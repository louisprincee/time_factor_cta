"""研究用的日频序列：收益、滚动 MAD、滚动分位信号。

三件事共用同一张「日期 × 品种」宽表，而且口径互相咬合：
因子在 t 日收盘才知道，赚取的是 day_ret[t+1]，仓位是信号的 shift(1)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..data import shard_io


def daily_prices_from_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """单品种分钟表 → 每个 trading_date 一根开盘价。"""
    missing = [c for c in ('open', 'openw', 'trading_date') if c not in df.columns]
    if missing:
        raise KeyError(
            f"分钟分片缺少 {missing}，无法按第 8.3 节计算 day_ret。\n"
            "请用含 open/openw 的原始面板重跑 step1_shard_minutes.py，"
            "不要用 close 代替 open。"
        )
    td = pd.to_datetime(df['trading_date']).dt.normalize()
    # 分钟表按时间升序，groupby.first 就是该交易日第一根 bar
    g = df.groupby(td, sort=True)
    out = pd.DataFrame({
        'open': g['open'].first(),
        'openw': g['openw'].first(),
    })
    out.index.name = 'trading_date'
    return out


def day_return_from_prices(prices: pd.DataFrame) -> pd.Series:
    """day_ret[t] = (openw[t+1] - openw[t]) / open[t]。末日没有 t+1，为 NaN。"""
    open_ = prices['open'].astype('float64')
    openw = prices['openw'].astype('float64')
    ret = (openw.shift(-1) - openw) / open_.where(open_ != 0)
    ret.name = 'day_ret'
    return ret


def load_day_returns(symbols: list[str],
                     directory=None) -> pd.DataFrame:
    """宽表 ``index=trading_date, columns=symbol``。

    读取走 shard_io.load_shard，因此指向 holdout_locked/ 会直接被拒绝。
    """
    cols = {}
    for s in symbols:
        df = shard_io.load_shard(s, directory=directory, columns=['open', 'openw', 'trading_date'])
        px = daily_prices_from_minutes(df)
        C.assert_no_holdout_dates(px.index, what=f"{s} 日频开盘价")
        cols[s] = day_return_from_prices(px)
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols).sort_index()
    out.index.name = 'trading_date'
    return out


def forward_return(day_ret: pd.DataFrame) -> pd.DataFrame:
    """factor[t] 所预测的那段收益：day_ret[t+1]。

    day_ret[t+1] = (openw[t+2] - openw[t+1]) / open[t+1]，
    即 t 日收盘形成信号、t+1 开盘成交、持有到 t+2 开盘。
    """
    return day_ret.shift(-1)


def _mad_1d(x: np.ndarray,
            window: int,
            min_periods: int,
            mult: float,
            clip: float) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(x[i]):
            continue
        a = max(0, i + 1 - window)
        w = x[a:i + 1]
        w = w[np.isfinite(w)]
        if w.size < min_periods:
            continue
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        if not np.isfinite(mad) or mad == 0.0:
            continue
        z = (float(x[i]) - med) / (mult * mad)
        if z > clip:
            z = clip
        elif z < -clip:
            z = -clip
        out[i] = z
    return out


def mad_standardize(df: pd.DataFrame | pd.Series,
                    window: int | None = None,
                    min_periods: int | None = None,
                    mult: float | None = None,
                    clip: float | None = None) -> pd.DataFrame | pd.Series:
    """逐列标准化。默认窗口 1000、除数 5、clip 到 [-1, 1]。"""
    window = int(C.STD_WINDOW if window is None else window)
    min_periods = window if min_periods is None else int(min_periods)
    mult = float(C.STD_MAD_MULT if mult is None else mult)
    clip = float(C.STD_CLIP if clip is None else clip)
    if isinstance(df, pd.Series):
        arr = _mad_1d(df.to_numpy(dtype='float64'), window, min_periods, mult, clip)
        return pd.Series(arr, index=df.index, name=df.name)
    cols = {
        c: _mad_1d(df[c].to_numpy(dtype='float64'), window, min_periods, mult, clip)
        for c in df.columns
    }
    return pd.DataFrame(cols, index=df.index)


def rolling_band(factor: pd.DataFrame,
                 window: int,
                 q_low: float,
                 q_high: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (低轨, 高轨)，都是「截止到昨日」的滚动分位数。

    q_low / q_high 用百分数（30 表示 30%），与 config.SIGNAL_BANDS 一致。
    窗口内有效观测不足 window 根时为 NaN，不降低 min_periods 去凑。
    """
    hist = factor.shift(1)
    roll = hist.rolling(int(window), min_periods=int(window))
    lo = roll.quantile(float(q_low) / 100.0)
    hi = roll.quantile(float(q_high) / 100.0)
    return lo, hi


def quantile_signal(factor: pd.DataFrame,
                    window: int,
                    q_low: float,
                    q_high: float) -> pd.DataFrame:
    """宽表信号，取值 -1 / 0 / +1，历史不足或因子缺失处为 NaN。"""
    lo, hi = rolling_band(factor, window, q_low, q_high)
    valid = factor.notna() & lo.notna() & hi.notna()
    sig = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    sig = sig.mask(valid, 0.0)
    sig = sig.mask(valid & (factor > hi), 1.0)
    sig = sig.mask(valid & (factor < lo), -1.0)
    return sig


def hold_signal(sig: pd.DataFrame) -> pd.DataFrame:
    """轨道中间的 0 沿用上一笔非零方向。开头的 0 和 NaN 保持原样。"""
    return confirm_signal(sig, 1)


def confirm_signal(sig: pd.DataFrame, k: int) -> pd.DataFrame:
    """中间的 0 沿用上一方向。反手要连续 k 日处在对面轨道，首次开仓立即生效。"""
    if int(k) < 1:
        raise ValueError(f"确认天数 k 至少为 1，收到 {k}")
    need = int(k)
    arr = sig.to_numpy(dtype='float64', copy=True)
    for j in range(arr.shape[1]):
        last = np.nan
        streak = 0
        col = arr[:, j]
        for i in range(len(col)):
            v = col[i]
            if not np.isfinite(v):
                continue
            if not np.isfinite(last):
                if v != 0.0:
                    last = v
                continue
            if v == 0.0 or v == last:
                col[i] = last
                streak = 0
                continue
            streak += 1
            if streak >= need:
                last = v
                streak = 0
            else:
                col[i] = last
    return pd.DataFrame(arr, index=sig.index, columns=sig.columns)


def execute_position(signal: pd.DataFrame) -> pd.DataFrame:
    """收盘信号 → 下一交易日开盘才持有的仓位。"""
    return signal.shift(1)
