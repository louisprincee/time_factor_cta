"""日频因子：当前策略用到的四个。

持续期族只算价格持续期上的 dfp_max、dfp_top3，依赖 (lookback, pct)。
时间戳族只算 ts_high、ts_low，不依赖参数。两边都以 trading_date 为 index，
落盘的是原始值。方向符号只在 apply_signs 里乘上。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..data import sessions
from . import duration as D


def day_codes_of(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """把 trading_date 转成连续整数编码，并返回去重后的交易日索引。"""
    td = pd.to_datetime(df['trading_date'])
    codes, uniq = pd.factorize(td, sort=True)
    return codes, pd.DatetimeIndex(uniq)


def _day_bounds(day_codes: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, day_codes[1:] != day_codes[:-1], True])


def dfp_factors(dur: np.ndarray,
                price: np.ndarray,
                day_codes: np.ndarray,
                days: pd.DatetimeIndex,
                top_ns: list[int] | None = None) -> pd.DataFrame:
    """公允均衡价格偏离。

        FP_t  = mean(closew 于持续期前 N 大的那些分钟)
        DFP_t = (FP_t - Close_t) / Close_t

    Close_t 取当日最后一根有效 closew。除以收盘价是为了跨品种可比。
    """
    top_ns = top_ns or C.FP_TOP_NS
    n = len(days)
    cols = {f'dfp_{"max" if N == 1 else f"top{N}"}': np.full(n, np.nan) for N in top_ns}
    bounds = _day_bounds(day_codes)

    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        d, p = dur[a:b], price[a:b]
        ok = np.isfinite(d) & np.isfinite(p)
        if not ok.any():
            continue
        d_ok, p_ok = d[ok], p[ok]
        close_t = p_ok[-1]
        if not np.isfinite(close_t) or close_t == 0:
            continue
        order = np.argsort(-d_ok, kind='mergesort')
        for N in top_ns:
            take = order[:min(N, d_ok.size)]
            fp = float(p_ok[take].mean())
            name = f'dfp_{"max" if N == 1 else f"top{N}"}'
            cols[name][k] = (fp - close_t) / close_t

    return pd.DataFrame(cols, index=days)


def duration_factors(df: pd.DataFrame,
                     lookback: int,
                     pct: float,
                     price_col: str = 'closew',
                     thr_p: pd.Series | None = None) -> pd.DataFrame:
    """一个 (lookback, pct) 下的 dfp_max 与 dfp_top3。"""
    for col in ('trading_date', 'gamma_norm', 'session'):
        if col not in df.columns:
            raise KeyError(f"缺少 {col} 列，请先调用 sessions.add_intraday_coords")

    codes, days = day_codes_of(df)
    price = df[price_col].to_numpy(dtype='float64')
    if thr_p is None:
        thr_p = D.rolling_threshold(D.intraday_abs_diff(price, codes), codes, lookback, pct)
    dur_p = D.duration_series(price, codes, thr_p)
    out = dfp_factors(dur_p, price, codes, days)
    out.index.name = 'trading_date'
    return out


def _extreme_timepoint(values: np.ndarray, gnorm: np.ndarray, mode: str) -> float:
    ok = np.isfinite(values) & np.isfinite(gnorm)
    if not ok.any():
        return np.nan
    pos = np.flatnonzero(ok)
    v = values[pos]
    j = pos[np.argmax(v) if mode == 'max' else np.argmin(v)]
    return float(gnorm[j])


def timestamp_factors(df: pd.DataFrame) -> pd.DataFrame:
    """ts_high、ts_low：全日最高价、最低价出现的归一化时点。"""
    need = ['trading_date', 'gamma_norm', 'highw', 'loww']
    for col in need:
        if col not in df.columns:
            raise KeyError(f"缺少 {col} 列，请先调用 sessions.add_intraday_coords")

    codes, days = day_codes_of(df)
    n = len(days)
    hi = df['highw'].to_numpy(dtype='float64')
    lo = df['loww'].to_numpy(dtype='float64')
    gnorm = df['gamma_norm'].to_numpy(dtype='float64')
    out = {k: np.full(n, np.nan) for k in ('ts_high', 'ts_low')}
    bounds = _day_bounds(codes)

    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if not np.isfinite(hi[a:b]).any():
            continue
        out['ts_high'][k] = _extreme_timepoint(hi[a:b], gnorm[a:b], 'max')
        out['ts_low'][k] = _extreme_timepoint(lo[a:b], gnorm[a:b], 'min')

    res = pd.DataFrame(out, index=days)
    res.index.name = 'trading_date'
    return res


class UnsignedFactor(KeyError):
    """出现了方向表里没有的因子列。"""


def apply_signs(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """按 config.FACTOR_SIGNS 把每列乘上方向，使因子统一为「越大越看多」。"""
    unknown = [c for c in df.columns if c not in C.FACTOR_SIGNS]
    if unknown and strict:
        raise UnsignedFactor(
            f"以下因子列没有登记方向: {unknown}\n"
            "请在 config.FACTOR_SIGNS 中登记。"
        )
    out = df.copy()
    for c in out.columns:
        out[c] = out[c] * C.FACTOR_SIGNS.get(c, 1)
    return out


def symbol_daily_factors(minute_df: pd.DataFrame,
                         lookback: int,
                         pct: float,
                         with_coords: bool = False) -> pd.DataFrame:
    """单品种、单参数组合下的四个日频因子（原始值，未施加方向）。"""
    df = minute_df if with_coords else sessions.add_intraday_coords(minute_df)
    dur = duration_factors(df, lookback=lookback, pct=pct)
    ts = timestamp_factors(df)
    out = pd.concat([dur, ts], axis=1)
    out.index.name = 'trading_date'
    return out
