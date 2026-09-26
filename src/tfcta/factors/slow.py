"""慢信号：时序动量与展期收益。给快信号（时间戳、持续期）当主体。

价格口径：分片里的 ``closew`` 是**加法**复权（``close - closew`` 日内恒定，跨日只在
换月日跳变）。所以收益一律写成 ``Δclosew / close``，不能用 ``closew`` 自己做分母：
加法复权后远期的 ``closew`` 水平和真实价位差得很远，比例会失真。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TSMOM_WINDOWS = (20, 60, 120, 250)
VOL_WINDOW = 60
CARRY_WINDOW = 250
CS_MIN_SYMBOLS = 5


def daily_close(minute: pd.DataFrame) -> pd.DataFrame:
    """单品种分钟表 → 每个 trading_date 的 close / closew。"""
    td = pd.to_datetime(minute['trading_date']).dt.normalize()
    g = minute.groupby(td, sort=True)
    out = pd.DataFrame({'close': g['close'].last(), 'closew': g['closew'].last()})
    out.index.name = 'trading_date'
    return out


def daily_vol(close: pd.DataFrame, closew: pd.DataFrame,
              window: int = VOL_WINDOW) -> pd.DataFrame:
    """截至当日的日收益波动。"""
    ret = closew.diff() / close.shift(1).where(close.shift(1) > 0)
    return ret.rolling(window, min_periods=window // 2).std()


def tsmom(close: pd.DataFrame, closew: pd.DataFrame,
          windows=TSMOM_WINDOWS) -> pd.DataFrame:
    """多窗口时序动量，取值 -1 到 1。

    每个窗口的收益除以 ``σ·sqrt(w)`` 变成 t 统计量样的尺度，截到 ±2 再除以 2，
    几个窗口等权平均。用连续值而不是符号，信号在 0 附近不会来回翻，换手低很多。
    """
    vol = daily_vol(close, closew)
    parts = []
    for w in windows:
        base = close.shift(w)
        r = (closew - closew.shift(w)) / base.where(base > 0)
        parts.append((r / (vol * np.sqrt(w))).clip(-2, 2) / 2)
    stacked = np.stack([p.to_numpy(dtype='float64') for p in parts])
    ok = np.isfinite(stacked)
    cnt = ok.sum(axis=0)
    acc = np.where(ok, stacked, 0.0).sum(axis=0)
    out = np.where(cnt == len(parts), acc / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def momentum_components(close: pd.DataFrame, closew: pd.DataFrame,
                        windows=TSMOM_WINDOWS,
                        vol_window: int = VOL_WINDOW) -> dict[str, pd.DataFrame]:
    """各窗口累计动量及其波动率缩放值，所有输入只用到当日收盘。"""
    daily_return = closew.diff() / close.shift(1).where(close.shift(1) > 0)
    log_return = np.log1p(daily_return.where(daily_return > -1))
    vol = daily_return.rolling(
        vol_window, min_periods=max(2, vol_window // 2)).std()
    out = {}
    for window in windows:
        total = log_return.rolling(int(window), min_periods=int(window)).sum()
        scaled = total / (vol * np.sqrt(int(window)))
        out[f'tsmom_{int(window)}'] = total
        out[f'tsmom_ra_{int(window)}'] = scaled.clip(-3, 3) / 3
    return out


def cross_sectional_rank(factor: pd.DataFrame,
                         universe: dict[int, list[str]],
                         min_symbols: int = CS_MIN_SYMBOLS) -> pd.DataFrame:
    """按每年事前确定的品种池做截面百分位排名，池外与样本不足日期保留 NaN。"""
    out = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    years = pd.DatetimeIndex(factor.index).year
    for year in sorted(set(years)):
        columns = [s for s in universe.get(int(year), []) if s in factor.columns]
        if not columns:
            continue
        dates = factor.index[years == year]
        block = factor.loc[dates, columns]
        valid = block.notna().sum(axis=1) >= int(min_symbols)
        ranks = block.rank(axis=1, method='average')
        count = block.notna().sum(axis=1).replace(0, np.nan)
        ranked = ranks.sub(0.5).div(count, axis=0) - 0.5
        out.loc[dates[valid], columns] = ranked.loc[valid]
    return out


def cross_sectional_momentum(close: pd.DataFrame, closew: pd.DataFrame,
                             universe: dict[int, list[str]],
                             windows=TSMOM_WINDOWS,
                             vol_window: int = VOL_WINDOW,
                             min_symbols: int = CS_MIN_SYMBOLS
                             ) -> dict[str, pd.DataFrame]:
    """截面动量与风险调整截面动量，返回中心化百分位排名，不负责策略合成。"""
    components = momentum_components(close, closew, windows, vol_window)
    ranked = {}
    for name, factor in components.items():
        if name.startswith('tsmom_ra_'):
            rank_name = name.replace('tsmom_ra_', 'cs_mom_ra_')
        else:
            rank_name = name.replace('tsmom_', 'cs_mom_')
        ranked[rank_name] = cross_sectional_rank(factor, universe, min_symbols)
    return ranked


def rolling_return_skewness(close: pd.DataFrame, closew: pd.DataFrame,
                            window: int = 60,
                            min_periods: int | None = None) -> pd.DataFrame:
    """截至当日的日收益偏度；不指定方向，供尾部风险假设做IC检验。"""
    daily_return = closew.diff() / close.shift(1).where(close.shift(1) > 0)
    minimum = max(3, int(window * 2 / 3)) if min_periods is None else int(min_periods)
    return daily_return.rolling(int(window), min_periods=minimum).skew()


def cross_sectional_low_volatility(close: pd.DataFrame,
                                   closew: pd.DataFrame,
                                   universe: dict[int, list[str]],
                                   window: int = 60,
                                   min_symbols: int = CS_MIN_SYMBOLS
                                   ) -> pd.DataFrame:
    """低已实现波动率的中心化截面秩；不称作特异性波动率。"""
    daily_return = closew.diff() / close.shift(1).where(close.shift(1) > 0)
    vol = daily_return.rolling(
        int(window), min_periods=max(2, int(window) // 2)).std()
    return cross_sectional_rank(-vol, universe, min_symbols)


def roll_gap(close: pd.DataFrame, closew: pd.DataFrame) -> pd.DataFrame:
    """换月日的相对价差 ``(P_new - P_old) / P_new``，其余日为 0。

    加法复权下 ``close - closew`` 只在换月日变化，变化量就是新旧合约的价差。
    """
    diff = (close - closew)
    gap = diff.diff()
    gap = gap.where(gap.abs() > 1e-6, 0.0)
    return gap / close.where(close > 0)


def carry(close: pd.DataFrame, closew: pd.DataFrame,
          window: int = CARRY_WINDOW) -> pd.DataFrame:
    """过去一年已实现的展期收益：换月时新合约比旧合约便宜的幅度之和。

    正值 = 贴水结构（backwardation），多头持有能赚展期。这是事后实现的 carry，
    比用远月合约报价算的即期 carry 滞后，但只用现有的主力连续数据就能得到。
    """
    g = roll_gap(close, closew)
    return -g.rolling(window, min_periods=window).sum()
