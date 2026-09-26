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
