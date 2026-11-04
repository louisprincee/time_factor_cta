"""股指策略共享行情因子。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfcta import config as C

SYMBOLS = ("IF", "IH", "IC")
THRESHOLD = 0.4
VOL_WINDOW = 20
VOL_TARGET = 0.20


def high_open_z(bars, symbols=SYMBOLS):
    """冲高以此前60日均值/标准差归一化，至少40日，不含当前日。"""
    high_open = (
        (bars["highw"] - bars["openw"])
        / bars["open"].where(bars["open"] > 0)
        * 1e4
    )
    high_open = high_open.reindex(columns=list(symbols))
    prior = high_open.shift(1)
    mean = prior.rolling(60, min_periods=40).mean()
    scale = prior.rolling(60, min_periods=40).std()
    z = (high_open - mean) / scale.where(scale > 0)
    return z.where(np.isfinite(z)).mean(axis=1, skipna=False).rename("high_open_z")


def close_vol_scale(open_to_open):
    """收盘时仅用截至当日开盘已经完成的20段收益。"""
    rv = (
        open_to_open.shift(1).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
        * np.sqrt(C.TRADING_DAYS_PER_YEAR)
    )
    return (VOL_TARGET / rv).clip(upper=1.0)


def market_close_location(
    bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> pd.Series:
    """IF/IH/IC 等权收盘位置；必须三品种都有有效日内区间。"""
    close = bars["closew"].reindex(index=index, columns=SYMBOLS)
    high = bars["highw"].reindex(index=index, columns=SYMBOLS)
    low = bars["loww"].reindex(index=index, columns=SYMBOLS)
    width = high - low
    if close.isna().any().any() or high.isna().any().any() or low.isna().any().any():
        raise ValueError("股指收盘位置行情未完全对齐")
    if width.le(0).any().any():
        raise ValueError("股指日内最高价必须高于最低价")
    clv = (close - low) / width
    if (
        not np.isfinite(clv.to_numpy()).all()
        or clv.lt(0).any().any()
        or clv.gt(1).any().any()
    ):
        raise ValueError("股指收盘位置超出 [0, 1] 或包含无效值")
    return clv.mean(axis=1, skipna=False).rename("market_clv")
