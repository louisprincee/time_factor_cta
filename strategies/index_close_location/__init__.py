"""IF/IH/IC 等权收盘位置。熊市空头策略和股指研究共用的信号函数，本身不是策略。

原来登记在这里的两条股指策略（只做多、收盘位置加展期加空头）已于 2026-10-29 按用户决定删除，
台账 data/index/oos/ledger.jsonl 里有 decision 记录。熊市空头 strategies/index_bear_reversal 从这里取信号，
那份冻结文件的哈希因此不变。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SYMBOLS = ("IF", "IH", "IC")


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
