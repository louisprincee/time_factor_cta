"""Research-only long clock strategy gated by lagged positive roll yield."""
from __future__ import annotations

import pandas as pd

from strategies.book import upper_tail

ID = "trend_high_atr_carry"
NAME = "趋势尾部高波动且正展期收益"
DETAIL = "10日上涨/全日时钟最高20%/高ATR/滞后主次月年化展期收益率>0/持有5日"


def signal(
    trend: pd.DataFrame,
    micro: pd.DataFrame,
    high_vol: pd.DataFrame,
    carry: pd.DataFrame,
) -> pd.DataFrame:
    trend = trend.reindex(index=micro.index, columns=micro.columns)
    high_vol = high_vol.reindex(index=micro.index, columns=micro.columns)
    carry = carry.reindex(index=micro.index, columns=micro.columns)
    tail = pd.DataFrame(
        {symbol: upper_tail(micro[symbol]) for symbol in micro.columns},
        index=micro.index,
    )
    return trend.gt(0) & tail.gt(0) & high_vol.fillna(False) & carry.gt(0)
