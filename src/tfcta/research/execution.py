"""从目标信号到收盘时的持仓意图。输出仍是"t 日收盘的信号"，成交滞后交给
:func:`panel.execute_position`，这里不做 shift。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def weekly(sig: pd.DataFrame) -> pd.DataFrame:
    """每周最后一个交易日取值，其余日沿用。"""
    s = pd.Series(sig.index, index=sig.index)
    iso = s.dt.isocalendar()
    key = iso.year.astype(str) + '-' + iso.week.astype(str)
    reb = pd.DatetimeIndex(s.groupby(key).tail(1).values)
    out = sig.copy()
    out.loc[~out.index.isin(reb)] = np.nan
    return out.ffill()
