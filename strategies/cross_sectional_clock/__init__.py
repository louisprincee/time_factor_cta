"""Research-only cross-sectional long-short clock ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd

ID = "cross_sectional_clock"
NAME = "时钟截面多空"
DETAIL = "年流动性池截面时钟前20%做多、后20%做空/持有5日"
TAIL_FRACTION = 0.20
MIN_SYMBOLS = 10


def signal(
    micro: pd.DataFrame,
    universe: dict[int, list[str]],
) -> pd.DataFrame:
    """每日等数量选取两端；分界并列时不按任意顺序拆分。"""
    result = pd.DataFrame(0, index=micro.index, columns=micro.columns, dtype="int8")
    years = pd.DatetimeIndex(micro.index).year
    for year in sorted(set(years)):
        columns = sorted(set(universe.get(int(year), ())).intersection(micro.columns))
        if len(columns) < MIN_SYMBOLS:
            continue
        dates = micro.index[years == year]
        values = micro.loc[dates, columns]
        for day, row in values.iterrows():
            valid = row.dropna()
            count = len(valid)
            if count < MIN_SYMBOLS:
                continue
            side_count = int(np.floor(count * TAIL_FRACTION))
            ordered = np.sort(valid.to_numpy(dtype="float64"))
            lower = ordered[side_count - 1]
            upper = ordered[count - side_count]
            if lower == ordered[side_count] or upper == ordered[count - side_count - 1]:
                continue
            result.loc[day, valid.index[valid.le(lower)]] = -1
            result.loc[day, valid.index[valid.ge(upper)]] = 1
    return result
