"""walk-forward 折与测试年拼接。这一层不跑 2022 及以后。"""
from __future__ import annotations

import pandas as pd

from .. import config as C


def walk_forward_folds(test_years: list[int] | None = None) -> list[dict]:
    years = C.WF_TEST_YEARS_LIST if test_years is None else list(test_years)
    rows = []
    for y in years:
        y = int(y)
        rows.append({
            'test_year': y,
            'train_start': y - C.WF_TRAIN_YEARS,
            'train_end': y - 1,
            'sparse_night': y in C.WF_FOLDS_WITH_SPARSE_NIGHT,
        })
    return rows


def stitch_test_years(port: pd.Series, years: list[int]) -> pd.Series:
    """把各测试年的组合收益按时间拼成一条。空年份跳过，不插 0。"""
    parts = []
    for y in years:
        sl = port.loc[port.index.year == int(y)].dropna()
        if len(sl):
            parts.append(sl)
    if not parts:
        return port.iloc[0:0]
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep='first')]
    out.name = getattr(port, 'name', None) or 'port_ret'
    return out
