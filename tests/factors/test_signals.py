"""信号装配口径的测试：复权换算、展期收益。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta.factors import daily


def _idx(n):
    return pd.bdate_range('2016-01-04', periods=n)

def test_tsmom_sign_is_trailing_month_direction_and_ignores_the_future():
    idx = _idx(30)
    close = pd.DataFrame({'A': np.linspace(100, 130, len(idx))}, index=idx)
    sign = daily.tsmom_sign(close, close.copy(), window=20)['A']
    assert np.isnan(sign.iloc[19])
    assert sign.iloc[20] == 1.0
    crashed = close.copy()
    crashed.iloc[-1, 0] = 50.0
    sign_crashed = daily.tsmom_sign(crashed, crashed, window=20)['A']
    pd.testing.assert_series_equal(sign.iloc[:-1], sign_crashed.iloc[:-1])
    assert sign_crashed.iloc[-1] == -1.0

def test_cross_sectional_momentum_ranks_only_current_year_universe():
    idx = pd.bdate_range('2020-01-01', periods=6)
    factor = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6],
        'B': [2, 3, 4, 5, 6, 7],
        'C': [3, 4, 5, 6, 7, 8],
        'D': [4, 5, 6, 7, 8, 9],
        'E': [5, 6, 7, 8, 9, 10],
        'OUT': [100, 100, 100, 100, 100, 100],
    }, index=idx)
    ranked = daily.cross_sectional_rank(
        factor, {2020: ['A', 'B', 'C', 'D', 'E']})
    assert ranked.loc[idx[-1], 'A'] == pytest.approx(-0.4)
    assert ranked.loc[idx[-1], 'E'] == pytest.approx(0.4)
    assert ranked['OUT'].isna().all()
    assert ranked.notna().sum(axis=1).eq(5).all()
