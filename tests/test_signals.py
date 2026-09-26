"""信号装配口径的测试：复权换算、展期收益。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta.data import bars as B
from tfcta.factors import daily


def _idx(n):
    return pd.bdate_range('2016-01-04', periods=n)


def test_multiplicative_prices_keep_true_returns_under_additive_adjustment():
    """加法复权后 closew 的比例失真，重建的乘法复权价必须还原真实日收益。"""
    idx = _idx(4)
    close = pd.DataFrame({'A': [100.0, 102.0, 51.0, 50.0]}, index=idx)
    # 第 3 天换月，新合约比旧合约低 50：加法复权把此前全部下移 50
    closew = pd.DataFrame({'A': [50.0, 52.0, 51.0, 50.0]}, index=idx)
    bars = {'close': close, 'closew': closew, 'highw': closew + 1, 'loww': closew - 1}
    px = B.multiplicative_prices(bars)['close']['A']
    assert px.iloc[1] / px.iloc[0] - 1 == pytest.approx(0.02)
    assert px.iloc[2] / px.iloc[1] - 1 == pytest.approx(-1.0 / 102.0)
    # 直接用 closew 算会得到 +4%，这就是要避免的失真
    assert closew['A'].iloc[1] / closew['A'].iloc[0] - 1 == pytest.approx(0.04)


def test_carry_is_positive_under_backwardation():
    """新合约比旧合约便宜（贴水）时，展期收益为正。"""
    n = 300
    idx = _idx(n)
    close = pd.Series(100.0, index=idx)
    closew = close.copy()
    for k in (50, 150, 250):          # 三次换月，每次新合约便宜 2
        close.iloc[k:] -= 2.0
    diff = pd.DataFrame({'A': close}), pd.DataFrame({'A': closew})
    c = daily.carry(*diff, window=250)['A']
    assert np.isnan(c.iloc[248])
    assert c.iloc[-1] > 0
    g = daily.roll_gap(*diff)['A']
    assert (g != 0).sum() == 3


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


def test_cross_sectional_momentum_uses_only_trailing_returns():
    idx = _idx(80)
    close = pd.DataFrame({s: np.linspace(100, 120 + i, len(idx))
                          for i, s in enumerate('ABCDE')}, index=idx)
    closew = close.copy()
    factors = daily.cross_sectional_momentum(
        close, closew, {2016: list('ABCDE')}, windows=(20,), vol_window=10)
    assert set(factors) == {'cs_mom_20', 'cs_mom_ra_20'}
    assert factors['cs_mom_20'].iloc[-1].notna().sum() == 5
    assert factors['cs_mom_ra_20'].iloc[-1].notna().sum() == 5


def test_rolling_skewness_does_not_use_future_returns():
    idx = _idx(80)
    close = pd.DataFrame({'A': np.exp(np.linspace(0, 0.2, len(idx)))}, index=idx)
    closew = close.copy()
    skew = daily.rolling_return_skewness(close, closew, window=20, min_periods=15)
    changed = close.copy()
    changed.iloc[-1, 0] *= 1.5
    skew_changed = daily.rolling_return_skewness(
        changed, changed, window=20, min_periods=15)
    pd.testing.assert_series_equal(skew['A'].iloc[:-1], skew_changed['A'].iloc[:-1])


def test_low_volatility_is_ranked_only_inside_each_year_pool():
    idx = _idx(80)
    phase = np.sin(np.arange(len(idx)))
    close = pd.DataFrame({s: np.exp(np.cumsum(0.001 * (i + 1) * phase))
                          for i, s in enumerate('ABCDE')}, index=idx)
    low_vol = daily.cross_sectional_low_volatility(
        close, close, {2016: list('ABCDE')}, window=20)
    assert low_vol['A'].iloc[-1] > low_vol['E'].iloc[-1]
    assert low_vol.iloc[-1].notna().sum() == 5
