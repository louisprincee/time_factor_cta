"""信号装配口径的测试：复权换算、展期收益。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta.factors import slow
from tfcta.research import panel


def _idx(n):
    return pd.bdate_range('2016-01-04', periods=n)


def test_multiplicative_prices_keep_true_returns_under_additive_adjustment():
    """加法复权后 closew 的比例失真，重建的乘法复权价必须还原真实日收益。"""
    idx = _idx(4)
    close = pd.DataFrame({'A': [100.0, 102.0, 51.0, 50.0]}, index=idx)
    # 第 3 天换月，新合约比旧合约低 50：加法复权把此前全部下移 50
    closew = pd.DataFrame({'A': [50.0, 52.0, 51.0, 50.0]}, index=idx)
    bars = {'close': close, 'closew': closew, 'highw': closew + 1, 'loww': closew - 1}
    px = panel.multiplicative_prices(bars)['close']['A']
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
    c = slow.carry(*diff, window=250)['A']
    assert np.isnan(c.iloc[248])
    assert c.iloc[-1] > 0
    g = slow.roll_gap(*diff)['A']
    assert (g != 0).sum() == 3
