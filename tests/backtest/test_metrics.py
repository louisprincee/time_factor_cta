"""回测绩效、紧缩夏普和过拟合概率。当前没有策略在跑。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import METRIC_KEYS, deflated_sharpe, performance, pbo_cscv, sharpe_ratio


def test_metrics_hand_values():
    # 两天 +10%、-5%：净值 1.1 * 0.95 = 1.045
    r = pd.Series([0.10, -0.05])
    m = performance(r, periods=252)
    nav = 1.045
    ann = nav ** (252 / 2) - 1
    vol = float(r.std(ddof=1) * np.sqrt(252))
    assert m['n_days'] == 2
    assert m['ann_return'] == pytest.approx(ann)
    assert m['ann_vol'] == pytest.approx(vol)
    assert m['ret_risk'] == pytest.approx(ann / vol)
    assert m['win_rate'] == pytest.approx(0.5)
    # 高点 1.1，回落到 1.045，回撤 1.045/1.1 - 1
    assert m['max_drawdown'] == pytest.approx(1.045 / 1.1 - 1)

def test_performance_and_sharpe_tolerate_tiny_samples():
    for r in (pd.Series(dtype='float64'), pd.Series([0.01])):
        m = performance(r)
        assert set(METRIC_KEYS) <= set(m)
        assert np.isnan(sharpe_ratio(r))

def test_pbo_separates_noise_from_a_real_edge():
    idx = pd.bdate_range('2016-01-01', periods=1200)
    noise_pbo = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noise = pd.DataFrame(rng.normal(0, .01, (1200, 20)), index=idx)
        edge = noise.copy()
        edge[0] += .002
        assert pbo_cscv(edge, 8)['pbo'] < .05
        noise_pbo.append(pbo_cscv(noise, 8)['pbo'])
    assert .3 < np.mean(noise_pbo) < .7  # 纯噪声时样本内最好的方案在样本外大约一半时间排在中位数以下
    with pytest.raises(ValueError):
        pbo_cscv(noise, 7)


def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(.0006, .01, 1500))
    trials = rng.normal(0, .5, 50)
    few = deflated_sharpe(ret, trials, 5)['dsr']
    many = deflated_sharpe(ret, trials, 500)['dsr']
    assert 0 <= many < few <= 1

def test_drawdown_includes_starting_capital():
    assert performance(pd.Series([-.10, .02]))['max_drawdown'] == pytest.approx(-.10)


