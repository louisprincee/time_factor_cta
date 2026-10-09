"""研究层与回测引擎的口径测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import bars as returns
from tfcta.research import stats as ic



def test_metrics_hand_values():
    # 两天 +10%、-5%：净值 1.1 * 0.95 = 1.045
    r = pd.Series([0.10, -0.05])
    m = ic.performance(r, periods=252)
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

def test_ic_sign_on_perfect_forecast():
    # 方向验收读 ic_ts，事前 z 与事前波动各要 120 日预热，所以从 2015 年中开始
    idx = pd.bdate_range('2015-06-01', '2016-12-30')
    rng = np.random.default_rng(0)
    day_ret = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 3)), index=idx, columns=list('ABC'))
    fwd = returns.forward_return(day_ret)
    universe = {2016: list('ABC')}
    pos = ic.factor_ic_table(fwd, fwd, universe, 'dfp_max', [2016], min_obs=30)
    neg = ic.factor_ic_table(-fwd, fwd, universe, 'ts_high', [2016], min_obs=30)
    assert pos.loc[pos['fold'] == '2016', 'ic'].iloc[0] == pytest.approx(1.0)
    assert pos.loc[pos['fold'] == 'mean_of_folds', 'sign'].iloc[0] == 'ok'
    # ts_high 的符号是 -1。原始因子 = -未来收益时，IC 为负，与符号一致
    assert neg.loc[neg['fold'] == '2016', 'ic'].iloc[0] == pytest.approx(-1.0)
    assert neg.loc[neg['fold'] == 'mean_of_folds', 'sign'].iloc[0] == 'ok'
    # 汇总行的 n_symbols 是折间均值（品种数），不是"品种×折"的累加
    mof = pos.loc[pos['fold'] == 'mean_of_folds'].iloc[0]
    assert int(mof['n_symbols']) == 3
    assert int(mof['n_folds']) == 1

def test_sign_gate_separates_significant_flip_from_unmeasurable():
    """"方向相反"和"测不出来"必须是两个标签，处置完全不同。"""
    assert ic.sign_status(+0.0008, 'ts_high', 0.22) == 'flip_weak'
    assert ic.sign_status(+0.05, 'ts_high', 4.0) == 'flip'
    assert ic.sign_status(-0.05, 'ts_high', 4.0) == 'ok'
    assert ic.sign_status(+0.0008, 'ts_high') == 'flip'
    assert ic.sign_status(+0.0008, 'ts_high', np.nan) == 'flip'
    assert ic.sign_status(0.0, 'ts_high', 0.1) == 'inconclusive'
    assert ic.sign_status(0.9, 'dur_mean', 9.0) == 'no_prior'

def test_ic_fold_is_a_time_slice_not_just_a_symbol_pool():
    """一折必须既切品种池也切时间。"""
    idx = pd.bdate_range('2016-01-04', periods=504)      # 覆盖 2016 与 2017
    rng = np.random.default_rng(7)
    syms = list('ABC')
    fwd = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 3)), index=idx, columns=syms)
    # 2016 年因子等于未来收益（IC=+1），2017 年取反（IC=-1）
    flip = np.where(pd.DatetimeIndex(idx).year == 2016, 1.0, -1.0)[:, None]
    factor = fwd * flip

    tab = ic.factor_ic_table(factor, fwd, {2016: syms, 2017: syms},
                             'dfp_max', [2016, 2017], min_obs=30).set_index('fold')
    assert tab.loc['2016', 'ic'] == pytest.approx(1.0)
    assert tab.loc['2017', 'ic'] == pytest.approx(-1.0)
    # 不切时间的话两折都会是混合后的同一个值，符号也不会相反
    assert tab.loc['mean_of_folds', 'ic'] == pytest.approx(0.0)

def test_cross_sectional_ic_uses_daily_cross_section_and_time_series_t():
    dates = pd.bdate_range('2018-01-01', periods=252)
    symbols = list('ABCDE')
    factor = pd.DataFrame(
        np.tile(np.arange(5, dtype='float64'), (len(dates), 1)),
        index=dates, columns=symbols)
    future = pd.DataFrame(
        np.tile(np.arange(5, dtype='float64'), (len(dates), 1)),
        index=dates, columns=symbols)
    tab = ic.cross_sectional_ic_table(
        factor, future, {2018: symbols}, 'cs_mom', [2018]).set_index('fold')
    assert tab.loc['2018', 'ic'] == pytest.approx(1.0)
    assert tab.loc['2018', 'n_symbols'] == 5
    assert tab.loc['2018', 'n_periods'] == 12
    assert tab.loc['mean_of_folds', 'sign'] == 'no_prior'

def test_ic_timeseries_t_is_far_smaller_than_cross_sectional_t():
    """时序 t 与横截面 t 的差别，用一个共同驱动的样本量化出来。"""
    idx = pd.bdate_range('2016-01-04', periods=504)
    rng = np.random.default_rng(11)
    syms = [f'S{i}' for i in range(8)]
    f_common = rng.normal(0, 1, len(idx))
    r_common = 0.25 * f_common + rng.normal(0, 1, len(idx))
    factor = pd.DataFrame(
        {s: f_common + 0.01 * rng.normal(0, 1, len(idx)) for s in syms}, index=idx)
    fwd = pd.DataFrame(
        {s: r_common + 0.01 * rng.normal(0, 1, len(idx)) for s in syms}, index=idx)

    tab = ic.factor_ic_table(factor, fwd, {2016: syms, 2017: syms},
                             'dfp_max', [2016, 2017], min_obs=30).set_index('fold')
    mof = tab.loc['mean_of_folds']
    assert 22 <= int(mof['n_periods']) <= 24          # 两年约 24 个月
    assert int(mof['nw_lag']) == ic.nw_lag(int(mof['n_periods']))
    assert mof['t'] > 2.0                             # 真信号，时序上也显著
    assert mof['ic_ts'] == pytest.approx(mof['ic'], abs=0.05)
    # 同质品种把横截面 t 吹得离谱；这就是不能拿它当显著性的原因
    assert tab.loc['2016', 't_cross'] > 10 * tab.loc['2016', 't']

def test_timeseries_ic_has_no_small_sample_bias_on_persistent_factor():
    """随机游走上的 RSI 不可能有预测力，时序 IC 必须测不出东西。"""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2014-01-01', periods=1500)
    syms = [f'S{i}' for i in range(40)]
    px = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.015, (len(idx), 40)), axis=0)),
                      index=idx, columns=syms)
    fwd = returns.forward_return(px.pct_change())
    # A persistent past-price statistic, independent of subsequent innovations.
    factor = px.rolling(20, min_periods=20).mean()
    ser = ic.ic_period_series(factor, fwd, syms)
    res = ic.timeseries_t(ser)
    assert abs(res['ic_ts']) < 0.02
    assert abs(res['t']) < 3.0

def test_newey_west_se_exceeds_plain_se_under_autocorrelation():
    """正自相关时 NW 标准误必须大于普通标准误，否则 t 值虚高。"""
    n = 120
    x = pd.Series(np.sin(np.arange(n) / 3.0) + 0.5)   # 强正自相关
    plain = ic._nw_se(x.to_numpy(), 0)
    assert plain == pytest.approx(float(x.std(ddof=0)) / np.sqrt(n))
    assert ic._nw_se(x.to_numpy(), ic.nw_lag(n)) > plain
    assert ic.nw_lag(12) == 2 and ic.nw_lag(72) == 3

    # 期数太少不给 t 值，宁可留空也不给一个假精度
    short = ic.timeseries_t(pd.Series([0.1] * (C.IC_PERIOD_MIN_COUNT - 1)))
    assert np.isnan(short['t']) and short['n_periods'] == C.IC_PERIOD_MIN_COUNT - 1
    # 逐期 IC 完全没有变异时也留空，而不是 inf
    flat = ic.timeseries_t(pd.Series([0.1] * 24))
    assert np.isnan(flat['t']) and flat['ic_ts'] == pytest.approx(0.1)

def test_performance_and_sharpe_tolerate_tiny_samples():
    for r in (pd.Series(dtype='float64'), pd.Series([0.01])):
        m = ic.performance(r)
        assert set(ic.METRIC_KEYS) <= set(m)
        assert np.isnan(ic.sharpe_ratio(r))

def test_holding_forward_return_sums_the_next_h_days_and_vol_uses_only_the_past():
    idx = pd.bdate_range('2021-01-04', periods=12)
    day_ret = pd.DataFrame({'A': np.arange(12, dtype=float)}, index=idx)
    fwd3 = returns.holding_forward_return(day_ret, 3)
    assert fwd3.attrs['horizon'] == 3
    # fwd3[t] = day_ret[t+1] + day_ret[t+2] + day_ret[t+3]
    assert fwd3['A'].iloc[0] == pytest.approx(1 + 2 + 3)
    assert fwd3['A'].iloc[8] == pytest.approx(9 + 10 + 11)
    assert fwd3['A'].iloc[9:].isna().all()
    pd.testing.assert_frame_equal(returns.holding_forward_return(day_ret, 1),
                                  returns.forward_return(day_ret))
    # 事前波动只能用到 t-1 为止已实现的收益：改动 t 及以后的收益不影响 t 日的分母
    bumped = day_ret.copy()
    bumped.iloc[6:] *= 100.0
    a = ic.exante_scaled_return(returns.holding_forward_return(day_ret, 3), 3, 3)
    b = ic.exante_scaled_return(returns.holding_forward_return(bumped, 3), 3, 3)
    # t=6 的分子变了，分母不变 → 比值正好放大 100 倍
    assert b['A'].iloc[6] == pytest.approx(100.0 * a['A'].iloc[6])


def test_pbo_separates_noise_from_a_real_edge():
    from tfcta.research import stats as S
    idx = pd.bdate_range('2016-01-01', periods=1200)
    noise_pbo = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noise = pd.DataFrame(rng.normal(0, .01, (1200, 20)), index=idx)
        edge = noise.copy()
        edge[0] += .002
        assert S.pbo_cscv(edge, 8)['pbo'] < .05
        noise_pbo.append(S.pbo_cscv(noise, 8)['pbo'])
    assert .3 < np.mean(noise_pbo) < .7  # 纯噪声时样本内最好的方案在样本外大约一半时间排在中位数以下
    with pytest.raises(ValueError):
        S.pbo_cscv(noise, 7)


def test_deflated_sharpe_falls_with_more_trials():
    from tfcta.research import stats as S
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(.0006, .01, 1500))
    trials = rng.normal(0, .5, 50)
    few = S.deflated_sharpe(ret, trials, 5)['dsr']
    many = S.deflated_sharpe(ret, trials, 500)['dsr']
    assert 0 <= many < few <= 1
