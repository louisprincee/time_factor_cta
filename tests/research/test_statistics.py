"""研究层与回测引擎的口径测试。

锁住的是几件静默就会算错的事：收益公式、信号不含当日、仓位晚一天成交、
手续费与滑点按换手扣除、IC 一折必须切时间、显著性用时序而非横截面 t、
中心点距离必须先 z-score、夜盘缺失不能被平均成 0。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import synth
from tfcta.data import bars as returns
from tfcta.factors import library
from tfcta.research.analysis import stats as ic



def test_day_return_matches_framework_formula():
    px = pd.DataFrame({
        'open': [100.0, 110.0, 90.0],
        'openw': [100.0, 121.0, 99.0],
    }, index=pd.to_datetime(['2016-01-04', '2016-01-05', '2016-01-06']))
    ret = returns.day_return_from_prices(px)
    # (121-100)/100 = 0.21； (99-121)/110 = -0.2；末日无 t+1
    assert ret.iloc[0] == pytest.approx(0.21)
    assert ret.iloc[1] == pytest.approx(-0.2)
    assert np.isnan(ret.iloc[2])

def test_daily_open_is_first_bar_of_trading_date():
    days = pd.bdate_range('2016-01-04', periods=4)
    df = synth.make_symbol(days, night_class='night_2300', seed=3)
    px = returns.daily_prices_from_minutes(df)
    td = pd.to_datetime(df['trading_date']).dt.normalize()
    day = px.index[1]
    first = df.loc[td == day].iloc[0]
    assert px.loc[day, 'open'] == pytest.approx(first['open'])
    assert px.loc[day, 'openw'] == pytest.approx(first['openw'])
    # 有夜盘时，该交易日的第一根应当在夜盘，而不是 09:00
    assert first.name.hour >= 20 or first.name.hour <= C.NIGHT_END_HOUR

def test_load_day_returns_refuses_holdout():
    with pytest.raises(C.HoldoutViolation):
        returns.load_day_returns(['RB'], directory=C.HOLDOUT_DIR)

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
    """"方向相反"和"测不出来"必须是两个标签，处置完全不同。

    pmt 在商品上的实测就是后者：IC = +0.0008、t = 0.22，六折里四折反向两折同向，
    量级全在 0.016 以内。判成 flip 等于宣称"发现了方向错误"并把人送去查实现，
    而真实结论是"论文的因子没迁移过来"。反过来，显著的反向必须照样拦——
    不给 t 时退回严格口径，就是为了防止哪天有人调用时忘了传 t 而悄悄放松闸门。
    """
    assert ic.sign_status(+0.0008, 'ts_high', 0.22) == 'flip_weak'
    assert ic.sign_status(+0.05, 'ts_high', 4.0) == 'flip'
    assert ic.sign_status(-0.05, 'ts_high', 4.0) == 'ok'
    assert ic.sign_status(+0.0008, 'ts_high') == 'flip'
    assert ic.sign_status(+0.0008, 'ts_high', np.nan) == 'flip'
    assert ic.sign_status(0.0, 'ts_high', 0.1) == 'inconclusive'
    assert ic.sign_status(0.9, 'dur_mean', 9.0) == 'no_prior'

def test_ic_fold_is_a_time_slice_not_just_a_symbol_pool():
    """一折必须**既切品种池也切时间**。

    这里曾经是个空转的闸门：`years` 只被当成品种池的键，`factor[s]` / `fwd[s]` 传的
    是完整历史，于是六折"逐折 IC"是同一个全样本 IC 的六个品种池变体，折间一致性看
    起来好得离谱。只切品种池、不切时间的向后窗口也会把研究期算进去。
    时序 t 值的前提就是"一折 = 一段时间"，所以造一个前后反向的样本来钉住它。
    """
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
    """时序 t 与横截面 t 的差别，用一个共同驱动的样本量化出来。

    商品之间同期高度相关（同一波宏观冲击推动整个板块）。跨品种口径把这 8 个品种当成
    8 个独立样本，分母是品种间 IC 的标准误——品种越同质它越小，t 越大，极限情况下
    "再加一个高度相关的品种"就能把 t 抬上去，这显然不是显著性。时序口径先在月内对
    品种取平均，一个月只贡献一个观测，分母来自时间上的变异。
    """
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
    """随机游走上的 RSI 不可能有预测力，时序 IC 必须测不出东西。

    旧口径在月内算 Spearman，对日间高度持续的因子，月内去均值带来 Stambaugh 型
    负偏差：这个样本上会给出 ic_ts≈-0.20、t≈-49，量价因子的"显著反转"就是这么来的。
    """
    from tfcta.factors import daily
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
    """正自相关时 NW 标准误必须大于普通标准误，否则 t 值虚高。

    lag=0 时它应当退化成总体标准差 / sqrt(n)，这条顺带钉住权重写法没写反。
    """
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
