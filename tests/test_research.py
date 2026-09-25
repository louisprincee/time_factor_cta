"""第 5-11 步的口径测试。

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
from tfcta.research import combo, costs, folds, ic, metrics, returns


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


def test_position_earns_next_day_not_same_day():
    idx = pd.bdate_range('2016-01-04', periods=4)
    sig = pd.DataFrame({'RB': [1.0, 0.0, 0.0, 0.0]}, index=idx)
    pos = returns.execute_position(sig)
    assert np.isnan(pos['RB'].iloc[0])
    assert pos['RB'].iloc[1] == 1.0
    day_ret = pd.DataFrame({'RB': [0.05, -0.01, 0.02, 0.03]}, index=idx)
    net = combo.symbol_net(pos, day_ret, fee=0.0)
    # 第一天还没持仓；第二天赚的是 -0.01，不是第一天的 0.05
    assert net['RB'].iloc[0] == 0.0
    assert net['RB'].iloc[1] == pytest.approx(-0.01)


def test_fee_is_one_way_turnover():
    idx = pd.bdate_range('2016-01-04', periods=4)
    pos = pd.DataFrame({'RB': [0.0, 1.0, 1.0, -1.0]}, index=idx)
    day_ret = pd.DataFrame({'RB': [0.01, 0.01, 0.01, 0.01]}, index=idx)
    fee = 0.001
    net = combo.symbol_net(pos, day_ret, fee)
    # 换手 0, 1, 0, 2
    assert net['RB'].iloc[0] == pytest.approx(0.0)
    assert net['RB'].iloc[1] == pytest.approx(0.01 - fee)
    assert net['RB'].iloc[2] == pytest.approx(0.01)
    # 当天仓位是 -1，毛收益为 -0.01，再扣两次单边费
    assert net['RB'].iloc[3] == pytest.approx(-0.01 - 2 * fee)


def test_portfolio_ignores_symbols_outside_the_year():
    idx = pd.to_datetime(['2016-01-04', '2017-01-04'])
    net = pd.DataFrame({'RB': [0.02, 0.02], 'CU': [0.10, 0.10]}, index=idx)
    universe = {2016: ['RB'], 2017: ['RB', 'CU']}
    port = combo.portfolio_return(net, universe)
    assert port.iloc[0] == pytest.approx(0.02)
    assert port.iloc[1] == pytest.approx(0.06)


def test_pool_exit_fee_lands_inside_the_pool():
    """退池那笔平仓费必须收在最后一个在池日，否则会掉进年度边界。

    落在第一个池外日的话，portfolio_return 只按当年池内品种取平均，
    那一行不进分母，费用就凭空消失了。
    """
    idx = pd.to_datetime(['2016-12-29', '2016-12-30', '2017-01-03', '2017-01-04'])
    day_ret = pd.DataFrame({'RB': [0.0] * 4, 'CU': [0.0] * 4}, index=idx)
    pos = pd.DataFrame({'RB': [1.0] * 4, 'CU': [1.0] * 4}, index=idx)
    universe = {2016: ['RB', 'CU'], 2017: ['CU']}    # RB 在 2016 年末退池
    fee = 0.001
    port = combo.run_book(pos, day_ret, universe, fee)
    # 12-29 两个品种各建仓一次；12-30 是 RB 最后一个在池日，只有 RB 付平仓费
    assert port.iloc[0] == pytest.approx(-fee)
    assert port.iloc[1] == pytest.approx(-fee / 2)
    assert port.iloc[2] == pytest.approx(0.0)


def test_mid_series_nan_gap_is_not_charged_twice():
    """信号中途断掉产生的 NaN 仓位已经在池内收过费，不能再补一笔。"""
    idx = pd.to_datetime(['2016-12-29', '2016-12-30', '2017-01-03', '2017-01-04'])
    day_ret = pd.DataFrame({'RB': [0.0] * 4}, index=idx)
    pos = pd.DataFrame({'RB': [1.0, np.nan, np.nan, 1.0]}, index=idx)
    fee = 0.001
    port = combo.run_book(pos, day_ret, {2016: ['RB'], 2017: ['RB']}, fee)
    assert port.iloc[0] == pytest.approx(-fee)     # 建仓
    assert port.iloc[1] == pytest.approx(-fee)     # 平掉，只收一次
    assert port.iloc[2] == pytest.approx(0.0)
    assert port.iloc[3] == pytest.approx(-fee)     # 重新建仓；末日不算退池


def test_metrics_hand_values():
    # 两天 +10%、-5%：净值 1.1 * 0.95 = 1.045
    r = pd.Series([0.10, -0.05])
    m = metrics.performance(r, periods=252)
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


def test_walk_forward_folds():
    rows = folds.walk_forward_folds()
    assert [r['test_year'] for r in rows] == [2016, 2017, 2018, 2019, 2020, 2021]
    assert rows[0]['train_start'] == 2013 and rows[0]['train_end'] == 2015
    assert rows[0]['sparse_night'] and rows[1]['sparse_night']
    assert not rows[2]['sparse_night']


def test_ic_sign_on_perfect_forecast():
    idx = pd.bdate_range('2016-01-04', periods=80)
    rng = np.random.default_rng(0)
    day_ret = pd.DataFrame(rng.normal(0, 0.01, (80, 3)), index=idx, columns=list('ABC'))
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


def test_tick_estimate_takes_the_smallest_frequent_step_not_the_mode():
    """tick 取"出现得足够频繁的最小档"，众数会系统性偏大一倍。

    活跃品种最常见的分钟变动是**两个** tick 而不是一个：焦炭实测 1.0 出现 20.5 万次、
    真 tick 0.5 只有 15.1 万次，取众数就把 J 的滑点翻倍（菜油 OI 同理，2 对 1）。
    这里造一个同样形状的序列钉住方向。反过来只取最小差分也不行，见下一条。
    """
    tick = 0.5
    # 两个 tick 的步长比一个 tick 更常见，众数会落在 1.0
    steps = np.array([2, -2, 2, -2, 2, 1, -1, 2, -2, 1, 0, 3, -2, 2] * 60)
    px = 2600.0 + np.cumsum(steps) * tick
    est, mode = costs.estimate_tick(px)
    assert est == pytest.approx(tick)
    assert mode == pytest.approx(2 * tick)      # 众数确实偏大一倍
    # 全程横盘时没有非零差分可用，返回 NaN 而不是 0——0 会静默变成"零成本"
    flat, fmode = costs.estimate_tick(np.full(50, 2600.0))
    assert np.isnan(flat) and np.isnan(fmode)


def test_tick_estimate_ignores_rare_steps_and_float32_noise():
    """低频档必须被频次下限滤掉，同一 tick 的浮点变体必须先归并。

    PVC 的 tick 是 5，但样本里有 1569 次 1.0 的跳动（占最高频档的 0.5%）——只取最小
    差分会把 V 的成本估高 5 倍。另一头，分钟价格是 float32 存的，黄金的 0.05 裂成
    0.049988 / 0.050003 / 0.050018 三个桶，不先按相对容差归并的话频次被摊薄，
    估计就退化成噪声。
    """
    rng = np.random.default_rng(5)
    steps = np.concatenate([np.repeat([1.0, -1.0], 3000), np.full(12, 0.2)])
    px = 8000.0 + np.cumsum(rng.permutation(steps)) * 5.0    # tick 5，掺 12 次 1.0
    est, _ = costs.estimate_tick(px)
    assert est == pytest.approx(5.0)
    # 同一个 tick 的三个浮点变体各自频次都不够，归并后才够
    noisy = np.cumsum(np.tile([0.049988, -0.050003, 0.050018, -0.049988], 300))
    est2, _ = costs.estimate_tick(275.0 + noisy)
    assert est2 == pytest.approx(0.05, rel=1e-3)


def test_slippage_is_tick_relative_so_cheap_symbols_cost_more():
    """同样穿一个 tick，低价位品种的比例成本必须更高。

    这就是不能把滑点写成一个固定比例数的原因：那样等于宣称玉米和铜的交易成本一样，
    而实测差三倍以上。等权组合的成本恰恰由低价位品种主导，抹平它会系统性低估成本。
    """
    tab = pd.DataFrame([
        {'symbol': 'C', 'year': 2016, 'tick': 1.0, 'median_close': 1600.0,
         'rate_per_tick': 1.0 / 1600.0},
        {'symbol': 'C', 'year': 2017, 'tick': 1.0, 'median_close': 2800.0,
         'rate_per_tick': 1.0 / 2800.0},
        {'symbol': 'CU', 'year': 2016, 'tick': 10.0, 'median_close': 70000.0,
         'rate_per_tick': 10.0 / 70000.0},
    ])
    idx = pd.to_datetime(['2016-06-01', '2017-06-01'])
    w = costs.slippage_wide(tab, idx, ['C', 'CU'], n_ticks=2.0)
    assert w.loc[idx[0], 'C'] == pytest.approx(2.0 / 1600.0)
    # 6.25bp/tick 对 1.43bp/tick，相差 4.4 倍
    assert w.loc[idx[0], 'C'] > 4 * w.loc[idx[0], 'CU']
    # 价位逐年变，比例成本跟着变：玉米 2017 年涨到 2800，同一个 tick 便宜了
    assert w.loc[idx[1], 'C'] == pytest.approx(2.0 / 2800.0)
    # CU 缺 2017 年（上市晚/退池早都可能），按最近有效年份补，不留 NaN
    assert w.loc[idx[1], 'CU'] == pytest.approx(2.0 * 10.0 / 70000.0)
    # 0 档直接给 0，不去建 tick 表
    assert (costs.slippage_wide(tab, idx, ['C'], n_ticks=0.0) == 0.0).all().all()
    # 整个品种不在表里必须报错。静默返回 NaN 会让该品种净值全 NaN、在等权里被跳过，
    # 结果是成本变成 0——方向恰好是低估。
    with pytest.raises(KeyError):
        costs.slippage_wide(tab, idx, ['C', 'RB'], n_ticks=1.0)


def test_slippage_is_charged_on_the_same_turnover_as_the_fee():
    idx = pd.bdate_range('2016-01-04', periods=4)
    pos = pd.DataFrame({'RB': [0.0, 1.0, 1.0, -1.0]}, index=idx)
    day_ret = pd.DataFrame({'RB': [0.0] * 4}, index=idx)
    slip = pd.DataFrame({'RB': [0.002] * 4}, index=idx)
    fee = 0.001
    net = combo.symbol_net(pos, day_ret, fee, slippage=slip)
    # 换手 0, 1, 0, 2；每单位换手扣 fee + slip
    assert net['RB'].iloc[0] == pytest.approx(0.0)
    assert net['RB'].iloc[1] == pytest.approx(-(fee + 0.002))
    assert net['RB'].iloc[2] == pytest.approx(0.0)
    assert net['RB'].iloc[3] == pytest.approx(-2 * (fee + 0.002))
    # 给标量就退化成"把费率调大"，这也是为什么标量版本没有研究价值
    flat = combo.symbol_net(pos, day_ret, fee, slippage=0.002)
    pd.testing.assert_frame_equal(net, flat)


def test_missing_slippage_rate_cannot_poison_a_zero_turnover_day():
    """``NaN × 0`` 在 numpy 里是 NaN，不能让它渗进净值。

    渗进去的话那一天该品种是 NaN，``portfolio_return`` 把它剔出分母——等权组合看起来
    毫发无损，成本却凭空消失了，而且是静默的。无换手的格子必须硬性记 0。
    """
    idx = pd.bdate_range('2016-01-04', periods=3)
    pos = pd.DataFrame({'RB': [1.0, 1.0, 1.0]}, index=idx)
    day_ret = pd.DataFrame({'RB': [0.01, 0.02, 0.03]}, index=idx)
    slip = pd.DataFrame({'RB': [np.nan] * 3}, index=idx)
    net = combo.symbol_net(pos, day_ret, fee=0.001, slippage=slip)
    # 第一天建仓（换手 1）费率缺失，只有这一天该是 NaN；后两天无换手，收益必须完好
    assert np.isnan(net['RB'].iloc[0])
    assert net['RB'].iloc[1] == pytest.approx(0.02)
    assert net['RB'].iloc[2] == pytest.approx(0.03)


def test_combo_skips_nan_instead_of_filling_zero():
    a = pd.DataFrame({'JD': [1.0, -1.0]})
    b = pd.DataFrame({'JD': [np.nan, np.nan]})
    out = combo.average_signals([a, b])
    assert out['JD'].iloc[0] == pytest.approx(1.0)
    assert out['JD'].iloc[1] == pytest.approx(-1.0)
