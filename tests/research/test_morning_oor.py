"""开盘过度反应回归：特征因果性、信号、仓位上限、成本与熔断。"""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
features = importlib.import_module('morning_features')
oor = importlib.import_module('research_morning_oor')

RULE = {'dev_threshold': .002, 'clock': .6667, 'relvol_max': 1.5, 'max_est_cost': .0006}
SIZING = {'risk_per_trade': .01, 'max_weight': 2., 'max_sector_gross': 3., 'max_gross': 4.}


def minutes(days, close_of):
    """每天 21:00–21:59 夜盘、09:00–11:30 和 13:30–15:00 日盘的一分钟线。"""
    parts = []
    for k, day in enumerate(days):
        night = pd.date_range(day - pd.Timedelta(1, unit='D') + pd.Timedelta(21, unit='h'), periods=60, freq='min')
        am = pd.date_range(day + pd.Timedelta(9, unit='h'), day + pd.Timedelta(11, unit='h') + pd.Timedelta(30, unit='min'), freq='min')
        pm = pd.date_range(day + pd.Timedelta(13, unit='h') + pd.Timedelta(30, unit='min'), day + pd.Timedelta(15, unit='h'), freq='min')
        idx = night.append(am).append(pm)
        close = close_of(k, idx)
        parts.append(pd.DataFrame({'close': close, 'open': close, 'closew': close, 'openw': close,
                                   'highw': close, 'loww': close, 'volume': 10., 'trading_date': day}, index=idx))
    return pd.concat(parts)


def run_rows(monkeypatch, frame):
    monkeypatch.setattr(features, 'load_minutes', lambda *args: frame.copy())
    days = pd.DatetimeIndex(sorted(frame.trading_date.unique()))
    fees = pd.DataFrame({'symbol': 'RB', 'trading_date': days, 'commission_type': 'by_money',
                         'open_commission': 1e-4, 'close_commission_today': 1e-4})
    ticks = pd.DataFrame({'symbol': ['RB'], 'year': [2020], 'tick': [1.]})
    return features.symbol_rows('RB', 'research', fees, ticks).set_index('trading_date')


def test_decision_features_ignore_bars_after_0916(monkeypatch):
    days = pd.DatetimeIndex(['2021-03-01', '2021-03-02'])
    base = lambda k, idx: 4000. + np.arange(len(idx)) * (1 + k)
    frame = minutes(days, base)
    changed = frame.copy()
    late = (changed.trading_date == days[1]) & (changed.index >= days[1] + pd.Timedelta(10, unit='h'))
    changed.loc[late, ['close', 'closew', 'highw', 'loww', 'open', 'openw']] *= 1.05
    changed.loc[late, 'volume'] = 999.
    a, b = run_rows(monkeypatch, frame), run_rows(monkeypatch, changed)
    decision = ['ref', 'ret_on', 'ret_night', 'gap', 'ret_pre', 'dev15', 'pre_volume', 'vwap_dev',
                'eff_pre', 'range_pre', 'hclock_td', 'lclock_td', 'hclock_pre', 'lclock_pre', 'est_cost']
    pd.testing.assert_series_equal(a.loc[days[1], decision], b.loc[days[1], decision])
    assert a.loc[days[1], 'gross_1130'] != b.loc[days[1], 'gross_1130']


def test_fee_rate_conventions():
    assert features.fee_rate('by_money', 1e-4, 4000., 4000., 'RB') == pytest.approx(1.01e-4)
    assert features.fee_rate('by_volume', 3., 4000., 4000., 'RB') == pytest.approx(3.01 / 40000)
    assert features.fee_rate('by_money', 0., 4000., 4000., 'RB') == pytest.approx(.01 / 40000)
    assert np.isnan(features.fee_rate(None, 1e-4, 4000., 4000., 'RB'))


def test_history_uses_only_prior_days():
    n = 80
    table = pd.DataFrame({'symbol': 'A', 'trading_date': pd.bdate_range('2020-01-01', periods=n),
                          'pre_volume': np.arange(1., n + 1), 'gross_1130': np.sin(np.arange(n)) / 100,
                          'dev15': 0.})
    changed = table.copy()
    changed.loc[70:, ['pre_volume', 'gross_1130']] = 1e6
    a = oor.add_history(table.copy(), 20, 60)
    b = oor.add_history(changed, 20, 60)
    np.testing.assert_allclose(a.loc[:70, ['sigma']], b.loc[:70, ['sigma']], equal_nan=True)
    np.testing.assert_allclose(a.loc[:69, ['relvol']], b.loc[:69, ['relvol']], equal_nan=True)
    assert a.loc[20, "relvol"] == pytest.approx(21 / 10.5)


def signal_table():
    return pd.DataFrame({
        'dev15': [.003, .003, -.003, -.003, .003, .001],
        'hclock_td': [.9, .9, .9, .1, .1, .9],
        'lclock_td': [.1, .1, .9, .9, .9, .1],
        'ret_night': [.01, -.01, -.01, -.01, .01, .01],
        'relvol': [1., 1., 1., 1., 1., 1.],
        'est_cost': [.0003] * 6,
        'member': [True] * 6,
    })


def test_extreme_clock_matches_direction_of_the_move():
    side = oor.sides(signal_table(), RULE, {'clock': 'extreme', 'night_aligned': True, 'volume': True})
    # 上冲且高点在后、夜盘同向 → 做空；下杀且低点在后 → 做多；
    # 夜盘反向、高点在前、偏离不够都不做
    np.testing.assert_array_equal(side, [-1, 0, 1, 1, 0, 0])


def test_old_high_clock_reads_early_high_as_long():
    side = oor.sides(signal_table(), RULE, {'clock': 'high', 'night_aligned': False, 'volume': False})
    np.testing.assert_array_equal(side, [-1, -1, 0, 1, 0, 0])


def test_sector_deviation_leaves_the_symbol_itself_out():
    day = pd.Timestamp('2020-01-02')
    table = pd.DataFrame({
        'trading_date': day,
        'sector': ['有色', '有色', '有色', '化工'],
        'dev15': [.020, .004, .004, .030],
    })
    out = oor.add_sector_deviation(table, min_peers=2)
    assert out.loc[0, 'sector_dev'] == pytest.approx(.004)
    assert out.loc[0, 'resid_dev'] == pytest.approx(.016)
    assert np.isnan(out.loc[3, 'sector_dev'])


def test_residual_fade_requires_a_quiet_sector_and_follow_requires_a_common_move():
    table = signal_table()
    table['sector_dev'] = [.0005, .0005, -.004, .004, .0005, .0005]
    table['resid_dev'] = [.003, .003, -.003, -.003, .003, .001]
    main = {'clock': 'extreme', 'night_aligned': True, 'volume': True}
    fade = oor.sides(table, RULE, {**main, 'sector_residual': True})
    follow = oor.sides(table, RULE, {**main, 'sector_follow': True})
    # 板块安静且残差够大才逆向；板块一起动且偏离同号才顺向。第 3 行残差为负、夜盘为负，顺向做空。
    np.testing.assert_array_equal(fade, [-1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(follow, [0, 0, -1, 0, 0, 0])


def test_volume_filter_and_cost_filter_leave_cash():
    table = signal_table()
    table.loc[0, 'relvol'] = 2.
    table.loc[2, 'est_cost'] = .001
    table.loc[3, 'relvol'] = np.nan
    side = oor.sides(table, RULE, {'clock': 'extreme', 'night_aligned': True, 'volume': True})
    np.testing.assert_array_equal(side, [0, 0, 0, 0, 0, 0])


def test_weights_respect_symbol_sector_and_gross_caps():
    day = pd.Timestamp('2020-01-02')
    table = pd.DataFrame({'trading_date': day, 'sector': ['有色'] * 3 + ['化工'] * 3,
                          'sigma': [.001, .01, .01, .005, .005, .005]})
    side = pd.Series([1., -1., 1., 1., 1., -1.])
    w = oor.weights(table, side, SIZING)
    raw = np.minimum(2., .01 / table.sigma)  # 2, 1, 1, 2, 2, 2
    sector = np.r_[raw[:3] * min(1, 3 / 4), raw[3:] * min(1, 3 / 6)]
    expected = side * sector * min(1, 4 / sector.sum())
    np.testing.assert_allclose(w, expected)
    assert w.abs().sum() == pytest.approx(4.)


def test_signal_without_sigma_is_rejected():
    table = pd.DataFrame({'trading_date': pd.Timestamp('2020-01-02'), 'sector': ['有色'], 'sigma': [np.nan]})
    with pytest.raises(ValueError, match='波动'):
        oor.weights(table, pd.Series([1.]), SIZING)


def test_held_trade_needs_quotes_and_cash_does_not():
    day = pd.Timestamp('2020-01-02')
    table = pd.DataFrame({'trading_date': day, 'gross_1130': [np.nan, -.01], 'cost_1130': [np.nan, .001]})
    pnl = oor.daily(table, pd.Series([0., -.5]), '主口径', pd.DatetimeIndex([day, day + pd.Timedelta(1, unit='D')]))
    np.testing.assert_allclose(pnl, [.005 - .0005, 0.])
    with pytest.raises(ValueError, match='持仓'):
        oor.daily(table, pd.Series([.5, 0.]), '主口径', pd.DatetimeIndex([day]))


def test_breaker_halts_on_paper_drawdown_and_resumes():
    ret = pd.Series([-.05, -.05, .01, .2, .01])
    guarded = oor.breaker(ret, .08)
    # 第二天收盘账面回撤 9.75%，第三、四天停手；第四天账面回到高点，第五天恢复
    np.testing.assert_allclose(guarded, [-.05, -.05, 0., 0., .01])
