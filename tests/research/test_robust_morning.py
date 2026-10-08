"""Causality and allocation properties for the bounded morning experiment."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def module():
    return importlib.import_module('research_robust_morning')


def features():
    dates = pd.bdate_range('2015-01-01', periods=85)
    return pd.DataFrame({'trading_date': dates, 'symbol': 'A',
        'dev16': np.arange(85) / 10000, 'range15': .01,
        'volume20': 100., 'gross17': np.sin(np.arange(85)) / 100})


def test_historical_scales_do_not_see_today_or_future():
    m = module()
    original = features()
    changed = original.copy()
    changed.loc[70:, ['dev16', 'range15', 'volume20', 'gross17']] = 10.
    a, b = m.add_history(original), m.add_history(changed)
    cols = ['q80', 'q90', 'median_range', 'median_volume', 'prior_sigma']
    np.testing.assert_allclose(a.loc[:70, cols], b.loc[:70, cols], equal_nan=True)


def test_filter_does_not_reallocate_unused_risk_budget():
    m = module()
    dates = pd.DatetimeIndex(['2020-01-02'])
    names = ['A', 'B', 'C', 'M', 'Y', 'P', 'RB', 'HC', 'I', 'CU']
    signal = pd.DataFrame(0., index=dates, columns=names)
    signal.loc[:, ['A', 'CU']] = 1.
    sigma = signal * 0 + .1
    universe = {2020: names}
    before = m.risk_weights(signal, sigma, universe)
    signal['CU'] = 0.
    after = m.risk_weights(signal, sigma, universe)
    assert before.loc[dates[0], 'A'] == after.loc[dates[0], 'A']
    assert after.abs().sum(axis=1).iloc[0] < before.abs().sum(axis=1).iloc[0]


def test_missing_risk_estimate_leaves_cash_and_caps_are_respected():
    m = module()
    dates = pd.DatetimeIndex(['2020-01-02'])
    signal = pd.DataFrame(1., index=dates, columns=['A', 'B', 'M', 'Y'])
    sigma = signal * 0 + .001
    sigma['B'] = np.nan
    weight = m.risk_weights(signal, sigma, {2020: list(signal)})
    assert weight['B'].iloc[0] == 0.
    assert weight.abs().to_numpy().max() <= .1
    assert weight.abs().sum(axis=1).iloc[0] <= .25 + 1e-12


def test_confirmation_requires_extreme_to_stop_and_price_to_pull_back():
    m = module()
    row = dict(dev16=.01, q80=.005, q90=.008, estimated16=.0001,
        estimated20=.0001, high_clock=.9, low_clock=.9, er=.2,
        close16=101., close20=100., high16=102., high_after=102.,
        low16=98., low_after=98., high15=100., low15=98., tick=1.,
        range15=.02, median_range=.03, volume20=100., median_volume=100., whole_clock=.9)
    frame = pd.DataFrame([row])
    assert m.make_signals(frame)['C_confirm80'][0] == -1.
    frame.loc[0, 'high_after'] = 103.
    assert m.make_signals(frame)['C_confirm80'][0] == 0.


def test_outcome_changes_cannot_change_decision():
    m = module()
    frame = pd.DataFrame([dict(dev16=.01, q80=.005, q90=.008,
        estimated16=.0001, estimated20=.0001, high_clock=.9, low_clock=.9,
        whole_clock=.9, er=.2, close16=101., close20=100., high16=102.,
        high_after=102., low16=98., low_after=98., high15=100., low15=98.,
        tick=1., range15=.02, median_range=.03, volume20=100., median_volume=100.,
        gross17=.01, gross21=.01, cost17=.001, cost21=.001)])
    before = m.make_signals(frame)
    frame.loc[:, ['gross17', 'gross21', 'cost17', 'cost21']] = np.nan
    for name, value in m.make_signals(frame).items():
        np.testing.assert_array_equal(value, before[name])


def test_walkforward_selection_cannot_see_test_year_or_2022():
    m = module()
    dates = pd.bdate_range('2016-01-01', '2022-12-31')
    oscillation = np.sin(np.arange(len(dates))) * .001
    columns = [f'{family}_{version}' for family in 'ABCD' for version in (1, 2)]
    daily = pd.DataFrame({name: oscillation + (.0001 if name.endswith('1') else .00005) for name in columns}, index=dates)
    trades = daily * 0 + 1
    paths, choices = m.walkforward(daily, daily, daily, trades)
    changed = daily.copy()
    changed.loc[dates.year >= 2019, :] *= -100
    _, changed_choices = m.walkforward(changed, changed, changed, trades)
    assert [c for c in choices if c['test_year'] == 2019] == [c for c in changed_choices if c['test_year'] == 2019]
    changed = daily.copy()
    changed.loc[dates.year == 2022, :] *= -100
    changed_paths, changed_choices = m.walkforward(changed, changed, changed, trades)
    assert choices == changed_choices
    pd.testing.assert_frame_equal(paths[0], changed_paths[0])
