"""高低点时钟个例：分类、抽样、截面回归和异质性检验。不读行情。"""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts' / 'commodity'))
cases = importlib.import_module('morning_clock_cases')


def frame_of(diff, gross, years=None):
    n = len(diff)
    years = years if years is not None else [2018] * n
    return pd.DataFrame({
        'diff': diff, 'gross_1130': gross, 'year': years,
        'trading_date': pd.to_datetime([f'{y}-01-01' for y in years]) + pd.to_timedelta(np.arange(n), 'D'),
    })


def test_label_cases_signs_returns_by_expected_direction_and_drops_flat_days():
    table = frame_of([0.9, -0.9, 0.9, -0.9, 0.1, 0.9], [0.01, 0.01, -0.02, -0.02, 0.05, 0.0])
    out = cases.label_cases(table, 0.6)
    assert list(out.side) == [1, -1, 1, -1]          # 中间段和收益为 0 的日子不进样本
    assert list(out.signed > 0) == [True, False, False, True]


def test_pick_cases_takes_two_days_per_category_and_year_and_is_repeatable():
    rng = np.random.default_rng(0)
    n = 400
    table = frame_of(rng.choice([0.9, -0.9], n), rng.normal(0, 0.01, n), years=[2018] * 200 + [2019] * 200)
    first = cases.pick_cases(table, seed=1)
    again = cases.pick_cases(table, seed=1)
    assert first.groupby(['类别', 'year']).size().eq(2).all()
    assert len(first) == len(cases.CATEGORIES) * 2 * 2
    pd.testing.assert_frame_equal(first, again)
    for _, row in first.iterrows():
        spec = cases.CATEGORIES[row.类别]
        assert np.sign(row['diff']) == spec['side']
        assert (spec['side'] * row.gross_1130 > 0) == spec['win']


def test_rank_regression_recovers_a_cross_sectional_rank_link_and_ignores_day_level_moves():
    rng = np.random.default_rng(3)
    days = np.repeat(pd.date_range('2020-01-01', periods=200), 20)
    x = rng.normal(size=len(days))
    market = np.repeat(rng.normal(0, 5, 200), 20)       # 当天整体涨跌，截面排名后应被去掉
    y = market + x + rng.normal(0, 0.5, len(days))
    frame = pd.DataFrame({'trading_date': days, 'diff': x, 'gross_1130': y})
    coef, t_stat = cases.rank_regression(frame, ['diff'])['diff']
    assert coef > 0.8 and t_stat > 20


def test_heterogeneity_q_is_small_for_equal_years_and_large_for_shifted_years():
    rng = np.random.default_rng(5)
    index = pd.date_range('2016-01-01', '2021-12-31', freq='D')
    same = pd.Series(rng.normal(0.02, 0.2, len(index)), index=index)
    _, p_same = cases.heterogeneity(same)
    shifted = same + np.where(index.year % 2 == 0, 0.1, -0.1)
    _, p_shift = cases.heterogeneity(shifted)
    assert p_same > 0.01 and p_shift < 1e-6


def test_pre_path_points_start_at_first_bar_and_end_at_decision():
    points = cases.pre_path_points(10, 130)
    assert points[0] == 10 and points[-1] == 130 and len(points) == cases.PRE_POINTS
    assert np.all(np.diff(points) >= 0)


def test_day_grid_runs_from_entry_to_close_and_ticks_exist():
    assert cases.DAY_GRID[0] == (9, 17) and cases.DAY_GRID[-1] == (15, 0)
    assert (10, 20) not in cases.DAY_GRID and (12, 0) not in cases.DAY_GRID
    assert all(hm in cases.DAY_GRID for hm in cases.TICKS)
    assert len(cases.DAY_GRID) == pytest.approx(59 + 60 + 90, abs=0)
