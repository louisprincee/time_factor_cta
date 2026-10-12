"""高低点位置与收益：聚类斜率和三条关系的方向。不读行情。"""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
clock = importlib.import_module('morning_clock_return')


def test_cluster_slope_recovers_the_line_and_withholds_a_single_day():
    day = np.arange(40)
    x = np.linspace(-1, 1, 40)
    out = clock.cluster_slope(day, x, 0.02 * x)
    assert out['斜率'] == pytest.approx(0.02, rel=1e-6)
    assert out['双尾p'] < 1e-6
    one_day = clock.cluster_slope(np.zeros(10), x[:10], x[:10])
    assert np.isnan(one_day['双尾p'])


def test_late_clock_minus_early_clock_is_the_difference_in_means():
    early = np.column_stack([np.arange(30), np.full(30, 0.1), np.zeros(30)])
    late = np.column_stack([np.arange(30, 60), np.full(30, 0.9), np.full(30, 0.01)])
    sample = np.vstack([early, late])
    out = clock.late_minus_early(sample[:, 0], sample[:, 1], sample[:, 2])
    assert out['斜率'] == pytest.approx(0.01)


def test_study_items_use_all_directions_and_unmodified_returns():
    frame = pd.DataFrame({
        'dev15': [0.01, -0.01, 0.0],
        'gross_1130': [0.002, -0.003, 0.004],
        'hclock_td': [0.9, 0.1, 0.5],
        'lclock_td': [0.2, 0.8, 0.4],
        'trading_date': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06']),
    })
    high, low, difference = clock.study_items(frame)
    np.testing.assert_allclose(high['x'], [0.9, 0.1, 0.5])
    np.testing.assert_allclose(high['y'], [0.002, -0.003, 0.004])
    np.testing.assert_allclose(low['x'], [0.2, 0.8, 0.4])
    np.testing.assert_allclose(low['y'], [0.002, -0.003, 0.004])
    np.testing.assert_allclose(difference['x'], [-0.7, 0.7, -0.1])
    np.testing.assert_allclose(difference['y'], [0.002, -0.003, 0.004])


def test_yearly_summary_matches_pooled_position_only_spearman():
    frame = pd.DataFrame({
        'trading_date': pd.to_datetime(['2020-01-02'] * 4 + ['2020-01-03'] * 4),
        'hclock_td': [0.9, 0.6, 0.4, 0.1, 0.1, 0.4, 0.6, 0.9],
        'lclock_td': [0.1, 0.3, 0.6, 0.9, 0.9, 0.6, 0.3, 0.1],
        'gross_1130': [0.01, 0.02, 0.03, 0.04] * 2,
        'dev15': [0.01, -0.01, 0.0, 0.01] * 2,
    })

    yearly = clock.yearly_pooled_summary(frame, '有夜盘', '全部').set_index('关系')

    for item in clock.study_items(frame):
        expected = clock.association(item['day'], item['x'], item['y'])
        actual = yearly.loc[item['关系']]
        assert actual['Spearman'] == pytest.approx(expected['Spearman'])
        if np.isnan(expected['Spearman_p']):
            assert np.isnan(actual['Spearman双尾p'])
        else:
            assert actual['Spearman双尾p'] == pytest.approx(expected['Spearman_p'])
        assert actual['笔数'] == expected['笔数']
        assert actual['天数'] == expected['天数']
