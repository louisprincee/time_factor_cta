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


def test_relations_match_the_extreme_to_the_direction_of_the_move():
    frame = pd.DataFrame({
        'dev15': [0.01, -0.01, 0.0],
        'gross_1130': [0.002, -0.003, 0.004],
        'hclock_td': [0.9, 0.1, 0.5],
        'lclock_td': [0.2, 0.8, 0.4],
        'trading_date': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06']),
    })
    up, down, aligned = clock.relations(frame)
    assert up['x'].tolist() == [0.9] and up['y'].tolist() == [0.002]
    assert down['x'].tolist() == [0.8] and down['y'].tolist() == [-0.003]
    np.testing.assert_allclose(aligned['x'], [0.9, 0.8])
    np.testing.assert_allclose(aligned['y'], [-0.002, -0.003])
