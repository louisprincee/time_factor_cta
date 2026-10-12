import importlib
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
core = importlib.import_module('morning')


def test_v1_keeps_original_late_extreme_reversal_and_night_alignment():
    table = pd.DataFrame({
        'member': [True] * 5,
        'ret_night': [0.01, -0.01, -0.01, 0.01, 0.01],
        'dev15': [0.01, -0.01, 0.01, 0.001, 0.01],
        'est_cost': [0.0005, 0.0005, 0.0005, 0.0005, 0.0007],
        'relvol': [1.0] * 5,
        'hclock_td': [0.7, 0.5, 0.9, 0.9, 0.9],
        'lclock_td': [0.5, 0.7, 0.5, 0.5, 0.5],
    })
    rule = {'dev_threshold': 0.002, 'max_est_cost': 0.0006,
            'clock': 0.6667, 'relvol_max': 1.5}

    side = core.sides(table, rule)

    assert side.tolist() == [-1.0, 1.0, 0.0, 0.0, 0.0]


def test_v2_uses_two_one_fifth_tails_and_retains_original_filters():
    table = pd.DataFrame({
        'member': [True] * 8,
        'ret_night': [0.01, 0.01, -0.01, -0.01, -0.01, 0.01, -0.01, 0.01],
        'dev15': [0.01, 0.01, 0.01, -0.01, -0.01, -0.01, 0.001, 0.01],
        'est_cost': [0.0005] * 7 + [0.0007],
        'relvol': [1.0] * 8,
        'hclock_td': [0.1, 0.9, 0.9, 0.5, 0.5, 0.5, 0.1, 0.1],
        'lclock_td': [0.5, 0.5, 0.5, 0.1, 0.9, 0.5, 0.5, 0.1],
        'gross_1130': [0.01] * 8,
        'cost_1130': [0.0005] * 8,
        'sigma': [0.01] * 8,
    })
    rule = {'dev_threshold': 0.002, 'max_est_cost': 0.0006, 'relvol_max': 1.5}

    side = core.v2_sides(table, rule)

    assert side.tolist() == [1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]


def test_loss_switch_flips_only_after_the_losing_window_is_known():
    table = pd.DataFrame({
        'trading_date': pd.to_datetime(['2021-01-04', '2021-01-05', '2021-01-06']),
        'symbol': 'RB',
        'gross_1130': [0.01, 0.01, 0.01],
        'cost_1130': [0.0, 0.0, 0.0],
    })
    side = pd.Series([-1.0, -1.0, -1.0])

    assert core.loss_switch(table, side, 2).tolist() == [-1.0, -1.0, 1.0]


def test_pure_clock_benchmark_uses_clock_spread_and_shared_execution_filters():
    table = pd.DataFrame({
        'member': [True] * 7,
        'ret_night': [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01],
        'hclock_td': [0.10, 0.50, 0.95, 0.20, 0.05, 0.10, 0.10],
        'lclock_td': [0.90, 0.30, 0.10, 0.85, 0.90, 0.95, 0.70],
        'est_cost': [0.0005] * 7,
        'relvol': [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
        'gross_1130': [0.01] * 7,
        'cost_1130': [0.0005] * 7,
        'sigma': [0.01] * 7,
    })
    rule = {'max_est_cost': 0.0006, 'relvol_max': 1.5}
    threshold = core.load()['benchmark']['threshold']

    assert threshold == 0.8
    side = core.benchmark_sides(table, rule, threshold)

    # 差为 0.80 / −0.20 / −0.85 / 0.65 / 0.85 / 0.85 / 0.60。
    # 0.65 和 0.60 在 0.6 档会交易，在 0.8 档不交易；第 5 行量能超限；第 6 行夜盘方向相反仍做多。
    assert side.tolist() == [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0]
