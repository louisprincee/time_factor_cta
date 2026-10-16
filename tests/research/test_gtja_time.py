"""国泰君安时间因子的定义、空仓规则和次日开盘成交。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tfcta.index import gtja_time as gtja


def test_duration_stops_at_the_nearest_strict_breach_and_restarts_each_day():
    got = gtja.duration_one_day(np.array([0.0, 1.0, 2.0, 6.0]), 3.0)
    np.testing.assert_array_equal(got, [0, 1, 2, 1])
    exact = gtja.duration_one_day(np.array([0.0, 3.0]), 3.0)
    np.testing.assert_array_equal(exact, [0, 1])


def test_open_close_gap_uses_the_report_clock_windows():
    minute = np.array([9 * 60 + 31, 10 * 60, 14 * 60 + 31, 15 * 60])
    duration = np.array([1.0, 3.0, 4.0, 8.0])
    assert gtja._window_gap(duration, minute) == 4.0


def test_later_observation_does_not_change_an_earlier_zscore():
    values = pd.Series(np.arange(30, dtype=float))
    before = gtja.rolling_z(values, window=10, min_periods=10)
    bumped = values.copy()
    bumped.iloc[-1] = 10_000
    after = gtja.rolling_z(bumped, window=10, min_periods=10)
    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])


def test_config_locks_the_research_sample_and_the_published_thresholds():
    path = Path(__file__).resolve().parents[2] / "config" / "research" / "gtja_time_factors.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert "不读 2022" in config["sample"]
    assert config["price_threshold"] == 3.0
    assert config["volume_threshold"] == 50.0
    assert config["status"] == "factor_research_only"
    assert len(gtja.COLUMNS) == 28
