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


def test_pair_signal_stays_flat_unless_both_sides_agree():
    frame = pd.DataFrame({
        "time_to_low": [1.0, -1.0, 1.0, np.nan],
        "Volume_duration_extreme": [-1.0, 1.0, 1.0, -1.0],
    })
    got = gtja.pair_signal(frame, {"time_to_low": 1, "Volume_duration_extreme": -1})
    np.testing.assert_array_equal(got.to_numpy(), [1.0, -1.0, 0.0, 0.0])


def test_later_observation_does_not_change_an_earlier_zscore():
    values = pd.Series(np.arange(30, dtype=float))
    before = gtja.rolling_z(values, window=10, min_periods=10)
    bumped = values.copy()
    bumped.iloc[-1] = 10_000
    after = gtja.rolling_z(bumped, window=10, min_periods=10)
    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])


def test_backtest_enters_next_open_and_charges_only_when_the_position_changes():
    dates = pd.to_datetime(["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07"])
    signal = pd.Series([1.0, 1.0, 0.0, 0.0], index=dates)
    forward = pd.Series([0.10, 0.10, 0.0, np.nan], index=dates)
    result = gtja.backtest(signal, forward, fee_per_side=0.0002)
    assert result.index[-1] == dates[2]
    assert result.position.tolist() == [0.0, 1.0, 1.0]
    assert result.net.iloc[1] == 0.10 - 0.0002
    assert result.cost.iloc[2] == 0.0


def test_config_locks_the_research_sample_and_the_published_thresholds():
    path = Path(__file__).resolve().parents[2] / "config" / "index" / "gtja_time.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert "不读 2022" in config["sample"]
    assert config["price_threshold"] == 3.0
    assert config["volume_threshold"] == 50.0
    assert {rule["symbol"] for rule in config["rules"]} == {"IF", "IH", "IC"}
    assert len(config["rules"]) == 23
