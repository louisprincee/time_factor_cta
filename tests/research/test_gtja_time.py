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


def test_open_to_open_divides_the_adjusted_gap_by_the_raw_open():
    stamps = pd.to_datetime(["2021-01-04 09:31", "2021-01-04 15:00", "2021-01-05 09:31", "2021-01-05 15:00"])
    frame = pd.DataFrame({
        "trading_date": pd.to_datetime(["2021-01-04", "2021-01-04", "2021-01-05", "2021-01-05"]),
        "open": [100.0, 101.0, 110.0, 111.0],
        "openw": [1100.0, 1101.0, 1110.0, 1111.0],
    }, index=stamps)
    got = gtja.open_to_open(frame)
    assert np.isclose(got.iloc[0], 0.10)
    assert np.isnan(got.iloc[1])


def test_representative_price_return_uses_the_previous_raw_price():
    day0 = pd.Timestamp("2021-01-04")
    day1 = pd.Timestamp("2021-01-05")
    frames = []
    for day, raw in ((day0, [100.0, 100.0, 100.0, 100.0]), (day1, [110.0, 110.0, 110.0, 130.0])):
        index = pd.date_range(f"{day:%Y-%m-%d} 09:31", periods=4, freq="min")
        frames.append(pd.DataFrame({
            "trading_date": day, "close": raw, "closew": np.array(raw) + 1000.0,
            "open": raw, "openw": np.array(raw) + 1000.0,
            "highw": raw, "loww": raw, "volume": [1.0, 1.0, 1.0, 1.0],
            "total_turnover": raw,
        }, index=index))
    out = gtja.daily_factors(pd.concat(frames), price_threshold=3.0, volume_threshold=50.0)
    # 第二天最后一根相对 110 跳出 3 点；最长持续期仍在 110，前一日代表价原始值是 100。
    assert np.isclose(out.loc[day1, "Price_duration_return"], (1110.0 - 1100.0) / 100.0)


def test_config_locks_the_research_sample_and_the_published_thresholds():
    path = Path(__file__).resolve().parents[2] / "config" / "research" / "gtja_time_factors.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert "不读 2022" in config["sample"]
    assert config["price_threshold"] == 3.0
    assert config["volume_threshold"] == 50.0
    assert config["status"] == "factor_research_only"
    assert len(gtja.COLUMNS) == 28
