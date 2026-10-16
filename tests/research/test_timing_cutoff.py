import numpy as np
import pandas as pd

from research.timing_cutoff import _at, path_features

def _window(end="09:05"):
    index = pd.date_range(f"2020-01-02 09:00", f"2020-01-02 {end}", freq="min")
    close = 100 + np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "volume": np.arange(len(index), dtype=float) + 10,
        },
        index=index,
    )


def test_path_features_require_complete_minute_window():
    complete = _window()
    missing_bar = complete.drop(pd.Timestamp("2020-01-02 09:03"))

    assert np.isfinite(path_features(complete)["open_move"])
    assert np.isnan(path_features(missing_bar)["open_move"])


def test_features_are_limited_to_the_supplied_cutoff_window():
    early = _window("09:05")
    later = pd.concat([early, _window("09:10").loc["2020-01-02 09:06":]])

    early_features = path_features(early)
    later_features = path_features(later)

    assert early_features["open_move"] == path_features(early)["open_move"]
    assert later_features["open_move"] > early_features["open_move"]


def test_cutoff_timestamp_and_path_efficiency_match_observed_session_window():
    day = pd.Timestamp("2020-01-02")
    features = path_features(_window())

    assert _at(day, "09:10") == pd.Timestamp("2020-01-02 09:10")
    assert np.isclose(features["path_efficiency"], 1.0)
    assert np.isclose(features["tail_move_5m"], 5 / 100)


def test_tail_move_requires_six_bars_to_measure_five_minute_change():
    features = path_features(_window("09:04"))

    assert np.isnan(features["tail_move_5m"])
