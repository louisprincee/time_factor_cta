import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
from morning_cross_market import index_morning_rows


def synthetic_index_day(day, offset=0.0):
    index = pd.date_range(f"{day} 09:31", f"{day} 11:30", freq="min")
    trend = np.arange(len(index), dtype=float) * 0.1 + offset
    return pd.DataFrame(
        {
            "trading_date": pd.Timestamp(day),
            "openw": 100 + trend,
            "closew": 100 + trend + 0.02,
            "highw": 100 + trend + 0.08,
            "loww": 100 + trend - 0.06,
            "volume": 100 + np.arange(len(index), dtype=float),
        },
        index=index,
    )


def test_index_features_use_only_the_first_16_bars_and_target_0947_to_1130():
    frame = pd.concat(
        [synthetic_index_day("2021-01-04"), synthetic_index_day("2021-01-05", 1.0)]
    )
    base = index_morning_rows(frame, "IH").set_index("trading_date")

    changed = frame.copy()
    later = changed.index >= pd.Timestamp("2021-01-05 09:47")
    changed.loc[later, ["highw", "loww", "volume"]] = [10000.0, 1.0, 1.0]
    changed.loc[changed.index >= pd.Timestamp("2021-01-05 09:48"), "closew"] += 10.0
    revised = index_morning_rows(changed, "IH").set_index("trading_date")

    day = pd.Timestamp("2021-01-05")
    for feature in (
        "open_move", "dev15", "early_clock_diff", "volume_16m", "volume_peak_clock",
        "price_duration_mean", "volume_duration_mean",
    ):
        assert revised.loc[day, feature] == base.loc[day, feature]
    assert revised.loc[day, "gross_1130"] != base.loc[day, "gross_1130"]


def test_index_features_require_a_complete_16_minute_window():
    frame = synthetic_index_day("2021-01-04")
    frame = frame.drop(pd.Timestamp("2021-01-04 09:40"))
    result = index_morning_rows(frame, "IH").iloc[0]

    assert np.isnan(result["open_move"])
    assert np.isnan(result["early_clock_diff"])
    assert np.isnan(result["gross_1130"])
