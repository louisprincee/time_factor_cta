import numpy as np
import pandas as pd

from research.index_close_discriminators import (
    FEATURES,
    _range_surprise_group_predictions,
    symbol_panel,
)


def _synthetic_daily(n=320):
    index = pd.bdate_range("2010-01-04", periods=n)
    trend = np.linspace(0.0, 8.0, n)
    oscillation = np.sin(np.arange(n) / 5.0)
    close = 3000 + trend + oscillation
    open_ = close - 0.3 * np.cos(np.arange(n) / 3.0)
    width = 8.0 + np.arange(n) % 7
    return pd.DataFrame(
        {
            "open": open_,
            "openw": open_,
            "close": close,
            "closew": close,
            "highw": close + width * 0.6,
            "loww": close - width * 0.4,
            "volume": 100000 + (np.arange(n) % 19) * 1000,
            "turnover": 1e9 + (np.arange(n) % 23) * 1e7,
        },
        index=index,
    )


def test_candidate_features_at_close_do_not_use_future_daily_bars():
    daily = _synthetic_daily()
    original = symbol_panel(daily)
    signal_day = daily.index[250]

    changed_future = daily.copy()
    changed_future.loc[signal_day + pd.offsets.BDay(1):, ["open", "openw", "close", "closew"]] *= 3
    changed = symbol_panel(changed_future)

    pd.testing.assert_series_equal(
        original.loc[signal_day, ["clv", *FEATURES]],
        changed.loc[signal_day, ["clv", *FEATURES]],
    )
    assert original.at[signal_day, "outcome_return"] != changed.at[signal_day, "outcome_return"]


def test_range_surprise_groups_use_training_only_tercile_cutoffs():
    index = pd.DatetimeIndex(
        [*pd.bdate_range("2010-01-04", periods=30), *pd.bdate_range("2011-01-04", periods=3)]
    )
    panel = pd.DataFrame(
        {
            "clv": [0.2] * len(index),
            "range_surprise": [*range(-15, 15), -10, 0, 10],
            "outcome_return": [-0.01, 0.01, 0.01] * 11,
        },
        index=index,
    )
    original = _range_surprise_group_predictions(
        panel,
        evaluation_years=(2011,),
        score_start=pd.Timestamp("2011-01-01"),
        score_end=pd.Timestamp("2011-12-31"),
    )

    changed_future = panel.copy()
    changed_future.loc[changed_future.index.year == 2011, "range_surprise"] = [100, 200, 300]
    changed = _range_surprise_group_predictions(
        changed_future,
        evaluation_years=(2011,),
        score_start=pd.Timestamp("2011-01-01"),
        score_end=pd.Timestamp("2011-12-31"),
    )

    np.testing.assert_allclose(
        original[["训练低切点", "训练高切点"]].to_numpy(),
        changed[["训练低切点", "训练高切点"]].to_numpy(),
    )
    assert original["分组"].tolist() == ["低", "中", "高"]
