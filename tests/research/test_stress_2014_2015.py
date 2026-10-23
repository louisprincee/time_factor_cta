import pandas as pd
import pytest

from research.stress_2014_2015 import load_protocol, stress_universe
from tfcta import config as C


def test_protocol_is_locked_to_frozen_books_and_recorded_pool_widths():
    protocol = load_protocol()
    assert protocol["status"] == "declared_reproduction_no_tuning"
    assert protocol["books"] == ["market_clock_slow_short"]
    assert protocol["universe"]["expected_pool_widths"] == {"2014": 11, "2015": 17}


def test_stress_universe_rejects_holdout_statistics():
    protocol = load_protocol()
    stats = pd.DataFrame({"year": [2013, 2014, 2022]})
    with pytest.raises(C.HoldoutViolation):
        stress_universe(stats, protocol)


def test_stress_universe_requires_the_predeclared_pool_widths():
    protocol = load_protocol()
    stats = pd.DataFrame({
        "symbol": ["A", "A", "B", "B"],
        "year": [2013, 2014, 2013, 2014],
        "turnover_median": [4e9] * 4,
        "valid_ratio": [0.99] * 4,
        "valid_days": [242] * 4,
        "night_class": ["no_night"] * 4,
    })
    with pytest.raises(ValueError, match="品种池"):
        stress_universe(stats, protocol)
