import numpy as np
import pandas as pd
import pytest

from research.index_market_clock import (
    FACTORS,
    SYMBOLS,
    _targets,
    load_protocol,
    require_index_carry_coverage,
)


def _synthetic_inputs(days=520):
    index = pd.bdate_range("2015-01-01", periods=days)
    rng = np.random.default_rng(17)
    signed = {
        factor: pd.DataFrame(
            rng.normal(size=(days, len(SYMBOLS))), index=index, columns=SYMBOLS
        )
        for factor in FACTORS
    }
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, (days, len(SYMBOLS))), axis=0)),
        index=index,
        columns=SYMBOLS,
    )
    bars = {"close": close, "closew": close - 25.0}
    carry = pd.DataFrame(0.1, index=index, columns=SYMBOLS)
    return index, signed, bars, carry


def test_protocol_declares_only_research_period_and_long_baseline_plus_current_book():
    protocol = load_protocol()
    assert protocol["status"] == "declared_test_not_a_strategy"
    assert protocol["instruments"] == list(SYMBOLS)
    assert protocol["books"] == ["market_clock_pool", "market_clock_slow_short_20_60_120"]
    assert protocol["parameters"]["short_trend_windows"] == [20, 60, 120]
    assert protocol["minimum_valid_instruments_for_market_score"] == 3


def test_index_carry_coverage_fails_above_five_percent():
    index = pd.bdate_range("2016-01-01", "2021-12-31")
    carry = pd.DataFrame(0.1, index=index, columns=SYMBOLS)
    require_index_carry_coverage(carry, index)
    carry.loc[index[index.year == 2016][:16], "IH"] = np.nan
    with pytest.raises(ValueError, match="IH 缺 6%"):
        require_index_carry_coverage(carry, index)


def test_index_targets_use_only_information_available_before_execution_open():
    index, signed, bars, carry = _synthetic_inputs()
    original = _targets(index, bars, signed, carry)
    changed_signed = {name: frame.copy() for name, frame in signed.items()}
    changed_bars = {name: frame.copy() for name, frame in bars.items()}
    for frame in changed_signed.values():
        frame.iloc[-1] = 100.0
    for frame in changed_bars.values():
        frame.iloc[-1] *= 2.0
    changed_carry = carry.copy()
    changed_carry.iloc[-1] = -1.0
    changed = _targets(index, changed_bars, changed_signed, changed_carry)
    for book in original:
        pd.testing.assert_frame_equal(original[book], changed[book])
