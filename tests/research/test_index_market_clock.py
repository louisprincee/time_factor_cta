import numpy as np
import pandas as pd
import pytest

from research.index_market_clock import (
    FACTORS,
    SCORE_START,
    SYMBOLS,
    _trim_to_first_target,
    _targets,
    load_protocol,
    require_index_carry_coverage,
    write_plots,
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
    assert protocol["sample"].startswith("Index research calendar 2015-04-16")
    assert SCORE_START == pd.Timestamp("2015-04-16")


def test_index_run_writes_per_book_and_comparison_performance_plots(tmp_path):
    index = pd.bdate_range("2015-04-16", periods=5)
    returns = pd.DataFrame(
        {
            "market_clock_pool": [0.01, -0.01, 0.02, 0.0, 0.01],
            "market_clock_slow_short_20_60_120": [0.0, -0.005, 0.01, 0.005, 0.0],
        },
        index=index,
    )
    market = pd.Series([0.005, -0.01, 0.01, 0.002, 0.0], index=index)

    write_plots(returns, market, tmp_path)

    assert (tmp_path / "plots" / "market_clock_pool" / "performance.png").is_file()
    assert (tmp_path / "plots" / "market_clock_slow_short_20_60_120" / "performance.png").is_file()
    assert (tmp_path / "plots" / "comparison" / "performance.png").is_file()


def test_backtest_outputs_start_on_first_nonzero_target_date():
    index = pd.bdate_range("2015-04-16", periods=6)
    returns = pd.DataFrame({"market": [0.1] * len(index)}, index=index)
    targets = {
        "long": pd.DataFrame({"IF": [0, 0, 0, 1, 0, 0]}, index=index),
        "current": pd.DataFrame({"IF": [0, 0, 0, 0, -1, 0]}, index=index),
    }

    actual, first_signal = _trim_to_first_target(returns, targets)

    assert first_signal == index[3]
    assert actual.index.equals(index[3:])


def test_backtest_output_start_requires_a_nonzero_target():
    index = pd.bdate_range("2015-04-16", periods=3)
    returns = pd.DataFrame({"market": [0.1] * len(index)}, index=index)
    targets = {"long": pd.DataFrame({"IF": [0, 0, 0]}, index=index)}

    with pytest.raises(ValueError, match="没有任何非零目标"):
        _trim_to_first_target(returns, targets)


def test_index_carry_coverage_fails_above_five_percent():
    index = pd.bdate_range(SCORE_START, "2021-12-31")
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


def test_index_carry_uses_the_largest_oi_later_contract_and_the_maturity_gap():
    from download.index.getIndexRollYield import main_sub_carry

    day = pd.Timestamp("2018-01-02")
    daily = pd.DataFrame({
        "date": day, "contract": ["IF1801", "IF1802", "IF1803", "IF1712"],
        "settlement": [4000.0, 3990.0, 3960.0, 4010.0], "open_interest": [20000, 500, 9000, 30000],
    })
    maturity = pd.Series(pd.to_datetime(["2018-01-19", "2018-02-23", "2018-03-16", "2017-12-15"]),
                         index=["IF1801", "IF1802", "IF1803", "IF1712"])
    out = main_sub_carry(daily, maturity, pd.Series({day: "IF1801"}))
    row = out.iloc[0]
    assert row["to_contract"] == "IF1803"                       # 晚于主力到期的合约里持仓最大，不看更早到期的
    assert row["yield"] == pytest.approx(40 / 3960)
    assert row["annualized_yield"] == pytest.approx(40 / 3960 * 365 / 56)
