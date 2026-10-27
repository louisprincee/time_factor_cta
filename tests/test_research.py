"""Research window, protocol, and candidate-signal contracts."""

# --- test_index_market_clock.py ---

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


# --- test_index_close_location_candidate.py ---

import numpy as np
import pandas as pd
import pytest

from research.index_close_location_candidate import (
    CANDIDATE_ID,
    BENCHMARK_ID,
    SYMBOLS,
    build_targets,
    load_protocol as load_candidate_protocol,
)
from strategies.catalog import BOOKS as COMMODITY_BOOKS
from strategies.index_catalog import BOOKS as INDEX_BOOKS, book_named


def test_candidate_uses_prior_close_location_and_registered_binary_rule():
    index = pd.date_range("2016-01-13", periods=4, freq="B")
    low = pd.DataFrame(0.0, index=index, columns=SYMBOLS)
    high = pd.DataFrame(10.0, index=index, columns=SYMBOLS)
    close = pd.DataFrame(
        [[2.0] * 3, [8.0] * 3, [1.0] * 3, [9.0] * 3],
        index=index,
        columns=SYMBOLS,
    )
    bars = {"loww": low, "highw": high, "closew": close}

    targets, market_clv = build_targets(bars, index)

    protocol = load_candidate_protocol()
    assert protocol["status"] == "exploratory_predeclared"
    assert protocol["parameters"]["score_start"] == "2016-01-13"
    assert book_named(CANDIDATE_ID) is INDEX_BOOKS[0]
    assert all(book.ID != CANDIDATE_ID for book in COMMODITY_BOOKS)
    assert market_clv.tolist() == pytest.approx([0.2, 0.8, 0.1, 0.9])
    assert targets[CANDIDATE_ID].iloc[:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert targets[BENCHMARK_ID].eq(1.0).all().all()

    changed = close.copy()
    changed.iloc[-1] = 0.0
    changed_targets, _ = build_targets({**bars, "closew": changed}, index)
    pd.testing.assert_frame_equal(
        targets[CANDIDATE_ID].iloc[:-1],
        changed_targets[CANDIDATE_ID].iloc[:-1],
    )


def test_short_leg_overrides_the_long_and_uses_only_the_previous_close():
    from research import index_close_location_candidate as K

    index = pd.bdate_range("2018-01-01", periods=200)
    close = pd.DataFrame({s: 100 * np.exp(-0.002 * np.arange(200)) for s in K.SYMBOLS}, index=index)
    bars = {"close": close, "closew": close - 10}
    long_target = pd.DataFrame(1.0, index=index, columns=list(K.SYMBOLS))
    carry = pd.DataFrame({"IF": -0.1, "IH": 0.1, "IC": -0.1}, index=index)
    out = K.short_leg_targets(long_target, bars, carry)
    assert out.iloc[-1].tolist() == [-1.0, 1.0, -1.0]             # 稳定下跌：熊市且趋势为负，展期为负的才做空
    assert out.iloc[:120].eq(1.0).all().all()                     # 120 日窗口不足时没有熊市判断
    moved = {k: v.copy() for k, v in bars.items()}
    for v in moved.values():
        v.iloc[-1] *= 5
    pd.testing.assert_frame_equal(out, K.short_leg_targets(long_target, moved, carry))


# --- test_three_strategy_clv_comparison.py ---

import numpy as np
import pandas as pd

from research.commodity_close_location_reversal import (
    attribute_positions,
    index_close_location_targets,
    load_protocol as load_comparison_protocol,
)
from research.index_market_clock import SYMBOLS


def test_comparison_protocol_declares_three_books_and_locked_windows():
    protocol = load_comparison_protocol()

    assert protocol["books"] == [
        "market_clock_slow_short_20_60_120",
        "commodity_close_location_reversal_strict_swap",
        "index_close_location_reversal_strict_swap",
    ]
    assert "Do not read 2022" in protocol["sample"]
    assert "2015-04-16" in protocol["sample"]


def test_index_close_location_target_uses_prior_close_and_preserves_carry_gate():
    index = pd.bdate_range("2015-04-16", periods=140)
    close = pd.DataFrame(
        np.repeat((100.0 + np.arange(len(index), dtype=float))[:, None], len(SYMBOLS), axis=1),
        index=index,
        columns=SYMBOLS,
    )
    high = close + 2.0
    low = close - 2.0
    high.loc[index[20]] = close.loc[index[20]] + 3.0
    low.loc[index[20]] = close.loc[index[20]] - 1.0
    bars = {"close": close, "closew": close, "highw": high, "loww": low}
    carry = pd.DataFrame(0.1, index=index, columns=SYMBOLS)

    target, market_clv = index_close_location_targets(index, bars, carry)

    assert market_clv.loc[index[20]] == 0.25
    assert target.loc[index[20]].eq(0.0).all()
    assert target.loc[index[21]].eq(1.0).all()
    changed = {name: frame.copy() for name, frame in bars.items()}
    for frame in changed.values():
        frame.loc[index[22]:] += 100.0
    changed_target, _ = index_close_location_targets(index, changed, carry)
    pd.testing.assert_frame_equal(target.loc[:index[21]], changed_target.loc[:index[21]])

    negative_carry = carry.copy()
    negative_carry.loc[index[19], "IF"] = -0.1
    filtered, _ = index_close_location_targets(index, bars, negative_carry)
    assert filtered.at[index[21], "IF"] == 0.0
    assert filtered.at[index[21], "IH"] == 1.0


def test_component_attribution_reconciles_to_ledger_net_return():
    day = pd.Timestamp("2018-01-02")
    daily = pd.DataFrame(
        {
            "期初权益": [1000.0],
            "毛盈亏": [30.0],
            "交易费用": [5.0],
            "权益清零调整": [0.0],
            "净收益": [0.025],
        },
        index=pd.DatetimeIndex([day]),
    )
    positions = pd.DataFrame(
        {
            "交易日": [day, day],
            "品种": ["IF", "IH"],
            "手数": [2, -1],
            "合约乘数": [10.0, 5.0],
        }
    )
    delta = pd.DataFrame({"IF": [1.0], "IH": [-2.0]}, index=daily.index)

    side, instruments = attribute_positions(daily, positions, delta)

    assert side.loc[day, "多头腿"] == 0.02
    assert side.loc[day, "空头腿"] == 0.01
    assert side.loc[day, "交易成本及清零调整"] == -0.005
    assert np.isclose(side.loc[day].sum(), daily.loc[day, "净收益"])
    assert np.isclose(instruments.loc[day].sum(), daily.loc[day, "净收益"])
