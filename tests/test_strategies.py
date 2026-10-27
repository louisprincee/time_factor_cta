"""Strategy signal, configuration, and holdout invariants."""

# --- test_market_clock_slow_short_20_60_120.py ---

import json

import numpy as np
import pandas as pd
import pytest

from strategies import market_clock_slow_short_20_60_120 as T


def _window_bars(index, symbols, drift=-0.001, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.exp(np.cumsum(drift + 0.01 * rng.normal(size=(len(index), len(symbols))), axis=0)),
                         index=index, columns=symbols)
    return {"close": close, "closew": close - 50.0}          # 加法复权：差分不变，比值会变


def _window_signed(index, symbols, seed=2):
    rng = np.random.default_rng(seed)
    return {name: pd.DataFrame(rng.normal(size=(len(index), len(symbols))), index=index, columns=symbols)
            for name in T.FACTORS}


def test_symbol_trend_uses_only_past_closes_and_needs_the_longest_window():
    index = pd.bdate_range("2015-01-01", periods=300)
    bars = _window_bars(index, ["A", "B"])
    trend = T.symbol_trend(bars, ["A", "B"])
    assert trend.iloc[:120].isna().all().all()
    assert set(np.unique(trend.dropna().to_numpy())) <= {-1.0, -1 / 3, 1 / 3, 1.0}
    later = {k: v.copy() for k, v in bars.items()}
    for frame in later.values():
        frame.iloc[-1] *= 3.0
    pd.testing.assert_frame_equal(trend.iloc[:-1], T.symbol_trend(later, ["A", "B"]).iloc[:-1])


def test_market_trend_is_negative_in_a_steady_decline_and_missing_before_120_days():
    daily = pd.Series(-0.001, index=pd.bdate_range("2015-01-01", periods=300))
    trend = T.market_trend(daily)
    assert trend.iloc[:119].isna().all() and trend.iloc[-1] == -1.0


def test_short_leg_needs_bear_trend_and_negative_carry_and_long_leg_is_unchanged():
    index = pd.RangeIndex(4)
    score = pd.Series([0.5, 0.5, 0.5, -0.5], index=index)
    carry = pd.DataFrame({"A": [0.1, 0.1, 0.1, 0.1], "B": [-0.1, -0.1, 0.1, -0.1]}, index=index)
    trend = pd.DataFrame({"A": [-1.0] * 4, "B": [-1 / 3, -1 / 3, -1.0, 1 / 3]}, index=index)
    bear = pd.Series([False, True, True, np.nan], index=index)
    weights = T.decision_weights(score, carry, trend, bear)
    assert weights["A"].tolist() == [1.0, 1.0, 1.0, 0.0]          # 展期为正：多头腿，跟分数走
    assert weights["B"].tolist() == [0.0, -1.0, 1.0, 0.0]         # 熊市、趋势和展期都为负才做空；展期转正回多头腿
    pd.testing.assert_frame_equal(weights.clip(lower=0.0), T.long_weights(score, carry))


def test_targets_use_only_the_previous_close():
    index = pd.bdate_range("2015-01-01", periods=420)
    symbols = [f"S{i:02d}" for i in range(12)]
    universe = {int(year): symbols for year in sorted(set(index.year))}
    signed, bars = _window_signed(index, symbols), _window_bars(index, symbols)
    carry = pd.DataFrame(np.where(np.arange(12) % 2, -0.1, 0.1), index=symbols).T
    carry = pd.DataFrame(np.repeat(carry.to_numpy(), len(index), axis=0), index=index, columns=symbols)
    weights = T.targets(signed, bars, universe, index, symbols, carry=carry)
    assert weights.iloc[0].eq(0.0).all() and weights.lt(0).any().any()
    changed_signed = {name: frame.copy() for name, frame in signed.items()}
    changed_bars = {name: frame.copy() for name, frame in bars.items()}
    for frame in [*changed_signed.values(), *changed_bars.values()]:
        frame.iloc[-1] *= 5.0
    changed_carry = carry.copy()
    changed_carry.iloc[-2:] *= -1.0           # 执行日 t 用的是 t−2 的展期收益
    pd.testing.assert_frame_equal(weights, T.targets(changed_signed, changed_bars, universe, index, symbols,
                                                     carry=changed_carry))
    score = T.market_score(signed, universe, index)
    long_only = T.long_weights(score, carry.shift(1)).shift(1).fillna(0.0)
    long_only = long_only.where(T.pool_mask(universe, index, symbols), 0.0)
    pd.testing.assert_frame_equal(weights.clip(lower=0.0), long_only)


def test_config_must_match_the_short_leg_and_is_frozen(tmp_path):
    config = T.BOOK.frozen()
    assert config["frozen_at"] == "2026-10-23" and config["chosen"] == config["parameters"]
    assert "不读 2026" in config["sample"] and "2022–2024" in config["sample"]
    for key, value in (("short_trend_windows", [10, 20, 30]), ("bear_trend_windows", [10, 20, 30]),
                       ("bear_rule", "market_trend <= 0"),
                       ("short_weight", -0.5), ("gross_exposure_limit", 1.5)):
        broken = {**config, "parameters": {**config["parameters"], key: value}}
        broken["chosen"] = broken["parameters"]
        with pytest.raises(ValueError):
            T.BOOK.check(json.loads(json.dumps(broken)))


def test_frozen_trend_windows_are_shared_by_symbol_and_market_rules():
    config = T.BOOK.load_config()["parameters"]
    assert T.TREND_WINDOWS == (20, 60, 120)
    assert config["short_trend_windows"] == list(T.TREND_WINDOWS)
    assert config["bear_trend_windows"] == list(T.TREND_WINDOWS)


# --- test_market_clock_slow_short_data.py ---

import json

import numpy as np
import pandas as pd
import pytest

from strategies import market_clock_slow_short_20_60_120 as S
from strategies.catalog import BOOKS, FROZEN_BOOKS, book_named
from tfcta import config as C


def _signed(index, symbols, seed=2):
    rng = np.random.default_rng(seed)
    return {
        name: pd.DataFrame(rng.normal(size=(len(index), len(symbols))), index=index, columns=symbols)
        for name in S.FACTORS
    }


def _bars(index, symbols):
    close = pd.DataFrame(100.0, index=index, columns=symbols)
    return {"close": close, "closew": close - 50.0}


def test_catalog_freezes_only_the_three_window_book():
    from strategies import commodity_close_location_reversal as R

    assert FROZEN_BOOKS == (S.BOOK,) and BOOKS == (S.BOOK, R.BOOK)
    assert (S.BOOK.ID, S.BOOK.NAME, S.BOOK.GROSS, S.TREND_WINDOWS) == (
        "market_clock_slow_short_20_60_120", "按池宽1倍加三窗口趋势空头", 1.0, (20, 60, 120),
    )
    parameters = S.BOOK.load_config()["parameters"]
    assert parameters["short_trend_windows"] == list(S.TREND_WINDOWS)
    assert parameters["bear_trend_windows"] == list(S.TREND_WINDOWS)
    assert parameters["gross_split"] == "pool" and parameters["rebalance_on_count_change"] is False
    assert parameters["roll_yield_lag_days"] == 1 and parameters["drawdown_breaker"] == 0.10
    assert book_named(S.BOOK.ID) is S.BOOK
    assert book_named(S.BOOK.NAME) is S.BOOK
    for removed in ("market_clock_pool", "market_clock_slow_short", "market_clock_slow_short_fast3"):
        with pytest.raises(ValueError):
            book_named(removed)


def test_current_trend_helpers_warm_up_on_the_120_day_window():
    index = pd.bdate_range("2010-01-01", periods=130)
    daily = pd.Series(0.01, index=index)
    close = pd.DataFrame({"A": np.arange(100.0, 230.0)}, index=index)
    bars = {"close": close, "closew": close}

    assert S.market_trend(daily).first_valid_index() == index[119]
    assert S.symbol_trend(bars, ["A"])["A"].first_valid_index() == index[120]


def _explore(config):
    out = {key: value for key, value in config.items() if key not in ("frozen_at", "chosen")}
    return {**out, "status": "exploratory_predeclared", "sample": "不读取2022及2023年以后，不读 2026。"}


def test_config_states_and_frozen_spec_validation(tmp_path):
    book = S.BOOK
    frozen = json.loads(book.CONFIG_PATH.read_text(encoding="utf-8"))
    explore = _explore(frozen)
    cases = {
        "explore_with_frozen_at": {**explore, "frozen_at": "2026-10-21"},
        "frozen_without_chosen": {key: value for key, value in frozen.items() if key != "chosen"},
        "frozen_chosen_differs": {**frozen, "chosen": {**frozen["parameters"], "gross_exposure_limit": 2.0}},
        "split_flag_wrong": {
            **frozen,
            "parameters": {**frozen["parameters"], "gross_split": "selected"},
            "chosen": {**frozen["parameters"], "gross_split": "selected"},
        },
        "reads_2026": {**frozen, "sample": "参数只在研究期上选。"},
        "unknown_status": {**frozen, "status": "draft"},
    }
    for name, broken in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError):
            book.check(json.loads(path.read_text(encoding="utf-8")))
    assert book.check(explore)["status"] == "exploratory_predeclared"
    assert book.check(frozen)["status"] == "frozen"


def test_unfrozen_book_refuses_later_windows(monkeypatch, tmp_path):
    book = S.BOOK
    path = tmp_path / "explore.json"
    path.write_text(json.dumps(_explore(book.load_config()), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(book, "CONFIG_PATH", path)
    with pytest.raises(C.HoldoutViolation, match="还没冻结"):
        book.frozen()


def test_both_score_and_lagged_carry_must_be_positive_for_long_leg():
    score = pd.Series([0.4, 0.4, -0.2], index=pd.RangeIndex(3))
    carry = pd.DataFrame({"A": [0.2, -0.1, 0.3]})
    assert S.long_weights(score, carry)["A"].tolist() == [1.0, 0.0, 0.0]


def test_names_outside_the_year_pool_get_zero_weight():
    index = pd.bdate_range("2015-01-01", periods=400)
    symbols = [f"S{i:02d}" for i in range(12)]
    universe = {2015: symbols, 2016: symbols[:11]}
    carry = pd.DataFrame(0.1, index=index, columns=symbols)
    weights = S.targets(_signed(index, symbols), _bars(index, symbols), universe, index, symbols, carry=carry)
    assert weights.loc[index.year == 2016, "S11"].eq(0.0).all()


def test_market_score_can_use_a_fixed_small_instrument_pool():
    index = pd.bdate_range("2015-01-01", periods=400)
    symbols = ["IF", "IH", "IC"]
    signed = _signed(index, symbols)
    universe = {int(year): symbols for year in sorted(set(index.year))}
    assert S.market_score(signed, universe, index).isna().all()
    assert S.market_score(signed, universe, index, min_symbols=3).notna().any()
    with pytest.raises(ValueError, match="正整数"):
        S.market_score(signed, universe, index, min_symbols=0)


def test_carry_coverage_names_the_missing_symbol_year():
    index = pd.bdate_range("2021-01-04", periods=40)
    carry = pd.DataFrame(0.1, index=index, columns=["A", "B"])
    pool = pd.DataFrame(True, index=index, columns=["A", "B"])
    S.require_carry_coverage(carry, pool)
    carry.iloc[:5, 1] = np.nan
    with pytest.raises(ValueError, match="2021 B"):
        S.require_carry_coverage(carry, pool)
    pool["B"] = False
    S.require_carry_coverage(carry, pool)


def test_roll_yield_refuses_2026_and_needs_final_evaluation_for_2023():
    with pytest.raises(C.HoldoutViolation, match="2026"):
        S.roll_yield(pd.DatetimeIndex(["2026-01-05"]), ["RB"])
    with pytest.raises(C.HoldoutViolation):
        S.roll_yield(pd.DatetimeIndex(["2023-01-03"]), ["RB"])


def test_roll_yield_merges_partitions_and_drops_rows_after_the_window(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "ROLL_YIELD_ROOT", tmp_path)

    def write(folder, dates, values):
        path = tmp_path / folder / "roll_yield"
        path.mkdir(parents=True, exist_ok=True)
        index = pd.MultiIndex.from_product([["RB"], pd.to_datetime(dates)], names=["underlying_symbol", "date"])
        pd.DataFrame({"annualized_yield": values}, index=index).to_pickle(path / "RB_main_sub.pkl")

    write(".", ["2021-12-30", "2021-12-31"], [0.1, 0.2])
    write("validation_2022", ["2022-01-04", "2022-01-05", "2022-01-06"], [0.3, -0.4, 9.9])
    index = pd.DatetimeIndex(["2021-12-31", "2022-01-04", "2022-01-05"])
    carry = S.roll_yield(index, ["RB", "CU"])
    assert carry["RB"].tolist() == [0.2, 0.3, -0.4]
    assert carry["CU"].isna().all()


# --- test_commodity_close_location_reversal.py ---

import numpy as np
import pandas as pd
import pytest

from strategies import commodity_close_location_reversal as R
from tfcta import config as C


def test_close_location_long_short_override_and_no_lookahead():
    index = pd.bdate_range("2011-01-03", periods=200)
    symbols = [f"S{i}" for i in range(12)]
    rng = np.random.default_rng(0)
    closew = pd.DataFrame(rng.normal(0, 1, (200, 12)).cumsum(axis=0), index=index, columns=symbols)
    highw, loww = closew + rng.uniform(0.1, 1, (200, 12)), closew - rng.uniform(0.1, 1, (200, 12))
    bars = {"closew": closew, "highw": highw, "loww": loww, "close": closew + 1000.0}
    universe = {2011: symbols}
    carry = pd.DataFrame(1.0, index=index, columns=symbols)
    carry["S0"] = -1.0
    target = R.targets(bars, universe, index, symbols, carry=carry)
    clv = R.market_close_location(bars, universe, index, symbols)
    longs = clv.lt(0.5).shift(1, fill_value=False)
    assert (target.loc[longs, "S1"] == 1).all() and (target.loc[~longs, "S1"] == 0).all()
    assert set(np.unique(target.to_numpy())) <= {-1.0, 0.0, 1.0}
    later = {k: v.copy() for k, v in bars.items()}
    day = index[150]
    for k in later:
        later[k].loc[index > day] += 50.0 * rng.normal(size=(int((index > day).sum()), 12))
    moved = R.targets(later, universe, index, symbols, carry=carry)
    pd.testing.assert_frame_equal(target.loc[:day + pd.offsets.BDay(1)], moved.loc[:day + pd.offsets.BDay(1)])
    trend = pd.DataFrame(-1.0, index=index, columns=symbols)
    weights = R.decision_weights(pd.Series(0.2, index=index), carry, trend, pd.Series(True, index=index))
    assert (weights["S0"] == -1).all() and (weights["S1"] == 1).all()


def test_exploratory_book_cannot_open_later_windows():
    with pytest.raises(C.HoldoutViolation):
        R.BOOK.frozen()
