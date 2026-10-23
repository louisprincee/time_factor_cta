"""慢趋势策略的市场分数、展期覆盖和数据分区保护。合成数据。"""
import json

import numpy as np
import pandas as pd
import pytest

from strategies import market_clock_slow_short as S
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


def test_catalog_contains_only_the_frozen_slow_short_book():
    assert BOOKS == FROZEN_BOOKS == (S.BOOK,)
    assert (S.BOOK.ID, S.BOOK.NAME, S.BOOK.GROSS, S.TREND_WINDOWS) == (
        "market_clock_slow_short", "按池宽1倍加慢趋势空头", 1.0, (60, 120, 250),
    )
    parameters = S.BOOK.load_config()["parameters"]
    assert parameters["short_trend_windows"] == list(S.TREND_WINDOWS)
    assert parameters["bear_trend_windows"] == list(S.TREND_WINDOWS)
    assert parameters["gross_split"] == "pool" and parameters["rebalance_on_count_change"] is False
    assert parameters["roll_yield_lag_days"] == 1 and parameters["drawdown_breaker"] == 0.10
    assert book_named(S.BOOK.ID) is S.BOOK
    assert book_named(S.BOOK.NAME) is S.BOOK
    for removed in ("market_clock_pool", "market_clock_slow_short_fast3", "按池宽1倍加三窗口趋势空头"):
        with pytest.raises(ValueError):
            book_named(removed)


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
