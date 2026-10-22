"""当天分数大于 0 且滞后一天的展期收益大于 0 才做多。合成数据。"""
import json

import numpy as np
import pandas as pd
import pytest

from strategies import market_clock_carry as S
from strategies.catalog import BOOKS, book_named
from tfcta import config as C


def _signed(index, symbols, seed=2):
    rng = np.random.default_rng(seed)
    return {
        name: pd.DataFrame(rng.normal(size=(len(index), len(symbols))), index=index, columns=symbols)
        for name in S.FACTORS
    }


def test_catalog_holds_the_two_frozen_books():
    assert [(book.ID, book.NAME, book.GROSS) for book in BOOKS] == [
        ("market_clock_pool", "按池宽1倍", 1.0),
        ("market_clock_slow_short", "按池宽1倍加慢趋势空头", 1.0),
    ]
    for book in BOOKS:
        parameters = book.load_config()["parameters"]
        assert parameters["gross_split"] == "pool" and parameters["rebalance_on_count_change"] is False
        assert parameters["roll_yield_lag_days"] == 1 and parameters["drawdown_breaker"] == 0.10
    assert book_named("按池宽1倍") is S.BOOK
    assert book_named("market_clock_slow_short").NAME == "按池宽1倍加慢趋势空头"
    for gone in ("没有这条", "入选平分1倍", "market_clock_pool_15", "market_clock_carry_five_factor_regime"):
        with pytest.raises(ValueError):
            book_named(gone)


def _explore(config):
    """冻结规格退回探索状态的样子。"""
    out = {k: v for k, v in config.items() if k not in ("frozen_at", "chosen")}
    return {**out, "status": "exploratory_predeclared", "sample": "不读取2022及2023年以后，不读 2026。"}


def test_config_states_explore_or_frozen_and_nothing_else(tmp_path):
    book = S.BOOK
    frozen = json.loads(book.CONFIG_PATH.read_text(encoding="utf-8"))
    explore = _explore(frozen)
    cases = {
        "explore_with_frozen_at": {**explore, "frozen_at": "2026-10-21"},
        "frozen_without_chosen": {k: v for k, v in frozen.items() if k != "chosen"},
        "frozen_chosen_differs": {**frozen, "chosen": {**frozen["parameters"], "gross_exposure_limit": 2.0}},
        "split_flag_wrong": {**frozen, "parameters": {**frozen["parameters"], "gross_split": "selected"},
                             "chosen": {**frozen["parameters"], "gross_split": "selected"}},
        "reads_2026": {**frozen, "sample": "参数只在研究期上选。"},
        "unknown_status": {**frozen, "status": "draft"},
    }
    for name, broken in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError):
            S.check_config(json.loads(path.read_text(encoding="utf-8")), book.ID)
    for name, good in (("explore", explore), ("frozen", frozen)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(good, ensure_ascii=False), encoding="utf-8")
        assert S.check_config(json.loads(path.read_text(encoding="utf-8")), book.ID)["status"] == good["status"]


def test_both_books_are_frozen_and_an_unfrozen_copy_refuses_later_windows(monkeypatch, tmp_path):
    frozen_at = {"market_clock_pool": "2026-10-21", "market_clock_slow_short": "2026-10-22"}
    for book in BOOKS:
        config = book.frozen()
        assert config["frozen_at"] == frozen_at[book.ID] and config["chosen"] == config["parameters"]
    for book in BOOKS:
        path = tmp_path / f"{book.ID}.json"
        path.write_text(json.dumps(_explore(book.load_config()), ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(book, "CONFIG_PATH", path)
        with pytest.raises(C.HoldoutViolation, match="还没冻结"):
            book.frozen()


def test_both_the_score_and_the_lagged_carry_must_be_positive():
    score = pd.Series([0.4, 0.4, -0.2], index=pd.RangeIndex(3))
    carry = pd.DataFrame({"A": [0.2, -0.1, 0.3]})
    assert S.decision_weights(score, carry)["A"].tolist() == [1.0, 0.0, 0.0]


def test_targets_use_only_the_previous_close_and_the_roll_yield_before_it():
    index = pd.bdate_range("2015-01-01", periods=400)
    symbols = [f"S{i:02d}" for i in range(12)]
    signed = _signed(index, symbols)
    universe = {int(year): symbols for year in sorted(set(index.year))}
    carry = pd.DataFrame(0.1, index=index, columns=symbols)
    weights = S.targets(signed, universe, index, symbols, carry=carry)
    changed = {name: frame.copy() for name, frame in signed.items()}
    for frame in changed.values():
        frame.iloc[-1] = 10.0
    carry_changed = carry.copy()
    carry_changed.iloc[-2:] = -1.0           # 执行日 t 用的是 t−2 的展期收益
    pd.testing.assert_frame_equal(weights, S.targets(changed, universe, index, symbols, carry=carry_changed))
    assert weights.iloc[0].eq(0.0).all()
    score = S.market_score(signed, universe, index)
    day = index[-1]
    assert weights.loc[day].eq(1.0 if score.shift(1).loc[day] > 0 else 0.0).all()


def test_names_outside_the_year_pool_get_zero_weight():
    index = pd.bdate_range("2015-01-01", periods=400)
    symbols = [f"S{i:02d}" for i in range(12)]
    universe = {2015: symbols, 2016: symbols[:11]}
    weights = S.targets(_signed(index, symbols), universe, index, symbols,
                        carry=pd.DataFrame(0.1, index=index, columns=symbols))
    assert weights.loc[index.year == 2016, "S11"].eq(0.0).all()


def test_carry_coverage_names_the_missing_symbol_year():
    index = pd.bdate_range("2021-01-04", periods=40)
    carry = pd.DataFrame(0.1, index=index, columns=["A", "B"])
    pool = pd.DataFrame(True, index=index, columns=["A", "B"])
    S.require_carry_coverage(carry, pool)
    carry.iloc[:5, 1] = np.nan                # B 缺 12.5%
    with pytest.raises(ValueError, match="2021 B"):
        S.require_carry_coverage(carry, pool)
    pool["B"] = False                         # 不在池里的缺失不管
    S.require_carry_coverage(carry, pool)


def test_roll_yield_refuses_2026_and_needs_the_final_evaluation_for_2023(monkeypatch, tmp_path):
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
