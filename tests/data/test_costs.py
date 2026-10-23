"""成本数据：tick 估计和郑商所平今费补齐。不读真实行情。"""
import numpy as np
import pandas as pd
import pytest
from tfcta import config as C
from tfcta.costs import (
    estimate_ticks, fill_close_today, load_fees, load_ticks, tick_path,
)

def test_estimate_ticks_ignores_rare_smaller_steps_and_thin_years():
    rng = np.random.default_rng(0)
    steps = rng.choice([-2.0, 2.0, -4.0, 4.0, 0.0], size=3000)
    steps[::1000] = 1.0  # 万分之几的错价，不算价位
    price = 5000 + np.cumsum(steps)
    days = np.repeat(pd.bdate_range("2017-01-02", periods=30), 100)
    thin = np.r_[price, 6000.0, 6002.0, 6001.0]
    dates = np.r_[days, np.repeat(np.datetime64("2018-01-02"), 3)]
    out = estimate_ticks(thin, dates).set_index("year")
    assert out.loc[2017, "tick"] == 2.0
    assert np.isnan(out.loc[2018, "tick"])  # 非零变动太少，不估

def test_oos_ticks_have_no_file():
    with pytest.raises(C.HoldoutViolation):
        tick_path("oos")

def test_research_ticks_reject_holdout_years(tmp_path, monkeypatch):
    path = tmp_path / "tick_size.csv"
    pd.DataFrame({"symbol": ["A"], "year": [2023], "tick": [1.0]}).to_csv(path, index=False)
    monkeypatch.setattr(C, "RESEARCH_OUT_DIR", tmp_path)
    with pytest.raises(C.HoldoutViolation):
        load_ticks()

def test_close_today_gap_takes_the_higher_of_close_and_later_close_today():
    days = pd.to_datetime(["2017-06-21", "2017-06-22", "2017-06-23", "2017-06-26"])
    table = pd.DataFrame({"symbol": ["FG"] * 4 + ["CF"] * 4, "trading_date": list(days) * 2,
                          "commission_type": "by_volume",
                          "close_commission": [3.0] * 4 + [6.0] * 4,
                          "close_commission_today": [np.nan, np.nan, 24.0, 24.0, np.nan, np.nan, 0.0, 0.0]})
    out = fill_close_today(table).set_index(["symbol", "trading_date"])
    assert out.loc[("FG", days[0]), "close_commission_today"] == 24.0  # 后来的平今费更贵
    assert out.loc[("CF", days[0]), "close_commission_today"] == 6.0   # 后来平今免费，仍按平仓费
    assert out.close_today_filled.sum() == 4

def test_close_today_gap_is_not_filled_across_a_fee_type_change():
    table = pd.DataFrame({"symbol": "X", "trading_date": pd.to_datetime(["2017-01-03", "2017-07-03"]),
                          "commission_type": ["by_money", "by_volume"], "close_commission": [1e-4, 3.0],
                          "close_commission_today": [np.nan, 6.0]})
    out = fill_close_today(table)
    assert np.isnan(out.close_commission_today.iloc[0])

def test_close_today_fill_only_covers_the_known_czce_gap():
    table = pd.DataFrame({"symbol": "X", "trading_date": pd.to_datetime(["2019-03-01", "2019-03-04"]),
                          "commission_type": "by_volume", "close_commission": [3.0, 3.0],
                          "close_commission_today": [np.nan, 6.0]})
    out = fill_close_today(table)
    assert np.isnan(out.close_commission_today.iloc[0])  # 2017-06-23 以后缺了不补，留给成本表在成交时报错

def test_close_today_fill_does_not_apply_to_other_exchanges():
    table = pd.DataFrame({"symbol": ["FG", "FG", "RB", "RB"],
                          "trading_date": pd.to_datetime([
                              "2017-06-22", "2017-06-23", "2017-06-22", "2017-06-23",
                          ]),
                          "commission_type": "by_volume", "close_commission": 3.0,
                          "close_commission_today": [np.nan, 6.0, np.nan, 6.0]})
    out = fill_close_today(table).set_index(["symbol", "trading_date"])
    assert out.loc[("FG", pd.Timestamp("2017-06-22")), "close_commission_today"] == 6.0
    assert np.isnan(out.loc[("RB", pd.Timestamp("2017-06-22")), "close_commission_today"])

def test_load_fees_applies_known_historical_close_today_repair(tmp_path, monkeypatch):
    table = pd.DataFrame({
        "symbol": ["FG", "FG"],
        "trading_date": pd.to_datetime(["2017-06-22", "2017-06-23"]),
        "commission_type": "by_volume",
        "open_commission": 1.0,
        "close_commission": 3.0,
        "close_commission_today": [np.nan, 6.0],
    })
    table.to_csv(tmp_path / "fee_history.csv", index=False)
    monkeypatch.setattr(C, "RESEARCH_OUT_DIR", tmp_path)

    loaded = load_fees("research")

    assert loaded.close_commission_today.tolist() == [6.0, 6.0]
    assert loaded.close_today_filled.tolist() == [True, False]


def test_contract_multiplier_follows_the_contract_not_the_symbol():
    from tfcta.costs import MULTIPLIER, contract_multiplier, contract_multiplier_overrides

    assert contract_multiplier("RU", "RU1205") == 5.0 and contract_multiplier("RU", "RU1207") == 5.0
    assert contract_multiplier("RU", "RU1208") == MULTIPLIER["RU"] == 10
    assert contract_multiplier("FU", "FU1812") == 50.0 and contract_multiplier("FU", "FU1901") == 10
    assert contract_multiplier("PB", "PB1308") == 25.0 and contract_multiplier("PB", "PB1309") == 5
    assert contract_multiplier("CU", "CU1105") == 5 and contract_multiplier("RU", None) == 10
    assert contract_multiplier_overrides("RU", ["RU1205", "RU1209", "RU1205"]) == {"RU1205": 5.0}


def test_by_volume_fee_uses_the_fee_rows_contract_multiplier():
    from tfcta.costs import fee_tables

    dates = pd.bdate_range("2012-04-02", periods=4)
    table = pd.DataFrame({
        "symbol": "RU", "trading_date": dates, "contract": ["RU1205", "RU1205", "RU1209", "RU1209"],
        "commission_type": "by_volume", "open_commission": 3.0, "close_commission": 3.0,
        "close_commission_today": 3.0,
    })
    prices = pd.DataFrame({"RU": 100.0}, index=dates)
    opened, _, rolled = fee_tables(table, prices)
    assert opened["RU"].tolist() == pytest.approx([3.01 / 500, 3.01 / 500, 3.01 / 1000, 3.01 / 1000])
    assert rolled["RU"].iloc[2] == pytest.approx(3.01 / 500)      # 换月当天平的是旧 5 吨合约


def test_shfe_fee_backfill_copies_the_first_rate_back_to_2011_and_nothing_else(tmp_path, monkeypatch):
    table = pd.DataFrame({
        "symbol": ["CU", "CU", "M"], "contract": ["CU1312", "CU1312", "M1405"],
        "trading_date": pd.to_datetime(["2013-10-08", "2013-10-09", "2013-10-08"]),
        "commission_type": ["by_money", "by_money", "by_volume"], "open_commission": [5e-5, 5e-5, 1.5],
        "close_commission": [5e-5, 5e-5, 1.5], "close_commission_today": [0.0, 0.0, 0.75],
    })
    table.to_csv(tmp_path / "fee_history.csv", index=False)
    monkeypatch.setattr(C, "RESEARCH_OUT_DIR", tmp_path)
    loaded = load_fees("research")
    cu = loaded[loaded.symbol == "CU"]
    assert cu.trading_date.min() == pd.Timestamp("2011-01-03")          # 2011-01-01 后第一个工作日
    early = cu[cu.trading_date < pd.Timestamp("2013-10-08")]
    assert early.open_commission.eq(5e-5).all() and early.contract.isna().all()
    assert loaded[loaded.symbol == "M"].trading_date.tolist() == [pd.Timestamp("2013-10-08")]
    bad = table.assign(trading_date=pd.to_datetime(["2013-10-09", "2013-10-10", "2013-10-08"]))
    bad.to_csv(tmp_path / "fee_history.csv", index=False)
    with pytest.raises(ValueError, match="回填假设"):
        load_fees("research")
