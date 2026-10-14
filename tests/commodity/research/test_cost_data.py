"""成本数据：tick 估计、郑商所平今费补齐、池内缺成本时停下。不读真实行情。"""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.research import estimate_ticks, load_ticks, tick_path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "commodity"))
features = importlib.import_module("morning_features")


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
    out = features.fill_close_today(table).set_index(["symbol", "trading_date"])
    assert out.loc[("FG", days[0]), "close_commission_today"] == 24.0  # 后来的平今费更贵
    assert out.loc[("CF", days[0]), "close_commission_today"] == 6.0   # 后来平今免费，仍按平仓费
    assert out.close_today_filled.sum() == 4


def test_close_today_gap_is_not_filled_across_a_fee_type_change():
    table = pd.DataFrame({"symbol": "X", "trading_date": pd.to_datetime(["2017-01-03", "2017-07-03"]),
                          "commission_type": ["by_money", "by_volume"], "close_commission": [1e-4, 3.0],
                          "close_commission_today": [np.nan, 6.0]})
    out = features.fill_close_today(table)
    assert np.isnan(out.close_commission_today.iloc[0])


def test_check_costs_stops_when_pool_symbols_cannot_be_costed():
    days = pd.bdate_range("2022-01-03", periods=20)
    table = pd.DataFrame({"symbol": "SA", "trading_date": days, "dev15": .003, "est_cost": np.nan})
    with pytest.raises(ValueError, match="缺成本"):
        features.check_costs(table, {2022: ["SA"]})
    features.check_costs(table, {2022: ["RB"]})  # 不在池里就不管
    table["est_cost"] = .0003
    features.check_costs(table, {2022: ["SA"]})


def test_close_today_fill_only_covers_the_known_czce_gap():
    table = pd.DataFrame({"symbol": "X", "trading_date": pd.to_datetime(["2019-03-01", "2019-03-04"]),
                          "commission_type": "by_volume", "close_commission": [3.0, 3.0],
                          "close_commission_today": [np.nan, 6.0]})
    out = features.fill_close_today(table)
    assert np.isnan(out.close_commission_today.iloc[0])  # 2017-06-23 以后缺了不补，让 check_costs 报出来
