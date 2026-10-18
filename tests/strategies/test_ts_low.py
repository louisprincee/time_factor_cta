import json

import pandas as pd
import pytest

from strategies.ts_low import (
    CONFIG_PATH, build_entry_schedule, build_positions, chosen_parameters,
    load_config, parameters, upper_tail,
)


def _factor(values):
    days = pd.bdate_range("2021-01-04", periods=len(values))
    return pd.DataFrame({"RB": values}, index=days)


def test_only_the_upper_tail_is_long_and_a_tighter_cutoff_fires_less():
    series = pd.Series([0, 0, 0, 1, 0.4, -1], index=pd.bdate_range("2021-01-04", periods=6))
    wide = upper_tail(series, 4, 0.7)
    tight = upper_tail(series, 4, 0.9)
    assert wide.iloc[4] == 1 and tight.iloc[4] == 0
    assert upper_tail(series, 4, 0.8).iloc[5] == 0




def test_position_starts_the_next_open_and_blocks_until_exit():
    factor = _factor([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    signal_days = factor.index
    returns = pd.DataFrame({"RB": 0.0}, index=signal_days)
    held = build_positions(factor, returns, {2021: ["RB"]}, 3, 3, 0.8)
    on = list(held.index[held["RB"] == 1])
    assert on[0] == signal_days[4]
    assert on == list(signal_days[4:7]) + list(signal_days[7:10])
    assert held.loc[signal_days[3], "RB"] == 0


def test_entry_schedule_preserves_back_to_back_holding_cycles():
    factor = _factor([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    entries = build_entry_schedule(
        factor, pd.DataFrame({"RB": 0.0}, index=factor.index),
        {2021: ["RB"]}, horizon=3, lookback=3, quantile=0.8,
    )
    assert list(entries.index[entries["RB"]]) == [factor.index[4], factor.index[7]]


def test_entry_schedule_applies_filter_on_signal_close():
    factor = _factor([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    eligible = pd.DataFrame(True, index=factor.index, columns=["RB"])
    eligible.loc[factor.index[3:6], "RB"] = False
    entries = build_entry_schedule(
        factor, pd.DataFrame({"RB": 0.0}, index=factor.index),
        {2021: ["RB"]}, horizon=3, lookback=3, quantile=0.8,
        entry_filter=eligible,
    )
    assert list(entries.index[entries["RB"]]) == [factor.index[7]]


def test_entry_schedule_rechecks_pool_membership_on_cross_year_entry():
    days = pd.bdate_range("2024-12-24", "2025-01-10").difference(
        pd.DatetimeIndex(["2025-01-01"])
    )
    factor = pd.DataFrame({"RB": 0.0}, index=days)
    signal_day = pd.Timestamp("2024-12-31")
    factor.loc[signal_day, "RB"] = 10.0
    returns = pd.DataFrame({"RB": 0.0}, index=days)
    lookback = int(days.get_loc(signal_day))

    excluded = build_entry_schedule(
        factor, returns, {2024: ["RB"], 2025: ["CU"]},
        horizon=2, lookback=lookback, quantile=0.8,
        start=pd.Timestamp("2024-12-24"), end=pd.Timestamp("2025-01-10"),
    )
    included = build_entry_schedule(
        factor, returns, {2024: ["RB"], 2025: ["RB"]},
        horizon=2, lookback=lookback, quantile=0.8,
        start=pd.Timestamp("2024-12-24"), end=pd.Timestamp("2025-01-10"),
    )

    assert not excluded["RB"].any()
    assert list(included.index[included["RB"]]) == [pd.Timestamp("2025-01-02")]


def test_symbol_outside_that_years_universe_stays_flat():
    factor = _factor([0, 0, 0, 1, 1])
    returns = pd.DataFrame({"RB": 0.0}, index=factor.index)
    held = build_positions(factor, returns, {2021: ["CU"]}, 1, 3, 0.8)
    assert held["RB"].sum() == 0


def test_config_keeps_the_parameter_grid_and_oos_closed(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["frozen_at"] == "2026-10-18"
    assert config["chosen"] == {"lookback": 250, "quantile": 0.8, "horizon": 5}
    research_config = json.loads(json.dumps(config))
    research_config.pop("frozen_at")
    research_config.pop("chosen")
    research_config["status"] = "strategy"
    research_config["sample"] = "研究期不读 2023 年以后。"
    path = tmp_path / "research.json"
    path.write_text(json.dumps(research_config), encoding="utf-8")
    config = load_config(path)
    grid = list(parameters(config))
    assert len(grid) == 27
    assert {item["回看交易日"] for item in grid} == {60, 250, 750}
    assert {item["持仓交易日"] for item in grid} == {1, 5, 20}
    assert {item["分位"] for item in grid} == {0.9, 0.85, 0.8}
    assert {item["阈值"] for item in grid} == {"前10%", "前15%", "前20%"}
    changed = json.loads(json.dumps(config))
    changed["parameters"]["quantiles"].append(0.5)
    path = tmp_path / "wider.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
    locked = json.loads(json.dumps(config))
    locked["frozen_at"] = "2026-10-18"
    locked["chosen"] = {"factor": "ts_low_minus_high"}
    locked_path = tmp_path / "locked.json"
    locked_path.write_text(json.dumps(locked), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(locked_path)


def test_frozen_config_selects_one_allowed_oos_parameter_set(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["sample"] = "冻结后才可进行样本外最终测试。"
    config["frozen_at"] = "2026-10-18"
    config["chosen"] = {"lookback": 250, "quantile": 0.8, "horizon": 5}
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    frozen = load_config(path, allow_frozen=True)
    selected = chosen_parameters(frozen)
    assert selected["回看交易日"] == 250
    assert selected["分位"] == 0.8
    assert selected["持仓交易日"] == 5

    frozen["chosen"]["horizon"] = 10
    with pytest.raises(ValueError, match="horizon"):
        chosen_parameters(frozen)
