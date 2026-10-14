"""供需状态：符号、因果 z 分数和事先写定的判定。不读行情。"""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "commodity"))
state = importlib.import_module("morning_clock_state")
CONFIG = json.loads((Path(__file__).resolve().parents[3] / "config" / "commodity" / "clock_state.json").read_text(encoding="utf-8"))


def test_config_locks_three_states_and_the_reversal_payoff_bar():
    assert CONFIG["declared_at"] == "2026-10-14"
    assert "不读 2022" in CONFIG["sample"]
    assert "back" in CONFIG["sign_convention"]
    assert [item["id"] for item in CONFIG["variables"]] == ["carry_align", "stock_align", "oi_state"]
    assert [item["expected_interaction"] for item in CONFIG["variables"]] == ["负", "负", "负"]
    decision = CONFIG["decision"]
    assert decision["interaction_abs_t"] == 2.5
    assert decision["min_years_same_sign"] == 5
    assert decision["min_sectors_same_sign"] == 4
    assert decision["min_reversal_bp"] == 4.0
    assert decision["reversal_bin"] == "低"


def test_zscore_at_earlier_dates_ignores_a_later_observation():
    series = pd.Series(np.arange(80, dtype=float))
    changed = series.copy()
    changed.iloc[-1] = 10_000
    original = state.zscore_known(series, window=20, min_periods=10)
    bumped = state.zscore_known(changed, window=20, min_periods=10)
    pd.testing.assert_series_equal(original.iloc[:-1], bumped.iloc[:-1])


def test_curve_alignment_is_positive_when_backwardation_agrees_with_the_rally():
    carry = pd.Series([2.0, -2.0, -2.0, 2.0, 2.0])
    out = state.curve_alignment(carry, state.move_sign(pd.Series([0.01, -0.02, 0.03, -0.04, 0.0])))
    assert out.iloc[0] == pytest.approx(2)
    assert out.iloc[1] == pytest.approx(2)
    assert out.iloc[2] == pytest.approx(-2)
    assert out.iloc[3] == pytest.approx(-2)
    assert np.isnan(out.iloc[4])
    assert state.move_sign(pd.Series([0.0, np.nan])).isna().all()


def test_stock_alignment_treats_destocking_plus_a_rally_as_continuation():
    move = state.move_sign(pd.Series([0.01, -0.02, -0.03]))
    stock = pd.Series([-1.5, 1.5, -1.5])
    out = state.stock_alignment(stock, move)
    assert out.iloc[0] == pytest.approx(1.5)
    assert out.iloc[1] == pytest.approx(1.5)
    assert out.iloc[2] == pytest.approx(-1.5)


def test_warrant_change_is_blank_when_either_stock_level_is_not_positive():
    level = pd.Series([100.0, 100.0, 110.0, 0.0, 120.0])
    change = state.warrant_change(level, periods=2)
    assert np.isnan(change.iloc[0]) and np.isnan(change.iloc[1])
    assert change.iloc[2] == pytest.approx(0.1)
    assert np.isnan(change.iloc[3])
    assert np.isnan(state.warrant_change(pd.Series([0.0, 10.0, 20.0]), periods=2).iloc[2])


def test_oi_impulse_scales_the_change_and_drops_a_roll_or_a_nonpositive_baseline():
    out = state.oi_impulse([10, 10, 10], [100, 100, 0], [True, False, True])
    assert out[0] == pytest.approx(0.1)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_judge_requires_the_reversal_bin_to_clear_four_basis_points():
    decision = CONFIG["decision"]
    passed = state.judge_state(-3.0, 5, 4, 4.0, -1, decision)
    assert passed["通过"]
    assert not state.judge_state(-3.0, 5, 4, 3.9, -1, decision)["通过"]
    assert not state.judge_state(3.0, 5, 4, 6.0, -1, decision)["通过"]
    assert not state.judge_state(-3.0, 4, 4, 6.0, -1, decision)["通过"]
    assert not state.judge_state(-3.0, 5, 3, 6.0, -1, decision)["通过"]
