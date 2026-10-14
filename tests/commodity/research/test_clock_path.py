"""路径形态变量：公式、5 分钟切法、交互项和事先写定的判定。不读行情。"""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "commodity"))
path = importlib.import_module("morning_clock_path")
CONFIG = json.loads((Path(__file__).resolve().parents[3] / "config" / "commodity" / "clock_path.json").read_text(encoding="utf-8"))


def test_config_locks_the_two_variables_and_the_decision_bar():
    assert CONFIG["declared_at"] == "2026-10-14"
    assert "不读 2022" in CONFIG["sample"]
    assert [item["id"] for item in CONFIG["variables"]] == ["impulse", "efficiency"]
    assert [item["expected_interaction"] for item in CONFIG["variables"]] == ["正", "负"]
    decision = CONFIG["decision"]
    assert decision["interaction_abs_t"] == 2.5
    assert decision["min_years_same_sign"] == 5
    assert decision["min_sectors_same_sign"] == 4


def test_tail_impulse_is_the_share_of_the_range_and_is_blank_when_the_range_is_flat():
    assert path.tail_impulse(-0.02, 0.05) == pytest.approx(0.4)
    assert path.tail_impulse(0.02, 0.05) == pytest.approx(0.4)
    assert np.isnan(path.tail_impulse(0.01, 0.0))
    assert np.isnan(path.tail_impulse(np.nan, 0.05))


def test_path_efficiency_is_one_on_a_straight_line_and_lower_when_the_path_retraces():
    assert path.path_efficiency_from_closes([100, 101, 102, 103]) == pytest.approx(1)
    assert path.path_efficiency_from_closes([100, 110, 90, 104]) == pytest.approx(4 / 44)
    assert np.isnan(path.path_efficiency_from_closes([100, 100, 100]))
    assert np.isnan(path.path_efficiency_from_closes([100]))


def test_five_minute_bins_match_the_case_chart_resample_and_keep_the_0916_bar():
    night = pd.date_range("2020-01-02 21:01", periods=10, freq="min")
    morning = pd.date_range("2020-01-03 09:00", "2020-01-03 09:16", freq="min")
    index = night.append(morning)
    values = np.arange(len(index), dtype=float)
    expected = pd.Series(values, index=index).resample(
        "5min", closed="right", label="right", origin="epoch", offset="1min").last().dropna()
    got = path.five_minute_last(values, index.asi8)
    assert np.allclose(got, expected.to_numpy())
    assert expected.index[-1] == pd.Timestamp("2020-01-03 09:16")


def test_still_extending_follows_the_later_extreme_and_is_blank_without_a_direction():
    out = path.still_extending([0.8, 0.8, -0.8, -0.8, 0.0], [-0.01, 0.01, 0.01, -0.01, -0.02])
    assert list(out[:4]) == [1.0, 0.0, 1.0, 0.0]
    assert np.isnan(out[4])
    push = path.aligned_dev([0.8, -0.8], [-0.01, 0.01])
    assert push[0] == pytest.approx(0.01) and push[1] == pytest.approx(0.01)


def test_tercile_is_balanced_inside_one_symbol_year_and_blank_when_too_short():
    labels = path.assign_tercile(pd.Series(np.arange(9)))
    assert list(labels.value_counts().sort_index()) == [3, 3, 3]
    assert path.assign_tercile(pd.Series([1.0, 2.0])).isna().all()


def test_interaction_uses_the_product_of_ranks_and_recovers_a_reversal_that_strengthens_with_state():
    rng = np.random.default_rng(7)
    days = np.repeat(pd.date_range("2020-01-01", periods=80), 12)
    diff = rng.normal(size=len(days))
    state = rng.normal(size=len(days))
    extend = rng.normal(size=len(days))
    y = diff * state + rng.normal(scale=0.15, size=len(days))
    frame = pd.DataFrame({
        "trading_date": days, "diff": diff, "impulse": state, "extend": extend, "gross_1130": y,
    })
    outcome, columns, day, n = path.interaction_terms(frame, "impulse", "extend")
    diff_r = path.centered_rank(frame, "diff")
    state_r = path.centered_rank(frame, "impulse")
    assert np.allclose(columns["低减高×状态"], (diff_r * state_r).to_numpy())
    assert n == len(frame)
    fitted = path.clustered_coefficients(outcome, columns, day)
    coef, t_stat = fitted["低减高×状态"]
    assert coef > 0 and t_stat > 8


def test_daily_spearman_keeps_a_datetime_index_and_is_empty_when_a_day_is_too_thin():
    days = np.repeat(pd.date_range("2020-01-01", periods=3), 6)
    frame = pd.DataFrame({
        "trading_date": days,
        "diff": np.tile(np.arange(6), 3),
        "gross_1130": np.tile(np.arange(6), 3) + np.repeat([0, 1, -1], 6),
    })
    ic = path.daily_spearman(frame, min_names=5)
    assert isinstance(ic.index, pd.DatetimeIndex) and len(ic) == 3
    thin = path.daily_spearman(frame.iloc[:4], min_names=5)
    assert isinstance(thin.index, pd.DatetimeIndex) and len(thin) == 0


def test_judge_passes_only_when_both_regressions_and_the_sign_counts_clear():
    decision = CONFIG["decision"]
    passed = path.judge(3.1, 2.8, 5, 4, 1, decision)
    assert passed["通过"]
    assert not path.judge(3.1, 2.4, 5, 4, 1, decision)["通过"]
    assert not path.judge(3.1, -2.8, 5, 4, 1, decision)["通过"]
    assert not path.judge(-3.1, -2.8, 5, 4, 1, decision)["通过"]
    assert not path.judge(3.1, 2.8, 4, 4, 1, decision)["通过"]
    assert not path.judge(3.1, 2.8, 5, 3, 1, decision)["通过"]
    assert path.judge(-3.1, -2.8, 5, 4, -1, decision)["通过"]
