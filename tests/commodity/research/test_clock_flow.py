import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "commodity"))
flow = importlib.import_module("morning_clock_flow")
CONFIG = json.loads((Path(__file__).resolve().parents[3] / "config" / "commodity" / "clock_flow.json").read_text(encoding="utf-8"))


def test_config_freezes_three_predecision_candidates_and_signs():
    assert CONFIG["declared_at"] == "2026-10-14"
    assert "不读 2022" in CONFIG["sample"]
    assert [spec["id"] for spec in CONFIG["variables"]] == [
        "open_volume_surprise", "impact_per_volume", "volume_peak_clock"
    ]
    assert [spec["expected_interaction"] for spec in CONFIG["variables"]] == ["负", "正", "负"]
    assert "shift(1)" in CONFIG["variables"][0]["formula"]


def test_volume_surprise_uses_only_positive_available_baselines():
    assert flow.volume_surprise(200.0, 100.0) == pytest.approx(2.0)
    assert np.isnan(flow.volume_surprise(200.0, 0.0))
    assert np.isnan(flow.volume_surprise(np.nan, 100.0))


def test_price_impact_is_move_scaled_by_square_root_of_relative_volume():
    assert flow.impact_per_volume(-0.02, 4.0) == pytest.approx(0.01)
    assert np.isnan(flow.impact_per_volume(0.02, 0.0))


def test_volume_peak_clock_is_scaled_to_open_window_and_handles_missing_values():
    assert flow.volume_peak_clock([1, 2, 8, 3, 2]) == pytest.approx(0.5)
    assert np.isnan(flow.volume_peak_clock([np.nan, 1, np.nan]))
    assert np.isnan(flow.volume_peak_clock([4]))