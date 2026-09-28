import numpy as np

from tfcta.research.analysis.screen import passes_screen


def test_passes_screen_for_stable_ic_and_significant_t():
    assert passes_screen([0.02, 0.03, 0.01, 0.04, 0.02, 0.03], 2.1)


def test_screen_rejects_sign_changes():
    assert not passes_screen([0.02, 0.03, -0.01, 0.04, 0.02, 0.03], 3.0)


def test_screen_uses_strict_mean_ic_and_t_thresholds():
    assert not passes_screen([0.01] * 6, 3.0)
    assert not passes_screen([0.02] * 6, 2.0)


def test_screen_requires_every_requested_year():
    assert not passes_screen([0.02, 0.03, np.nan, 0.04, 0.02, 0.03], 3.0)


def test_screen_allows_a_shorter_symbol_history_when_requested():
    assert passes_screen([0.02, 0.03, 0.04, 0.02], 2.1, min_years=4)
