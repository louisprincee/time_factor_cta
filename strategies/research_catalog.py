"""Exploratory strategies; deliberately separate from the frozen OOS catalog."""
from __future__ import annotations

from strategies import (
    cross_sectional_clock,
    trend_carry_full_clock,
    trend_high_atr_carry,
)

RESEARCH_EXPERIMENTS = (
    cross_sectional_clock,
    trend_high_atr_carry,
    trend_carry_full_clock,
)
