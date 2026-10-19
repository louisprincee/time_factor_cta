"""当前保留的四条策略。2026-10-19 已一起冻结，样本外只打开一次。"""
from __future__ import annotations

from strategies import regime_two_leg, trend_adx_range, trend_ema3, trend_high_atr

SELECTED = (trend_ema3, trend_high_atr, trend_adx_range, regime_two_leg)
