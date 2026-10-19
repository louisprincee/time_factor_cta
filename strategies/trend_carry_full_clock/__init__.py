"""Exploratory trend/carry agreement timed by the full-trading-day clock."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from strategies.book import HOLDING_DAYS, LOOKBACK
from strategies.trend_carry_common import RISK_SCALE_RULE, risk_scale
from strategies.trend_carry_common import signal as _direction_signal
from tfcta import config as C

ID = "trend_carry_full_clock"
NAME = "趋势展期一致全日时钟"
DETAIL = "10日趋势与滞后carry同向/全交易日时钟同向尾部/ATR风险缩放0.5-2/持有5日"
TAIL_QUANTILE = 0.20
CONFIG_PATH = C.CONFIG_DIR / "research" / "trend_carry_full_clock.json"


def load_config(path=CONFIG_PATH) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = config.get("rules", {})
    if (
        config.get("id") != ID
        or config.get("status") != "exploratory_post_hoc_candidate"
        or "不读取2022及2023年以后" not in config.get("sample", "")
        or rules.get("clock_lookback_days") != LOOKBACK
        or rules.get("clock_tail_quantile") != TAIL_QUANTILE
        or rules.get("clock_threshold_comparison") != "strict"
        or rules.get("clock_scope") != "full_trading_day"
        or rules.get("risk_scale") != RISK_SCALE_RULE
        or rules.get("holding_days") != HOLDING_DAYS
    ):
        raise ValueError("全交易日时钟组合配置与实现或样本边界不一致")
    return config


def signal(
    trend: pd.DataFrame,
    carry: pd.DataFrame,
    full_clock: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the same direction rule to the full-trading-day clock input."""
    return _direction_signal(trend, carry, full_clock)
