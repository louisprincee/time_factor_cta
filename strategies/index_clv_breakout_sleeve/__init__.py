"""CLV 与固定 55/20 突破趋势袖套组合策略。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.index_breakout import TREND_BUDGET, trend_targets_from_bars
from strategies.index_common import SYMBOLS
from strategies.index_tactical_books import TACTICAL_ONLY_BOOK
from tfcta import config as C

ID = "index_clv_breakout_sleeve"
NAME = "股指CLV加55/20突破趋势袖套"
GROSS = 1.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
INITIAL_CAPITAL = 10_000_000
CONFIG_PATH = Path(__file__).with_name(f"{ID}.json")
PARAMETERS = {
    "instruments": list(SYMBOLS),
    "entry_lookback": 55,
    "entry_price": "closew > prior_55_session_max_highw",
    "exit_lookback": 20,
    "exit_price": "closew < prior_20_session_min_loww",
    "trend_budget": TREND_BUDGET,
    "allocation": "equal across active symbols",
    "short_rule": "none",
    "clv_strategy": TACTICAL_ONLY_BOOK.ID,
    "clv_rally_multiplier": 2.0,
    "vol_window": 20,
    "vol_target": 0.2,
    "drawdown_breaker": DRAWDOWN_LIMIT,
    "halt_days": HALT_DAYS,
    "initial_capital": INITIAL_CAPITAL,
    "execution": "l1_30m",
    "open_limits": True,
    "signal_execution": "close t -> open t+1",
    "maximum_pre_vol_target": 2.5,
}
DETAIL = "纯CLV目标叠加固定55日突破/20日低点退出的多头袖套；趋势目标预算0.5，冲高只放大CLV袖套"


def declared_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sample = config.get("sample", "")
    if (
        config.get("id") != ID
        or config.get("status") != "frozen"
        or config.get("declared_at") != "2026-11-07"
        or config.get("frozen_at") != "2026-11-07"
        or config.get("parameters") != PARAMETERS
        or config.get("chosen") != PARAMETERS
        or "2015-04-16" not in sample
        or "2021-12-30" not in sample
        or "未读取2022至2025" not in sample
        or "OOS尚未运行" not in sample
    ):
        raise ValueError("55/20突破趋势策略冻结配置与实现不一致")
    return config


def targets(market_clv, bars, *, rally_z):
    index = market_clv.index
    if not rally_z.index.equals(index):
        raise ValueError("冲高因子必须与市场 CLV 日期对齐")
    scores = pd.DataFrame(0, index=index, columns=SYMBOLS, dtype="int64")
    clv = TACTICAL_ONLY_BOOK.targets((market_clv, scores), bars, rally_z=rally_z)
    _, trend = trend_targets_from_bars(bars, index)
    combined = clv.add(trend, fill_value=0.0)
    if (
        not np.isfinite(combined.to_numpy(dtype=float)).all()
        or combined.lt(0).any().any()
        or combined.gt(PARAMETERS["maximum_pre_vol_target"] + 1e-12).any().any()
    ):
        raise ValueError("CLV 与趋势目标超出冻结组合范围")
    return combined


class Book:
    ID, NAME, SYMBOLS = ID, NAME, SYMBOLS
    GROSS, DRAWDOWN_LIMIT, HALT_DAYS = GROSS, DRAWDOWN_LIMIT, HALT_DAYS
    CONFIG_PATH = CONFIG_PATH
    DETAIL = DETAIL
    targets = staticmethod(targets)

    @staticmethod
    def declared():
        return declared_config()

    frozen = declared

    @property
    def STATUS(self):
        return declared_config()["status"]

    @property
    def FILES(self):
        return [Path(__file__), Path(__file__).parents[1] / "index_breakout.py", CONFIG_PATH]


BOOK = Book()
