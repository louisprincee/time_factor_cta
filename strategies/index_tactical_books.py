"""股指纯择时与因子倾斜策略。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from research.index.factors.index_tactical_factor_candidates import (
    FACTOR_TILT,
    FACTOR_DIRECTIONS,
    STANDARDIZATION_WINDOW,
    TACTICAL_ONLY,
    factor_votes,
    target_weights as tactical_target_weights,
)
from strategies.index_common import SYMBOLS, close_vol_scale, market_close_location
from tfcta import config as C

RALLY_MULTIPLIER = 2.0
INITIAL_CAPITAL = 10_000_000
DRAW_DOWN_LIMIT = 0.10
HALT_DAYS = 10
_ROOT = Path(__file__).parent

_COMMON_PARAMETERS = {
    "instruments": list(SYMBOLS),
    "market_clv_threshold": 0.4,
    "momentum_base": "none",
    "rally_multiplier": RALLY_MULTIPLIER,
    "vol_window": 20,
    "vol_target": 0.2,
    "drawdown_breaker": DRAW_DOWN_LIMIT,
    "halt_days": HALT_DAYS,
    "initial_capital": INITIAL_CAPITAL,
    "execution": "l1_30m",
    "open_limits": True,
    "signal_execution": "close t -> open t+1",
}

STRATEGY_PARAMETERS = {
    TACTICAL_ONLY.ID: {
        **_COMMON_PARAMETERS,
        "allocation_rule": "equal",
        "factor_directions": {},
    },
    FACTOR_TILT.ID: {
        **_COMMON_PARAMETERS,
        "allocation_rule": "factor shares proportional to 1 + favorable votes",
        "factor_directions": FACTOR_DIRECTIONS,
        "factor_standardization_window": STANDARDIZATION_WINDOW,
    },
}


@lru_cache(maxsize=1)
def _research_factor_panels():
    from research.index.factors.index_clv_cross_product_factors import load_panels

    return load_panels()[0]


@lru_cache(maxsize=1)
def _oos_factor_panels(end: str):
    from research.index.factors.index_tactical_factor_candidates import load_oos_factor_panels

    return load_oos_factor_panels(end)


def _factor_scores(index: pd.DatetimeIndex) -> pd.DataFrame:
    if index.empty or not isinstance(index, pd.DatetimeIndex):
        raise ValueError("股指因子评分日期必须为非空DatetimeIndex")
    research_index = index[index < pd.Timestamp(C.OOS_START)]
    oos_index = index[index >= pd.Timestamp(C.OOS_START)]
    scores = pd.DataFrame(0, index=index, columns=SYMBOLS, dtype="int64")
    if len(research_index):
        C.assert_research_only(C.INDEX_RESEARCH_DIR)
        research_panels = _research_factor_panels()
        research, _ = factor_votes(research_panels, research_index)
        scores.loc[research_index] = research
    if len(oos_index):
        if not len(research_index):
            raise C.HoldoutViolation("OOS因子预热必须包含研究期因子历史")
        oos_panels = _oos_factor_panels(str(oos_index.max().date()))
        combined_panels = {}
        for symbol in SYMBOLS:
            combined = pd.concat(
                [research_panels[symbol], oos_panels[symbol]]
            ).sort_index()
            if combined.index.has_duplicates:
                raise ValueError(f"{symbol} 研究/OOS因子日期重复")
            combined_panels[symbol] = combined
        combined_index = pd.DatetimeIndex([])
        for symbol in SYMBOLS:
            combined_index = combined_index.union(combined_panels[symbol].index)
        combined_index = combined_index.sort_values()
        combined_scores, _ = factor_votes(combined_panels, combined_index)
        scores.loc[oos_index] = combined_scores.reindex(oos_index)
    return scores


def _open_to_open(bars: dict[str, pd.DataFrame]) -> pd.Series:
    return (
        (bars["openw"].shift(-1) - bars["openw"]) / bars["open"]
    ).mean(axis=1, skipna=False)


def _validate_frozen_config(strategy_id: str) -> dict:
    config_path = _ROOT / strategy_id / f"{strategy_id}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parameters = STRATEGY_PARAMETERS[strategy_id]
    if (
        config.get("id") != strategy_id
        or config.get("status") != "frozen"
        or config.get("declared_at") != "2026-11-05"
        or config.get("frozen_at") != "2026-11-05"
        or config.get("parameters") != parameters
        or config.get("chosen") != parameters
        or "2015-04-16" not in config.get("sample", "")
        or "2021-12-30" not in config.get("sample", "")
        or "未读取2022至2025" not in config.get("sample", "")
        or "OOS尚未运行" not in config.get("sample", "")
    ):
        raise ValueError(f"{strategy_id} 冻结配置与实现不一致")
    return config


class TacticalBook:
    SYMBOLS = SYMBOLS
    DRAWDOWN_LIMIT = DRAW_DOWN_LIMIT
    HALT_DAYS = HALT_DAYS

    def __init__(self, strategy_id: str, name: str, detail: str, gross: float):
        self.ID = strategy_id
        self.NAME = name
        self.DETAIL = detail
        self.GROSS = gross
        self.CONFIG_PATH = _ROOT / strategy_id / f"{strategy_id}.json"

    def declared(self):
        return _validate_frozen_config(self.ID)

    frozen = declared

    @property
    def STATUS(self):
        return json.loads(self.CONFIG_PATH.read_text(encoding="utf-8"))["status"]

    @property
    def FILES(self):
        return [Path(__file__), self.CONFIG_PATH]

    def signal_inputs(self, bars, index):
        clv = market_close_location(bars, index)
        if self.ID == TACTICAL_ONLY.ID:
            scores = pd.DataFrame(0, index=index, columns=SYMBOLS, dtype="int64")
        else:
            scores = _factor_scores(index)
        return clv, scores

    def targets(self, inputs, bars, *, rally_z):
        market_clv, scores = inputs
        index = market_clv.index
        if tuple(scores.columns) != SYMBOLS or not scores.index.equals(index):
            raise ValueError("因子票数必须与市场 CLV 日期和品种对齐")
        if not rally_z.index.equals(index):
            raise ValueError("冲高因子必须与市场 CLV 日期对齐")

        open_returns = _open_to_open(bars)
        scale = close_vol_scale(open_returns).fillna(1.0).reindex(index)
        if scale.isna().any():
            raise ValueError("股指波动缩放未与市场CLV日期对齐")
        return tactical_target_weights(
            self.ID, market_clv, scores, rally_z, scale
        )


TACTICAL_ONLY_BOOK = TacticalBook(
    TACTICAL_ONLY.ID,
    "股指CLV纯择时等权",
    "市场CLV<0.4时全预算等权择时；冲高×2；无动量底仓",
    gross=1.0,
)
FACTOR_TILT_BOOK = TacticalBook(
    FACTOR_TILT.ID,
    "股指CLV因子仓位倾斜",
    "市场CLV<0.4时三品种均参与；按已登记正向因子票倾斜；冲高×2",
    gross=2.0,
)
