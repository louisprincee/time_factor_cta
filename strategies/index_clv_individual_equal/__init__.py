"""股指单品种弱收盘信号，活跃品种等预算做多。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.index_close_location import (
    SYMBOLS,
    close_vol_scale,
    high_open_z,
)

ID = "index_clv_individual_equal_long"
NAME = "股指单品种弱收盘活跃等权多头"
THRESHOLD = 0.4
GROSS = 1.5
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
VOL_WINDOW = 20
VOL_TARGET = 0.20
RALLY_MULTIPLIER = 1.5
BASE_WEIGHT_SUM = 2.0
CONFIG_PATH = Path(__file__).with_name(f"{ID}.json")
PARAMETERS = {
    "instruments": list(SYMBOLS),
    "individual_clv_threshold": THRESHOLD,
    "signal_rule": "each symbol long when its own CLV < 0.4",
    "allocation_rule": "equal across signaled symbols; flat when none signal",
    "base_target_weight_sum": BASE_WEIGHT_SUM,
    "short_rule": "none",
    "rally_window": 60,
    "rally_minimum": 40,
    "rally_gate": 0.0,
    "rally_multiplier": RALLY_MULTIPLIER,
    "vol_window": VOL_WINDOW,
    "vol_target": VOL_TARGET,
    "vol_scale_cap": 1.0,
    "gross_exposure_limit": GROSS,
    "drawdown_breaker": DRAWDOWN_LIMIT,
    "halt_days": HALT_DAYS,
    "initial_capital": 10000000,
    "execution": "l1_30m",
    "open_limits": True,
    "signal_execution": "close t -> open t+1",
    "rebalance": "only target change; whole lots",
}


def declared_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config.get("id") != ID
        or config.get("status") != "frozen"
        or not config.get("frozen_at")
        or config.get("parameters") != PARAMETERS
        or config.get("chosen") != PARAMETERS
    ):
        raise ValueError("股指单品种等权冻结规则与实现不一致")
    sample = config.get("sample", "")
    if "2015-04-16" not in sample or "尚未回测" not in sample or "不读 2026" not in sample:
        raise ValueError("股指单品种等权规则缺少样本边界")
    return config


def individual_close_locations(
    bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> pd.DataFrame:
    close = bars["closew"].reindex(index=index, columns=SYMBOLS)
    high = bars["highw"].reindex(index=index, columns=SYMBOLS)
    low = bars["loww"].reindex(index=index, columns=SYMBOLS)
    width = high - low
    clv = (close - low) / width
    if (
        close.isna().any().any()
        or high.isna().any().any()
        or low.isna().any().any()
        or width.le(0).any().any()
        or not np.isfinite(clv.to_numpy()).all()
        or clv.lt(0).any().any()
        or clv.gt(1).any().any()
    ):
        raise ValueError("股指单品种收盘位置缺失或越界")
    return clv


def individual_equal_targets(clv, bars, *, rally_z=None):
    if not isinstance(clv.index, pd.DatetimeIndex):
        raise TypeError("股指单品种CLV索引必须为DatetimeIndex")
    if (
        clv.index.has_duplicates
        or not clv.index.is_monotonic_increasing
        or tuple(clv.columns) != SYMBOLS
    ):
        raise ValueError("股指单品种CLV日期或品种列无效")
    if clv.isna().any().any() or not clv.apply(lambda col: col.between(0, 1).all()).all():
        raise ValueError("股指单品种CLV缺失或越界")

    index = clv.index
    opened = bars["open"].reindex(index=index, columns=SYMBOLS)
    adjusted = bars["openw"].reindex(index=index, columns=SYMBOLS)
    if opened.isna().any().any() or adjusted.isna().any().any() or opened.le(0).any().any():
        raise ValueError("波动缩放缺少完整正开盘价")
    returns = ((adjusted.shift(-1) - adjusted) / opened).mean(axis=1, skipna=False)
    scale = close_vol_scale(returns).fillna(1.0)

    if rally_z is None:
        rally_z = high_open_z(bars)
    tilt = pd.Series(
        np.where(rally_z.reindex(index).gt(0), RALLY_MULTIPLIER, 1.0),
        index=index,
    )

    signals = clv.lt(THRESHOLD)
    active_count = signals.sum(axis=1)
    per_symbol_weight = BASE_WEIGHT_SUM / active_count.where(active_count > 0)
    weighted = signals.mul(per_symbol_weight, axis=0).mul(scale * tilt, axis=0)
    return weighted.shift(1).fillna(0.0)


class Book:
    ID, NAME, SYMBOLS = ID, NAME, SYMBOLS
    GROSS, DRAWDOWN_LIMIT, HALT_DAYS = GROSS, DRAWDOWN_LIMIT, HALT_DAYS
    STATUS, CONFIG_PATH = "frozen", CONFIG_PATH
    DETAIL = "各品种自身CLV<0.4；活跃品种等分风险预算；保留冲高放大和20日波动缩放；无空头"
    declared = staticmethod(declared_config)
    frozen = staticmethod(declared_config)
    signal_inputs = staticmethod(individual_close_locations)
    targets = staticmethod(individual_equal_targets)

    @property
    def FILES(self):
        shared_rules = Path(__file__).parents[1] / "index_close_location" / "__init__.py"
        return [Path(__file__), shared_rules, CONFIG_PATH]


BOOK = Book()
