"""股指双半仓：中期动量决定底仓多空，弱收盘冲高决定择时多头。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.index_common import SYMBOLS, THRESHOLD, close_vol_scale
from tfcta import config as C

ID = "index_clv_rally_base_long"
NAME = "股指弱收盘中期动量底仓"
GROSS = 1.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
RALLY_MULTIPLIER = 1.5
BASE_GROSS = 0.5
TACTICAL_GROSS = 0.5
MOMENTUM_WINDOWS = (63, 126, 252)
CONFIG_PATH = Path(__file__).with_name(f"{ID}.json")
PARAMETERS = {
    "instruments": list(SYMBOLS),
    "clv_threshold": THRESHOLD,
    "signal_rule": (
        "half the unit budget follows the majority sign of 63/126/252-session "
        "compounded equal-weight close returns, long when at least two votes "
        "are positive and short otherwise; add the other half long when mean of "
        "IF/IH/IC CLV < 0.4"
    ),
    "base_gross": BASE_GROSS,
    "momentum_windows": list(MOMENTUM_WINDOWS),
    "base_rule": "each complete window votes +1 for compounded return > 0, -1 otherwise; incomplete windows vote +1; majority positive gives base +0.5, otherwise -0.5",
    "tactical_gross": TACTICAL_GROSS,
    "allocation_rule": "equal multipliers 1/1/1",
    "short_rule": "short the base half when at least two momentum votes are negative",
    "rally_window": 60,
    "rally_minimum": 40,
    "rally_gate": 0.0,
    "rally_multiplier": RALLY_MULTIPLIER,
    "rally_applies_to": "tactical half only",
    "vol_window": 20,
    "vol_target": 0.2,
    "vol_scale_cap": 1.0,
    "vol_warmup": "scale 1 until the 20-session estimate exists",
    "gross_exposure_limit": GROSS,
    "drawdown_breaker": DRAWDOWN_LIMIT,
    "halt_days": HALT_DAYS,
    "initial_capital": 10000000,
    "execution": "l1_30m",
    "open_limits": True,
    "signal_execution": "close t -> open t+1",
    "rebalance": "only target change; whole lots",
}
PRICE_FIELDS = ("open", "openw", "highw", "loww", "close", "closew")


def declared_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config.get("id") != ID
        or config.get("status") != "exploratory_predeclared"
        or config.get("declared_at") != "2026-11-04"
        or config.get("parameters") != PARAMETERS
        or config.get("chosen") != PARAMETERS
        or config.get("parameters", {}).get("base_gross") != BASE_GROSS
        or config.get("sample") is None
        or "2015-04-16" not in config["sample"]
        or "2021-12-30" not in config["sample"]
        or "不运行OOS" not in config["sample"]
    ):
        raise ValueError("股指双半仓参数与实现不一致")
    return config


def _open_to_open(frames: dict[str, pd.DataFrame]) -> pd.Series:
    return (
        (frames["openw"].shift(-1) - frames["openw"]) / frames["open"]
    ).mean(axis=1, skipna=False)


def _close_to_close(frames: dict[str, pd.DataFrame]) -> pd.Series:
    prior = frames["close"].shift(1)
    return (
        (frames["closew"] - frames["closew"].shift(1)) / prior.where(prior > 0)
    ).mean(axis=1, skipna=False)


def _base_sign(close_to_close: pd.Series) -> pd.Series:
    """3/6/12月绝对动量多数表决；完整窗口含当日收盘，缺预热票先为多。"""
    votes = []
    for window in MOMENTUM_WINDOWS:
        momentum = (1.0 + close_to_close).rolling(window, min_periods=window).apply(
            np.prod, raw=True
        ) - 1.0
        votes.append(pd.Series(
            np.where(momentum.isna() | momentum.gt(0), 1.0, -1.0),
            index=close_to_close.index,
        ))
    return pd.Series(
        np.where(pd.concat(votes, axis=1).sum(axis=1).gt(0), 1.0, -1.0),
        index=close_to_close.index,
    )


def _by_partition(index: pd.DatetimeIndex, frames: dict[str, pd.DataFrame], build) -> pd.Series:
    oos_mask = index >= pd.Timestamp(C.OOS_START)
    if not oos_mask.any():
        return build(frames)
    research_index, oos_index = index[~oos_mask], index[oos_mask]
    if research_index.empty:
        raise C.HoldoutViolation("OOS目标计算必须包含研究期历史")
    parts = []
    for part in (research_index, oos_index):
        subset = {field: frame.reindex(part) for field, frame in frames.items()}
        parts.append(build(subset))
    return pd.concat(parts)


def rally_base_targets(
    market_clv: pd.Series,
    bars: dict[str, pd.DataFrame],
    *,
    rally_z: pd.Series | None = None,
) -> pd.DataFrame:
    index = market_clv.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("股指CLV索引必须为DatetimeIndex")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("股指信号日期必须唯一升序")
    if index.empty:
        raise ValueError("股指信号日期不能为空")
    if index.min() < pd.Timestamp("2015-04-16") or index.max() > pd.Timestamp(
        C.DEFAULT_OOS_END
    ):
        raise C.HoldoutViolation("股指策略日期须在2015-04-16至2025年范围内")
    if market_clv.isna().any() or not market_clv.between(0, 1).all():
        raise ValueError("股指CLV缺失或越界")

    frames = {
        field: bars[field].reindex(index=index, columns=list(SYMBOLS))
        for field in PRICE_FIELDS
    }
    if any(not np.isfinite(frame.to_numpy(dtype=float)).all() for frame in frames.values()):
        raise ValueError("常驻多头行情未完全对齐")
    if frames["open"].le(0).any().any() or frames["close"].le(0).any().any() or (
        frames["highw"] <= frames["loww"]
    ).any().any():
        raise ValueError("常驻多头价格区间无效")

    if rally_z is None:
        from strategies.index_common import high_open_z

        rally_z = high_open_z(bars)
    if not rally_z.index.equals(index):
        raise ValueError("冲高因子必须与信号日期对齐")

    scale = _by_partition(
        index, frames, lambda subset: close_vol_scale(_open_to_open(subset)).fillna(1.0)
    )
    base_sign = _by_partition(index, frames, lambda subset: _base_sign(_close_to_close(subset)))
    # 冲高只放大择时半仓；两半在同一合约净额持有。
    tilt = pd.Series(np.where(rally_z.gt(0), RALLY_MULTIPLIER, 1.0), index=index)
    gross = scale * (
        BASE_GROSS * base_sign
        + market_clv.lt(THRESHOLD).astype(float) * TACTICAL_GROSS * tilt
    )
    return pd.DataFrame({symbol: gross for symbol in SYMBOLS}, index=index).shift(1).fillna(0.0)


class Book:
    ID, NAME, SYMBOLS = ID, NAME, SYMBOLS
    GROSS, DRAWDOWN_LIMIT, HALT_DAYS = GROSS, DRAWDOWN_LIMIT, HALT_DAYS
    CONFIG_PATH = CONFIG_PATH
    DETAIL = "底仓一半按63/126/252日动量多数表决多空；弱收盘加另一半多头，冲高只放大择时半仓"
    declared = staticmethod(declared_config)
    frozen = staticmethod(declared_config)
    targets = staticmethod(rally_base_targets)

    @property
    def STATUS(self):
        return declared_config()["status"]

    @property
    def FILES(self):
        return [Path(__file__), Path(__file__).parents[1] / "index_common.py", CONFIG_PATH]


BOOK = Book()
