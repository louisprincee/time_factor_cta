"""IF/IH/IC 收盘位置长仓策略，尚未冻结，不开放样本外。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from strategies import market_clock_slow_short_20_60_120 as T

SYMBOLS = ("IF", "IH", "IC")
TARGET_THRESHOLD = 0.5
GROSS = 1.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
ID = "index_close_location_reversal"
NAME = "股指收盘位置反转（研究候选）"
CONFIG_PATH = Path(__file__).with_name(f"{ID}.json")
STATUS = "exploratory_predeclared"
STRICT_SWAP_ID = "index_close_location_reversal_strict_swap"
STRICT_SWAP_CONFIG_PATH = Path(__file__).with_name(f"{STRICT_SWAP_ID}.json")


def declared_config(path: Path = CONFIG_PATH) -> dict:
    """读取并核对候选规则；探索性策略不具备冻结或样本外状态。"""
    config = json.loads(path.read_text(encoding="utf-8"))
    parameters = config.get("parameters", {})
    if (
        config.get("id") != ID
        or config.get("status") != STATUS
        or config.get("frozen_at")
        or config.get("chosen")
        or parameters.get("instruments") != list(SYMBOLS)
        or parameters.get("target_threshold") != TARGET_THRESHOLD
        or parameters.get("gross_exposure_limit") != GROSS
        or parameters.get("risk")
        != (
            f"Use the shared {DRAWDOWN_LIMIT:.0%} peak drawdown breaker and halt new "
            f"positions for {HALT_DAYS} trading days."
        )
        or parameters.get("score_start") != "2016-01-13"
        or parameters.get("score_end") != "2021-12-31"
        or "Do not read 2022" not in config.get("sample", "")
        or "2023-2025 locked partitions" not in config.get("sample", "")
        or "or 2026" not in config.get("sample", "")
    ):
        raise ValueError("股指收盘位置策略登记与实现或样本边界不一致")
    return config


def targets(market_close_location: pd.Series) -> pd.DataFrame:
    """收盘位置低于固定中点时，下一交易日等权持有股指多头。"""
    if not isinstance(market_close_location, pd.Series):
        raise TypeError("市场收盘位置必须是 pandas Series")
    if not isinstance(market_close_location.index, pd.DatetimeIndex):
        raise TypeError("市场收盘位置索引必须为 DatetimeIndex")
    if market_close_location.index.has_duplicates:
        raise ValueError("市场收盘位置日期不能重复")
    if not market_close_location.index.is_monotonic_increasing:
        raise ValueError("市场收盘位置日期必须升序")
    if market_close_location.isna().any():
        raise ValueError("市场收盘位置不能缺失")
    if not market_close_location.between(0.0, 1.0).all():
        raise ValueError("市场收盘位置必须在 [0, 1] 内")

    exposure = market_close_location.lt(TARGET_THRESHOLD).shift(
        1, fill_value=False
    ).astype(float)
    return pd.DataFrame({symbol: exposure for symbol in SYMBOLS}, index=exposure.index)


def market_close_location(
    bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> pd.Series:
    """IF/IH/IC 等权收盘位置；必须三品种都有有效日内区间。"""
    close = bars["closew"].reindex(index=index, columns=SYMBOLS)
    high = bars["highw"].reindex(index=index, columns=SYMBOLS)
    low = bars["loww"].reindex(index=index, columns=SYMBOLS)
    width = high - low
    if close.isna().any().any() or high.isna().any().any() or low.isna().any().any():
        raise ValueError("股指收盘位置行情未完全对齐")
    if width.le(0).any().any():
        raise ValueError("股指日内最高价必须高于最低价")
    clv = (close - low) / width
    if (
        not np.isfinite(clv.to_numpy()).all()
        or clv.lt(0).any().any()
        or clv.gt(1).any().any()
    ):
        raise ValueError("股指收盘位置超出 [0, 1] 或包含无效值")
    return clv.mean(axis=1, skipna=False).rename("market_clv")


def strict_swap_targets(
    market_close_location: pd.Series,
    bars: dict[str, pd.DataFrame],
    carry: pd.DataFrame,
) -> pd.DataFrame:
    """替换原市场分数为 CLV，并保留滞后展期筛选和趋势空头。"""
    if not isinstance(market_close_location.index, pd.DatetimeIndex):
        raise TypeError("市场收盘位置索引必须为 DatetimeIndex")
    if market_close_location.index.has_duplicates or not market_close_location.index.is_monotonic_increasing:
        raise ValueError("市场收盘位置日期必须唯一且升序")
    if market_close_location.isna().any() or not market_close_location.between(0.0, 1.0).all():
        raise ValueError("市场收盘位置必须完整且在 [0, 1] 内")
    index = market_close_location.index
    carry = carry.reindex(index=index, columns=SYMBOLS)
    lagged = carry.shift(1)
    trend = T.symbol_trend(bars, SYMBOLS).reindex(index)
    market_return = T.close_return(bars, SYMBOLS).reindex(index).mean(axis=1)
    bear = T.market_trend(market_return).lt(0)
    long = pd.DataFrame(1.0, index=index, columns=SYMBOLS)
    long = long.where(lagged.gt(0), 0.0).mul(
        market_close_location.lt(TARGET_THRESHOLD).astype(float), axis=0
    )
    bear_mask = pd.DataFrame(
        {symbol: bear.reindex(index).eq(True) for symbol in SYMBOLS},
        index=index,
    )
    short = trend.lt(0) & lagged.lt(0) & bear_mask
    weights = long.where(~short, -1.0)
    return weights.shift(1).fillna(0.0)


class StrictSwapBook:
    ID = STRICT_SWAP_ID
    NAME = "股指原策略（仅将分数换成收盘位置）"
    SYMBOLS = SYMBOLS
    GROSS = GROSS
    DRAWDOWN_LIMIT = DRAWDOWN_LIMIT
    HALT_DAYS = HALT_DAYS
    CONFIG_PATH = STRICT_SWAP_CONFIG_PATH
    DETAIL = (
        "将原股指市场分数替换为 IF/IH/IC 等权收盘位置小于0.5，保留滞后一日展期收益大于0的多头筛选；"
        "市场熊市、品种趋势、负展期空头及执行风控按冻结规格执行"
    )

    @staticmethod
    def declared() -> dict:
        config = json.loads(STRICT_SWAP_CONFIG_PATH.read_text(encoding="utf-8"))
        parameters = config.get("parameters") or {}
        if (
            config.get("id") != STRICT_SWAP_ID
            or config.get("status") != "frozen"
            or not config.get("frozen_at")
            or config.get("chosen") != parameters
            or parameters.get("instruments") != list(SYMBOLS)
            or parameters.get("long_signal")
            != "Equal-weight IF/IH/IC close location (closew - loww) / (highw - loww) < 0.5"
            or parameters.get("long_carry_filter") != "Prior trading day's annualized_yield > 0"
            or parameters.get("roll_yield") != "annualized_yield"
            or parameters.get("roll_yield_lag_days") != 1
            or parameters.get("short_rule")
            != "market_trend < 0 and symbol_trend < 0 and lagged annualized_yield < 0"
            or parameters.get("trend_windows") != list(T.TREND_WINDOWS)
            or parameters.get("short_weight") != -1.0
            or parameters.get("conflict_rule") != "short_overrides_long"
            or parameters.get("gross_exposure_limit") != GROSS
            or parameters.get("gross_split") != "pool"
            or parameters.get("execution")
            != "Close signal enters at the next trading day's open; unchanged weight stays on."
            or parameters.get("drawdown_breaker") != DRAWDOWN_LIMIT
            or parameters.get("halt_days") != HALT_DAYS
            or parameters.get("index_tick_2015") != 0.2
            or parameters.get("later_slippage") != "Prior-year estimated tick, one tick per side."
            or "Do not read 2022" not in config.get("sample", "")
            or "do not read 2026" not in config.get("sample", "").lower()
        ):
            raise ValueError("股指严格替换冻结规格与实现不一致")
        return config

    @staticmethod
    def frozen() -> dict:
        return StrictSwapBook.declared()

    @staticmethod
    def targets(
        market_close_location: pd.Series,
        bars: dict[str, pd.DataFrame],
        carry: pd.DataFrame,
    ) -> pd.DataFrame:
        return strict_swap_targets(market_close_location, bars, carry)


class Book:
    ID = ID
    NAME = NAME
    SYMBOLS = SYMBOLS
    GROSS = GROSS
    DRAWDOWN_LIMIT = DRAWDOWN_LIMIT
    HALT_DAYS = HALT_DAYS
    STATUS = STATUS

    @staticmethod
    def declared() -> dict:
        return declared_config()

    @staticmethod
    def targets(market_close_location: pd.Series) -> pd.DataFrame:
        return targets(market_close_location)


BOOK = Book()
BOOK.declared()
STRICT_SWAP_BOOK = StrictSwapBook()
STRICT_SWAP_BOOK.declared()
