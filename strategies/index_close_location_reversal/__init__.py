"""IF/IH/IC 收盘位置长仓策略，尚未冻结，不开放样本外。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SYMBOLS = ("IF", "IH", "IC")
TARGET_THRESHOLD = 0.5
GROSS = 1.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
ID = "index_close_location_reversal"
NAME = "股指收盘位置反转（研究候选）"
CONFIG_PATH = Path(__file__).with_name(f"{ID}.json")
STATUS = "exploratory_predeclared"


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
