"""收盘位置低于 0.4 做多；熊市且收盘不弱时做空。仓位按已实现波动缩小，不超过原来的 1 倍。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from strategies import commodity_original as T
from strategies.index_close_location import SYMBOLS, market_close_location
from tfcta import config as C

THRESHOLD = 0.4
GROSS = 1.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
VOL_TARGET = 0.20
VOL_WINDOW = 20
VOL_RULE = "min(1, 0.20 / trailing 20-day equal-weight open-to-open volatility); do not scale above 1"
ID = "index_close_location_bear_short"
NAME = "股指收盘位置熊市空头"
CONFIG_PATH = Path(__file__).with_name(f"{ID}.json")
STATUS = "frozen"


def declared_config(path: Path = CONFIG_PATH) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    parameters = config.get("parameters", {})
    sample = config.get("sample", "")
    if (
        config.get("id") != ID
        or config.get("status") != STATUS
        or not config.get("frozen_at")
        or config.get("chosen") != parameters
        or parameters.get("instruments") != list(SYMBOLS)
        or parameters.get("trend_windows") != list(T.TREND_WINDOWS)
        or parameters.get("carry_filter") != "none"
        or parameters.get("gross_exposure_limit") != GROSS
        or parameters.get("drawdown_breaker") != DRAWDOWN_LIMIT
        or parameters.get("halt_days") != HALT_DAYS
        or parameters.get("position_scale") != VOL_RULE
        or parameters.get("short_rule")
        != "market_trend < 0 and equal-weight close location >= 0.4, next open, weight -1 on IF, IH and IC"
        or "不读 2022" not in sample
        or "不读 2026" not in sample
        or config.get("threshold_choice", {}).get("result", "").find("选定 0.4") < 0
    ):
        raise ValueError("熊市空头策略登记与实现不一致")
    return config


def close_vol_scale(open_to_open: pd.Series) -> pd.Series:
    """收盘 t 已知的乘数。用截至当开盘的已完成开盘到开盘段，不超过 1。"""
    done = open_to_open.shift(1)
    realized = done.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(C.TRADING_DAYS_PER_YEAR)
    return (VOL_TARGET / realized).clip(upper=1.0)


def _open_to_open(bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.Series:
    opened = bars["open"].reindex(index=index, columns=list(SYMBOLS))
    adjusted = bars["openw"].reindex(index=index, columns=list(SYMBOLS))
    if opened.isna().any().any() or adjusted.isna().any().any() or opened.le(0).any().any():
        raise ValueError("波动率缩放缺少完整的正开盘价")
    return ((adjusted.shift(-1) - adjusted) / opened).mean(axis=1)


def decision_side(market_clv: pd.Series, bars: dict[str, pd.DataFrame]) -> pd.Series:
    """收盘当时的方向：弱收盘 +1，熊市且收盘不弱 −1，否则 0。尚未后移。"""
    if not isinstance(market_clv.index, pd.DatetimeIndex):
        raise TypeError("市场收盘位置索引必须为 DatetimeIndex")
    if market_clv.index.has_duplicates or not market_clv.index.is_monotonic_increasing:
        raise ValueError("市场收盘位置日期必须唯一且升序")
    if market_clv.isna().any() or not market_clv.between(0.0, 1.0).all():
        raise ValueError("市场收盘位置必须完整且在 [0, 1] 内")
    index = market_clv.index
    market_return = T.close_return(bars, SYMBOLS).reindex(index).mean(axis=1)
    bear = T.market_trend(market_return).reindex(index).lt(0)
    long = market_clv.lt(THRESHOLD)
    short = bear & ~long
    side = pd.Series(0.0, index=index)
    return side.mask(long, 1.0).mask(short, -1.0)


def bear_reversal_targets(market_clv: pd.Series, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """收盘决策，下一开盘执行。方向乘以波动率乘数，乘数不超过 1。"""
    side = decision_side(market_clv, bars)
    scale = close_vol_scale(_open_to_open(bars, side.index)).reindex(side.index).fillna(1.0)
    weighted = side * scale
    return pd.DataFrame({symbol: weighted for symbol in SYMBOLS}, index=side.index).shift(1).fillna(0.0)


def unscaled_targets(market_clv: pd.Series, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """同一方向、不缩放。只给已经跑过的对照用。"""
    side = decision_side(market_clv, bars)
    return pd.DataFrame({symbol: side for symbol in SYMBOLS}, index=side.index).shift(1).fillna(0.0)


def long_only_targets(market_clv: pd.Series) -> pd.DataFrame:
    """同一阈值的多头对照，没有空头。"""
    side = market_clv.lt(THRESHOLD).astype(float)
    return pd.DataFrame({symbol: side for symbol in SYMBOLS}, index=market_clv.index).shift(1).fillna(0.0)


class Book:
    ID = ID
    NAME = NAME
    SYMBOLS = SYMBOLS
    GROSS = GROSS
    DRAWDOWN_LIMIT = DRAWDOWN_LIMIT
    HALT_DAYS = HALT_DAYS
    STATUS = STATUS
    CONFIG_PATH = CONFIG_PATH
    DETAIL = "等权收盘位置小于 0.4 则次日做多；熊市且收盘位置不低于 0.4 则次日做空；仓位按过去 20 日波动缩小，不超过 1 倍；无展期过滤"

    @staticmethod
    def declared() -> dict:
        return declared_config()

    @staticmethod
    def frozen() -> dict:
        return declared_config()

    @property
    def FILES(self) -> list[Path]:
        """打开 2022 时记进台账的文件哈希。"""
        return [Path(__file__), CONFIG_PATH]

    @staticmethod
    def targets(market_clv: pd.Series, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return bear_reversal_targets(market_clv, bars)


BOOK = Book()
BOOK.declared()
__all__ = ["BOOK", "THRESHOLD", "VOL_RULE", "bear_reversal_targets", "close_vol_scale", "long_only_targets", "market_close_location", "unscaled_targets"]
