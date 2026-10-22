"""按池宽1倍加慢趋势空头：多头腿就是 market_clock_pool，市场慢趋势向下时再加一条趋势加展期的空头腿。

空头腿不用时间因子（对偶空头研究期失败，分数的信息只在多头一侧），用品种自身趋势向下且滞后一日展期收益为负。
两条腿都按当年池宽分 1 倍名义敞口；一个品种同一天只会在一条腿上（多头要展期 > 0，空头要展期 < 0）。
规格在同目录 JSON，导入时跟代码核对。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategies import market_clock_carry as S

TREND_WINDOWS = (60, 120, 250)


def close_return(bars: dict, columns) -> pd.DataFrame:
    """收盘到收盘日收益：复权收盘之差除以前一日原始收盘，收盘 t 已知。"""
    close, closew = bars["close"].reindex(columns=columns), bars["closew"].reindex(columns=columns)
    prior = close.shift(1)
    return (closew - closew.shift(1)) / prior.where(prior > 0)


def symbol_trend(bars: dict, columns) -> pd.DataFrame:
    """60、120、250 日收益符号的平均，取值 −1、−1/3、1/3、1；任一窗口缺失则为缺失。"""
    close, closew = bars["close"].reindex(columns=columns), bars["closew"].reindex(columns=columns)
    signs = []
    for window in TREND_WINDOWS:
        base = close.shift(window)
        change = (closew - closew.shift(window)) / base.where(base > 0)
        signs.append(np.sign(change))
    return sum(signs) / len(signs)


def market_trend(daily: pd.Series) -> pd.Series:
    """池子日收益累计 60、120、250 日后的符号平均；任一窗口不足则为缺失。"""
    signs = [np.sign(daily.rolling(window, min_periods=window).sum()) for window in TREND_WINDOWS]
    return sum(signs) / len(signs)


def decision_weights(score: pd.Series, carry: pd.DataFrame, trend: pd.DataFrame,
                     bear: pd.Series) -> pd.DataFrame:
    """收盘决策。carry 已滞后一天。多头同 market_clock_pool；熊市（缺失当作不是）时趋势和展期都为负的做空。"""
    long_side = S.decision_weights(score, carry)
    short_side = -(trend.reindex_like(long_side).lt(0) & carry.reindex_like(long_side).lt(0)).astype(float)
    bear = bear.reindex(long_side.index).eq(True)
    return long_side + short_side.mul(bear.astype(float), axis=0)


def targets(signed: dict, bars: dict, universe: dict, index: pd.DatetimeIndex, columns,
            carry: pd.DataFrame | None = None) -> pd.DataFrame:
    """收盘决策后移到下一交易日第一根开盘。市场慢趋势只用 index 上的池内日收益。"""
    columns = pd.Index(columns)
    pool = S.pool_mask(universe, index, columns)
    if carry is None:
        carry = S.roll_yield(index, columns)
        S.require_carry_coverage(carry, pool)
    lagged = carry.reindex(index=index, columns=columns).shift(1)
    score = S.market_score(signed, universe, index).reindex(index)
    trend = symbol_trend(bars, columns).reindex(index)
    daily = close_return(bars, columns).reindex(index).where(pool).mean(axis=1)
    bear = market_trend(daily).lt(0)
    return decision_weights(score, lagged, trend, bear).shift(1).fillna(0.0).where(pool, 0.0)


class Book(S.Book):
    ID = "market_clock_slow_short"
    NAME = "按池宽1倍加慢趋势空头"
    BEAR_TREND_WINDOWS = TREND_WINDOWS
    BEAR_RULE = "market_trend < 0"
    CONFIG_PATH = Path(__file__).with_name("market_clock_slow_short.json")
    DETAIL = (
        "多头同按池宽1倍；池子60/120/250日累计收益符号均值小于0时，"
        "品种自身60/120/250日收益符号均值小于0且滞后一日年化展期收益小于0的做空1份"
        "/两腿都按当年池宽分1倍名义敞口/下一交易日第一根开盘成交/权重不变则留仓/回撤10%清仓并停10个交易日"
    )

    @property
    def FILES(self) -> list[Path]:
        return [Path(S.__file__), Path(__file__), self.CONFIG_PATH]

    def check(self, config: dict) -> dict:
        config = S.check_config(config, self.ID, self.GROSS)
        parameters = config["parameters"]
        if (
            parameters.get("short_trend_windows") != list(TREND_WINDOWS)
            or parameters.get("short_rule") != "symbol_trend < 0 and lagged annualized_yield < 0"
            or parameters.get("short_weight") != -1.0
            or parameters.get("bear_trend_windows") != list(self.BEAR_TREND_WINDOWS)
            or parameters.get("bear_rule") != self.BEAR_RULE
        ):
            raise ValueError(f"{self.ID} 的空头腿登记与实现不一致")
        return config

    def targets(self, data, columns) -> pd.DataFrame:
        return targets(data.factors.signed, data.factors.bars, data.universe, data.returns.index, columns)


BOOK = Book()
BOOK.load_config()
