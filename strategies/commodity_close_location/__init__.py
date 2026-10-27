"""商品收盘位置策略及其未过滤探索版本。

多头：池内等权收盘位置 < 0.5 时池内每个品种 +1，与股指候选同一规则，不用五个时间因子，也不看展期。
空头：与冻结的 market_clock_slow_short_20_60_120 完全相同，直接复用它的函数；同一品种同一天多空冲突时以空头为准。
只能跑研究期。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from strategies import commodity_original as S
from tfcta import config as C

LONG_THRESHOLD = 0.5
LONG_RULE = "market_clv < 0.5"
CONFLICT_RULE = "short_overrides_long"
STRICT_SWAP_ID = "commodity_close_location_reversal_strict_swap"
STRICT_SWAP_CONFIG_PATH = Path(__file__).with_name(f"{STRICT_SWAP_ID}.json")


def market_close_location(bars: dict, universe: dict, index: pd.DatetimeIndex, columns) -> pd.Series:
    """当年池内品种收盘位置的等权平均，收盘 t 已知；复权列只取差。池内有效品种不足 MIN_SYMBOLS 时缺失。"""
    close, high, low = (bars[k].reindex(index=index, columns=columns) for k in ("closew", "highw", "loww"))
    width = high - low
    clv = ((close - low) / width.where(width > 0)).where(S.pool_mask(universe, index, columns))
    enough = clv.notna().sum(axis=1) >= S.MIN_SYMBOLS
    return clv.mean(axis=1).where(enough)


def decision_weights(market_clv: pd.Series, carry: pd.DataFrame, trend: pd.DataFrame, bear: pd.Series,
                     long_carry: bool = False) -> pd.DataFrame:
    """收盘决策。carry 已滞后一天。long_carry 只给研究对照：多头另要求展期 > 0（此时不会和空头冲突）。"""
    longs = market_clv.lt(LONG_THRESHOLD)
    long_side = pd.DataFrame(1.0, index=carry.index, columns=carry.columns)
    if long_carry:
        long_side = long_side.where(carry.gt(0), 0.0)
    long_side = long_side.mul(longs.reindex(carry.index).eq(True).astype(float), axis=0)
    bear_days = pd.DataFrame({c: bear.reindex(carry.index).eq(True) for c in carry.columns}, index=carry.index)
    short = trend.reindex_like(carry).lt(0) & carry.lt(0) & bear_days
    return long_side.where(~short, -1.0)


def targets(bars: dict, universe: dict, index: pd.DatetimeIndex, columns,
            carry: pd.DataFrame | None = None, long_only: bool = False, long_carry: bool = False) -> pd.DataFrame:
    """收盘决策后移到下一交易日第一根开盘。long_only、long_carry 只给研究对照用。"""
    columns = pd.Index(columns)
    pool = S.pool_mask(universe, index, columns)
    if carry is None:
        carry = S.roll_yield(index, columns)
        S.require_carry_coverage(carry, pool)
    lagged = carry.reindex(index=index, columns=columns).shift(1)
    clv = market_close_location(bars, universe, index, columns)
    trend = S.symbol_trend(bars, columns).reindex(index)
    daily = S.close_return(bars, columns).reindex(index).where(pool).mean(axis=1)
    bear = S.market_trend(daily).lt(0)
    if long_only:
        bear = pd.Series(False, index=index)
    return decision_weights(clv, lagged, trend, bear, long_carry).shift(1).fillna(0.0).where(pool, 0.0)


def check_config(config: dict, strategy_id: str) -> dict:
    parameters = config.get("parameters") or {}
    if (
        config.get("status") != "exploratory_predeclared"
        or config.get("frozen_at") or config.get("chosen")
        or config.get("id") != strategy_id
        or "不读取2022及2023年以后" not in config.get("sample", "")
        or "不读 2026" not in config.get("sample", "")
        or parameters.get("long_threshold") != LONG_THRESHOLD
        or parameters.get("long_rule") != LONG_RULE
        or parameters.get("conflict_rule") != CONFLICT_RULE
        or parameters.get("roll_yield") != "annualized_yield"
        or parameters.get("roll_yield_lag_days") != 1
        or parameters.get("gross_exposure_limit") != S.GROSS
        or parameters.get("gross_split") != "pool"
        or parameters.get("rebalance_on_count_change") is not False
        or parameters.get("drawdown_breaker") != S.DRAWDOWN_LIMIT
        or parameters.get("halt_days") != S.HALT_DAYS
        or parameters.get("short_trend_windows") != list(S.TREND_WINDOWS)
        or parameters.get("short_rule") != S.SHORT_RULE
        or parameters.get("short_weight") != -1.0
        or parameters.get("bear_trend_windows") != list(S.TREND_WINDOWS)
        or parameters.get("bear_rule") != S.BEAR_RULE
    ):
        raise ValueError(f"{strategy_id} 的登记与实现不一致")
    return config


class Book:
    ID = "commodity_close_location_reversal"
    NAME = "商品收盘位置反转探索策略"
    GROSS = S.GROSS
    CONFIG_PATH = Path(__file__).with_name("commodity_close_location_reversal.json")
    DETAIL = (
        "池内等权收盘位置小于0.5时池内每个品种做多1份；池子20/60/120日累计收益符号均值小于0时，"
        "品种自身20/60/120日收益符号均值小于0且滞后一日年化展期收益小于0的做空1份，多空冲突以空头为准"
        "/两腿都按当年池宽分1倍名义敞口/下一交易日第一根开盘成交/权重不变则留仓/回撤10%清仓并停10个交易日"
    )

    @property
    def FILES(self) -> list[Path]:
        return [Path(S.__file__), Path(__file__), self.CONFIG_PATH]

    def load_config(self) -> dict:
        return check_config(json.loads(self.CONFIG_PATH.read_text(encoding="utf-8")), self.ID)

    def frozen(self) -> dict:
        self.load_config()
        raise C.HoldoutViolation(f"{self.NAME}（{self.ID}）是探索期策略，没冻结，不能打开 2022 及以后")

    def targets(self, data, columns) -> pd.DataFrame:
        return targets(data.factors.bars, data.universe, data.returns.index, columns)


class StrictSwapBook:
    ID = STRICT_SWAP_ID
    NAME = "商品收盘位置策略"
    GROSS = S.GROSS
    CONFIG_PATH = STRICT_SWAP_CONFIG_PATH
    DETAIL = (
        "将原策略多头市场分数替换为池内等权收盘位置小于0.5，保留滞后一日展期收益大于0的多头筛选；"
        "空头腿、预算、执行和风控均与冻结原策略一致"
    )

    @property
    def FILES(self) -> list[Path]:
        return [Path(S.__file__), Path(__file__), self.CONFIG_PATH]

    def load_config(self) -> dict:
        config = json.loads(self.CONFIG_PATH.read_text(encoding="utf-8"))
        parameters = config.get("parameters") or {}
        if (
            config.get("id") != self.ID
            or config.get("status") != "frozen"
            or not config.get("frozen_at")
            or config.get("chosen") != parameters
            or "do not read 2026" not in config.get("sample", "").lower()
            or parameters.get("long_signal")
            != "Equal-weight annual-pool close location (closew - loww) / (highw - loww) < 0.5"
            or parameters.get("long_carry_filter") != "Prior trading day's annualized_yield > 0"
            or parameters.get("roll_yield") != "annualized_yield"
            or parameters.get("roll_yield_lag_days") != 1
            or parameters.get("gross_split") != "pool"
            or parameters.get("rebalance_on_count_change") is not False
            or parameters.get("short_trend_windows") != list(S.TREND_WINDOWS)
            or parameters.get("short_rule") != S.SHORT_RULE
            or parameters.get("short_weight") != -1.0
            or parameters.get("bear_trend_windows") != list(S.TREND_WINDOWS)
            or parameters.get("bear_rule") != S.BEAR_RULE
            or parameters.get("conflict_rule") != CONFLICT_RULE
            or parameters.get("gross_exposure_limit") != S.GROSS
            or parameters.get("drawdown_breaker") != S.DRAWDOWN_LIMIT
            or parameters.get("halt_days") != S.HALT_DAYS
        ):
            raise ValueError(f"{self.ID} 的冻结规格与实现不一致")
        return config

    def frozen(self) -> dict:
        return self.load_config()

    def targets(self, data, columns) -> pd.DataFrame:
        return targets(
            data.factors.bars,
            data.universe,
            data.returns.index,
            columns,
            long_carry=True,
        )


BOOK = Book()
BOOK.load_config()
STRICT_SWAP_BOOK = StrictSwapBook()
STRICT_SWAP_BOOK.load_config()
