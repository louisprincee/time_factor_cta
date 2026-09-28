"""经典日内突破策略的盘前参数计算。

策略只使用当前交易日前已经完成的日线，开盘区间策略除外。实际成交由
``engine.run_intraday`` 统一处理，避免各策略各自定义成交和成本口径。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    lookback: int = 4
    k1: float = 0.5
    k2: float = 0.5
    atr_window: int = 20
    atr_multiple: float = 1.0
    rbreaker_setup: float = 0.35
    rbreaker_break: float = 0.25
    rbreaker_reverse: float = 0.50
    opening_range_minutes: int = 30
    max_entries_per_day: int = 1
    stop_atr_multiple: float | None = None
    target_atr_multiple: float | None = None


def available_strategies() -> tuple[str, ...]:
    return ("dual_thrust", "atr", "rbreaker", "fali", "opening_range_assumption")


def _daily_context(df: pd.DataFrame, atr_window: int) -> pd.DataFrame:
    grouped = df.groupby("trading_date", sort=True)
    daily = grouped.agg(high=("high", "max"), low=("low", "min"),
                        close=("close", "last"), open=("open", "first"))
    previous_close = daily["close"].shift(1)
    true_range = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - previous_close).abs(),
        (daily["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    daily["atr"] = true_range.shift(1).rolling(
        int(atr_window), min_periods=int(atr_window)).mean()
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily["prev_close"] = daily["close"].shift(1)
    return daily


def _range_levels(daily: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    high = daily["high"].shift(1).rolling(cfg.lookback, min_periods=cfg.lookback).max()
    low = daily["low"].shift(1).rolling(cfg.lookback, min_periods=cfg.lookback).min()
    close_high = daily["close"].shift(1).rolling(
        cfg.lookback, min_periods=cfg.lookback).max()
    close_low = daily["close"].shift(1).rolling(
        cfg.lookback, min_periods=cfg.lookback).min()
    price_range = pd.concat([high - close_low, close_high - low], axis=1).max(axis=1)
    return pd.DataFrame({
        "long_level": daily["open"] + cfg.k1 * price_range,
        "short_level": daily["open"] - cfg.k2 * price_range,
        "atr": daily["atr"],
    }, index=daily.index)


def _atr_levels(daily: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    return pd.DataFrame({
        "long_level": daily["open"] + cfg.atr_multiple * daily["atr"],
        "short_level": daily["open"] - cfg.atr_multiple * daily["atr"],
        "atr": daily["atr"],
    }, index=daily.index)


def _rbreaker_levels(daily: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    high, low, close = daily["prev_high"], daily["prev_low"], daily["prev_close"]
    setup_buy = high + cfg.rbreaker_setup * (close - low)
    setup_sell = low - cfg.rbreaker_setup * (high - close)
    width = setup_buy - setup_sell
    return pd.DataFrame({
        "long_level": setup_buy + cfg.rbreaker_break * width,
        "short_level": setup_sell - cfg.rbreaker_break * width,
        "atr": daily["atr"],
    }, index=daily.index)


def _fali_levels(daily: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    return pd.DataFrame({
        "long_level": daily["prev_high"],
        "short_level": daily["prev_low"],
        "atr": daily["atr"],
    }, index=daily.index)


def _opening_range_levels(day: pd.DataFrame, cfg: StrategyConfig) -> tuple[float, float]:
    first = day.iloc[: int(cfg.opening_range_minutes)]
    if len(first) < int(cfg.opening_range_minutes):
        return np.nan, np.nan
    return float(first["high"].max()), float(first["low"].min())


def prepared_daily_levels(minute: pd.DataFrame,
                          cfg: StrategyConfig) -> pd.DataFrame:
    """一次性计算所有交易日的盘前水平线，避免重复扫描历史数据。"""
    daily = _daily_context(minute, cfg.atr_window)
    if cfg.name == "dual_thrust":
        return _range_levels(daily, cfg)
    if cfg.name == "atr":
        return _atr_levels(daily, cfg)
    if cfg.name == "rbreaker":
        return _rbreaker_levels(daily, cfg)
    if cfg.name == "fali":
        return _fali_levels(daily, cfg)
    if cfg.name == "opening_range_assumption":
        return daily[["atr"]].assign(long_level=np.nan, short_level=np.nan)
    raise ValueError(f"未知日内策略: {cfg.name}; 可选 {available_strategies()}")


def levels_for_day(history: pd.DataFrame, day: pd.DataFrame,
                   cfg: StrategyConfig) -> tuple[float, float, float]:
    """返回 (多头突破价, 空头突破价, ATR)。

    ``history`` 只包含当前日以前的分钟数据；开盘区间策略在当前日的前 N 根
    bar 完成后才产生水平线，调用方必须从第 N 根之后开始寻找突破。
    """
    if cfg.name == "opening_range_assumption":
        long_level, short_level = _opening_range_levels(day, cfg)
        atr = float(history.groupby("trading_date").agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last")
        ).pipe(lambda x: pd.concat([
            x["high"] - x["low"],
            (x["high"] - x["close"].shift(1)).abs(),
            (x["low"] - x["close"].shift(1)).abs(),
        ], axis=1).max(axis=1).shift(1).rolling(
            cfg.atr_window, min_periods=cfg.atr_window).mean().iloc[-1]))
        return long_level, short_level, atr

    daily = _daily_context(pd.concat([history, day]), cfg.atr_window)
    if cfg.name == "dual_thrust":
        row = _range_levels(daily, cfg).iloc[-1]
    elif cfg.name == "atr":
        row = _atr_levels(daily, cfg).iloc[-1]
    elif cfg.name == "rbreaker":
        row = _rbreaker_levels(daily, cfg).iloc[-1]
    elif cfg.name == "fali":
        row = _fali_levels(daily, cfg).iloc[-1]
    else:
        raise ValueError(f"未知日内策略: {cfg.name}; 可选 {available_strategies()}")
    return float(row["long_level"]), float(row["short_level"]), float(row["atr"])