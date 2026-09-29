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
    rbreaker_enter: float = 0.07
    opening_range_minutes: int = 30
    # first：交易日第一根（夜盘品种从夜盘算起）；day：日盘第一根（约 09:00）
    open_anchor: str = "first"
    # 每日最多持仓段数（止损/止盈后再入场、反手开出的新仓都各算一段）
    max_entries_per_day: int = 1
    stop_atr_multiple: float | None = None
    target_atr_multiple: float | None = None
    # 固定比例止损/止盈（占当日首根 bar 原始开盘价），与 ATR 止损二选一
    stop_pct: float | None = None
    target_pct: float | None = None
    # 突破反向轨道时平仓并反手（Dual Thrust 原文逻辑）；R-Breaker 的反手由自身规则决定
    reverse: bool = False
    # 空中花园：开盘相对昨收的最小跳空幅度，以及上下轨取开盘后前几根 bar
    gap_pct: float = 0.01
    sky_bars: int = 1


def available_strategies() -> tuple[str, ...]:
    return ("dual_thrust", "atr", "rbreaker", "rbreaker_breakout", "fali", "sky_garden",
            "opening_range_assumption")


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
    """六个价位（原文系数 0.35 / 0.07 / 0.25）。

    ``long_level``/``short_level`` 为突破买入/卖出价；``sell_setup``/``buy_setup`` 为
    观察卖出/买入价；``sell_enter``/``buy_enter`` 为反转卖出/买入价。
    """
    high, low, close = daily["prev_high"], daily["prev_low"], daily["prev_close"]
    sell_setup = high + cfg.rbreaker_setup * (close - low)
    buy_setup = low - cfg.rbreaker_setup * (high - close)
    f = cfg.rbreaker_enter
    sell_enter = (1 + f) / 2 * (high + low) - f * low
    buy_enter = (1 + f) / 2 * (high + low) - f * high
    width = sell_setup - buy_setup
    return pd.DataFrame({
        "long_level": sell_setup + cfg.rbreaker_break * width,
        "short_level": buy_setup - cfg.rbreaker_break * width,
        "sell_setup": sell_setup, "buy_setup": buy_setup,
        "sell_enter": sell_enter, "buy_enter": buy_enter,
        "atr": daily["atr"],
    }, index=daily.index)


def _fali_levels(daily: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    return pd.DataFrame({
        "long_level": daily["prev_high"],
        "short_level": daily["prev_low"],
        "atr": daily["atr"],
    }, index=daily.index)


def prepared_daily_levels(minute: pd.DataFrame,
                          cfg: StrategyConfig) -> pd.DataFrame:
    """一次性计算所有交易日的盘前水平线，避免重复扫描历史数据。"""
    daily = _daily_context(minute, cfg.atr_window)
    if cfg.name == "dual_thrust":
        return _range_levels(daily, cfg)
    if cfg.name == "atr":
        return _atr_levels(daily, cfg)
    if cfg.name in ("rbreaker", "rbreaker_breakout"):
        return _rbreaker_levels(daily, cfg)
    if cfg.name == "fali":
        return _fali_levels(daily, cfg)
    if cfg.name in ("opening_range_assumption", "sky_garden"):
        # 上下轨由开盘后的分钟 bar 给出；gap_points 为开盘减昨收的复权价差，
        # 引擎再除以原始价格的昨收（= 当日原始开盘 − gap_points）得到跳空幅度
        return daily[["atr"]].assign(long_level=np.nan, short_level=np.nan,
                                     gap_points=daily["open"] - daily["prev_close"])
    raise ValueError(f"未知日内策略: {cfg.name}; 可选 {available_strategies()}")

