"""Exploratory trend/carry agreement timed by the full-trading-day clock."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.book import HOLDING_DAYS, LOOKBACK
from tfcta import config as C

ID = "trend_carry_full_clock"
NAME = "趋势展期一致全日时钟"
DETAIL = "10日趋势与滞后carry同向/全交易日时钟同向尾部/ATR风险缩放0.5-2/持有5日"
CONFIG_PATH = C.CONFIG_DIR / "research" / "trend_carry_full_clock.json"
TAIL_QUANTILE = 0.20
RISK_SCALE_MIN = 0.5
RISK_SCALE_MAX = 2.0
MIN_SYMBOLS = 5
RISK_SCALE_RULE = "cross-sectional pool median ATR% / own ATR%, clipped to [0.5, 2.0]"


def load_config(path=CONFIG_PATH) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = config.get("rules", {})
    if (
        config.get("id") != ID
        or config.get("status") != "exploratory_post_hoc_candidate"
        or "不读取2022及2023年以后" not in config.get("sample", "")
        or rules.get("clock_lookback_days") != LOOKBACK
        or rules.get("clock_tail_quantile") != TAIL_QUANTILE
        or rules.get("clock_threshold_comparison") != "strict"
        or rules.get("clock_scope") != "full_trading_day"
        or rules.get("risk_scale") != RISK_SCALE_RULE
        or rules.get("holding_days") != HOLDING_DAYS
    ):
        raise ValueError("全交易日时钟组合配置与实现或样本边界不一致")
    return config


def symmetric_tail(
    series: pd.Series,
    lookback: int = LOOKBACK,
    quantile: float = TAIL_QUANTILE,
) -> pd.Series:
    """Strict two-sided tails against prior observations only."""
    if lookback < 1 or not 0 < quantile < 0.5:
        raise ValueError("lookback 必须为正数，quantile 必须在 (0, 0.5) 内")
    values = series.to_numpy(dtype="float64")
    result = np.zeros(values.size, dtype="int8")
    finite = np.flatnonzero(np.isfinite(values))
    for nth, loc in enumerate(finite):
        if nth < lookback:
            continue
        history = values[finite[nth - lookback:nth]]
        lower, upper = np.quantile(history, [quantile, 1.0 - quantile], method="linear")
        if values[loc] > upper:
            result[loc] = 1
        elif values[loc] < lower:
            result[loc] = -1
    return pd.Series(result, index=series.index, name=series.name)


def signal(
    trend: pd.DataFrame,
    carry: pd.DataFrame,
    clock: pd.DataFrame,
) -> pd.DataFrame:
    """趋势与滞后 carry 同号，并且全交易日时钟落在同方向尾部时才交易。"""
    index, columns = clock.index, clock.columns
    trend = trend.reindex(index=index, columns=columns)
    carry = carry.reindex(index=index, columns=columns)
    tails = pd.DataFrame(
        {symbol: symmetric_tail(clock[symbol]) for symbol in columns},
        index=index,
    )
    trend_side = np.sign(trend.to_numpy(dtype="float64"))
    carry_side = np.sign(carry.to_numpy(dtype="float64"))
    clock_side = tails.to_numpy(dtype="int8")
    confirmed = (trend_side != 0) & (trend_side == carry_side)
    direction = np.where(confirmed & (trend_side == clock_side), trend_side, 0)
    return pd.DataFrame(direction.astype("int8"), index=index, columns=columns)


def risk_scale(
    atr_pct: pd.DataFrame,
    positions: pd.DataFrame,
    universe: dict[int, list[str]],
) -> pd.DataFrame:
    """Scale live signals by pool-median ATR% / own ATR%, clipped to [0.5, 2]."""
    atr_pct = atr_pct.reindex(index=positions.index, columns=positions.columns)
    result = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)
    years = pd.DatetimeIndex(positions.index).year
    for year in sorted(set(years)):
        columns = sorted(set(universe.get(int(year), ())).intersection(positions.columns))
        if len(columns) < MIN_SYMBOLS:
            continue
        dates = positions.index[years == year]
        block = atr_pct.loc[dates, columns]
        benchmark = block.median(axis=1)
        scale = block.rdiv(benchmark, axis="index").where(block > 0).clip(
            lower=RISK_SCALE_MIN,
            upper=RISK_SCALE_MAX,
        )
        live = positions.loc[dates, columns].ne(0)
        result.loc[dates, columns] = scale.where(live, 0.0).fillna(0.0)
    return result
