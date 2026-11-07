"""股指 55/20 突破袖套的共用信号和目标仓位。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.index_common import SYMBOLS, close_vol_scale
from tfcta import config as C

ENTRY_LOOKBACK = 55
EXIT_LOOKBACK = 20
TREND_BUDGET = 0.5


def breakout_states(
    bars: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    *,
    entry_allowed: pd.Series | None = None,
) -> pd.DataFrame:
    """生成收盘确认的逐品种多头状态；通道只使用此前交易日。"""
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("突破信号日期必须是非空、唯一、升序的 DatetimeIndex")
    if index.min() < pd.Timestamp("2015-04-16") or index.max() > pd.Timestamp(C.DEFAULT_OOS_END):
        raise C.HoldoutViolation("突破趋势策略日期须在2015-04-16至2025年范围内")
    if entry_allowed is not None and (
        not entry_allowed.index.equals(index)
        or entry_allowed.isna().any()
        or not pd.api.types.is_bool_dtype(entry_allowed.dtype)
    ):
        raise ValueError("突破开仓过滤器必须与信号日期完全对齐且只含布尔值")

    frames = {
        field: bars[field].reindex(index=index, columns=list(SYMBOLS))
        for field in ("closew", "highw", "loww")
    }
    if any(not np.isfinite(frame.to_numpy(dtype=float)).all() for frame in frames.values()):
        raise ValueError("突破趋势策略行情未完全对齐")
    if (frames["highw"] <= frames["loww"]).any().any():
        raise ValueError("突破趋势策略日内高低价区间无效")

    entry_channel = frames["highw"].shift(1).rolling(
        ENTRY_LOOKBACK, min_periods=ENTRY_LOOKBACK
    ).max()
    exit_channel = frames["loww"].shift(1).rolling(
        EXIT_LOOKBACK, min_periods=EXIT_LOOKBACK
    ).min()
    states = pd.DataFrame(False, index=index, columns=SYMBOLS)
    for symbol in SYMBOLS:
        active = False
        for date in index:
            entry = entry_channel.at[date, symbol]
            exit_ = exit_channel.at[date, symbol]
            close = frames["closew"].at[date, symbol]
            if not np.isfinite(entry) or not np.isfinite(exit_) or not np.isfinite(close):
                active = False
            elif active:
                if close < exit_:
                    active = False
            elif close > entry and (
                entry_allowed is None or bool(entry_allowed.at[date])
            ):
                active = True
            states.at[date, symbol] = active
    return states


def trend_targets(
    states: pd.DataFrame,
    scale: pd.Series,
) -> pd.DataFrame:
    """将收盘状态滞后一交易日执行，并限制趋势袖套目标预算为 0.5。"""
    if tuple(states.columns) != SYMBOLS or not states.index.equals(scale.index):
        raise ValueError("趋势状态、波动缩放和品种列必须完全对齐")
    if states.empty or states.index.has_duplicates or not states.index.is_monotonic_increasing:
        raise ValueError("趋势状态日期必须非空、唯一、升序")
    if scale.isna().any() or not np.isfinite(scale.to_numpy(dtype=float)).all() or scale.lt(0).any():
        raise ValueError("趋势波动缩放必须为非负有限值")
    active = states.astype(float)
    count = active.sum(axis=1).replace(0.0, np.nan)
    weights = active.div(count, axis=0).fillna(0.0) * TREND_BUDGET
    targets = weights.mul(scale, axis=0).shift(1).fillna(0.0)
    if targets.abs().sum(axis=1).gt(TREND_BUDGET + 1e-12).any():
        raise ArithmeticError("趋势袖套目标预算超过固定上限")
    return targets


def trend_targets_from_bars(
    bars: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    *,
    entry_allowed: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从行情生成状态和使用历史波动缩放后的趋势目标。"""
    open_to_open = (
        (bars["openw"].shift(-1) - bars["openw"]) / bars["open"]
    ).mean(axis=1, skipna=False)
    scale = close_vol_scale(open_to_open).fillna(1.0).reindex(index)
    if scale.isna().any():
        raise ValueError("突破趋势策略波动缩放覆盖不完整")
    states = breakout_states(bars, index, entry_allowed=entry_allowed)
    return states, trend_targets(states, scale)
