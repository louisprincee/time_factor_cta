"""低点减高点的尾部分位策略。参数在 strategies/ts_low.json。

回测不在这里读行情。引擎调用 build_positions，再计算三个时期。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "strategies" / "ts_low.json"
EVAL_START = pd.Timestamp("2016-01-01")
EVAL_END = pd.Timestamp("2021-12-31")

LOOKBACKS = (
    {"days": 60, "label": "一季"},
    {"days": 250, "label": "一年"},
    {"days": 750, "label": "三年"},
)
HORIZONS = (
    {"days": 1, "label": "隔夜"},
    {"days": 5, "label": "一周"},
    {"days": 20, "label": "一月"},
)
TAILS = (
    {"quantile": 0.9, "label": "前10%"},
    {"quantile": 0.85, "label": "前15%"},
    {"quantile": 0.8, "label": "前20%"},
)


def load_config(path=CONFIG_PATH, *, allow_frozen=False):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not allow_frozen and (config.get("frozen_at") or config.get("chosen") is not None):
        raise ValueError("当前入口不接受冻结配置")
    sample = config.get("sample", "")
    if not allow_frozen and "不读 2023" not in sample:
        raise ValueError("样本口径必须写明不读 2023 年以后")
    chosen = config.get("parameters") or {}
    if (
        chosen.get("factor") != "ts_low_minus_high"
        or chosen.get("direction") != "upper_tail_long"
    ):
        raise ValueError("当前策略只允许高尾做多")
    if chosen.get("comparison") != "strict":
        raise ValueError("分位比较必须是严格高于")
    if chosen.get("lookbacks") != [item["days"] for item in LOOKBACKS]:
        raise ValueError("回看只能是 60、250、750 个交易日")
    if chosen.get("horizons") != [item["days"] for item in HORIZONS]:
        raise ValueError("持仓只能是 1、5、20 个交易日")
    if chosen.get("quantiles") != [item["quantile"] for item in TAILS]:
        raise ValueError("阈值只能是前 10%、20%、30%")
    return config


def chosen_parameters(config):
    """从已冻结配置中解析唯一的样本外参数组合。"""
    chosen = config.get("chosen")
    required = {"lookback", "quantile", "horizon"}
    if not isinstance(chosen, dict) or set(chosen) != required:
        raise ValueError("样本外 chosen 必须且只能包含 lookback、quantile、horizon")

    lookback = chosen["lookback"]
    quantile = chosen["quantile"]
    horizon = chosen["horizon"]
    if lookback not in config["parameters"]["lookbacks"]:
        raise ValueError(f"chosen lookback 不在允许的参数范围内: {lookback}")
    if quantile not in config["parameters"]["quantiles"]:
        raise ValueError(f"chosen quantile 不在允许的参数范围内: {quantile}")
    if horizon not in config["parameters"]["horizons"]:
        raise ValueError(f"chosen horizon 不在允许的参数范围内: {horizon}")

    for spec in parameters(config):
        if (
            spec["回看交易日"] == lookback
            and spec["分位"] == quantile
            and spec["持仓交易日"] == horizon
        ):
            return spec
    raise ValueError("chosen 没有对应的已登记参数组合")




def parameters(config):
    labels = {
        "lookback": {item["days"]: item["label"] for item in LOOKBACKS},
        "horizon": {item["days"]: item["label"] for item in HORIZONS},
    }
    chosen = config["parameters"]
    for lookback in chosen["lookbacks"]:
        for quantile in chosen["quantiles"]:
            for horizon in chosen["horizons"]:
                yield {
                    "回看": labels["lookback"][lookback],
                    "回看交易日": int(lookback),
                    "阈值": f"前{1 - quantile:.0%}",
                    "分位": float(quantile),
                    "持仓": labels["horizon"][horizon],
                    "持仓交易日": int(horizon),
                }


def upper_tail(series: pd.Series, lookback: int, quantile: float) -> pd.Series:
    """过去 lookback 个有值的交易日，不含当天。只在严格高于分位时做多。"""
    values = series.to_numpy(dtype="float64")
    sides = np.zeros(values.size, dtype="float64")
    finite_at = np.flatnonzero(np.isfinite(values))
    for nth, loc in enumerate(finite_at):
        if nth < lookback:
            continue
        hist = values[finite_at[nth - lookback:nth]]
        hi = float(np.quantile(hist, quantile, method="linear"))
        if float(values[loc]) > hi:
            sides[loc] = 1.0
    return pd.Series(sides, index=series.index)




def build_entry_schedule(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    universe: dict,
    horizon: int,
    lookback: int,
    quantile: float,
    start=EVAL_START,
    end=EVAL_END,
    entry_filter: pd.DataFrame | None = None,
    require_exit_day: bool = False,
) -> pd.DataFrame:
    """收盘信号对应的下一交易日开仓事件，显式保留相邻持仓的独立入场。"""
    if horizon < 1:
        raise ValueError("持仓期必须至少为 1 个交易日")
    if entry_filter is not None and not entry_filter.index.is_unique:
        raise ValueError("入场过滤器的日期不能重复")

    entries = pd.DataFrame(False, index=returns.index, columns=returns.columns)
    pools = {int(year): set(names) for year, names in universe.items()}
    for symbol in returns.columns:
        if symbol not in factor.columns:
            continue
        signal = upper_tail(factor[symbol], lookback, quantile)
        tradable = returns.index[np.isfinite(returns[symbol].to_numpy(dtype="float64"))]
        if len(tradable) == 0:
            continue
        order = {pd.Timestamp(day): i for i, day in enumerate(tradable)}
        blocked = 0
        for day, side in signal.items():
            if side != 1.0:
                continue
            day = pd.Timestamp(day).normalize()
            if day < start or day > end or symbol not in pools.get(int(day.year), ()):
                continue
            if entry_filter is not None:
                if day not in entry_filter.index or symbol not in entry_filter.columns:
                    continue
                eligible = entry_filter.at[day, symbol]
                if pd.isna(eligible) or not bool(eligible):
                    continue
            later = tradable[tradable > day]
            if len(later) == 0:
                continue
            entry_day = pd.Timestamp(later[0])
            if symbol not in pools.get(int(entry_day.year), ()):
                continue
            entry_i = order[entry_day]
            if (
                entry_i < blocked
                or entry_i + horizon > len(tradable)
                or (require_exit_day and entry_i + horizon >= len(tradable))
            ):
                continue
            entries.loc[tradable[entry_i], symbol] = True
            blocked = entry_i + horizon
    return entries


def build_positions(factor: pd.DataFrame, returns: pd.DataFrame, universe: dict,
                    horizon: int, lookback: int, quantile: float,
                    start=EVAL_START, end=EVAL_END,
                    entry_filter: pd.DataFrame | None = None) -> pd.DataFrame:
    """收盘信号，下一交易日开盘持有。未到期的品种不再开新仓。"""
    entries = build_entry_schedule(
        factor, returns, universe, horizon, lookback, quantile,
        start, end, entry_filter,
    )
    position = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for symbol in entries.columns:
        tradable = returns.index[
            np.isfinite(returns[symbol].to_numpy(dtype="float64"))
        ]
        order = {pd.Timestamp(day): i for i, day in enumerate(tradable)}
        for day in entries.index[entries[symbol]]:
            entry_i = order.get(pd.Timestamp(day))
            if entry_i is not None:
                position.loc[tradable[entry_i:entry_i + horizon], symbol] = 1.0
    return position
