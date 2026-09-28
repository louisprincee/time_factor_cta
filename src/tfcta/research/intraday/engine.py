"""1 分钟日内策略撮合与绩效计算。

成交规则固定为保守口径：突破在当前 bar 内触发时按突破价成交，若开盘已越过
突破价则按开盘价成交；止盈和止损同一根 bar 同时触发时先按止损成交；每个交易日
收盘强制平仓，不持仓过夜。该模块不读取或修改现有日频研究产物。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategies import StrategyConfig, _opening_range_levels, prepared_daily_levels


@dataclass(frozen=True)
class BacktestConfig:
    fee_rate: float = 0.00025
    slippage_points: float = 0.0
    margin_rate: float = 0.30
    initial_capital: float = 1.0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily: pd.DataFrame
    metrics: dict[str, float]


def combine_daily_returns(books: dict[str, pd.DataFrame],
                          weights: dict[str, float] | None = None) -> BacktestResult:
    """合成已完成单品种回测的日收益，权重按可用书归一化。"""
    if not books:
        raise ValueError("至少需要一本日内策略书")
    names = list(books)
    weight = {name: 1.0 / len(names) for name in names}
    if weights is not None:
        if set(weights) != set(names):
            raise ValueError("weights 必须覆盖且只覆盖所有组合成员")
        total = sum(float(v) for v in weights.values())
        if total <= 0:
            raise ValueError("组合权重之和必须为正")
        weight = {name: float(value) / total for name, value in weights.items()}
    panel = pd.concat({name: frame["net_return"] for name, frame in books.items()}, axis=1)
    weighted = panel.mul(pd.Series(weight)).sum(axis=1, min_count=1)
    equity = (1.0 + weighted).cumprod()
    daily = pd.DataFrame({"net_return": weighted, "equity": equity})
    ret = weighted.dropna()
    std = ret.std(ddof=1) if len(ret) > 1 else np.nan
    metrics = {
        "ann_return": float((1.0 + ret).prod() ** (252.0 / len(ret)) - 1.0)
        if len(ret) and (1.0 + ret).prod() > 0 else np.nan,
        "ann_vol": float(std * np.sqrt(252.0)) if np.isfinite(std) else np.nan,
        "sharpe": float(ret.mean() / std * np.sqrt(252.0))
        if np.isfinite(std) and std > 0 else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min())
        if len(equity) else np.nan,
        "n_days": float(len(daily)),
        "n_trades": float(sum(len(frame) for frame in books.values())),
    }
    return BacktestResult(pd.DataFrame(), daily, metrics)


def _fill_entry(row: pd.Series, long_level: float, short_level: float):
    if np.isfinite(long_level) and row.high >= long_level:
        return (1, float(row.open if row.open >= long_level else long_level))
    if np.isfinite(short_level) and row.low <= short_level:
        return (-1, float(row.open if row.open <= short_level else short_level))
    return None


def _exit_bar(row: pd.Series, direction: int, stop: float, target: float):
    if direction > 0:
        if np.isfinite(stop) and row.low <= stop:
            return float(row.open if row.open <= stop else stop), "stop_loss"
        if np.isfinite(target) and row.high >= target:
            return float(row.open if row.open >= target else target), "take_profit"
    else:
        if np.isfinite(stop) and row.high >= stop:
            return float(row.open if row.open >= stop else stop), "stop_loss"
        if np.isfinite(target) and row.low <= target:
            return float(row.open if row.open <= target else target), "take_profit"
    return None


def _trade_record(symbol: str, direction: int, entry_time, exit_time,
                  entry: float, exit: float, entry_reference: float,
                  reason: str,
                  cfg: BacktestConfig) -> dict:
    entry_exec = entry + direction * cfg.slippage_points
    exit_exec = exit - direction * cfg.slippage_points
    denominator = entry_reference if np.isfinite(entry_reference) and entry_reference > 0 else entry_exec
    gross = direction * (exit_exec - entry_exec) / denominator
    cost = 2.0 * cfg.fee_rate
    return {
        "symbol": symbol, "direction": direction, "entry_time": entry_time,
        "exit_time": exit_time, "entry_price": entry_exec, "exit_price": exit_exec,
        "exit_reason": reason, "gross_return": gross,
        "net_return": gross - cost,
    }


def run_intraday(minute: pd.DataFrame, strategy: StrategyConfig,
                 config: BacktestConfig = BacktestConfig(),
                 symbol: str = "") -> BacktestResult:
    df = minute.copy()
    raw_open = pd.to_numeric(df["open"], errors="coerce") if "open" in df else None
    aliases = {"open": "openw", "high": "highw", "low": "loww", "close": "closew"}
    for raw, adjusted in aliases.items():
        if adjusted in df.columns:
            df[raw] = df[adjusted]
    required = {"open", "high", "low", "close", "trading_date"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"分钟数据缺少字段: {missing}")
    df.index = pd.to_datetime(df.index)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.normalize()
    df = df.sort_index(kind="mergesort")
    if df.index.has_duplicates:
        raise ValueError("分钟数据存在重复时间戳")

    trades: list[dict] = []
    daily_rows: list[dict] = []
    day_groups = list(df.groupby("trading_date", sort=True))
    level_table = prepared_daily_levels(df, strategy)
    for date, day in day_groups:
        if strategy.name == "opening_range_assumption":
            start = int(strategy.opening_range_minutes)
            long_level, short_level = _opening_range_levels(day, strategy)
        else:
            start = 0
            level = level_table.loc[date]
            long_level, short_level = float(level["long_level"]), float(level["short_level"])
        atr = float(level_table.loc[date, "atr"])
        position = 0
        entry_price = np.nan
        entry_reference = np.nan
        entry_time = None
        stop = target = np.nan
        entries = 0
        day_net = 0.0
        for bar_no, (timestamp, row) in enumerate(day.iterrows()):
            if (position == 0 and bar_no >= start
                    and entries < int(strategy.max_entries_per_day)):
                fill = _fill_entry(row, long_level, short_level)
                if fill is not None:
                    position, entry_price = fill
                    entries += 1
                    entry_time = timestamp
                    entry_reference = (float(raw_open.loc[timestamp])
                                       if raw_open is not None and np.isfinite(raw_open.loc[timestamp])
                                       else entry_price)
                    stop = target = np.nan
                    if np.isfinite(atr):
                        if strategy.stop_atr_multiple is not None:
                            stop = entry_price - position * strategy.stop_atr_multiple * atr
                        if strategy.target_atr_multiple is not None:
                            target = entry_price + position * strategy.target_atr_multiple * atr
            if position != 0:
                exit_fill = _exit_bar(row, position, stop, target)
                if exit_fill is not None:
                    exit_price, reason = exit_fill
                    rec = _trade_record(symbol, position, entry_time, timestamp,
                                        entry_price, exit_price, entry_reference, reason, config)
                    trades.append(rec)
                    day_net += rec["net_return"] / config.margin_rate
                    position = 0
                    entry_price = np.nan
        if position != 0:
            timestamp, row = day.iloc[-1].name, day.iloc[-1]
            rec = _trade_record(symbol, position, entry_time, timestamp,
                                entry_price, float(row.close), entry_reference,
                                "end_of_day", config)
            trades.append(rec)
            day_net += rec["net_return"] / config.margin_rate
        daily_rows.append({"trading_date": date, "net_return": day_net})

    trade_df = pd.DataFrame(trades)
    daily = pd.DataFrame(daily_rows).set_index("trading_date")
    daily["equity"] = config.initial_capital * (1.0 + daily["net_return"]).cumprod()
    ret = daily["net_return"].dropna()
    ann = float((1.0 + ret).prod() ** (252.0 / len(ret)) - 1.0) if len(ret) else np.nan
    vol = float(ret.std(ddof=1) * np.sqrt(252.0)) if len(ret) > 1 else np.nan
    sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(252.0)) if len(ret) > 1 and ret.std(ddof=1) > 0 else np.nan
    drawdown = daily["equity"] / daily["equity"].cummax() - 1.0
    metrics = {
        "ann_return": ann, "ann_vol": vol, "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "n_days": float(len(daily)), "n_trades": float(len(trade_df)),
    }
    return BacktestResult(trade_df, daily, metrics)