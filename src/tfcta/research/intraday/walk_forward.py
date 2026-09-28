"""日内经典策略的真实 walk-forward 选参与测试。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestResult, run_intraday
from .strategies import StrategyConfig


@dataclass(frozen=True)
class WalkForwardConfig:
    train_years: int = 3
    test_years: int = 1
    min_train_trades: int = 20


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def _metrics(daily: pd.DataFrame) -> dict[str, float]:
    ret = pd.Series(daily["net_return"], dtype="float64").dropna()
    std = ret.std(ddof=1) if len(ret) > 1 else np.nan
    equity = (1.0 + ret).cumprod()
    growth = float(equity.iloc[-1]) if len(equity) else np.nan
    return {
        "ann_return": float(growth ** (252.0 / len(ret)) - 1.0)
        if len(ret) and growth > 0 else np.nan,
        "ann_vol": float(std * np.sqrt(252.0)) if np.isfinite(std) else np.nan,
        "sharpe": float(ret.mean() / std * np.sqrt(252.0))
        if np.isfinite(std) and std > 0 else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min())
        if len(equity) else np.nan,
        "n_days": float(len(ret)),
    }


def _candidate_key(cfg: StrategyConfig) -> str:
    return str(asdict(cfg))


def _fold_years(dates: pd.DatetimeIndex, cfg: WalkForwardConfig) -> list[tuple[int, int, int, int]]:
    years = sorted(set(dates.year))
    out = []
    for i in range(cfg.train_years, len(years), cfg.test_years):
        train_start = years[max(0, i - cfg.train_years)]
        train_end = years[i - 1]
        test_start = years[i]
        test_end = years[min(i + cfg.test_years - 1, len(years) - 1)]
        if test_start <= test_end:
            out.append((train_start, train_end, test_start, test_end))
    return out


def run_walk_forward(minute: pd.DataFrame,
                     candidates: list[StrategyConfig],
                     wf: WalkForwardConfig = WalkForwardConfig(),
                     backtest: BacktestConfig = BacktestConfig(),
                     symbol: str = "") -> WalkForwardResult:
    if not candidates:
        raise ValueError("candidates 不能为空")
    if wf.train_years < 1 or wf.test_years < 1:
        raise ValueError("train_years 和 test_years 必须为正数")
    df = minute.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.normalize()
    dates = pd.DatetimeIndex(sorted(df["trading_date"].dropna().unique()))
    folds = []
    test_daily = []
    test_trades = []
    for train_start, train_end, test_start, test_end in _fold_years(dates, wf):
        train_mask = (df["trading_date"].dt.year >= train_start) & (df["trading_date"].dt.year <= train_end)
        test_mask = (df["trading_date"].dt.year >= test_start) & (df["trading_date"].dt.year <= test_end)
        train = df.loc[train_mask]
        through_test = df.loc[train_mask | test_mask]
        scored = []
        for candidate in candidates:
            result = run_intraday(train, candidate, backtest, symbol)
            scored.append((result.metrics.get("sharpe", np.nan),
                           result.metrics.get("n_trades", 0), candidate))
        eligible = [item for item in scored
                    if item[1] >= wf.min_train_trades and np.isfinite(item[0])]
        if not eligible:
            eligible = [item for item in scored if np.isfinite(item[0])]
        if not eligible:
            continue
        train_sharpe, train_trades, selected = max(
            eligible, key=lambda item: (item[0], -candidates.index(item[2])))
        tested = run_intraday(through_test, selected, backtest, symbol)
        test_days = tested.daily.loc[
            (tested.daily.index.year >= test_start) & (tested.daily.index.year <= test_end)
        ].copy()
        test_trade = tested.trades.copy()
        if not test_trade.empty:
            exit_date = pd.to_datetime(test_trade["exit_time"]).dt.year
            test_trade = test_trade[exit_date.between(test_start, test_end)]
        test_metric = _metrics(test_days)
        folds.append({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "selected": _candidate_key(selected),
            "train_sharpe": train_sharpe, "train_trades": train_trades,
            "test_sharpe": test_metric["sharpe"],
            "test_ann_return": test_metric["ann_return"],
            "test_max_drawdown": test_metric["max_drawdown"],
        })
        test_daily.append(test_days)
        if not test_trade.empty:
            test_trades.append(test_trade)
    daily = pd.concat(test_daily).sort_index() if test_daily else pd.DataFrame(
        columns=["net_return", "equity"])
    if not daily.empty:
        daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    trades = pd.concat(test_trades, ignore_index=True) if test_trades else pd.DataFrame()
    return WalkForwardResult(pd.DataFrame(folds), daily, trades, _metrics(daily))


def classic_candidate_grid(names: tuple[str, ...] = (
        "dual_thrust", "atr", "rbreaker")) -> list[StrategyConfig]:
    """返回事前冻结的经典参数网格；网格只在训练窗口内搜索。"""
    candidates = []
    for name in names:
        if name == "dual_thrust":
            for lookback in (2, 4, 8):
                for k in (0.25, 0.5, 0.75):
                    candidates.append(StrategyConfig(name=name, lookback=lookback,
                                                     k1=k, k2=k,
                                                     stop_atr_multiple=1.0,
                                                     target_atr_multiple=2.0))
        elif name == "atr":
            for window in (10, 20, 40):
                for multiple in (0.5, 1.0, 1.5):
                    candidates.append(StrategyConfig(name=name, atr_window=window,
                                                     atr_multiple=multiple,
                                                     stop_atr_multiple=1.0,
                                                     target_atr_multiple=2.0))
        elif name == "rbreaker":
            for setup in (0.25, 0.35, 0.5):
                candidates.append(StrategyConfig(name=name, rbreaker_setup=setup,
                                                 stop_atr_multiple=1.0,
                                                 target_atr_multiple=2.0))
        else:
            raise ValueError(f"不支持的候选策略: {name}")
    return candidates