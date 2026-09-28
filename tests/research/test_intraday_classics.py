"""隔离日内经典策略模块的撮合与信息时点测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta.research.intraday import (
    BacktestConfig,
    StrategyConfig,
    WalkForwardConfig,
    run_intraday,
    run_walk_forward,
)


def _minutes(days: int = 3) -> pd.DataFrame:
    rows = []
    for day in pd.bdate_range("2021-01-04", periods=days):
        for minute in range(5):
            value = 100.0
            rows.append({
                "timestamp": day + pd.Timedelta(hours=9, minutes=minute),
                "openw": value, "highw": value, "loww": value,
                "closew": value, "trading_date": day,
            })
    return pd.DataFrame(rows).set_index("timestamp")


def test_requires_completed_previous_day_for_fali():
    df = _minutes(2)
    df.loc[df.index[5], "highw"] = 105.0
    result = run_intraday(df, StrategyConfig(name="fali"),
                          BacktestConfig(margin_rate=0.3), "RB")
    assert result.trades.iloc[0]["entry_time"].date() == pd.Timestamp("2021-01-05").date()


def test_same_bar_stop_loss_wins_over_target():
    df = _minutes(2)
    df.loc[df.index[5], "highw"] = 105.0
    df.loc[df.index[6], ["openw", "highw", "loww", "closew"]] = [105.0, 110.0, 90.0, 100.0]
    result = run_intraday(
        df, StrategyConfig(name="fali", atr_window=1,
                           stop_atr_multiple=1, target_atr_multiple=1),
        BacktestConfig(margin_rate=0.3), "RB")
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "stop_loss"


def test_every_position_is_closed_at_end_of_day():
    df = _minutes(2)
    df.loc[df.index[5], "highw"] = 105.0
    result = run_intraday(df, StrategyConfig(name="fali"),
                          BacktestConfig(margin_rate=0.3), "RB")
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "end_of_day"
    assert result.trades.iloc[0]["exit_time"].date() == pd.Timestamp("2021-01-05").date()


def test_margin_rate_scales_daily_return():
    df = _minutes(2)
    df.loc[df.index[5], "highw"] = 105.0
    df.loc[df.index[6], ["openw", "highw", "loww", "closew"]] = [105.0, 105.0, 105.0, 110.0]
    one = run_intraday(df, StrategyConfig(name="fali"),
                       BacktestConfig(fee_rate=0, margin_rate=1), "RB")
    half = run_intraday(df, StrategyConfig(name="fali"),
                        BacktestConfig(fee_rate=0, margin_rate=0.5), "RB")
    assert half.daily.iloc[-1]["net_return"] == pytest.approx(
        2 * one.daily.iloc[-1]["net_return"])


def test_walk_forward_test_dates_follow_training_dates():
    rows = []
    for year in range(2016, 2021):
        for day_no in range(3):
            day = pd.Timestamp(year=year, month=1, day=day_no + 2)
            for minute in range(5):
                value = 100.0 + year - 2016 + minute * 0.1
                rows.append({
                    "timestamp": day + pd.Timedelta(hours=9, minutes=minute),
                    "openw": value, "highw": value + 0.2,
                    "loww": value - 0.2, "closew": value,
                    "trading_date": day,
                })
    minute = pd.DataFrame(rows).set_index("timestamp")
    result = run_walk_forward(
        minute,
        [StrategyConfig(name="dual_thrust", lookback=1, atr_window=1)],
        WalkForwardConfig(train_years=2, test_years=1, min_train_trades=0),
        BacktestConfig(fee_rate=0, margin_rate=1),
        "TEST",
    )
    assert not result.folds.empty
    assert (result.folds["train_end"] < result.folds["test_start"]).all()
    assert result.daily.index.year.min() >= result.folds["test_start"].min()