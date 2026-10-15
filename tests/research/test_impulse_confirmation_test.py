import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
from impulse_confirmation_test import (
    commodity_factor_state_daily,
    commodity_incremental_coefficients,
    commodity_strategy_daily,
    index_incremental_regression_summary,
    index_strategy_daily,
)


def _commodity_panel():
    size = 20
    half = size // 2
    return pd.DataFrame({
        "trading_date": pd.Timestamp("2018-01-02"),
        "symbol": [f"C{i}" for i in range(size)],
        "open_move": np.ones(size) * 0.01,
        "tail_move_5m": np.array([0.001] * half + [-0.001] * half),
        "gross_11:30": np.array([0.02] * half + [-0.02] * half),
        "cost_11:30": np.ones(size) * 0.001,
        "clock_diff": np.arange(size, dtype=float),
        "path_efficiency": np.arange(size, dtype=float),
        "volume_peak_clock": np.arange(size, dtype=float),
    })


def test_commodity_strategy_uses_trend_after_confirmation_and_fades_without_it():
    daily = commodity_strategy_daily(_commodity_panel())
    conditional = daily.loc[
        daily.strategy.eq("confirmed_momentum_unconfirmed_reversal")
    ].iloc[0]

    assert conditional.n_names == 20
    assert conditional.active_fraction == 1.0
    assert np.isclose(conditional.gross_return, 0.02)
    assert np.isclose(conditional.net_return, 0.019)


def test_commodity_factor_spreads_are_reported_separately_by_confirmation_state():
    daily = commodity_factor_state_daily(_commodity_panel())

    assert set(daily.confirmation_state) == {"confirmed", "unconfirmed"}
    assert set(daily.factor) == {
        "clock_diff", "path_efficiency", "volume_peak_clock"
    }
    assert daily.n_names.eq(10).all()


def test_index_strategies_expose_gross_breakout_states_with_positions():
    panel = pd.DataFrame({
        "trading_date": [pd.Timestamp("2018-01-02")] * 2,
        "symbol": ["IF", "IF"],
        "gap_return": [0.01, -0.01],
        "open_move": [0.01, 0.01],
        "gross_11:30": [0.02, -0.03],
    })
    daily = index_strategy_daily(panel)
    unconfirmed_fade = daily.loc[
        daily.strategy.eq("fade_unconfirmed")
    ].set_index("trading_date").iloc[0]

    assert np.isclose(unconfirmed_fade.gross_return, -0.015)
    assert unconfirmed_fade.active_fraction == 0.5


def test_commodity_regression_measures_time_factor_after_price_controls():
    rng = np.random.default_rng(21)
    size = 60
    open_move = rng.normal(size=size)
    tail_move = rng.normal(size=size)
    confirmation_centered = (
        2 * (np.sign(open_move) * np.sign(tail_move) > 0).astype(float) - 1
    )
    clock = rng.normal(size=size)
    path = rng.normal(size=size)
    volume = rng.normal(size=size)
    z_open = (open_move - open_move.mean()) / open_move.std()
    z_tail = (tail_move - tail_move.mean()) / tail_move.std()
    z_clock = (clock - clock.mean()) / clock.std()
    outcome = (
        0.001 * z_open + 0.0003 * z_tail
        + 0.0002 * z_clock + 0.0005 * z_clock * confirmation_centered
        + rng.normal(scale=0.00001, size=size)
    )
    panel = pd.DataFrame({
        "trading_date": pd.Timestamp("2018-01-02"),
        "open_move": open_move,
        "tail_move_5m": tail_move,
        "gross_11:30": outcome,
        "clock_diff": clock,
        "path_efficiency": path,
        "volume_peak_clock": volume,
    })

    coefficients = commodity_incremental_coefficients(panel)

    assert len(coefficients) == 1
    assert coefficients.loc[0, "n_names"] == size
    assert np.isclose(
        coefficients.loc[0, "clock_diff_confirmed_slope"], 0.0007, atol=0.0001
    )


def test_index_hac_regression_controls_gap_and_open_move_before_timing_factors():
    rng = np.random.default_rng(22)
    size = 240
    gap = np.exp(rng.normal(scale=0.01, size=size))
    open_move = rng.normal(scale=0.01, size=size)
    confirmation_centered = (
        2 * (np.sign(gap) * np.sign(open_move) > 0).astype(float) - 1
    )
    clock = rng.normal(size=size)
    path = rng.normal(size=size)
    volume = rng.normal(size=size)
    z_gap = (np.abs(gap) - np.abs(gap).mean()) / np.abs(gap).std()
    aligned = np.sign(gap) * open_move
    z_open = (aligned - aligned.mean()) / aligned.std()
    z_clock = (clock - clock.mean()) / clock.std()
    directional_return = (
        0.0004 * z_gap + 0.0002 * z_open + 0.0003 * z_clock
        + 0.0001 * z_clock * confirmation_centered
        + rng.normal(scale=0.0002, size=size)
    )
    panel = pd.DataFrame({
        "trading_date": pd.bdate_range("2016-01-04", periods=size),
        "symbol": "IF",
        "gap_return": gap,
        "open_move": open_move,
        "gross_11:30": directional_return,
        "clock_diff": clock,
        "path_efficiency": path,
        "volume_peak_clock": volume,
    })

    summary = index_incremental_regression_summary(panel)
    estimate = summary.loc[
        summary.coefficient.eq("clock_diff_confirmed_slope")
    ].iloc[0]

    assert estimate.days == size
    assert np.isclose(estimate.mean_bp_per_1sd, 4.0, atol=1.0)
    assert np.isfinite(estimate.nw_t_lag5)
