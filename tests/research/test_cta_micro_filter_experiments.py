import numpy as np
import pandas as pd

from research.cta_micro_filter_experiments import (
    _entry_schedule,
    _validation_duration_factors,
    build_signal_variants,
    market_regimes,
)


def test_micro_filter_variants_do_not_change_when_future_values_change():
    index = pd.bdate_range("2015-01-01", periods=400)
    symbols = ["RB", "CU"]
    rng = np.random.default_rng(412)
    micro = pd.DataFrame(rng.normal(size=(len(index), 2)), index=index, columns=symbols)
    trend = pd.DataFrame(1.0, index=index, columns=symbols)
    pmt = pd.DataFrame(rng.uniform(size=(len(index), 2)), index=index, columns=symbols)
    volume_duration_ratio = pd.DataFrame(
        rng.uniform(size=(len(index), 2)), index=index, columns=symbols,
    )
    duration_summary = {
        name: pd.DataFrame(
            rng.uniform(size=(len(index), 2)), index=index, columns=symbols,
        )
        for name in (
            "price_duration_mean", "price_duration_vol",
            "volume_duration_mean", "volume_duration_vol",
        )
    }
    cutoff = 350

    original = build_signal_variants(
        trend, micro, pmt, volume_duration_ratio, duration_summary,
    )
    changed_micro = micro.copy()
    changed_micro.iloc[cutoff:] = 1_000.0
    changed_pmt = pmt.copy()
    changed_pmt.iloc[cutoff:] = -1_000.0
    changed_volume_duration_ratio = volume_duration_ratio.copy()
    changed_volume_duration_ratio.iloc[cutoff:] = -1_000.0
    changed_duration_summary = {
        name: values.copy() for name, values in duration_summary.items()
    }
    for values in changed_duration_summary.values():
        values.iloc[cutoff:] = -1_000.0
    changed = build_signal_variants(
        trend, changed_micro, changed_pmt, changed_volume_duration_ratio,
        changed_duration_summary,
    )

    for name in original:
        pd.testing.assert_frame_equal(
            original[name].iloc[:cutoff],
            changed[name].iloc[:cutoff],
        )


def test_micro_filters_keep_trend_as_the_required_directional_gate():
    index = pd.bdate_range("2015-01-01", periods=400)
    symbols = ["RB"]
    micro = pd.DataFrame(
        np.tile(np.r_[np.linspace(-1, 1, 200), np.linspace(1, -1, 200)], (1, 1)).T,
        index=index,
        columns=symbols,
    )
    pmt = pd.DataFrame(np.linspace(0.0, 1.0, len(index)), index=index, columns=symbols)
    volume_duration_ratio = pd.DataFrame(
        np.linspace(0.0, 1.0, len(index)), index=index, columns=symbols,
    )
    duration_summary = {
        name: pd.DataFrame(
            np.linspace(0.0, 1.0, len(index)), index=index, columns=symbols,
        )
        for name in (
            "price_duration_mean", "price_duration_vol",
            "volume_duration_mean", "volume_duration_vol",
        )
    }
    trend = pd.DataFrame(-1.0, index=index, columns=symbols)

    variants = build_signal_variants(
        trend, micro, pmt, volume_duration_ratio, duration_summary,
    )

    assert not variants["trend_20d"].to_numpy().any()
    assert not variants["trend_plus_daily_micro_filter"].to_numpy().any()
    assert not variants["trend_plus_ema3_micro_filter"].to_numpy().any()
    assert not variants["trend_plus_ema5_micro_filter"].to_numpy().any()
    assert not variants["trend_plus_duration_clock_stability"].to_numpy().any()
    assert not variants["trend_plus_micro_hysteresis"].to_numpy().any()
    assert not variants["trend_plus_daily_micro_filter_and_volume_duration"].to_numpy().any()
    assert not variants["trend_plus_daily_micro_filter_price_duration_stable"].to_numpy().any()
    assert not variants["trend_plus_daily_micro_filter_volume_duration_stable"].to_numpy().any()
    assert not variants["trend_plus_daily_micro_filter_both_durations_stable"].to_numpy().any()


def test_entry_schedule_does_not_reopen_on_the_simulator_exit_day():
    index = pd.bdate_range("2020-01-01", periods=20)
    signal = pd.DataFrame(False, index=index, columns=["RB"])
    signal.iloc[[0, 5, 6], 0] = True
    returns = pd.DataFrame(0.01, index=index, columns=["RB"])

    entries = _entry_schedule(
        signal, returns, {2020: ["RB"]}, index[0], index[-1],
    )

    assert entries.index[entries["RB"]].tolist() == [index[1], index[7]]


def test_validation_duration_factors_returns_empty_features_for_empty_universe():
    pmt, volume_duration_ratio, duration = _validation_duration_factors([])

    assert pmt.empty
    assert volume_duration_ratio.empty
    assert set(duration) == {
        "price_duration_mean", "price_duration_vol",
        "volume_duration_mean", "volume_duration_vol",
    }
    assert all(values.empty for values in duration.values())


def test_daily_market_regimes_are_ex_ante_and_ignore_future_prices():
    index = pd.bdate_range("2014-01-01", periods=500)
    close = np.linspace(100.0, 200.0, len(index))
    bars = {
        "close": pd.DataFrame({"RB": close}, index=index),
        "closew": pd.DataFrame({"RB": close}, index=index),
        "highw": pd.DataFrame({"RB": close + 1.0}, index=index),
        "loww": pd.DataFrame({"RB": close - 1.0}, index=index),
    }
    regime, volatility = market_regimes(bars)

    changed_bars = {name: frame.copy() for name, frame in bars.items()}
    cutoff = 400
    for name in ("close", "closew"):
        changed_bars[name].iloc[cutoff:, 0] = np.linspace(200.0, 100.0, len(index) - cutoff)
    changed_bars["highw"].iloc[cutoff:, 0] = changed_bars["closew"].iloc[cutoff:, 0] + 1.0
    changed_bars["loww"].iloc[cutoff:, 0] = changed_bars["closew"].iloc[cutoff:, 0] - 1.0
    changed_regime, changed_volatility = market_regimes(changed_bars)

    pd.testing.assert_series_equal(regime.iloc[:cutoff, 0], changed_regime.iloc[:cutoff, 0])
    pd.testing.assert_series_equal(
        volatility.iloc[:cutoff, 0], changed_volatility.iloc[:cutoff, 0],
    )
    assert set(regime["RB"].dropna().unique()) <= {
        "range", "transition", "trend_up", "trend_down",
    }
    assert set(volatility["RB"].dropna().unique()) <= {"low_vol", "high_vol"}
