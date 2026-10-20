import numpy as np
import pandas as pd

from strategies.book import adx_above_range, adx_is_range, entry_schedule, high_volatility, upper_tail
from strategies.catalog import SELECTED
from strategies.regime_two_leg import signal as regime_signal
from strategies.trend_adx_range import signal as adx_signal
from strategies.trend_ema3 import signal as ema_signal
from strategies.trend_high_atr import signal as atr_signal
from strategies.research_catalog import RESEARCH_EXPERIMENTS
from tfcta.factors import tsmom_sign


def test_only_the_upper_tail_is_long_and_a_tighter_cutoff_fires_less():
    series = pd.Series(
        [0, 0, 0, 1, 0.4, -1],
        index=pd.bdate_range("2021-01-04", periods=6),
    )
    wide = upper_tail(series, 4, 0.7)
    tight = upper_tail(series, 4, 0.9)
    assert wide.iloc[4] == 1 and tight.iloc[4] == 0
    assert upper_tail(series, 4, 0.8).iloc[5] == 0


def test_entry_schedule_skips_the_exit_day_and_respects_the_pool():
    index = pd.bdate_range("2020-01-01", periods=20)
    signal = pd.DataFrame(False, index=index, columns=["RB"])
    signal.iloc[[0, 5, 6], 0] = True
    returns = pd.DataFrame(0.01, index=index, columns=["RB"])

    entries = entry_schedule(signal, returns, {2020: ["RB"]}, index[0], index[-1], holding_days=5)

    assert entries.index[entries["RB"]].tolist() == [index[1], index[7]]
    outside = entry_schedule(signal, returns, {2020: ["CU"]}, index[0], index[-1], holding_days=5)
    assert not outside["RB"].any()


def test_research_experiments_are_registered_separately_from_frozen_strategies():
    assert [item.ID for item in SELECTED] == [
        "trend_ema3", "trend_high_atr", "trend_adx_range", "regime_two_leg",
    ]
    assert [item.ID for item in RESEARCH_EXPERIMENTS] == ["market_clock_long"]


def _reference_regimes(bars):
    """逐行写出的 14 日 Wilder ATR% 与 ADX，作为冻结信号的独立对照。"""
    high, low, close = bars["highw"], bars["loww"], bars["closew"]
    prior = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prior).abs(), (low - prior).abs()], keys=range(3),
    ).groupby(level=1).max().reindex(close.index)
    true_range.iloc[0] = (high - low).iloc[0]

    def wilder(frame):
        return frame.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    up, down = high.diff(), -low.diff()
    plus = wilder(up.where((up > down) & (up > 0), 0.0)) / wilder(true_range) * 100
    minus = wilder(down.where((down > up) & (down > 0), 0.0)) / wilder(true_range) * 100
    adx = wilder((plus - minus).abs() / (plus + minus) * 100)
    atr = wilder(true_range) / close
    median = atr.shift(1).rolling(250, min_periods=250).median()
    return adx, atr.gt(median)


def test_selected_signals_match_an_independent_reference():
    index = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(19)
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(scale=0.4, size=(len(index), 2)), axis=0),
        index=index,
        columns=["RB", "CU"],
    )
    bars = {
        "close": close,
        "closew": close,
        "highw": close + 1.0 + rng.uniform(0, 1, size=close.shape),
        "loww": close - 1.0 - rng.uniform(0, 1, size=close.shape),
    }
    micro = pd.DataFrame(rng.normal(size=(len(index), 2)), index=index, columns=["RB", "CU"])
    trend = tsmom_sign(bars["close"], bars["closew"], window=10)
    adx, high_vol = _reference_regimes(bars)
    ranging = adx.notna() & adx.le(20)
    above_range = adx.notna() & adx.gt(20)

    def tail(values):
        return pd.DataFrame({s: upper_tail(values[s], 250, 0.8) for s in values}, index=index).gt(0)

    raw_tail = tail(micro)
    ema_tail = tail(micro.ewm(span=3, min_periods=3, adjust=False).mean())
    long = trend.gt(0)

    pd.testing.assert_frame_equal(high_volatility(bars), high_vol)
    pd.testing.assert_frame_equal(adx_is_range(bars), ranging)
    pd.testing.assert_frame_equal(adx_above_range(bars), above_range)
    pd.testing.assert_frame_equal(
        ema_signal(trend, micro, high_vol, ranging, above_range), long & ema_tail,
    )
    pd.testing.assert_frame_equal(
        atr_signal(trend, micro, high_vol, ranging, above_range), long & raw_tail & high_vol,
    )
    pd.testing.assert_frame_equal(
        adx_signal(trend, micro, high_vol, ranging, above_range), long & raw_tail & ranging,
    )
    pd.testing.assert_frame_equal(
        regime_signal(trend, micro, high_vol, ranging, above_range),
        (ranging & raw_tail) | (above_range & long & raw_tail & high_vol),
    )
    assert ranging.to_numpy().any() and above_range.to_numpy().any() and high_vol.to_numpy().any()


def test_downtrend_blocks_the_trend_leg_and_the_range_leg_ignores_it():
    index = pd.bdate_range("2015-01-01", periods=300)
    micro = pd.DataFrame(
        np.linspace(-1, 2, len(index)), index=index, columns=["RB"],
    )
    trend = pd.DataFrame(-1.0, index=index, columns=["RB"])
    high_vol = pd.DataFrame(True, index=index, columns=["RB"])
    ranging = pd.DataFrame(True, index=index, columns=["RB"])
    quiet = pd.DataFrame(False, index=index, columns=["RB"])
    tail = upper_tail(micro["RB"]).gt(0)

    assert not ema_signal(trend, micro, high_vol, ranging, quiet).to_numpy().any()
    assert not atr_signal(trend, micro, high_vol, ranging, quiet).to_numpy().any()
    assert not adx_signal(trend, micro, high_vol, ranging, quiet).to_numpy().any()
    pd.testing.assert_series_equal(
        regime_signal(trend, micro, high_vol, ranging, quiet)["RB"],
        tail,
        check_names=False,
    )
    assert not regime_signal(trend, micro, high_vol, quiet, high_vol).to_numpy().any()
    assert not regime_signal(trend, micro, high_vol, quiet, quiet).to_numpy().any()


def test_configs_are_frozen_together_and_match_the_catalog():
    assert [item.ID for item in SELECTED] == [
        "trend_ema3", "trend_high_atr", "trend_adx_range", "regime_two_leg",
    ]
    frozen_at = set()
    for strategy in SELECTED:
        config = strategy.load_config()
        assert config["status"] == "frozen"
        assert config["frozen_at"] == "2026-10-19"
        assert config["chosen"] == config["parameters"]
        assert "不读 2023" in config["sample"]
        frozen_at.add(config["frozen_at"])
    assert frozen_at == {"2026-10-19"}
