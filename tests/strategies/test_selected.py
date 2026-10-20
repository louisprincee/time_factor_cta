import numpy as np
import pandas as pd

from strategies.book import adx_above_range, adx_is_range, entry_schedule, high_volatility, upper_tail
from strategies.catalog import SELECTED
from strategies.research_catalog import RESEARCH_EXPERIMENTS


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
    assert SELECTED == ()
    assert [item.ID for item in RESEARCH_EXPERIMENTS] == [
        "market_clock_carry",
    ]


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


def test_volatility_and_adx_match_an_independent_reference():
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
    adx, high_vol = _reference_regimes(bars)
    ranging = adx.notna() & adx.le(20)
    above_range = adx.notna() & adx.gt(20)
    pd.testing.assert_frame_equal(high_volatility(bars), high_vol)
    pd.testing.assert_frame_equal(adx_is_range(bars), ranging)
    pd.testing.assert_frame_equal(adx_above_range(bars), above_range)
