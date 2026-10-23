"""慢趋势空头：趋势只用过去收盘、熊市才开空头腿、多头腿同按池宽1倍、规格与代码一致。合成数据。"""
import json

import numpy as np
import pandas as pd
import pytest

from strategies import market_clock_carry as S
from strategies import market_clock_slow_short as T


def _bars(index, symbols, drift=-0.001, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.exp(np.cumsum(drift + 0.01 * rng.normal(size=(len(index), len(symbols))), axis=0)),
                         index=index, columns=symbols)
    return {"close": close, "closew": close - 50.0}          # 加法复权：差分不变，比值会变


def _signed(index, symbols, seed=2):
    rng = np.random.default_rng(seed)
    return {name: pd.DataFrame(rng.normal(size=(len(index), len(symbols))), index=index, columns=symbols)
            for name in S.FACTORS}


def test_symbol_trend_uses_only_past_closes_and_needs_the_longest_window():
    index = pd.bdate_range("2015-01-01", periods=300)
    bars = _bars(index, ["A", "B"])
    trend = T.symbol_trend(bars, ["A", "B"])
    assert trend.iloc[:250].isna().all().all()
    assert set(np.unique(trend.dropna().to_numpy())) <= {-1.0, -1 / 3, 1 / 3, 1.0}
    later = {k: v.copy() for k, v in bars.items()}
    for frame in later.values():
        frame.iloc[-1] *= 3.0
    pd.testing.assert_frame_equal(trend.iloc[:-1], T.symbol_trend(later, ["A", "B"]).iloc[:-1])


def test_market_trend_is_negative_in_a_steady_decline_and_missing_before_250_days():
    daily = pd.Series(-0.001, index=pd.bdate_range("2015-01-01", periods=300))
    trend = T.market_trend(daily)
    assert trend.iloc[:249].isna().all() and trend.iloc[-1] == -1.0


def test_short_leg_needs_bear_trend_and_negative_carry_and_long_leg_is_unchanged():
    index = pd.RangeIndex(4)
    score = pd.Series([0.5, 0.5, 0.5, -0.5], index=index)
    carry = pd.DataFrame({"A": [0.1, 0.1, 0.1, 0.1], "B": [-0.1, -0.1, 0.1, -0.1]}, index=index)
    trend = pd.DataFrame({"A": [-1.0] * 4, "B": [-1 / 3, -1 / 3, -1.0, 1 / 3]}, index=index)
    bear = pd.Series([False, True, True, np.nan], index=index)
    weights = T.decision_weights(score, carry, trend, bear)
    assert weights["A"].tolist() == [1.0, 1.0, 1.0, 0.0]          # 展期为正：多头腿，跟分数走
    assert weights["B"].tolist() == [0.0, -1.0, 1.0, 0.0]         # 熊市、趋势和展期都为负才做空；展期转正回多头腿
    pd.testing.assert_frame_equal(weights.clip(lower=0.0), S.decision_weights(score, carry))


def test_targets_use_only_the_previous_close():
    index = pd.bdate_range("2015-01-01", periods=420)
    symbols = [f"S{i:02d}" for i in range(12)]
    universe = {int(year): symbols for year in sorted(set(index.year))}
    signed, bars = _signed(index, symbols), _bars(index, symbols)
    carry = pd.DataFrame(np.where(np.arange(12) % 2, -0.1, 0.1), index=symbols).T
    carry = pd.DataFrame(np.repeat(carry.to_numpy(), len(index), axis=0), index=index, columns=symbols)
    weights = T.targets(signed, bars, universe, index, symbols, carry=carry)
    assert weights.iloc[0].eq(0.0).all() and weights.lt(0).any().any()
    changed_signed = {name: frame.copy() for name, frame in signed.items()}
    changed_bars = {name: frame.copy() for name, frame in bars.items()}
    for frame in [*changed_signed.values(), *changed_bars.values()]:
        frame.iloc[-1] *= 5.0
    changed_carry = carry.copy()
    changed_carry.iloc[-2:] *= -1.0           # 执行日 t 用的是 t−2 的展期收益
    pd.testing.assert_frame_equal(weights, T.targets(changed_signed, changed_bars, universe, index, symbols,
                                                     carry=changed_carry))
    long_only = S.targets(signed, universe, index, symbols, carry=carry)
    pd.testing.assert_frame_equal(weights.clip(lower=0.0), long_only)


def test_config_must_match_the_short_leg_and_is_frozen(tmp_path):
    config = T.BOOK.frozen()
    assert config["frozen_at"] == "2026-10-22" and config["chosen"] == config["parameters"]
    assert "不读 2026" in config["sample"] and "2022、2023、2024 已经打开过" in config["sample"]
    for key, value in (("short_trend_windows", [20, 60, 120]), ("bear_rule", "market_trend <= 0"),
                       ("short_weight", -0.5), ("gross_exposure_limit", 1.5)):
        broken = {**config, "parameters": {**config["parameters"], key: value}}
        broken["chosen"] = broken["parameters"]
        with pytest.raises(ValueError):
            T.BOOK.check(json.loads(json.dumps(broken)))


def test_three_window_book_only_changes_the_bear_windows():
    from strategies import market_clock_slow_short_fast3 as F3

    assert F3.BOOK.BEAR_TREND_WINDOWS == (20, 60, 120) and T.BOOK.BEAR_TREND_WINDOWS == (60, 120, 250)
    config = F3.BOOK.frozen()
    assert config["frozen_at"] == "2026-10-23" and config["parameters"]["bear_trend_windows"] == [20, 60, 120]
    assert {k: v for k, v in config["parameters"].items() if k != "bear_trend_windows"} == {
        k: v for k, v in T.BOOK.load_config()["parameters"].items() if k != "bear_trend_windows"}
    assert len(F3.BOOK.FILES) == 3
    short = pd.Series(-0.001, index=pd.bdate_range("2015-01-01", periods=130))
    assert T.market_trend(short, (20, 60, 120)).iloc[-1] == -1.0 and np.isnan(T.market_trend(short).iloc[-1])
    index = pd.bdate_range("2015-01-01", periods=420)
    symbols = [f"S{i:02d}" for i in range(12)]
    universe = {int(year): symbols for year in sorted(set(index.year))}
    signed, bars = _signed(index, symbols), _bars(index, symbols, drift=-0.0005)
    carry = pd.DataFrame(np.where(np.arange(12) % 2, -0.1, 0.1), index=symbols).T
    carry = pd.DataFrame(np.repeat(carry.to_numpy(), len(index), axis=0), index=index, columns=symbols)
    fast = T.targets(signed, bars, universe, index, symbols, carry=carry, bear_windows=(20, 60, 120))
    pool = S.pool_mask(universe, index, symbols)
    daily = T.close_return(bars, symbols).where(pool).mean(axis=1)
    bear = T.market_trend(daily, (20, 60, 120)).lt(0)
    expected = T.decision_weights(S.market_score(signed, universe, index), carry.shift(1),
                                  T.symbol_trend(bars, symbols), bear).shift(1).fillna(0.0).where(pool, 0.0)
    pd.testing.assert_frame_equal(fast, expected)
    pd.testing.assert_frame_equal(fast.clip(lower=0.0), S.targets(signed, universe, index, symbols, carry=carry))
