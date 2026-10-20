"""当天分数大于 0 且滞后一天的展期收益大于 0 才做多。合成数据。"""
import numpy as np
import pandas as pd

from strategies import market_clock_carry as S
from strategies.catalog import SELECTED
from strategies.research_catalog import RESEARCH_EXPERIMENTS


def test_config_locks_the_one_times_gross_and_the_carry_lag():
    config = S.load_config()
    assert config["parameters"]["gross_exposure_limit"] == 1.0
    assert config["parameters"]["roll_yield_lag_days"] == 1
    assert config["parameters"]["drawdown_breaker"] == 0.10
    assert S in RESEARCH_EXPERIMENTS
    assert all(strategy.ID != S.ID for strategy in SELECTED)


def test_both_the_score_and_the_lagged_carry_must_be_positive():
    score = pd.Series([0.4, 0.4, -0.2], index=pd.RangeIndex(3))
    carry = pd.DataFrame({"A": [0.2, -0.1, 0.3]})
    weights = S.decision_weights(score, carry)
    assert weights["A"].tolist() == [1.0, 0.0, 0.0]


def test_targets_use_the_previous_close_and_the_previous_roll_yield():
    index = pd.bdate_range("2015-01-01", periods=400)
    symbols = [f"S{i:02d}" for i in range(12)]
    rng = np.random.default_rng(2)
    signed = {
        name: pd.DataFrame(rng.normal(size=(400, len(symbols))), index=index, columns=symbols)
        for name in S.FACTORS
    }
    universe = {int(year): symbols for year in sorted(set(index.year))}
    carry = pd.DataFrame(0.1, index=index, columns=symbols)
    weights = S.targets(signed, {}, universe, index, symbols, carry=carry)
    changed = {name: frame.copy() for name, frame in signed.items()}
    for frame in changed.values():
        frame.iloc[-1] = 10.0
    carry_changed = carry.copy()
    carry_changed.iloc[-1] = -1.0
    pd.testing.assert_frame_equal(
        weights, S.targets(changed, {}, universe, index, symbols, carry=carry_changed),
    )
    assert weights.iloc[0].eq(0.0).all()
