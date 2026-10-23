"""熊市识别比较：事后标注的状态机、候选规则只用收盘 t 已知的数据、广度的最少品种数、登记一致。合成数据。"""
import numpy as np
import pandas as pd
import pytest

from research import market_bear_regimes as B


def _level(path):
    return pd.Series(path, index=pd.bdate_range("2011-01-03", periods=len(path)), dtype="float64")


def test_protocol_matches_the_code():
    protocol = B.load_protocol()
    assert [c["id"] for c in protocol["candidates"]] == list(B.CANDIDATES)
    assert protocol["ex_post_labels"]["primary"] == {"down": 0.15, "up": 0.15}


def test_bear_episodes_start_at_the_peak_and_end_at_the_trough():
    path = np.r_[np.linspace(100, 120, 21), np.linspace(119, 90, 30), np.linspace(91, 110, 20)]
    level = _level(path)
    episodes = B.bear_episodes(level, 0.15, 0.15)
    assert episodes == [(level.index[20], level.index[50])]
    labels = B.label_days(level.index, episodes)
    assert not labels.iloc[20] and labels.iloc[21] and labels.iloc[50] and not labels.iloc[51]
    # 涨回不够 15% 就还在熊市里，直到样本结束
    assert B.bear_episodes(level.iloc[:60], 0.15, 0.15) == [(level.index[20], level.index[50])]
    assert B.bear_episodes(_level(np.linspace(100, 95, 30)), 0.15, 0.15) == []


def test_breadth_needs_enough_names():
    index = pd.RangeIndex(2)
    flags = pd.DataFrame([[True] * 4 + [False], [True] * 3 + [False] * 1 + [np.nan]], index=index).fillna(False)
    valid = pd.DataFrame([[True] * 5, [True] * 4 + [False]], index=index)
    share = B.breadth(flags.astype(bool), valid)
    assert share.iloc[0] == pytest.approx(0.8) and np.isnan(share.iloc[1])


def _inputs(n=600, seed=5):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=n)
    cols = [f"S{i}" for i in range(8)]
    daily = pd.Series(rng.normal(-0.0005, 0.01, n), index=index)
    trend = pd.DataFrame(rng.choice([-1.0, -1 / 3, 1 / 3, 1.0], size=(n, 8)), index=index, columns=cols)
    lagged = pd.DataFrame(rng.normal(size=(n, 8)), index=index, columns=cols)
    pool = pd.DataFrame(True, index=index, columns=cols)
    return daily, trend, lagged, pool


def test_candidates_at_t_do_not_use_later_data_and_are_boolean():
    daily, trend, lagged, pool = _inputs()
    day = daily.index[400]
    base = B.candidate_bears(daily, trend, lagged, pool)
    later = daily.copy()
    later[later.index > day] = -0.2
    t2, l2 = trend.copy(), lagged.copy()
    t2[t2.index > day] = -1.0
    l2[l2.index > day] = -5.0
    moved = B.candidate_bears(later, t2, l2, pool)
    assert list(base) == list(B.CANDIDATES)
    for name in B.CANDIDATES:
        assert base[name].dtype == bool
        assert base[name].loc[:day].equals(moved[name].loc[:day])


def test_drawdown_candidate_triggers_at_ten_percent():
    index = pd.bdate_range("2010-01-04", periods=200)
    daily = pd.Series(0.0, index=index)
    daily.iloc[100:105] = -0.025                              # 累计约 −11.9%
    flags = B.candidate_bears(daily, pd.DataFrame(index=index), pd.DataFrame(index=index),
                              pd.DataFrame(index=index))
    assert not flags["drawdown"].iloc[102] and flags["drawdown"].iloc[104]
    assert not flags["drawdown"].iloc[:60].any()              # 不足 60 日历史不判
