"""持续期计算测试。

核心是拿暴力实现当参照系。持续期的定义有一个极易做错的分支（当日无满足阈值的
历史观测时从开盘累计），上一轮小时频实现就错在这里，所以这里用随机化交叉验证。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta.factors import intraday as D


def brute_duration(values, threshold):
    """论文定义的逐字翻译：对每个 i 向前扫描，取最近的满足者；无则从开盘累计。"""
    v = np.asarray(values, dtype='float64')
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if not np.isfinite(v[i]):
            continue
        hit = None
        for j in range(i - 1, -1, -1):
            if np.isfinite(v[j]) and abs(v[i] - v[j]) >= threshold:
                hit = j
                break
        out[i] = (i - hit) if hit is not None else i
    return out


def test_matches_brute_force_on_random_walks():
    """300 组随机样本上逐元素比对向量化实现与暴力实现。"""
    rng = np.random.default_rng(42)
    for _ in range(300):
        n = int(rng.integers(2, 60))
        v = 100 + np.cumsum(rng.normal(0, 1, n))
        thr = float(rng.uniform(0.2, 3.0))
        np.testing.assert_array_equal(D.duration_one_day(v, thr), brute_duration(v, thr))


def test_no_qualifying_history_accumulates_from_open():
    """阈值极大时无任何满足者，持续期应为 0,1,2,...（从开盘累计）而非全 0。"""
    v = np.array([100.0, 100.1, 100.2, 100.15, 100.05])
    got = D.duration_one_day(v, threshold=1e9)
    np.testing.assert_array_equal(got, np.arange(5, dtype='float64'))


def test_picks_most_recent_not_earliest():
    """必须取最近的满足者。若错取最早者，这个用例会给出 3 而不是 1。"""
    v = np.array([100.0, 110.0, 120.0, 130.0])
    got = D.duration_one_day(v, threshold=5.0)
    assert got[3] == 1.0


def test_nan_values_propagate_not_zero():
    v = np.array([100.0, np.nan, 102.0, 105.0])
    got = D.duration_one_day(v, threshold=1.0)
    assert np.isnan(got[1])
    assert np.isfinite(got[2]) and np.isfinite(got[3])


def test_does_not_cross_days():
    """每日独立计算：第二日首根必须重新从 0 起，不得沿用前一日的历史。"""
    v = np.array([100.0, 101.0, 102.0, 200.0, 201.0, 202.0])
    days = np.array([0, 0, 0, 1, 1, 1])
    thr = pd.Series({0: 0.5, 1: 0.5})
    got = D.duration_series(v, days, thr)
    np.testing.assert_array_equal(got, np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0]))


def test_abs_diff_no_cross_day():
    v = np.array([100.0, 101.0, 500.0, 501.0])
    days = np.array([0, 0, 1, 1])
    d = D.intraday_abs_diff(v, days)
    assert np.isnan(d[0]) and np.isnan(d[2])   # 每日首根
    np.testing.assert_allclose(d[[1, 3]], [1.0, 1.0])


def test_rolling_threshold_excludes_current_day():
    """当日样本不得进入自身阈值，否则前视。第 0 日无历史，必须为 NaN。"""
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(6), 20)
    v = 100 + np.cumsum(rng.normal(0, 1, len(days)))
    d = D.intraday_abs_diff(v, days)
    thr = D.rolling_threshold(d, days, lookback=3, pct=55.0)
    assert np.isnan(thr.iloc[0])
    assert thr.iloc[1:].notna().all()


def test_rolling_threshold_pools_not_averages():
    """必须把 N 日样本汇总成一个池子再取分位数，而非"每日分位数再平均"。

    构造两日**样本量悬殊**的分布：样本多的一日应主导池化分位数。若实现错成按日
    取分位数再平均，结果会是 50.5 而不是 1.0。
    （注意样本量必须不等——各 99 个时池化中位数正好插值到 50.5，与均值巧合相同，
    区分不出两种实现。）
    """
    days = np.r_[np.zeros(60), np.ones(60), np.full(10, 2), np.full(5, 3)]
    d = np.concatenate([
        np.full(60, np.nan),                       # 第 0 日：无历史，不参与
        np.r_[np.nan, np.full(59, 1.0)],           # 第 1 日：59 个 1
        np.r_[np.nan, np.full(9, 100.0)],          # 第 2 日：9 个 100
        np.full(5, np.nan),                        # 第 3 日：取值无关，只看历史
    ])
    thr = D.rolling_threshold(d, days, lookback=10, pct=50.0)

    assert thr.loc[2] == pytest.approx(1.0)        # 第 2 日只看第 1 日
    # 第 3 日池子 = 59 个 1 + 9 个 100 -> 中位数被样本多的一侧主导
    assert thr.loc[3] == pytest.approx(1.0)
    # 若错误实现为"每日分位数再平均"，会得到 (1 + 100) / 2
    assert thr.loc[3] != pytest.approx(50.5)
