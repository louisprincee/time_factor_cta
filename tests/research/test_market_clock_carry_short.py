"""对偶空头检验：决策对称、按方向拆盈亏、回撤段落。合成数据。"""
import numpy as np
import pandas as pd
import pytest

from research import market_clock_carry_short as R


def test_protocol_reads_only_the_research_period():
    protocol = R.load_protocol()
    assert "不读 2022" in protocol["sample"] and "不读 2026" in protocol["sample"]
    assert [v["id"] for v in protocol["variants"]] == ["long", "short", "long_short"]


def test_dual_weights_are_the_mirror_of_the_long_rule_and_zero_on_ties():
    score = pd.Series([0.5, 0.5, -0.5, -0.5, 0.0], index=pd.RangeIndex(5))
    carry = pd.DataFrame({"A": [0.1, -0.1, -0.1, 0.1, -0.1]})
    assert R.dual_weights(score, carry, ["long"])["A"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert R.dual_weights(score, carry, ["short"])["A"].tolist() == [0.0, 0.0, -1.0, 0.0, 0.0]
    assert R.dual_weights(score, carry, ["long", "short"])["A"].tolist() == [1.0, 0.0, -1.0, 0.0, 0.0]
    carry.iloc[2, 0] = np.nan
    assert R.dual_weights(score, carry, ["short"])["A"].iloc[2] == 0.0


def test_side_pnl_splits_long_and_short_lots():
    days = pd.bdate_range("2020-01-02", periods=2)
    positions = pd.DataFrame({"交易日": [days[0], days[0], days[1]], "品种": ["A", "B", "A"],
                              "手数": [2, -3, 2], "合约乘数": [10.0, 5.0, 10.0]})
    delta = pd.DataFrame({"A": [1.0, -2.0], "B": [4.0, 0.0]}, index=days)
    out = R.side_pnl(positions, delta)
    assert out["多头毛盈亏"] == pytest.approx(2 * 10 * 1.0 + 2 * 10 * -2.0)
    assert out["空头毛盈亏"] == pytest.approx(-3 * 5 * 4.0)


def test_drawdown_episodes_are_deepest_first_and_do_not_overlap():
    # 净值 1 → 0.8 → 0.85 → 0.7 → 1.1 → 1.0：第一段水下跌到 0.7，第二段从 1.1 跌到 1.0
    nav = pd.Series([1.0, 0.8, 0.85, 0.7, 1.1, 1.0], index=pd.bdate_range("2020-01-02", periods=6))
    returns = nav.pct_change().fillna(0.0)
    episodes = R.drawdown_episodes(returns, 3)
    assert len(episodes) == 2
    (p0, t0, d0), (p1, t1, d1) = episodes
    assert d0 == pytest.approx(-0.3) and (p0, t0) == (nav.index[0], nav.index[3])
    assert d1 == pytest.approx(1.0 / 1.1 - 1) and (p1, t1) == (nav.index[4], nav.index[5])
