"""商品图表报告及记账契约。"""
import numpy as np
import pandas as pd
import pytest

# --- strategy_report ---

from research.report import strategy_report as SR

def test_report_protocol_lists_the_three_frozen_books_and_locked_windows():
    protocol = SR.load_protocol()
    assert protocol["books"] == list(SR.BOOK_IDS)
    assert "不读 2022" in protocol["sample"] and "不读 2026" in protocol["sample"]


def test_report_components_rebuild_the_frozen_rule_and_short_overrides():
    index = pd.bdate_range("2018-01-01", periods=4)
    cols = ["A", "B"]
    signal = pd.DataFrame([[True, True], [False, False], [True, True], [True, True]], index=index, columns=cols)
    lagged = pd.DataFrame([[0.1, -0.1]] * 4, index=index, columns=cols)
    trend = pd.DataFrame(-1.0, index=index, columns=cols)
    bear = pd.Series([False, False, True, True], index=index)
    out = SR.component_decisions(signal, lagged, trend, bear)
    assert out["full"]["A"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert out["full"]["B"].tolist() == [0.0, 0.0, -1.0, -1.0]
    assert out["signal_only"]["B"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert out["carry_only"]["A"].eq(1.0).all() and out["carry_only"]["B"].eq(0.0).all()
    pd.testing.assert_frame_equal(out["full"], out["long_leg"] + out["short_leg"])


def test_leg_contributions_reconcile_to_ledger_net_return():
    day = pd.Timestamp("2018-01-02")
    daily = pd.DataFrame({"期初权益": [1000.0], "毛盈亏": [30.0], "交易费用": [5.0],
                          "权益清零调整": [0.0], "净收益": [0.025]}, index=pd.DatetimeIndex([day]))
    positions = pd.DataFrame({"交易日": [day, day], "品种": ["IF", "IH"], "手数": [2, -1],
                              "合约乘数": [10.0, 5.0]})
    delta = pd.DataFrame({"IF": [1.0], "IH": [-2.0]}, index=daily.index)
    legs = SR.leg_contributions(daily, positions, delta)
    assert legs.loc[day].tolist() == [0.02, 0.01, -0.005]

