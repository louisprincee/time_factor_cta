"""还在用的研究契约：展期口径、熊市空头规则、图表报告。"""

import numpy as np
import pandas as pd
import pytest

def test_index_carry_uses_the_largest_oi_later_contract_and_the_maturity_gap():
    from download.index.getIndexRollYield import main_sub_carry

    day = pd.Timestamp("2018-01-02")
    daily = pd.DataFrame({
        "date": day, "contract": ["IF1801", "IF1802", "IF1803", "IF1712"],
        "settlement": [4000.0, 3990.0, 3960.0, 4010.0], "open_interest": [20000, 500, 9000, 30000],
    })
    maturity = pd.Series(pd.to_datetime(["2018-01-19", "2018-02-23", "2018-03-16", "2017-12-15"]),
                         index=["IF1801", "IF1802", "IF1803", "IF1712"])
    out = main_sub_carry(daily, maturity, pd.Series({day: "IF1801"}))
    row = out.iloc[0]
    assert row["to_contract"] == "IF1803"                       # 晚于主力到期的合约里持仓最大，不看更早到期的
    assert row["yield"] == pytest.approx(40 / 3960)
    assert row["annualized_yield"] == pytest.approx(40 / 3960 * 365 / 56)


# --- 股指熊市空头（唯一保留的股指策略） ---

import numpy as np
import pandas as pd
import pytest

from strategies.index_bear_reversal import BOOK as BEAR_BOOK
from strategies.index_catalog import INDEX_FROZEN_BOOKS
from strategies.index_close_location import SYMBOLS, market_close_location


def test_index_catalog_keeps_only_the_bear_short_book_and_clv_uses_the_day_range():
    assert INDEX_FROZEN_BOOKS == (BEAR_BOOK,)
    index = pd.bdate_range("2016-01-13", periods=2)
    bars = {"loww": pd.DataFrame(0.0, index=index, columns=SYMBOLS),
            "highw": pd.DataFrame(10.0, index=index, columns=SYMBOLS),
            "closew": pd.DataFrame([[2.0] * 3, [9.0] * 3], index=index, columns=SYMBOLS)}
    assert market_close_location(bars, index).tolist() == pytest.approx([0.2, 0.9])


def test_bear_short_waits_for_the_next_open_and_keeps_a_weak_close_long():
    from strategies.index_bear_reversal import THRESHOLD, bear_reversal_targets
    INDEX_SYMBOLS = SYMBOLS

    index = pd.bdate_range("2016-01-04", periods=140)
    level = pd.Series(np.linspace(200.0, 60.0, len(index)), index=index)
    close = pd.DataFrame({symbol: level for symbol in INDEX_SYMBOLS})
    high = close + 8.0
    low = close - 2.0
    high.iloc[126] = close.iloc[126] + 1.0
    low.iloc[126] = close.iloc[126] - 9.0
    high.iloc[50] = close.iloc[50] + 1.0
    low.iloc[50] = close.iloc[50] - 9.0
    bars = {"close": close, "closew": close.copy(), "highw": high, "loww": low, "open": close * 0 + 100.0, "openw": close * 0 + 100.0}
    market = ((close - low) / (high - low)).mean(axis=1)

    targets = bear_reversal_targets(market, bars)

    assert THRESHOLD == 0.4
    assert targets.iloc[51, 0] == 0.0
    assert targets.iloc[126, 0] == 1.0
    assert targets.iloc[127, 0] == -1.0
    moved = close.copy()
    moved.iloc[-1] = 1.0
    unchanged = bear_reversal_targets(market, {**bars, "close": moved, "closew": moved})
    pd.testing.assert_frame_equal(targets, unchanged)


def test_vol_scale_shrinks_a_wild_market_and_ignores_the_unfinished_open():
    from strategies.index_bear_reversal import bear_reversal_targets

    index = pd.bdate_range("2016-01-04", periods=160)
    level = pd.Series(np.linspace(200.0, 40.0, len(index)), index=index)
    close = pd.DataFrame({symbol: level for symbol in ("IF", "IH", "IC")})
    high = close + 8.0
    low = close - 2.0
    shock = np.array([0.08 if i % 2 == 0 else -0.07 for i in range(len(index))])
    path = 100.0 * np.cumprod(1.0 + shock)
    opened = pd.DataFrame({symbol: path for symbol in ("IF", "IH", "IC")}, index=index)
    bars = {"close": close, "closew": close.copy(), "highw": high, "loww": low, "open": opened, "openw": opened.copy()}
    market = ((close - low) / (high - low)).mean(axis=1)
    targets = bear_reversal_targets(market, bars)
    assert targets.iloc[-1, 0] > 0
    assert targets.iloc[-1, 0] < 0.5
    bumped = opened.copy()
    bumped.iloc[-1] *= 2
    pd.testing.assert_frame_equal(
        targets,
        bear_reversal_targets(market, {**bars, "open": bumped, "openw": bumped}),
    )

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

