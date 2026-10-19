import numpy as np
import pandas as pd
import pytest

from backtest.engine import simulate_integer_lots


def _inputs(days=4):
    index = pd.bdate_range("2020-01-02", periods=days)
    columns = pd.Index(["CU"])
    open_price = pd.DataFrame(100.0, index=index, columns=columns)
    close_price = pd.DataFrame(100.0, index=index, columns=columns)
    adjusted_open = pd.DataFrame(100.0, index=index, columns=columns)
    high = pd.DataFrame(101.0, index=index, columns=columns)
    low = pd.DataFrame(99.0, index=index, columns=columns)
    rolls = pd.DataFrame(False, index=index, columns=columns)
    contract = pd.DataFrame("CU2002", index=index, columns=columns)
    fees = pd.DataFrame(0.0, index=index, columns=columns)
    entries = pd.DataFrame(False, index=index, columns=columns)
    delta = pd.DataFrame(0.0, index=index, columns=columns)
    return {
        "entries": entries,
        "open_price": open_price,
        "close_price": close_price,
        "adjusted_open": adjusted_open,
        "adjusted_open_delta": delta,
        "adjusted_high": high,
        "adjusted_low": low,
        "rolls": rolls,
        "contract_codes": contract,
        "universe": {2020: ["CU"]},
        "multipliers": {"CU": 10.0},
        "open_fee": fees.copy(),
        "close_fee": fees.copy(),
        "roll_close_fee": fees.copy(),
        "slippage": fees.copy(),
        "initial_capital": 10_000.0,
        "margin_rate": 0.20,
        "holding_days": 2,
        "exposure_limit": 1.0,
        "drawdown_limit": 0.20,
    }


def test_integer_lots_use_floor_sizing_and_mark_to_adjusted_price_changes():
    inputs = _inputs()
    inputs["entries"].iloc[0, 0] = True
    inputs["adjusted_open_delta"]["CU"] = [1.0, 2.0, 3.0, 0.0]

    daily, positions, events = simulate_integer_lots(**inputs)

    assert positions.iloc[0]["手数"] == 10
    assert positions.iloc[0]["合约"] == "CU2002"
    assert daily.iloc[0]["保证金需求_20pct_assumption"] == pytest.approx(2_000.0)
    assert daily.iloc[0]["毛盈亏"] == pytest.approx(100.0)
    assert daily.iloc[1]["毛盈亏"] == pytest.approx(200.0)
    assert daily.iloc[2]["平仓笔数"] == 1
    assert daily.iloc[2]["净值"] == pytest.approx(1.03)
    assert events["事件"].tolist() == ["开仓", "到期平仓"]


def test_fractional_commission_preserves_cash_and_nav_precision():
    inputs = _inputs()
    inputs["entries"].iloc[0, 0] = True
    inputs["adjusted_open_delta"]["CU"] = [1.0, 0.0, 0.0, 0.0]
    inputs["open_fee"].iloc[0, 0] = 0.00012345
    inputs["slippage"] = inputs["slippage"].copy()
    inputs["slippage"].iloc[0, 0] = 0.0

    daily, _positions, _events = simulate_integer_lots(**inputs)

    assert daily.iloc[0]["交易费用"] == pytest.approx(1.2345)
    assert daily.iloc[0]["权益"] == pytest.approx(10_098.7655)
    assert np.allclose(
        daily["期初权益"] - daily["交易费用"] + daily["毛盈亏"],
        daily["权益"],
        rtol=0,
        atol=1e-9,
    )
    assert (1.0 + daily["净收益"]).prod() == pytest.approx(
        daily["权益"].iloc[-1] / inputs["initial_capital"],
    )


def test_single_price_lock_proxy_delays_scheduled_exit():
    inputs = _inputs()
    inputs["holding_days"] = 1
    inputs["entries"].iloc[0, 0] = True
    inputs["adjusted_open_delta"]["CU"] = [1.0, 0.0, 0.0, 0.0]
    inputs["adjusted_high"].iloc[1, 0] = 100.0
    inputs["adjusted_low"].iloc[1, 0] = 100.0

    daily, positions, events = simulate_integer_lots(**inputs)

    assert daily.iloc[1]["受阻委托数"] == 1
    assert positions.loc[positions["交易日"] == daily.index[1], "手数"].iloc[0] == 10
    assert daily.iloc[2]["平仓笔数"] == 1
    assert "退出受阻" in events["事件"].tolist()


def test_roll_charges_old_close_and_new_open_costs_and_updates_contract():
    inputs = _inputs(days=3)
    inputs["entries"].iloc[0, 0] = True
    inputs["rolls"].iloc[1, 0] = True
    inputs["contract_codes"].iloc[1, 0] = "CU2003"
    inputs["close_price"].iloc[0, 0] = 110.0
    inputs["open_price"].iloc[1, 0] = 120.0
    inputs["open_fee"].iloc[1, 0] = 0.002
    inputs["roll_close_fee"].iloc[1, 0] = 0.001
    inputs["adjusted_open_delta"]["CU"] = [0.0, 2.0, 0.0]

    daily, positions, events = simulate_integer_lots(**inputs)

    assert daily.iloc[1]["换月笔数"] == 1
    assert daily.iloc[1]["交易费用"] == pytest.approx(35.0)
    assert daily.iloc[1]["毛盈亏"] == pytest.approx(200.0)
    assert positions.loc[positions["交易日"] == daily.index[1], "合约"].iloc[0] == "CU2003"
    roll = events.loc[events["事件"] == "换月"].iloc[0]
    assert "CU2002 -> CU2003" in roll["说明"]
    assert "前一交易日收盘" in roll["说明"]


def test_missing_held_position_mark_fails_instead_of_counting_zero_pnl():
    inputs = _inputs()
    inputs["entries"].iloc[0, 0] = True
    inputs["adjusted_open_delta"].iloc[1, 0] = np.nan

    with pytest.raises(ValueError, match="持仓期间复权开盘变动缺失"):
        simulate_integer_lots(**inputs)


def test_missing_open_and_close_block_exit_until_a_price_is_available():
    inputs = _inputs(days=3)
    inputs["entries"].iloc[0, 0] = True
    inputs["holding_days"] = 1
    inputs["open_price"].iloc[1, 0] = np.nan
    inputs["close_price"].iloc[1, 0] = np.nan
    inputs["adjusted_open_delta"]["CU"] = [0.0, 1.0, 1.0]

    daily, positions, events = simulate_integer_lots(**inputs)

    assert daily.iloc[1]["受阻委托数"] == 1
    assert daily.iloc[1]["毛盈亏"] == pytest.approx(100.0)
    assert positions.loc[positions["交易日"] == daily.index[1], "主力原始开盘价"].iloc[0] == 100.0
    assert daily.iloc[2]["平仓笔数"] == 1
    assert "退出受阻" in events["事件"].tolist()


def test_holding_period_and_position_carry_across_calendar_year():
    inputs = _inputs()
    index = pd.to_datetime([
        "2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03",
    ])
    for frame in inputs.values():
        if isinstance(frame, pd.DataFrame):
            frame.index = index
    inputs["universe"] = {2019: ["CU"], 2020: ["CU"]}
    inputs["entries"].iloc[0, 0] = True
    inputs["entries"].iloc[2, 0] = True
    inputs["holding_days"] = 3
    inputs["adjusted_open_delta"]["CU"] = [0.0, 0.0, 0.0, 0.0]

    daily, positions, events = simulate_integer_lots(**inputs)

    assert daily.loc[index[2], "持仓总手数"] == 10
    assert daily.loc[index[3], "平仓笔数"] == 1
    assert not (positions["交易日"] == index[3]).any()
    assert events.loc[events["事件"] == "到期平仓", "交易日"].tolist() == [index[3]]


def test_open_position_at_end_of_available_data_is_not_synthetically_liquidated():
    inputs = _inputs(days=1)
    inputs["entries"].iloc[0, 0] = True
    inputs["holding_days"] = 1

    daily, positions, events = simulate_integer_lots(**inputs)

    assert daily.iloc[-1]["持仓总手数"] == 10
    assert positions.iloc[-1]["手数"] == 10
    assert events["事件"].tolist() == ["开仓"]


def test_drawdown_breaker_stops_new_entries_and_liquidates_at_20_percent():
    inputs = _inputs()
    inputs["entries"].iloc[0:2, 0] = True
    inputs["adjusted_open_delta"]["CU"] = [-21.0, 0.0, 0.0, 0.0]

    daily, _positions, events = simulate_integer_lots(**inputs)

    assert daily.iloc[1]["停止新开仓"]
    assert daily.iloc[1]["平仓笔数"] == 1
    assert "回撤熔断清仓触发" in events["事件"].tolist()
    assert events.loc[events["事件"] == "开仓", "交易日"].tolist() == [daily.index[0]]


def test_drawdown_freeze_resets_next_year_without_repeated_trigger():
    inputs = _inputs()
    index = pd.to_datetime([
        "2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03",
    ])
    for key in (
        "entries", "open_price", "close_price", "adjusted_open",
        "adjusted_open_delta", "adjusted_high", "adjusted_low", "rolls",
        "contract_codes", "open_fee", "close_fee", "roll_close_fee", "slippage",
    ):
        inputs[key].index = index
    inputs["universe"] = {2019: ["CU"], 2020: ["CU"]}
    inputs["entries"].iloc[0, 0] = True
    inputs["entries"].iloc[2, 0] = True
    inputs["adjusted_open_delta"]["CU"] = [-11.0, 0.0, 0.0, 0.0]
    inputs["holding_days"] = 1
    inputs["drawdown_limit"] = 0.10

    daily, _positions, events = simulate_integer_lots(**inputs)

    assert daily.loc[index[1], "停止新开仓"]
    assert daily.loc[index[2], "开仓笔数"] == 1
    assert events.loc[events["事件"] == "开仓", "交易日"].tolist() == [
        index[0], index[2],
    ]


def test_short_entries_are_rejected_by_long_only_simulator():
    inputs = _inputs()
    inputs["entries"] = inputs["entries"].astype("int8")
    inputs["entries"].iloc[0, 0] = -1

    with pytest.raises(ValueError, match="纯多头入场表"):
        simulate_integer_lots(**inputs)


def test_short_entries_use_signed_pnl_and_gross_exposure():
    inputs = _inputs()
    inputs["entries"] = inputs["entries"].astype("int8")
    inputs["entries"].iloc[0, 0] = -1
    inputs["adjusted_open_delta"]["CU"] = [1.0, 2.0, 3.0, 0.0]

    daily, positions, events = simulate_integer_lots(**inputs, allow_shorts=True)

    assert positions.iloc[0]["手数"] == -10
    assert positions.iloc[0]["方向"] == "空"
    assert positions.iloc[0]["名义敞口"] == pytest.approx(10_000.0)
    assert daily.iloc[0]["保证金需求_20pct_assumption"] == pytest.approx(2_000.0)
    assert daily.iloc[0]["毛盈亏"] == pytest.approx(-100.0)
    assert daily.iloc[1]["毛盈亏"] == pytest.approx(-200.0)
    assert daily.iloc[2]["平仓笔数"] == 1
    assert daily.iloc[2]["净值"] == pytest.approx(0.97)
    assert "空头" in events.iloc[0]["说明"]


def test_weighted_short_entry_scales_integer_lots_and_preserves_direction():
    inputs = _inputs()
    inputs["entries"] = inputs["entries"].astype("float64")
    inputs["entries"].iloc[0, 0] = -0.5
    inputs["adjusted_open_delta"]["CU"] = [1.0, 0.0, 0.0, 0.0]

    daily, positions, _events = simulate_integer_lots(
        **inputs,
        allow_shorts=True,
        weighted_entries=True,
    )

    assert positions.iloc[0]["手数"] == -5
    assert positions.iloc[0]["名义敞口"] == pytest.approx(5_000.0)
    assert daily.iloc[0]["毛盈亏"] == pytest.approx(-50.0)


def test_exposure_limit_scales_integer_lot_target():
    inputs = _inputs()
    inputs["entries"].iloc[0, 0] = True
    inputs["exposure_limit"] = 1.5

    daily, positions, _events = simulate_integer_lots(**inputs)

    assert positions.iloc[0]["手数"] == 15
    assert daily.iloc[0]["敞口倍数"] == pytest.approx(1.5)


def test_rejects_non_aligned_fee_inputs():
    inputs = _inputs()
    inputs["open_fee"] = inputs["open_fee"].iloc[:-1]
    with pytest.raises(ValueError, match="open_fee"):
        simulate_integer_lots(**inputs)


def test_locked_proxy_requires_finite_daily_range():
    inputs = _inputs()
    inputs["entries"].iloc[0, 0] = True
    inputs["adjusted_high"].iloc[0, 0] = np.nan
    inputs["adjusted_low"].iloc[0, 0] = np.nan
    daily, _positions, events = simulate_integer_lots(**inputs)
    assert daily.iloc[0]["开仓笔数"] == 1
    assert "开仓" in events["事件"].tolist()
