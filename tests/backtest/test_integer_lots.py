"""目标仓位整数手账本：只成交差额、换月、缺价、熔断、做空、入选平分。合成数据。"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import TARGET_CAP, max_drawdown, simulate_target_lots

FRAMES = (
    "open_price", "close_price", "adjusted_open", "adjusted_open_delta", "adjusted_high",
    "adjusted_low", "rolls", "contract_codes", "open_fee", "close_fee", "roll_close_fee", "slippage",
)


def _inputs(weights, delta=0.0, fee=0.0, symbols=("CU",)):
    """每列一个品种的目标权重；价格恒为 100，乘数 10，初始资金 1 万。"""
    if not isinstance(weights, dict):
        weights = {symbols[0]: weights}
    days = len(next(iter(weights.values())))
    targets = pd.DataFrame(weights, index=pd.bdate_range("2020-01-02", periods=days))[list(symbols)]
    index, columns = targets.index, targets.columns

    def frame(value, dtype=None):
        return pd.DataFrame(value, index=index, columns=columns, dtype=dtype)

    inputs = {
        "open_price": frame(100.0), "close_price": frame(100.0), "adjusted_open": frame(100.0),
        "adjusted_open_delta": frame(float(delta)), "adjusted_high": frame(101.0),
        "adjusted_low": frame(99.0), "rolls": frame(False), "contract_codes": frame("CU2002", object),
        "universe": {2020: list(columns)}, "multipliers": {s: 10.0 for s in columns},
        "open_fee": frame(float(fee)), "close_fee": frame(float(fee)),
        "roll_close_fee": frame(float(fee)), "slippage": frame(0.0),
        "initial_capital": 10_000.0, "margin_rate": 0.20, "exposure_limit": 1.0, "drawdown_limit": 0.20,
    }
    return targets, inputs


def test_trade_only_the_difference_and_hold_when_the_weight_is_unchanged():
    targets, inputs = _inputs([0.0, 1.0, 1.0, 0.5, 0.5, 0.0], fee=0.001)
    daily, positions, events = simulate_target_lots(targets, **inputs)
    # 1.0 × 10000 / (100 × 10) = 10 手；付过 10 元费用后 0.5 × 9990 / 1000 向下取整为 4 手
    assert positions.set_index("交易日")["手数"].tolist() == [10, 10, 4, 4]
    assert daily["开仓笔数"].tolist() == [0, 1, 0, 0, 0, 0]
    assert daily["平仓笔数"].tolist() == [0, 0, 0, 1, 0, 1]
    assert daily["交易费用"].tolist() == pytest.approx([0, 10.0, 0, 6.0, 0, 4.0])
    assert events["事件"].tolist() == ["开仓", "减仓", "平仓"]


def test_pnl_is_lots_times_multiplier_times_adjusted_open_change():
    targets, inputs = _inputs([1.0, 1.0, 1.0])
    inputs["adjusted_open_delta"]["CU"] = [1.0, 2.0, 3.0]
    daily, _, _ = simulate_target_lots(targets, **inputs)
    assert daily["毛盈亏"].tolist() == pytest.approx([100.0, 200.0, 300.0])
    assert daily["保证金需求_20pct_assumption"].iloc[0] == pytest.approx(2_000.0)
    assert daily["净值"].iloc[-1] == pytest.approx(1.06)


def test_roll_charges_old_close_and_new_open_and_updates_the_contract():
    targets, inputs = _inputs([1.0, 1.0, 1.0])
    inputs["rolls"].iloc[1, 0] = True
    inputs["contract_codes"].iloc[1:, 0] = "CU2003"
    inputs["close_price"].iloc[0, 0] = 110.0
    inputs["open_price"].iloc[1, 0] = 120.0
    inputs["open_fee"].iloc[1, 0] = 0.002
    inputs["roll_close_fee"].iloc[1, 0] = 0.001
    daily, positions, events = simulate_target_lots(targets, **inputs)
    # 旧合约 10 手按前一日收盘 110 付 0.1%，新合约按开盘 120 付 0.2%：11 + 24 = 35
    assert daily["换月笔数"].iloc[1] == 1
    assert daily["交易费用"].iloc[1] == pytest.approx(35.0)
    assert positions.loc[positions["交易日"] == daily.index[1], "合约"].iloc[0] == "CU2003"
    assert "CU2002 -> CU2003" in events.loc[events["事件"] == "换月", "说明"].iloc[0]


def test_missing_mark_while_holding_fails_instead_of_counting_zero_pnl():
    targets, inputs = _inputs([1.0, 1.0, 1.0])
    inputs["adjusted_open_delta"].iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match="持仓期间复权开盘变动缺失"):
        simulate_target_lots(targets, **inputs)


def test_locked_day_blocks_the_order_and_retries_next_day():
    targets, inputs = _inputs([1.0, 1.0, 1.0, 1.0])
    inputs["adjusted_high"].iloc[0, 0] = inputs["adjusted_low"].iloc[0, 0]
    daily, _, _ = simulate_target_lots(targets, **inputs)
    assert daily["受阻委托数"].iloc[0] == 1 and daily["开仓笔数"].iloc[1] == 1


def test_names_outside_the_year_pool_are_never_held():
    targets, inputs = _inputs([1.0, 1.0])
    inputs["universe"] = {2020: ["AL"]}
    _, positions, _ = simulate_target_lots(targets, **inputs)
    assert positions.empty


def test_inputs_must_share_the_target_index_and_columns():
    targets, inputs = _inputs([1.0, 1.0])
    inputs["open_fee"] = inputs["open_fee"].iloc[:1]
    with pytest.raises(ValueError, match="open_fee"):
        simulate_target_lots(targets, **inputs)


def test_breaker_without_halt_days_waits_for_next_year():
    targets, inputs = _inputs([1.0] * 5, delta=-30.0)
    daily, _, events = simulate_target_lots(targets, **inputs)
    assert "回撤熔断清仓触发" in set(events["事件"])
    assert daily["持仓总手数"].iloc[-1] == 0 and daily["停止新开仓"].iloc[-1]


def test_breaker_with_halt_days_pauses_then_reopens_and_true_drawdown_ignores_the_reset():
    targets, inputs = _inputs([0.0] + [1.0] * 15)
    inputs["adjusted_open_delta"].iloc[1, 0] = -20.0
    inputs["drawdown_limit"] = 0.15
    daily, _, events = simulate_target_lots(targets, **inputs, halt_days=10)
    assert "暂停 10 个交易日" in events.loc[events["事件"] == "回撤熔断清仓触发", "说明"].iloc[0]
    assert daily["停止新开仓"].iloc[2:12].all() and not bool(daily["停止新开仓"].iloc[12])
    assert daily["开仓笔数"].iloc[12] == 1
    # 恢复当日熔断口径把峰值重置，真实回撤仍从 1 万算起
    assert daily["熔断口径回撤"].iloc[-1] == pytest.approx(0.0)
    assert max_drawdown(daily["权益"], 10_000.0) == pytest.approx(-0.2)


def test_weights_outside_the_cap_are_rejected():
    for bad in (-TARGET_CAP - 0.1, TARGET_CAP + 0.1, float("nan")):
        targets, inputs = _inputs([0.0, bad])
        with pytest.raises(ValueError, match="目标权重"):
            simulate_target_lots(targets, **inputs)


def test_short_loses_when_the_open_rises_and_flip_pays_both_sides():
    targets, inputs = _inputs([0.0, -1.0, -1.0, 0.0], delta=1.0, fee=0.001)
    daily, positions, events = simulate_target_lots(targets, **inputs)
    assert positions["手数"].tolist() == [-10, -10]
    assert daily["毛盈亏"].tolist() == pytest.approx([0.0, -100.0, -100.0, 0.0])
    assert events["事件"].tolist() == ["开空", "平空"]
    targets, inputs = _inputs([0.0, 1.0, -1.0], fee=0.001)
    daily, positions, events = simulate_target_lots(targets, **inputs)
    assert positions["手数"].tolist() == [10, -9]
    assert events["事件"].tolist() == ["开仓", "平仓", "开空"]


def test_weight_above_one_scales_the_pool_budget():
    targets, inputs = _inputs([0.0, 2.0, 2.0])
    _, positions, _ = simulate_target_lots(targets, **inputs)
    assert positions["手数"].tolist() == [20, 20]               # 2 × 1.0 × 10000 / (100 × 10)


def test_pool_budget_divides_by_pool_width_and_selected_by_names_with_a_target():
    targets, inputs = _inputs({"CU": [0.0, 1.0, 1.0], "AL": [0.0, 0.0, 0.0]}, symbols=("CU", "AL"))
    _, pooled, _ = simulate_target_lots(targets, **inputs)
    _, selected, _ = simulate_target_lots(targets, **inputs, budget_count="selected")
    assert pooled.loc[pooled["品种"] == "CU", "手数"].iloc[0] == 5      # 10000 / 2 / 1000
    assert selected.loc[selected["品种"] == "CU", "手数"].iloc[0] == 10  # 10000 / 1 / 1000
    assert selected.loc[selected["品种"] == "AL"].empty


def test_selected_budget_resplits_every_holding_when_the_count_changes():
    weights = {"CU": [1.0, 1.0, 1.0, 1.0], "AL": [0.0, 1.0, 1.0, 0.0]}
    targets, inputs = _inputs(weights, symbols=("CU", "AL"))
    daily, positions, events = simulate_target_lots(targets, **inputs, budget_count="selected")
    lots = positions.pivot(index="交易日", columns="品种", values="手数").fillna(0)
    # 一个入选时 CU 拿满 10 手；两个入选时各 5 手，CU 减 5；回到一个时 CU 再加回 10
    assert lots["CU"].tolist() == [10, 5, 5, 10]
    assert lots["AL"].tolist() == [0, 5, 5, 0]
    assert daily["敞口倍数"].max() == pytest.approx(1.0)
    pooled_daily, _, _ = simulate_target_lots(targets, **inputs)
    assert pooled_daily["敞口倍数"].max() == pytest.approx(1.0)
    assert list(events["事件"]) == ["开仓", "减仓", "开仓", "加仓", "平仓"]   # 同一天先 CU 后 AL


def test_selected_budget_holds_when_neither_the_count_nor_the_weight_changes():
    weights = {"CU": [1.0, 1.0, 1.0], "AL": [1.0, 1.0, 1.0]}
    targets, inputs = _inputs(weights, delta=5.0, symbols=("CU", "AL"))
    daily, _, _ = simulate_target_lots(targets, **inputs, budget_count="selected")
    assert daily["开仓笔数"].tolist() == [2, 0, 0]
    assert daily["平仓笔数"].sum() == 0
