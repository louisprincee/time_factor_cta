"""整数手数期货账本、绩效统计、行情装载和命令行入口。

    python -m backtest.engine --research                     # 商品 2011–2021
    python -m backtest.engine --oos                          # 商品 2022–2025
    python -m backtest.engine --market index --research      # 股指 2015–2021
    python -m backtest.engine --market index --oos           # 股指 2022–2025
    python -m backtest.engine <策略> --start 2022 --end 2023

研究期和样本外用同一条命令。行情仍按 research/ 与 oos/ 分目录读取。
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.commodity_original import FACTORS
from tfcta import config as C
from tfcta.costs import (
    INDEX_MULTIPLIER, MULTIPLIER, contract_multiplier_overrides, estimate_ticks, fee_tables,
    index_open_limits, load_fees, load_index_fees, load_index_oos_fees, load_ticks,
    require_multipliers, slippage_tables,
)
from tfcta.data import (
    add_intraday_coords, build_universe, daily_bars, find_shard, list_shards,
    load_minutes, load_roll_calendar, load_shard, load_universe, oos_universe,
    read_frame,
)
from tfcta.factors import SignalSet, load, raw_price_path, timestamp_factors

INITIAL_CAPITAL = 100_000_000.0
# 股指策略固定1000万、开盘后30分钟L1。研究期与样本外共用这个口径。
INDEX_DEFAULT_CAPITAL = 10_000_000.0
INDEX_EXECUTIONS = ("l1_30m",)
INDEX_DEFAULT_EXECUTION = "l1_30m"
MARGIN_RATE = 0.20
TARGET_CAP = 3.0
RESEARCH_YEARS = range(C.EARLY_START.year, C.RESEARCH_END.year + 1)   # 用户 2026-10-23：研究账本从 2011 起
OOS_COLUMNS = ["trading_date", "open", "openw", "close", "closew", "highw", "loww",
               "volume", "total_turnover"]
WARMUP_DAYS = 320          # 持续期阈值要 250 日分钟历史，2022 起现算因子时留足余量


# --------------------------------------------------------------------------
# 目标仓位账本
# --------------------------------------------------------------------------
def simulate_target_lots(
    targets: pd.DataFrame,
    open_price: pd.DataFrame,
    close_price: pd.DataFrame,
    adjusted_open: pd.DataFrame,
    adjusted_open_delta: pd.DataFrame,
    adjusted_high: pd.DataFrame,
    adjusted_low: pd.DataFrame,
    rolls: pd.DataFrame,
    contract_codes: pd.DataFrame,
    universe: dict[int, list[str]],
    multipliers: dict[str, float],
    open_fee: pd.DataFrame,
    close_fee: pd.DataFrame,
    roll_close_fee: pd.DataFrame,
    slippage: pd.DataFrame,
    *,
    initial_capital: float,
    margin_rate: float,
    exposure_limit: float,
    drawdown_limit: float,
    halt_days: int,
    contract_multipliers: dict[str, dict[str, float]] | None = None,
    open_limits: pd.DataFrame | None = None,
    open_slippage: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按目标仓位交易整数手，只成交目标与现有手数的差额。负权重做空。

    targets[t, s] 是 t 日开盘的目标权重（-TARGET_CAP..TARGET_CAP）。名义金额用权重的绝对值：
    |权重| × 敞口上限 × 期初权益 / 当年池宽。等权满仓时每个品种的权重绝对值为 1。
    手数与权重同号，盈亏 = 有符号手数 × 乘数 × 复权开盘变动。权重不变就继续持有，不因价格或权益变化调整手数。
    无开盘价或单价锁定代理当日不成交，次日重试。整本账一个峰值回撤熔断：触发后清仓。
    熔断后停 halt_days 个交易日（含触发日），恢复当日把峰值改成当时权益。
    contract_multipliers 给出乘数不等于品种现行乘数的合约（如 RU1207 以前 5 吨）。持仓按所持合约的乘数算盈亏和名义；
    换月到乘数不同的合约时按名义不变换算新手数（向下取整）。
    open_limits[t, s] 是 t 日该品种最多能开的手数（交易所日内开仓限额；不限为 inf）。新开、加仓、换月的新合约开仓、
    多空翻转的新方向都算开仓，平仓不受限。超过额度只开到额度，之后每天按额度补到目标手数；只会少持仓。
    open_slippage {"buy", "sell"} 给出按开盘价成交的买入、卖出各自的成本率（可为负）；不给时两边都用 slippage。
    可再给 "auction_lots"：当日该品种前这么多手（各开盘成交腿合计）按 slippage 成交（集合竞价），其余按买卖成本率。
    或给 "auction_buy_lots" 和 "auction_buy_rate"：只有买入腿先用这个额度，按 auction_buy_rate 成交；卖出不进竞价。
    换月旧合约按前收平仓，始终用 slippage。
    """
    frames = {
        "open_price": open_price, "close_price": close_price,
        "adjusted_open": adjusted_open, "adjusted_open_delta": adjusted_open_delta,
        "adjusted_high": adjusted_high, "adjusted_low": adjusted_low,
        "rolls": rolls, "contract_codes": contract_codes,
        "open_fee": open_fee, "close_fee": close_fee,
        "roll_close_fee": roll_close_fee, "slippage": slippage,
    }
    index, columns = targets.index, targets.columns
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("交易日索引必须唯一且按升序排列")
    if open_limits is not None:
        frames["open_limits"] = open_limits
    if open_slippage is not None:
        frames["open_slippage_buy"], frames["open_slippage_sell"] = open_slippage["buy"], open_slippage["sell"]
        if "auction_lots" in open_slippage:
            frames["auction_lots"] = open_slippage["auction_lots"]
            auction = open_slippage["auction_lots"].to_numpy(dtype="float64")
            if (
                not np.isfinite(auction).all()
                or (auction < 0).any()
                or not np.equal(auction, np.floor(auction)).all()
            ):
                raise ValueError("集合竞价可成交手数必须是非负有限整数")
        if "auction_buy_lots" in open_slippage:
            if "auction_lots" in open_slippage or "auction_buy_rate" not in open_slippage:
                raise ValueError("auction_buy_lots 要配 auction_buy_rate，且不能和 auction_lots 同时给")
            frames["auction_buy_lots"] = open_slippage["auction_buy_lots"]
            frames["auction_buy_rate"] = open_slippage["auction_buy_rate"]
            lots = open_slippage["auction_buy_lots"].to_numpy(dtype="float64")
            rate = open_slippage["auction_buy_rate"].to_numpy(dtype="float64")
            if (
                not np.isfinite(lots).all() or (lots < 0).any()
                or not np.equal(lots, np.floor(lots)).all()
                or not np.isfinite(rate[lots > 0]).all()
            ):
                raise ValueError("买入竞价额度必须是非负整数，有额度的格子必须有成本率")
    for name, frame in frames.items():
        if not frame.index.equals(index) or not frame.columns.equals(columns):
            raise ValueError(f"{name} 的日期或品种索引与目标仓位表不一致")
    values = targets.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (np.abs(values) > TARGET_CAP).any():
        raise ValueError(f"目标权重必须是 -{TARGET_CAP} 到 {TARGET_CAP} 之间的有限数")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("初始资金必须是正的有限数")
    if not 0 < margin_rate <= 1 or exposure_limit <= 0 or not 0 < drawdown_limit < 1:
        raise ValueError("保证金率、敞口上限或回撤熔断阈值无效")
    if not isinstance(halt_days, int) or isinstance(halt_days, bool) or halt_days < 1:
        raise ValueError("熔断停牌天数必须是正整数")
    for symbol in columns:
        multiplier = multipliers.get(symbol)
        if multiplier is None or not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f"{symbol} 缺少有效合约乘数")
    overrides = contract_multipliers or {}
    for symbol, table in overrides.items():
        if any(not np.isfinite(v) or v <= 0 for v in table.values()):
            raise ValueError(f"{symbol} 的合约乘数无效")

    high, low = adjusted_high.to_numpy(dtype="float64"), adjusted_low.to_numpy(dtype="float64")
    locked = np.isfinite(high) & np.isfinite(low) & np.isclose(high, low, rtol=0.0, atol=1e-10)
    lots, applied, desired = dict.fromkeys(columns, 0), dict.fromkeys(columns, 0.0), dict.fromkeys(columns, 0)
    if open_limits is not None:
        limit_values = open_limits.to_numpy(dtype="float64")
        if np.isnan(limit_values).any() or (limit_values < 0).any():
            raise ValueError("开仓限额必须是非负数，不限用 inf")
    contracts, pending_roll = dict.fromkeys(columns), set()
    last_price, last_close = dict.fromkeys(columns, np.nan), dict.fromkeys(columns, np.nan)
    equity = peak_equity = float(initial_capital)
    halt_until_year, halt_until_i, previous_drawdown = 0, -1, 0.0
    daily_rows, position_rows, events = [], [], []

    def event(day, kind, symbol=None, quantity=0, price=np.nan, detail=""):
        events.append({
            "交易日": day, "事件": kind, "品种": symbol, "手数": int(quantity),
            "成交代理价": float(price) if np.isfinite(price) else np.nan,
            "合约": contracts.get(symbol) if symbol else None, "说明": detail,
        })

    def mult_of(symbol, contract):
        return float(overrides.get(symbol, {}).get(contract, multipliers[symbol]))

    def held_mult(symbol):
        return mult_of(symbol, contracts[symbol])

    def charge(table, day, symbol, quantity, price, description, mult):
        rate = float(table.at[day, symbol])
        if not np.isfinite(rate) or not np.isfinite(price) or price <= 0:
            raise ValueError(f"{day.date()} {symbol} 的{description}费用或价格缺失")
        return quantity * mult * price * rate

    def open_cost(direction, day, symbol, quantity, price, description, mult):
        """按开盘价成交那一腿的滑点；集合竞价额度先用完，再按买卖成本率。"""
        nonlocal opening_slippage_cost, auction_slippage_cost, other_open_slippage_cost
        nonlocal opening_tick_cost_base
        if quantity <= 0:
            return 0.0
        opening_lots[symbol] += int(quantity)
        opening_tick_cost_base += charge(
            slippage, day, symbol, quantity, price, f"{description}一跳基准", mult
        )
        if open_slippage is None:
            cost = charge(slippage, day, symbol, quantity, price, description, mult)
            opening_slippage_cost += cost
            other_open_slippage_cost += cost
            return cost
        room, auction_table = 0, slippage
        if "auction_lots" in open_slippage:
            room = max(0, int(open_slippage["auction_lots"].at[day, symbol]) - auction_used[symbol])
        elif "auction_buy_lots" in open_slippage and direction > 0:
            room = max(0, int(open_slippage["auction_buy_lots"].at[day, symbol]) - auction_used[symbol])
            auction_table = open_slippage["auction_buy_rate"]
        first = min(int(quantity), room)
        auction_used[symbol] += first
        auction_cost = charge(auction_table, day, symbol, first, price, description, mult) if first else 0.0
        opening_slippage_cost += auction_cost
        auction_slippage_cost += auction_cost
        other_cost = 0.0
        if quantity > first:
            table = open_slippage["buy" if direction > 0 else "sell"]
            other_cost = charge(table, day, symbol, quantity - first, price, description, mult)
            opening_slippage_cost += other_cost
            other_open_slippage_cost += other_cost
        return auction_cost + other_cost

    def mark(day, symbol):
        price = open_price.at[day, symbol]
        return float(price) if np.isfinite(price) else last_price[symbol]

    for i, day in enumerate(index):
        day = pd.Timestamp(day)
        year = int(day.year)
        cost = opened = closed = rolled = blocked = capped = 0
        opened_today = dict.fromkeys(columns, 0)
        auction_used = dict.fromkeys(columns, 0)
        opening_lots = dict.fromkeys(columns, 0)
        opening_slippage_cost = auction_slippage_cost = other_open_slippage_cost = 0.0
        opening_tick_cost_base = 0.0
        start_equity = equity
        if i == halt_until_i:
            peak_equity = equity
        peak_equity = max(peak_equity, equity)
        drawdown_at_open = equity / peak_equity - 1.0
        if previous_drawdown > -drawdown_limit and drawdown_at_open <= -drawdown_limit:
            halt_until_i = i + halt_days
            event(day, "回撤熔断清仓触发", detail=f"峰值回撤 {drawdown_at_open:.2%}；清仓并暂停 {halt_days} 个交易日")
        for symbol in columns:
            raw_open = open_price.at[day, symbol]
            if np.isfinite(raw_open) and raw_open > 0:
                last_price[symbol] = float(raw_open)
            elif lots[symbol] and not np.isfinite(last_price[symbol]):
                raise ValueError(f"{day.date()} {symbol} 持仓开盘价缺失")
        margin_call = sum(
            abs(lots[s]) * held_mult(s) * mark(day, s) * margin_rate for s in columns if lots[s]
        ) > equity
        if margin_call:
            event(day, "保证金不足", detail="清仓")
        halted = year < halt_until_year or i < halt_until_i or margin_call
        members = universe.get(year, [])
        slots = len(members)

        for symbol in columns:
            raw_open = open_price.at[day, symbol]
            prior_close = last_close[symbol]
            if np.isfinite(close_price.at[day, symbol]):
                last_close[symbol] = float(close_price.at[day, symbol])
            tradable = bool(np.isfinite(raw_open) and raw_open > 0) and not bool(
                locked[i, columns.get_loc(symbol)]
            )
            if bool(rolls.at[day, symbol]) and lots[symbol]:
                pending_roll.add(symbol)
            target = 0.0 if halted or symbol not in members else float(targets.at[day, symbol])
            if not tradable:
                if (lots[symbol] and symbol in pending_roll) or target != applied[symbol]:
                    blocked += 1
                continue
            raw_open = float(raw_open)
            room = np.inf if open_limits is None else float(open_limits.at[day, symbol])
            if lots[symbol] and symbol in pending_roll:
                contract = contract_codes.at[day, symbol]
                if not isinstance(contract, str) or not contract:
                    raise ValueError(f"{day.date()} {symbol} 缺少新主力合约代码")
                if not np.isfinite(prior_close) or prior_close <= 0:
                    raise ValueError(f"{day.date()} {symbol} 换月缺少旧合约收盘价")
                quantity = abs(lots[symbol])
                old_mult, new_mult = held_mult(symbol), mult_of(symbol, contract)
                new_quantity = quantity if new_mult == old_mult else int(math.floor(quantity * old_mult / new_mult))
                desired[symbol] = int(np.sign(lots[symbol])) * new_quantity
                if new_quantity > room:
                    capped += 1
                    event(day, "开仓限额", symbol, new_quantity, raw_open,
                          f"换月要开 {new_quantity} 手，当日上限 {room:g} 手，只换 {int(room)} 手，其余之后补")
                    new_quantity = int(room)
                opened_today[symbol] += new_quantity
                cost += charge(roll_close_fee, day, symbol, quantity, prior_close, "换月平仓", old_mult)
                cost += charge(slippage, day, symbol, quantity, prior_close, "换月平仓滑点", old_mult)
                cost += charge(open_fee, day, symbol, new_quantity, raw_open, "换月开仓", new_mult)
                cost += open_cost(np.sign(lots[symbol]), day, symbol, new_quantity, raw_open, "换月开仓滑点", new_mult)
                detail = f"{contracts[symbol]} -> {contract}"
                if new_mult != old_mult:
                    detail += f"；乘数 {old_mult:g} -> {new_mult:g}，手数 {quantity} -> {new_quantity}"
                event(day, "换月", symbol, quantity, raw_open, detail)
                lots[symbol] = int(np.sign(lots[symbol])) * new_quantity
                contracts[symbol] = contract if new_quantity else None
                pending_roll.discard(symbol)
                rolled += 1
            if target == applied[symbol] and lots[symbol] == desired[symbol] and not (target == 0.0 and lots[symbol]):
                continue
            want = 0
            today_mult = mult_of(symbol, contract_codes.at[day, symbol])
            if target == applied[symbol] and target != 0.0:
                want = desired[symbol]                     # 目标没变，继续补前几天受限额没成交的手数
            elif slots and target > 0.0:
                budget = target * exposure_limit * max(start_equity, 0.0) / slots
                want = int(math.floor(budget / (raw_open * today_mult)))
            elif slots and target < 0.0:
                budget = -target * exposure_limit * max(start_equity, 0.0) / slots
                want = -int(math.floor(budget / (raw_open * today_mult)))
            current = lots[symbol]
            desired[symbol] = wanted = want
            left = room - opened_today[symbol]
            if current * want >= 0 and abs(want) > abs(current):
                opening = min(abs(want) - abs(current), left)
                want = current + int(np.sign(want)) * int(opening)
            elif current * want < 0:
                want = int(np.sign(want)) * int(min(abs(want), left))
            if want != wanted:
                capped += 1
                event(day, "开仓限额", symbol, abs(wanted - want), raw_open,
                      f"目标 {wanted} 手，当日上限 {room:g} 手，只做到 {want} 手，其余之后补")
            opened_today[symbol] += abs(want) if current * want < 0 else max(0, abs(want) - abs(current))
            if current >= 0 and want >= 0:
                change = want - current
                if current and not want:
                    cost += charge(close_fee, day, symbol, current, raw_open, "平仓", held_mult(symbol))
                    cost += open_cost(-1, day, symbol, current, raw_open, "平仓滑点", held_mult(symbol))
                    closed += 1
                    event(day, "清仓" if halted else "平仓", symbol, current, raw_open)
                elif change > 0:
                    contract = contract_codes.at[day, symbol]
                    if not isinstance(contract, str) or not contract:
                        raise ValueError(f"{day.date()} {symbol} 开仓缺少主力合约代码")
                    cost += charge(open_fee, day, symbol, change, raw_open, "开仓", today_mult)
                    cost += open_cost(1, day, symbol, change, raw_open, "开仓滑点", today_mult)
                    opened += 1
                    if not current:
                        contracts[symbol] = contract
                    event(day, "加仓" if current else "开仓", symbol, change, raw_open,
                          f"目标权重 {target:.2f}")
                elif change < 0:
                    cost += charge(close_fee, day, symbol, -change, raw_open, "减仓", held_mult(symbol))
                    cost += open_cost(-1, day, symbol, -change, raw_open, "减仓滑点", held_mult(symbol))
                    closed += 1
                    event(day, "减仓", symbol, -change, raw_open, f"目标权重 {target:.2f}")
                if target != 0.0 and not wanted:
                    event(day, "不足一手", symbol, detail=f"目标权重 {target:.2f}")
            else:
                def close_lots(quantity, kind, detail=""):
                    nonlocal cost, closed
                    cost += charge(close_fee, day, symbol, quantity, raw_open, "平仓", held_mult(symbol))
                    cost += open_cost(-np.sign(lots[symbol]), day, symbol, quantity, raw_open, "平仓滑点", held_mult(symbol))
                    closed += 1
                    event(day, kind, symbol, quantity, raw_open, detail)

                def open_lots(quantity, kind, detail):
                    nonlocal cost, opened
                    contract = contract_codes.at[day, symbol]
                    if not isinstance(contract, str) or not contract:
                        raise ValueError(f"{day.date()} {symbol} 开仓缺少主力合约代码")
                    cost += charge(open_fee, day, symbol, quantity, raw_open, "开仓", today_mult)
                    cost += open_cost(np.sign(want), day, symbol, quantity, raw_open, "开仓滑点", today_mult)
                    opened += 1
                    if not lots[symbol]:
                        contracts[symbol] = contract
                    event(day, kind, symbol, quantity, raw_open, detail)

                if current > 0 and want < 0:
                    close_lots(current, "清仓" if halted else "平仓")
                    current = 0
                    lots[symbol] = 0
                elif current < 0 and want >= 0:
                    close_lots(-current, "清仓" if halted else "平空")
                    current = 0
                    lots[symbol] = 0
                if want < 0 and current == 0:
                    if -want:
                        open_lots(-want, "开空", f"目标权重 {target:.2f}")
                    else:
                        event(day, "不足一手", symbol, detail=f"目标权重 {target:.2f}")
                elif want < 0 and current < 0:
                    change = want - current
                    if change < 0:
                        open_lots(-change, "加空", f"目标权重 {target:.2f}")
                    elif change > 0:
                        close_lots(change, "减空", f"目标权重 {target:.2f}")
                elif want > 0 and current == 0:
                    open_lots(want, "开仓", f"目标权重 {target:.2f}")
                if target < 0.0 and not wanted:
                    event(day, "不足一手", symbol, detail=f"目标权重 {target:.2f}")
            lots[symbol] = want
            applied[symbol] = target
            if not want:
                contracts[symbol] = None
                pending_roll.discard(symbol)

        gross_pnl = 0.0
        for symbol in columns:
            if lots[symbol]:
                delta = adjusted_open_delta.at[day, symbol]
                if not np.isfinite(delta) or not np.isfinite(adjusted_open.at[day, symbol]):
                    raise ValueError(f"{day.date()} {symbol} 持仓期间复权开盘变动缺失")
                gross_pnl += lots[symbol] * held_mult(symbol) * float(delta)
        equity = float(equity - cost + gross_pnl)
        writeoff = 0.0
        if equity <= 0:
            event(day, "账户权益耗尽", detail=f"本日损益 {gross_pnl - cost:.2f}")
            writeoff, equity = -equity, 0.0
            halt_until_year = max(halt_until_year, year + 1)
        peak_equity = max(peak_equity, equity)
        previous_drawdown = drawdown_at_open
        notional = sum(abs(lots[s]) * held_mult(s) * mark(day, s) for s in columns if lots[s])
        margin = notional * margin_rate
        daily_rows.append({
            "交易日": day, "期初权益": start_equity, "毛盈亏": gross_pnl,
            "交易费用": cost, "净盈亏": float(gross_pnl - cost + writeoff), "权益清零调整": writeoff,
            "权益": equity, "净值": equity / initial_capital,
            "熔断口径回撤": equity / peak_equity - 1.0 if peak_equity else -1.0,
            "名义敞口": notional, "敞口倍数": notional / equity if equity > 0 else np.nan,
            "保证金需求_20pct_assumption": margin,
            "保证金权益占用": margin / equity if equity > 0 else np.nan,
            "持仓品种数": sum(value != 0 for value in lots.values()),
            "持仓总手数": sum(abs(value) for value in lots.values()), "开仓笔数": opened,
            "开盘成交手数": sum(opening_lots.values()),
            "集合竞价成交手数": sum(auction_used.values()),
            "开盘滑点成本": opening_slippage_cost,
            "集合竞价滑点成本": auction_slippage_cost,
            "非竞价开盘滑点成本": other_open_slippage_cost,
            "开盘一跳基准费用": opening_tick_cost_base,
            "平仓笔数": closed, "换月笔数": rolled, "受阻委托数": blocked, "开仓受限数": capped,
            "停止新开仓": bool(halted), "强制清仓待执行": bool(halted),
            "保证金追缴": bool(margin_call),
        })
        for symbol in columns:
            if lots[symbol]:
                exposure = abs(lots[symbol]) * held_mult(symbol) * mark(day, symbol)
                position_rows.append({
                    "交易日": day, "品种": symbol, "合约": contracts[symbol], "手数": lots[symbol],
                    "主力原始开盘价": mark(day, symbol),
                    "合约乘数": held_mult(symbol), "名义敞口": exposure,
                    "保证金_20pct_assumption": exposure * margin_rate, "目标权重": applied[symbol],
                })

    daily = pd.DataFrame(daily_rows).set_index("交易日")
    expected_equity = (
        daily["期初权益"] - daily["交易费用"] + daily["毛盈亏"] + daily["权益清零调整"]
    )
    if not np.allclose(expected_equity, daily["权益"], rtol=1e-12, atol=1e-7):
        raise ArithmeticError("逐日权益无法由毛盈亏、费用和清零调整复算")
    daily["净收益"] = daily["净盈亏"] / daily["期初权益"].replace(0, np.nan)
    positions = pd.DataFrame(position_rows, columns=[
        "交易日", "品种", "合约", "手数", "主力原始开盘价", "合约乘数", "名义敞口",
        "保证金_20pct_assumption", "目标权重",
    ])
    return daily, positions, pd.DataFrame(events)


# --------------------------------------------------------------------------
# 绩效
# --------------------------------------------------------------------------
PERIODS = C.TRADING_DAYS_PER_YEAR

METRIC_KEYS = ['ann_return', 'ann_vol', 'ret_risk', 'calmar', 'win_rate', 'max_drawdown']

def _empty(n: int) -> dict:
    out = {k: np.nan for k in METRIC_KEYS}
    out['n_days'] = int(n)
    return out


def performance(ret: pd.Series, periods: int = PERIODS) -> dict:
    r = pd.Series(ret, dtype='float64').dropna()
    n = int(len(r))
    if n < 2:
        return _empty(n)
    growth = (1.0 + r).to_numpy()
    if np.any(growth <= 0):
        return _empty(n)
    nav = np.cumprod(growth)
    ann_return = float(nav[-1] ** (periods / n) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods))
    ret_risk = ann_return / ann_vol if ann_vol > 0 else np.nan
    peak = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    dd = nav / peak - 1.0
    max_dd = float(dd.min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    win = float((r.to_numpy() > 0).mean())
    return {
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'ret_risk': float(ret_risk) if np.isfinite(ret_risk) else np.nan,
        'calmar': float(calmar) if np.isfinite(calmar) else np.nan,
        'win_rate': win,
        'max_drawdown': max_dd,
        'n_days': n,
    }


def sharpe_ratio(returns: pd.Series, periods: int = PERIODS) -> float:
    """日均值 / 日标准差 × sqrt(年交易日数)，无风险利率 0。"""
    values = pd.Series(returns, dtype='float64').dropna()
    if len(values) < 2:
        return np.nan
    vol = float(values.std(ddof=1))
    if not np.isfinite(vol) or vol <= 0:
        return np.nan
    return float(values.mean() / vol * np.sqrt(periods))


def newey_west_t(values, lags: int = 10) -> float:
    """均值的 Newey–West t，Bartlett 权重。少于 lags + 3 个观测时为 NaN。"""
    sample = np.asarray(values, dtype="float64")
    sample = sample[np.isfinite(sample)]
    n = len(sample)
    if n < lags + 3:
        return float("nan")
    centered = sample - sample.mean()
    variance = float(centered @ centered) / n
    for lag in range(1, lags + 1):
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * float(centered[lag:] @ centered[:-lag]) / n
    return float(sample.mean() / math.sqrt(variance / n)) if variance > 0 else float("nan")


def max_drawdown(equity, initial_capital: float) -> float:
    """权益相对历史最高点的最大回撤，起点算初始资金。不受熔断后重置峰值影响。"""
    values = np.asarray(equity, dtype="float64")
    peak = np.maximum.accumulate(np.r_[float(initial_capital), values])[1:]
    return float((values / peak - 1.0).min()) if len(values) else float("nan")


def deflated_sharpe(returns: pd.Series, trial_sharpes, n_trials: int | None = None) -> dict:
    """Bailey & López de Prado (2014) 的 DSR：扣掉“试了 n_trials 次取最好”带来的期望最大夏普后，
    观测夏普仍大于 0 的概率。trial_sharpes 是各次尝试的年化夏普（用来估计尝试之间的离散程度）。"""
    from scipy.stats import kurtosis, norm, skew
    r = pd.Series(returns, dtype='float64').dropna()
    trials = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype='float64')
    n = int(n_trials or len(trials))
    if len(r) < 30 or len(trials) < 2 or n < 2:
        return {'dsr': np.nan, 'sr0': np.nan, 'sr': np.nan, 'n_trials': n}
    scale = np.sqrt(PERIODS)
    sr = r.mean() / r.std(ddof=1)                      # 日频，不年化
    spread = np.std(trials / scale, ddof=1)
    gamma = 0.5772156649
    sr0 = spread * ((1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * np.e)))
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    z = (sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return {'dsr': float(norm.cdf(z)), 'sr0': float(sr0 * scale), 'sr': float(sr * scale), 'n_trials': n}


def pbo_cscv(matrix: pd.DataFrame, blocks: int = 16) -> dict:
    """Bailey 等 (2017) 的组合对称交叉验证：把日收益按时间切成 blocks 段，每次取一半做样本内选夏普最高的方案，
    看它在另一半里排在第几。PBO = 样本内最好的方案在样本外排到中位数以下的比例。"""
    from itertools import combinations
    m = pd.DataFrame(matrix).dropna(how='all').fillna(0.0)
    if blocks % 2 or m.shape[1] < 2 or len(m) < blocks * 5:
        raise ValueError("需要偶数段、至少两个方案、每段至少 5 天")
    edges = np.linspace(0, len(m), blocks + 1).astype(int)
    parts = [m.iloc[a:b].to_numpy() for a, b in zip(edges[:-1], edges[1:])]

    def sharpe(rows):
        sd = rows.std(axis=0, ddof=1)
        return np.where(sd > 0, rows.mean(axis=0) / np.where(sd > 0, sd, 1.0), -np.inf)

    logits, degradation = [], []
    for train in combinations(range(blocks), blocks // 2):
        test = [k for k in range(blocks) if k not in train]
        inside = sharpe(np.vstack([parts[k] for k in train]))
        outside = sharpe(np.vstack([parts[k] for k in test]))
        best = int(np.argmax(inside))
        rank = (outside < outside[best]).sum() + 0.5 * ((outside == outside[best]).sum() - 1) + 1
        omega = rank / (m.shape[1] + 1)
        logits.append(np.log(omega / (1 - omega)))
        degradation.append((inside[best] * np.sqrt(PERIODS), outside[best] * np.sqrt(PERIODS)))
    logits = np.asarray(logits)
    pairs = np.asarray(degradation)
    return {'pbo': float((logits <= 0).mean()), 'n_splits': int(len(logits)),
            'is_best_sharpe': float(pairs[:, 0].mean()), 'oos_of_best_sharpe': float(pairs[:, 1].mean()),
            'oos_of_best_below_zero': float((pairs[:, 1] < 0).mean())}


# --------------------------------------------------------------------------
# 行情装载
# --------------------------------------------------------------------------
@dataclass
class StudyData:
    factors: SignalSet
    returns: pd.DataFrame
    universe: dict
    rolls: pd.DataFrame
    open_fee: pd.DataFrame
    close_fee: pd.DataFrame
    roll_close_fee: pd.DataFrame
    slippage: pd.DataFrame
    roll_calendar: dict | None = None


def load_research(n_ticks=1., first_year=None):
    """研究分片里的行情、因子、换月、费用和滑点。默认从 2016 起；first_year 早于 2016 时，
    之前几年的品种池用落盘的 yearly_stats 按同一门槛重建（第 y 年只看第 y−1 年），重建的 2015 必须等于落盘的。"""
    first = C.STUDY_START.year if first_year is None else int(first_year)
    if not C.EARLY_START.year <= first <= C.STUDY_START.year:
        raise ValueError(f"研究账本起始年必须在 {C.EARLY_START.year}–{C.STUDY_START.year}，收到 {first}")
    stored = load_universe()
    universe = {y:s for y,s in stored.items() if 2016<=y<=2021}
    if first < C.STUDY_START.year:
        stats = pd.read_csv(C.UNIVERSE_DIR / "yearly_stats.csv")
        C.assert_no_holdout_dates(pd.to_datetime(stats["year"].astype(str) + "-01-01"), "品种池统计")
        early, _ = build_universe(stats, years=list(range(first, C.STUDY_START.year)))
        if early[C.STUDY_START.year - 1] != sorted(stored[C.STUDY_START.year - 1]):
            raise ValueError("重建的 2015 品种池与落盘的不一致")
        universe = {**{y: early[y] for y in range(first, C.STUDY_START.year)}, **universe}
    symbols = sorted({s for ss in universe.values() for s in ss})
    absent = set(symbols)-set(list_shards(C.RESEARCH_DIR))
    if absent:
        raise FileNotFoundError(f"研究期品种池缺分钟分片: {sorted(absent)}")
    factors = load(symbols)
    prices = factors.bars['open']
    returns = (factors.bars['openw'].shift(-1)-factors.bars['openw'])/prices
    C.assert_no_holdout_dates(returns.index,'研究收益')
    # Final open has no next open in this partition; never borrow 2022 prices.
    start = pd.Timestamp(f"{first}-01-01")
    if first < C.STUDY_START.year:
        # 持续期阈值要 250 日分钟历史，dfp_top3 从 2011-01-14 才有值；从五个因子都有值的那天起计分
        start = max(start, *(factors.signed[name].dropna(how="all").index.min() for name in FACTORS))
    index = returns.index[(returns.index>=start) & returns.notna().any(axis=1)]
    returns = returns.loc[index]
    rolls = pd.DataFrame(False,index=index,columns=symbols)
    for symbol,frame in load_roll_calendar(symbols,C.RESEARCH_END).items():
        days = index.intersection(frame.trading_date)
        rolls.loc[days,symbol] = True
    o,c,r = fee_tables(load_fees(),prices)
    slip = slippage_tables(load_ticks(),prices.loc[index],n_ticks)
    return StudyData(factors,returns,universe,rolls,o.loc[index],c.loc[index],r.loc[index],slip)


def _keep_last(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _require_cost_coverage(symbols, ticks, fees, period):
    available_ticks = set(ticks["symbol"].dropna())
    available_fees = set(fees["symbol"].dropna())
    missing_ticks = sorted(set(symbols) - available_ticks)
    missing_fees = sorted(set(symbols) - available_fees)
    if missing_ticks or missing_fees:
        details = []
        if missing_ticks:
            details.append(f"tick 缓存缺 {missing_ticks}")
        if missing_fees:
            details.append(f"手续费缓存缺 {missing_fees}")
        raise ValueError(
            f"{period} 品种池成本数据不完整，拒绝静默剔除品种："
            + "；".join(details)
            + "。请先运行 tick 准备流程并补齐对应手续费缓存。"
        )


def _fresh_time_factors(minute: pd.DataFrame) -> pd.DataFrame:
    """在一段分钟数据上现算五个因子，口径与研究期缓存相同（N250/M55 持续期阈值）。"""
    from tfcta.factors import duration_factors, report_factors

    minute = add_intraday_coords(minute)
    clock = timestamp_factors(minute)
    duration = duration_factors(minute, C.IC_REFERENCE_LOOKBACK, C.IC_REFERENCE_PCT)
    report = report_factors(minute, C.IC_REFERENCE_LOOKBACK, C.IC_REFERENCE_PCT)
    return clock.join(duration, how="outer").join(report, how="outer")[list(FACTORS)]


def load_through_year(year: int, n_ticks: float = 1.0) -> StudyData:
    """研究期缓存接上样本外至 year 的现算因子和日线。"""
    from tfcta.costs import estimate_ticks

    year = int(year)
    if year > C.DEFAULT_OOS_END.year:
        raise C.HoldoutViolation("不读 2026")
    end = pd.Timestamp(f"{year}-12-31")
    research = load_research(n_ticks)
    pools, _ = oos_universe(C.COMMODITY_SYMBOLS, end.date())
    names = sorted({s for members in pools.values() for s in members})
    fees = pd.concat([load_fees("research"), load_fees("oos")], ignore_index=True)
    ticks = load_ticks()
    _require_cost_coverage(pools[C.OOS_START.year], ticks, fees, f"{year} 年")

    cutoff = research.returns.index[-WARMUP_DAYS]
    factors = {name: {} for name in FACTORS}
    bars = {field: {} for field in ("open", "openw", "close", "closew", "highw", "loww")}
    tick_parts, available = [], {}
    for symbol in names:
        minute = load_minutes(
            symbol, oos_end=end.date(), columns=OOS_COLUMNS,
        )
        minute = minute[pd.to_datetime(minute["trading_date"]) >= cutoff]
        fresh = _fresh_time_factors(minute)
        if symbol in research.factors.signed["ts_low"].columns:
            # 研究期最后几十天两边都有完整的阈值历史，必须一致，否则现算口径和缓存不一样
            overlap = fresh.index[(fresh.index > research.returns.index[-60]) & (fresh.index <= research.returns.index[-1])]
            for name in FACTORS:
                cached = research.factors.raw(name)[symbol].reindex(overlap)
                if not np.allclose(cached, fresh.loc[overlap, name], equal_nan=True, atol=1e-9):
                    raise ValueError(f"{symbol} {name} 现算值与研究期缓存不一致")
        later_days = fresh.index[fresh.index >= pd.Timestamp(C.OOS_START)]
        for name in FACTORS:
            factors[name][symbol] = fresh.loc[later_days, name]
        daily = daily_bars(minute)
        daily = daily.loc[(daily.index >= pd.Timestamp(C.OOS_START)) & (daily.index <= end)]
        for field in bars:
            bars[field][symbol] = daily[field]
        available[symbol] = set(daily.index.year)
        if year > C.OOS_START.year:
            estimated = estimate_ticks(raw_price_path(minute), minute["trading_date"])
            estimated = estimated[(estimated["year"] >= C.OOS_START.year) & (estimated["year"] < year)].copy()
            if not estimated.empty:
                estimated["symbol"] = symbol
                tick_parts.append(estimated)

    signed = {
        name: _keep_last([research.factors.signed[name], pd.DataFrame(values) * C.FACTOR_SIGNS[name]])
        for name, values in factors.items()
    }
    prices = {
        field: _keep_last([research.factors.bars[field], pd.DataFrame(values)])
        for field, values in bars.items()
    }
    returns = (prices["openw"].shift(-1) - prices["openw"]) / prices["open"]
    returns = returns.loc[(returns.index >= pd.Timestamp(C.STUDY_START)) & (returns.index <= end)]
    returns = returns.loc[returns.notna().any(axis=1)]
    if returns.index.max() > end:
        raise C.HoldoutViolation(f"单年测试收益越过 {end.date()}")
    universe = {int(y): list(pool) for y, pool in research.universe.items()}
    for y, members in pools.items():
        kept = [s for s in members if y in available.get(s, set())]
        removed = sorted(set(members) - set(kept))
        if removed:
            print(f"{y} 年池内品种没有当年行情，不进组合: {removed}", flush=True)
        universe[int(y)] = kept
    symbols = list(returns.columns)
    if tick_parts:
        ticks = pd.concat([ticks, *tick_parts], ignore_index=True)
    open_px = prices["open"].reindex(index=returns.index, columns=symbols)
    opened_fee, closed_fee, roll_fee = fee_tables(fees, open_px)
    slip = slippage_tables(ticks, open_px, n_ticks)
    rolls = pd.DataFrame(False, index=returns.index, columns=symbols)
    for symbol, frame in load_roll_calendar(symbols, end.date()).items():
        rolls.loc[rolls.index.intersection(frame.trading_date), symbol] = True
    return StudyData(
        SignalSet(bars=prices, signed=signed), returns, universe, rolls,
        opened_fee, closed_fee, roll_fee, slip,
    )


def ledger_inputs(data: StudyData, period_years, end_date) -> dict:
    """一个区间的整数手账本输入：当年池、价格、换月、合约代码、费用和滑点，按交易日 × 品种对齐。"""
    symbols = sorted({
        symbol
        for year, members in data.universe.items()
        if int(year) in period_years
        for symbol in members
        if symbol in data.returns.columns
    })
    require_multipliers(symbols)
    selected_universe = {
        int(year): sorted(set(members).intersection(symbols))
        for year, members in data.universe.items()
        if int(year) in period_years
    }
    selected_universe = {
        year: members for year, members in selected_universe.items() if members
    }
    selected_symbols = sorted({
        symbol for members in selected_universe.values() for symbol in members
    })
    if not selected_symbols:
        raise ValueError("流动性池与收益没有交集")

    index = data.returns.index[data.returns.index.year.isin(period_years)]
    if index.empty:
        raise ValueError(f"{period_years} 整数手数回测没有可计分的交易日")
    columns = pd.Index(selected_symbols)
    prices = data.factors.bars
    open_price = prices["open"].reindex(index=index, columns=columns)
    close_price = prices["close"].reindex(index=index, columns=columns)
    adjusted_open = prices["openw"].reindex(index=index, columns=columns)
    adjusted_open_delta = (
        prices["openw"].shift(-1) - prices["openw"]
    ).reindex(index=index, columns=columns)
    adjusted_high = prices["highw"].reindex(index=index, columns=columns)
    adjusted_low = prices["loww"].reindex(index=index, columns=columns)
    rolls = data.rolls.reindex(index=index, columns=columns, fill_value=False)
    fee_open = data.open_fee.reindex(index=index, columns=columns)
    fee_close = data.close_fee.reindex(index=index, columns=columns)
    fee_roll = data.roll_close_fee.reindex(index=index, columns=columns)
    slip = data.slippage.reindex(index=index, columns=columns)
    multipliers = {symbol: MULTIPLIER[symbol] for symbol in selected_symbols}

    roll_calendar = data.roll_calendar
    if roll_calendar is None:
        roll_calendar = load_roll_calendar(selected_symbols, end_date.date())
    contract_codes = pd.DataFrame(index=index, columns=columns, dtype=object)
    for symbol in selected_symbols:
        calendar = roll_calendar.get(symbol)
        if calendar is None or calendar.empty:
            raise FileNotFoundError(f"{symbol} 缺少主力合约换月表")
        calendar = calendar.drop_duplicates("trading_date", keep="last").set_index(
            "trading_date"
        ).sort_index()
        changes = calendar["new_contract"].copy()
        if calendar.index.min() > index.min():
            changes.loc[index.min()] = calendar.iloc[0]["prev_contract"]
        contract_codes[symbol] = changes.reindex(
            changes.index.union(index)
        ).sort_index().ffill().reindex(index)
        if contract_codes[symbol].isna().any():
            raise ValueError(f"{symbol} 研究期存在无法映射的主力合约代码")
    contract_multipliers = {
        symbol: table for symbol in selected_symbols
        if (table := contract_multiplier_overrides(symbol, contract_codes[symbol].dropna()))
    }

    return {
        "index": index, "columns": columns, "universe": selected_universe,
        "symbols": selected_symbols, "open_price": open_price, "close_price": close_price,
        "adjusted_open": adjusted_open, "adjusted_open_delta": adjusted_open_delta,
        "adjusted_high": adjusted_high, "adjusted_low": adjusted_low, "rolls": rolls,
        "contract_codes": contract_codes, "open_fee": fee_open, "close_fee": fee_close,
        "roll_close_fee": fee_roll, "slippage": slip, "multipliers": multipliers,
        "contract_multipliers": contract_multipliers,
    }


INDEX_SYMBOLS = ("IF", "IH", "IC")
INDEX_RESEARCH_START = pd.Timestamp("2015-04-16")
INDEX_RESEARCH_END = pd.Timestamp("2021-12-31")
INDEX_EXCHANGE_TICK = 0.2


def _load_index_research_inputs():
    """读取 IF/IH/IC 研究期分钟分片，构造共享整数手账本所需的日线输入。"""
    from strategies.index_common import high_open_z
    frames, contracts_by_symbol, tick_parts, rally_parts = {}, {}, [], {}
    for symbol in INDEX_SYMBOLS:
        minute = load_shard(
            symbol,
            directory=C.INDEX_RESEARCH_DIR,
            columns=list(dict.fromkeys([*C.FACTOR_FIELDS, *C.PRICE_FIELDS, "dominant_id"])),
        )
        dates = pd.to_datetime(minute["trading_date"]).dt.normalize()
        C.assert_no_holdout_dates(dates, what=f"{symbol} 股指分钟行情")
        daily = daily_bars(minute)
        frames[symbol] = daily
        rally_parts[symbol] = high_open_z({f: daily[[f]].rename(columns={f: symbol}) for f in ("open", "openw", "highw")}, (symbol,))
        contracts = minute["dominant_id"].groupby(dates, sort=True).first()
        contracts.index = pd.DatetimeIndex(contracts.index)
        contracts_by_symbol[symbol] = contracts
        ticks = estimate_ticks(minute["close"], dates)
        ticks.insert(0, "symbol", symbol)
        tick_parts.append(ticks)

    common_start = max(frame.index.min() for frame in frames.values())
    if common_start > INDEX_RESEARCH_START:
        raise ValueError(
            f"IF/IH/IC 共同行情从 {common_start.date()} 才开始，"
            f"晚于约定起点 {INDEX_RESEARCH_START.date()}"
        )
    index = pd.DatetimeIndex(
        sorted(
            set.union(*(set(frame.index) for frame in frames.values()))
        )
    )
    index = index[
        (index >= INDEX_RESEARCH_START) & (index <= INDEX_RESEARCH_END)
    ]
    bars = {
        field: pd.DataFrame(
            {symbol: frames[symbol][field].reindex(index) for symbol in INDEX_SYMBOLS},
            index=index,
        )
        for field in ("open", "openw", "close", "closew", "highw", "loww")
    }
    contracts = pd.DataFrame(
        {symbol: contracts_by_symbol[symbol].reindex(index) for symbol in INDEX_SYMBOLS},
        index=index,
    )
    if contracts.isna().any().any():
        missing = contracts.columns[contracts.isna().any()].tolist()
        raise ValueError(f"股指主力合约代码覆盖不完整：{missing}")
    for field in ("open", "openw", "close", "closew", "highw", "loww"):
        if bars[field].isna().any().any():
            raise ValueError(f"IF/IH/IC 研究期 {field} 行情未完全对齐")

    rally_z = pd.DataFrame(rally_parts).reindex(index).mean(axis=1, skipna=False).rename("high_open_z")
    ticks = pd.concat(tick_parts, ignore_index=True)
    first_year_ticks = (
        ticks[ticks["year"] == INDEX_RESEARCH_START.year]
        .set_index("symbol")["tick"]
        .reindex(INDEX_SYMBOLS)
    )
    if first_year_ticks.isna().any() or not np.allclose(
        first_year_ticks.to_numpy(dtype=float), INDEX_EXCHANGE_TICK
    ):
        raise ValueError(
            f"2015 年估计 tick 与中金所固定 tick {INDEX_EXCHANGE_TICK} 不符："
            f"{first_year_ticks.to_dict()}"
        )
    # 2015 年初没有可用的上一年 tick，使用已核实的固定交易所最小变动价位。
    ticks = pd.concat(
        [
            ticks,
            pd.DataFrame(
                {"symbol": INDEX_SYMBOLS, "year": 2014, "tick": INDEX_EXCHANGE_TICK}
            ),
        ],
        ignore_index=True,
    )
    return index, bars, contracts, rally_z, ticks


def _simulate_index_targets(
    target: pd.DataFrame,
    scored_index: pd.DatetimeIndex,
    bars: dict[str, pd.DataFrame],
    contracts: pd.DataFrame,
    ticks: pd.DataFrame,
    end_year: int,
    symbols: tuple[str, ...],
    *,
    exposure_limit: float,
    drawdown_limit: float,
    halt_days: int,
    initial_capital: float = INITIAL_CAPITAL,
    open_limits: bool = False,
    open_slippage: dict[str, pd.DataFrame] | None = None,
    fee_table: pd.DataFrame | None = None,
):
    """股指整数手账本。open_limits=True 时按中金所每日开仓限额成交（tfcta.costs.CFFEX_OPEN_LIMITS）。
    open_slippage 见 simulate_target_lots；给出时按计分日、品种对齐。"""
    if scored_index.empty:
        raise ValueError("股指目标仓位没有可计分的交易日")
    prices = bars["open"].reindex(index=scored_index, columns=symbols)
    if fee_table is None:
        fee_table = load_index_fees("research")
        fee_table = fee_table[fee_table["trading_date"] <= INDEX_RESEARCH_END]
    open_fee, close_fee, roll_fee = fee_tables(fee_table, prices)
    slippage = slippage_tables(ticks[ticks["year"] < end_year], prices)
    selected_contracts = contracts.reindex(index=scored_index, columns=symbols)
    rolls = selected_contracts.ne(selected_contracts.shift()).fillna(False)
    if not rolls.empty:
        rolls.iloc[0] = False
    universe = {
        int(year): list(symbols)
        for year in range(int(scored_index.min().year), int(scored_index.max().year) + 1)
    }
    return simulate_target_lots(
        target.reindex(index=scored_index, columns=symbols).fillna(0.0),
        prices,
        bars["close"].reindex(index=scored_index, columns=symbols),
        bars["openw"].reindex(index=scored_index, columns=symbols),
        (bars["openw"].shift(-1) - bars["openw"]).reindex(
            index=scored_index, columns=symbols
        ),
        bars["highw"].reindex(index=scored_index, columns=symbols),
        bars["loww"].reindex(index=scored_index, columns=symbols),
        rolls,
        selected_contracts,
        universe,
        {symbol: INDEX_MULTIPLIER[symbol] for symbol in symbols},
        open_fee,
        close_fee,
        roll_fee,
        slippage,
        initial_capital=initial_capital,
        margin_rate=MARGIN_RATE,
        exposure_limit=exposure_limit,
        drawdown_limit=drawdown_limit,
        halt_days=halt_days,
        open_limits=index_open_limits(scored_index, symbols) if open_limits else None,
        open_slippage=None if open_slippage is None else {
            side: table.reindex(index=scored_index, columns=symbols) for side, table in open_slippage.items()
        },
    )




# --------------------------------------------------------------------------
# 股指回测
# --------------------------------------------------------------------------
def _read_index_oos_symbol(
    symbol: str, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按请求窗口筛读一个品种的样本外分钟分片。"""
    if start < pd.Timestamp(C.OOS_START) or end > pd.Timestamp(C.DEFAULT_OOS_END) or start > end:
        raise C.HoldoutViolation(
            f"股指样本外行情窗口须在 {C.OOS_START}..{C.DEFAULT_OOS_END}，收到 {start.date()}..{end.date()}"
        )
    columns = list(dict.fromkeys([*C.FACTOR_FIELDS, *C.PRICE_FIELDS, "dominant_id"]))
    price_path = find_shard(C.INDEX_OOS_DIR, symbol)
    if price_path is None:
        raise FileNotFoundError(f"{symbol} 股指样本外分片缺失")
    C.assert_index_oos_only(price_path)
    if price_path.suffix != ".parquet":
        raise ValueError("股指样本外分片必须是可按日期筛读的 Parquet")
    filters = [("trading_date", ">=", start), ("trading_date", "<=", end)]
    minute = read_frame(price_path, columns=columns, filters=filters)
    if minute.empty:
        raise ValueError(f"{symbol} 股指样本外行情为空")
    dates = pd.to_datetime(minute["trading_date"]).dt.normalize()
    if dates.min() < start or dates.max() > end:
        raise C.HoldoutViolation(f"{symbol} 股指样本外分片越过 {start.date()}..{end.date()}")
    daily = daily_bars(minute)
    contracts = minute["dominant_id"].groupby(dates, sort=True).first()
    contracts.index = pd.DatetimeIndex(contracts.index)
    tick_rows = estimate_ticks(minute["close"], dates)
    tick_rows.insert(0, "symbol", symbol)
    return daily, contracts, tick_rows


def _load_index_oos_inputs(start: pd.Timestamp, end: pd.Timestamp):
    """读取请求年份内的 IF/IH/IC 样本外日线和 L1 开盘摘要。"""
    from research.index.index_execution_10m import opening_quotes

    frames, contract_parts, tick_parts, date_indexes = {}, {}, [], []
    for symbol in INDEX_SYMBOLS:
        daily, contracts, tick_rows = _read_index_oos_symbol(symbol, start, end)
        frames[symbol] = daily
        contract_parts[symbol] = contracts
        tick_parts.append(tick_rows)
        date_indexes.append(daily.index)

    oos_index = pd.DatetimeIndex(date_indexes[0]).sort_values()
    if any(not oos_index.equals(pd.DatetimeIndex(other).sort_values()) for other in date_indexes[1:]):
        raise ValueError("IF/IH/IC 样本外日历不一致")
    if oos_index.min() < start or oos_index.max() > end:
        raise C.HoldoutViolation(f"股指样本外日线越过 {start.date()}..{end.date()}")
    bars = {
        field: pd.DataFrame(
            {symbol: frames[symbol][field].reindex(oos_index) for symbol in INDEX_SYMBOLS},
            index=oos_index,
        )
        for field in ("open", "openw", "close", "closew", "highw", "loww")
    }
    contracts = pd.DataFrame(
        {symbol: contract_parts[symbol].reindex(oos_index) for symbol in INDEX_SYMBOLS},
        index=oos_index,
    )
    if contracts.isna().any().any() or any(
        frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype="float64")).all()
        for frame in bars.values()
    ):
        raise ValueError("IF/IH/IC 样本外行情或合约代码未完全对齐")

    root = C.INDEX_EXTERNAL_OOS_DIR.resolve()
    quote_parts = []
    tick_columns = ["datetime", "trading_date", "open", "volume", "a1", "b1", "contract"]
    for symbol in INDEX_SYMBOLS:
        for year in range(start.year, end.year + 1):
            year_start = max(start, pd.Timestamp(year=year, month=1, day=1))
            year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
            folder = C.INDEX_EXTERNAL_OOS_DIR / "l1_ticks" / symbol / str(year)
            files = sorted(folder.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"{symbol} {year} 年股指样本外 L1 不存在: {folder}")
            filters = [("trading_date", ">=", year_start), ("trading_date", "<=", year_end)]
            for path in files:
                if root not in path.resolve().parents:
                    raise C.HoldoutViolation(f"股指样本外 L1 路径越界: {path}")
                ticks = pd.read_parquet(path, columns=tick_columns, filters=filters)
                if ticks.empty:
                    continue
                ticks["datetime"] = pd.to_datetime(ticks["datetime"])
                ticks["trading_date"] = pd.to_datetime(ticks["trading_date"]).dt.normalize()
                if (
                    not ticks["trading_date"].dt.year.eq(year).all()
                    or ticks["trading_date"].min() < year_start
                    or ticks["trading_date"].max() > year_end
                ):
                    raise C.HoldoutViolation(f"{symbol} {year}年L1越过 {year_start.date()}..{year_end.date()}")
                ticks = ticks[ticks["datetime"] - ticks["trading_date"] < pd.Timedelta(10 * 3600 + 5 * 60, unit="s")]
                summary = opening_quotes(ticks)
                summary.insert(1, "symbol", symbol)
                quote_parts.append(summary)
    if not quote_parts:
        raise ValueError("股指样本外 L1 开盘报价为空")
    quotes = pd.concat(quote_parts, ignore_index=True)
    if quotes.duplicated(["trading_date", "symbol"]).any():
        raise ValueError("股指样本外 L1 开盘摘要日期重复")
    return oos_index, bars, contracts, pd.concat(tick_parts, ignore_index=True), quotes


def run_index(
    start,
    end,
    *,
    by_year=False,
    capital=INDEX_DEFAULT_CAPITAL,
    execution=INDEX_DEFAULT_EXECUTION,
    book=None,
):
    """运行已登记股指策略。研究期只读 research/，样本外只读 oos/，预热历史从研究期接上。"""
    from strategies.index_common import high_open_z, market_close_location
    from research.index.index_execution_10m import load_opening_quotes, l1_slippage

    if book is None:
        from strategies.index_catalog import INDEX_BOOKS, INDEX_FROZEN_BOOKS
        selected = INDEX_FROZEN_BOOKS if int(end) >= C.OOS_START.year else INDEX_BOOKS
        if not selected:
            raise C.HoldoutViolation("当前没有已冻结股指策略可用于样本外回测")
        for item in selected:
            run_index(
                start, end, by_year=by_year, capital=capital, execution=execution, book=item,
            )
        return
    book.frozen()
    start, end = int(start), int(end)
    if start < 2015 or end > C.DEFAULT_OOS_END.year or start > end:
        raise C.HoldoutViolation(f"股指回测年份须在 2015–{C.DEFAULT_OOS_END.year}，收到 {start}–{end}")
    if end >= C.OOS_START.year and book.STATUS != "frozen":
        raise C.HoldoutViolation("股指策略未冻结，不允许运行样本外")
    if capital != INDEX_DEFAULT_CAPITAL or execution != INDEX_DEFAULT_EXECUTION:
        raise ValueError("股指口径固定为1000万、l1_30m")
    if by_year and start != end:
        for year in range(start, end + 1):
            run_index(year, year, capital=capital, execution=execution, book=book)
        return

    _, research_bars, research_contracts, _, research_ticks = _load_index_research_inputs()
    bars, contracts, ticks = research_bars, research_contracts, research_ticks
    quotes = load_opening_quotes()
    fee_table = None
    if end >= C.OOS_START.year:
        oos_start = pd.Timestamp(C.OOS_START)
        oos_end = pd.Timestamp(f"{end}-12-31")
        _, oos_bars, oos_contracts, oos_ticks, oos_quotes = _load_index_oos_inputs(oos_start, oos_end)
        bars = {
            field: pd.concat([research_bars[field], oos_bars[field]]).sort_index()
            for field in oos_bars
        }
        contracts = pd.concat([research_contracts, oos_contracts]).sort_index()
        ticks = pd.concat([research_ticks, oos_ticks], ignore_index=True)
        quotes = pd.concat([quotes, oos_quotes], ignore_index=True)
        if contracts.index.has_duplicates or contracts.isna().any().any():
            raise ValueError("研究期与样本外的主力合约代码拼接重复或缺失")
        if quotes.duplicated(["trading_date", "symbol"]).any():
            raise ValueError("研究期与样本外的 L1 开盘摘要日期重复")
        oos_fees = load_index_oos_fees(oos_start, oos_end)
        fee_table = oos_fees if start >= C.OOS_START.year else pd.concat(
            [load_index_fees("research"), oos_fees], ignore_index=True
        )

    signal_index = bars["close"].index
    signal_builder = getattr(book, "signal_inputs", None)
    signal_inputs = (
        market_close_location(bars, signal_index)
        if signal_builder is None
        else signal_builder(bars, signal_index)
    )
    target = book.targets(signal_inputs, bars, rally_z=high_open_z(bars))
    returns = ((bars["openw"].shift(-1) - bars["openw"]) / bars["open"]).dropna(how="any")
    scored = returns.loc[(returns.index.year >= start) & (returns.index.year <= end)].index
    if scored.empty:
        raise ValueError(f"{start}–{end} 没有可计分的股指交易日")
    slip, _, _ = l1_slippage(quotes, bars, contracts, scored)
    daily, positions, events = _simulate_index_targets(
        target.reindex(scored),
        scored,
        bars,
        contracts,
        ticks,
        end,
        tuple(book.SYMBOLS),
        exposure_limit=book.GROSS,
        drawdown_limit=book.DRAWDOWN_LIMIT,
        halt_days=book.HALT_DAYS,
        initial_capital=capital,
        open_limits=True,
        open_slippage=slip,
        fee_table=fee_table,
    )
    held = pd.DataFrame(1.0, index=scored, columns=list(book.SYMBOLS))
    benchmark, _, _ = _simulate_index_targets(
        held,
        scored,
        bars,
        contracts,
        ticks,
        end,
        tuple(book.SYMBOLS),
        exposure_limit=1.0,
        drawdown_limit=0.99,
        halt_days=book.HALT_DAYS,
        initial_capital=capital,
        open_limits=True,
        open_slippage=slip,
        fee_table=fee_table,
    )
    if (start, end) == (2015, 2021):
        period = "index_research"
    elif (start, end) == (C.OOS_START.year, C.DEFAULT_OOS_END.year):
        period = "oos"
    elif start == end:
        period = start
    else:
        period = (start, end)
    write_results(
        period, book, daily, positions, events, list(book.SYMBOLS),
        initial_capital=capital, market="index", benchmark=benchmark["净收益"],
    )
    out = C.backtest_dir("index", book.ID, period_layout(period)[2])
    target.reindex(scored).to_csv(out / "targets.csv", encoding="utf-8-sig")
    return daily, positions, events



def period_layout(period) -> tuple[int, int, str, str]:
    """research、oos、单年或 (起, 止) → 起止年、结果子目录和图标题。"""
    if period == "research":
        return C.EARLY_START.year, C.RESEARCH_END.year, "research", "2011–2021"
    if period == "index_research":
        return INDEX_RESEARCH_START.year, INDEX_RESEARCH_END.year, "research", "2015–2021"
    if period == "oos":
        return C.OOS_START.year, C.DEFAULT_OOS_END.year, "oos", "2022–2025"
    if isinstance(period, int):
        return period, period, str(period), f"{period}年"
    if isinstance(period, tuple) and len(period) == 2:
        start, end = int(period[0]), int(period[1])
        return start, end, f"{start}-{end}", f"{start}–{end}"
    raise ValueError(f"不认识的区间 {period!r}")


def plot_performance(
    returns,
    out_dir,
    title="净值与最大回撤",
    series=None,
    *,
    filename="performance.png",
) -> Path:
    """日收益（列为策略）→ 净值和回撤同一张图，写到 out_dir/performance.png。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(returns).apply(pd.to_numeric, errors="coerce")
    if series:
        missing = set(series) - set(frame.columns)
        if missing:
            raise KeyError(f"结果中没有这些序列: {sorted(missing)}")
        frame = frame[list(series)]
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all").fillna(0.0)
    if frame.empty:
        raise ValueError("没有可绘制的日收益数据")
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS", "PingFang SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    nav = (1 + frame).cumprod()
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1
    fig, (nav_axis, drawdown_axis) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    dates = nav.index.to_numpy(dtype="datetime64[ns]")
    for name in nav.columns:
        nav_axis.plot(dates, nav[name].to_numpy(), label=name, lw=1.2)
    nav_axis.set_title(title)
    nav_axis.set_ylabel("净值")
    nav_axis.grid(alpha=0.3)
    nav_axis.legend(loc="upper left", fontsize=8)
    for name in drawdown.columns:
        drawdown_axis.plot(dates, drawdown[name].to_numpy(), lw=1.0)
    drawdown_axis.fill_between(
        dates, drawdown.min(axis=1).to_numpy(), 0, alpha=0.2
    )
    drawdown_axis.set_ylabel("回撤")
    drawdown_axis.grid(alpha=0.3)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_annual_performance(returns, benchmark, out_dir, strategy_name) -> list[Path]:
    """按自然年重置净值，逐年绘制策略与池内等权买入持有。"""
    frame = pd.DataFrame({strategy_name: returns, "等权买入持有": benchmark})
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    if frame.empty:
        raise ValueError("没有可绘制的年度收益数据")
    paths = []
    for year, part in frame.groupby(frame.index.year, sort=True):
        paths.append(
            plot_performance(
                part,
                out_dir,
                title=f"{strategy_name} vs 等权买入持有 {year}",
                filename=f"{year}.png",
            )
        )
    return paths


def benchmark_comparison(returns: pd.Series, benchmark: pd.Series) -> dict:
    """策略相对等权买入持有：两者年化、差值、逐日超额 NW t、beta 与年化 alpha。"""
    bench = pd.Series(benchmark).reindex(returns.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    excess = returns - bench
    strat_metrics, bench_metrics = performance(returns), performance(bench)
    variance = float(bench.var())
    beta = float(returns.cov(bench) / variance) if variance > 0 else np.nan
    alpha = float((returns.mean() - beta * bench.mean()) * PERIODS) if np.isfinite(beta) else np.nan
    return {
        "买入持有累计收益": float((1.0 + bench).prod() - 1.0),
        "买入持有年化收益": bench_metrics["ann_return"],
        "买入持有Sharpe": sharpe_ratio(bench),
        "买入持有最大回撤": max_drawdown((1.0 + bench).cumprod(), 1.0),
        "年化超额收益": strat_metrics["ann_return"] - bench_metrics["ann_return"],
        "逐日超额均值年化": float(excess.mean() * PERIODS),
        "逐日超额NW_t": newey_west_t(excess.to_numpy(), lags=10),
        "相对买入持有beta": beta,
        "相对买入持有年化alpha": alpha,
    }


def _capital_label(capital: float) -> str:
    capital = float(capital)
    return f"{capital / 1e8:g}亿元" if capital >= 1e8 else f"{capital / 1e4:g}万元"


def write_results(
    period,
    strategy,
    daily,
    positions,
    events,
    symbols,
    initial_capital=INITIAL_CAPITAL,
    *,
    market="commodity",
    folder_suffix="",
    benchmark=None,
):
    """写一条策略在一个区间的汇总、分年、逐日、持仓和事件，返回汇总行和分年行。
    folder_suffix 给非冻结规格的资金/成交口径另开子目录。"""
    start, end, folder, title = period_layout(period)
    if folder_suffix:
        folder = f"{folder}_{folder_suffix}"
    years = tuple(range(start, end + 1))
    seen = set(int(year) for year in daily.index.year)
    if not seen or seen - set(years):
        raise C.HoldoutViolation(f"净值日期必须落在 {start}–{end}")
    if end > C.DEFAULT_OOS_END.year:
        raise C.HoldoutViolation("不读 2026")
    label = strategy.NAME
    daily = daily.copy()
    daily["策略"] = label
    positions = positions.copy()
    positions.insert(0, "策略", label)
    if not events.empty:
        events = events.copy()
        events.insert(0, "策略", label)
    returns = daily["净收益"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    metrics = performance(returns)
    summary = {
        "市场": "股指" if market == "index" else "商品",
        "策略": label,
        "参数": strategy.DETAIL,
        "初始资金": float(initial_capital),
        "池内品种数": len(symbols),
        "名义敞口上限": float(strategy.GROSS),
        "保证金率假设": MARGIN_RATE,
        "累计净收益": float(daily["权益"].iloc[-1] / initial_capital - 1.0),
        "年化收益": metrics["ann_return"],
        "Sharpe": sharpe_ratio(returns),
        "最大回撤": max_drawdown(daily["权益"], initial_capital),
        "Calmar比率": metrics["calmar"],
        "期末权益": float(daily["权益"].iloc[-1]),
        "最大名义敞口倍数": float(daily["敞口倍数"].max()),
        "最大保证金权益占用": float(daily["保证金权益占用"].max()),
        "开仓笔数": int(daily["开仓笔数"].sum()),
        "平仓笔数": int(daily["平仓笔数"].sum()),
        "换月笔数": int(daily["换月笔数"].sum()),
        "涨跌停代理阻单数": int(daily["受阻委托数"].sum()),
    }
    if benchmark is not None:
        summary.update(benchmark_comparison(returns, benchmark))
    annual_rows = []
    for year in years:
        part = daily.loc[daily.index.year == year]
        if part.empty:
            continue
        opening = float(part["期初权益"].iloc[0])
        annual_metrics = performance(part["净收益"].fillna(0.0))
        annual_rows.append({
            "策略": label,
            "年份": year,
            "净收益": float(part["权益"].iloc[-1] / opening - 1.0),
            "Sharpe": sharpe_ratio(part["净收益"].fillna(0.0)),
            "Calmar比率": annual_metrics["calmar"],
            "期末权益": float(part["权益"].iloc[-1]),
            "年度内最大回撤": max_drawdown(part["权益"], opening),
            "开仓笔数": int(part["开仓笔数"].sum()),
            "平仓笔数": int(part["平仓笔数"].sum()),
        })
        if benchmark is not None:
            bench = pd.Series(benchmark).reindex(part.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            bench_return = float((1.0 + bench).prod() - 1.0)
            annual_rows[-1].update({
                "买入持有净收益": bench_return,
                "超额收益": annual_rows[-1]["净收益"] - bench_return,
                "买入持有年度内最大回撤": max_drawdown((1.0 + bench).cumprod(), 1.0),
            })

    result_dir = C.backtest_dir(market, strategy.ID, folder)
    result_dir.mkdir(parents=True, exist_ok=True)
    chart = pd.DataFrame({label: returns})
    if benchmark is not None:
        chart["等权买入持有"] = (
            pd.Series(benchmark).reindex(returns.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
    plot_performance(
        chart,
        result_dir / "plots",
        title=f"{label} {title}",
        filename="performance.png",
    )
    if market == "index" and benchmark is not None:
        plot_annual_performance(
            chart[label],
            chart["等权买入持有"],
            result_dir / "plots",
            label,
        )
    tables = {
        "summary.csv": pd.DataFrame([summary]),
        "annual.csv": pd.DataFrame(annual_rows),
        "daily.csv": daily.reset_index(),
        "positions.csv": positions,
        "events.csv": events,
    }
    for name, table in tables.items():
        table.to_csv(result_dir / name, index=False, encoding="utf-8-sig", float_format="%.10f")
    print(f"{label} {title}：结果在 {result_dir}", flush=True)
    return summary, annual_rows


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
            description="商品和股指整数手数回测。--research 跑研究期，--oos 跑 2022–2025。"
    )
    parser.add_argument("strategy", nargs="?", help="策略 ID 或中文名；不提供时使用市场默认策略")
    parser.add_argument(
        "--market",
        choices=("commodity", "index"),
        default="commodity",
        help="未点名策略时选择商品或股指（默认商品）",
    )
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--research", action="store_true", help="2011–2021（2011–2015 品种池按同一门槛重建）")
    period.add_argument("--oos", action="store_true", help="2022–2025")
    period.add_argument("--year", type=int, help="单独一年，2011–2025")
    period.add_argument("--start", type=int, help="区间起始年，和 --end 一起用；最早 2011（2016 以前只能在研究期里）")
    parser.add_argument("--end", type=int, help="区间结束年，和 --start 一起用")
    parser.add_argument("--by-year", action="store_true", help="区间里每年一本新的账")
    parser.add_argument("--capital", type=float, help="股指冻结资金为1000万，仅接受1e7")
    parser.add_argument("--execution", choices=INDEX_EXECUTIONS, help="股指冻结成交口径为l1_30m")
    args = parser.parse_args(argv)
    if (args.start is None) != (args.end is None):
        parser.error("--start 和 --end 要一起给")
    if args.start is not None:
        window = (args.start, args.end)
    elif args.research:
        window = (C.EARLY_START.year, C.RESEARCH_END.year)
    elif args.oos:
        window = (C.OOS_START.year, C.DEFAULT_OOS_END.year)
    elif args.year is not None:
        window = (args.year, args.year)
    else:
        parser.error("要指定 --research、--oos、--year 或 --start/--end")
    if args.by_year and window[0] == window[1]:
        parser.error("--by-year 需要跨年的区间")
    from strategies.commodity_catalog import book_named as commodity_book_named
    from strategies.index_catalog import book_named as index_book_named
    market = args.market
    try:
        if args.strategy:
            try:
                book = commodity_book_named(args.strategy)
            except ValueError:
                book = index_book_named(args.strategy)
                market = "index"
            else:
                if market == "index":
                    raise ValueError(f"{args.strategy!r} 是商品策略，不能用于股指回测")
        elif market == "index":
            book = None
        else:
            book = None
    except ValueError as error:
        parser.error(str(error))

    basis_given = args.capital is not None or args.execution is not None
    if basis_given and market != "index":
        parser.error("--capital 和 --execution 只用于股指")
    if market == "index":
        if args.research and args.start is None:
            window = (2015, 2021)
        capital = INDEX_DEFAULT_CAPITAL if args.capital is None else args.capital
        execution = args.execution or INDEX_DEFAULT_EXECUTION
        if capital != INDEX_DEFAULT_CAPITAL or execution != INDEX_DEFAULT_EXECUTION:
            parser.error("股指口径固定为1000万、l1_30m")
        if window[0] < 2015 or window[1] > C.DEFAULT_OOS_END.year:
            parser.error(f"股指回测年份须在 2015–{C.DEFAULT_OOS_END.year}")
        from strategies.index_catalog import INDEX_BOOKS, INDEX_FROZEN_BOOKS
        if window[1] >= C.OOS_START.year and (
            (book is None and not INDEX_FROZEN_BOOKS)
            or (book is not None and book.STATUS != "frozen")
        ):
            parser.error("当前没有已冻结股指策略可用于样本外回测")
        # 样本外只选择冻结策略；研究期默认运行全部登记策略。
        default_books = INDEX_FROZEN_BOOKS if window[1] >= C.OOS_START.year else INDEX_BOOKS
        books = default_books if book is None else (book,)
        for item in books:
            run_index(
                *window,
                by_year=args.by_year,
                capital=capital,
                execution=execution,
                book=item,
            )
        return

    from research.commodity import market_clock_slow_short_20_60_120 as driver

    driver.run_span(*window, by_year=args.by_year, book=book)


if __name__ == "__main__":
    main()
