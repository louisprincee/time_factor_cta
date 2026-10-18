"""整数手数期货账本、成本和绩效。调用 strategies/ts_low.py。

    python -m backtest.engine --research
    python -m backtest.engine --validation

样本外读取只通过显式冻结并审计的 OOS 命令。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tfcta import config as C
from tfcta.costs import (
    MULTIPLIER, fee_tables, load_fees, load_ticks, require_multipliers,
    slippage_tables,
)
from tfcta.data import (
    add_intraday_coords, daily_bars, find_shard, holding_forward_return, list_shards,
    load_minutes, load_roll_calendar, load_universe, load_validation_shard, oos_universe,
    validation_universe,
)
from tfcta.factors import (
    CACHE_VERSION, SIGNED_PRIORS, SignalSet, combine, combo_dir, load, report_dir,
    raw_price_path, timestamp_dir, timestamp_factors, to_signal,
)
from tfcta.ic import cross_sectional_ic_table, factor_ic_table
def simulate_integer_lots(
    entries: pd.DataFrame,
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
    holding_days: int,
    exposure_limit: float,
    drawdown_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate a long-only integer-lot account with a peak-drawdown breaker."""
    frames = {
        "open_price": open_price, "close_price": close_price,
        "adjusted_open": adjusted_open, "adjusted_open_delta": adjusted_open_delta,
        "adjusted_high": adjusted_high, "adjusted_low": adjusted_low,
        "rolls": rolls, "contract_codes": contract_codes,
        "open_fee": open_fee, "close_fee": close_fee,
        "roll_close_fee": roll_close_fee, "slippage": slippage,
    }
    index, columns = entries.index, entries.columns
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("交易日索引必须唯一且按升序排列")
    for name, frame in frames.items():
        if not frame.index.equals(index) or not frame.columns.equals(columns):
            raise ValueError(f"{name} 的日期或品种索引与入场表不一致")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("初始资金必须是正的有限数")
    if not np.isfinite(margin_rate) or not 0 < margin_rate <= 1:
        raise ValueError("保证金率必须在 (0, 1] 内")
    if not np.isfinite(exposure_limit) or exposure_limit <= 0:
        raise ValueError("名义敞口上限必须是正的有限数")
    if not np.isfinite(drawdown_limit) or not 0 < drawdown_limit < 1:
        raise ValueError("回撤熔断阈值必须在 (0, 1) 内")
    if not isinstance(holding_days, int) or holding_days < 1:
        raise ValueError("持仓日数必须是正整数")
    values = entries.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("纯多头入场表只能包含 0、1，且不能缺失")
    for symbol in columns:
        multiplier = multipliers.get(symbol)
        if multiplier is None or not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f"{symbol} 缺少有效合约乘数")

    high, low = adjusted_high.to_numpy(dtype="float64"), adjusted_low.to_numpy(dtype="float64")
    locked = np.isfinite(high) & np.isfinite(low) & np.isclose(
        high, low, rtol=0.0, atol=1e-10,
    )
    lots, age = dict.fromkeys(columns, 0), dict.fromkeys(columns, 0)
    contracts = dict.fromkeys(columns)
    pending_roll, blocked_reasons = set(), {}
    last_price, last_close = dict.fromkeys(columns, np.nan), dict.fromkeys(columns, np.nan)
    equity = peak_equity = float(initial_capital)
    pause_until_year = stop_new_until_year = 0
    previous_drawdown = 0.0
    forced_liquidation = margin_call_active = False
    daily_rows, position_rows, events = [], [], []

    def event(day, kind, symbol=None, quantity=0, price=np.nan, detail=""):
        events.append({
            "交易日": day, "事件": kind, "品种": symbol, "手数": int(quantity),
            "成交代理价": float(price) if np.isfinite(price) else np.nan,
            "合约": contracts.get(symbol) if symbol else None, "说明": detail,
        })

    def charge(table, day, symbol, quantity, price, description):
        rate = float(table.at[day, symbol])
        if not np.isfinite(rate) or not np.isfinite(price) or price <= 0:
            raise ValueError(f"{day.date()} {symbol} 的{description}费用或价格缺失")
        return quantity * multipliers[symbol] * price * rate

    for i, day in enumerate(index):
        day = pd.Timestamp(day)
        cost = entries_count = exits_count = roll_count = blocked_count = 0
        start_equity = equity
        peak_equity = max(peak_equity, equity)
        drawdown_at_open = equity / peak_equity - 1.0
        year = int(day.year)
        margin_at_open = sum(
            lots[s] * multipliers[s] * float(
                open_price.at[day, s] if np.isfinite(open_price.at[day, s]) else last_price[s]
            ) * margin_rate for s in columns if lots[s]
        )
        margin_call = margin_at_open > equity
        if margin_call and not margin_call_active:
            event(day, "保证金不足", detail=f"需求 {margin_at_open:.2f} > 权益 {equity:.2f}")
        margin_call_active = margin_call
        if previous_drawdown > -drawdown_limit and drawdown_at_open <= -drawdown_limit:
            stop_new_until_year = max(stop_new_until_year, year + 1)
            pause_until_year = year + 1
            forced_liquidation = True
            event(day, "回撤熔断清仓触发", detail=(
                f"峰值回撤 {drawdown_at_open:.2%}；停止新开仓并暂停至 {pause_until_year} 年"
            ))
        elif pause_until_year and year >= pause_until_year:
            pause_until_year = 0
            forced_liquidation = any(lot != 0 for lot in lots.values())
            event(day, "熔断暂停期结束")
        if margin_call:
            forced_liquidation = True
        elif not pause_until_year and not margin_call:
            forced_liquidation = False
        stop_new = (
            year < stop_new_until_year or year < pause_until_year
            or forced_liquidation or margin_call
        )

        for symbol in columns:
            raw_open, raw_close = open_price.at[day, symbol], close_price.at[day, symbol]
            prior_close = last_close[symbol]
            has_open = bool(np.isfinite(raw_open) and raw_open > 0)
            if has_open:
                raw_open = float(raw_open)
                last_price[symbol] = raw_open
            if np.isfinite(raw_close):
                last_close[symbol] = float(raw_close)
            if lots[symbol] and (not np.isfinite(raw_open) or raw_open <= 0):
                if np.isfinite(last_price[symbol]) and last_price[symbol] > 0:
                    raw_open = last_price[symbol]
                else:
                    raise ValueError(f"{day.date()} {symbol} 持仓开盘价缺失")
            day_locked = not has_open or bool(locked[i, columns.get_loc(symbol)])
            wants_entry = bool(entries.at[day, symbol])
            if bool(rolls.at[day, symbol]) and lots[symbol]:
                pending_roll.add(symbol)
            exit_due = bool(lots[symbol]) and (
                age[symbol] >= holding_days or forced_liquidation or margin_call
            )
            if lots[symbol] and exit_due:
                if day_locked:
                    blocked_count += 1
                    reason = "强平/到期退出遇单价锁定代理，延迟至可成交日"
                    if blocked_reasons.get(symbol) != reason:
                        event(day, "退出受阻", symbol, lots[symbol], raw_open, reason)
                    blocked_reasons[symbol] = reason
                else:
                    quantity = lots[symbol]
                    cost += charge(close_fee, day, symbol, quantity, raw_open, "平仓")
                    cost += charge(slippage, day, symbol, quantity, raw_open, "平仓滑点")
                    exits_count += 1
                    event(day, "强制平仓" if forced_liquidation or margin_call else "到期平仓",
                          symbol, quantity, raw_open, "多头；连续主力开盘价代理")
                    lots[symbol] = age[symbol] = 0
                    contracts[symbol] = None
                    pending_roll.discard(symbol)
                    blocked_reasons.pop(symbol, None)
                    continue
            if lots[symbol] and symbol in pending_roll:
                if day_locked:
                    blocked_count += 1
                    reason = "换月遇单价锁定代理，延迟至可成交日"
                    if blocked_reasons.get(symbol) != reason:
                        event(day, "换月受阻", symbol, lots[symbol], raw_open, reason)
                    blocked_reasons[symbol] = reason
                else:
                    contract = contract_codes.at[day, symbol]
                    if not isinstance(contract, str) or not contract:
                        raise ValueError(f"{day.date()} {symbol} 缺少新主力合约代码")
                    if not np.isfinite(prior_close) or prior_close <= 0:
                        raise ValueError(f"{day.date()} {symbol} 换月缺少旧合约收盘价")
                    quantity = lots[symbol]
                    cost += charge(roll_close_fee, day, symbol, quantity, prior_close, "换月平仓")
                    cost += charge(slippage, day, symbol, quantity, prior_close, "换月平仓滑点")
                    cost += charge(open_fee, day, symbol, quantity, raw_open, "换月开仓")
                    cost += charge(slippage, day, symbol, quantity, raw_open, "换月开仓滑点")
                    old_contract = contracts[symbol]
                    contracts[symbol] = contract
                    pending_roll.discard(symbol)
                    roll_count += 1
                    event(day, "换月", symbol, quantity, raw_open,
                          f"多头；{old_contract or '未知'} -> {contract}；旧合约平仓价用前一交易日收盘")
            if lots[symbol]:
                if wants_entry:
                    event(day, "信号重叠跳过", symbol, detail="已有持仓或退出受阻")
                continue
            if not wants_entry:
                continue
            if stop_new or equity - cost <= 0:
                event(day, "入场暂停", symbol, detail="熔断或保证金风控")
                continue
            if day_locked:
                blocked_count += 1
                event(day, "开仓受阻", symbol, detail=(
                    "单价锁定代理；本次信号不延迟追单" if has_open
                    else "当日无可用开盘价；本次信号不延迟追单"
                ))
                continue
            if not np.isfinite(raw_open) or raw_open <= 0:
                raise ValueError(f"{day.date()} {symbol} 入场开盘价缺失")
            contract = contract_codes.at[day, symbol]
            if not isinstance(contract, str) or not contract:
                raise ValueError(f"{day.date()} {symbol} 入场缺少主力合约代码")
            members = universe.get(year, [])
            if symbol not in members or not members:
                raise ValueError(f"{day.date()} {symbol} 入场时不在年度流动性池")
            current_notional = sum(
                lots[s] * multipliers[s] * float(
                    open_price.at[day, s] if np.isfinite(open_price.at[day, s]) else last_price[s]
                ) for s in columns if lots[s]
            )
            available_equity = max(equity - cost, 0.0)
            budget = min(
                exposure_limit * available_equity / len(members),
                max(exposure_limit * available_equity - current_notional, 0.0),
            )
            quantity = int(math.floor(budget / (raw_open * multipliers[symbol])))
            if quantity < 1:
                event(day, "不足一手", symbol, detail=f"目标名义金额 {budget:.2f}")
                continue
            cost += charge(open_fee, day, symbol, quantity, raw_open, "开仓")
            cost += charge(slippage, day, symbol, quantity, raw_open, "开仓滑点")
            lots[symbol] = quantity
            age[symbol] = 0
            contracts[symbol] = contract
            entries_count += 1
            event(day, "开仓", symbol, quantity, raw_open, f"多头；{contract}；目标名义金额 {budget:.2f}")

        gross_pnl = 0.0
        for symbol in columns:
            if lots[symbol]:
                delta = adjusted_open_delta.at[day, symbol]
                if not np.isfinite(delta):
                    raise ValueError(f"{day.date()} {symbol} 持仓期间复权开盘变动缺失")
                if not np.isfinite(adjusted_open.at[day, symbol]):
                    raise ValueError(f"{day.date()} {symbol} 持仓期间复权开盘价缺失")
                gross_pnl += lots[symbol] * multipliers[symbol] * float(delta)
                age[symbol] += 1
        equity = float(equity - cost + gross_pnl)
        writeoff = 0.0
        if equity <= 0:
            event(day, "账户权益耗尽", detail=f"本日损益 {gross_pnl - cost:.2f}")
            writeoff = -equity
            equity = 0.0
            forced_liquidation = True
        net_pnl = float(gross_pnl - cost + writeoff)
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0 if peak_equity else -1.0
        previous_drawdown = drawdown_at_open
        notional = sum(
            lots[s] * multipliers[s] * float(
                open_price.at[day, s] if np.isfinite(open_price.at[day, s]) else last_price[s]
            ) for s in columns if lots[s]
        )
        margin = notional * margin_rate
        daily_rows.append({
            "交易日": day, "期初权益": start_equity, "毛盈亏": gross_pnl,
            "交易费用": cost, "净盈亏": net_pnl, "权益清零调整": writeoff,
            "权益": equity, "净值": equity / initial_capital, "峰值回撤": drawdown,
            "名义敞口": notional, "敞口倍数": notional / equity if equity > 0 else np.nan,
            "保证金需求_20pct_assumption": margin,
            "保证金权益占用": margin / equity if equity > 0 else np.nan,
            "持仓品种数": sum(value != 0 for value in lots.values()),
            "持仓总手数": sum(lots.values()), "开仓笔数": entries_count,
            "平仓笔数": exits_count, "换月笔数": roll_count,
            "受阻委托数": blocked_count, "停止新开仓": bool(stop_new),
            "强制清仓待执行": bool(forced_liquidation), "保证金追缴": bool(margin_call),
        })
        for symbol in columns:
            if not lots[symbol]:
                continue
            price = open_price.at[day, symbol]
            price = price if np.isfinite(price) else last_price[symbol]
            if not np.isfinite(price):
                raise ValueError(f"{day.date()} {symbol} 持仓账本缺少开盘价")
            exposure = lots[symbol] * multipliers[symbol] * float(price)
            position_rows.append({
                "交易日": day, "品种": symbol, "合约": contracts[symbol],
                "手数": lots[symbol], "方向": "多", "主力原始开盘价": float(price),
                "合约乘数": multipliers[symbol], "名义敞口": exposure,
                "保证金_20pct_assumption": exposure * margin_rate,
                "已持有交易日": age[symbol], "强平待执行": forced_liquidation,
            })

    daily = pd.DataFrame(daily_rows).set_index("交易日")
    expected_equity = (
        daily["期初权益"] - daily["交易费用"] + daily["毛盈亏"] + daily["权益清零调整"]
    )
    if not np.allclose(expected_equity, daily["权益"], rtol=1e-12, atol=1e-7):
        raise ArithmeticError("逐日权益无法由毛盈亏、费用和清零调整复算")
    daily["净收益"] = daily["净盈亏"] / daily["期初权益"].replace(0, np.nan)
    positions = pd.DataFrame(position_rows)
    if positions.empty:
        positions = pd.DataFrame(columns=[
            "交易日", "品种", "合约", "手数", "方向", "主力原始开盘价",
            "合约乘数", "名义敞口", "保证金_20pct_assumption", "已持有交易日",
            "强平待执行",
        ])
    return daily, positions, pd.DataFrame(events)


def run_dir(step: str) -> Path:
    d = C.RUNS_DIR / "commodity" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{step}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def clean_json(o):
    """numpy 标量转 Python，NaN/inf 转 None。"""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o

def dump_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2), encoding='utf-8')

def rebalance(signal, days=1, mode="single", phase=0):
    """收盘目标仓位：单一周期组合，或等权错开的几组。"""
    if int(days) != days or days < 1 or mode not in ("single","staggered"):
        raise ValueError("调仓周期须为正整数，模式为 single/staggered")
    if not 0 <= phase < days:
        raise ValueError("调仓相位越界")
    order = np.arange(len(signal)) % days
    if mode == "staggered":
        return sum(rebalance(signal, days, "single", k) for k in range(days))/days
    scheduled = pd.Series(order == phase, index=signal.index)
    return signal.fillna(0.).where(scheduled, axis=0).ffill().fillna(0.)

def position(signal, vol=None, vol_target=.20, cap=1., days=1, mode="single", phase=0):
    if vol_target:
        if vol is None:
            raise ValueError("波动率目标需要事前波动率")
        ann = vol.reindex_like(signal)*np.sqrt(C.TRADING_DAYS_PER_YEAR)
        signal = signal * (vol_target/ann.where(ann>0)).clip(upper=cap)
    return rebalance(signal.clip(-cap,cap), days, mode, phase).shift(1).fillna(0.)

def allocate(position, universe):
    out = position*0.
    for year, members in universe.items():
        members = list(dict.fromkeys(members))
        present = [s for s in members if s in position.columns]
        if members:
            mask = position.index.year == int(year)
            out.loc[mask,present] = position.loc[mask,present].fillna(0.)/len(members)
    return out.fillna(0.)

def trade_legs(exposure, rolls=None):
    previous = exposure.shift(1).fillna(0.)
    same = exposure*previous > 0
    retained = pd.DataFrame(np.minimum(exposure.abs(), previous.abs()),
        index=exposure.index, columns=exposure.columns).where(same,0.)
    if rolls is not None:
        retained = retained.where(~rolls.reindex_like(exposure).fillna(False).astype(bool),0.)
    return exposure.abs()-retained, previous.abs()-retained

def backtest(position, returns, universe, open_fee=0., close_fee=0., slippage=0.,
             rolls=None, roll_close_fee=None):
    position = position.reindex_like(returns)
    exposure = allocate(position, universe)
    opened, closed = trade_legs(exposure, rolls)
    if ((exposure != 0) & ~np.isfinite(returns)).any().any():
        raise ValueError("持仓对应的市场收益缺失，不能跳过品种或填零")
    def cost(size, rate):
        if isinstance(rate,pd.DataFrame):
            rate = rate.reindex_like(size)
        values = (size*rate).where(size>0,0.)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("实际成交对应的成本缺失")
        return values
    closing = close_fee
    if rolls is not None and roll_close_fee is not None:
        closing = close_fee.where(~rolls.reindex_like(exposure).fillna(False), roll_close_fee)
    charge = cost(opened,open_fee)+cost(closed,closing)+cost(opened+closed,slippage)
    gross = (exposure*returns.fillna(0.)).sum(axis=1)
    return pd.DataFrame({"gross":gross, "net":gross-charge.sum(axis=1),
        "turnover":(opened+closed).sum(axis=1), "cost":charge.sum(axis=1)})

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

def load_research(n_ticks=1.):
    universe = {y:s for y,s in load_universe().items() if 2016<=y<=2021}
    symbols = sorted({s for ss in universe.values() for s in ss})
    absent = set(symbols)-set(list_shards(C.RESEARCH_DIR))
    if absent:
        raise FileNotFoundError(f"研究期品种池缺分钟分片: {sorted(absent)}")
    factors = load(symbols)
    prices = factors.bars['open']
    returns = (factors.bars['openw'].shift(-1)-factors.bars['openw'])/prices
    C.assert_no_holdout_dates(returns.index,'研究收益')
    # Final open has no next open in this partition; never borrow 2022 prices.
    index = returns.index[(returns.index>=pd.Timestamp(C.STUDY_START)) & returns.notna().any(axis=1)]
    returns = returns.loc[index]
    rolls = pd.DataFrame(False,index=index,columns=symbols)
    for symbol,frame in load_roll_calendar(symbols,C.RESEARCH_END).items():
        days = index.intersection(frame.trading_date)
        rolls.loc[days,symbol] = True
    o,c,r = fee_tables(load_fees(),prices)
    slip = slippage_tables(load_ticks(),prices.loc[index],n_ticks)
    return StudyData(factors,returns,universe,rolls,o.loc[index],c.loc[index],r.loc[index],slip)

def validate_specs(specs):
    seen = set()
    allowed = {'id','factors','timing','combine','days','mode','phase','vol_target','cap','hypothesis'}
    for spec in specs:
        if set(spec)-allowed:
            raise ValueError(f"未知候选字段 {set(spec)-allowed}")
        if spec['id'] in seen:
            raise ValueError('候选 id 重复')
        seen.add(spec['id'])
        if not spec.get('factors'):
            raise ValueError('候选没有因子')
        for name,weight in spec['factors'].items():
            if name not in SIGNED_PRIORS or not np.isfinite(weight) or weight==0:
                raise ValueError(f'无效因子/权重: {name}:{weight}')
        if spec.get('timing','z') not in ('z','quantile') or spec.get('combine','mean') not in ('mean','agree','filter'):
            raise ValueError('无效信号/组合方法')
        days,phase = spec.get('days',1),spec.get('phase',0)
        if not isinstance(days,int) or days<1 or not isinstance(phase,int) or not 0<=phase<days:
            raise ValueError('无效调仓周期/相位')
        if spec.get('mode','single') not in ('single','staggered'):
            raise ValueError('无效调仓模式')
        if spec.get('mode','single') == 'staggered' and phase != 0:
            raise ValueError('错开持有已经平均全部相位，不接受额外相位选择')
        if not 0 < spec.get('cap',1.) <= 1. or not np.isfinite(spec.get('vol_target',.20)) or spec.get('vol_target',.20)<0:
            raise ValueError('本研究只支持最大一倍名义暴露，非负波动目标')

def combined(data,factors,timing='z',method='mean'):
    frames,weights = [],[]
    for name,weight in factors.items():
        # JSON weights multiply the already economically directed factor.
        direction = SIGNED_PRIORS[name]*np.sign(weight)
        frames.append(to_signal(name,data.factors.raw(name),timing,direction))
        weights.append(abs(weight))
    return combine(frames,method,weights)

def signal_for(data,spec):
    timing = spec.get('timing','z')
    return combined(data,spec['factors'],timing,spec.get('combine','mean'))

def run_spec(data,spec):
    signal = signal_for(data,spec)
    held = position(signal,data.factors.vol,spec.get('vol_target',.20),
        spec.get('cap',1.),spec.get('days',1),spec.get('mode','single'),spec.get('phase',0))
    return backtest(held.reindex_like(data.returns),data.returns,data.universe,
        data.open_fee,data.close_fee,data.slippage,data.rolls,data.roll_close_fee)

def performance_rows(result,spec):
    periods = [('2016-2021',result)]
    periods += [(str(year),result.loc[result.index.year==year]) for year in range(2016,2022)]
    rows = []
    for label,frame in periods:
        row = {'id':spec['id'],'period':label}
        for kind in ('gross','net'):
            row.update({kind+'_'+k:v for k,v in performance(frame[kind]).items()})
            row[kind+'_sharpe'] = sharpe_ratio(frame[kind])
        row['annual_turnover'] = (
            frame.turnover.sum() * C.TRADING_DAYS_PER_YEAR / len(frame)
            if len(frame) else np.nan
        )
        row['annual_cost'] = (
            frame.cost.sum() * C.TRADING_DAYS_PER_YEAR / len(frame)
            if len(frame) else np.nan
        )
        rows.append(row)
    return rows

def factor_diagnostics(data,horizons=(1,3,5,10)):
    """只算原始因子的 IC，Spearman 和标准化预测分数分开记录。"""
    rows = []
    historical = (data.factors.bars['openw'].shift(-1)-data.factors.bars['openw'])/data.factors.bars['open']
    for name in data.factors.signed:
        raw = data.factors.raw(name)
        for h in horizons:
            forward = holding_forward_return(historical,h)
            function = cross_sectional_ic_table if name.startswith('cs_') else factor_ic_table
            table = function(raw,forward,data.universe,name,list(range(2016,2022)))
            table = table.rename(columns={'ic':'spearman_ic','ic_ts':'predictive_score','t':'score_nw_t'})
            table.insert(0,'horizon',h)
            rows.extend(table.to_dict('records'))
    return pd.DataFrame(rows)

def provenance(specs,n_ticks):
    roots = (C.PROJECT_ROOT / 'src', C.PROJECT_ROOT / 'research', C.PROJECT_ROOT / 'backtest')
    files = [path for root in roots for path in sorted(root.rglob('*.py'))]
    hashes = {str(p.relative_to(C.PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source_hashes = {}
    for p in [C.UNIVERSE_DIR/'universe_by_year.json',C.RESEARCH_OUT_DIR/'fee_history.csv',
              C.RESEARCH_OUT_DIR/'tick_size.csv']:
        source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    for directory in (timestamp_dir(),report_dir(),
                      combo_dir(C.IC_REFERENCE_LOOKBACK,C.IC_REFERENCE_PCT),
                      C.FACTOR_DAILY_DIR/'external'/'research'):
        for p in sorted(directory.glob('*.parquet')) + sorted(directory.glob('*.pkl')):
            source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {'status':'research_only_unfrozen','candidates':specs,'slippage_ticks_per_side':n_ticks,
        'cache_version':CACHE_VERSION,'code_sha256':hashes,'input_sha256':source_hashes,
        'window':'2016-2021','execution':'close signal, next trading-date open, open-to-open returns',
        'roll_fee':'old-contract close rate approximated by preceding dominant rate',
        'tick_source':'previous-year estimates, current raw open denominator',
        'phase_checks':'all single-book phases, reported separately; none automatically selected',
        'missing_factor':'fixed weight left in cash','validation_2022':'not_run','oos_2023_2025':'locked'}

def plot_performance(returns, out_dir, title="净值与最大回撤", series=None) -> Path:
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
        "Microsoft YaHei", "SimHei", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    nav = (1 + frame).cumprod()
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1
    fig, (nav_axis, drawdown_axis) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    nav.plot(ax=nav_axis, lw=1.2)
    nav_axis.set_title(title)
    nav_axis.set_ylabel("净值")
    nav_axis.grid(alpha=0.3)
    nav_axis.legend(loc="upper left", fontsize=8)
    drawdown.plot(ax=drawdown_axis, lw=1.0, legend=False)
    drawdown_axis.fill_between(drawdown.index, drawdown.min(axis=1), 0, alpha=0.2)
    drawdown_axis.set_ylabel("回撤")
    drawdown_axis.grid(alpha=0.3)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "performance.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


VAL_COLUMNS = ["trading_date", "open", "openw", "close", "closew", "highw", "loww"]
VAL_END = pd.Timestamp("2022-12-31")
RESEARCH_YEARS = range(2016, 2022)
LONG_ONLY_SECTORS = {
    "农产品": "A B C CS M Y P JD LH RR AP CF CY SR OI RM RS PK CJ WH PM RI LR JR".split(),
    "黑色钢铁": "I J JM RB HC WR SS SF SM ZC".split(),
    "有色金属": "CU AL ZN PB NI SN BC AO AD SI LC PS".split(),
    "贵金属": "AU AG".split(),
    "能源": "BU FU LU SC PG".split(),
    "化工": "L V PP EG EB RU NR BR SP TA MA FG SA UR PF PX PR SH".split(),
}
LONG_ONLY_SECTOR_POOLS = {
    "农产品+有色+化工等权板块池": ("农产品", "有色金属", "化工"),
    "黑色钢铁+有色金属等权板块池": ("黑色钢铁", "有色金属"),
}
OOS_COLUMNS = ["trading_date", "open", "openw", "close", "closew", "highw", "loww",
               "volume", "total_turnover"]


def _compound(ret: pd.Series) -> float:
    values = pd.Series(ret, dtype="float64").dropna()
    if values.empty or np.any(1.0 + values.to_numpy() <= 0):
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def _daily_equity_frame(result: pd.DataFrame, labels: dict) -> pd.DataFrame:
    daily = result[["gross", "net", "turnover", "cost"]].copy()
    daily["nav"] = (1.0 + daily["net"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax().clip(lower=1.0) - 1.0
    daily = daily.rename(columns={
        "gross": "毛收益",
        "net": "净收益",
        "turnover": "换手",
        "cost": "成本",
        "nav": "净值",
        "drawdown": "回撤",
    })
    daily.insert(0, "交易日", daily.index)
    for column, value in reversed(list(labels.items())):
        daily.insert(1, column, value)
    return daily.reset_index(drop=True)


def _exposure(position: pd.DataFrame, universe: dict, years) -> float:
    held = position.loc[position.index.year.isin(list(years))]
    if held.empty:
        return np.nan
    pools = {int(year): len(names) for year, names in universe.items()}
    width = pd.Series(held.index.year, index=held.index).map(pools).astype(float)
    return float((held.sum(axis=1) / width.replace(0, np.nan)).mean())


def _keep_last(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _validation_time_factor(history: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    low_clock = panel.pivot_table(
        index="trading_date", columns="symbol", values="ts_low", aggfunc="first",
    )
    high_clock = panel.pivot_table(
        index="trading_date", columns="symbol", values="ts_high", aggfunc="first",
    )
    return _keep_last([history, low_clock - high_clock])


def _newey_west_mean_test(values: pd.Series, lags: int = 4) -> tuple[float, float]:
    sample = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    array = sample.to_numpy()
    count = len(array)
    if count < max(20, lags + 3):
        return np.nan, np.nan
    centered = array - array.mean()
    variance = float(centered @ centered / count)
    used_lags = min(lags, count - 1)
    for lag in range(1, used_lags + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / count)
        variance += 2.0 * (1.0 - lag / (used_lags + 1.0)) * covariance
    standard_error = math.sqrt(max(variance, 0.0) / count)
    t_value = float(array.mean() / standard_error) if standard_error > 0 else np.nan
    p_value = math.erfc(abs(t_value) / math.sqrt(2.0)) if np.isfinite(t_value) else np.nan
    return t_value, p_value


def _benjamini_hochberg(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype="float64")
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted




def _run_integer_lot_period(
    period: str,
    data: StudyData,
    factor: pd.DataFrame,
    initial_capital: float = 100_000_000.0,
) -> None:
    """Run the fixed long-only integer-lot portfolio for one isolated period."""
    from strategies.ts_low import build_entry_schedule

    period_years = {
        "research": tuple(RESEARCH_YEARS),
        "validation": (2022,),
        "oos": tuple(range(C.STRICT_OOS_START.year, C.DEFAULT_OOS_END.year + 1)),
    }[period]
    start_date = pd.Timestamp(f"{period_years[0]}-01-01")
    end_date = pd.Timestamp(f"{period_years[-1]}-12-31")
    if period == "research" and (
        data.returns.empty or data.returns.index.max() >= pd.Timestamp(C.HOLDOUT_START)
    ):
        raise C.HoldoutViolation("整数手数研究收益越过 2021")
    if period == "validation":
        validation_dates = data.returns.index[data.returns.index.year == 2022]
        C.assert_validation_2022_dates(validation_dates, "整数手数验证收益")
    if period == "oos" and (
        data.returns.empty
        or data.returns.index.min() < pd.Timestamp(C.STRICT_OOS_START)
        or data.returns.index.max() > pd.Timestamp(C.DEFAULT_OOS_END)
    ):
        raise C.HoldoutViolation("整数手数样本外收益超出 2023–2025")

    sector_names = ("农产品", "黑色钢铁", "有色金属", "化工")
    symbols = sorted({
        symbol
        for sector in sector_names
        for symbol in LONG_ONLY_SECTORS[sector]
        if symbol in data.returns.columns
        and any(
            symbol in members and int(year) in period_years
            for year, members in data.universe.items()
        )
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
        raise ValueError("四板块与逐年流动性池没有交集")

    index = data.returns.index[data.returns.index.year.isin(period_years)]
    if index.empty:
        raise ValueError(f"{period} 整数手数回测没有可计分的交易日")
    columns = pd.Index(selected_symbols)
    prices = data.factors.bars
    base_entries = build_entry_schedule(
        factor.reindex(columns=columns),
        data.returns.reindex(columns=columns),
        selected_universe,
        horizon=5,
        lookback=250,
        quantile=0.8,
        start=start_date,
        end=end_date,
        require_exit_day=True,
    ).reindex(index=index, columns=columns, fill_value=False)

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
    strategy = "纯多头_基准"
    entries = base_entries.astype("int8")

    roll_calendar = load_roll_calendar(selected_symbols, end_date.date())
    contract_codes = pd.DataFrame(index=index, columns=columns, dtype=object)
    for symbol in selected_symbols:
        calendar = roll_calendar.get(symbol)
        if calendar is None or calendar.empty:
            raise FileNotFoundError(f"{symbol} 缺少研究期主力合约换月表")
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

    annual_rows = []
    daily_frames = []
    position_frames = []
    event_frames = []
    summary_rows = []
    output_dir = C.RUNS_DIR / "strategies" / "ts_low"
    output_dir.mkdir(parents=True, exist_ok=True)

    daily, positions, events = simulate_integer_lots(
        entries=entries,
        open_price=open_price,
        close_price=close_price,
        adjusted_open=adjusted_open,
        adjusted_open_delta=adjusted_open_delta,
        adjusted_high=adjusted_high,
        adjusted_low=adjusted_low,
        rolls=rolls,
        contract_codes=contract_codes,
        universe=selected_universe,
        multipliers=multipliers,
        open_fee=fee_open,
        close_fee=fee_close,
        roll_close_fee=fee_roll,
        slippage=slip,
        initial_capital=float(initial_capital),
        margin_rate=0.20,
        holding_days=5,
        exposure_limit=1.5,
        drawdown_limit=0.20,
    )
    if period == "research":
        C.assert_no_holdout_dates(daily.index, "整数手数研究净值")
    elif period == "validation":
        C.assert_validation_2022_dates(daily.index, "整数手数验证净值")
    else:
        if daily.index.min() < pd.Timestamp(C.STRICT_OOS_START):
            raise C.HoldoutViolation("整数手数样本外净值进入2022")
        C.assert_oos_research_locked(daily.index.max())
    daily["策略"] = strategy
    positions.insert(0, "策略", strategy)
    if not events.empty:
        events.insert(0, "策略", strategy)
    daily_frames.append(daily.reset_index())
    position_frames.append(positions)
    event_frames.append(events)
    returns = daily["净收益"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    metrics = performance(returns)
    summary_rows.append({
        "策略": strategy,
        "板块池": "+".join(sector_names),
        "参数": "250日回看/前20%/持有5日/名义敞口上限1.5倍/回撤熔断20%",
        "初始资金": float(initial_capital),
        "池内品种数": len(selected_symbols),
        "名义敞口上限": 1.5,
        "保证金率假设": 0.20,
        "累计净收益": float(daily["权益"].iloc[-1] / initial_capital - 1.0),
        "年化收益": metrics["ann_return"],
        "Sharpe": sharpe_ratio(returns),
        "最大回撤": float(daily["峰值回撤"].min()),
        "期末权益": float(daily["权益"].iloc[-1]),
        "最大名义敞口倍数": float(daily["敞口倍数"].max()),
        "最大保证金权益占用": float(daily["保证金权益占用"].max()),
        "开仓笔数": int(daily["开仓笔数"].sum()),
        "平仓笔数": int(daily["平仓笔数"].sum()),
        "换月笔数": int(daily["换月笔数"].sum()),
        "涨跌停代理阻单数": int(daily["受阻委托数"].sum()),
    })

    for year in period_years:
        annual = daily.loc[daily.index.year == year, "净收益"].fillna(0.0)
        annual_nav = (1.0 + annual).cumprod()
        annual_peak = np.maximum.accumulate(np.r_[1.0, annual_nav.to_numpy()])[1:]
        annual_drawdown = annual_nav.to_numpy() / annual_peak - 1.0
        annual_rows.append({
            "策略": strategy,
            "年份": year,
            "净收益": float((1.0 + annual).prod() - 1.0),
            "Sharpe": sharpe_ratio(annual),
            "期末权益": float(daily.loc[daily.index.year == year, "权益"].iloc[-1]),
            "年度内最大回撤": float(annual_drawdown.min()) if len(annual_drawdown) else np.nan,
            "开仓笔数": int(daily.loc[daily.index.year == year, "开仓笔数"].sum()),
            "平仓笔数": int(daily.loc[daily.index.year == year, "平仓笔数"].sum()),
        })

    period_label = {
        "research": "2016–2021研究期",
        "validation": "2022验证期",
        "oos": "2023–2025 OOS",
    }[period]
    plot_performance(
        daily["净收益"].rename(strategy),
        output_dir / "plots" / f"integer_lot_{period}" / strategy,
        title=f"{period_label}整数手数：{strategy}，1亿元、1.5倍敞口、20%保证金假设",
    )

    stem = f"integer_lot_{period}"
    summary_path = output_dir / f"{stem}.csv"
    annual_path = output_dir / f"{stem}_annual.csv"
    daily_path = output_dir / f"{stem}_daily.csv"
    positions_path = output_dir / f"{stem}_positions.csv"
    events_path = output_dir / f"{stem}_events.csv"
    pd.DataFrame(summary_rows).to_csv(
        summary_path, index=False, encoding="utf-8-sig", float_format="%.10f",
    )
    pd.DataFrame(annual_rows).to_csv(
        annual_path, index=False, encoding="utf-8-sig", float_format="%.10f",
    )
    pd.concat(daily_frames, ignore_index=True).to_csv(
        daily_path, index=False, encoding="utf-8-sig", float_format="%.10f",
    )
    pd.concat(position_frames, ignore_index=True).to_csv(
        positions_path, index=False, encoding="utf-8-sig", float_format="%.10f",
    )
    pd.concat(event_frames, ignore_index=True).to_csv(
        events_path, index=False, encoding="utf-8-sig", float_format="%.10f",
    )
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(f"年度结果：{annual_path}", flush=True)
    print(f"逐日权益：{daily_path}", flush=True)
    print(f"整数手持仓：{positions_path}", flush=True)
    print(f"订单/风控事件：{events_path}", flush=True)


def run_integer_lot_research(initial_capital=100_000_000.0):
    from strategies.ts_low import load_config

    load_config(allow_frozen=True)
    data = load_research(1.0)
    factor = data.factors.raw("ts_low") - data.factors.raw("ts_high")
    _run_integer_lot_period("research", data, factor, initial_capital)


def run_integer_lot_validation(initial_capital=100_000_000.0):
    from strategies.ts_low import load_config

    load_config(allow_frozen=True)
    data, factor = load_validation(1.0)
    _run_integer_lot_period("validation", data, factor, initial_capital)


def run_integer_lot_oos(initial_capital=100_000_000.0):
    from strategies.ts_low import chosen_parameters, load_config

    config = load_config(allow_frozen=True)
    chosen = chosen_parameters(config)
    if (
        chosen["回看交易日"] != 250
        or chosen["分位"] != 0.8
        or chosen["持仓交易日"] != 5
        or config["parameters"].get("direction") != "upper_tail_long"
    ):
        raise ValueError(
            "整数手数OOS要求先冻结纯多头参数：250日回看、quantile=0.8、持有5日"
        )
    end = C.DEFAULT_OOS_END
    with C.final_evaluation(
        config, _oos_audit_files(), end, strategy=f"{config['id']}_integer_lots",
        note="ts_low fixed integer-lot OOS backtest",
    ):
        if not list_shards(C.HOLDOUT_DIR):
            from tfcta.prepare.shard import shard_oos_from_monolith

            results = shard_oos_from_monolith(C.MINUTE_MONOLITH)
            written = sum(item["status"] == "ok" for item in results)
            print(f"已从锁定的分钟单体生成 {written} 个 2023–2025 分片", flush=True)
        data, factor = load_oos(end)
        _run_integer_lot_period("oos", data, factor, initial_capital)


def _validation_clock(symbol: str) -> pd.DataFrame:
    frame = add_intraday_coords(load_validation_shard(symbol, columns=VAL_COLUMNS))
    clock = timestamp_factors(frame).join(
        daily_bars(frame)[["open", "openw", "close", "closew", "highw", "loww"]],
        how="outer",
    )
    C.assert_validation_2022_dates(clock.index, f"{symbol} 2022")
    return clock


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


def load_validation(n_ticks=1.0):
    """研究期缓存接到 2022。不读 2023 及以后。"""
    research = load_research(n_ticks)
    names, _detail = validation_universe()
    ticks = load_ticks("research")
    fees = pd.concat([load_fees("research"), load_fees("validation_2022")], ignore_index=True)
    _require_cost_coverage(names, ticks, fees, "2022 验证")
    missing = [symbol for symbol in names if find_shard(C.VALIDATION_DIR, symbol) is None]
    if missing:
        raise FileNotFoundError(f"2022 品种池缺验证分片: {missing}")
    panel = pd.concat([_validation_clock(symbol).assign(symbol=symbol) for symbol in names])
    panel.index.name = "trading_date"
    panel = panel.reset_index()
    opened = panel.pivot_table(index="trading_date", columns="symbol", values="open", aggfunc="first")
    adjusted = panel.pivot_table(index="trading_date", columns="symbol", values="openw", aggfunc="first")
    closed = panel.pivot_table(index="trading_date", columns="symbol", values="close", aggfunc="first")
    adjusted_close = panel.pivot_table(index="trading_date", columns="symbol", values="closew", aggfunc="first")
    bar_high = panel.pivot_table(index="trading_date", columns="symbol", values="highw", aggfunc="first")
    bar_low = panel.pivot_table(index="trading_date", columns="symbol", values="loww", aggfunc="first")
    history = research.factors.raw("ts_low") - research.factors.raw("ts_high")
    factor = _validation_time_factor(history, panel)
    open_px = _keep_last([research.factors.bars["open"], opened])
    open_w = _keep_last([research.factors.bars["openw"], adjusted])
    close_px = _keep_last([research.factors.bars["close"], closed])
    close_w = _keep_last([research.factors.bars["closew"], adjusted_close])
    high_w = _keep_last([research.factors.bars["highw"], bar_high])
    low_w = _keep_last([research.factors.bars["loww"], bar_low])
    returns = (open_w.shift(-1) - open_w) / open_px
    returns = returns.loc[(returns.index >= pd.Timestamp(C.STUDY_START)) & (returns.index <= VAL_END)]
    returns = returns.loc[returns.notna().any(axis=1)]
    if returns.empty or returns.index.max() >= pd.Timestamp(C.STRICT_OOS_START):
        raise C.HoldoutViolation("验证收益越过 2022")
    symbols = list(returns.columns)
    index = returns.index
    rolls = pd.DataFrame(False, index=index, columns=symbols)
    for symbol, frame in load_roll_calendar(symbols, VAL_END.date()).items():
        if symbol not in rolls.columns:
            continue
        rolls.loc[rolls.index.intersection(frame.trading_date), symbol] = True
    prices = open_px.reindex(index).reindex(columns=symbols)
    opened_fee, closed_fee, roll_fee = fee_tables(fees, prices)
    slip = slippage_tables(ticks, prices, n_ticks)
    universe = {int(year): list(pool) for year, pool in research.universe.items()}
    universe[2022] = list(names)
    data = StudyData(
        SignalSet(
            bars={
                "open": open_px, "openw": open_w, "close": close_px,
                "closew": close_w, "highw": high_w, "loww": low_w,
            },
            signed=research.factors.signed,
        ),
        returns, universe, rolls.reindex(columns=symbols).fillna(False),
        opened_fee, closed_fee, roll_fee, slip,
    )
    return data, factor.reindex(columns=symbols)


def load_oos(end=None, n_ticks=1.0):
    """只在最终评估守卫打开时读取冻结测试数据。"""
    from tfcta.costs import estimate_ticks

    end = C.to_date(end or C.final_evaluation_end())
    C.assert_oos_research_locked(end)
    holdout_symbols = set(list_shards(C.HOLDOUT_DIR))
    pools, _detail = oos_universe(C.COMMODITY_SYMBOLS, end)
    excluded = {
        symbol for members in pools.values() for symbol in members
        if symbol not in holdout_symbols
    }
    if excluded:
        print(f"样本外品种池缺锁定分片，不进组合: {sorted(excluded)}", flush=True)
    symbols = sorted({symbol for members in pools.values() for symbol in members}
                     & holdout_symbols)
    if not symbols:
        raise ValueError("样本外品种池没有可用的锁定行情分片")

    high_factors, low_factors = {}, {}
    opens, adjusted_opens, closes, highs, lows = {}, {}, {}, {}, {}
    tick_parts = []
    available_years = {}
    for symbol in symbols:
        minute = load_minutes(
            symbol, include_validation=True, oos_end=end, columns=OOS_COLUMNS,
        )
        minute = add_intraday_coords(minute)
        clock = timestamp_factors(minute)
        bars = daily_bars(minute)
        bars = bars.loc[bars.index <= pd.Timestamp(end)]
        high_factors[symbol] = clock["ts_high"]
        low_factors[symbol] = clock["ts_low"]
        opens[symbol] = bars["open"]
        adjusted_opens[symbol] = bars["openw"]
        closes[symbol] = bars["close"]
        highs[symbol] = bars["highw"]
        lows[symbol] = bars["loww"]
        available_years[symbol] = set(bars.index.year)

        ticks = estimate_ticks(raw_price_path(minute), minute["trading_date"])
        ticks = ticks[
            (ticks["year"] >= C.STRICT_OOS_START.year)
            & (ticks["year"] <= end.year)
        ].copy()
        if not ticks.empty:
            ticks["symbol"] = symbol
            tick_parts.append(ticks)

    universe = {
        int(year): [
            symbol for symbol in members
            if symbol in available_years and int(year) in available_years[symbol]
        ]
        for year, members in pools.items()
    }
    for year, members in pools.items():
        removed = sorted(set(members) - set(universe[int(year)]))
        if removed:
            print(f"{year} 年样本外品种没有当年行情，不进组合: {removed}", flush=True)

    high_factor = pd.DataFrame(high_factors).sort_index()
    low_factor = pd.DataFrame(low_factors).sort_index()
    factor = low_factor - high_factor
    open_px = pd.DataFrame(opens).sort_index()
    openw = pd.DataFrame(adjusted_opens).sort_index()
    close_px = pd.DataFrame(closes).sort_index()
    highw = pd.DataFrame(highs).sort_index()
    loww = pd.DataFrame(lows).sort_index()
    all_returns = (openw.shift(-1) - openw) / open_px
    index = all_returns.index[
        (all_returns.index >= pd.Timestamp(C.STRICT_OOS_START))
        & (all_returns.index <= pd.Timestamp(end))
        & all_returns.notna().any(axis=1)
    ]
    if len(index) == 0:
        raise ValueError(f"样本外窗口没有可计分的收益: {C.STRICT_OOS_START} 至 {end}")
    returns = all_returns.loc[index]
    prices = open_px.reindex(index)

    fees = pd.concat([
        load_fees("research"),
        load_fees("validation_2022"),
        load_fees("oos"),
    ], ignore_index=True)
    ticks = load_ticks("validation_2022")
    if tick_parts:
        ticks = pd.concat([ticks, *tick_parts], ignore_index=True)
    opened_fee, closed_fee, roll_fee = fee_tables(fees, prices)
    slip = slippage_tables(ticks, prices, n_ticks)
    rolls = pd.DataFrame(False, index=index, columns=symbols)
    for symbol, frame in load_roll_calendar(symbols, end).items():
        rolls.loc[rolls.index.intersection(frame.trading_date), symbol] = True

    return StudyData(
        factors=SignalSet(
            bars={
                "open": open_px, "openw": openw, "close": close_px,
                "highw": highw, "loww": loww,
            },
            signed={"ts_high": -high_factor, "ts_low": low_factor},
        ),
        returns=returns,
        universe=universe,
        rolls=rolls.reindex(columns=returns.columns).fillna(False),
        open_fee=opened_fee,
        close_fee=closed_fee,
        roll_close_fee=roll_fee,
        slippage=slip,
    ), factor.reindex(columns=returns.columns)


def _oos_audit_files() -> list[Path]:
    """Files whose contents determine the frozen strategy's OOS result."""
    from strategies import ts_low

    symbols = C.COMMODITY_SYMBOLS
    files = [
        Path(__file__),
        Path(ts_low.CONFIG_PATH),
        Path(ts_low.__file__),
        C.PROJECT_ROOT / "src" / "tfcta" / "config.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "data.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "costs.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "factors.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "prepare" / "shard.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "prepare" / "ticks.py",
        C.MINUTE_MONOLITH,
        C.UNIVERSE_DIR / "yearly_stats.csv",
        C.UNIVERSE_DIR / "screen_params.json",
        C.COMMODITY_DATA / "validation_2022" / "fee_history.csv",
        C.COMMODITY_DATA / "oos" / "fee_history.csv",
        C.RESEARCH_OUT_DIR / "fee_history.csv",
        C.RESEARCH_OUT_DIR / "tick_size.csv",
        C.COMMODITY_DATA / "validation_2022" / "tick_size.csv",
    ]
    for symbol in symbols:
        for directory in (C.RESEARCH_DIR, C.VALIDATION_DIR, C.HOLDOUT_DIR):
            shard = find_shard(directory, symbol)
            if shard is not None:
                files.append(shard)
        roll_file = C.ROLL_DIR / f"{symbol}.csv"
        if roll_file.is_file():
            files.append(roll_file)
    return [path for path in dict.fromkeys(files) if path.is_file()]


def _window_row(result, position, universe, years, labels) -> dict:
    net = result.loc[result.index.year.isin(list(years)), "net"]
    stats = performance(net)
    yearly = []
    for year in years:
        part = net.loc[net.index.year == year]
        if not part.notna().any():
            continue
        yearly.append(_compound(part))
    yearly = pd.Series(yearly, dtype="float64")
    return {
        **labels,
        "收益": _compound(net),
        "年化": stats["ann_return"],
        "波动": stats["ann_vol"],
        "夏普": sharpe_ratio(net),
        "回撤": stats["max_drawdown"],
        "平均仓位": _exposure(position, universe, years),
        "为正年份": int((yearly > 0).sum()) if len(yearly) else 0,
        "年份数": int(yearly.notna().sum()) if len(yearly) else 0,
        "最差年份": float(yearly.min()) if len(yearly) else np.nan,
        "交易日": int(net.notna().sum()),
    }


def run(period: str):
    """按研究、验证或已冻结的最终测试窗口计分。"""
    from strategies.ts_low import (
        build_positions, chosen_parameters, load_config, parameters,
    )

    if period == "oos":
        config = load_config(allow_frozen=True)
        chosen = chosen_parameters(config)
        end = C.DEFAULT_OOS_END
        files = _oos_audit_files()
        with C.final_evaluation(
            config, files, end, strategy=config["id"],
            note="ts_low frozen strategy OOS backtest",
        ):
            data, factor = load_oos(end)
            _run_period(
                data, factor, [chosen], end, range(C.STRICT_OOS_START.year, end.year + 1),
                "测试期", "oos.csv", allow_oos=True,
            )
        return

    config = load_config()
    if period == "research":
        data = load_research(1.0)
        factor = data.factors.raw("ts_low") - data.factors.raw("ts_high")
        end = pd.Timestamp(C.RESEARCH_END)
        years = RESEARCH_YEARS
        label = "研究期"
        filename = "research.csv"
    elif period == "validation":
        data, factor = load_validation(1.0)
        end = VAL_END
        years = range(2022, 2023)
        label = "验证期"
        filename = "validation.csv"
    else:
        raise ValueError("只能选 --research、--validation 或 --oos")
    _run_period(data, factor, list(parameters(config)), end, years, label, filename)








def _run_period(data, factor, specs, end, years, label, filename, *, allow_oos=False):
    from strategies.ts_low import build_positions

    rows = []
    daily_frames = []
    plot_root = C.RUNS_DIR / "strategies" / "ts_low" / "plots" / Path(filename).stem
    for spec in specs:
        position = build_positions(
            factor, data.returns, data.universe, spec["持仓交易日"],
            spec["回看交易日"], spec["分位"],
            start=pd.Timestamp(f"{min(years)}-01-01"), end=end,
        )
        result = backtest(
            position, data.returns, data.universe, data.open_fee, data.close_fee,
            data.slippage, data.rolls, data.roll_close_fee,
        )
        if not allow_oos and result.index.max() >= pd.Timestamp(C.STRICT_OOS_START):
            raise C.HoldoutViolation("净值进入 2023")
        row = _window_row(result, position, data.universe, years, spec)
        rows.append(row)
        period_result = result.loc[result.index.year.isin(list(years))]
        daily_frames.append(_daily_equity_frame(period_result, spec))
        print(
            f"{label} {spec['回看']} {spec['阈值']} {spec['持仓']} "
            f"收益 {row['收益']:.2%} 年化 {row['年化']:.2%} 夏普 {row['夏普']:.2f}",
            flush=True,
        )
        plot_name = (
            f"lookback_{spec['回看交易日']}_quantile_{spec['分位']:g}_"
            f"horizon_{spec['持仓交易日']}"
        )
        plot_performance(
            period_result[["net"]].rename(columns={"net": "净收益"}),
            plot_root / plot_name,
            title=(
                f"{label} ts_low 回看{spec['回看交易日']}日 "
                f"前{(1 - spec['分位']):.0%} 持有{spec['持仓交易日']}日"
            ),
        )
    table = pd.DataFrame(rows).drop(columns=["分位"])
    out = C.RUNS_DIR / "strategies" / "ts_low"
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    table.to_csv(path, index=False, encoding="utf-8-sig")
    daily_path = out / f"{Path(filename).stem}_daily.csv"
    pd.concat(daily_frames, ignore_index=True).to_csv(
        daily_path, index=False, encoding="utf-8-sig",
    )
    print(table.to_string(index=False), flush=True)
    print(f"结果：{path}", flush=True)
    print(f"逐日净值：{daily_path}", flush=True)
    print(f"净值与回撤图：{plot_root}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "运行 ts_low 整数手数回测；--oos 仅可用于已冻结策略的最终测试。"
        )
    )
    period = parser.add_mutually_exclusive_group(required=True)
    period.add_argument(
        "--research", action="store_true",
        help="运行 2016–2021 整数手数研究期回测",
    )
    period.add_argument(
        "--validation", action="store_true",
        help="运行 2022 整数手数验证期回测",
    )
    period.add_argument(
        "--oos", action="store_true",
        help="冻结并审计后运行 2023–2025 整数手数最终测试",
    )
    period.add_argument(
        "--integer-lot-research", action="store_true",
        help="研究期四板块池整数手数回测：固定250/前20%%/5日，仅用2016–2021",
    )
    period.add_argument(
        "--integer-lot-validation", action="store_true",
        help="2022整数手数验证：固定250/前20%%/5日、1.5倍敞口、20%%回撤熔断",
    )
    period.add_argument(
        "--integer-lot-oos", action="store_true",
        help="冻结并审计后运行2023–2025整数手数OOS",
    )
    args = parser.parse_args(argv)
    if args.integer_lot_research or args.research:
        run_integer_lot_research()
    elif args.integer_lot_validation or args.validation:
        run_integer_lot_validation()
    elif args.integer_lot_oos or args.oos:
        run_integer_lot_oos()


if __name__ == "__main__":
    main()
