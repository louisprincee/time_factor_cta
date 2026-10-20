"""整数手数期货账本、成本和绩效。策略在 strategies/catalog.py。

    python -m backtest.engine --market commodity --research
    python -m backtest.engine --market index --research
    python -m backtest.engine --validation

--market 选择商品流动性池或股指期货，策略名单不变。
四条策略已在 2026-10-19 冻结。--oos 先核对冻结名单，再打开一次 2023–2025。
"""
from __future__ import annotations

import argparse
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
    add_intraday_coords, daily_bars, find_shard, list_shards,
    load_minutes, load_roll_calendar, load_universe, load_validation_shard, oos_universe,
    validation_universe,
)
from tfcta.factors import SignalSet, load, raw_price_path, timestamp_factors


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
    """纯多头整数手账本：入场表为 1 的下一开盘按等名义预算开仓，持有到期平仓，带峰值回撤熔断。"""
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
    if not np.isin(values, [0.0, 1.0]).all():
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
                          symbol, quantity, raw_open, "连续主力开盘价代理")
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
                          f"{old_contract or '未知'} -> {contract}；旧合约平仓价用前一交易日收盘")
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
            event(day, "开仓", symbol, quantity, raw_open,
                f"{contract}；目标名义金额 {budget:.2f}")

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
                "手数": lots[symbol],
                "主力原始开盘价": float(price),
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
            "交易日", "品种", "合约", "手数", "主力原始开盘价",
            "合约乘数", "名义敞口", "保证金_20pct_assumption", "已持有交易日",
            "强平待执行",
        ])
    return daily, positions, pd.DataFrame(events)


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
    roll_calendar: dict | None = None

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
OOS_COLUMNS = ["trading_date", "open", "openw", "close", "closew", "highw", "loww",
               "volume", "total_turnover"]


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


TARGET_CAP = 3.0


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """纯多头，按目标仓位交易整数手，只成交目标与现有手数的差额。

    targets[t, s] 是 t 日开盘的目标权重（0..TARGET_CAP），名义金额 = 权重 × 敞口上限 × 期初权益 / 当年池宽；
    等权满仓时每个品种的权重为 1。权重不变就继续持有，不因价格或权益变化调整手数。
    无开盘价或单价锁定代理当日不成交，次日重试。整本账一个峰值回撤熔断：触发后清仓，停到下一个日历年。
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
    for name, frame in frames.items():
        if not frame.index.equals(index) or not frame.columns.equals(columns):
            raise ValueError(f"{name} 的日期或品种索引与目标仓位表不一致")
    values = targets.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values < 0).any() or (values > TARGET_CAP).any():
        raise ValueError(f"目标权重必须是 0 到 {TARGET_CAP} 之间的有限数")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("初始资金必须是正的有限数")
    if not 0 < margin_rate <= 1 or exposure_limit <= 0 or not 0 < drawdown_limit < 1:
        raise ValueError("保证金率、敞口上限或回撤熔断阈值无效")
    for symbol in columns:
        multiplier = multipliers.get(symbol)
        if multiplier is None or not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f"{symbol} 缺少有效合约乘数")

    high, low = adjusted_high.to_numpy(dtype="float64"), adjusted_low.to_numpy(dtype="float64")
    locked = np.isfinite(high) & np.isfinite(low) & np.isclose(high, low, rtol=0.0, atol=1e-10)
    lots, applied = dict.fromkeys(columns, 0), dict.fromkeys(columns, 0.0)
    contracts, pending_roll = dict.fromkeys(columns), set()
    last_price, last_close = dict.fromkeys(columns, np.nan), dict.fromkeys(columns, np.nan)
    equity = peak_equity = float(initial_capital)
    halt_until_year, previous_drawdown = 0, 0.0
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

    def mark(day, symbol):
        price = open_price.at[day, symbol]
        return float(price) if np.isfinite(price) else last_price[symbol]

    for i, day in enumerate(index):
        day = pd.Timestamp(day)
        year = int(day.year)
        cost = opened = closed = rolled = blocked = 0
        start_equity = equity
        peak_equity = max(peak_equity, equity)
        drawdown_at_open = equity / peak_equity - 1.0
        if previous_drawdown > -drawdown_limit and drawdown_at_open <= -drawdown_limit:
            halt_until_year = year + 1
            event(day, "回撤熔断清仓触发", detail=(
                f"峰值回撤 {drawdown_at_open:.2%}；清仓并暂停至 {halt_until_year} 年"
            ))
        for symbol in columns:
            raw_open = open_price.at[day, symbol]
            if np.isfinite(raw_open) and raw_open > 0:
                last_price[symbol] = float(raw_open)
            elif lots[symbol] and not np.isfinite(last_price[symbol]):
                raise ValueError(f"{day.date()} {symbol} 持仓开盘价缺失")
        margin_call = sum(
            lots[s] * multipliers[s] * mark(day, s) * margin_rate for s in columns if lots[s]
        ) > equity
        if margin_call:
            event(day, "保证金不足", detail="清仓")
        halted = year < halt_until_year or margin_call
        members = universe.get(year, [])

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
            if lots[symbol] and symbol in pending_roll:
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
                event(day, "换月", symbol, quantity, raw_open, f"{contracts[symbol]} -> {contract}")
                contracts[symbol] = contract
                pending_roll.discard(symbol)
                rolled += 1
            if target == applied[symbol] and not (target == 0.0 and lots[symbol]):
                continue
            want = 0
            if target > 0.0:
                budget = target * exposure_limit * max(start_equity, 0.0) / len(members)
                want = int(math.floor(budget / (raw_open * multipliers[symbol])))
            current = lots[symbol]
            change = want - current
            if current and not want:
                cost += charge(close_fee, day, symbol, current, raw_open, "平仓")
                cost += charge(slippage, day, symbol, current, raw_open, "平仓滑点")
                closed += 1
                event(day, "清仓" if halted else "平仓", symbol, current, raw_open)
            elif change > 0:
                contract = contract_codes.at[day, symbol]
                if not isinstance(contract, str) or not contract:
                    raise ValueError(f"{day.date()} {symbol} 开仓缺少主力合约代码")
                cost += charge(open_fee, day, symbol, change, raw_open, "开仓")
                cost += charge(slippage, day, symbol, change, raw_open, "开仓滑点")
                opened += 1
                if not current:
                    contracts[symbol] = contract
                event(day, "加仓" if current else "开仓", symbol, change, raw_open,
                      f"目标权重 {target:.2f}")
            elif change < 0:
                cost += charge(close_fee, day, symbol, -change, raw_open, "减仓")
                cost += charge(slippage, day, symbol, -change, raw_open, "减仓滑点")
                closed += 1
                event(day, "减仓", symbol, -change, raw_open, f"目标权重 {target:.2f}")
            if target > 0.0 and not want:
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
                gross_pnl += lots[symbol] * multipliers[symbol] * float(delta)
        equity = float(equity - cost + gross_pnl)
        writeoff = 0.0
        if equity <= 0:
            event(day, "账户权益耗尽", detail=f"本日损益 {gross_pnl - cost:.2f}")
            writeoff, equity = -equity, 0.0
            halt_until_year = max(halt_until_year, year + 1)
        peak_equity = max(peak_equity, equity)
        previous_drawdown = drawdown_at_open
        notional = sum(lots[s] * multipliers[s] * mark(day, s) for s in columns if lots[s])
        margin = notional * margin_rate
        daily_rows.append({
            "交易日": day, "期初权益": start_equity, "毛盈亏": gross_pnl,
            "交易费用": cost, "净盈亏": float(gross_pnl - cost + writeoff), "权益清零调整": writeoff,
            "权益": equity, "净值": equity / initial_capital,
            "峰值回撤": equity / peak_equity - 1.0 if peak_equity else -1.0,
            "名义敞口": notional, "敞口倍数": notional / equity if equity > 0 else np.nan,
            "保证金需求_20pct_assumption": margin,
            "保证金权益占用": margin / equity if equity > 0 else np.nan,
            "持仓品种数": sum(value != 0 for value in lots.values()),
            "持仓总手数": sum(lots.values()), "开仓笔数": opened,
            "平仓笔数": closed, "换月笔数": rolled, "受阻委托数": blocked,
            "停止新开仓": bool(halted), "强制清仓待执行": bool(halted),
            "保证金追缴": bool(margin_call),
        })
        for symbol in columns:
            if lots[symbol]:
                exposure = lots[symbol] * multipliers[symbol] * mark(day, symbol)
                position_rows.append({
                    "交易日": day, "品种": symbol, "合约": contracts[symbol], "手数": lots[symbol],
                    "主力原始开盘价": mark(day, symbol),
                    "合约乘数": multipliers[symbol], "名义敞口": exposure,
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


def _market_name(market: str) -> str:
    if market not in ("commodity", "index"):
        raise ValueError("市场只能是 commodity 或 index")
    return market


def _oos_ledger_path(market: str):
    if market == "commodity":
        return C.OOS_LEDGER
    return C.INDEX_DATA_ROOT / "oos" / "ledger.jsonl"


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

    return {
        "index": index, "columns": columns, "universe": selected_universe,
        "symbols": selected_symbols, "open_price": open_price, "close_price": close_price,
        "adjusted_open": adjusted_open, "adjusted_open_delta": adjusted_open_delta,
        "adjusted_high": adjusted_high, "adjusted_low": adjusted_low, "rolls": rolls,
        "contract_codes": contract_codes, "open_fee": fee_open, "close_fee": fee_close,
        "roll_close_fee": fee_roll, "slippage": slip, "multipliers": multipliers,
    }


def _run_integer_lot_period(
    period: str,
    data: StudyData,
    micro: pd.DataFrame,
    initial_capital: float = 100_000_000.0,
    market: str = "commodity",
    strategies=None,
) -> None:
    """Run registered strategies on one isolated period."""
    from strategies.book import (
        TREND_WINDOW, adx_above_range, adx_is_range, entry_schedule,
        high_volatility,
    )
    if strategies is None:
        from strategies.catalog import SELECTED
        strategies = SELECTED
    from tfcta.factors import tsmom_sign

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

    market = _market_name(market)
    sector_names = ("股指期货",) if market == "index" else ("全品种",)
    inputs = ledger_inputs(data, period_years, end_date)
    index, columns = inputs["index"], inputs["columns"]
    selected_universe, selected_symbols = inputs["universe"], inputs["symbols"]
    open_price, close_price = inputs["open_price"], inputs["close_price"]
    adjusted_open, adjusted_open_delta = inputs["adjusted_open"], inputs["adjusted_open_delta"]
    adjusted_high, adjusted_low = inputs["adjusted_high"], inputs["adjusted_low"]
    rolls, contract_codes = inputs["rolls"], inputs["contract_codes"]
    fee_open, fee_close = inputs["open_fee"], inputs["close_fee"]
    fee_roll, slip, multipliers = inputs["roll_close_fee"], inputs["slippage"], inputs["multipliers"]

    trend = tsmom_sign(
        data.factors.bars["close"], data.factors.bars["closew"], window=TREND_WINDOW,
    )
    volatile = high_volatility(data.factors.bars)
    ranging = adx_is_range(data.factors.bars)
    above_range = adx_above_range(data.factors.bars)
    for strategy in strategies:
        if getattr(strategy, "EXECUTION", None) == "target":
            strategy.load_config()
            targets = strategy.targets(
                data.factors.signed, data.factors.bars, selected_universe, index, columns,
            )
            result = simulate_target_lots(
                targets, open_price, close_price, adjusted_open, adjusted_open_delta,
                adjusted_high, adjusted_low, rolls, contract_codes, selected_universe,
                multipliers, fee_open, fee_close, fee_roll, slip,
                initial_capital=float(initial_capital), margin_rate=0.20,
                exposure_limit=1.5, drawdown_limit=0.20,
            )
            _write_results(
                period, strategy, *result, selected_symbols, initial_capital, sector_names, market,
            )
            continue
        signal = strategy.signal(trend, micro, volatile, ranging, above_range)
        entries = entry_schedule(
            signal,
            data.returns.reindex(columns=columns),
            selected_universe,
            start_date,
            end_date,
        ).reindex(index=index, columns=columns, fill_value=False).astype("int8")
        _write_integer_lot_strategy(
            period, strategy, entries, index, columns, selected_universe, selected_symbols,
            open_price, close_price, adjusted_open, adjusted_open_delta,
            adjusted_high, adjusted_low, rolls, contract_codes,
            fee_open, fee_close, fee_roll, slip, multipliers,
            initial_capital, sector_names, market,
        )


def _write_integer_lot_strategy(
    period, strategy, entries, index, columns, selected_universe, selected_symbols,
    open_price, close_price, adjusted_open, adjusted_open_delta,
    adjusted_high, adjusted_low, rolls, contract_codes,
    fee_open, fee_close, fee_roll, slip, multipliers,
    initial_capital, sector_names, market="commodity",
) -> None:
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
    _write_results(
        period, strategy, daily, positions, events, selected_symbols, initial_capital,
        sector_names, market,
    )


def _write_results(
    period, strategy, daily, positions, events, selected_symbols, initial_capital,
    sector_names, market="commodity",
) -> None:
    """写一条策略在一个区间的汇总、分年、逐日、持仓和事件。period 为整数时是冻结后的单年测试。"""
    annual_rows = []
    output_dir = C.RUNS_DIR / "strategies" / strategy.ID / market
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(period, int):
        if period == C.VALIDATION_YEAR:
            C.assert_validation_2022_dates(daily.index, f"{period} 单年测试净值")
        else:
            C.assert_strict_oos_dates(daily.index, f"{period} 单年测试净值")
        if set(daily.index.year) != {period}:
            raise C.HoldoutViolation(f"单年测试净值必须只含 {period} 年")
        period_years = (period,)
    elif period == "research":
        C.assert_no_holdout_dates(daily.index, "整数手数研究净值")
        period_years = tuple(RESEARCH_YEARS)
    elif period == "validation":
        C.assert_validation_2022_dates(daily.index, "整数手数验证净值")
        period_years = (2022,)
    elif period == "oos":
        C.assert_strict_oos_dates(daily.index, "整数手数样本外净值")
        if daily.index.max() > pd.Timestamp(C.DEFAULT_OOS_END):
            raise C.HoldoutViolation("整数手数样本外净值越过 2025")
        period_years = tuple(range(C.STRICT_OOS_START.year, C.DEFAULT_OOS_END.year + 1))
    else:
        raise ValueError("整数手数只跑研究期、2022 验证期或 2023–2025 样本外")
    label = strategy.NAME
    daily["策略"] = label
    positions.insert(0, "策略", label)
    if not events.empty:
        events.insert(0, "策略", label)
    returns = daily["净收益"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    metrics = performance(returns)
    summary_rows = [{
        "策略": label,
        "市场": "股指" if market == "index" else "商品",
        "板块池": "+".join(sector_names),
        "参数": strategy.DETAIL,
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
    }]
    for year in period_years:
        annual = daily.loc[daily.index.year == year, "净收益"].fillna(0.0)
        annual_nav = (1.0 + annual).cumprod()
        annual_peak = np.maximum.accumulate(np.r_[1.0, annual_nav.to_numpy()])[1:]
        annual_drawdown = annual_nav.to_numpy() / annual_peak - 1.0
        annual_rows.append({
            "策略": label,
            "年份": year,
            "净收益": float((1.0 + annual).prod() - 1.0),
            "Sharpe": sharpe_ratio(annual),
            "期末权益": float(daily.loc[daily.index.year == year, "权益"].iloc[-1]),
            "年度内最大回撤": float(annual_drawdown.min()) if len(annual_drawdown) else np.nan,
            "开仓笔数": int(daily.loc[daily.index.year == year, "开仓笔数"].sum()),
            "平仓笔数": int(daily.loc[daily.index.year == year, "平仓笔数"].sum()),
        })

    result_dir = output_dir / str(period)
    result_dir.mkdir(parents=True, exist_ok=True)
    period_label = f"{period}年单年测试" if isinstance(period, int) else {
        "research": "2016–2021研究期",
        "validation": "2022验证期",
        "oos": "2023–2025 OOS",
    }[period]
    book = "股指期货" if market == "index" else "商品"
    plot_performance(
        daily["净收益"].rename(label),
        result_dir / "plots" / label,
        title=f"{period_label}整数手数：{label}，{book}，1亿元、1.5倍敞口、20%保证金假设",
    )

    summary_path = result_dir / "summary.csv"
    annual_path = result_dir / "annual.csv"
    daily_path = result_dir / "daily.csv"
    positions_path = result_dir / "positions.csv"
    events_path = result_dir / "events.csv"
    pd.DataFrame(summary_rows).to_csv(
        summary_path, index=False, encoding="utf-8-sig", float_format="%.10f",
    )
    pd.DataFrame(annual_rows).to_csv(
        annual_path, index=False, encoding="utf-8-sig", float_format="%.10f",
    )
    daily.reset_index().to_csv(
        daily_path, index=False, encoding="utf-8-sig", float_format="%.10f",
    )
    positions.to_csv(
        positions_path, index=False, encoding="utf-8-sig", float_format="%.10f",
    )
    events.to_csv(
        events_path, index=False, encoding="utf-8-sig", float_format="%.10f",
    )
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(f"年度结果：{annual_path}", flush=True)
    print(f"逐日权益：{daily_path}", flush=True)
    print(f"整数手持仓：{positions_path}", flush=True)
    print(f"订单/风控事件：{events_path}", flush=True)
    return summary_rows[0], annual_rows


def run_integer_lot_research(
    initial_capital=100_000_000.0,
    market="commodity",
    strategy_set="frozen",
):
    market = _market_name(market)
    if strategy_set not in ("frozen", "experiments"):
        raise ValueError("strategy_set 必须是 frozen 或 experiments")
    if strategy_set == "experiments" and market != "commodity":
        raise ValueError("当前探索策略仅注册在商品市场")
    if market == "index":
        from backtest.index_market import load_research as load_index_research
        data = load_index_research(1.0)
    else:
        data = load_research(1.0)
    micro = data.factors.raw("ts_low") - data.factors.raw("ts_high")
    if strategy_set == "frozen":
        _run_integer_lot_period("research", data, micro, initial_capital, market=market)
        return

    from strategies.research_catalog import RESEARCH_EXPERIMENTS

    _run_integer_lot_period(
        "research",
        data,
        micro,
        initial_capital,
        market=market,
        strategies=RESEARCH_EXPERIMENTS,
    )


def run_integer_lot_validation(initial_capital=100_000_000.0, market="commodity"):
    market = _market_name(market)
    if market == "index":
        from backtest.index_market import load_validation as load_index_validation
        data, micro = load_index_validation(1.0)
    else:
        data, micro = load_validation(1.0)
    _run_integer_lot_period("validation", data, micro, initial_capital, market=market)


def _frozen_strategy_specs() -> list[dict]:
    """冻结名单核对发生在任何样本外读取之前。"""
    from strategies.catalog import SELECTED

    expected = ("trend_ema3", "trend_high_atr", "trend_adx_range", "regime_two_leg")
    if tuple(strategy.ID for strategy in SELECTED) != expected:
        raise ValueError("样本外名单与 2026-10-19 冻结的四条策略不一致")
    specs = []
    frozen_at = None
    for strategy in SELECTED:
        config = strategy.load_config()
        if config.get("status") != "frozen" or not config.get("frozen_at") or not isinstance(
            config.get("chosen"), dict,
        ):
            raise ValueError("当前保留的策略没有冻结，不能读取 2023–2025")
        if frozen_at is None:
            frozen_at = config["frozen_at"]
        elif config["frozen_at"] != frozen_at:
            raise ValueError("四条策略必须在同一天冻结")
        specs.append(config)
    return specs


def _refuse_repeat_oos(strategy: str, ledger=None) -> None:
    for entry in C.read_ledger(ledger):
        if (
            entry.get("kind") == "final_evaluation"
            and entry.get("strategy") == strategy
            and entry.get("status") == "finished"
        ):
            raise ValueError("这四条策略的样本外已经跑完，不能再读 2023–2025")


def run_integer_lot_oos(initial_capital=100_000_000.0, market="commodity"):
    """四条冻结策略的样本外，从 2023 起各用 1 亿元新账。商品和股指分开记账。"""
    market = _market_name(market)
    specs = _frozen_strategy_specs()
    window = specs[0]["chosen"]["trend_window"]
    ledger = _oos_ledger_path(market)
    if market == "index":
        from backtest.index_market import require_oos_inputs
        require_oos_inputs()
        ledger_name = f"integer_lot_index_w{window}"
        note = f"股指期货，趋势窗口 {window} 日。日盘时钟。"
        universe = "index_futures"
    else:
        ledger_name = f"integer_lot_all_symbols_w{window}"
        note = f"全品种流动性池，趋势窗口 {window} 日。"
        universe = "all_liquidity"
    _refuse_repeat_oos(ledger_name, ledger)
    files = _oos_audit_files(market)
    spec = {
        "id": ledger_name,
        "frozen_at": specs[0]["frozen_at"],
        "chosen": {item["id"]: item["chosen"] for item in specs},
        "universe": universe,
    }
    with C.final_evaluation(
        spec, files, C.DEFAULT_OOS_END, strategy=ledger_name, note=note, ledger=ledger,
    ):
        if market == "index":
            from backtest.index_market import load_oos as load_index_oos
            data, micro = load_index_oos()
        else:
            data, micro = load_oos()
        _run_integer_lot_period("oos", data, micro, initial_capital, market=market)
    C.append_ledger({
        "kind": "final_evaluation",
        "status": "finished",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": ledger_name,
        "frozen_at": spec["frozen_at"],
        "chosen": spec["chosen"],
        "universe": spec["universe"],
        "oos_start": str(C.STRICT_OOS_START),
        "oos_end": str(C.DEFAULT_OOS_END),
        "note": note,
    }, ledger)


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
    opens, adjusted_opens, closes, adjusted_closes, highs, lows = {}, {}, {}, {}, {}, {}
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
        adjusted_closes[symbol] = bars["closew"]
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
    closew = pd.DataFrame(adjusted_closes).sort_index()
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
                "open": open_px, "openw": openw, "close": close_px, "closew": closew,
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


# --------------------------------------------------------------------------
# 冻结的全市场时钟：按年依次打开 2022、2023、2024、2025
# --------------------------------------------------------------------------
YEAR_TEST_YEARS = (2022, 2023, 2024, 2025)
YEAR_TEST_WARMUP_DAYS = 320          # 持续期阈值要 250 日分钟历史，留足余量
YEAR_TEST_FACTORS = ("ts_low", "ts_high", "dfp_top3", "ts_low_pm", "ts_high_pm")


def year_test_name(year: int) -> str:
    return f"market_clock_long_{int(year)}"


def check_year_order(year: int, ledger=None) -> None:
    """只能按年顺序打开；前面每一年都已跑完，本年还没打开过。"""
    year = int(year)
    if year not in YEAR_TEST_YEARS:
        raise C.HoldoutViolation(f"单年测试只允许 {YEAR_TEST_YEARS}，收到 {year}")
    entries = C.read_ledger(ledger)
    finished = {
        entry.get("strategy") for entry in entries
        if entry.get("status") == "finished" and str(entry.get("strategy", "")).startswith("market_clock_long_")
    }
    missing = [y for y in YEAR_TEST_YEARS if y < year and year_test_name(y) not in finished]
    if missing:
        raise C.HoldoutViolation(f"必须先跑完 {missing} 才能打开 {year}")
    if any(entry.get("strategy") == year_test_name(year) for entry in entries):
        raise C.HoldoutViolation(f"{year} 已经在台账里打开过，不能再读")


def _fresh_time_factors(minute: pd.DataFrame) -> pd.DataFrame:
    """在一段分钟数据上现算五个因子，口径与研究期缓存相同（N250/M55 持续期阈值）。"""
    from tfcta.factors import duration_factors, report_factors

    minute = add_intraday_coords(minute)
    clock = timestamp_factors(minute)
    duration = duration_factors(minute, C.IC_REFERENCE_LOOKBACK, C.IC_REFERENCE_PCT)
    report = report_factors(minute, C.IC_REFERENCE_LOOKBACK, C.IC_REFERENCE_PCT)
    return clock.join(duration, how="outer").join(report, how="outer")[list(YEAR_TEST_FACTORS)]


def load_market_clock_year(year: int, n_ticks: float = 1.0) -> StudyData:
    """研究期缓存接上 2022 至 year 的现算因子和日线。year ≥ 2023 只能在最终评估块内调用。"""
    from tfcta.costs import estimate_ticks

    year = int(year)
    end = pd.Timestamp(f"{year}-12-31")
    if year >= C.STRICT_OOS_START.year:
        C.assert_oos_research_locked(end.date())
    research = load_research(n_ticks)
    pools = {C.VALIDATION_YEAR: validation_universe()[0]}
    if year >= C.STRICT_OOS_START.year:
        later, _ = oos_universe(C.COMMODITY_SYMBOLS, end.date())
        pools.update({y: names for y, names in later.items() if y <= year})
    names = sorted({s for members in pools.values() for s in members})
    fees = [load_fees("research"), load_fees("validation_2022")]
    if year >= C.STRICT_OOS_START.year:
        fees.append(load_fees("oos"))
    fees = pd.concat(fees, ignore_index=True)
    ticks = load_ticks("research" if year == C.VALIDATION_YEAR else "validation_2022")
    _require_cost_coverage(pools[C.VALIDATION_YEAR], ticks, fees, f"{year} 单年测试")

    cutoff = research.returns.index[-YEAR_TEST_WARMUP_DAYS]
    factors = {name: {} for name in YEAR_TEST_FACTORS}
    bars = {field: {} for field in ("open", "openw", "close", "closew", "highw", "loww")}
    tick_parts, available = [], {}
    for symbol in names:
        if find_shard(C.VALIDATION_DIR, symbol) is None and year == C.VALIDATION_YEAR:
            raise FileNotFoundError(f"{symbol} 缺 2022 验证分片")
        minute = load_minutes(
            symbol, include_validation=True,
            oos_end=end.date() if year >= C.STRICT_OOS_START.year else None, columns=OOS_COLUMNS,
        )
        minute = minute[pd.to_datetime(minute["trading_date"]) >= cutoff]
        fresh = _fresh_time_factors(minute)
        if symbol in research.factors.signed["ts_low"].columns:
            # 研究期最后几十天两边都有完整的阈值历史，必须一致，否则现算口径和缓存不一样
            overlap = fresh.index[(fresh.index > research.returns.index[-60]) & (fresh.index <= research.returns.index[-1])]
            for name in YEAR_TEST_FACTORS:
                cached = research.factors.raw(name)[symbol].reindex(overlap)
                if not np.allclose(cached, fresh.loc[overlap, name], equal_nan=True, atol=1e-9):
                    raise ValueError(f"{symbol} {name} 现算值与研究期缓存不一致")
        later_days = fresh.index[fresh.index >= pd.Timestamp(C.HOLDOUT_START)]
        for name in YEAR_TEST_FACTORS:
            factors[name][symbol] = fresh.loc[later_days, name]
        daily = daily_bars(minute)
        daily = daily.loc[(daily.index >= pd.Timestamp(C.HOLDOUT_START)) & (daily.index <= end)]
        for field in bars:
            bars[field][symbol] = daily[field]
        available[symbol] = set(daily.index.year)
        if year > C.STRICT_OOS_START.year:
            estimated = estimate_ticks(raw_price_path(minute), minute["trading_date"])
            estimated = estimated[(estimated["year"] >= C.STRICT_OOS_START.year) & (estimated["year"] < year)].copy()
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


def _year_audit_files(year: int, symbols) -> list[Path]:
    from strategies import market_clock_long

    files = [
        Path(__file__), market_clock_long.CONFIG_PATH, Path(market_clock_long.__file__),
        C.PROJECT_ROOT / "src" / "tfcta" / "config.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "data.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "costs.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "factors.py",
        C.UNIVERSE_DIR / "yearly_stats.csv", C.UNIVERSE_DIR / "screen_params.json",
        C.RESEARCH_OUT_DIR / "fee_history.csv", C.RESEARCH_OUT_DIR / "tick_size.csv",
        C.COMMODITY_DATA / "validation_2022" / "fee_history.csv",
        C.COMMODITY_DATA / "validation_2022" / "tick_size.csv",
    ]
    directories = [C.RESEARCH_DIR, C.VALIDATION_DIR]
    if year >= C.STRICT_OOS_START.year:
        files.append(C.COMMODITY_DATA / "oos" / "fee_history.csv")
        directories.append(C.HOLDOUT_DIR)
    for symbol in symbols:
        for directory in directories:
            shard = find_shard(directory, symbol)
            if shard is not None:
                files.append(shard)
        if (C.ROLL_DIR / f"{symbol}.csv").is_file():
            files.append(C.ROLL_DIR / f"{symbol}.csv")
    return [path for path in dict.fromkeys(files) if path.is_file()]


def _run_year_book(year: int, data: StudyData, initial_capital: float):
    from strategies import market_clock_long

    inputs = ledger_inputs(data, (year,), pd.Timestamp(f"{year}-12-31"))
    full_index = data.returns.index
    every = sorted({s for members in data.universe.values() for s in members})
    targets = market_clock_long.targets(
        data.factors.signed, data.factors.bars, data.universe, full_index, every,
    ).reindex(index=inputs["index"], columns=inputs["columns"]).fillna(0.0)
    keys = ("open_price", "close_price", "adjusted_open", "adjusted_open_delta", "adjusted_high",
            "adjusted_low", "rolls", "contract_codes")
    result = simulate_target_lots(
        targets, *(inputs[k] for k in keys), inputs["universe"], inputs["multipliers"],
        inputs["open_fee"], inputs["close_fee"], inputs["roll_close_fee"], inputs["slippage"],
        initial_capital=float(initial_capital), margin_rate=0.20, exposure_limit=1.5, drawdown_limit=0.20,
    )
    return _write_results(
        year, market_clock_long, *result, inputs["symbols"], initial_capital, ("全品种",), "commodity",
    )


def run_market_clock_year(year: int, initial_capital: float = 100_000_000.0, ledger=None) -> dict:
    """冻结的全市场时钟打开一年：先核对冻结和顺序，再记台账，最后读数据、计分、写结果。"""
    from strategies import market_clock_long

    year = int(year)
    config = market_clock_long.load_config()
    check_year_order(year, ledger)
    name = year_test_name(year)
    pools = set(validation_universe()[0])
    note = (f"冻结的只做多全市场时钟，{year} 年单独一本 1 亿元新账；之前年份只做因子和仓位预热。"
            "这些年份的行情和其他策略结果以前看过，规则没有在其上拟合，不是干净样本外。")
    spec = {"id": market_clock_long.ID, "frozen_at": config["frozen_at"], "chosen": config["chosen"]}
    if year == C.VALIDATION_YEAR:
        files = _year_audit_files(year, sorted(pools))
        C.append_ledger({
            "kind": "year_test", "status": "opened",
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "strategy": name, "chosen": config["chosen"], "frozen_at": config["frozen_at"],
            "test_start": f"{year}-01-01", "test_end": f"{year}-12-31",
            "sha256": {C._ledger_key(path): C._sha256(path) for path in files}, "note": note,
        }, ledger)
        summary, annual = _run_year_book(year, load_market_clock_year(year), initial_capital)
    else:
        files = _year_audit_files(year, C.COMMODITY_SYMBOLS)
        with C.final_evaluation(spec, files, f"{year}-12-31", strategy=name, note=note, ledger=ledger):
            summary, annual = _run_year_book(year, load_market_clock_year(year), initial_capital)
    C.append_ledger({
        "kind": "year_test" if year == C.VALIDATION_YEAR else "final_evaluation",
        "status": "finished", "run_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": name, "frozen_at": config["frozen_at"], "scored_year": year,
        "net_return": summary["累计净收益"], "sharpe": summary["Sharpe"],
        "max_drawdown": summary["最大回撤"], "note": note,
    }, ledger)
    return summary



def _oos_audit_files(market: str = "commodity") -> list[Path]:
    if market == "index":
        return _index_oos_audit_files()
    return _commodity_oos_audit_files()


def _index_oos_audit_files() -> list[Path]:
    from strategies import book, regime_two_leg, trend_adx_range, trend_ema3, trend_high_atr
    from tfcta.costs import index_fee_path

    files = [
        Path(__file__),
        Path(__file__).with_name("index_market.py"),
        trend_ema3.CONFIG_PATH,
        Path(trend_ema3.__file__),
        trend_high_atr.CONFIG_PATH,
        Path(trend_high_atr.__file__),
        trend_adx_range.CONFIG_PATH,
        Path(trend_adx_range.__file__),
        regime_two_leg.CONFIG_PATH,
        Path(regime_two_leg.__file__),
        Path(book.__file__),
        C.PROJECT_ROOT / "src" / "tfcta" / "config.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "data.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "costs.py",
        C.PROJECT_ROOT / "src" / "tfcta" / "factors.py",
        index_fee_path("research"),
        index_fee_path("validation_2022"),
        index_fee_path("oos"),
    ]
    for symbol in ("IF", "IH", "IC", "IM"):
        for directory in (C.INDEX_RESEARCH_DIR, C.INDEX_HOLDOUT_DIR):
            shard = find_shard(directory, symbol)
            if shard is not None:
                files.append(shard)
            oi = find_shard(directory / "open_interest", symbol)
            if oi is not None:
                files.append(oi)
    return [path for path in dict.fromkeys(files) if path.is_file()]


def _commodity_oos_audit_files() -> list[Path]:
    """Files whose contents determine the frozen strategy's OOS result."""
    from strategies import book, regime_two_leg, trend_adx_range, trend_ema3, trend_high_atr

    symbols = C.COMMODITY_SYMBOLS
    files = [
        Path(__file__),
        trend_ema3.CONFIG_PATH,
        Path(trend_ema3.__file__),
        trend_high_atr.CONFIG_PATH,
        Path(trend_high_atr.__file__),
        trend_adx_range.CONFIG_PATH,
        Path(trend_adx_range.__file__),
        regime_two_leg.CONFIG_PATH,
        Path(regime_two_leg.__file__),
        Path(book.__file__),
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "运行已冻结的四条整数手数策略。--market 选择商品或股指。"
            "--oos 只允许打开一次 2023–2025。"
        )
    )
    parser.add_argument(
        "--market",
        choices=("commodity", "index"),
        default="commodity",
        help="commodity 为商品流动性池，index 为股指期货。默认商品。",
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
        "--experimental-research", action="store_true",
        help="跑 research_catalog 里的探索策略，限定商品 2016–2021",
    )
    period.add_argument(
        "--year", type=int, choices=YEAR_TEST_YEARS,
        help="冻结的全市场时钟按年打开一次（2022、2023、2024、2025 依次），先记台账",
    )
    args = parser.parse_args(argv)
    if args.year is not None:
        if args.market != "commodity":
            parser.error("--year 只用于商品")
        run_market_clock_year(args.year)
    elif args.experimental_research:
        run_integer_lot_research(market=args.market, strategy_set="experiments")
    elif args.research:
        run_integer_lot_research(market=args.market)
    elif args.validation:
        run_integer_lot_validation(market=args.market)
    elif args.oos:
        run_integer_lot_oos(market=args.market)


if __name__ == "__main__":
    main()
