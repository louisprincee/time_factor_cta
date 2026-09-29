"""1 分钟日内策略撮合与绩效计算。

成交规则固定为保守口径（撮合在 ``simulate_path``）：
- 突破在当前 bar 内触发时按突破价成交，开盘已越过突破价则按开盘价成交；
- 同一根 bar 同时触及上下轨无法判断先后，当日不交易；成交量为 0 的 bar 不能入场或反手；
- 入场 bar 内只检查止损；之后止盈和止损同一根 bar 同时触发时按止损；
- 反手（Dual Thrust ``reverse=True``、R-Breaker 反转）在入场 bar 之后按反手价平仓并同价反向开仓；
- 每个交易日最后一根 bar 按收盘价强制平仓，不持仓到下一个交易日。

``trading_date`` 以夜盘开盘为一日之始，所以有夜盘品种的"日内"持仓会跨过夜盘收盘到
次日 9:00 的休市。价格用加法复权（``*w``）列，收益分母用入场 bar 的原始开盘价。
手续费：``fee_schedule="flat"`` 为每边 ``fee_rate``；``"hist"`` 为当日主力合约的历史交易所标准
（``costs.load_fee_history``，研究期主口径）；``"table"`` 为 ``costs.FEES_2026``。后两者都按
开仓 + 平今（比例 + 元/手 ÷ 原始价格 × 乘数）。
该模块不读取或修改现有日频研究产物。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest import costs
from .strategies import StrategyConfig, prepared_daily_levels

TRADE_COLUMNS = ["symbol", "direction", "entry_time", "exit_time", "entry_price",
                 "exit_price", "exit_reason", "gross_return", "cost", "net_return"]


@dataclass(frozen=True)
class BacktestConfig:
    fee_rate: float = 0.00025
    slippage_points: float = 0.0
    margin_rate: float = 0.30
    initial_capital: float = 1.0
    # 每边滑点的 tick 数，tick 大小由 run_intraday(tick=...) 按年份给出
    slippage_ticks: float = 0.0
    # "flat"：每边 fee_rate；"hist"：历史交易所费率；"table"：2026 费率表（见 backtest/costs.py）
    fee_schedule: str = "flat"
    fee_scale: float = 1.0
    close_today_as_open: bool = False

    def fee_model(self) -> costs.FeeModel:
        schedule = {"table": "2026", "hist": "hist"}[self.fee_schedule]
        return costs.FeeModel(schedule, self.fee_scale, self.close_today_as_open)


def trade_cost(config: BacktestConfig, symbol: str, raw_price, dates=None) -> np.ndarray:
    """日内一次开平（平今）的手续费占名义金额的比例。"hist" 口径需要逐笔交易日 ``dates``。"""
    price = np.asarray(raw_price, dtype="float64")
    if config.fee_schedule == "flat":
        return np.full(price.shape, 2.0 * config.fee_rate)
    if config.fee_schedule in ("table", "hist"):
        return costs.round_trip(symbol, price, config.fee_model(), close_today=True, dates=dates)
    raise ValueError(f"未知手续费口径: {config.fee_schedule}")


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily: pd.DataFrame
    metrics: dict[str, float]


def performance(net_return: pd.Series) -> dict[str, float]:
    ret = pd.Series(net_return, dtype="float64").dropna()
    std = ret.std(ddof=1) if len(ret) > 1 else np.nan
    equity = (1.0 + ret).cumprod()
    growth = float(equity.iloc[-1]) if len(equity) else np.nan
    return {
        "ann_return": float(growth ** (252.0 / len(ret)) - 1.0)
        if len(ret) and growth > 0 else np.nan,
        "ann_vol": float(std * np.sqrt(252.0)) if np.isfinite(std) else np.nan,
        "sharpe": float(ret.mean() / std * np.sqrt(252.0))
        if np.isfinite(std) and std > 0 else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min())
        if len(equity) else np.nan,
        "n_days": float(len(ret)),
    }


@dataclass
class Prepared:
    """一次性整理好的分钟数据，同一品种跑多组参数时复用。"""
    df: pd.DataFrame
    raw_open: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    codes: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    dates: pd.DatetimeIndex
    tradable: np.ndarray
    prior_high: np.ndarray   # 当日此前（不含当前 bar）的最高价，首根 bar 为 -inf
    prior_low: np.ndarray


def prepare(minute: pd.DataFrame) -> Prepared:
    df, raw_open = _prepare(minute)
    o, h, l, c = (df[k].to_numpy("float64") for k in ("open", "high", "low", "close"))
    codes, starts, ends = day_layout(df["trading_date"].to_numpy())
    dates = pd.DatetimeIndex(df["trading_date"].to_numpy()[starts])
    tradable = (pd.to_numeric(df["volume"], errors="coerce").to_numpy("float64") > 0
                if "volume" in df else np.ones(len(df), bool))
    first = np.zeros(len(df), bool)
    first[starts] = True
    ph = pd.Series(h).groupby(codes).cummax().shift(1).to_numpy()
    pl = pd.Series(l).groupby(codes).cummin().shift(1).to_numpy()
    return Prepared(df, raw_open, o, h, l, c, codes, starts, ends, dates, tradable,
                    np.where(first, -np.inf, ph), np.where(first, np.inf, pl))


def _prepare(minute: pd.DataFrame):
    df = minute.copy()
    raw_open = pd.to_numeric(df["open"], errors="coerce") if "open" in df else None
    for raw, adjusted in {"open": "openw", "high": "highw", "low": "loww", "close": "closew"}.items():
        if adjusted in df.columns:
            df[raw] = df[adjusted]
    missing = sorted({"open", "high", "low", "close", "trading_date"} - set(df.columns))
    if missing:
        raise KeyError(f"分钟数据缺少字段: {missing}")
    df.index = pd.to_datetime(df.index)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.normalize()
    df = df.sort_index(kind="mergesort")
    if df.index.has_duplicates:
        raise ValueError("分钟数据存在重复时间戳")
    raw_open = df["open"] if raw_open is None else raw_open.reindex(df.index)
    return df, raw_open.to_numpy("float64")


def tick_per_day(tick, dates: pd.DatetimeIndex) -> np.ndarray:
    """``tick`` 为标量或 {年份: tick}；返回逐交易日的 tick。

    表里没有的年份用此前最近一年的 tick（第一条记录之前用最早一年）。不能记 0：
    tick 表只覆盖研究期，记 0 会让之后年份的滑点静默消失。
    """
    if tick is None:
        return np.zeros(len(dates))
    if np.isscalar(tick):
        return np.full(len(dates), float(tick))
    known = sorted((int(k), float(v)) for k, v in dict(tick).items() if np.isfinite(v))
    if not known:
        raise ValueError("tick 表没有有效值，滑点无法定价")
    years = np.array([k for k, _ in known])
    values = np.array([v for _, v in known])
    i = np.clip(np.searchsorted(years, pd.DatetimeIndex(dates).year.to_numpy(), side="right") - 1,
                0, len(years) - 1)
    return values[i]


def bar_levels(p: Prepared, strategy: StrategyConfig) -> dict[str, np.ndarray]:
    """逐 bar 的开仓/反手价，及逐日的 ATR、止损止盈距离、最早入场 bar、是否可交易。"""
    level = prepared_daily_levels(p.df, strategy).reindex(p.dates)
    atr = level["atr"].to_numpy("float64")
    n_days, codes = len(p.starts), p.codes
    day_ok = np.ones(n_days, bool)
    start_bar = np.zeros(n_days, dtype=np.int64)
    if strategy.name in ("opening_range_assumption", "sky_garden"):
        minutes = int(strategy.sky_bars if strategy.name == "sky_garden"
                      else strategy.opening_range_minutes)
        if strategy.name == "opening_range_assumption" and strategy.open_anchor == "day":
            origin = day_open_offset(p.df.index, codes, p.starts)
        else:
            origin = np.zeros(n_days, np.int64)
        long_d, short_d = opening_range(p.h, p.l, codes, p.starts, p.ends, minutes, origin)
        start_bar[:] = np.where(origin >= 0, origin + minutes, np.iinfo(np.int64).max // 4)
        if strategy.open_anchor == "day":
            day_ok = origin >= 0
        if strategy.name == "sky_garden":
            # 跳空幅度 = 复权价差 / 原始昨收（= 当日原始开盘 − 复权价差）
            gap_pts = level["gap_points"].to_numpy("float64")
            raw_prev = p.raw_open[p.starts] - gap_pts
            with np.errstate(invalid="ignore", divide="ignore"):
                gap = gap_pts / np.where(raw_prev > 0, raw_prev, np.nan)
                day_ok = (gap >= strategy.gap_pct) | (gap <= -strategy.gap_pct)
    else:
        long_d = level["long_level"].to_numpy("float64")
        short_d = level["short_level"].to_numpy("float64")
    enter_long, enter_short = long_d[codes], short_d[codes]
    rev_to_short = rev_to_long = np.full(len(p.o), np.nan)
    if strategy.name == "rbreaker":
        # 当日此前最高价超过观察卖出价后，跌破反转卖出价即做空（持多则反手）；做多对称
        with np.errstate(invalid="ignore"):
            gate_s = p.prior_high > level["sell_setup"].to_numpy("float64")[codes]
            gate_l = p.prior_low < level["buy_setup"].to_numpy("float64")[codes]
        rev_to_short = np.where(gate_s, level["sell_enter"].to_numpy("float64")[codes], np.nan)
        rev_to_long = np.where(gate_l, level["buy_enter"].to_numpy("float64")[codes], np.nan)
        enter_long = np.where(gate_l, rev_to_long, enter_long)
        enter_short = np.where(gate_s, rev_to_short, enter_short)
    elif strategy.reverse:
        rev_to_short, rev_to_long = enter_short, enter_long
    day_open = p.raw_open[p.starts]
    nan = np.full(n_days, np.nan)
    if strategy.stop_pct is not None:
        stop_dist = strategy.stop_pct * day_open
    elif strategy.stop_atr_multiple is not None:
        stop_dist = atr * strategy.stop_atr_multiple
    else:
        stop_dist = nan
    if strategy.target_pct is not None:
        target_dist = strategy.target_pct * day_open
    elif strategy.target_atr_multiple is not None:
        target_dist = atr * strategy.target_atr_multiple
    else:
        target_dist = nan
    return {"enter_long": enter_long, "enter_short": enter_short,
            "rev_to_short": rev_to_short, "rev_to_long": rev_to_long, "atr": atr,
            "stop_dist": stop_dist, "target_dist": target_dist,
            "start_bar": start_bar, "day_ok": day_ok}


def run_intraday(minute: pd.DataFrame | Prepared, strategy: StrategyConfig,
                 config: BacktestConfig = BacktestConfig(),
                 symbol: str = "", tick=None,
                 allow_long: np.ndarray | None = None,
                 allow_short: np.ndarray | None = None) -> BacktestResult:
    """单品种日内回测。``allow_long``/``allow_short`` 是逐交易日的方向过滤（按交易日顺序）。

    ``minute`` 可以直接传 ``prepare()`` 的结果，同一品种跑多组参数时不必重复整理数据。
    """
    p = minute if isinstance(minute, Prepared) else prepare(minute)
    lv = bar_levels(p, strategy)
    n_days = len(p.starts)
    base_long = np.ones(n_days, bool) if allow_long is None else np.asarray(allow_long, bool)
    base_short = np.ones(n_days, bool) if allow_short is None else np.asarray(allow_short, bool)
    sim = simulate_path(p.o, p.h, p.l, p.c, p.codes, p.starts, p.ends,
                             lv["enter_long"], lv["enter_short"], lv["rev_to_short"],
                             lv["rev_to_long"], lv["stop_dist"], lv["target_dist"],
                             lv["start_bar"], base_long & lv["day_ok"],
                             base_short & lv["day_ok"], p.tradable,
                             max_legs=int(strategy.max_entries_per_day))
    pieces = [sim] if len(sim["day"]) else []
    trades = _trades_frame(pieces, p.df.index, p.raw_open, config,
                           tick_per_day(tick, p.dates), symbol, p.dates)

    # atr_pct 是开盘前已知的 ATR 占当日首根 bar 原始开盘价的比例，组合层按它做风险缩放
    first_open = p.raw_open[p.starts]
    daily = pd.DataFrame({"net_return": 0.0, "gross_return": 0.0, "n_trades": 0,
                          "atr_pct": lv["atr"] / np.where(first_open > 0, first_open, np.nan)},
                         index=pd.Index(p.dates, name="trading_date"))
    if len(trades):
        agg = trades.groupby("day").agg(net=("net_return", "sum"),
                                        gross=("gross_return", "sum"),
                                        size=("net_return", "size"))
        pos = agg.index.to_numpy()
        daily.iloc[pos, 0] = agg["net"].to_numpy() / config.margin_rate
        daily.iloc[pos, 1] = agg["gross"].to_numpy() / config.margin_rate
        daily.iloc[pos, 2] = agg["size"].to_numpy()
    daily["equity"] = config.initial_capital * (1.0 + daily["net_return"]).cumprod()
    metrics = performance(daily["net_return"])
    metrics["n_trades"] = float(len(trades))
    return BacktestResult(trades.drop(columns="day"), daily, metrics)


def _trades_frame(pieces, index, raw_open, config, tick, symbol, dates) -> pd.DataFrame:
    if not pieces:
        return pd.DataFrame(columns=[*TRADE_COLUMNS, "day"])
    sim = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    day, direction = sim["day"], sim["direction"].astype(np.int64)
    slip = config.slippage_points + config.slippage_ticks * tick[day]
    entry = sim["entry"] + direction * slip
    exit_ = sim["exit"] - direction * slip
    ref = raw_open[sim["entry_bar"]]
    ref = np.where(np.isfinite(ref) & (ref > 0), ref, entry)
    gross = direction * (exit_ - entry) / ref
    cost = trade_cost(config, symbol, ref, dates[day])
    out = pd.DataFrame({
        "symbol": symbol, "direction": direction,
        "entry_time": index[sim["entry_bar"]], "exit_time": index[sim["exit_bar"]],
        "entry_price": entry, "exit_price": exit_, "exit_reason": sim["reason"],
        "gross_return": gross, "cost": cost, "net_return": gross - cost, "day": day,
    })
    return out.sort_values("entry_time", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# 向量化撮合
# --------------------------------------------------------------------------
# 单次入场日内突破的向量化撮合。
#
# 把"当日第一根触发的 bar"换成按交易日分段的 ``minimum.reduceat``，整段分钟数据只扫常数遍。
# 测试里与逐 bar 参考循环互相对照。
#
# 成交口径：
# - 入场：多头 bar 最高价严格高于上轨（价格穿过而非只碰到），开盘已越过上轨则按开盘价，
#   否则按上轨；空头对称。同一根 bar 上下轨都被穿过时，开盘已在某一侧之外就按该侧，
#   否则先后无法判断，当日不交易。
# - 入场 bar 内只检查止损（按止损价成交），不检查止盈：bar 内先后未知，取保守一侧。
# - 之后的 bar：止损、止盈同一根 bar 都触发时按止损；跳空越过时按开盘价。
# - 其余持仓在当日最后一根 bar 按收盘价平仓。

def day_layout(trading_date: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每根 bar 的交易日编号，及每个交易日首末 bar 的位置。要求已按时间排序。"""
    td = np.asarray(trading_date)
    change = np.r_[True, td[1:] != td[:-1]]
    if len(td) > 1 and (td[1:] < td[:-1]).any():
        raise ValueError("trading_date 必须随时间单调不减")
    codes = np.cumsum(change) - 1
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(td)] - 1
    return codes, starts, ends


def first_true(mask: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """每个交易日内第一根为真的 bar 的全局位置；没有则为 ``len(mask)``。"""
    n = len(mask)
    return np.minimum.reduceat(np.where(mask, np.arange(n), n), starts)


def day_open_offset(index: pd.DatetimeIndex, codes: np.ndarray,
                    starts: np.ndarray) -> np.ndarray:
    """每个交易日日盘第一根 bar 在当日的序号。没有日盘则为 -1。

    夜盘是 20:00 及以后、或凌晨 4 点及以前。日盘从 8:30 起，取第一根不是夜盘且小时不少于 8 的 bar。
    """
    hour = np.asarray(pd.DatetimeIndex(index).hour)
    is_day = ~((hour >= 20) | (hour <= 4)) & (hour >= 8)
    pos = np.arange(len(hour)) - starts[codes]
    big = np.where(is_day, pos, np.iinfo(np.int64).max)
    origin = np.minimum.reduceat(big, starts)
    return np.where(origin == np.iinfo(np.int64).max, -1, origin).astype(np.int64)


def opening_range(high: np.ndarray, low: np.ndarray, codes: np.ndarray,
                  starts: np.ndarray, ends: np.ndarray, minutes: int,
                  origin: np.ndarray | None = None):
    """从每个交易日的 ``origin`` 起、连续 ``minutes`` 根 bar 的最高最低价。

    ``origin`` 缺省为 0，即交易日第一根（夜盘品种含夜盘）。当日从起点算起不足
    ``minutes`` 根，或起点为 -1 时，结果为 NaN。
    """
    if origin is None:
        origin = np.zeros(len(starts), np.int64)
    else:
        origin = np.asarray(origin)
    pos = np.arange(len(high)) - starts[codes]
    rel = pos - origin[codes]
    take = (origin[codes] >= 0) & (rel >= 0) & (rel < int(minutes))
    hi = np.maximum.reduceat(np.where(take, high, -np.inf), starts)
    lo = np.minimum.reduceat(np.where(take, low, np.inf), starts)
    enough = (origin >= 0) & ((ends - starts + 1 - np.maximum(origin, 0)) >= int(minutes))
    return np.where(enough, hi, np.nan), np.where(enough, lo, np.nan)


def simulate(o, h, l, c, codes, starts, ends, long_level, short_level,
             stop_dist, target_dist, start_bar, allow_long, allow_short, tradable=None):
    """返回逐笔成交的数组字典（只含有成交的交易日），价格未含滑点。

    日级数组长度等于交易日数：``long_level``/``short_level`` 为突破价，
    ``stop_dist``/``target_dist`` 为距入场价的止损/止盈距离（NaN 表示不设），
    ``start_bar`` 为当日最早可入场的 bar 序号，``allow_*`` 为当日允许的方向。
    ``tradable`` 为逐 bar 布尔数组（通常是成交量 > 0），为假的 bar 不能入场。
    """
    n = len(o)
    bar = np.arange(n)
    pos = bar - starts[codes]
    ok = pos >= start_bar[codes]
    if tradable is not None:
        ok &= np.asarray(tradable, dtype=bool)
    with np.errstate(invalid="ignore"):
        hit_long = ok & allow_long[codes] & (h > long_level[codes])
        hit_short = ok & allow_short[codes] & (l < short_level[codes])
        open_long = o >= long_level[codes]
        open_short = o <= short_level[codes]
    # 同一根 bar 两侧都穿过：开盘已越过的一侧在先；都没越过（或两侧都越过）则无法判断
    both = hit_long & hit_short
    hit_long = hit_long & ~(both & ~(open_long & ~open_short))
    hit_short = hit_short & ~(both & ~(open_short & ~open_long))
    ambiguous = both & (open_long == open_short)
    fl, fs = first_true(hit_long, starts), first_true(hit_short, starts)
    fa = first_true(ambiguous, starts)
    entry_bar = np.minimum(fl, fs)
    traded = (entry_bar < n) & (fa > entry_bar)
    days = np.flatnonzero(traded)
    ei = entry_bar[days]
    direction = np.where(fl[days] < fs[days], 1, -1)
    entry = np.where(direction > 0, np.maximum(o[ei], long_level[days]),
                     np.minimum(o[ei], short_level[days]))
    stop = entry - direction * stop_dist[days]
    target = entry + direction * target_dist[days]

    n_days = len(starts)
    d_dir = np.zeros(n_days, dtype=np.int8)
    d_dir[days] = direction
    d_entry_bar = np.full(n_days, n)
    d_entry_bar[days] = ei
    d_stop = np.full(n_days, np.nan)
    d_stop[days] = stop
    d_target = np.full(n_days, np.nan)
    d_target[days] = target
    b_dir, b_entry = d_dir[codes], d_entry_bar[codes]
    b_stop, b_target = d_stop[codes], d_target[codes]
    at, after = bar == b_entry, bar > b_entry
    with np.errstate(invalid="ignore"):
        stop_hit = (at | after) & (((b_dir > 0) & (l <= b_stop)) | ((b_dir < 0) & (h >= b_stop)))
        target_hit = after & (((b_dir > 0) & (h >= b_target)) | ((b_dir < 0) & (l <= b_target)))
    fstop = first_true(stop_hit, starts)[days]
    ftarget = first_true(target_hit, starts)[days]
    eod = ends[days]
    by_stop = (fstop < n) & (fstop <= ftarget)
    by_target = ~by_stop & (ftarget < n)
    xi = np.where(by_stop, fstop, np.where(by_target, ftarget, eod))
    ox = o[xi]
    stop_px = np.where(xi == ei, stop,
                       np.where(direction > 0, np.minimum(ox, stop), np.maximum(ox, stop)))
    target_px = np.where(direction > 0, np.maximum(ox, target), np.minimum(ox, target))
    exit_px = np.where(by_stop, stop_px, np.where(by_target, target_px, c[eod]))
    reason = np.where(by_stop, "stop_loss", np.where(by_target, "take_profit", "end_of_day"))
    return {"day": days, "entry_bar": ei, "exit_bar": xi, "direction": direction,
            "entry": entry, "exit": exit_px, "reason": reason}


def _entries(o, h, l, n, codes, pos, flat, nxt, allow_long, allow_short, enter_long,
             enter_short, tradable, starts):
    """空仓交易日在 ``nxt`` 之后第一根触发的入场 bar（口径同 ``simulate``）。"""
    ok = flat[codes] & (pos >= nxt[codes]) & tradable
    with np.errstate(invalid="ignore"):
        hit_long = ok & allow_long[codes] & (h > enter_long)
        hit_short = ok & allow_short[codes] & (l < enter_short)
        open_long = o >= enter_long
        open_short = o <= enter_short
    both = hit_long & hit_short
    hit_long = hit_long & ~(both & ~(open_long & ~open_short))
    hit_short = hit_short & ~(both & ~(open_short & ~open_long))
    ambiguous = both & (open_long == open_short)
    fl, fs = first_true(hit_long, starts), first_true(hit_short, starts)
    fa = first_true(ambiguous, starts)
    eb = np.minimum(fl, fs)
    return eb, (eb < n) & (fa > eb), np.where(fl < fs, 1, -1)


def simulate_path(o, h, l, c, codes, starts, ends, enter_long, enter_short,
                  rev_to_short, rev_to_long, stop_dist, target_dist, start_bar,
                  allow_long, allow_short, tradable=None, max_legs=1):
    """带反手的日内撮合。返回字段同 ``simulate``，``reason`` 多一种 ``reverse``。

    逐 bar 数组：``enter_long``/``enter_short`` 为空仓时的开仓价（NaN 为不开）；
    ``rev_to_short`` 为持多时的反手价（最低价严格跌破即平多，允许做空则按同价开空），
    ``rev_to_long`` 对称。逐日数组：止损/止盈距离、最早入场 bar、允许的方向。

    口径（与 ``simulate`` 一致并补充反手）：
    - 反手只在入场 bar 之后、且成交量 > 0 的 bar 上触发，按 min(开盘, 反手价)（多头）成交；
    - 同一根 bar 止损与反手都触发时，按成交价先到的一侧（多头取较高者；相同则算反手）；
      止损/反手与止盈同 bar 时按不利一侧；
    - 反手开出的新仓在同一根 bar 只检查止损；最后一根 bar 上只平不反；
    - 反手不允许的方向（``allow_*`` 为假）时只平仓，之后仍可按开仓价重新入场；
    - ``max_legs`` 为每日最多持仓段数（反手开出的仓也算一段）。
    """
    n, n_days = len(o), len(starts)
    bar = np.arange(n)
    pos = bar - starts[codes]
    tradable = np.ones(n, bool) if tradable is None else np.asarray(tradable, bool)
    allow_long = np.asarray(allow_long, bool)
    allow_short = np.asarray(allow_short, bool)
    side = np.zeros(n_days, np.int64)
    ebar = np.full(n_days, n)
    epx = np.full(n_days, np.nan)
    nxt = np.asarray(start_bar, np.int64).copy()
    legs = np.zeros(n_days, np.int64)
    alive = ends >= starts
    out = {k: [] for k in ("day", "entry_bar", "exit_bar", "direction", "entry", "exit", "reason")}
    while True:
        flat = alive & (side == 0)
        if flat.any():
            eb, got, dirn = _entries(o, h, l, n, codes, pos, flat, nxt, allow_long, allow_short,
                                     enter_long, enter_short, tradable, starts)
            alive &= ~(flat & ~got)
            d = np.flatnonzero(flat & got)
            ei, dd = eb[d], dirn[d]
            side[d], ebar[d], legs[d] = dd, ei, legs[d] + 1
            epx[d] = np.where(dd > 0, np.maximum(o[ei], enter_long[ei]),
                              np.minimum(o[ei], enter_short[ei]))
        held = alive & (side != 0)
        if not held.any():
            break
        d = np.flatnonzero(held)
        stop_d = np.where(held, epx - side * stop_dist, np.nan)
        target_d = np.where(held, epx + side * target_dist, np.nan)
        b_dir = np.where(held, side, 0)[codes]
        b_e, b_stop, b_target = ebar[codes], stop_d[codes], target_d[codes]
        at, after = bar == b_e, bar > b_e
        longs, shorts = b_dir > 0, b_dir < 0
        with np.errstate(invalid="ignore"):
            stop_hit = (at | after) & ((longs & (l <= b_stop)) | (shorts & (h >= b_stop)))
            target_hit = after & ((longs & (h >= b_target)) | (shorts & (l <= b_target)))
            rev_hit = after & tradable & ((longs & (l < rev_to_short)) | (shorts & (h > rev_to_long)))
        fsx = first_true(stop_hit, starts)[d]
        ftx = first_true(target_hit, starts)[d]
        frx = first_true(rev_hit, starts)[d]
        dirn, ei, stop, target = side[d], ebar[d], stop_d[d], target_d[d]

        def _px(idx, lvl_long, lvl_short):
            j = np.minimum(idx, n - 1)
            return np.where(dirn > 0, np.minimum(o[j], lvl_long), np.maximum(o[j], lvl_short)), j

        stop_px, js = _px(fsx, stop, stop)
        stop_px = np.where(js == ei, stop, stop_px)
        jr = np.minimum(frx, n - 1)
        rev_px = np.where(dirn > 0, np.minimum(o[jr], rev_to_short[jr]),
                          np.maximum(o[jr], rev_to_long[jr]))
        jt = np.minimum(ftx, n - 1)
        target_px = np.where(dirn > 0, np.maximum(o[jt], target), np.minimum(o[jt], target))
        with np.errstate(invalid="ignore"):
            rev_first = (frx < fsx) | ((frx == fsx) & (frx < n) & (dirn * (rev_px - stop_px) >= 0))
        adverse = np.minimum(fsx, frx)
        is_adv = (adverse < n) & (adverse <= ftx)
        is_rev = is_adv & rev_first
        is_stop = is_adv & ~rev_first
        is_tgt = ~is_adv & (ftx < n)
        xi = np.where(is_rev, frx, np.where(is_stop, fsx, np.where(is_tgt, ftx, ends[d])))
        exit_px = np.where(is_rev, rev_px, np.where(is_stop, stop_px,
                                                    np.where(is_tgt, target_px, c[ends[d]])))
        reason = np.where(is_rev, "reverse", np.where(is_stop, "stop_loss",
                                                      np.where(is_tgt, "take_profit", "end_of_day")))
        for k, v in (("day", d), ("entry_bar", ei), ("exit_bar", xi), ("direction", dirn),
                     ("entry", epx[d]), ("exit", exit_px), ("reason", reason)):
            out[k].append(v)
        # 状态更新
        eod = ~(is_rev | is_stop | is_tgt)
        can_flip = is_rev & (xi < ends[d]) & (legs[d] < max_legs) & np.where(
            dirn > 0, allow_short[d], allow_long[d])
        side[d] = np.where(can_flip, -dirn, 0)
        ebar[d] = np.where(can_flip, xi, n)
        epx[d] = np.where(can_flip, exit_px, np.nan)
        legs[d] += can_flip
        nxt[d] = xi - starts[d] + 1
        alive[d] = ~eod & (can_flip | (legs[d] < max_legs))
    if not out["day"]:
        empty = np.array([], dtype=np.int64)
        return {"day": empty, "entry_bar": empty, "exit_bar": empty, "direction": empty,
                "entry": np.array([]), "exit": np.array([]), "reason": np.array([], dtype="<U11")}
    res = {k: np.concatenate(v) for k, v in out.items()}
    order = np.lexsort((res["entry_bar"], res["day"]))
    return {k: v[order] for k, v in res.items()}
