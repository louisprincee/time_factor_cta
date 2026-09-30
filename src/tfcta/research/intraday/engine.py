"""1 分钟开盘区间突破（ORB）等日内单边撮合、多日持仓记账与绩效计算。

成交规则固定为保守口径：
- 开盘区间 = 从开盘起点（交易日第一根，或日盘第一根）起连续 ``minutes`` 根 bar 的最高最低价，
  区间走完后的下一根 bar 起才可入场；
- 做多：bar 最高价严格高于上轨（穿过而非碰到）触发，开盘已越过上轨按开盘价成交，否则按上轨；
  做空对称。成交量为 0 的 bar 不能入场；
- 每天最多一笔，当日最后一根 bar 按收盘价平仓，不持仓到下一个交易日。

同样的撮合口径推广到 ``touch_trades``（任意触价水平）和 ``next_open_trades``（bar 收盘信号、
下一根开盘入场）；多日持仓策略用 ``position_returns`` 按 bar 记仓位、按交易日记收益与换手成本。

多空两条单边路径分别撮合（``orb_trades(..., side=±1)``），不存在同一根 bar 上下轨都被穿过、
先后无法判断的情况。

``trading_date`` 以夜盘开盘为一日之始，所以有夜盘品种的"日内"持仓会跨过夜盘收盘到
次日 9:00 的休市。价格用加法复权（``*w``）列，收益分母用入场 bar 的原始开盘价。
撮合只给不含滑点、不含手续费的收益；成本在 ``walk_forward`` 里按笔拆开计算。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OrbConfig:
    minutes: int = 30
    # "first"：交易日第一根 bar 起（夜盘品种含夜盘）；"day"：日盘第一根 bar 起
    open_anchor: str = "first"


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


def prepare(minute: pd.DataFrame) -> Prepared:
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
    o, h, l, c = (df[k].to_numpy("float64") for k in ("open", "high", "low", "close"))
    codes, starts, ends = day_layout(df["trading_date"].to_numpy())
    dates = pd.DatetimeIndex(df["trading_date"].to_numpy()[starts])
    tradable = (pd.to_numeric(df["volume"], errors="coerce").to_numpy("float64") > 0
                if "volume" in df else np.ones(len(df), bool))
    return Prepared(df, raw_open.to_numpy("float64"), o, h, l, c, codes, starts, ends, dates,
                    tradable)


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


def orb_origin(p: Prepared, orb: OrbConfig) -> np.ndarray:
    """每个交易日开盘区间起点在当日的序号（``"day"`` 口径没有日盘时为 -1）。"""
    if orb.open_anchor == "day":
        return day_open_offset(p.df.index, p.codes, p.starts)
    if orb.open_anchor == "first":
        return np.zeros(len(p.starts), np.int64)
    raise ValueError(f"未知开盘定义: {orb.open_anchor}")


def orb_trades(p: Prepared, orb: OrbConfig, side: int) -> dict[str, np.ndarray]:
    """单边 ORB 撮合（``side`` = 1 做多 / -1 做空），只返回有成交的交易日。

    逐笔数组：``day``（交易日编号）、``entry_bar``/``exit_bar``（全局 bar 位置）、
    ``entry``/``exit``（复权价）、``ref``（入场 bar 原始开盘价，收益分母）、
    ``gross``（未扣滑点和手续费的收益）。
    """
    origin = orb_origin(p, orb)
    upper, lower = opening_range(p.h, p.l, p.codes, p.starts, p.ends, orb.minutes, origin)
    start = np.where(origin >= 0, origin + int(orb.minutes), -1)
    return touch_trades(p, upper if side > 0 else lower, start, side)


def _check_side(side: int) -> None:
    if side not in (1, -1):
        raise ValueError("side 只能是 1 或 -1")


def _trade_arrays(p: Prepared, days, ei, xi, entry, side) -> dict[str, np.ndarray]:
    exit_ = p.c[xi]
    ref = p.raw_open[ei]
    ref = np.where(np.isfinite(ref) & (ref > 0), ref, entry)
    return {"day": days, "entry_bar": ei, "exit_bar": xi, "entry": entry, "exit": exit_,
            "ref": ref, "gross": side * (exit_ - entry) / ref}


def touch_trades(p: Prepared, level: np.ndarray, start: np.ndarray,
                 side: int) -> dict[str, np.ndarray]:
    """单边触价入场、尾盘平：当日序号不小于 ``start`` 的 bar 里，第一根穿过 ``level`` 的成交。

    ``level``/``start`` 逐交易日给出，``start`` < 0 或 ``level`` 缺失当天不做。做多要求最高价
    严格高于 ``level``，开盘已越过按开盘价成交，否则按 ``level``；做空对称。
    """
    _check_side(side)
    n = len(p.o)
    start = np.asarray(start)
    lv = np.asarray(level, np.float64)[p.codes]
    pos = np.arange(n) - p.starts[p.codes]
    ok = (start[p.codes] >= 0) & (pos >= start[p.codes]) & p.tradable
    with np.errstate(invalid="ignore"):
        hit = ok & ((p.h > lv) if side > 0 else (p.l < lv))
    first = first_true(hit, p.starts)
    days = np.flatnonzero(first < n)
    ei, xi = first[days], p.ends[days]
    entry = np.maximum(p.o[ei], lv[ei]) if side > 0 else np.minimum(p.o[ei], lv[ei])
    return _trade_arrays(p, days, ei, xi, entry, side)


def next_open_trades(p: Prepared, signal: np.ndarray, side: int) -> dict[str, np.ndarray]:
    """信号在 bar 收盘时成立 → 同一交易日下一根可成交 bar 按开盘价入场，尾盘平。

    每天只取第一个信号。信号出在当日最后一根，或之后没有可成交的 bar，当天不做。
    """
    _check_side(side)
    n = len(p.o)
    fired = first_true(np.asarray(signal, bool), p.starts)
    after = p.tradable & (np.arange(n) > fired[p.codes])
    first = first_true(after, p.starts)
    days = np.flatnonzero(first < n)
    ei, xi = first[days], p.ends[days]
    return _trade_arrays(p, days, ei, xi, p.o[ei], side)


def position_returns(o: np.ndarray, c: np.ndarray, raw: np.ndarray, codes: np.ndarray,
                     n_days: int, signal: np.ndarray, weight: np.ndarray,
                     side_costs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """多日持仓：逐 bar 目标方向 → 逐交易日收益与换手成本（均已乘风险权重）。

    - ``signal[b]``：bar b 收盘时的目标方向（-1..1），缺失沿用上一个，下一根 bar 开盘成交；
    - 名义仓位 = 方向 × 当日风险权重 ``weight[day]``（开盘前已知，缺失记 0），权重变化也算换手；
    - bar 之间的跳空归调仓前的仓位；价差用复权价，分母用该 bar 原始开盘价；
    - ``side_costs``：逐交易日、每单位名义单边的成本（如每边 1 tick、单边手续费），乘换手后按日加总；
    - ``n``：当日方向改变的次数（调仓笔数），``position``：当日平均名义敞口。
    """
    sgn = pd.Series(np.asarray(signal, np.float64)).ffill().fillna(0.0).to_numpy()
    held = np.r_[0.0, sgn[:-1]]
    x = held * np.nan_to_num(np.asarray(weight, np.float64))[codes]
    x_prev = np.r_[0.0, x[:-1]]
    c_prev = np.r_[np.nan, c[:-1]]
    ref = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan)
    with np.errstate(invalid="ignore"):
        gap = np.where(np.isfinite(c_prev), x_prev * (o - c_prev), 0.0)
        pnl = np.nan_to_num((gap + x * (c - o)) / ref)
    turn = np.abs(x - x_prev)
    out = {"gross": np.bincount(codes, pnl, minlength=n_days),
           "n": np.bincount(codes, (held != np.r_[0.0, held[:-1]]).astype(float), minlength=n_days),
           "position": np.bincount(codes, np.abs(x), minlength=n_days)
           / np.maximum(np.bincount(codes, minlength=n_days), 1)}
    for name, rate in side_costs.items():
        out[name] = np.bincount(codes, np.nan_to_num(turn * np.asarray(rate)[codes]),
                                minlength=n_days)
    return out
