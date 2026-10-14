"""国泰君安股指时间维度因子。只服务 IF、IH、IC 的研究期复现。

持续期按研报正文：从当前分钟向更早的分钟看，价格或成交量差第一次大于阈值就停下，
持续期是这两根分钟的间隔。当天没有越过阈值时，从开盘累加。交易日之间不衔接。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfcta.factors import _day_spans, day_codes_of

MORNING = (9 * 60 + 31, 10 * 60)
AFTERNOON_CLOSE = (14 * 60 + 31, 15 * 60)
COLUMNS = (
    "Price_duration_max_min", "Price_duration_vol", "Price_duration_averg",
    "Price_duration_corr", "Price_duration_return", "Price_duration_slope",
    "Price_duration_extreme", "Price_duration_open_close", "Price_duration_adj_vol",
    "Price_volume_persistence_ratio",
    "Volume_duration_max_min", "Volume_duration_vol", "Volume_duration_averg",
    "Volume_duration_corr", "Volume_duration_return", "Volume_duration_slope",
    "Volume_duration_extreme", "Volume_duration_open_close", "Volume_duration_adj_vol",
    "Volume_volume_persistence_ratio",
    "time_to_high", "time_to_low", "time_to_max_volume", "time_to_max_turnover",
    "time_to_close_above_open", "time_to_close_below_open",
    "afternoon_high_time", "time_to_half_volume",
)


def duration_one_day(values: np.ndarray, threshold: float) -> np.ndarray:
    """差必须严格大于阈值。相等不算中断。"""
    values = np.asarray(values, dtype="float64")
    n = len(values)
    if n == 0:
        return np.zeros(0)
    if not np.isfinite(threshold) or threshold <= 0:
        return np.full(n, np.nan)
    gap = np.abs(values[:, None] - values[None, :])
    earlier = np.arange(n)[None, :] < np.arange(n)[:, None]
    hit = (gap > threshold) & earlier & np.isfinite(gap)
    found = hit.any(axis=1)
    nearest = np.where(found, n - 1 - hit[:, ::-1].argmax(axis=1), 0)
    duration = np.where(found, np.arange(n) - nearest, np.arange(n)).astype("float64")
    duration[~np.isfinite(values)] = np.nan
    return duration


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    return values[np.isfinite(values)]


def _std(values: np.ndarray) -> float:
    kept = _finite(values)
    if kept.size < 2:
        return np.nan
    return float(kept.std(ddof=1))


def _mean(values: np.ndarray) -> float:
    kept = _finite(values)
    if kept.size == 0:
        return np.nan
    return float(kept.mean())


def _slope(duration: np.ndarray) -> float:
    x = np.arange(len(duration), dtype="float64")
    ok = np.isfinite(duration)
    if int(ok.sum()) < 2:
        return np.nan
    x = x[ok] - x[ok].mean()
    y = duration[ok] - duration[ok].mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def _corr(price: np.ndarray, duration: np.ndarray) -> float:
    ok = np.isfinite(price) & np.isfinite(duration)
    if int(ok.sum()) < 3:
        return np.nan
    price_scale, duration_scale = _std(price[ok]), _std(duration[ok])
    if not np.isfinite(price_scale) or not np.isfinite(duration_scale) or price_scale == 0 or duration_scale == 0:
        return np.nan
    return float(np.corrcoef(price[ok], duration[ok])[0, 1])


def _extreme_share(duration: np.ndarray) -> float:
    kept = _finite(duration)
    scale = _std(kept)
    if kept.size < 2 or not np.isfinite(scale):
        return np.nan
    return float(np.mean(kept > kept.mean() + 2 * scale))


def _range(duration: np.ndarray) -> float:
    kept = _finite(duration)
    if kept.size == 0:
        return np.nan
    return float(kept.max() - kept.min())


def _window_gap(duration: np.ndarray, minute: np.ndarray) -> float:
    morning = (minute >= MORNING[0]) & (minute <= MORNING[1])
    close = (minute >= AFTERNOON_CLOSE[0]) & (minute <= AFTERNOON_CLOSE[1])
    if not (np.isfinite(duration[morning]).any() and np.isfinite(duration[close]).any()):
        return np.nan
    return _mean(duration[close]) - _mean(duration[morning])


def _persistence(duration: np.ndarray, volume: np.ndarray) -> float:
    kept = _finite(duration)
    total = _finite(volume).sum()
    if kept.size < 5 or not np.isfinite(total) or total <= 0:
        return np.nan
    cut = float(np.quantile(kept, 0.8))
    chosen = np.isfinite(duration) & np.isfinite(volume) & (duration > cut)
    return float(volume[chosen].sum() / total)


def _location(values: np.ndarray, mode: str) -> float:
    if len(values) < 2 or not np.isfinite(values).any():
        return np.nan
    if mode == "max":
        scored = np.where(np.isfinite(values), values, -np.inf)
        loc = int(np.argmax(scored))
    else:
        scored = np.where(np.isfinite(values), values, np.inf)
        loc = int(np.argmin(scored))
    return float(loc / (len(values) - 1))


def _first_cross(close: np.ndarray, open_px: float, above: bool) -> float:
    if len(close) < 2 or not np.isfinite(open_px):
        return np.nan
    hit = close > open_px if above else close < open_px
    hit &= np.isfinite(close)
    where = np.flatnonzero(hit)
    if where.size == 0:
        return np.nan
    return float(where[0] / (len(close) - 1))


def _half_volume(volume: np.ndarray) -> float:
    if len(volume) < 2 or not np.isfinite(volume).all() or volume.sum() <= 0:
        return np.nan
    reached = np.flatnonzero(np.cumsum(volume) >= 0.5 * volume.sum())
    if reached.size == 0:
        return np.nan
    return float(reached[0] / (len(volume) - 1))


def _day_row(price_dur, volume_dur, close, close_w, high_w, low_w, volume, turnover, minute, day_open):
    price_scale = _std(close)
    row = {
        "Price_duration_max_min": _range(price_dur),
        "Price_duration_vol": _std(price_dur),
        "Price_duration_averg": _mean(price_dur),
        "Price_duration_corr": _corr(close, price_dur),
        "Price_duration_slope": _slope(price_dur),
        "Price_duration_extreme": _extreme_share(price_dur),
        "Price_duration_open_close": _window_gap(price_dur, minute),
        "Price_duration_adj_vol": _std(price_dur) / price_scale if price_scale and price_scale > 0 else np.nan,
        "Price_volume_persistence_ratio": _persistence(price_dur, volume),
        "Volume_duration_max_min": _range(volume_dur),
        "Volume_duration_vol": _std(volume_dur),
        "Volume_duration_averg": _mean(volume_dur),
        "Volume_duration_corr": _corr(close, volume_dur),
        "Volume_duration_slope": _slope(volume_dur),
        "Volume_duration_extreme": _extreme_share(volume_dur),
        "Volume_duration_open_close": _window_gap(volume_dur, minute),
        "Volume_duration_adj_vol": _std(volume_dur) / price_scale if price_scale and price_scale > 0 else np.nan,
        "Volume_volume_persistence_ratio": _persistence(volume_dur, volume),
        "time_to_high": _location(high_w, "max"),
        "time_to_low": _location(low_w, "min"),
        "time_to_max_volume": _location(volume, "max"),
        "time_to_max_turnover": _location(turnover, "max"),
        "time_to_close_above_open": _first_cross(close, day_open, True),
        "time_to_close_below_open": _first_cross(close, day_open, False),
        "time_to_half_volume": _half_volume(volume),
    }
    afternoon = minute >= 13 * 60
    row["afternoon_high_time"] = _location(high_w[afternoon], "max")
    price_at = _finite_argmax(price_dur)
    volume_at = _finite_argmax(volume_dur)
    row["price_rep"] = close_w[price_at] if price_at is not None else np.nan
    row["volume_rep"] = close_w[volume_at] if volume_at is not None else np.nan
    return row


def _finite_argmax(values: np.ndarray):
    if not np.isfinite(values).any():
        return None
    scored = np.where(np.isfinite(values), values, -np.inf)
    return int(np.argmax(scored))


def daily_factors(frame: pd.DataFrame, price_threshold: float, volume_threshold: float) -> pd.DataFrame:
    """一个品种的日频因子。代表价格用复权收盘，避免换月把收益因子打断。"""
    needed = ("trading_date", "close", "closew", "open", "openw", "highw", "loww", "volume", "total_turnover")
    missing = [col for col in needed if col not in frame.columns]
    if missing:
        raise KeyError(f"分钟分片缺少 {missing}")
    frame = frame.sort_index()
    codes, days = day_codes_of(frame)
    close = frame.close.to_numpy(dtype="float64")
    close_w = frame.closew.to_numpy(dtype="float64")
    high_w = frame.highw.to_numpy(dtype="float64")
    low_w = frame.loww.to_numpy(dtype="float64")
    volume = frame.volume.to_numpy(dtype="float64")
    turnover = frame.total_turnover.to_numpy(dtype="float64")
    minute = (frame.index.hour * 60 + frame.index.minute).to_numpy()
    opens = frame.open.to_numpy(dtype="float64")
    rows = []
    for a, b in zip(*_day_spans(codes)):
        price_dur = duration_one_day(close[a:b], price_threshold)
        volume_dur = duration_one_day(volume[a:b], volume_threshold)
        rows.append(_day_row(price_dur, volume_dur, close[a:b], close_w[a:b], high_w[a:b],
                             low_w[a:b], volume[a:b], turnover[a:b], minute[a:b], opens[a]))
    out = pd.DataFrame(rows, index=days)
    out.index.name = "trading_date"
    out["Price_duration_return"] = out.price_rep / out.price_rep.shift(1) - 1
    out["Volume_duration_return"] = out.volume_rep / out.volume_rep.shift(1) - 1
    return out.drop(columns=["price_rep", "volume_rep"])


def rolling_z(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """含当天的滚动标准化。后面的观测不改前面的值。标准差为 0 时留空。"""
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=min_periods).mean()
    scale = values.rolling(window, min_periods=min_periods).std(ddof=1)
    return (values - mean) / scale.where(scale > 0)


def standardize(factors: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    return pd.DataFrame({col: rolling_z(factors[col], window, min_periods) for col in factors.columns})


def pair_signal(standardized: pd.DataFrame, long_signs: dict) -> pd.Series:
    """做多要求每个因子都落在规定的一侧。全部翻到对面则做空，否则空仓。"""
    long = pd.Series(True, index=standardized.index)
    short = pd.Series(True, index=standardized.index)
    for name, sign in long_signs.items():
        values = standardized[name]
        long &= values * sign > 0
        short &= values * sign < 0
    signal = pd.Series(0.0, index=standardized.index)
    signal = signal.mask(long, 1.0).mask(short, -1.0)
    return signal


def open_to_open(frame: pd.DataFrame) -> pd.Series:
    """当天开盘到下一次开盘。最后一天没有下一次开盘，留空，避免用到样本外价格。"""
    opens = frame.sort_index().groupby(pd.to_datetime(frame.trading_date).dt.normalize(), sort=True).openw.first()
    opens.index = pd.DatetimeIndex(opens.index)
    return opens.shift(-1) / opens - 1


def backtest(signal: pd.Series, forward: pd.Series, fee_per_side: float) -> pd.DataFrame:
    """收盘信号在下一交易日开盘成交。仓位不变不扣费。"""
    aligned = pd.concat({"signal": signal, "forward": forward}, axis=1).sort_index()
    position = aligned.signal.shift(1).fillna(0.0)
    turnover = (position - position.shift(1).fillna(0.0)).abs()
    net = position * aligned.forward - fee_per_side * turnover
    out = pd.DataFrame({"position": position, "gross": position * aligned.forward,
                        "cost": fee_per_side * turnover, "net": net})
    return out.loc[out.gross.notna()]


def profit_factor(position: pd.Series, net: pd.Series) -> float:
    """连续同一仓位算一笔，盈利笔合计除以亏损笔合计的绝对值。"""
    held = position.reindex(net.index).fillna(0.0)
    changed = held.ne(held.shift(1).fillna(0.0))
    trade = changed.cumsum()
    pnl = net.where(held != 0).groupby(trade).sum().dropna()
    wins = float(pnl[pnl > 0].sum())
    losses = float(pnl[pnl < 0].sum())
    if losses == 0:
        return np.nan
    return wins / abs(losses)


def performance(result: pd.DataFrame) -> dict:
    net = result.net.dropna()
    if net.empty:
        return {"天数": 0, "持仓占比": np.nan, "年化": np.nan, "夏普": np.nan,
                "最大回撤": np.nan, "卡玛": np.nan, "盈亏比": np.nan}
    growth = 1 + net
    if growth.le(0).any():
        return {"天数": int(len(net)), "持仓占比": np.nan, "年化": np.nan, "夏普": np.nan,
                "最大回撤": np.nan, "卡玛": np.nan, "盈亏比": np.nan}
    equity = growth.cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = float(drawdown.min())
    annual = float(equity.iloc[-1] ** (252 / len(net)) - 1)
    scale = float(net.std(ddof=1))
    sharpe = float(net.mean() / scale * np.sqrt(252)) if scale > 0 else np.nan
    calmar = float(annual / abs(max_dd)) if max_dd < 0 else np.nan
    held = result.position.reindex(net.index).fillna(0.0)
    return {
        "天数": int(len(net)),
        "持仓占比": float((held != 0).mean()),
        "年化": annual,
        "夏普": sharpe,
        "最大回撤": max_dd,
        "卡玛": calmar,
        "盈亏比": profit_factor(held, net),
    }
