"""保留策略共用的时钟尾部、波动标记和开仓日程。

这里不读行情。趋势符号用 tfcta.factors.tsmom_sign。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 250
QUANTILE = 0.8
HOLDING_DAYS = 5
TREND_WINDOW = 10
ATR_PERIOD = 14
ATR_REFERENCE = 250
ADX_PERIOD = 14
ADX_RANGE_MAX = 20.0


def accept_lock(config: dict) -> None:
    """冻结必须同时写明日期和 chosen，而且 chosen 等于 parameters。"""
    frozen_at = config.get("frozen_at")
    chosen = config.get("chosen")
    if bool(frozen_at) != (chosen is not None):
        raise ValueError("冻结必须同时写明 frozen_at 和 chosen")
    if "不读 2023" not in config.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2023 年以后")
    if chosen is not None and chosen != config.get("parameters"):
        raise ValueError("chosen 必须与 parameters 一致")


def upper_tail(series: pd.Series, lookback: int = LOOKBACK, quantile: float = QUANTILE) -> pd.Series:
    """过去 lookback 个有值的交易日，不含当天。只在严格高于分位时为 1。"""
    values = series.to_numpy(dtype="float64")
    sides = np.zeros(values.size, dtype="float64")
    finite_at = np.flatnonzero(np.isfinite(values))
    for nth, loc in enumerate(finite_at):
        if nth < lookback:
            continue
        hist = values[finite_at[nth - lookback:nth]]
        hi = float(np.quantile(hist, quantile, method="linear"))
        if float(values[loc]) > hi:
            sides[loc] = 1.0
    return pd.Series(sides, index=series.index)


def high_volatility(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """14 日真实波幅除以复权收盘，严格高于自己过去 250 日、不含当天的中位数。"""
    atr = atr_percent(bars)
    prior_median = atr.shift(1).rolling(ATR_REFERENCE, min_periods=ATR_REFERENCE).median()
    return atr.gt(prior_median)


def atr_percent(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """14 日 Wilder 真实波幅除以加法复权收盘。"""
    close = bars["closew"].astype("float64")
    average_tr = _true_range(bars).ewm(
        alpha=1.0 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False,
    ).mean()
    return average_tr / close.where(close > 0)


def _true_range(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    high = bars["highw"].astype("float64")
    low = bars["loww"].astype("float64")
    close = bars["closew"].astype("float64")
    prior_close = close.shift(1)
    return pd.DataFrame(
        np.fmax.reduce([
            (high - low).to_numpy(),
            (high - prior_close).abs().to_numpy(),
            (low - prior_close).abs().to_numpy(),
        ]),
        index=close.index,
        columns=close.columns,
    )


def _adx(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """14 日 Wilder ADX。预热不足的位置是缺失值。"""
    high = bars["highw"].astype("float64")
    low = bars["loww"].astype("float64")
    true_range = _true_range(bars)
    upward = high.diff()
    downward = -low.diff()
    plus_dm = upward.where((upward > downward) & (upward > 0), 0.0)
    minus_dm = downward.where((downward > upward) & (downward > 0), 0.0)
    alpha = 1.0 / ADX_PERIOD
    average_tr = true_range.ewm(alpha=alpha, min_periods=ADX_PERIOD, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(
        alpha=alpha, min_periods=ADX_PERIOD, adjust=False,
    ).mean() / average_tr.where(average_tr > 0)
    minus_di = 100.0 * minus_dm.ewm(
        alpha=alpha, min_periods=ADX_PERIOD, adjust=False,
    ).mean() / average_tr.where(average_tr > 0)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).where(
        (plus_di + minus_di) > 0,
    )
    return dx.ewm(alpha=alpha, min_periods=ADX_PERIOD, adjust=False).mean()


def adx_is_range(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """14 日 ADX 有值，并且小于等于 20。这是实验里的震荡状态。"""
    adx = _adx(bars)
    return adx.notna() & adx.le(ADX_RANGE_MAX)


def adx_above_range(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """14 日 ADX 有值，并且大于 20。含过渡段和单边段，不含 ADX 尚未算出来的日子。"""
    adx = _adx(bars)
    return adx.notna() & adx.gt(ADX_RANGE_MAX)


def entry_schedule(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    universe: dict,
    start,
    end,
    holding_days: int = HOLDING_DAYS,
    signed: bool = False,
    preserve_weights: bool = False,
) -> pd.DataFrame:
    """收盘信号的下一可交易日开仓。持有期内以及到期当日都不再开新仓。"""
    if holding_days < 1:
        raise ValueError("持仓期必须至少为 1 个交易日")
    if preserve_weights and not signed:
        raise ValueError("保留仓位权重时必须同时启用 signed")
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    entries = pd.DataFrame(
        0 if signed else False,
        index=returns.index,
        columns=returns.columns,
        dtype="float64" if preserve_weights else "int8" if signed else bool,
    )
    pools = {int(year): set(names) for year, names in universe.items()}
    for symbol in returns.columns:
        if symbol not in signal.columns:
            continue
        tradable = returns.index[np.isfinite(returns[symbol].to_numpy(dtype="float64"))]
        order = {pd.Timestamp(day): i for i, day in enumerate(tradable)}
        blocked_until = -1
        active = signal[symbol].fillna(0)
        for day in signal.index[active.ne(0)]:
            day = pd.Timestamp(day)
            if day < start or day > end or symbol not in pools.get(day.year, set()):
                continue
            later = tradable[tradable > day]
            if not len(later):
                continue
            entry_day = pd.Timestamp(later[0])
            entry_position = order[entry_day]
            if (
                symbol not in pools.get(entry_day.year, set())
                or entry_position <= blocked_until
                or entry_position + holding_days >= len(tradable)
            ):
                continue
            entries.at[entry_day, symbol] = (
                float(active.at[day]) if preserve_weights
                else int(np.sign(active.at[day])) if signed else True
            )
            blocked_until = entry_position + holding_days
    return entries
