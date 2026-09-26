"""第 6、7 步的数据准备：把研究期分钟与已允许的验证期 / 已结束样本外窗口接起来，
用与第 3 步相同的公式重算时间因子，并给出日线、日收益和逐年 tick 表。

窗口在分区之间连续：阈值、z 分数等滚动量在 2022 或 2023 开头不会被截断。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from ... import config as C
from ...data import bars as B
from ...data import sessions
from ...factors import intraday, library
from ..backtest import costs


@dataclass
class History:
    factors: dict[str, pd.DataFrame] = field(default_factory=dict)
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    ticks: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def load_history(symbols: list[str], *,
                 include_validation: bool,
                 oos_end: dt.date | None = None,
                 lookback: int | None = None,
                 pct: float | None = None) -> History:
    lookback = C.IC_REFERENCE_LOOKBACK if lookback is None else int(lookback)
    pct = C.IC_REFERENCE_PCT if pct is None else float(pct)
    out = History()
    tick_rows = []
    for symbol in symbols:
        try:
            minute = B.load_minutes(symbol, include_validation=include_validation, oos_end=oos_end)
        except FileNotFoundError as exc:
            out.skipped.append((symbol, str(exc).splitlines()[0]))
            continue
        coords = sessions.add_intraday_coords(minute)
        out.factors[symbol] = intraday.symbol_daily_factors(
            coords, lookback=lookback, pct=pct, with_coords=True)
        out.bars[symbol] = B.daily_bars(minute)
        tick_rows.extend(costs.tick_rows(symbol, minute['close'], minute['trading_date']))
        del minute, coords
    out.ticks = pd.DataFrame(tick_rows)
    return out


def signal_set(history: History, universe: dict,
               external_partitions=('research',)) -> library.SignalSet:
    time_raw = {
        name: pd.DataFrame({s: frame[name] for s, frame in history.factors.items()}).sort_index()
        for name in C.FACTOR_SIGNS
    }
    return library.assemble(B.wide_by_field(history.bars), time_raw, universe,
                            external_partitions)


def day_returns(history: History) -> pd.DataFrame:
    return pd.DataFrame({
        s: B.day_return_from_prices(frame[['open', 'openw']])
        for s, frame in history.bars.items()
    }).sort_index()
