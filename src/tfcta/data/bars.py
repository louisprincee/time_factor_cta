"""日频行情与收益口径（设计文档第 8.3 节），以及跨时间分区拼接分钟数据。

价格口径：分片里带 ``w`` 的字段是**加法**复权（``close - closew`` 日内恒定，只在换月日
跳变，而且离上市越远偏离越大，个别品种甚至为负）。价差可以用复权价，比例的分母
一律用原始价。收益因此写成 ``Δ复权价 / 原始价``。

时间口径：因子在 t 日收盘才知道，赚取的是 ``day_ret[t+1]``，仓位是信号的 ``shift(1)``。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .. import config as C
from . import shard_io

MINUTE_COLUMNS = list(dict.fromkeys([*C.FACTOR_FIELDS, *C.PRICE_FIELDS]))
DAILY_AGG = {
    'open': 'first', 'openw': 'first', 'close': 'last', 'closew': 'last',
    'highw': 'max', 'loww': 'min', 'volume': 'sum',
}
DAILY_FIELDS = tuple(DAILY_AGG)


def daily_bars(minute: pd.DataFrame) -> pd.DataFrame:
    """单品种分钟表 → 日线。开盘取首根，收盘取末根，高低取极值，量求和。"""
    td = pd.to_datetime(minute['trading_date']).dt.normalize()
    agg = {k: v for k, v in DAILY_AGG.items() if k in minute.columns}
    out = minute.groupby(td, sort=True).agg(agg)
    out.index.name = 'trading_date'
    return out


def daily_prices_from_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """单品种分钟表 → 每个 trading_date 一根开盘价。"""
    missing = [c for c in ('open', 'openw', 'trading_date') if c not in df.columns]
    if missing:
        raise KeyError(
            f"分钟分片缺少 {missing}，无法按第 8.3 节计算 day_ret。\n"
            "请用含 open/openw 的原始面板重跑 step1_shard_minutes.py，"
            "不要用 close 代替 open。"
        )
    return daily_bars(df[['open', 'openw', 'trading_date']])


def day_return_from_prices(prices: pd.DataFrame) -> pd.Series:
    """day_ret[t] = (openw[t+1] - openw[t]) / open[t]。末日没有 t+1，为 NaN。"""
    open_ = prices['open'].astype('float64')
    openw = prices['openw'].astype('float64')
    ret = (openw.shift(-1) - openw) / open_.where(open_ != 0)
    ret.name = 'day_ret'
    return ret


def forward_return(day_ret: pd.DataFrame) -> pd.DataFrame:
    """factor[t] 所预测的那段收益：day_ret[t+1]。

    day_ret[t+1] = (openw[t+2] - openw[t+1]) / open[t+1]，
    即 t 日收盘形成信号、t+1 开盘成交、持有到 t+2 开盘。
    """
    return day_ret.shift(-1)


def wide_by_field(bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """{品种: 日线} → {字段: 交易日 × 品种}。"""
    fields = next(iter(bars_by_symbol.values())).columns
    return {
        field: pd.DataFrame({s: frame[field] for s, frame in bars_by_symbol.items()}).sort_index()
        for field in fields
    }


def load_day_returns(symbols: list[str], directory=None) -> pd.DataFrame:
    """研究期日收益宽表。读取走 shard_io.load_shard，指向 holdout_locked/ 会被拒绝。"""
    cols = {}
    for s in symbols:
        df = shard_io.load_shard(s, directory=directory,
                                 columns=['open', 'openw', 'trading_date'])
        px = daily_prices_from_minutes(df)
        C.assert_no_holdout_dates(px.index, what=f"{s} 日频开盘价")
        cols[s] = day_return_from_prices(px)
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols).sort_index()
    out.index.name = 'trading_date'
    return out


def load_daily_bars(symbols: list[str], directory=None) -> dict[str, pd.DataFrame]:
    """研究期日线，每个字段一张 ``trading_date × 品种`` 宽表。"""
    per_symbol = {}
    for s in symbols:
        df = shard_io.load_shard(s, directory=directory,
                                 columns=list(DAILY_FIELDS) + ['trading_date'])
        day = daily_bars(df)
        C.assert_no_holdout_dates(day.index, what=f"{s} 日频行情")
        per_symbol[s] = day
    if not per_symbol:
        return {k: pd.DataFrame() for k in DAILY_FIELDS}
    return wide_by_field(per_symbol)


def multiplicative_prices(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """加法复权 → 乘法复权的收盘、最高、最低价。

    ``pct_change`` 和"相对均线偏离"这类比例指标不能直接用加法复权价。这里用
    ``Δclosew / close[t-1]`` 重建一条乘法复权收盘价，当日高低价按与收盘价的真实价差折算。
    """
    close, closew = bars['close'], bars['closew']
    prev = close.shift(1)
    r = (closew.diff() / prev.where(prev > 0)).fillna(0.0).where(closew.notna())
    px = (1.0 + r).cumprod()
    base = close.where(close > 0)
    return {
        'close': px,
        'high': px * (1.0 + (bars['highw'] - closew) / base),
        'low': px * (1.0 + (bars['loww'] - closew) / base),
    }


def load_minutes(symbol: str, *,
                 include_validation: bool,
                 oos_end: dt.date | None = None,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """研究期 + （可选）2022 验证期 + （可选）已结束的样本外窗口，按时间拼成一张分钟表。

    研究期分片可缺（2022 年后上市的品种）；验证期分片只在不含样本外窗口时必须存在。
    样本外部分经 ``load_oos_shard``，窗口未结束会直接拒绝。
    """
    columns = MINUTE_COLUMNS if columns is None else columns
    parts = []
    if shard_io.find_shard(C.RESEARCH_DIR, symbol) is not None:
        research = shard_io.load_shard(symbol, directory=C.RESEARCH_DIR, columns=columns)
        C.assert_no_holdout_dates(research['trading_date'], what=f"{symbol} 研究期")
        parts.append(research)
    if include_validation or oos_end is not None:
        if shard_io.find_shard(C.VALIDATION_DIR, symbol) is not None:
            parts.append(shard_io.load_validation_shard(symbol, columns=columns))
        elif oos_end is None:
            raise FileNotFoundError(f"找不到 {symbol} 的 2022 验证分片")
    if oos_end is not None:
        parts.append(shard_io.load_oos_shard(symbol, end=oos_end, columns=columns))
    if not parts:
        raise FileNotFoundError(f"找不到 {symbol} 的任何分片")
    minute = pd.concat(parts).sort_index(kind='mergesort')
    if minute.index.has_duplicates:
        raise ValueError(f"{symbol} 各时间分区有重复时间戳")
    return minute
