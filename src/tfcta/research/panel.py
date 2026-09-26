"""研究用的日频序列：收益、滚动 MAD、滚动分位信号。

三件事共用同一张「日期 × 品种」宽表，而且口径互相咬合：
因子在 t 日收盘才知道，赚取的是 day_ret[t+1]，仓位是信号的 shift(1)。
"""
from __future__ import annotations

import pandas as pd

from .. import config as C
from ..data import shard_io


def daily_prices_from_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """单品种分钟表 → 每个 trading_date 一根开盘价。"""
    missing = [c for c in ('open', 'openw', 'trading_date') if c not in df.columns]
    if missing:
        raise KeyError(
            f"分钟分片缺少 {missing}，无法按第 8.3 节计算 day_ret。\n"
            "请用含 open/openw 的原始面板重跑 step1_shard_minutes.py，"
            "不要用 close 代替 open。"
        )
    td = pd.to_datetime(df['trading_date']).dt.normalize()
    # 分钟表按时间升序，groupby.first 就是该交易日第一根 bar
    g = df.groupby(td, sort=True)
    out = pd.DataFrame({
        'open': g['open'].first(),
        'openw': g['openw'].first(),
    })
    out.index.name = 'trading_date'
    return out


def day_return_from_prices(prices: pd.DataFrame) -> pd.Series:
    """day_ret[t] = (openw[t+1] - openw[t]) / open[t]。末日没有 t+1，为 NaN。"""
    open_ = prices['open'].astype('float64')
    openw = prices['openw'].astype('float64')
    ret = (openw.shift(-1) - openw) / open_.where(open_ != 0)
    ret.name = 'day_ret'
    return ret


def load_day_returns(symbols: list[str],
                     directory=None) -> pd.DataFrame:
    """宽表 ``index=trading_date, columns=symbol``。

    读取走 shard_io.load_shard，因此指向 holdout_locked/ 会直接被拒绝。
    """
    cols = {}
    for s in symbols:
        df = shard_io.load_shard(s, directory=directory, columns=['open', 'openw', 'trading_date'])
        px = daily_prices_from_minutes(df)
        C.assert_no_holdout_dates(px.index, what=f"{s} 日频开盘价")
        cols[s] = day_return_from_prices(px)
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols).sort_index()
    out.index.name = 'trading_date'
    return out


DAILY_FIELDS = ('open', 'openw', 'close', 'closew', 'highw', 'loww', 'volume')


def load_daily_bars(symbols: list[str], directory=None) -> dict[str, pd.DataFrame]:
    """每个字段一张 ``trading_date × 品种`` 宽表。开盘取首根，收盘取末根，高低取极值，量求和。"""
    agg = {'open': 'first', 'openw': 'first', 'close': 'last', 'closew': 'last',
           'highw': 'max', 'loww': 'min', 'volume': 'sum'}
    cols: dict[str, dict[str, pd.Series]] = {k: {} for k in DAILY_FIELDS}
    for s in symbols:
        df = shard_io.load_shard(s, directory=directory,
                                 columns=list(DAILY_FIELDS) + ['trading_date'])
        td = pd.to_datetime(df['trading_date']).dt.normalize()
        day = df.groupby(td, sort=True).agg(agg)
        C.assert_no_holdout_dates(day.index, what=f"{s} 日频行情")
        for k in DAILY_FIELDS:
            cols[k][s] = day[k]
    return {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}


def multiplicative_prices(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """加法复权 → 乘法复权的收盘、最高、最低价。

    分片的 ``closew`` 是加法复权，价差对、比例不对：远期价位被整体平移，
    ``pct_change`` 和"相对均线偏离"这类比例指标会失真。这里用 ``Δclosew / close[t-1]``
    重建一条乘法复权收盘价，当日高低价按与收盘价的真实价差折算。
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


def forward_return(day_ret: pd.DataFrame) -> pd.DataFrame:
    """factor[t] 所预测的那段收益：day_ret[t+1]。

    day_ret[t+1] = (openw[t+2] - openw[t+1]) / open[t+1]，
    即 t 日收盘形成信号、t+1 开盘成交、持有到 t+2 开盘。
    """
    return day_ret.shift(-1)


def execute_position(signal: pd.DataFrame) -> pd.DataFrame:
    """收盘信号 → 下一交易日开盘才持有的仓位。"""
    return signal.shift(1)
