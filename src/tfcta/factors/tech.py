"""广发《125 个经典技术指标》里的五类构造，各留一到两个代表。

不复刻全部 125 个。时间戳和持续期因子已经覆盖日内极值时刻和均衡价偏离，
这里补的是日频上的趋势形态、区间位置、成交量和价量关系。
价格用复权价，窗口用到当日收盘，不含下一日。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ER_N = 20
PO_SHORT, PO_LONG = 12, 26
BIAS_N = 20
RSV_N = 20
RSI_N = 14
VOL_SHORT, VOL_LONG = 5, 20
OBV_N = 20
PV_N = 20
ATR_N = 20


def _ema(price: pd.DataFrame, span: int) -> pd.DataFrame:
    return price.ewm(span=int(span), min_periods=int(span), adjust=False).mean()


def efficiency_ratio(close: pd.DataFrame, n: int = ER_N) -> pd.DataFrame:
    """ER：N 日净位移除以路径长度。趋势越干净，值越接近 1。"""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return net / path.where(path > 0)


def signed_efficiency_ratio(close: pd.DataFrame, n: int = ER_N) -> pd.DataFrame:
    """带方向的 ER：N 日净位移除以路径长度，保留涨跌号，取值 -1 到 1。"""
    net = close - close.shift(n)
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return net / path.where(path > 0)


def price_oscillator(close: pd.DataFrame,
                     short: int = PO_SHORT, long: int = PO_LONG) -> pd.DataFrame:
    """PO：短均线相对长均线的偏离。"""
    slow = _ema(close, long)
    return (_ema(close, short) - slow) / slow.where(slow != 0)


def bias(close: pd.DataFrame, n: int = BIAS_N) -> pd.DataFrame:
    """BIAS：收盘价相对 N 日均线的乖离。"""
    ma = close.rolling(n, min_periods=n).mean()
    return (close - ma) / ma.where(ma != 0)


def range_position(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                   n: int = RSV_N) -> pd.DataFrame:
    """RSV：收盘价在过去 N 日最高、最低之间的位置，取值 0 到 1。"""
    lo = low.rolling(n, min_periods=n).min()
    hi = high.rolling(n, min_periods=n).max()
    span = (hi - lo).where(hi > lo)
    return (close - lo) / span


def rsi(close: pd.DataFrame, n: int = RSI_N) -> pd.DataFrame:
    """RSI：上涨幅度占涨跌绝对幅度的比例，取值 0 到 1。"""
    diff = close.diff()
    up = diff.clip(lower=0).rolling(n, min_periods=n).mean()
    down = (-diff.clip(upper=0)).rolling(n, min_periods=n).mean()
    total = up + down
    return up / total.where(total > 0)


def volume_ratio(volume: pd.DataFrame,
                 short: int = VOL_SHORT, long: int = VOL_LONG) -> pd.DataFrame:
    """短均量相对长均量。大于 1 表示近期放量。"""
    base = volume.rolling(long, min_periods=long).mean()
    return volume.rolling(short, min_periods=short).mean() / base.where(base > 0)


def obv_change(close: pd.DataFrame, volume: pd.DataFrame, n: int = OBV_N) -> pd.DataFrame:
    """OBV 的 N 日变化，除以同期成交量，去掉品种间的量纲。"""
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    base = volume.rolling(n, min_periods=n).sum()
    return (obv - obv.shift(n)) / base.where(base > 0)


def pvt_change(close: pd.DataFrame, volume: pd.DataFrame, n: int = PV_N) -> pd.DataFrame:
    """PVT 的 N 日变化。收益率乘成交量后再累加，同样用成交量归一。"""
    ret = close.pct_change(fill_method=None)
    pvt = (ret * volume).cumsum()
    base = volume.rolling(n, min_periods=n).sum()
    return (pvt - pvt.shift(n)) / base.where(base > 0)


def price_volume_corr(close: pd.DataFrame, volume: pd.DataFrame,
                      n: int = PV_N) -> pd.DataFrame:
    """N 日收益与成交量变化的相关系数。"""
    return close.pct_change(fill_method=None).rolling(n, min_periods=n).corr(
        volume.pct_change(fill_method=None))


def atr_pct(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
            n: int = ATR_N) -> pd.DataFrame:
    """ATR 除以收盘价。真实波幅取高低价和相对昨收的缺口。"""
    prev = close.shift(1)
    a = (high - low).to_numpy(dtype='float64')
    b = (high - prev).abs().to_numpy(dtype='float64')
    c = (low - prev).abs().to_numpy(dtype='float64')
    tr = pd.DataFrame(np.fmax(np.fmax(a, b), c), index=close.index, columns=close.columns)
    return tr.rolling(n, min_periods=n).mean() / close.where(close != 0)


def build(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
          volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """五类代表因子。值是日期×品种的原始因子。"""
    return {
        'er': efficiency_ratio(close),
        'er_signed': signed_efficiency_ratio(close),
        'po': price_oscillator(close),
        'bias': bias(close),
        'rsv': range_position(close, high, low),
        'rsi': rsi(close),
        'vol_ratio': volume_ratio(volume),
        'obv': obv_change(close, volume),
        'pvt': pvt_change(close, volume),
        'pv_corr': price_volume_corr(close, volume),
        'atr_pct': atr_pct(close, high, low),
    }
