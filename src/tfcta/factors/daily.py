"""日频因子：传统量价指标与慢信号。输入都是 ``trading_date × 品种`` 宽表，只用到当日收盘。

价格口径
--------
* 量价指标（广发《125 个经典技术指标》五类构造，各留一到两个代表）接收
  ``data.bars.multiplicative_prices`` 重建的乘法复权价，比例指标因此不失真。
* 慢信号直接接收原始 ``close`` 与加法复权 ``closew``，收益一律写成 ``Δclosew / close``。
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

TSMOM_WINDOWS = (20, 60, 120, 250)
VOL_WINDOW = 60
CARRY_WINDOW = 250
CS_MIN_SYMBOLS = 5


# --------------------------------------------------------------------------
# 量价指标（乘法复权价）
# --------------------------------------------------------------------------
def _ema(price: pd.DataFrame, span: int) -> pd.DataFrame:
    return price.ewm(span=int(span), min_periods=int(span), adjust=False).mean()


def efficiency_ratio(close: pd.DataFrame, n: int = ER_N) -> pd.DataFrame:
    """ER：N 日净位移除以路径长度。趋势越干净，值越接近 1。"""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return net / path.where(path > 0)


def signed_efficiency_ratio(close: pd.DataFrame, n: int = ER_N) -> pd.DataFrame:
    """带方向的 ER，取值 -1 到 1。"""
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
    return (close - lo) / (hi - lo).where(hi > lo)


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
    obv = (np.sign(close.diff()) * volume).cumsum()
    base = volume.rolling(n, min_periods=n).sum()
    return (obv - obv.shift(n)) / base.where(base > 0)


def pvt_change(close: pd.DataFrame, volume: pd.DataFrame, n: int = PV_N) -> pd.DataFrame:
    """PVT 的 N 日变化。收益率乘成交量后再累加，同样用成交量归一。"""
    pvt = (close.pct_change(fill_method=None) * volume).cumsum()
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


def tech_indicators(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                    volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """五类代表因子的原始值。价格必须是乘法复权价。"""
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


# --------------------------------------------------------------------------
# 慢信号（原始 close + 加法复权 closew）
# --------------------------------------------------------------------------
def daily_return(close: pd.DataFrame, closew: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    return closew.diff() / prev.where(prev > 0)


def daily_vol(close: pd.DataFrame, closew: pd.DataFrame,
              window: int = VOL_WINDOW) -> pd.DataFrame:
    """截至当日的日收益波动。"""
    return daily_return(close, closew).rolling(window, min_periods=window // 2).std()


def tsmom(close: pd.DataFrame, closew: pd.DataFrame,
          windows=TSMOM_WINDOWS) -> pd.DataFrame:
    """多窗口时序动量，取值 -1 到 1。

    每个窗口的收益除以 ``σ·sqrt(w)`` 变成 t 统计量样的尺度，截到 ±2 再除以 2，
    几个窗口等权平均，任何一个窗口缺失就留空。
    """
    vol = daily_vol(close, closew)
    parts = []
    for w in windows:
        base = close.shift(w)
        r = (closew - closew.shift(w)) / base.where(base > 0)
        parts.append((r / (vol * np.sqrt(w))).clip(-2, 2) / 2)
    stacked = np.stack([p.to_numpy(dtype='float64') for p in parts])
    ok = np.isfinite(stacked)
    cnt = ok.sum(axis=0)
    acc = np.where(ok, stacked, 0.0).sum(axis=0)
    out = np.where(cnt == len(parts), acc / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def momentum_components(close: pd.DataFrame, closew: pd.DataFrame,
                        windows=TSMOM_WINDOWS,
                        vol_window: int = VOL_WINDOW) -> dict[str, pd.DataFrame]:
    """各窗口累计对数动量及其波动率缩放值。"""
    ret = daily_return(close, closew)
    log_return = np.log1p(ret.where(ret > -1))
    vol = ret.rolling(vol_window, min_periods=max(2, vol_window // 2)).std()
    out = {}
    for window in windows:
        w = int(window)
        total = log_return.rolling(w, min_periods=w).sum()
        out[f'tsmom_{w}'] = total
        out[f'tsmom_ra_{w}'] = (total / (vol * np.sqrt(w))).clip(-3, 3) / 3
    return out


def cross_sectional_rank(factor: pd.DataFrame,
                         universe: dict[int, list[str]],
                         min_symbols: int = CS_MIN_SYMBOLS) -> pd.DataFrame:
    """按每年事前确定的品种池做截面百分位排名（中心化到 ±0.5），池外与样本不足日期留 NaN。"""
    out = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    years = pd.DatetimeIndex(factor.index).year
    for year in sorted(set(years)):
        columns = [s for s in universe.get(int(year), []) if s in factor.columns]
        if not columns:
            continue
        dates = factor.index[years == year]
        block = factor.loc[dates, columns]
        count = block.notna().sum(axis=1)
        valid = count >= int(min_symbols)
        ranked = block.rank(axis=1, method='average').sub(0.5).div(
            count.replace(0, np.nan), axis=0) - 0.5
        out.loc[dates[valid], columns] = ranked.loc[valid]
    return out


def cross_sectional_momentum(close: pd.DataFrame, closew: pd.DataFrame,
                             universe: dict[int, list[str]],
                             windows=TSMOM_WINDOWS,
                             vol_window: int = VOL_WINDOW,
                             min_symbols: int = CS_MIN_SYMBOLS) -> dict[str, pd.DataFrame]:
    """截面动量与风险调整截面动量的中心化百分位排名。"""
    ranked = {}
    for name, factor in momentum_components(close, closew, windows, vol_window).items():
        rank_name = (name.replace('tsmom_ra_', 'cs_mom_ra_') if name.startswith('tsmom_ra_')
                     else name.replace('tsmom_', 'cs_mom_'))
        ranked[rank_name] = cross_sectional_rank(factor, universe, min_symbols)
    return ranked


def rolling_return_skewness(close: pd.DataFrame, closew: pd.DataFrame,
                            window: int = 60,
                            min_periods: int | None = None) -> pd.DataFrame:
    """截至当日的日收益偏度；不指定方向。"""
    minimum = max(3, int(window * 2 / 3)) if min_periods is None else int(min_periods)
    return daily_return(close, closew).rolling(int(window), min_periods=minimum).skew()


def cross_sectional_low_volatility(close: pd.DataFrame, closew: pd.DataFrame,
                                   universe: dict[int, list[str]],
                                   window: int = 60,
                                   min_symbols: int = CS_MIN_SYMBOLS) -> pd.DataFrame:
    """低已实现波动率的中心化截面秩。"""
    vol = daily_return(close, closew).rolling(
        int(window), min_periods=max(2, int(window) // 2)).std()
    return cross_sectional_rank(-vol, universe, min_symbols)


def roll_gap(close: pd.DataFrame, closew: pd.DataFrame) -> pd.DataFrame:
    """换月日的相对价差 ``(P_new - P_old) / P_new``，其余日为 0。

    加法复权下 ``close - closew`` 只在换月日变化，变化量就是新旧合约的价差。
    """
    gap = (close - closew).diff()
    gap = gap.where(gap.abs() > 1e-6, 0.0)
    return gap / close.where(close > 0)


def carry(close: pd.DataFrame, closew: pd.DataFrame,
          window: int = CARRY_WINDOW) -> pd.DataFrame:
    """过去一年已实现的展期收益：换月时新合约比旧合约便宜的幅度之和。

    正值 = 贴水结构（backwardation），多头持有能赚展期。这是事后实现的 carry，
    比用远月报价算的即期 carry 滞后，但只用主力连续数据就能得到。
    """
    return -roll_gap(close, closew).rolling(window, min_periods=window).sum()
