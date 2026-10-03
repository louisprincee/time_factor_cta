"""Traditional controls: absolute time-series trend and cross-sectional momentum."""
import numpy as np
import pandas as pd

TSMOM_WINDOWS = (20, 60, 120, 250)
VOL_WINDOW = 60
CS_MIN_SYMBOLS = 5

def daily_return(close: pd.DataFrame, closew: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    return closew.diff() / prev.where(prev > 0)


def daily_vol(close: pd.DataFrame, closew: pd.DataFrame,
              window: int = VOL_WINDOW) -> pd.DataFrame:
    """截至当日的日收益波动。"""
    return daily_return(close, closew).rolling(window, min_periods=window // 2).std()


def tsmom_sign(close: pd.DataFrame, closew: pd.DataFrame,
               window: int = 20) -> pd.DataFrame:
    """过去 ``window`` 个交易日收益的符号：涨为 +1，跌为 −1。

    Cho 等对中国商品时序动量的口径是大约一个月的形成期，方向本身就是信号；
    波动率缩放放到组合的波动率目标上，这里不再除一次波动。只用到 t 日收盘。
    """
    w = int(window)
    base = close.shift(w)
    r = (closew - closew.shift(w)) / base.where(base > 0)
    values = np.sign(r.to_numpy(dtype='float64'))
    values[~np.isfinite(r.to_numpy(dtype='float64'))] = np.nan
    return pd.DataFrame(values, index=close.index, columns=close.columns)


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
