"""Traditional controls: absolute time-series trend and cross-sectional momentum."""
import numpy as np
import pandas as pd

TSMOM_WINDOWS = (20, 60, 120, 250)
VOL_WINDOW = 60
CS_MIN_SYMBOLS = 5
# 火富牛《2022年度期货策略复盘与展望》里写明的数字。
# 3 日、两端各 5 个是截面动量的例子；20 日是文中波动率图的窗口。
# 均价突破没有给窗口，沿用同一张 20 日图，不另做搜索。
REVIEW_MOM_WINDOW = 3
REVIEW_VOL_WINDOW = 20
REVIEW_TAIL = 5

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


def trailing_return(close: pd.DataFrame, closew: pd.DataFrame,
                    window: int = REVIEW_MOM_WINDOW) -> pd.DataFrame:
    """过去 ``window`` 个交易日的简单收益。分母用原始价，价差用复权价。"""
    base = close.shift(int(window))
    return (closew - closew.shift(int(window))) / base.where(base > 0)


def realized_vol(close: pd.DataFrame, closew: pd.DataFrame,
                 window: int = REVIEW_VOL_WINDOW) -> pd.DataFrame:
    """截至当日的收益年化波动，窗口默认 20 日。"""
    ret = daily_return(close, closew)
    return ret.rolling(int(window), min_periods=int(window)).std() * np.sqrt(252)


def vol_change_sign(close: pd.DataFrame, closew: pd.DataFrame,
                    window: int = REVIEW_VOL_WINDOW) -> pd.DataFrame:
    """自身波动相对 ``window`` 日前是抬升还是回落：抬升 +1，回落 −1。"""
    vol = realized_vol(close, closew, window)
    delta = vol - vol.shift(int(window))
    values = np.sign(delta.to_numpy(dtype="float64"))
    values[~np.isfinite(delta.to_numpy(dtype="float64"))] = np.nan
    return pd.DataFrame(values, index=close.index, columns=close.columns)


def ma_breakout(closew: pd.DataFrame, window: int = REVIEW_VOL_WINDOW) -> pd.DataFrame:
    """收盘相对自身均线：在均线上方 +1，下方 −1。

    用复权价，避免换月把均线打断。窗口含当日，收盘后才知道。
    """
    average = closew.rolling(int(window), min_periods=int(window)).mean()
    gap = closew - average
    values = np.sign(gap.to_numpy(dtype="float64"))
    values[~np.isfinite(gap.to_numpy(dtype="float64"))] = np.nan
    return pd.DataFrame(values, index=closew.index, columns=closew.columns)


def cross_sectional_tails(factor: pd.DataFrame,
                          universe: dict[int, list[str]],
                          n: int = REVIEW_TAIL,
                          min_symbols: int | None = None) -> pd.DataFrame:
    """截面两端：最大的 ``n`` 个为 +1，最小的 ``n`` 个为 −1，中间为 0。

    只在当年事前品种池里排序。有效品种少于 ``2n`` 的日期整行留空。
    池外为 NaN。两端独立用最小排名，边界并列都计入该端；同时落入两端时留现金。
    """
    n = int(n)
    need = 2 * n if min_symbols is None else int(min_symbols)
    out = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    years = pd.DatetimeIndex(factor.index).year
    for year in sorted(set(years)):
        columns = [s for s in universe.get(int(year), []) if s in factor.columns]
        if len(columns) < need:
            continue
        dates = factor.index[years == year]
        block = factor.loc[dates, columns]
        count = block.notna().sum(axis=1)
        valid = count >= need
        short = block.rank(axis=1, method="min", ascending=True).le(n) & block.notna()
        long = block.rank(axis=1, method="min", ascending=False).le(n) & block.notna()
        both = long & short
        side = np.zeros(block.shape)
        side[short.to_numpy()] = -1.0
        side[long.to_numpy()] = 1.0
        side[both.to_numpy()] = 0.0
        side[~valid.to_numpy()] = np.nan
        out.loc[dates, columns] = side
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
