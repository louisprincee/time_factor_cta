"""分钟级因子：持续期核心（设计文档第 6.2 节）与当前策略用到的四个日频时点因子。

持续期定义（论文原文）
----------------------
    Price Duration_j  = t_j - t_i,  t_j > t_i,  if |P_j - P_i| >= Threshold

逐交易日独立计算，不跨日。对当日时刻 i，向前遍历当日已发生的 j < i，取**最近的**满足
``|P_i - P_j| >= Threshold_t`` 的 j，``Duration_{t,i} = i - j``（bar 数）。当日此前没有
满足条件的 j 时，持续期**从开盘累计**，0-based 下即 ``i``，不是 0、不是 NaN。

阈值动态滚动：``Threshold_t`` = 过去 N 个交易日全部「日内相邻 bar 一阶差分绝对值」汇总后
的第 M 分位数，不含当日。逐品种各自计算。

日频因子
--------
* ``dfp_max`` / ``dfp_top3``：持续期前 N 大的分钟上的均衡价 FP 相对收盘价的偏离，依赖 (N, M)。
* ``ts_high`` / ``ts_low``：全日最高价、最低价出现的归一化时点，不依赖参数。

价格口径：持续期使用经过报价精度修复的原始 close。加法复权价的理论平移不变性
不能消除 float32 在阈值边界的误差，因此不以 closew 计算持续期。``dfp`` 是比例，分子分母都用原始 ``close``——加法复权价离
上市越远偏离越大，做分母会把量纲放大或缩小数倍，接近 0 时还会变号。

落盘的是原始值，方向只在 ``factors.library`` 装配时乘上。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..data import sessions


# --------------------------------------------------------------------------
# 持续期核心
# --------------------------------------------------------------------------
def intraday_abs_diff(values: np.ndarray, day_codes: np.ndarray) -> np.ndarray:
    """日内相邻 bar 的一阶差分绝对值；每日首根记为 NaN（不跨日）。"""
    v = np.asarray(values, dtype='float64')
    if not len(v):
        return v
    d = np.abs(np.diff(v, prepend=np.nan))
    same_day = np.empty(len(v), dtype=bool)
    same_day[0] = False
    same_day[1:] = day_codes[1:] == day_codes[:-1]
    d[~same_day] = np.nan
    return d


def rolling_threshold(abs_diff: np.ndarray,
                      day_codes: np.ndarray,
                      lookback: int,
                      pct: float) -> pd.Series:
    """过去 ``lookback`` 日全部分钟变化绝对值的第 ``pct`` 分位数，以日编码为 index。

    只用 t-lookback..t-1，不含当日。"严格要求 N 日预热；过去 N 日全部分钟观测的分位数"是把 N 天的分钟样本
    **汇总成一个池子**再取分位数，不是"每日分位数再平均"。这是参照实现，
    :func:`rolling_threshold_grid` 与它逐元素一致（有测试钉住）。
    """
    s = pd.Series(abs_diff)
    per_day = [g.dropna().to_numpy() for _, g in s.groupby(day_codes, sort=True)]
    days = np.array(sorted(pd.unique(day_codes)))

    out = np.full(len(days), np.nan)
    for k in range(lookback, len(days)):
        pool = [a for a in per_day[max(0, k - lookback):k] if a.size]
        if not pool:
            continue
        cat = np.concatenate(pool)
        if cat.size:
            out[k] = np.percentile(cat, pct)
    return pd.Series(out, index=days)


def rolling_threshold_grid(abs_diff: np.ndarray,
                           day_codes: np.ndarray,
                           lookbacks: list[int],
                           pcts: list[float]) -> dict[tuple[int, float], pd.Series]:
    """一次算出 ``lookbacks × pcts`` 全部阈值序列。切分 1 遍，每个 N 拼接 1 遍，
    ``np.percentile`` 一次给出全部分位数。"""
    s = pd.Series(abs_diff)
    per_day = [g.dropna().to_numpy() for _, g in s.groupby(day_codes, sort=True)]
    days = np.array(sorted(pd.unique(day_codes)))
    qs = list(pcts)

    out = {(lb, p): np.full(len(days), np.nan) for lb in lookbacks for p in qs}
    for lb in lookbacks:
        for k in range(lb, len(days)):
            pool = [a for a in per_day[max(0, k - lb):k] if a.size]
            if not pool:
                continue
            cat = np.concatenate(pool)
            if not cat.size:
                continue
            vals = np.percentile(cat, qs)
            for p, v in zip(qs, np.atleast_1d(vals)):
                out[(lb, p)][k] = v
    return {key: pd.Series(arr, index=days) for key, arr in out.items()}


def duration_one_day(values: np.ndarray, threshold: float) -> np.ndarray:
    """单交易日的持续期序列（向量化，已用暴力双循环交叉验证）。

    O(n^2)，n <= 555，峰值内存约 2.4MB。
    """
    v = np.asarray(values, dtype='float64')
    n = len(v)
    if n == 0:
        return np.zeros(0)
    if not np.isfinite(threshold) or threshold <= 0:
        return np.full(n, np.nan)

    D = np.abs(v[:, None] - v[None, :])
    # Decimal quotes such as 400.2-400.1 must qualify at an exact 0.1 boundary.
    ok = (D >= threshold - 1e-9) & (np.arange(n)[None, :] < np.arange(n)[:, None])
    has = ok.any(axis=1)
    last_j = np.where(has, n - 1 - ok[:, ::-1].argmax(axis=1), 0)
    dur = np.where(has, np.arange(n) - last_j, np.arange(n)).astype('float64')
    dur[~np.isfinite(v)] = np.nan
    return dur


def duration_series(values: np.ndarray,
                    day_codes: np.ndarray,
                    thresholds: pd.Series) -> np.ndarray:
    """整段序列的持续期，逐交易日独立计算。阈值为 NaN 的交易日（预热不足）整日 NaN，不回填。"""
    v = np.asarray(values, dtype='float64')
    out = np.full(len(v), np.nan)
    thr_map = thresholds.to_dict()

    order = np.argsort(day_codes, kind='mergesort')
    if not np.array_equal(order, np.arange(len(day_codes))):
        raise ValueError("day_codes 未按时间排序，持续期会算错；请先排序")

    for a, b in zip(*_day_spans(day_codes)):
        out[a:b] = duration_one_day(v[a:b], thr_map.get(day_codes[a], np.nan))
    return out


# --------------------------------------------------------------------------
# 日频因子
# --------------------------------------------------------------------------
def day_codes_of(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """把 trading_date 转成连续整数编码，并返回去重后的交易日索引。"""
    td = pd.to_datetime(df['trading_date'])
    codes, uniq = pd.factorize(td, sort=True)
    return codes, pd.DatetimeIndex(uniq)


def _day_spans(day_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.flatnonzero(np.r_[True, day_codes[1:] != day_codes[:-1], True])
    return bounds[:-1], bounds[1:]


def dfp_name(n: int) -> str:
    return 'dfp_max' if n == 1 else f'dfp_top{n}'


def dfp_factors(dur: np.ndarray,
                close: np.ndarray,
                day_codes: np.ndarray,
                days: pd.DatetimeIndex,
                top_ns: list[int] | None = None) -> pd.DataFrame:
    """公允均衡价格偏离。

        FP_t  = mean(close 于持续期前 N 大的那些分钟)
        DFP_t = (FP_t - Close_t) / Close_t

    ``close`` 必须是**原始**价，Close_t 取当日最后一根有效值。
    """
    top_ns = top_ns or C.FP_TOP_NS
    cols = {dfp_name(n): np.full(len(days), np.nan) for n in top_ns}
    for k, (a, b) in enumerate(zip(*_day_spans(day_codes))):
        d, p = dur[a:b], close[a:b]
        ok = np.isfinite(d) & np.isfinite(p)
        if not ok.any():
            continue
        d_ok, p_ok = d[ok], p[ok]
        close_t = p_ok[-1]
        if not close_t > 0:
            continue
        order = np.argsort(-d_ok, kind='mergesort')
        for n in top_ns:
            fp = float(p_ok[order[:min(n, d_ok.size)]].mean())
            cols[dfp_name(n)][k] = (fp - close_t) / close_t
    return pd.DataFrame(cols, index=days)


def raw_price_path(df: pd.DataFrame) -> np.ndarray:
    """Recover the dataset's decimal raw quotes, never duration from adjusted float32.

    Historical commodity shards store raw quotes with at most two decimal places
    as float32. Round those to 0.01 price precision (NOT a presumed exchange tick).
    Promoted float32 values are recognized too, so concatenating partitions
    with different dtypes cannot change old history. Other float64 inputs are
    kept intact. The cleaned path is used consistently for
    thresholds, duration and FP; converting float32 to float64 alone cannot repair it.
    """
    values = df['close'].to_numpy()
    clean = values.astype('float64')
    is_stored_float32 = clean == clean.astype('float32').astype('float64')
    clean = np.where(is_stored_float32, np.round(clean, 2), clean)
    return clean


def duration_factors(df: pd.DataFrame,
                     lookback: int,
                     pct: float,
                     thr_p: pd.Series | None = None) -> pd.DataFrame:
    """一个 (lookback, pct) 下的 dfp_max 与 dfp_top3。阈值、持续期、FP 与分母都用原始报价路径。"""
    for col in ('trading_date', 'gamma_norm', 'session', 'closew', 'close'):
        if col not in df.columns:
            raise KeyError(f"缺少 {col} 列，请先调用 sessions.add_intraday_coords")

    codes, days = day_codes_of(df)
    price = raw_price_path(df)
    if thr_p is None:
        thr_p = rolling_threshold(intraday_abs_diff(price, codes), codes, lookback, pct)
    dur = duration_series(price, codes, thr_p)
    out = dfp_factors(dur, price, codes, days)
    out.index.name = 'trading_date'
    return out


def _extreme_timepoint(values: np.ndarray, gnorm: np.ndarray, mode: str) -> float:
    ok = np.isfinite(values) & np.isfinite(gnorm)
    if not ok.any():
        return np.nan
    pos = np.flatnonzero(ok)
    v = values[pos]
    j = pos[np.argmax(v) if mode == 'max' else np.argmin(v)]
    return float(gnorm[j])


def timestamp_factors(df: pd.DataFrame) -> pd.DataFrame:
    """ts_high、ts_low：全日最高价、最低价首次出现的归一化时点。"""
    for col in ('trading_date', 'gamma_norm', 'highw', 'loww'):
        if col not in df.columns:
            raise KeyError(f"缺少 {col} 列，请先调用 sessions.add_intraday_coords")

    codes, days = day_codes_of(df)
    hi = df['highw'].to_numpy(dtype='float64')
    lo = df['loww'].to_numpy(dtype='float64')
    gnorm = df['gamma_norm'].to_numpy(dtype='float64')
    out = {k: np.full(len(days), np.nan) for k in ('ts_high', 'ts_low')}
    for k, (a, b) in enumerate(zip(*_day_spans(codes))):
        if not np.isfinite(hi[a:b]).any():
            continue
        out['ts_high'][k] = _extreme_timepoint(hi[a:b], gnorm[a:b], 'max')
        out['ts_low'][k] = _extreme_timepoint(lo[a:b], gnorm[a:b], 'min')

    res = pd.DataFrame(out, index=days)
    res.index.name = 'trading_date'
    return res


# 报告里除 DFP 与全日高低点之外、并给出次日方向的因子。原始值，方向在 library 里乘。
REPORT_COLUMNS = (
    'pmt', 'vmt', 'vd_ratio',
    'ts_volume', 'ts_turnover',
    'ts_high_pm', 'ts_low_pm',
    'spike_am', 'vol_pm',
)


def _window_clock(values: np.ndarray, mask: np.ndarray, mode: str) -> float:
    """窗口内极值首次出现的位置，0 为窗口第一根，1 为最后一根。"""
    pos = np.flatnonzero(mask)
    if pos.size < 2:
        return np.nan
    chosen = values[pos].astype('float64', copy=False)
    if not np.isfinite(chosen).any():
        return np.nan
    if mode == 'max':
        ranked = np.where(np.isfinite(chosen), chosen, -np.inf)
        rel = int(np.argmax(ranked))
    else:
        ranked = np.where(np.isfinite(chosen), chosen, np.inf)
        rel = int(np.argmin(ranked))
    return float(rel / (pos.size - 1))


def _argmax_time(values: np.ndarray, gnorm: np.ndarray) -> float:
    """有限值中最大值首次出现的全日归一化时点。"""
    return _extreme_timepoint(values, gnorm, 'max')


def report_factors(df: pd.DataFrame,
                   lookback: int,
                   pct: float,
                   price_threshold: pd.Series | None = None,
                   volume_threshold: pd.Series | None = None) -> pd.DataFrame:
    """报告中其余时序因子。早盘/午后只用日盘 AM、PM，夜盘不混进这两个时段。

    全日时点仍用含夜盘的 ``gamma_norm``。``vol_pm`` 仅在全日最大成交量落在
    日盘时取 0/1；落在夜盘为缺失，因为报告的早晚对比没有夜盘这一档。
    """
    needed = ('trading_date', 'gamma_norm', 'session', 'close', 'volume',
              'total_turnover', 'highw', 'loww')
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"缺少 {col} 列，请先调用 sessions.add_intraday_coords")
    codes, days = day_codes_of(df)
    price = raw_price_path(df)
    volume = df['volume'].to_numpy(dtype='float64')
    if price_threshold is None:
        price_threshold = rolling_threshold(
            intraday_abs_diff(price, codes), codes, lookback, pct)
    if volume_threshold is None:
        volume_threshold = rolling_threshold(
            intraday_abs_diff(volume, codes), codes, lookback, pct)
    price_dur = duration_series(price, codes, price_threshold)
    volume_dur = duration_series(volume, codes, volume_threshold)
    gnorm = df['gamma_norm'].to_numpy(dtype='float64')
    session = df['session'].to_numpy()
    high = df['highw'].to_numpy(dtype='float64')
    low = df['loww'].to_numpy(dtype='float64')
    turnover = df['total_turnover'].to_numpy(dtype='float64')
    out = {name: np.full(len(days), np.nan) for name in REPORT_COLUMNS}
    for k, (a, b) in enumerate(zip(*_day_spans(codes))):
        am = session[a:b] == C.SESSION_AM
        pm = session[a:b] == C.SESSION_PM
        out['pmt'][k] = _argmax_time(price_dur[a:b], gnorm[a:b])
        out['vmt'][k] = _argmax_time(volume_dur[a:b], gnorm[a:b])
        am_dur = volume_dur[a:b][am]
        pm_dur = volume_dur[a:b][pm]
        am_dur = am_dur[np.isfinite(am_dur)]
        pm_dur = pm_dur[np.isfinite(pm_dur)]
        if am_dur.size and pm_dur.size and float(pm_dur.mean()) > 0:
            out['vd_ratio'][k] = float(am_dur.mean() / pm_dur.mean())
        out['ts_volume'][k] = _argmax_time(volume[a:b], gnorm[a:b])
        out['ts_turnover'][k] = _argmax_time(turnover[a:b], gnorm[a:b])
        out['ts_high_pm'][k] = _window_clock(high[a:b], pm, 'max')
        out['ts_low_pm'][k] = _window_clock(low[a:b], pm, 'min')
        day_high = high[a:b]
        if np.isfinite(day_high).any():
            peak = np.nanmax(day_high)
            out['spike_am'][k] = float(np.sum(am & np.isfinite(day_high) & (day_high == peak)))
        if np.isfinite(volume[a:b]).any():
            where = int(np.argmax(np.where(np.isfinite(volume[a:b]), volume[a:b], -np.inf)))
            label = session[a + where]
            if label == C.SESSION_PM:
                out['vol_pm'][k] = 1.0
            elif label == C.SESSION_AM:
                out['vol_pm'][k] = 0.0
    res = pd.DataFrame(out, index=days)
    res.index.name = 'trading_date'
    return res


def symbol_daily_factors(minute_df: pd.DataFrame,
                         lookback: int,
                         pct: float,
                         with_coords: bool = False) -> pd.DataFrame:
    """单品种、单参数组合下的日频时间因子（原始值，未施加方向）。"""
    df = minute_df if with_coords else sessions.add_intraday_coords(minute_df)
    out = pd.concat([duration_factors(df, lookback=lookback, pct=pct),
                     timestamp_factors(df),
                     report_factors(df, lookback=lookback, pct=pct)], axis=1)
    out.index.name = 'trading_date'
    return out
