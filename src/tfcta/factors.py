"""分钟因子、日频因子、外部数据和因子装配。"""
from __future__ import annotations

import json
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tfcta import config as C
from tfcta.data import bars as B
from tfcta.data import sessions, shard_io
from tfcta.data import universe as U


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
    """过去 ``window`` 个交易日收益的符号：涨为 +1，跌为 −1。"""
    w = int(window)
    base = close.shift(w)
    r = (closew - closew.shift(w)) / base.where(base > 0)
    values = np.sign(r.to_numpy(dtype='float64'))
    values[~np.isfinite(r.to_numpy(dtype='float64'))] = np.nan
    return pd.DataFrame(values, index=close.index, columns=close.columns)


def tsmom(close: pd.DataFrame, closew: pd.DataFrame,
          windows=TSMOM_WINDOWS) -> pd.DataFrame:
    """多窗口时序动量，取值 -1 到 1。"""
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
    """过去 ``window`` 个交易日的简单收益。"""
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
    """收盘相对自身均线：在均线上方 +1，下方 −1。"""
    average = closew.rolling(int(window), min_periods=int(window)).mean()
    gap = closew - average
    values = np.sign(gap.to_numpy(dtype="float64"))
    values[~np.isfinite(gap.to_numpy(dtype="float64"))] = np.nan
    return pd.DataFrame(values, index=closew.index, columns=closew.columns)


def cross_sectional_tails(factor: pd.DataFrame,
                          universe: dict[int, list[str]],
                          n: int = REVIEW_TAIL,
                          min_symbols: int | None = None) -> pd.DataFrame:
    """截面两端：最大的 ``n`` 个为 +1，最小的 ``n`` 个为 −1，中间为 0。"""
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
    """过去 ``lookback`` 日全部分钟变化绝对值的第 ``pct`` 分位数，以日编码为 index。"""
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
    """一次算出 ``lookbacks × pcts`` 全部阈值序列。"""
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
    """单交易日的持续期序列（向量化，已用暴力双循环交叉验证）。"""
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
    """整段序列的持续期，逐交易日独立计算。"""
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
    """公允均衡价格偏离。"""
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
    """用数据集里的原始报价，不用 float32 复权价算持续期。"""
    values = df['close'].to_numpy()
    clean = values.astype('float64')
    is_stored_float32 = clean == clean.astype('float32').astype('float64')
    clean = np.where(is_stored_float32, np.round(clean, 2), clean)
    return clean


def duration_factors(df: pd.DataFrame,
                     lookback: int,
                     pct: float,
                     thr_p: pd.Series | None = None) -> pd.DataFrame:
    """一个 (lookback, pct) 下的 dfp_max 与 dfp_top3。"""
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
    """报告中其余时序因子。"""
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


EXTERNAL_DATA_ROOT = C.DATA_ROOT / "external_rqdata"
EXTERNAL_FACTOR_ROOT = C.FACTOR_DAILY_DIR / "external"


def guard_dates(dates,partition):
    if partition == "research":
        C.assert_no_holdout_dates(dates,"carry缓存")
    elif partition == "validation_2022":
        C.assert_validation_2022_dates(dates,"carry缓存")
    elif partition == "oos":
        end = C.final_evaluation_end()  # 未打开最终评估时拒绝
        C.assert_strict_oos_dates(dates,"carry缓存")
        if len(dates) and pd.DatetimeIndex(dates).max() > pd.Timestamp(end):
            raise C.HoldoutViolation(f"carry缓存越过样本外截止日 {end}")
    else:
        raise C.HoldoutViolation(f"未知 carry 分区 {partition}")

def _date_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        for name in ("date", "trading_date"):
            if name in index.names:
                return pd.DatetimeIndex(pd.to_datetime(index.get_level_values(name)).normalize())
    return pd.DatetimeIndex(pd.to_datetime(index).normalize())


def _series(data, column: str | None = None, numeric: bool = True) -> pd.Series:
    if data is None:
        return pd.Series(dtype="float64")
    if isinstance(data, pd.Series):
        values = data
    elif isinstance(data, pd.DataFrame) and column in data.columns:
        values = data[column]
    else:
        return pd.Series(dtype="float64")
    result = pd.Series(values.to_numpy(), index=_date_index(data.index))
    result = result[~result.index.isna()]
    result = result[~result.index.duplicated(keep="last")].sort_index()
    if numeric:
        result = pd.to_numeric(result, errors="coerce")
    return result


def align_asof(source: pd.Series, calendar: pd.DatetimeIndex,
               lag: int = 1, fill_limit: int = 5) -> pd.Series:
    """只向前填充已经观测到的值，再整体滞后 ``lag`` 个交易日。"""
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().sort_values()
    source = source[~source.index.duplicated(keep="last")].sort_index()
    if source.empty:
        return pd.Series(np.nan, index=calendar, dtype="float64")
    return source.reindex(calendar, method="ffill", limit=fill_limit).shift(lag)


def build_partition(partition="research", symbols=None):
    if partition not in ("research","validation_2022","oos"):
        raise C.HoldoutViolation(f"未知 carry 分区 {partition}")
    # (源目录, 该目录数据所属分区)；样本外接在研究期和 2022 后面，滞后对齐才能跨年衔接
    roots = [(EXTERNAL_DATA_ROOT, "research")]
    if partition in ("validation_2022", "oos"):
        roots.append((EXTERNAL_DATA_ROOT / "validation_2022", "validation_2022"))
    if partition == "oos":
        end = pd.Timestamp(C.final_evaluation_end())
        roots.append((EXTERNAL_DATA_ROOT / "holdout_locked", "oos"))
    summary = {}
    for symbol in symbols or shard_io.list_shards(C.RESEARCH_DIR):
        pieces = []
        for root, part in roots:
            manifest = root / "coverage.json"
            if not manifest.exists():
                continue
            entry = json.loads(manifest.read_text()).get("items",{}).get(f"roll_yield/{symbol}_main_sub")
            if not entry or not entry.get("coverage"):
                continue
            path = (root / entry["file"]).resolve()
            if root.resolve() not in path.parents:
                raise ValueError("carry源路径越界")
            series = _series(pd.read_pickle(path),"annualized_yield")
            if part == "oos":
                series = series[(series.index >= pd.Timestamp(C.STRICT_OOS_START)) & (series.index <= end)]
            guard_dates(series.index, part)
            pieces.append(series)
        if not pieces:
            continue
        minutes = []
        if shard_io.find_shard(C.RESEARCH_DIR, symbol) is not None:
            minutes.append(shard_io.load_shard(symbol,columns=["trading_date"]))
        if partition in ("validation_2022", "oos") and shard_io.find_shard(C.VALIDATION_DIR, symbol) is not None:
            minutes.append(shard_io.load_validation_shard(symbol,columns=["trading_date"]))
        if partition == "oos" and shard_io.find_shard(C.HOLDOUT_DIR, symbol) is not None:
            minutes.append(shard_io.load_oos_shard(symbol, end=end.date(), columns=["trading_date"]))
        if not minutes:
            continue
        minute = pd.concat(minutes)
        calendar = pd.DatetimeIndex(pd.to_datetime(minute.trading_date).unique()).sort_values()
        aligned = align_asof(pd.concat(pieces),calendar)
        if partition == "research":
            keep = calendar.year<=2021
        elif partition == "validation_2022":
            keep = calendar.year==2022
        else:
            keep = (calendar >= pd.Timestamp(C.STRICT_OOS_START)) & (calendar <= end)
        frame = pd.DataFrame({"carry_main_sub_annualized":aligned.loc[keep]})
        guard_dates(frame.index,partition)
        if frame.empty:
            continue
        path = shard_io.save_shard(frame,EXTERNAL_FACTOR_ROOT/partition,symbol)
        summary[symbol] = {"file":str(path),"lag_days":1,"fill_limit":5}
    return summary


def load_panel(symbols=None, partition="research", root=None):
    if partition not in ("research","validation_2022","oos"):
        raise C.HoldoutViolation(f"未知 carry 分区 {partition}")
    if partition == "oos":
        C.assert_oos_research_locked()
    directory = Path(root or EXTERNAL_FACTOR_ROOT)/partition
    values = {}
    for symbol in symbols or shard_io.list_shards(directory):
        path = shard_io.find_shard(directory,symbol)
        if path is None:
            continue
        if partition == 'research':
            C.assert_research_only(path)
        frame = shard_io.read_frame(path,["carry_main_sub_annualized"])
        guard_dates(frame.index,partition)
        if frame.index.has_duplicates:
            raise ValueError(f"{path} carry日期重复")
        if "carry_main_sub_annualized" in frame:
            values[symbol] = frame["carry_main_sub_annualized"]
    return {"carry_main_sub_annualized":pd.DataFrame(values).sort_index()} if values else {}


def load_wide(symbols,partitions,index):
    if isinstance(partitions,str):
        partitions = (partitions,)
    frames = [load_panel(symbols,p).get("carry_main_sub_annualized") for p in partitions]
    frames = [f for f in frames if f is not None]
    if not frames:
        return {}
    joined = pd.concat(frames).sort_index()
    if joined.index.has_duplicates:
        raise ValueError("carry分区重叠")
    return {"carry_main_sub_annualized":joined.reindex(index=index,columns=symbols)}


intraday = types.SimpleNamespace(intraday_abs_diff=intraday_abs_diff, rolling_threshold=rolling_threshold, rolling_threshold_grid=rolling_threshold_grid, duration_one_day=duration_one_day, duration_series=duration_series, day_codes_of=day_codes_of, _day_spans=_day_spans, dfp_name=dfp_name, dfp_factors=dfp_factors, raw_price_path=raw_price_path, duration_factors=duration_factors, _extreme_timepoint=_extreme_timepoint, timestamp_factors=timestamp_factors, REPORT_COLUMNS=REPORT_COLUMNS, _window_clock=_window_clock, _argmax_time=_argmax_time, report_factors=report_factors)

TIMESTAMP_DIR_NAME = 'timestamp'
REPORT_DIR_NAME = 'report'
# v3: raw decimal-price durations, positive threshold, full lookback warm-up;
# each individual file is versioned, so a partial rebuild cannot bless old files.
CACHE_VERSION = 3
TIMEPOINT_FACTORS = list(C.TIMESTAMP_FACTORS)
DURATION_FACTORS = [intraday.dfp_name(n) for n in C.FP_TOP_NS]


def combo_grid(lookbacks: list[int] | None = None,
               pcts: list[float] | None = None) -> list[tuple[int, float]]:
    lbs = lookbacks if lookbacks is not None else C.THRESHOLD_LOOKBACKS
    ps = pcts if pcts is not None else C.THRESHOLD_PCTS
    return [(int(n), float(m)) for n in lbs for m in ps]


def combo_name(lookback: int, pct: float) -> str:
    """组合目录名。"""
    return f"N{int(lookback)}_M{pct:g}"


def combo_dir(lookback: int, pct: float, root: Path | None = None) -> Path:
    return (root or C.FACTOR_DAILY_DIR) / combo_name(lookback, pct)


def timestamp_dir(root: Path | None = None) -> Path:
    return (root or C.FACTOR_DAILY_DIR) / TIMESTAMP_DIR_NAME


def report_dir(root: Path | None = None) -> Path:
    return (root or C.FACTOR_DAILY_DIR) / REPORT_DIR_NAME


# --------------------------------------------------------------------------
# 版本
# --------------------------------------------------------------------------
def check_version(path: Path) -> None:
    """每个已有因子文件都要带当前版本的旁路说明。"""
    path = Path(path)
    metadata = path.with_suffix(path.suffix + '.json')
    if path.exists() and (not metadata.exists() or
        json.loads(metadata.read_text()).get('version') != CACHE_VERSION):
        raise RuntimeError(f'{path} 因子公式版本不符，请用 build_factors.py --overwrite 重建')


# --------------------------------------------------------------------------
# 计算
# --------------------------------------------------------------------------
def build_symbol(symbol: str,
                 combos: list[tuple[int, float]] | None = None,
                 root: Path | None = None,
                 overwrite: bool = False,
                 fmt: str = 'auto',
                 timestamp: bool = True) -> dict:
    """算一个品种的全部日频因子并落盘。"""
    combos = combos or combo_grid()
    root = root or C.FACTOR_DAILY_DIR
    lookbacks = sorted({n for n, _ in combos})
    pcts = sorted({m for _, m in combos})

    if not overwrite:
        directories = [combo_dir(n,m,root) for n,m in combos]
        if timestamp:
            directories.append(timestamp_dir(root))
        directories.append(report_dir(root))
        for directory in directories:
            path = shard_io.find_shard(directory,symbol)
            if path is not None:
                check_version(path)
    need_ts = timestamp and (overwrite or shard_io.find_shard(timestamp_dir(root), symbol) is None)
    need_report = overwrite or shard_io.find_shard(report_dir(root), symbol) is None
    todo = [(n, m) for n, m in combos
            if overwrite or shard_io.find_shard(combo_dir(n, m, root), symbol) is None]
    if not need_ts and not todo and not need_report:
        return {'symbol': symbol, 'skipped': True, 'combos_written': 0,
                'timestamp_written': False, 'report_written': False}

    df = sessions.add_intraday_coords(shard_io.load_shard(symbol, columns=C.FACTOR_FIELDS))
    codes, days = intraday.day_codes_of(df)
    info: dict = {'symbol': symbol, 'skipped': False, 'n_days': int(len(days)),
                  'n_bars': int(len(df)),
                  'first_day': str(days[0].date()) if len(days) else '',
                  'last_day': str(days[-1].date()) if len(days) else ''}

    if need_ts:
        ts = intraday.timestamp_factors(df)
        path = shard_io.save_shard(ts, timestamp_dir(root), symbol, fmt)
        path.with_suffix(path.suffix + '.json').write_text(json.dumps({'version': CACHE_VERSION}))
        info['n_timestamp_cols'] = int(ts.shape[1])
    info['timestamp_written'] = bool(need_ts)

    if todo:
        price = intraday.raw_price_path(df)
        grid = intraday.rolling_threshold_grid(
            intraday.intraday_abs_diff(price, codes), codes, lookbacks, pcts)
        for n, m in todo:
            dur = intraday.duration_factors(df, lookback=n, pct=m, thr_p=grid[(n, m)])
            path = shard_io.save_shard(dur, combo_dir(n, m, root), symbol, fmt)
            path.with_suffix(path.suffix + '.json').write_text(json.dumps({'version': CACHE_VERSION}))
            info['n_duration_cols'] = int(dur.shape[1])
    info['combos_written'] = len(todo)
    info['report_written'] = False
    if need_report:
        # 与 DFP 使用同一组 (N, M)。多个组合时取调用方列出的第一组，避免同一目录被后一组覆盖。
        n0, m0 = combos[0]
        report = intraday.report_factors(df, lookback=n0, pct=m0)
        path = shard_io.save_shard(report, report_dir(root), symbol, fmt)
        path.with_suffix(path.suffix + '.json').write_text(json.dumps({'version': CACHE_VERSION}))
        info['report_written'] = True
        info['n_report_cols'] = int(report.shape[1])
    return info


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    """只取当前定义的因子列。"""
    C.assert_research_only(path)
    check_version(path)
    df = shard_io.read_frame(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{path} 缺少因子列 {missing}，请用 build_factors.py --overwrite 重建")
    df = df[columns]
    df.index = pd.to_datetime(df.index)
    C.assert_no_holdout_dates(df.index, what=str(path))
    return df


def load_symbol(symbol: str, lookback: int, pct: float,
                root: Path | None = None,
                with_timestamp: bool = True) -> pd.DataFrame:
    """单品种在某个 (N, M) 下的日频因子表（持续期族 + 时间戳族 outer 对齐）。"""
    root = root or C.FACTOR_DAILY_DIR
    p = shard_io.find_shard(combo_dir(lookback, pct, root), symbol)
    if p is None:
        raise FileNotFoundError(
            f"缺少 {combo_name(lookback, pct)}/{symbol}，请先运行 build_factors.py")
    out = _read(p, DURATION_FACTORS)
    if with_timestamp:
        q = shard_io.find_shard(timestamp_dir(root), symbol)
        if q is None:
            raise FileNotFoundError(f"缺少 timestamp/{symbol}")
        # outer：两族交易日理论上一致，一旦不一致能看见 NaN 而不是被静默截断
        out = out.join(_read(q, TIMEPOINT_FACTORS), how='outer')
    return out.sort_index()


def load_report(symbol: str, root: Path | None = None) -> pd.DataFrame:
    """报告其余时序因子。"""
    root = root or C.FACTOR_DAILY_DIR
    path = shard_io.find_shard(report_dir(root), symbol)
    if path is None:
        raise FileNotFoundError(f"缺少 report/{symbol}，请先运行 build_factors.py")
    return _read(path, list(intraday.REPORT_COLUMNS)).sort_index()


Z_WINDOW, Z_MIN = 252, 120
SIGNED_PRIORS = {**C.FACTOR_SIGNS, "time_spread":1., "tsmom":1.,
                 "tsmom_20":1., "tsmom_3":1., "ma_break_20":1.,
                 "carry_ms":1., "cs_mom_ra_250":1., "cs_mom_3":1.,
                 "vol_tail_20":1., "neg_clv":1.}
TIME_FACTORS = frozenset([*C.FACTOR_SIGNS,"time_spread"])


def exante_z(factor, window=None, min_periods=None):
    w = C.IC_Z_WINDOW if window is None else window
    m = C.IC_Z_MIN if min_periods is None else min_periods
    mean = factor.rolling(w,min_periods=m).mean()
    std = factor.rolling(w,min_periods=m).std()
    return ((factor-mean)/std.where(std>0)).clip(-3,3)


def trail_z(factor):
    return exante_z(factor,Z_WINDOW,Z_MIN)


def to_signal(name, raw, timing="z", direction=None):
    signed = raw * (SIGNED_PRIORS[name] if direction is None else direction)
    if name in TIME_FACTORS:
        if timing == "z":
            return trail_z(signed).clip(-1,1)
        if timing == "quantile":
            history = signed.shift(1).rolling(Z_WINDOW,min_periods=Z_MIN)
            lo, hi = history.quantile(.2), history.quantile(.8)
            out = signed*0.
            # Equal boundaries / tied flat data give no trade, not +1 bias.
            out = out.mask(signed>hi,1.).mask(signed<lo,-1.)
            return out.where(lo.notna() & hi.notna() & (hi>lo))
        raise ValueError("时间信号须用 z/quantile")
    if name == "carry_ms":
        rms = (signed.shift(1)**2).rolling(Z_WINDOW,min_periods=Z_MIN).mean()**.5
        return (signed/rms.where(rms>0)).clip(-1,1)  # do not demean carry
    if name == "cs_mom_ra_250":
        return (2*signed).clip(-1,1)  # rank is already standardized across symbols
    return signed.clip(-1,1)  # bounded trend / CLV, preserves economic sign


def combine(frames, method="mean", weights=None):
    if not frames:
        raise ValueError("至少选择一个因子")
    if method not in ("mean","agree","filter"):
        raise ValueError("组合方法须为 mean/agree/filter")
    values = np.stack([f.reindex_like(frames[0]).fillna(0.).to_numpy() for f in frames])
    w = np.ones(len(frames)) if weights is None else np.asarray(weights,dtype=float)
    if len(w)!=len(frames) or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError("组合权重须为有限正数；反向通过因子 direction 指定")
    mean = np.average(values,axis=0,weights=w)
    if method == "agree":
        mean = np.where(np.all(values>0,axis=0) | np.all(values<0,axis=0),mean,0.)
    if method == "filter":
        if len(frames)<2:
            raise ValueError("filter 需要主信号和至少一个过滤信号")
        control = np.average(values[1:],axis=0,weights=w[1:])
        mean = np.where((values[0]*control)>0,values[0],0.)
    return pd.DataFrame(mean,index=frames[0].index,columns=frames[0].columns).clip(-1,1)


@dataclass
class SignalSet:
    bars: dict
    signed: dict = field(default_factory=dict)
    unsigned: dict = field(default_factory=dict)
    family: dict = field(default_factory=dict)
    vol: pd.DataFrame | None = None

    def raw(self,name):
        if name in self.signed:
            return self.signed[name]/SIGNED_PRIORS[name]
        if name in self.unsigned:
            return self.unsigned[name]
        raise KeyError(f"未知因子 {name}，可用 {sorted(self.signed)}")


def assemble(bars,time_raw,universe,external_partitions=("research",)):
    result = SignalSet(bars=bars)
    for name,sign in C.FACTOR_SIGNS.items():
        result.signed[name] = time_raw[name]*sign
        result.family[name] = "时间戳" if name.startswith("ts_") else "持续期"
    result.signed["time_spread"] = time_raw["ts_low"]-time_raw["ts_high"]
    result.family["time_spread"] = "高低点时间差"
    close, adjusted = bars["close"], bars["closew"]
    result.signed["tsmom"] = daily.tsmom(close,adjusted)
    result.signed["tsmom_20"] = daily.tsmom_sign(close,adjusted)
    result.signed["tsmom_3"] = daily.tsmom_sign(close,adjusted,daily.REVIEW_MOM_WINDOW)
    result.signed["ma_break_20"] = daily.ma_breakout(adjusted,daily.REVIEW_VOL_WINDOW)
    source = external.load_wide(list(close.columns),external_partitions,close.index)
    if "carry_main_sub_annualized" in source:
        result.signed["carry_ms"] = source["carry_main_sub_annualized"]
        result.family["carry_ms"] = "期限结构对照"
    mom = daily.momentum_components(close,adjusted,windows=(250,))["tsmom_ra_250"]
    result.signed["cs_mom_ra_250"] = daily.cross_sectional_rank(mom,universe)
    result.family["cs_mom_ra_250"] = "截面动量对照"
    ret3 = daily.trailing_return(close,adjusted,daily.REVIEW_MOM_WINDOW)
    result.signed["cs_mom_3"] = daily.cross_sectional_tails(ret3,universe,daily.REVIEW_TAIL)
    vol20 = daily.realized_vol(close,adjusted,daily.REVIEW_VOL_WINDOW)
    result.unsigned["vol_20"] = vol20
    result.unsigned["vol_rise_20"] = daily.vol_change_sign(close,adjusted,daily.REVIEW_VOL_WINDOW)
    result.signed["vol_tail_20"] = daily.cross_sectional_tails(vol20,universe,daily.REVIEW_TAIL)
    high, low = bars["highw"],bars["loww"]
    result.signed["neg_clv"] = -(2*adjusted-high-low)/(high-low).where(high>low)
    result.family.update(
        tsmom="趋势对照", tsmom_20="趋势对照", tsmom_3="时序动量",
        ma_break_20="均价突破", cs_mom_3="截面动量", vol_tail_20="截面波动",
        neg_clv="价格位置对照")
    result.vol = daily.daily_vol(close,adjusted)
    return result


def load(symbols):
    panels = {s:cache.load_symbol(s,C.IC_REFERENCE_LOOKBACK,C.IC_REFERENCE_PCT) for s in symbols}
    for symbol, frame in panels.items():
        extra = cache.load_report(symbol)
        overlap = frame.index.intersection(extra.index)
        panels[symbol] = frame.join(extra, how='outer')
        if overlap.empty and len(frame) and len(extra):
            raise ValueError(f"{symbol} 的持续期缓存与报告因子缓存没有共同交易日")
    raw = {n:pd.DataFrame({s:f[n] for s,f in panels.items() if n in f.columns}) for n in C.FACTOR_SIGNS}
    missing = [n for n, frame in raw.items() if frame.empty]
    if missing:
        raise KeyError(f"因子缓存缺少 {missing}")
    return assemble(B.load_daily_bars(symbols),raw,U.load_universe())

daily = types.SimpleNamespace(TSMOM_WINDOWS=TSMOM_WINDOWS, VOL_WINDOW=VOL_WINDOW, CS_MIN_SYMBOLS=CS_MIN_SYMBOLS, REVIEW_MOM_WINDOW=REVIEW_MOM_WINDOW, REVIEW_VOL_WINDOW=REVIEW_VOL_WINDOW, REVIEW_TAIL=REVIEW_TAIL, daily_return=daily_return, daily_vol=daily_vol, tsmom_sign=tsmom_sign, tsmom=tsmom, momentum_components=momentum_components, trailing_return=trailing_return, realized_vol=realized_vol, vol_change_sign=vol_change_sign, ma_breakout=ma_breakout, cross_sectional_tails=cross_sectional_tails, cross_sectional_rank=cross_sectional_rank)
external = types.SimpleNamespace(EXTERNAL_DATA_ROOT=EXTERNAL_DATA_ROOT, EXTERNAL_FACTOR_ROOT=EXTERNAL_FACTOR_ROOT, guard_dates=guard_dates, _date_index=_date_index, _series=_series, align_asof=align_asof, build_partition=build_partition, load_panel=load_panel, load_wide=load_wide)
cache = types.SimpleNamespace(TIMESTAMP_DIR_NAME=TIMESTAMP_DIR_NAME, REPORT_DIR_NAME=REPORT_DIR_NAME, CACHE_VERSION=CACHE_VERSION, TIMEPOINT_FACTORS=TIMEPOINT_FACTORS, DURATION_FACTORS=DURATION_FACTORS, combo_grid=combo_grid, combo_name=combo_name, combo_dir=combo_dir, timestamp_dir=timestamp_dir, report_dir=report_dir, check_version=check_version, build_symbol=build_symbol, _read=_read, load_symbol=load_symbol, load_report=load_report)
library = types.SimpleNamespace(Z_WINDOW=Z_WINDOW, Z_MIN=Z_MIN, SIGNED_PRIORS=SIGNED_PRIORS, TIME_FACTORS=TIME_FACTORS, exante_z=exante_z, trail_z=trail_z, to_signal=to_signal, combine=combine, SignalSet=SignalSet, assemble=assemble, load=load)
