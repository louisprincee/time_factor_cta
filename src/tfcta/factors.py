"""持续期与时间戳因子、日频因子缓存和装配。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tfcta import config as C
from tfcta.data import add_intraday_coords, find_shard, load_daily_bars, load_shard, read_frame, save_shard


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
            raise KeyError(f"缺少 {col} 列，请先调用 add_intraday_coords")

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
    """全交易日（含夜盘）高低点首次出现的位置，按 gamma_norm 归一化到 0–1。"""
    for col in ('trading_date', 'highw', 'loww', 'gamma_norm'):
        if col not in df.columns:
            raise KeyError(f"缺少 {col} 列，请先调用 add_intraday_coords")

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
            raise KeyError(f"缺少 {col} 列，请先调用 add_intraday_coords")
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


# 国泰君安股指报告（2025-02-25）的固定阈值持续期统计和行为时间戳。股指只有日盘，位置按当日日盘分钟数归一化。
GTJA_COLUMNS = (
    'time_to_max_turnover', 'time_to_half_volume', 'afternoon_high_time', 'time_to_low',
    'Price_duration_vol', 'Price_duration_corr', 'Price_duration_adj_vol', 'Price_duration_max_min',
    'Price_volume_persistence_ratio',
    'Volume_duration_adj_vol', 'Volume_duration_slope', 'Volume_duration_extreme',
    'Volume_duration_open_close', 'Volume_duration_return',
)


def _first_position(values: np.ndarray, mode: str) -> float:
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return np.nan
    ranked = np.where(ok, values, -np.inf if mode == 'max' else np.inf)
    j = int(np.argmax(ranked) if mode == 'max' else np.argmin(ranked))
    return float(j / (len(values) - 1))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def gtja_index_factors(df: pd.DataFrame,
                       price_threshold: float = 3.0,
                       volume_threshold: float = 50.0) -> pd.DataFrame:
    """单品种股指分钟表 → 每个交易日的报告因子（收盘 t 已知）。

    持续期：分钟 j 往回找最近一个变化 ≥ 阈值的分钟 i，持续期 = j − i；找不到记为距当日首根的分钟数。
    价格用原始分钟收盘价，成交量用分钟成交量。Volume_duration_open_close 按报告文字取收盘前 30 分钟减开盘后 30 分钟。
    Volume_duration_return 用加法复权价之差除以前一日原始价，跨换月不跳。"""
    for col in ('trading_date', 'close', 'closew', 'highw', 'loww', 'volume', 'total_turnover'):
        if col not in df.columns:
            raise KeyError(f"缺少 {col} 列")
    frame = df.sort_index(kind='mergesort')
    codes, days = day_codes_of(frame)
    price = raw_price_path(frame)
    closew = frame['closew'].to_numpy(dtype='float64')
    volume = frame['volume'].to_numpy(dtype='float64')
    turnover = frame['total_turnover'].to_numpy(dtype='float64')
    high = frame['highw'].to_numpy(dtype='float64')
    low = frame['loww'].to_numpy(dtype='float64')
    clock = frame.index.hour * 100 + frame.index.minute
    afternoon = clock > 1300
    opening = (clock >= 931) & (clock <= 1000)
    closing = (clock >= 1431) & (clock <= 1500)
    out = {name: np.full(len(days), np.nan) for name in GTJA_COLUMNS}
    anchor_w = np.full(len(days), np.nan)
    anchor_raw = np.full(len(days), np.nan)
    for k, (a, b) in enumerate(zip(*_day_spans(codes))):
        p, v = price[a:b], volume[a:b]
        if b - a < 10 or not np.isfinite(p).all() or not np.isfinite(v).all():
            continue
        pd_ = duration_one_day(p, price_threshold)
        vd = duration_one_day(v, volume_threshold)
        steps = np.arange(b - a, dtype='float64')
        price_sd = float(np.std(p))
        out['time_to_max_turnover'][k] = _first_position(turnover[a:b], 'max')
        cum = np.cumsum(v)
        if cum[-1] > 0:
            out['time_to_half_volume'][k] = float(np.flatnonzero(cum >= 0.5 * cum[-1])[0] / (b - a - 1))
        pm = afternoon[a:b]
        if pm.sum() >= 2:
            out['afternoon_high_time'][k] = _first_position(high[a:b][pm], 'max')
        out['time_to_low'][k] = _first_position(low[a:b], 'min')
        out['Price_duration_vol'][k] = float(np.std(pd_))
        out['Price_duration_corr'][k] = _safe_corr(pd_, p)
        out['Price_duration_adj_vol'][k] = float(np.std(pd_) / price_sd) if price_sd > 0 else np.nan
        out['Price_duration_max_min'][k] = float(pd_.max() - pd_.min())
        if v.sum() > 0:
            out['Price_volume_persistence_ratio'][k] = float(v[pd_ >= np.quantile(pd_, 0.8)].sum() / v.sum())
        out['Volume_duration_adj_vol'][k] = float(np.std(vd) / price_sd) if price_sd > 0 else np.nan
        out['Volume_duration_slope'][k] = float(np.polyfit(steps, vd, 1)[0])
        out['Volume_duration_extreme'][k] = float(np.mean(vd > vd.mean() + 2 * vd.std()))
        o, c = opening[a:b], closing[a:b]
        if o.any() and c.any():
            out['Volume_duration_open_close'][k] = float(vd[c].mean() - vd[o].mean())
        j = int(np.argmax(vd))
        anchor_w[k], anchor_raw[k] = closew[a + j], p[j]
    res = pd.DataFrame(out, index=days)
    previous_w, previous_raw = pd.Series(anchor_w).shift(1).to_numpy(), pd.Series(anchor_raw).shift(1).to_numpy()
    res['Volume_duration_return'] = (anchor_w - previous_w) / np.where(previous_raw > 0, previous_raw, np.nan)
    res.index.name = 'trading_date'
    return res


TIMESTAMP_DIR_NAME = 'timestamp'
REPORT_DIR_NAME = 'report'
# v3: raw decimal-price durations, positive threshold, full lookback warm-up;
# each individual file is versioned, so a partial rebuild cannot bless old files.
CACHE_VERSION = 3
TIMEPOINT_FACTORS = list(C.TIMESTAMP_FACTORS)
DURATION_FACTORS = [dfp_name(n) for n in C.FP_TOP_NS]


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
        raise RuntimeError(f'{path} 因子公式版本不符，请用 python src/tfcta/prepare/factors.py --overwrite 重建')


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
            path = find_shard(directory,symbol)
            if path is not None:
                check_version(path)
    need_ts = timestamp and (overwrite or find_shard(timestamp_dir(root), symbol) is None)
    need_report = overwrite or find_shard(report_dir(root), symbol) is None
    todo = [(n, m) for n, m in combos
            if overwrite or find_shard(combo_dir(n, m, root), symbol) is None]
    if not need_ts and not todo and not need_report:
        return {'symbol': symbol, 'skipped': True, 'combos_written': 0,
                'timestamp_written': False, 'report_written': False}

    df = add_intraday_coords(load_shard(symbol, columns=C.FACTOR_FIELDS))
    codes, days = day_codes_of(df)
    info: dict = {'symbol': symbol, 'skipped': False, 'n_days': int(len(days)),
                  'n_bars': int(len(df)),
                  'first_day': str(days[0].date()) if len(days) else '',
                  'last_day': str(days[-1].date()) if len(days) else ''}

    if need_ts:
        ts = timestamp_factors(df)
        path = save_shard(ts, timestamp_dir(root), symbol, fmt)
        path.with_suffix(path.suffix + '.json').write_text(json.dumps({'version': CACHE_VERSION}))
        info['n_timestamp_cols'] = int(ts.shape[1])
    info['timestamp_written'] = bool(need_ts)

    if todo:
        price = raw_price_path(df)
        grid = rolling_threshold_grid(
            intraday_abs_diff(price, codes), codes, lookbacks, pcts)
        for n, m in todo:
            dur = duration_factors(df, lookback=n, pct=m, thr_p=grid[(n, m)])
            path = save_shard(dur, combo_dir(n, m, root), symbol, fmt)
            path.with_suffix(path.suffix + '.json').write_text(json.dumps({'version': CACHE_VERSION}))
            info['n_duration_cols'] = int(dur.shape[1])
    info['combos_written'] = len(todo)
    info['report_written'] = False
    if need_report:
        # 与 DFP 使用同一组 (N, M)。多个组合时取调用方列出的第一组，避免同一目录被后一组覆盖。
        n0, m0 = combos[0]
        report = report_factors(df, lookback=n0, pct=m0)
        path = save_shard(report, report_dir(root), symbol, fmt)
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
    df = read_frame(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{path} 缺少因子列 {missing}，请用 python src/tfcta/prepare/factors.py --overwrite 重建")
    df = df[columns]
    df.index = pd.to_datetime(df.index)
    C.assert_no_holdout_dates(df.index, what=str(path))
    return df


def load_symbol(symbol: str, lookback: int, pct: float,
                root: Path | None = None,
                with_timestamp: bool = True) -> pd.DataFrame:
    """单品种在某个 (N, M) 下的日频因子表（持续期族 + 时间戳族 outer 对齐）。"""
    root = root or C.FACTOR_DAILY_DIR
    p = find_shard(combo_dir(lookback, pct, root), symbol)
    if p is None:
        raise FileNotFoundError(
            f"缺少 {combo_name(lookback, pct)}/{symbol}，请先运行 python src/tfcta/prepare/factors.py")
    out = _read(p, DURATION_FACTORS)
    if with_timestamp:
        q = find_shard(timestamp_dir(root), symbol)
        if q is None:
            raise FileNotFoundError(f"缺少 timestamp/{symbol}")
        # outer：两族交易日理论上一致，一旦不一致能看见 NaN 而不是被静默截断
        out = out.join(_read(q, TIMEPOINT_FACTORS), how='outer')
    return out.sort_index()


def load_report(symbol: str, root: Path | None = None) -> pd.DataFrame:
    """报告其余时序因子。"""
    root = root or C.FACTOR_DAILY_DIR
    path = find_shard(report_dir(root), symbol)
    if path is None:
        raise FileNotFoundError(f"缺少 report/{symbol}，请先运行 python src/tfcta/prepare/factors.py")
    return _read(path, list(REPORT_COLUMNS)).sort_index()


SIGNED_PRIORS = dict(C.FACTOR_SIGNS)


@dataclass
class SignalSet:
    bars: dict
    signed: dict = field(default_factory=dict)

    def raw(self,name):
        if name in self.signed:
            return self.signed[name]/SIGNED_PRIORS[name]
        raise KeyError(f"未知因子 {name}，可用 {sorted(self.signed)}")


def assemble(bars,time_raw):
    """装配时间因子，乘上事前方向。"""
    result = SignalSet(bars=bars)
    for name,sign in C.FACTOR_SIGNS.items():
        result.signed[name] = time_raw[name]*sign
    return result


def load(symbols):
    panels = {s:load_symbol(s,C.IC_REFERENCE_LOOKBACK,C.IC_REFERENCE_PCT) for s in symbols}
    for symbol, frame in panels.items():
        extra = load_report(symbol)
        overlap = frame.index.intersection(extra.index)
        panels[symbol] = frame.join(extra, how='outer')
        if overlap.empty and len(frame) and len(extra):
            raise ValueError(f"{symbol} 的持续期缓存与报告因子缓存没有共同交易日")
    raw = {n:pd.DataFrame({s:f[n] for s,f in panels.items() if n in f.columns}) for n in C.FACTOR_SIGNS}
    missing = [n for n, frame in raw.items() if frame.empty]
    if missing:
        raise KeyError(f"因子缓存缺少 {missing}")
    return assemble(load_daily_bars(symbols),raw)
