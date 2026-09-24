"""日频因子：持续期族、时间戳族，以及入模前的方向符号。

持续期族依赖 (lookback, pct)，时间戳族不依赖参数。两边都输出以 trading_date
为 index 的原始值。方向符号只在 apply_signs 里乘上，不写进计算过程。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..data import sessions
from . import duration as D


# 基础聚合列名（价格族 dur_*，成交量族 vdur_*）
_AGG_SUFFIX = ['mean', 'std', 'max', 'gap', 'extreme']


def day_codes_of(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """把 trading_date 转成连续整数编码，并返回去重后的交易日索引。"""
    td = pd.to_datetime(df['trading_date'])
    codes, uniq = pd.factorize(td, sort=True)
    return codes, pd.DatetimeIndex(uniq)


def _day_bounds(day_codes: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, day_codes[1:] != day_codes[:-1], True])


def _nan_safe(fn, arr):
    """对可能全为 NaN 的切片调用 nan* 聚合而不触发 RuntimeWarning。"""
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    return float(fn(arr))


def basic_aggregations(dur: np.ndarray,
                       day_codes: np.ndarray,
                       days: pd.DatetimeIndex,
                       prefix: str) -> pd.DataFrame:
    """论文 6.3 节的五个基础聚合：均值、波动、极值、极差、极端占比。

    波动用**总体标准差**（ddof=0），与论文公式 (1/N)Σ(x-x̄)² 一致。
    极端占比 = 持续期 >= 均值 + 2σ 的 bar 占比。
    """
    n = len(days)
    out = {f'{prefix}_{s}': np.full(n, np.nan) for s in _AGG_SUFFIX}
    bounds = _day_bounds(day_codes)

    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg = dur[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        mu = float(seg.mean())
        sd = float(seg.std(ddof=0))
        out[f'{prefix}_mean'][k] = mu
        out[f'{prefix}_std'][k] = sd
        out[f'{prefix}_max'][k] = float(seg.max())
        out[f'{prefix}_gap'][k] = float(seg.max() - seg.min())
        out[f'{prefix}_extreme'][k] = float((seg >= mu + C.EXTREME_SIGMA * sd).mean())

    return pd.DataFrame(out, index=days)


def dfp_factors(dur: np.ndarray,
                price: np.ndarray,
                day_codes: np.ndarray,
                days: pd.DatetimeIndex,
                top_ns: list[int] | None = None) -> pd.DataFrame:
    """公允均衡价格偏离 DFP（论文 6.4 节因子 1）。

        FP_t  = mean(closew 于持续期前 N 大的那些分钟)
        DFP_t = FP_t - Close_t

    Close_t 取当日最后一根有效 closew。返回已按当日收盘价归一化的相对偏离
    ``DFP_t / Close_t``——不归一化则 AU 与 RB 的量纲差几个数量级，跨品种不可比。
    """
    top_ns = top_ns or C.FP_TOP_NS
    n = len(days)
    cols = {f'dfp_{"max" if N == 1 else f"top{N}"}': np.full(n, np.nan) for N in top_ns}
    bounds = _day_bounds(day_codes)

    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        d, p = dur[a:b], price[a:b]
        ok = np.isfinite(d) & np.isfinite(p)
        if not ok.any():
            continue
        d_ok, p_ok = d[ok], p[ok]
        close_t = p_ok[-1]
        if not np.isfinite(close_t) or close_t == 0:
            continue
        # 降序取前 N 大；mergesort 保证同值时取更早出现者，结果可复现
        order = np.argsort(-d_ok, kind='mergesort')
        for N in top_ns:
            take = order[:min(N, d_ok.size)]
            fp = float(p_ok[take].mean())
            name = f'dfp_{"max" if N == 1 else f"top{N}"}'
            cols[name][k] = (fp - close_t) / close_t

    return pd.DataFrame(cols, index=days)


def argmax_timepoint(dur: np.ndarray,
                     gamma_norm: np.ndarray,
                     day_codes: np.ndarray,
                     days: pd.DatetimeIndex,
                     name: str) -> pd.DataFrame:
    """持续期最大值出现的归一化日内时点（PMT / VMT，论文 6.4 节因子 2 与 4）。

    归一化到 [0,1] 是硬要求：每日 bar 数在 225/345/465/555 之间变化，不归一化则
    等权合成会被长夜盘品种主导。
    """
    n = len(days)
    out = np.full(n, np.nan)
    bounds = _day_bounds(day_codes)
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        d, g = dur[a:b], gamma_norm[a:b]
        ok = np.isfinite(d) & np.isfinite(g)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        out[k] = float(g[idx[np.argmax(d[idx])]])
    return pd.DataFrame({name: out}, index=days)


def volume_ratio_factors(vdur: np.ndarray,
                         session: np.ndarray,
                         day_codes: np.ndarray,
                         days: pd.DatetimeIndex,
                         min_night_bars: int = C.NIGHT_BARS_MIN) -> pd.DataFrame:
    """成交量持续期比值 VR 及其商品夜盘版 VR_night（论文 6.4 节因子 3 + 商品扩展）。

        VR_t       = VDam_t / VDpm_t                 （论文原版）
        VR_night_t = VDnight_t / VDday_t             （商品扩展，VDday = AM+PM）

    商品的隔夜信息消化主要在夜盘而非早盘，且夜盘结构性领先于日盘，所以两个版本
    都入池，由数据决定优劣。无夜盘（或夜盘 bar 数不足）的交易日 VR_night = NaN。
    """
    n = len(days)
    vr = np.full(n, np.nan)
    vrn = np.full(n, np.nan)
    bounds = _day_bounds(day_codes)

    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        d = vdur[a:b]
        s = session[a:b]
        fin = np.isfinite(d)
        if not fin.any():
            continue
        am = _nan_safe(np.nanmean, d[fin & (s == C.SESSION_AM)])
        pm = _nan_safe(np.nanmean, d[fin & (s == C.SESSION_PM)])
        if np.isfinite(am) and np.isfinite(pm) and pm > 0:
            vr[k] = am / pm

        night_mask = fin & (s == C.SESSION_NIGHT)
        if night_mask.sum() >= min_night_bars:
            nt = _nan_safe(np.nanmean, d[night_mask])
            day = _nan_safe(np.nanmean, d[fin & (s != C.SESSION_NIGHT)])
            if np.isfinite(nt) and np.isfinite(day) and day > 0:
                vrn[k] = nt / day
        # 无夜盘 -> 保持 NaN，绝不填 0

    return pd.DataFrame({'vr': vr, 'vr_night': vrn}, index=days)


def duration_factors(df: pd.DataFrame,
                     lookback: int,
                     pct: float,
                     price_col: str = 'closew',
                     vol_col: str = 'volume',
                     thr_p: pd.Series | None = None,
                     thr_v: pd.Series | None = None) -> pd.DataFrame:
    """一个 (lookback, pct) 参数组合下的全部持续期族日频因子。

    参数
    ----
    df : 单品种分钟表，须含 trading_date / gamma_norm / session 与价量列
         （即已过 sessions.add_intraday_coords）
    thr_p, thr_v : 可选的预算阈值（以日编码为 index），用于第 3 步跨 15 个参数组合
         复用 ``duration.rolling_threshold_grid`` 的结果。给了就不再重算；
         **传入者自己负责保证它来自同样的 (lookback, pct)**——这里不做校验，
         因为阈值序列里没有携带参数信息，校验只能是假的。

    返回
    ----
    以 trading_date 为 index 的日频因子表，列为原始（未乘方向符号）因子值。
    方向符号由 factors.apply_signs 统一施加，便于分别检查原始值与入模值。
    """
    for col in ('trading_date', 'gamma_norm', 'session'):
        if col not in df.columns:
            raise KeyError(f"缺少 {col} 列，请先调用 sessions.add_intraday_coords")

    codes, days = day_codes_of(df)
    price = df[price_col].to_numpy(dtype='float64')
    vol = df[vol_col].to_numpy(dtype='float64')
    gnorm = df['gamma_norm'].to_numpy(dtype='float64')
    sess = df['session'].to_numpy()

    # 价量各自算阈值：成交量的一阶差分量纲与价格完全不同，不能共用阈值
    if thr_p is None:
        thr_p = D.rolling_threshold(D.intraday_abs_diff(price, codes), codes, lookback, pct)
    if thr_v is None:
        thr_v = D.rolling_threshold(D.intraday_abs_diff(vol, codes), codes, lookback, pct)
    dur_p = D.duration_series(price, codes, thr_p)
    dur_v = D.duration_series(vol, codes, thr_v)

    parts = [
        basic_aggregations(dur_p, codes, days, 'dur'),
        basic_aggregations(dur_v, codes, days, 'vdur'),
        dfp_factors(dur_p, price, codes, days),
        argmax_timepoint(dur_p, gnorm, codes, days, 'pmt'),
        argmax_timepoint(dur_v, gnorm, codes, days, 'vmt'),
        volume_ratio_factors(dur_v, sess, codes, days),
    ]
    out = pd.concat(parts, axis=1)
    out.index.name = 'trading_date'
    return out


def _norm_pos(idx_in_seg: int, n_seg: int) -> float:
    """段内归一化位置，与 sessions.add_intraday_coords 的 gamma_norm 同一口径。"""
    if n_seg <= 1:
        return np.nan
    return idx_in_seg / (n_seg - 1)


def _extreme_timepoint(values: np.ndarray, gnorm: np.ndarray, mode: str) -> float:
    ok = np.isfinite(values) & np.isfinite(gnorm)
    if not ok.any():
        return np.nan
    pos = np.flatnonzero(ok)
    v = values[pos]
    # 同值时取**最早**出现者：事件的"首次发生时点"才是信息，argmax/argmin 默认即如此
    j = pos[np.argmax(v) if mode == 'max' else np.argmin(v)]
    return float(gnorm[j])


def _count_new_highs(high: np.ndarray) -> np.ndarray:
    """逐 bar 标记是否刷新当日新高（首根不算）。"""
    ok = np.isfinite(high)
    out = np.zeros(len(high), dtype=bool)
    if ok.sum() < 2:
        return out
    run = np.fmax.accumulate(np.where(ok, high, -np.inf))
    prev = np.r_[-np.inf, run[:-1]]
    out = ok & (high > prev)
    out[np.flatnonzero(ok)[0]] = False   # 当日第一根不算"刷新"
    return out


def timestamp_factors(df: pd.DataFrame, min_night_bars: int = C.NIGHT_BARS_MIN) -> pd.DataFrame:
    """全部时间戳族日频因子。

    参数
    ----
    df : 单品种分钟表，须含 trading_date / gamma_norm / session 与 highw/loww/
         volume/total_turnover（即已过 sessions.add_intraday_coords）
    """
    need = ['trading_date', 'gamma_norm', 'session',
            'highw', 'loww', 'volume', 'total_turnover']
    for col in need:
        if col not in df.columns:
            raise KeyError(f"缺少 {col} 列，请先调用 sessions.add_intraday_coords")

    codes, days = day_codes_of(df)
    n = len(days)
    arr = {c: df[c].to_numpy(dtype='float64')
           for c in ('highw', 'loww', 'volume', 'total_turnover', 'gamma_norm')}
    sess = df['session'].to_numpy()

    names = ['ts_high', 'ts_low', 'ts_vmax', 'ts_tomax',
             'ts_high_am', 'ts_high_pm', 'ts_low_am', 'ts_low_pm',
             'ts_high_night', 'ts_low_night',
             'cnt_high_am', 'cnt_high_pm', 'is_high_am',
             'night_vol_share', 'night_day_range']
    out = {k: np.full(n, np.nan) for k in names}
    bounds = _day_bounds(codes)

    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        g = arr['gamma_norm'][a:b]
        hi, lo = arr['highw'][a:b], arr['loww'][a:b]
        vol, to = arr['volume'][a:b], arr['total_turnover'][a:b]
        s = sess[a:b]
        if not np.isfinite(hi).any():
            continue

        # --- 全日极值时点 ---
        out['ts_high'][k] = _extreme_timepoint(hi, g, 'max')
        out['ts_low'][k] = _extreme_timepoint(lo, g, 'min')
        out['ts_vmax'][k] = _extreme_timepoint(vol, g, 'max')
        out['ts_tomax'][k] = _extreme_timepoint(to, g, 'max')

        # --- 分时段极值时点（段内归一化，与全日归一化不是一回事）---
        night = s == C.SESSION_NIGHT
        has_night = np.isfinite(hi[night]).sum() >= min_night_bars
        for tag, mask in (('am', s == C.SESSION_AM),
                          ('pm', s == C.SESSION_PM),
                          ('night', night)):
            if tag == 'night' and not has_night:
                continue           # 无夜盘 -> 保持 NaN，不填 0
            m = np.flatnonzero(mask)
            if m.size == 0:
                continue
            seg_h, seg_l = hi[m], lo[m]
            local = np.arange(m.size, dtype='float64')
            denom = m.size - 1 if m.size > 1 else np.nan
            gl = local / denom if m.size > 1 else np.full(m.size, np.nan)
            out[f'ts_high_{tag}'][k] = _extreme_timepoint(seg_h, gl, 'max')
            out[f'ts_low_{tag}'][k] = _extreme_timepoint(seg_l, gl, 'min')

        # --- 刷新新高次数与最高价归属 ---
        new_hi = _count_new_highs(hi)
        out['cnt_high_am'][k] = float(new_hi[s == C.SESSION_AM].sum())
        out['cnt_high_pm'][k] = float(new_hi[s == C.SESSION_PM].sum())
        fin_h = np.isfinite(hi)
        if fin_h.any():
            j = np.flatnonzero(fin_h)[np.argmax(hi[fin_h])]
            out['is_high_am'][k] = float(s[j] == C.SESSION_AM)

        # --- 夜盘结构因子（商品扩展）---
        if has_night:
            day = ~night
            v_night = np.nansum(vol[night])
            v_all = np.nansum(vol)
            if np.isfinite(v_all) and v_all > 0:
                out['night_vol_share'][k] = v_night / v_all
            rng_n = np.nanmax(hi[night]) - np.nanmin(lo[night])
            if np.isfinite(hi[day]).any() and np.isfinite(lo[day]).any():
                rng_d = np.nanmax(hi[day]) - np.nanmin(lo[day])
                if np.isfinite(rng_d) and rng_d > 0 and np.isfinite(rng_n):
                    out['night_day_range'][k] = rng_n / rng_d

    res = pd.DataFrame(out, index=days)
    res.index.name = 'trading_date'
    return res


class UnsignedFactor(KeyError):
    """出现了方向表里没有的因子列。"""


def apply_signs(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """按 config.ALL_SIGNS 把每列乘上方向符号，使所有因子统一为「越大越看多」。

    参数
    ----
    strict : True 时遇到方向表中不存在的列直接报错。默认严格——静默按 +1 处理
             等于埋下一个无人知晓的方向假设。
    """
    unknown = [c for c in df.columns if c not in C.ALL_SIGNS]
    if unknown and strict:
        raise UnsignedFactor(
            f"以下因子列没有登记方向: {unknown}\n"
            "请在 config.FACTOR_SIGNS（有论文先验）或 EXPLORATORY_SIGNS（无先验）中登记。"
        )
    out = df.copy()
    for c in out.columns:
        out[c] = out[c] * C.ALL_SIGNS.get(c, 1)
    return out


def symbol_daily_factors(minute_df: pd.DataFrame,
                         lookback: int,
                         pct: float,
                         with_coords: bool = False) -> pd.DataFrame:
    """单品种、单参数组合下的全部日频因子（原始值，未施加方向）。

    参数
    ----
    minute_df   : 单品种分钟表，须含 config.FACTOR_FIELDS 全部字段
    with_coords : True 表示 minute_df 已含 gamma_norm/session，跳过坐标计算
    """
    df = minute_df if with_coords else sessions.add_intraday_coords(minute_df)
    dur = duration_factors(df, lookback=lookback, pct=pct)
    ts = timestamp_factors(df)
    out = pd.concat([dur, ts], axis=1)
    out.index.name = 'trading_date'
    return out


def factor_health(df: pd.DataFrame) -> pd.DataFrame:
    """因子层诊断表：缺失率、量级、是否退化为常数。

    设计文档第 11 节要求每一步都有验收项。这张表就是因子步的验收依据：
    * miss_ratio 接近 1 -> 该因子在此品种上不可用（通常是无夜盘）
    * nunique <= 2      -> 因子退化，等权合成里是纯噪声
    * 全为 0            -> 大概率是把 NaN 填成了 0（上一轮踩过的坑）
    """
    rows = []
    for c in df.columns:
        s = df[c]
        v = s.dropna()
        rows.append({
            'factor': c,
            'miss_ratio': float(s.isna().mean()),
            'nunique': int(v.nunique()),
            'mean': float(v.mean()) if len(v) else np.nan,
            'std': float(v.std(ddof=0)) if len(v) else np.nan,
            'p05': float(v.quantile(0.05)) if len(v) else np.nan,
            'p50': float(v.quantile(0.50)) if len(v) else np.nan,
            'p95': float(v.quantile(0.95)) if len(v) else np.nan,
            'all_zero': bool(len(v) > 0 and (v == 0).all()),
        })
    return pd.DataFrame(rows).set_index('factor')
