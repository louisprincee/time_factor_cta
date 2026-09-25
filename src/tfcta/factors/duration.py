"""持续期计算（设计文档第 6.2 节，论文核心定义）。

论文原文定义
------------
    Price Duration_j  = t_j - t_i,  t_j > t_i,  if |P_j - P_i| >= Threshold
    Volume Duration_j = t_j - t_i,  t_j > t_i,  if |V_j - V_i| >= Threshold

逐交易日独立计算，不跨日。对当日时刻 i 的观测值 Value_i，向前遍历当日全部已发生
观测 j < i，取**最近的（最后一个）**满足 |Value_i - Value_j| >= Threshold_t 的 j，
则 Duration_{t,i} = i - j（单位为分钟/bar 数间隔）。

极易做错的细节（上一轮小时频实现就错在这里）
--------------------------------------------
若当日此前不存在任何满足阈值条件的 j，则持续期**从当日开盘时刻起累计至当前时刻**，
即 0-based 下 Duration = i，而**不是 0、不是 NaN**。

阈值为动态滚动（论文明确反对固定阈值）
--------------------------------------
    Threshold_t = 过去 N 个交易日全部「分钟变化绝对值」的第 M 分位数
其中「分钟变化」指日内相邻 bar 的一阶差分绝对值，跨日首根不计。逐品种各自计算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def intraday_abs_diff(values: np.ndarray, day_codes: np.ndarray) -> np.ndarray:
    """日内相邻 bar 的一阶差分绝对值；每日首根记为 NaN（不跨日）。

    参数
    ----
    values    : 一维观测值（价格或成交量）
    day_codes : 同长度的整数日编码（同一 trading_date 取同一值，且已按时间排序）
    """
    v = np.asarray(values, dtype='float64')
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
    """按交易日滚动计算阈值：过去 ``lookback`` 日全部分钟变化绝对值的第 ``pct`` 分位数。

    返回以日编码为 index 的 Series，值为**当日可用**的阈值（仅使用 t-lookback..t-1，
    不含当日，避免自包含导致的前视）。

    实现说明
    --------
    "过去 N 日全部分钟观测的分位数" 是把 N 天的分钟样本**汇总成一个池子**再取分位数，
    不是"每日分位数再平均"。两者不等价，论文要的是前者。

    为在可接受的时间内得到精确结果，这里按日分组保存排序后的样本，滚动窗口内用
    np.concatenate 合并后取分位数。N<=300 日、每日<=555 根，池子最大约 16.6 万个
    浮点数，单日 np.percentile 约 2ms，全样本约 2900 日 -> 约 6 秒/品种/参数组。
    """
    s = pd.Series(abs_diff)
    per_day = [g.dropna().to_numpy() for _, g in s.groupby(day_codes, sort=True)]
    days = np.array(sorted(pd.unique(day_codes)))

    out = np.full(len(days), np.nan)
    for k in range(len(days)):
        lo = max(0, k - lookback)
        if k == 0:
            continue
        pool = [a for a in per_day[lo:k] if a.size]
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
    """一次算出 ``lookbacks × pcts`` 全部阈值序列，结果与逐个调用 rolling_threshold 相同。

    为什么要有这个函数
    ------------------
    第 3 步可以一次算多组 (N, M)。逐个调用会把「按日切分样本」和「拼接滚动窗口」
    按组合数重复做，而这两件事里只有最后取分位数那一步与 M 有关。
    这里改成：切分做 1 遍，每个 N 拼接 1 遍，然后 ``np.percentile`` 一次给出全部
    分位数。逐元素结果完全一致（有测试钉住）。

    返回
    ----
    ``{(lookback, pct): 以日编码为 index 的阈值 Series}``
    """
    s = pd.Series(abs_diff)
    per_day = [g.dropna().to_numpy() for _, g in s.groupby(day_codes, sort=True)]
    days = np.array(sorted(pd.unique(day_codes)))
    qs = list(pcts)

    out = {(lb, p): np.full(len(days), np.nan) for lb in lookbacks for p in qs}
    for lb in lookbacks:
        for k in range(1, len(days)):
            pool = [a for a in per_day[max(0, k - lb):k] if a.size]
            if not pool:
                continue
            cat = np.concatenate(pool)
            if not cat.size:
                continue
            vals = np.percentile(cat, qs)      # 一次给出全部分位数
            for p, v in zip(qs, np.atleast_1d(vals)):
                out[(lb, p)][k] = v
    return {key: pd.Series(arr, index=days) for key, arr in out.items()}


def duration_one_day(values: np.ndarray, threshold: float) -> np.ndarray:
    """单交易日的持续期序列（向量化）。

    已用暴力双循环在 300 组随机样本上交叉验证，逐元素完全一致，含"无满足条件的 j
    则从开盘累计"这一分支。

    复杂度 O(n^2) 但 n <= 555，峰值内存 555^2 * 8B ~ 2.4MB；实测 0.10 / 0.83 / 2.14 ms
    对应 n = 225 / 345 / 555。
    """
    v = np.asarray(values, dtype='float64')
    n = len(v)
    if n == 0:
        return np.zeros(0)
    if not np.isfinite(threshold):
        return np.full(n, np.nan)

    D = np.abs(v[:, None] - v[None, :])
    # j < i 且 |v_i - v_j| >= threshold
    ok = (D >= threshold) & (np.arange(n)[None, :] < np.arange(n)[:, None])
    has = ok.any(axis=1)
    # 最后一个 True 的列号：对反转数组取 argmax 再换算
    last_j = np.where(has, n - 1 - ok[:, ::-1].argmax(axis=1), 0)
    # 无满足条件者：从当日开盘累计，0-based 即 i
    dur = np.where(has, np.arange(n) - last_j, np.arange(n)).astype('float64')
    # NaN 观测无法判定，置 NaN
    dur[~np.isfinite(v)] = np.nan
    return dur


def threshold_diagnostics(abs_diff: np.ndarray,
                          thresholds: pd.Series,
                          day_codes: np.ndarray,
                          tick_hint: float | None = None) -> dict:
    """阈值退化诊断（设计文档第 6.2.1 节，第 3 步必备验收项）。

    分钟价格是离散 tick 网格，盘中大量整分钟零变动。当「零变动占比 >= M%」时，
    第 M 分位数阈值会塌缩到 1 个 tick，持续期退化为"距上次 tick 跳动几分钟"。
    这不是 bug，但必须知道它在哪些品种/年份发生了。

    返回
    ----
    zero_ratio    : 非 NaN 的分钟变化中恰好为 0 的占比
    thr_median    : 阈值中位数
    thr_over_tick : 阈值 / 最小变动价位（tick_hint 为 None 时用非零变化的最小值估计）
    degenerate    : thr_over_tick <= 1.5，即阈值基本锁死在一个 tick
    """
    d = np.asarray(abs_diff, dtype='float64')
    fin = d[np.isfinite(d)]
    zero_ratio = float((fin == 0).mean()) if fin.size else np.nan

    nz = fin[fin > 0]
    tick = float(tick_hint) if tick_hint else (float(nz.min()) if nz.size else np.nan)
    # 预热期不足时整段阈值都是 NaN，这是正常情形（不是异常），所以自己判空，
    # 不让 np.nanmedian 抛 All-NaN slice 警告——那个警告会淹没真正值得看的输出。
    thr_arr = np.asarray(thresholds, dtype='float64').ravel()
    thr_fin = thr_arr[np.isfinite(thr_arr)]
    thr_med = float(np.median(thr_fin)) if thr_fin.size else np.nan
    ratio = thr_med / tick if (np.isfinite(thr_med) and tick and np.isfinite(tick)) else np.nan

    return {
        'zero_ratio': zero_ratio,
        'thr_median': thr_med,
        'tick_est': tick,
        'thr_over_tick': ratio,
        'degenerate': bool(np.isfinite(ratio) and ratio <= 1.5),
    }


def duration_series(values: np.ndarray,
                    day_codes: np.ndarray,
                    thresholds: pd.Series) -> np.ndarray:
    """整段序列的持续期，逐交易日独立计算。

    参数
    ----
    values     : 一维观测值（价格或成交量），已按时间排序
    day_codes  : 同长度日编码
    thresholds : 以日编码为 index 的当日阈值（来自 rolling_threshold）

    阈值为 NaN 的交易日（预热期不足）整日返回 NaN——这是正确行为，不要回填。
    """
    v = np.asarray(values, dtype='float64')
    out = np.full(len(v), np.nan)
    thr_map = thresholds.to_dict()

    order = np.argsort(day_codes, kind='mergesort')
    if not np.array_equal(order, np.arange(len(day_codes))):
        raise ValueError("day_codes 未按时间排序，持续期会算错；请先排序")

    # 连续同日区间的起止位置
    bounds = np.flatnonzero(np.r_[True, day_codes[1:] != day_codes[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        thr = thr_map.get(day_codes[a], np.nan)
        out[a:b] = duration_one_day(v[a:b], thr)
    return out
