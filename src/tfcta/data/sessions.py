"""日内时序坐标与时段划分（设计文档第 6.1 节）。

本模块建立整个项目的"日"和"日内位置"定义。所有因子都依赖它，任何错误都会
静默传播到全部结果，所以这里的每个函数都配有测试。

核心纪律
--------
1. 唯一合法的"日"是 ``trading_date``，绝不是 ``datetime.date``。
   中国商品期货夜盘 21:00 开盘，归属**次一**交易日，且结构性地领先于日盘。
2. 日内位置 gamma 必须归一化到 [0, 1]，因为每日 bar 数跨品种（225/345/465/555）
   和跨年份都不同。不归一化则等权合成会被长夜盘品种主导。
3. 无夜盘品种的夜盘类指标取 NaN，绝不填 0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C


def tag_session(index: pd.DatetimeIndex) -> pd.Series:
    """按墙钟时间给每根 bar 打 NIGHT / AM / PM 标签。

    夜盘判定用小时而非精确区间，因为夜盘收盘时间随品种（23:00/01:00/02:30）
    和年份（2016 年有一轮调整）变化，用 ``hour >= 20 or hour <= 4`` 覆盖全部情形。

    区间之外的 bar（如集合竞价残留、异常时间戳）标记为 NaN，由调用方决定是否剔除。
    """
    idx = pd.DatetimeIndex(index)
    hour = idx.hour
    t = idx.time

    out = pd.Series(np.nan, index=idx, dtype=object)
    is_night = (hour >= C.NIGHT_START_HOUR) | (hour <= C.NIGHT_END_HOUR)
    is_am = np.array([C.AM_START <= x <= C.AM_END for x in t])
    is_pm = np.array([C.PM_START <= x <= C.PM_END for x in t])

    # 夜盘优先：夜盘时段与 AM/PM 的墙钟区间不重叠，但先赋值可防止边界意外
    out[is_pm] = C.SESSION_PM
    out[is_am] = C.SESSION_AM
    out[is_night] = C.SESSION_NIGHT
    return out


def add_intraday_coords(df: pd.DataFrame) -> pd.DataFrame:
    """为单品种分钟数据添加日内坐标列。

    参数
    ----
    df : 单品种分钟表，index 为 datetime，必须含 ``trading_date`` 列。

    新增列
    ------
    gamma      : 当日第几根 bar，从 1 开始（论文的 gamma = 1..N）
    n_bars     : 当日总 bar 数 N_t
    gamma_norm : (gamma - 1) / (N_t - 1)，落在 [0, 1]；N_t == 1 时为 NaN
    session    : NIGHT / AM / PM
    """
    if 'trading_date' not in df.columns:
        raise KeyError("缺少 trading_date 列——这是唯一合法的'日'定义，不能用 index.date 替代")

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    # 必须先按时间排序，否则 gamma 编号错乱
    out = out.sort_index(kind='mergesort')
    out['trading_date'] = pd.to_datetime(out['trading_date']).dt.normalize()

    # transform 取的列必须在建 grp 时就已存在。原来这里取的是 grp['gamma']，
    # 而 gamma 是建完 grp 之后才赋的——能跑通只是因为 groupby 持有的是 out 的引用，
    # 属于实现细节，跨 pandas 版本不保证。改成对 trading_date 自己 transform。
    grp = out.groupby('trading_date', sort=False)
    out['gamma'] = grp.cumcount() + 1
    out['n_bars'] = grp['trading_date'].transform('size')

    denom = (out['n_bars'] - 1).astype('float64')
    out['gamma_norm'] = np.where(denom > 0, (out['gamma'] - 1) / denom, np.nan)

    out['session'] = tag_session(out.index).to_numpy()
    return out


def has_night_session(df: pd.DataFrame,
                      min_bars: int = C.NIGHT_BARS_MIN) -> pd.Series:
    """每个 trading_date 是否存在夜盘（bar 数达到 min_bars 才算）。

    返回以 trading_date 为 index 的布尔 Series。

    为什么要 min_bars 而不是 ``> 0``：节假日后的首个交易日常有零星夜盘 bar，
    把它们当成完整夜盘会让夜盘类因子的分母极小、数值爆炸。

    门槛默认取 ``config.NIGHT_BARS_MIN``，与 ``classify_night_bars`` 同源——
    两处判定"有没有夜盘"必须用同一个数，否则会出现"归类成有夜盘但因子按无夜盘算"。
    """
    if 'session' not in df.columns:
        df = add_intraday_coords(df)
    night = (df['session'] == C.SESSION_NIGHT).groupby(df['trading_date']).sum()
    return night >= min_bars


def session_mask(df: pd.DataFrame, session: str) -> np.ndarray:
    return (df['session'] == session).to_numpy()


def day_bar_counts(df: pd.DataFrame) -> pd.DataFrame:
    """每个 trading_date 的分时段 bar 数，用于 step1 验收与结构诊断。"""
    if 'session' not in df.columns:
        df = add_intraday_coords(df)
    tab = (df.groupby(['trading_date', 'session'], dropna=False)
             .size().unstack(fill_value=0))
    for s in C.SESSIONS:
        if s not in tab.columns:
            tab[s] = 0
    tab['TOTAL'] = tab[C.SESSIONS].sum(axis=1)
    return tab[C.SESSIONS + ['TOTAL']]


def classify_night_bars(median_night_bars: float) -> str:
    """由夜盘 bar 数的中位数归类网格类别。门槛只写在这一处。"""
    med = float(median_night_bars)
    if not np.isfinite(med) or med < C.NIGHT_BARS_MIN:
        return 'no_night'
    if med <= C.NIGHT_BARS_2300:
        return 'night_2300'
    if med <= C.NIGHT_BARS_0100:
        return 'night_0100'
    return 'night_0230'


def classify_night_length(bar_counts: pd.DataFrame) -> str:
    """按夜盘 bar 数中位数归类品种的网格类别，用于验收比对。"""
    return classify_night_bars(bar_counts[C.SESSION_NIGHT].median())
