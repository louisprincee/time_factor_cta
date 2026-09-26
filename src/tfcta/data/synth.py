"""合成分钟数据：复刻真实面板结构，用于在没有 10.7GB 单体文件时验证整条管道。

复刻的结构特征（设计文档第 2.1 / 2.2 节）
----------------------------------------
* 两层 MultiIndex 列 ``(品种, 字段)``，15 个字段，float32
* index 为分钟 datetime
* **夜盘归属次一交易日**（21:00 的 bar 其 trading_date 为次日）
* 参差的每日 bar 数：225 / 345 / 465 / 555，按品种类别
* 换月：``dominant_id`` 定期跳变，且换月日复权价有跳变
* 缺失：早期年份部分品种尚未上市 -> 全 NaN

合成数据**不用于任何研究结论**，只用于验证代码正确性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C

# 品种类别 -> 夜盘收盘墙钟 (小时, 分钟)；None 表示无夜盘
# 夜盘 bar 数 = 收盘时刻 - 21:00，配合日盘 225 根得到 225/345/465/555 四种网格
NIGHT_CLASS = {
    'no_night': None,
    'night_2300': (23, 0),     # 120 分钟 -> 全日 345
    'night_0100': (25, 0),     # 240 分钟 -> 全日 465（25 = 次日 01:00）
    'night_0230': (26, 30),    # 330 分钟 -> 全日 555（26:30 = 次日 02:30）
}


def _session_minutes(night_class: str):
    """返回 (时段, 起小时, 起分钟, 止小时, 止分钟) 的构造参数。

    日盘沿用中国商品期货的真实网格（合计 225 根）：
        AM 09:00-10:15 (75), 10:30-11:30 (60)
        PM 13:30-15:00 (90)
    夜盘 21:00 起，收盘时刻按品种类别，小时数可 >= 24 表示跨自然日。
    """
    blocks = []
    end = NIGHT_CLASS[night_class]
    if end is not None:
        blocks.append(('night', 21, 0, end[0], end[1]))
    blocks.append(('am', 9, 0, 10, 15))
    blocks.append(('am', 10, 30, 11, 30))
    blocks.append(('pm', 13, 30, 15, 0))
    return blocks


def _day_timestamps(trading_day: pd.Timestamp, night_class: str,
                    prev_trading_day: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """构造某个 trading_date 的全部分钟时间戳。

    夜盘部分的墙钟日期是**前一个交易日**，不是前一自然日。这个区别不是细节：
    周一的夜盘在上周五 21:00（间隔 3 天），节后第一个交易日的夜盘在节前最后一个
    交易日。若按自然日减一，周一的夜盘会落在周日——一个不存在的交易时刻，
    且会让"夜盘归属次一交易日"的验收项拿到错误的样本。
    """
    stamps = []
    prev = None if prev_trading_day is None else pd.Timestamp(prev_trading_day).normalize()
    for kind, h0, m0, h1, m1 in _session_minutes(night_class):
        if kind == 'night':
            if prev is None:
                continue          # 样本首日没有"前一交易日"，其夜盘不在数据范围内
            start = prev + pd.to_timedelta(h0, unit='h') + pd.to_timedelta(m0, unit='m')
            end = prev + pd.to_timedelta(h1, unit='h') + pd.to_timedelta(m1, unit='m')
        else:
            start = trading_day + pd.to_timedelta(h0, unit='h') + pd.to_timedelta(m0, unit='m')
            end = trading_day + pd.to_timedelta(h1, unit='h') + pd.to_timedelta(m1, unit='m')
        stamps.append(pd.date_range(start + pd.to_timedelta(1, unit='min'), end, freq='1min'))
    return pd.DatetimeIndex(np.concatenate([s.values for s in stamps])).sort_values()


def _vol_shape(n: int, has_night: bool) -> np.ndarray:
    """日内波动率的 U 形（开盘与收盘高、盘中低），夜盘段整体略高。

    这一项不是装饰：持续期的分布完全由「一阶差分绝对值的日内结构」决定，用平坦的
    iid 波动率会得到几乎恒为 1-2 分钟的持续期，无法反映真实数据的量级。
    """
    u = np.linspace(0, 1, n)
    shape = 0.6 + 1.4 * (2 * u - 1) ** 2          # 端点 2.0，中点 0.6
    if has_night:
        shape[:int(n * 0.35)] *= 1.25             # 夜盘承接隔夜信息，波动更高
    return shape


def make_symbol(trading_days: pd.DatetimeIndex,
                night_class: str = 'night_2300',
                start_price: float = 3000.0,
                seed: int = 0,
                roll_every: int = 60,
                tick_ratio: float = 3e-4,
                vol_scale=1.0,
                listed_from: pd.Timestamp | None = None) -> pd.DataFrame:
    """生成单品种分钟表（含 15 个字段中因子需要的全部列 + dominant_id）。

    参数
    ----
    tick_ratio : 最小变动价位相对价格的比例。**必须离散化到 tick**，否则连续价格
                 的一阶差分几乎处处非零，持续期恒为 1-2 分钟，与真实商品期货
                 （盘中大量零变动分钟）的分布相差一个数量级。
    vol_scale  : 成交量倍数。可以是常数，也可以是 ``f(trading_day) -> float``，
                 后者用于构造"先活跃后塌缩"的僵尸品种（ZC 型），以检验品种池的
                 时点有效性确实能把它挡在后期年份之外。
    """
    rng = np.random.default_rng(seed)
    frames = []
    price = start_price
    contract_no = 0
    tick = max(start_price * tick_ratio, 1e-8)
    has_night = NIGHT_CLASS[night_class] is not None

    for k, day in enumerate(trading_days):
        prev_day = pd.Timestamp(trading_days[k - 1]) if k > 0 else None
        idx = _day_timestamps(pd.Timestamp(day), night_class, prev_day)
        n = len(idx)
        if n == 0:
            continue
        if k % roll_every == 0:
            contract_no += 1

        # 随机游走 + 日内 U 形波动率 + 日间波动聚集，再离散化到最小变动价位
        day_vol = start_price * 2e-4 * rng.lognormal(0, 0.35)
        step = rng.normal(0, 1, n) * day_vol * _vol_shape(n, has_night)
        close = np.round((price + np.cumsum(step)) / tick) * tick
        price = float(close[-1])

        # 日内极值：在收盘价上下各加不到一个 tick 的随机幅度，再对齐 tick
        high = close + np.ceil(np.abs(rng.normal(0, 0.8, n))) * tick
        low = close - np.ceil(np.abs(rng.normal(0, 0.8, n))) * tick
        open_ = np.r_[close[0], close[:-1]]
        # 成交量同样呈 U 形，且取整（真实成交量是整数手，大量重复值影响量能持续期）
        scale = vol_scale(pd.Timestamp(day)) if callable(vol_scale) else float(vol_scale)
        vol = np.round(rng.lognormal(6.0, 0.8, n) * _vol_shape(n, has_night) * scale)
        turnover = vol * close

        # 未复权 vs 复权：给复权价加一个随合约递增的乘数，模拟 889 的拼接
        adj = 1.0 + 0.001 * contract_no
        df = pd.DataFrame({
            'open': open_, 'high': high, 'low': low, 'close': close,
            'volume': vol, 'total_turnover': turnover,
            'open_interest': rng.lognormal(9, 0.3, n),
            'trading_date': pd.Timestamp(day),
            'dominant_id': f'SYN{contract_no:04d}',
            'openw': open_ * adj, 'closew': close * adj,
            'highw': high * adj, 'loww': low * adj,
            'open_interest99': rng.lognormal(9, 0.3, n),
            'volume99': vol * 1.3,
        }, index=idx)
        frames.append(df)

    out = pd.concat(frames, axis=0).sort_index()
    if listed_from is not None:
        num = out.columns.difference(['trading_date', 'dominant_id'])
        out.loc[out.index < listed_from, num] = np.nan
    return out


def make_panel(symbols: dict[str, str],
               start: str = '2015-01-01',
               end: str = '2016-06-30',
               seed: int = 7) -> pd.DataFrame:
    """生成多品种宽面板，列为 (品种, 字段) 两层 MultiIndex——与真实文件同构。

    参数
    ----
    symbols : ``{品种代码: 夜盘类别}``，或 ``{品种代码: {make_symbol 的关键字参数}}``。
              后者用于构造流动性各异、上市时间各异的品种，检验品种池筛选。
    """
    days = pd.bdate_range(start, end)
    parts = []
    for i, (sym, spec) in enumerate(symbols.items()):
        kw = {'night_class': spec} if isinstance(spec, str) else dict(spec)
        kw.setdefault('night_class', 'night_2300')
        kw.setdefault('start_price', 1000.0 * (i + 1))
        kw.setdefault('seed', seed + i)
        df = make_symbol(days, **kw)
        df.columns = pd.MultiIndex.from_product([[sym], df.columns],
                                                names=['future', 'field'])
        parts.append(df)
    panel = pd.concat(parts, axis=1).sort_index()
    return panel


DEFAULT_SYNTH_SYMBOLS = {
    'RB': 'night_2300',
    'CU': 'night_0100',
    'AU': 'night_0230',
    'JD': 'no_night',
    'M': 'night_2300',
}
