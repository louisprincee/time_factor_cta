"""测试用合成分钟行情。只在测试里使用，不进入研究或回测管线。"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 品种类别 -> 夜盘收盘墙钟 (小时, 分钟)；None 表示无夜盘
# 夜盘 bar 数 = 收盘时刻 - 21:00，配合日盘 225 根得到 225/345/465/555 四种网格
NIGHT_CLASS = {
    'no_night': None,
    'night_2300': (23, 0),     # 120 分钟 -> 全日 345
    'night_0100': (25, 0),     # 240 分钟 -> 全日 465（25 = 次日 01:00）
    'night_0230': (26, 30),    # 330 分钟 -> 全日 555（26:30 = 次日 02:30）
}


def _session_minutes(night_class: str):
    """返回 (时段, 起小时, 起分钟, 止小时, 止分钟) 的构造参数。"""
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
    """构造某个 trading_date 的全部分钟时间戳。"""
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
    """日内波动率的 U 形（开盘与收盘高、盘中低），夜盘段整体略高。"""
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
    """生成单品种分钟表（含 15 个字段中因子需要的全部列 + dominant_id）。"""
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

