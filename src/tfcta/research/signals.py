"""各步共用的信号装配：时间戳/持续期、传统量价、慢信号、反转代理。

方向在这里一次性乘好（``FACTOR_SIGNS`` / ``TECH_PRIOR_SIGNS``），下游拿到的
``signed`` 一律"越大越看多"。不带方向的量价指标放在 ``unsigned``，只报 IC。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .. import config as C
from ..data import universe as U
from ..factors import external_factor_cache, slow, tech
from . import book, jobs, panel, stats

Z_WINDOW = 252
Z_MIN = 120


def trail_z(raw: pd.DataFrame) -> pd.DataFrame:
    return stats.exante_z(raw, Z_WINDOW, Z_MIN)


@dataclass
class SignalSet:
    bars: dict[str, pd.DataFrame]
    signed: dict[str, pd.DataFrame] = field(default_factory=dict)
    unsigned: dict[str, pd.DataFrame] = field(default_factory=dict)
    family: dict[str, str] = field(default_factory=dict)
    vol: pd.DataFrame | None = None


def reversal_proxies(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """CLV：收盘价在当日区间里的位置（-1 最低，+1 最高）；ret_day：当日开到收。"""
    hi, lo = bars['highw'], bars['loww']
    clv = (2 * bars['closew'] - hi - lo) / (hi - lo).where(hi > lo)
    ret_day = (bars['closew'] - bars['openw']) / bars['open'].where(bars['open'] > 0)
    return {'clv': clv, 'ret_day': ret_day}


def load(symbols: list[str]) -> SignalSet:
    bars = panel.load_daily_bars(symbols)
    out = SignalSet(bars=bars)

    for n, s in C.FACTOR_SIGNS.items():
        raw = jobs.load_factor(n, C.IC_REFERENCE_LOOKBACK, C.IC_REFERENCE_PCT, symbols)
        out.signed[n] = raw * s
        out.family[n] = '时间戳' if n in C.TIMESTAMP_FACTORS else '持续期'
    out.signed['time_combo'] = book.average_signals(
        [trail_z(out.signed[n]) for n in C.FACTOR_SIGNS])
    out.family['time_combo'] = '时间戳+持续期'

    mp = panel.multiplicative_prices(bars)
    tech_raw = tech.build(mp['close'], mp['high'], mp['low'], bars['volume'])
    for n, s in C.TECH_PRIOR_SIGNS.items():
        out.signed[n] = tech_raw[n] * s
        out.family[n] = '量价'
    for n in C.TECH_UNSIGNED:
        out.unsigned[n] = tech_raw[n]
        out.family[n] = '量价(无方向)'

    out.signed['tsmom'] = slow.tsmom(bars['close'], bars['closew'])
    out.family['tsmom'] = '慢信号'
    out.signed['carry'] = slow.carry(bars['close'], bars['closew'])
    out.family['carry'] = '慢信号'

    universe = U.load_universe()
    for n, factor in slow.cross_sectional_momentum(
            bars['close'], bars['closew'], universe).items():
        out.unsigned[n] = factor
        out.family[n] = '截面动量(独立候选)'
    out.unsigned['return_skew_60'] = slow.rolling_return_skewness(
        bars['close'], bars['closew'], window=60)
    out.family['return_skew_60'] = '尾部风险(独立候选)'
    out.unsigned['cs_low_vol_60'] = slow.cross_sectional_low_volatility(
        bars['close'], bars['closew'], universe, window=60)
    out.family['cs_low_vol_60'] = '截面低波动(独立候选)'

    external_families = {
        'carry_main_sub_yield': '外部期限结构(候选)',
        'carry_main_sub_annualized': '外部期限结构(候选)',
        'carry_main_sub_annualized_trading': '外部期限结构(候选)',
        'warehouse_on_warrant': '外部库存(候选)',
        'warehouse_log_level': '外部库存(候选)',
        'warehouse_low': '外部库存(候选)',
        'warehouse_change_20d': '外部库存(候选)',
        'warehouse_drawdown_20d': '外部库存(候选)',
        'main_open_interest': '外部持仓(候选)',
        'main_oi_change_20d': '外部持仓(候选)',
        'main_oi_change_60d': '外部持仓(候选)',
        'spot_basis_morning': '外部基差(候选)',
        'spot_basis_morning_pct': '外部基差(候选)',
        'spot_basis_noon': '外部基差(候选)',
        'spot_basis_noon_pct': '外部基差(候选)',
    }
    for name, factor in external_factor_cache.load_external_panel(symbols).items():
        out.unsigned[name] = factor.reindex(bars['close'].index)
        out.family[name] = external_families.get(name, '外部数据(候选)')

    rev = reversal_proxies(bars)
    out.signed['neg_clv'] = -rev['clv']
    out.signed['neg_ret_day'] = -rev['ret_day']
    out.family['neg_clv'] = out.family['neg_ret_day'] = '反转代理'

    out.vol = slow.daily_vol(bars['close'], bars['closew'])
    return out
