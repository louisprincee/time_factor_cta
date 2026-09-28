"""因子库：全部日频因子的登记与装配，以及标准化、等权合成两个变换。

新因子一律在这里登记（分钟级因子在 ``intraday``/``cache`` 构造，外部因子在 ``external``
构造，日频量价/慢信号在 ``daily`` 构造），第 3 步的因子目录与第 4–7 步自动可用。

方向在装配时一次性乘好（``FACTOR_SIGNS`` / ``TECH_PRIOR_SIGNS``）：``signed`` 一律
"越大越看多"，``unsigned`` 不带方向，只报 IC，入书时必须显式给符号。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config as C
from ..data import bars as B
from ..data import universe as U
from . import cache, daily, external

Z_WINDOW = 252
Z_MIN = 120

# 已定向因子相对原始值的符号：signed = raw × prior。反转代理、time_combo 本身就是定向后的量。
# time_combo 在装配时已经是各成员 trail_z 的等权平均，入书时不再做第二次 z 分数；
# time_combo_trend 是它按趋势方向过滤后的版本，同样不再标准化。
STANDARDIZED_FACTORS = frozenset({'time_combo', 'time_combo_trend'})
SIGNED_PRIORS = {
    **C.FACTOR_SIGNS,
    'time_combo': +1,
    'time_combo_trend': +1,
    **C.TECH_PRIOR_SIGNS,
    'tsmom': +1,
    'tsmom_20': +1,
    'ts_low_slow': +1,
    'carry': +1,
    'carry_roll': +1,
    'cs_carry_roll': +1,
    'carry_ms': +1,
    'cs_carry_ms': +1,
    'neg_clv': +1,
    'neg_ret_day': +1,
}
EXTERNAL_FAMILIES = external.FAMILIES


# --------------------------------------------------------------------------
# 变换
# --------------------------------------------------------------------------
def exante_z(factor: pd.DataFrame,
             window: int | None = None,
             min_periods: int | None = None) -> pd.DataFrame:
    """截至当日的滚动均值、标准差标准化，截到 ±3。只用过去，不含未来。"""
    window = C.IC_Z_WINDOW if window is None else int(window)
    min_periods = C.IC_Z_MIN if min_periods is None else int(min_periods)
    mu = factor.rolling(window, min_periods=min_periods).mean()
    sd = factor.rolling(window, min_periods=min_periods).std()
    return ((factor - mu) / sd.where(sd > 0)).clip(-3, 3)


def trail_z(raw: pd.DataFrame) -> pd.DataFrame:
    """入书用的标准化：252 日窗口、至少 120 日。"""
    return exante_z(raw, Z_WINDOW, Z_MIN)


def trailing_mean(factor: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """截至当日的滚动均值。窗口里有缺失就不给出值，避免用更早的数填平缺口。"""
    return factor.rolling(int(window), min_periods=int(window)).mean()


def average_signals(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """等权平均，NaN 不投票；全 NaN 的位置保持 NaN。"""
    if not frames:
        raise ValueError('组合里没有因子')
    idx, cols = frames[0].index, frames[0].columns
    for f in frames[1:]:
        idx, cols = idx.union(f.index), cols.union(f.columns)
    idx = idx.sort_values()
    acc = np.zeros((len(idx), len(cols)), dtype='float64')
    cnt = np.zeros((len(idx), len(cols)), dtype='float64')
    for f in frames:
        x = f.reindex(index=idx, columns=cols).to_numpy(dtype='float64')
        ok = np.isfinite(x)
        acc += np.where(ok, x, 0.0)
        cnt += ok
    out = np.full_like(acc, np.nan)
    good = cnt > 0
    out[good] = acc[good] / cnt[good]
    return pd.DataFrame(out, index=idx, columns=list(cols))


# --------------------------------------------------------------------------
# 装配
# --------------------------------------------------------------------------
@dataclass
class SignalSet:
    bars: dict[str, pd.DataFrame]
    signed: dict[str, pd.DataFrame] = field(default_factory=dict)
    unsigned: dict[str, pd.DataFrame] = field(default_factory=dict)
    family: dict[str, str] = field(default_factory=dict)
    vol: pd.DataFrame | None = None

    def raw(self, name: str) -> pd.DataFrame:
        """因子的原始值（已定向因子除回先验符号）。"""
        if name in self.signed:
            return self.signed[name] * SIGNED_PRIORS.get(name, 1)
        if name in self.unsigned:
            return self.unsigned[name]
        known = sorted({*self.signed, *self.unsigned})
        raise KeyError(f"没有因子 {name}。可选: {', '.join(known)}")


def trend_filtered(combo: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    """``combo × 1{sign(combo) = sign(trend)}``：只保留与趋势同向的时间因子观点。

    方向相反或任一为 0 时记 0（空仓，是一个观点）；任一缺失时是 NaN（没有观点）。
    """
    trend = trend.reindex(index=combo.index, columns=combo.columns)
    agree = np.sign(combo) == np.sign(trend)
    return combo.where(agree & (combo != 0), 0.0).where(combo.notna() & trend.notna())


def reversal_proxies(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """CLV：收盘价在当日区间里的位置（-1 最低，+1 最高）；ret_day：当日开到收。"""
    hi, lo = bars['highw'], bars['loww']
    clv = (2 * bars['closew'] - hi - lo) / (hi - lo).where(hi > lo)
    ret_day = (bars['closew'] - bars['openw']) / bars['open'].where(bars['open'] > 0)
    return {'clv': clv, 'ret_day': ret_day}


def assemble(bars: dict[str, pd.DataFrame],
             time_raw: dict[str, pd.DataFrame],
             universe: dict,
             external_partitions=('research',)) -> SignalSet:
    """用已经算好的时间戳/持续期原始值，装配全部日频信号。

    ``time_raw`` 不带方向。研究期由因子缓存提供；验证期和已结束的样本外窗口由同一套
    公式在允许的分钟数据上重算后传入。
    """
    symbols = list(bars['close'].columns)
    out = SignalSet(bars=bars)

    for n, s in C.FACTOR_SIGNS.items():
        if n not in time_raw:
            raise KeyError(f"缺少时间因子 {n}，请先在第 3 步构造")
        out.signed[n] = time_raw[n] * s
        out.family[n] = '时间戳' if n in C.TIMESTAMP_FACTORS else '持续期'
    out.signed['time_combo'] = average_signals(
        [trail_z(out.signed[n]) for n in C.FACTOR_SIGNS])
    out.family['time_combo'] = '时间戳+持续期'

    mp = B.multiplicative_prices(bars)
    tech_raw = daily.tech_indicators(mp['close'], mp['high'], mp['low'], bars['volume'])
    for n, s in C.TECH_PRIOR_SIGNS.items():
        out.signed[n] = tech_raw[n] * s
        out.family[n] = '量价'
    for n in C.TECH_UNSIGNED:
        out.unsigned[n] = tech_raw[n]
        out.family[n] = '量价(无方向)'

    close, closew = bars['close'], bars['closew']
    out.signed['tsmom'] = daily.tsmom(close, closew)
    out.signed['tsmom_20'] = daily.tsmom_sign(close, closew, window=20)
    out.signed['carry'] = daily.carry(close, closew)
    out.family['tsmom'] = out.family['tsmom_20'] = out.family['carry'] = '慢信号'
    out.signed['carry_roll'] = daily.carry_roll(
        close, closew, B.load_roll_calendar(symbols, close.index.max()))
    out.family['carry_roll'] = '慢信号'
    # 年化后不同换月节奏的品种可比，做截面排名：贴水越深越偏多
    out.signed['cs_carry_roll'] = daily.cross_sectional_rank(out.signed['carry_roll'], universe)
    out.family['cs_carry_roll'] = '慢信号(截面)'
    out.signed['ts_low_slow'] = trailing_mean(out.signed['ts_low'], window=20)
    out.family['ts_low_slow'] = '时间戳'
    out.signed['time_combo_trend'] = trend_filtered(out.signed['time_combo'], out.signed['tsmom'])
    out.family['time_combo_trend'] = '时间戳+持续期×趋势同向'

    for n, factor in daily.cross_sectional_momentum(close, closew, universe).items():
        out.unsigned[n] = factor
        out.family[n] = '截面动量(独立候选)'
    out.unsigned['return_skew_60'] = daily.rolling_return_skewness(close, closew, window=60)
    out.family['return_skew_60'] = '尾部风险(独立候选)'
    out.unsigned['cs_low_vol_60'] = daily.cross_sectional_low_volatility(
        close, closew, universe, window=60)
    out.family['cs_low_vol_60'] = '截面低波动(独立候选)'

    for name, factor in external.load_wide(symbols, external_partitions, close.index).items():
        out.unsigned[name] = factor
        out.family[name] = EXTERNAL_FAMILIES.get(name, '外部数据(候选)')
    # 米筐主力-次主力年化展期收益，正值 = 贴水（与 carry_roll 同向，按定义而非 IC 定向）
    if 'carry_main_sub_annualized' in out.unsigned:
        out.signed['carry_ms'] = out.unsigned['carry_main_sub_annualized']
        out.signed['cs_carry_ms'] = daily.cross_sectional_rank(out.signed['carry_ms'], universe)
        out.family['carry_ms'] = '外部期限结构'
        out.family['cs_carry_ms'] = '外部期限结构(截面)'

    rev = reversal_proxies(bars)
    out.signed['neg_clv'] = -rev['clv']
    out.signed['neg_ret_day'] = -rev['ret_day']
    out.family['neg_clv'] = out.family['neg_ret_day'] = '反转代理'

    out.vol = daily.daily_vol(close, closew)
    return out


def load(symbols: list[str]) -> SignalSet:
    """研究期信号。日频行情和因子缓存都停在 2021。"""
    time_raw = {
        n: cache.load_wide(n, C.IC_REFERENCE_LOOKBACK, C.IC_REFERENCE_PCT, symbols)
        for n in C.FACTOR_SIGNS
    }
    return assemble(B.load_daily_bars(symbols), time_raw, U.load_universe(), ('research',))
