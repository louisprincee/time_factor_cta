"""Small factor registry and explicit economically different signal/combination rules."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .. import config as C
from ..data import bars as B, universe as U
from . import daily, cache, external

Z_WINDOW, Z_MIN = 252, 120
SIGNED_PRIORS = {**C.FACTOR_SIGNS, "time_spread":1., "tsmom":1.,
                 "tsmom_20":1., "carry_ms":1., "cs_mom_ra_250":1., "neg_clv":1.}
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
    source = external.load_wide(list(close.columns),external_partitions,close.index)
    if "carry_main_sub_annualized" in source:
        result.signed["carry_ms"] = source["carry_main_sub_annualized"]
        result.family["carry_ms"] = "期限结构对照"
    mom = daily.momentum_components(close,adjusted,windows=(250,))["tsmom_ra_250"]
    result.signed["cs_mom_ra_250"] = daily.cross_sectional_rank(mom,universe)
    result.family["cs_mom_ra_250"] = "截面动量对照"
    high, low = bars["highw"],bars["loww"]
    result.signed["neg_clv"] = -(2*adjusted-high-low)/(high-low).where(high>low)
    result.family.update(tsmom="趋势对照",tsmom_20="趋势对照",neg_clv="价格位置对照")
    result.vol = daily.daily_vol(close,adjusted)
    return result


def load(symbols):
    panels = {s:cache.load_symbol(s,C.IC_REFERENCE_LOOKBACK,C.IC_REFERENCE_PCT) for s in symbols}
    raw = {n:pd.DataFrame({s:f[n] for s,f in panels.items()}) for n in C.FACTOR_SIGNS}
    return assemble(B.load_daily_bars(symbols),raw,U.load_universe())
