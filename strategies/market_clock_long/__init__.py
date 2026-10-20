"""只做多全市场时钟：研报五因子聚合成一个市场分数，看多时全池持有多头，否则空仓。

冻结参数在同目录的 market_clock_long.json（波动降仓 + 趋势帽 + 波动倒数权重），按目标仓位净额执行
（backtest.engine.simulate_target_lots）。研究期各变体登记在 config/research/market_clock_long_robustness.json，
由 research/market_clock_long.py 运行；冻结后的逐年测试用 python -m backtest.engine --year。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATH = Path(__file__).with_name("market_clock_long.json")
ID = "market_clock_long"
NAME = "只做多全市场时钟"
DETAIL = "五因子自身z/池内平均/大于0看多否则空仓/5日看多占比×波动降仓×趋势帽×波动倒数权重/净额执行"
EXECUTION = "target"
FACTORS = ("ts_low", "ts_high", "dfp_top3", "ts_low_pm", "ts_high_pm")
Z_WINDOW = 250
Z_MIN_PERIODS = 125
Z_CLIP = 3.0
MIN_FACTORS = 3
MIN_SYMBOLS = 10
HOLDING_DAYS = 5
VOL_SHORT, VOL_LONG, VOL_MIN = 20, 250, 120
TREND_WINDOW, TREND_MIN, TREND_CUT = 250, 120, 0.5
INV_VOL_WINDOW, INV_VOL_MIN, WEIGHT_CAP = 60, 40, 3.0
SPEC_KEYS = ("score", "factors", "z_window", "holding_days", "vol_scale", "trend_cap", "weights")
BASE = {
    "score": "time5", "factors": FACTORS, "z_window": Z_WINDOW, "holding_days": HOLDING_DAYS,
    "vol_scale": False, "trend_cap": False, "weights": "equal",
}


def load_config(path=CONFIG_PATH) -> dict:
    """冻结必须同时写明 frozen_at 和 chosen，chosen 等于 parameters，并且和代码常量一致。"""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    chosen = config.get("chosen") or {}
    if (
        config.get("id") != ID
        or config.get("status") != "frozen"
        or not config.get("frozen_at")
        or chosen != config.get("parameters")
        or "不读 2026" not in config.get("sample", "")
        or chosen.get("direction") != "long_only"
        or tuple(chosen.get("factors", ())) != FACTORS
        or chosen.get("z_window") != Z_WINDOW
        or chosen.get("holding_days") != HOLDING_DAYS
        or chosen.get("gross_exposure_limit") != 1.5
        or chosen.get("drawdown_breaker") != 0.2
    ):
        raise ValueError("只做多全市场时钟冻结文件与实现不一致")
    return config


def chosen_spec() -> dict:
    chosen = load_config()["chosen"]
    return {key: tuple(chosen[key]) if key == "factors" else chosen[key] for key in SPEC_KEYS}


def own_history_z(frame: pd.DataFrame, window: int = Z_WINDOW) -> pd.DataFrame:
    """每列只用自己过去 window 个有值观测（不含当天）的均值和标准差，至少一半窗口。"""
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    minimum = Z_MIN_PERIODS if window == Z_WINDOW else window // 2
    for symbol in frame.columns:
        values = frame[symbol].dropna()
        prior = values.shift(1).rolling(window, min_periods=minimum)
        std = prior.std()
        z = (values - prior.mean()) / std.where(std > 0)
        out[symbol] = z.clip(-Z_CLIP, Z_CLIP).reindex(frame.index)
    return out


def pool_mask(universe: dict, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    pool = pd.DataFrame(False, index=index, columns=columns)
    for year, members in universe.items():
        pool.loc[index.year == int(year), [s for s in members if s in pool.columns]] = True
    return pool


def close_location(bars: dict) -> pd.DataFrame:
    """收盘在当日区间里的位置，−1 在最低、+1 在最高。复权价之差，不做比值。"""
    high, low, close = bars["highw"], bars["loww"], bars["closew"]
    span = (high - low).where(high > low)
    return (2 * close - high - low) / span


def market_score(signed: dict, universe: dict, index: pd.DatetimeIndex,
                 factors=FACTORS, window: int = Z_WINDOW) -> pd.Series:
    """收盘时已知的全市场分数。输入已乘事前方向，越大越看多。"""
    missing = [name for name in factors if name not in signed]
    if missing:
        raise KeyError(f"全市场时钟缺少因子 {missing}")
    columns = sorted({s for members in universe.values() for s in members})
    frames = {}
    for name in factors:
        frame = signed[name].reindex(columns=columns)
        covered = frame.dropna(how="all").index
        if covered.empty or covered.max() < index.max() or covered.min() > index.min():
            raise ValueError(f"{name} 没有覆盖回测区间 {index.min().date()}..{index.max().date()}")
        frames[name] = own_history_z(frame, window).reindex(index)
    stacked = np.stack([frames[name].to_numpy(dtype="float64") for name in factors])
    count = np.isfinite(stacked).sum(axis=0)
    with np.errstate(invalid="ignore"):
        per_symbol = np.where(
            count >= min(MIN_FACTORS, len(factors)), np.nanmean(stacked, axis=0), np.nan,
        )
    per_symbol = pd.DataFrame(per_symbol, index=index, columns=columns).where(
        pool_mask(universe, index, columns)
    )
    enough = per_symbol.notna().sum(axis=1) >= MIN_SYMBOLS
    return per_symbol.mean(axis=1).where(enough)


def pool_return(bars: dict, universe: dict, index: pd.DatetimeIndex) -> pd.Series:
    """池内收盘到收盘收益的等权平均，收盘时已知。"""
    columns = sorted({s for members in universe.values() for s in members})
    close, closew = bars["close"].reindex(columns=columns), bars["closew"].reindex(columns=columns)
    prior = close.shift(1)
    daily = ((closew - closew.shift(1)) / prior.where(prior > 0)).reindex(index)
    return daily.where(pool_mask(universe, index, columns)).mean(axis=1)


def account_multiplier(bars: dict, universe: dict, index: pd.DatetimeIndex, spec: dict) -> pd.Series:
    """收盘时的账户乘数：波动放大时降仓、长期下跌时减半，只降不升。"""
    market = pool_return(bars, universe, index)
    multiplier = pd.Series(1.0, index=index)
    if spec.get("vol_scale"):
        short = market.rolling(VOL_SHORT, min_periods=VOL_SHORT).std()
        long = market.rolling(VOL_LONG, min_periods=VOL_MIN).std()
        multiplier *= (long / short.where(short > 0)).clip(upper=1.0).fillna(1.0)
    if spec.get("trend_cap"):
        trend = market.rolling(TREND_WINDOW, min_periods=TREND_MIN).sum()
        multiplier *= np.where(trend.lt(0), TREND_CUT, 1.0)
    return multiplier


def relative_weights(bars: dict, universe: dict, index: pd.DatetimeIndex, columns, spec: dict) -> pd.DataFrame:
    """收盘时的品种相对权重，池内平均为 1。等权时全为 1。"""
    pool = pool_mask(universe, index, columns)
    if spec.get("weights", "equal") == "equal":
        return pool.astype("float64")
    if spec["weights"] != "inverse_vol":
        raise ValueError("权重只能是 equal 或 inverse_vol")
    close, closew = bars["close"].reindex(columns=columns), bars["closew"].reindex(columns=columns)
    prior = close.shift(1)
    daily = (closew - closew.shift(1)) / prior.where(prior > 0)
    inverse = 1.0 / daily.rolling(INV_VOL_WINDOW, min_periods=INV_VOL_MIN).std().reindex(index)
    inverse = inverse.where(pool & np.isfinite(inverse))
    inverse = inverse.T.fillna(inverse.mean(axis=1)).T.where(pool)
    weights = inverse.div(inverse.mean(axis=1), axis=0).clip(upper=WEIGHT_CAP)
    return weights.where(pool, 0.0).fillna(0.0)


def score_for(spec: dict, signed: dict, bars: dict, universe: dict, index) -> pd.Series:
    if spec["score"] == "time5":
        return market_score(signed, universe, index, tuple(spec["factors"]), spec["z_window"])
    if spec["score"] == "clv":
        return market_score({"clv": -close_location(bars)}, universe, index, ("clv",), spec["z_window"])
    raise ValueError("分数只能是 time5 或 clv")


def targets(signed: dict, bars: dict, universe: dict, index: pd.DatetimeIndex, columns,
            spec: dict | None = None) -> pd.DataFrame:
    """执行日开盘的目标权重：前一收盘的看多占比 × 账户乘数 × 相对权重，只在看多占比变化时重算。

    不给 spec 时用冻结参数；给了就在等权主规格上覆盖，研究脚本用它跑登记变体。
    """
    spec = chosen_spec() if spec is None else {**BASE, **spec}
    columns = pd.Index(columns)
    pool = pool_mask(universe, index, columns)
    bullish = score_for(spec, signed, bars, universe, index).gt(0)
    side = pool.mul(bullish.astype("float64"), axis=0)
    fraction = side.rolling(spec["holding_days"], min_periods=1).mean()
    desired = fraction.mul(account_multiplier(bars, universe, index, spec), axis=0)
    desired = desired * relative_weights(bars, universe, index, columns, spec)
    changed = fraction.ne(fraction.shift(1))
    held = desired.where(changed).ffill().where(fraction.gt(0), 0.0).fillna(0.0)
    return held.shift(1).fillna(0.0).where(pool, 0.0)


load_config()
