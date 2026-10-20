"""当天分数大于 0，且前一日年化展期收益大于 0，下一开盘做多。

规格在同目录的 market_clock_carry.json。名义敞口上限 1 倍。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tfcta import config as C

CONFIG_PATH = Path(__file__).with_name("market_clock_carry.json")
FACTORS = ("ts_low", "ts_high", "dfp_top3", "ts_low_pm", "ts_high_pm")
Z_WINDOW = 250
Z_MIN_PERIODS = 125
Z_CLIP = 3.0
MIN_FACTORS = 3
MIN_SYMBOLS = 10
ID = "market_clock_carry"
NAME = "当日分数且展期为正"
DETAIL = "五因子当天分数大于0且滞后一日年化展期收益大于0才等权做多/名义敞口1倍/回撤10%清仓并停10个交易日/下一开盘"
EXECUTION = "target"
GROSS = 1.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
ROLL_YIELD_END = pd.Timestamp("2025-12-31")


def load_config(path=CONFIG_PATH) -> dict:
    """规格必须事先写明，且不能带冻结标记。"""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    parameters = config.get("parameters") or {}
    sample = config.get("sample", "")
    if (
        config.get("id") != ID
        or config.get("status") != "exploratory_predeclared"
        or config.get("frozen_at")
        or config.get("chosen")
        or "不读取2022及2023年以后" not in sample
        or "不读 2026" not in sample
        or tuple(parameters.get("factors", ())) != FACTORS
        or parameters.get("roll_yield") != "annualized_yield"
        or parameters.get("roll_yield_lag_days") != 1
        or parameters.get("gross_exposure_limit") != GROSS
        or parameters.get("drawdown_breaker") != DRAWDOWN_LIMIT
        or parameters.get("halt_days") != HALT_DAYS
    ):
        raise ValueError("当日分数且展期为正的登记与实现不一致")
    return config


def pool_mask(universe: dict, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    pool = pd.DataFrame(False, index=index, columns=columns)
    for year, members in universe.items():
        pool.loc[index.year == int(year), [s for s in members if s in pool.columns]] = True
    return pool


def _own_history_z(frame: pd.DataFrame) -> pd.DataFrame:
    """每列只用自己过去 250 个有值观测（不含当天）的均值和标准差。"""
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for symbol in frame.columns:
        values = frame[symbol].dropna()
        prior = values.shift(1).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS)
        std = prior.std()
        z = (values - prior.mean()) / std.where(std > 0)
        out[symbol] = z.clip(-Z_CLIP, Z_CLIP).reindex(frame.index)
    return out


def market_score(signed: dict, universe: dict, index: pd.DatetimeIndex) -> pd.Series:
    """收盘时已知的全市场分数。输入已乘事前方向，越大越看多。"""
    missing = [name for name in FACTORS if name not in signed]
    if missing:
        raise KeyError(f"当日分数缺少因子 {missing}")
    columns = sorted({s for members in universe.values() for s in members})
    frames = {}
    for name in FACTORS:
        frame = signed[name].reindex(columns=columns)
        covered = frame.dropna(how="all").index
        if covered.empty or covered.max() < index.max() or covered.min() > index.min():
            raise ValueError(f"{name} 没有覆盖回测区间 {index.min().date()}..{index.max().date()}")
        frames[name] = _own_history_z(frame).reindex(index)
    stacked = np.stack([frames[name].to_numpy(dtype="float64") for name in FACTORS])
    count = np.isfinite(stacked).sum(axis=0)
    with np.errstate(invalid="ignore"):
        per_symbol = np.where(count >= MIN_FACTORS, np.nanmean(stacked, axis=0), np.nan)
    per_symbol = pd.DataFrame(per_symbol, index=index, columns=columns).where(
        pool_mask(universe, index, columns)
    )
    enough = per_symbol.notna().sum(axis=1) >= MIN_SYMBOLS
    return per_symbol.mean(axis=1).where(enough)


def window_return(bars: dict, columns, window: int) -> pd.DataFrame:
    """过去 window 日涨跌：复权收盘之差除以 window 日前的原始收盘。"""
    close = bars["close"].reindex(columns=columns)
    closew = bars["closew"].reindex(columns=columns)
    base = close.shift(window)
    return (closew - closew.shift(window)) / base.where(base > 0)


def _roll_yield_dirs(end: pd.Timestamp) -> list[Path]:
    """研究期在根目录。2022、2023–2025 在各自的分区里。2026 不读。"""
    base = C.COMMODITY_DATA / "external_rqdata"
    folders = [base / "roll_yield"]
    if end >= pd.Timestamp("2022-01-01"):
        folders.append(base / "validation_2022" / "roll_yield")
    if end >= pd.Timestamp("2023-01-01"):
        folders.append(base / "holdout_locked" / "roll_yield")
    return folders


def roll_yield(index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """主力对次主力的年化展期收益。晚于 2025 的行直接丢掉。"""
    end = pd.Timestamp(index.max())
    if end > ROLL_YIELD_END:
        raise C.HoldoutViolation("展期收益不读 2026")
    frames = {}
    for symbol in columns:
        parts = []
        for folder in _roll_yield_dirs(end):
            path = folder / f"{symbol}_main_sub.pkl"
            if not path.is_file():
                continue
            frame = pd.read_pickle(path)
            dates = pd.DatetimeIndex(frame.index.get_level_values(-1))
            keep = dates <= ROLL_YIELD_END
            series = pd.Series(frame["annualized_yield"].to_numpy()[keep], index=dates[keep])
            parts.append(series)
        if not parts:
            raise C.HoldoutViolation(f"{symbol} 没有展期收益")
        merged = pd.concat(parts)
        frames[symbol] = merged[~merged.index.duplicated(keep="last")].sort_index()
    return pd.DataFrame(frames).reindex(index)


def decision_weights(score: pd.Series, carry: pd.DataFrame) -> pd.DataFrame:
    """收盘决策。分数和已经滞后一天的展期收益都严格大于 0 才做多。"""
    allowed = carry.reindex(index=score.index).gt(0)
    side = score.gt(0).astype("float64")
    return pd.DataFrame(1.0, index=score.index, columns=carry.columns).mul(side, axis=0).where(allowed, 0.0)


def targets(signed: dict, bars: dict, universe: dict, index: pd.DatetimeIndex, columns,
            spec: dict | None = None, carry: pd.DataFrame | None = None) -> pd.DataFrame:
    """下一开盘的目标权重。展期收益先滞后一天，整行权重再后移到下一开盘。"""
    del bars, spec
    columns = pd.Index(columns)
    pool = pool_mask(universe, index, columns)
    score = market_score(signed, universe, index).reindex(index)
    lagged = (carry if carry is not None else roll_yield(index, columns)).reindex(index=index, columns=columns).shift(1)
    return decision_weights(score, lagged).shift(1).fillna(0.0).where(pool, 0.0)


load_config()
