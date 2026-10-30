"""冻结熊市空头的变体：弱收盘日只在“扣掉跌幅和收盘位置后冲高仍偏高”时做多。探索期，未冻结，不能打开样本外。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from strategies import commodity_original as T
from strategies.index_bear_reversal import (
    DRAWDOWN_LIMIT, GROSS, HALT_DAYS, THRESHOLD, VOL_RULE, _open_to_open, close_vol_scale, decision_side,
)
from strategies.index_close_location import SYMBOLS

ID = "index_bear_short_rally_residual"
NAME = "股指熊市空头·冲高残差择多"
CONFIG_PATH = Path(__file__).with_name(f"{ID}.json")
STATUS = "exploratory_predeclared"
FEATURE_WINDOW = 60
FEATURE_MINIMUM = 40
RESIDUAL_MINIMUM = 60
RESIDUAL_CUT = 0.0


def declared_config(path: Path = CONFIG_PATH) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    p = config.get("parameters", {})
    sample = config.get("sample", "")
    if (
        config.get("id") != ID
        or config.get("status") != STATUS
        or config.get("frozen_at")
        or p.get("instruments") != list(SYMBOLS)
        or p.get("trend_windows") != list(T.TREND_WINDOWS)
        or p.get("base_book") != "index_close_location_bear_short"
        or p.get("weak_close") != f"Equal-weight close location < {THRESHOLD}"
        or p.get("feature_window") != FEATURE_WINDOW
        or p.get("feature_minimum") != FEATURE_MINIMUM
        or p.get("residual_minimum_days") != RESIDUAL_MINIMUM
        or p.get("residual_cut") != RESIDUAL_CUT
        or p.get("gross_exposure_limit") != GROSS
        or p.get("drawdown_breaker") != DRAWDOWN_LIMIT
        or p.get("halt_days") != HALT_DAYS
        or p.get("position_scale") != VOL_RULE
        or not p.get("execution", "").startswith("l1_30m")
        or "不读 2022" not in sample
        or "不读 2026" not in sample
    ):
        raise ValueError("冲高残差策略登记与实现不一致")
    return config


def _z(raw: pd.Series) -> pd.Series:
    prior = raw.shift(1)
    mean = prior.rolling(FEATURE_WINDOW, min_periods=FEATURE_MINIMUM).mean()
    scale = prior.rolling(FEATURE_WINDOW, min_periods=FEATURE_MINIMUM).std()
    return (raw - mean) / scale.where(scale > 0)


def day_features(bars: dict[str, pd.DataFrame], symbols=SYMBOLS) -> pd.DataFrame:
    """信号日收盘可知：冲高 (高−开)/开 与开盘到收盘涨跌 (收−开)/开，逐品种 60 日 z 后等权。任一品种缺则为缺。"""
    rally, ret = {}, {}
    for symbol in symbols:
        opened = bars["open"][symbol]
        rally[symbol] = _z((bars["highw"][symbol] - bars["openw"][symbol]) / opened)
        ret[symbol] = _z((bars["closew"][symbol] - bars["openw"][symbol]) / opened)
    return pd.DataFrame({
        "rally_z": pd.DataFrame(rally).mean(axis=1, skipna=False),
        "return_z": pd.DataFrame(ret).mean(axis=1, skipna=False),
    })


def rally_residual(features: pd.DataFrame, clv: pd.Series) -> pd.Series:
    """弱收盘日：用此前弱收盘日（不含当天）拟合 冲高_z ~ 1 + 涨跌_z + CLV，返回当天残差。非弱收盘日或历史不足为缺。"""
    clv = clv.reindex(features.index)
    weak = clv.lt(THRESHOLD)
    design = np.column_stack([np.ones(len(clv)), features["return_z"].to_numpy(), clv.to_numpy()])
    target = features["rally_z"].to_numpy()
    usable = weak.to_numpy() & np.isfinite(design).all(axis=1) & np.isfinite(target)
    out = pd.Series(np.nan, index=features.index)
    rows = np.flatnonzero(usable)
    for k, row in enumerate(rows):
        if k < RESIDUAL_MINIMUM:
            continue
        past = rows[:k]
        beta = np.linalg.lstsq(design[past], target[past], rcond=None)[0]
        out.iloc[row] = target[row] - design[row] @ beta
    return out


def long_weight(residual: pd.Series) -> pd.Series:
    """残差 > 0 做多，≤ 0 空仓，未知时沿用冻结规则做多。只在弱收盘日起作用。"""
    return pd.Series(np.where(residual.le(RESIDUAL_CUT), 0.0, 1.0), index=residual.index)


def weighted_targets(side: pd.Series, scale: pd.Series, weight, symbols=SYMBOLS) -> pd.DataFrame:
    """收盘方向 × 波动乘数，多头再乘 weight（标量或序列）；下一开盘执行。"""
    weight = pd.Series(weight, index=side.index) if np.isscalar(weight) else weight.reindex(side.index)
    weighted = side * scale.reindex(side.index).fillna(1.0) * weight.where(side > 0, 1.0)
    return pd.DataFrame({symbol: weighted for symbol in symbols}, index=side.index).shift(1).fillna(0.0)


def residual_targets(market_clv: pd.Series, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    side = decision_side(market_clv, bars)
    scale = close_vol_scale(_open_to_open(bars, side.index))
    residual = rally_residual(day_features(bars).reindex(side.index), market_clv)
    return weighted_targets(side, scale, long_weight(residual))


class Book:
    ID = ID
    NAME = NAME
    SYMBOLS = SYMBOLS
    GROSS = GROSS
    DRAWDOWN_LIMIT = DRAWDOWN_LIMIT
    HALT_DAYS = HALT_DAYS
    STATUS = STATUS
    CONFIG_PATH = CONFIG_PATH
    DETAIL = "同冻结熊市空头；弱收盘日只在扣掉当天涨跌和收盘位置后冲高仍偏高时做多，否则空仓"

    @staticmethod
    def declared() -> dict:
        return declared_config()

    @staticmethod
    def targets(market_clv: pd.Series, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return residual_targets(market_clv, bars)


BOOK = Book()
BOOK.declared()
__all__ = ["BOOK", "day_features", "long_weight", "rally_residual", "residual_targets", "weighted_targets"]
