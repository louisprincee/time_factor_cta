"""唯一股指冻结策略：弱收盘做多、冲高日加仓、固定品种份额，无空头。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tfcta import config as C

SYMBOLS = ("IF", "IH", "IC")
ID = "index_clv_rally_fixed_long"
NAME = "股指弱收盘冲高固定权重多头"
THRESHOLD = 0.4
FIXED_MULTIPLIERS = (0.891744548286604, 0.623831775700935, 1.484423676012461)
GROSS = 1.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
VOL_WINDOW = 20
VOL_TARGET = 0.20
CONFIG_PATH = Path(__file__).with_name(f"{ID}.json")
PARAMETERS = {
    "instruments": list(SYMBOLS), "clv_threshold": THRESHOLD,
    "fixed_multipliers": dict(zip(SYMBOLS, FIXED_MULTIPLIERS)),
    "short_rule": "none", "rally_window": 60, "rally_minimum": 40,
    "rally_gate": 0.0, "rally_multiplier": 1.5,
    "vol_window": VOL_WINDOW, "vol_target": VOL_TARGET, "vol_scale_cap": 1.0,
    "gross_exposure_limit": GROSS, "drawdown_breaker": DRAWDOWN_LIMIT, "halt_days": HALT_DAYS,
    "initial_capital": 10000000, "execution": "l1_30m", "open_limits": True,
    "signal_execution": "close t -> open t+1", "rebalance": "only target change; whole lots",
}


def declared_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (config.get("id") != ID or config.get("status") != "frozen" or not config.get("frozen_at")
            or config.get("parameters") != PARAMETERS or config.get("chosen") != PARAMETERS):
        raise ValueError("股指唯一冻结规则与实现不一致")
    sample = config.get("sample", "")
    if "2015-04-16" not in sample or "尚未回测" not in sample or "不读 2026" not in sample:
        raise ValueError("股指冻结规则缺少样本边界")
    return config


def high_open_z(bars, symbols=SYMBOLS):
    """冲高以此前60日均值/标准差归一化，至少40日，不含当前日。"""
    high_open = (bars["highw"] - bars["openw"]) / bars["open"].where(bars["open"] > 0) * 1e4
    high_open = high_open.reindex(columns=list(symbols))
    prior = high_open.shift(1)
    mean = prior.rolling(60, min_periods=40).mean()
    scale = prior.rolling(60, min_periods=40).std()
    z = (high_open - mean) / scale.where(scale > 0)
    return z.where(np.isfinite(z)).mean(axis=1, skipna=False).rename("high_open_z")


def close_vol_scale(open_to_open):
    """收盘时仅用截至当日开盘已经完成的20段收益。"""
    rv = open_to_open.shift(1).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(C.TRADING_DAYS_PER_YEAR)
    return (VOL_TARGET / rv).clip(upper=1.0)


def fixed_long_targets(market_clv, bars, *, rally_z=None):
    if not isinstance(market_clv.index, pd.DatetimeIndex):
        raise TypeError("市场CLV索引必须为DatetimeIndex")
    if market_clv.index.has_duplicates or not market_clv.index.is_monotonic_increasing:
        raise ValueError("信号日期必须唯一升序")
    if market_clv.isna().any() or not market_clv.between(0, 1).all():
        raise ValueError("市场CLV缺失或越界")
    index = market_clv.index
    opened = bars["open"].reindex(index=index, columns=list(SYMBOLS))
    adjusted = bars["openw"].reindex(index=index, columns=list(SYMBOLS))
    if opened.isna().any().any() or adjusted.isna().any().any() or opened.le(0).any().any():
        raise ValueError("波动缩放缺少完整正开盘价")
    returns = ((adjusted.shift(-1)-adjusted)/opened).mean(axis=1, skipna=False)
    scale = close_vol_scale(returns).fillna(1.0)
    if rally_z is None:
        rally_z = high_open_z(bars)
    tilt = pd.Series(np.where(rally_z.reindex(index).gt(0), 1.5, 1.0), index=index)
    weighted = market_clv.lt(THRESHOLD).astype(float) * scale * tilt
    return pd.DataFrame({s: weighted*m for s,m in zip(SYMBOLS,FIXED_MULTIPLIERS)}, index=index).shift(1).fillna(0.0)


class Book:
    ID, NAME, SYMBOLS = ID, NAME, SYMBOLS
    GROSS, DRAWDOWN_LIMIT, HALT_DAYS = GROSS, DRAWDOWN_LIMIT, HALT_DAYS
    STATUS, CONFIG_PATH = "frozen", CONFIG_PATH
    DETAIL = "等权CLV<0.4次日做多；冲高z>0加仓1.5倍；固定IF/IH/IC约29.7/20.8/49.5%；无空头"
    declared = staticmethod(declared_config)
    frozen = staticmethod(declared_config)
    targets = staticmethod(fixed_long_targets)

    @property
    def FILES(self):
        return [Path(__file__), CONFIG_PATH]


BOOK = Book()


def market_close_location(
    bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> pd.Series:
    """IF/IH/IC 等权收盘位置；必须三品种都有有效日内区间。"""
    close = bars["closew"].reindex(index=index, columns=SYMBOLS)
    high = bars["highw"].reindex(index=index, columns=SYMBOLS)
    low = bars["loww"].reindex(index=index, columns=SYMBOLS)
    width = high - low
    if close.isna().any().any() or high.isna().any().any() or low.isna().any().any():
        raise ValueError("股指收盘位置行情未完全对齐")
    if width.le(0).any().any():
        raise ValueError("股指日内最高价必须高于最低价")
    clv = (close - low) / width
    if (
        not np.isfinite(clv.to_numpy()).all()
        or clv.lt(0).any().any()
        or clv.gt(1).any().any()
    ):
        raise ValueError("股指收盘位置超出 [0, 1] 或包含无效值")
    return clv.mean(axis=1, skipna=False).rename("market_clv")
