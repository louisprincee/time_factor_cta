"""已退出自适应股指策略的研究期信号组件，供历史诊断复现使用。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.index_allocation import allocation_multipliers
from strategies.index_common import SYMBOLS, THRESHOLD, high_open_z
from tfcta import config as C

GROSS = 1.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
VOL_WINDOW = 20
VOL_TARGET = 0.20
EWMA_LAMBDA = 0.94
EWMA_MINIMUM = 20
ONE_SIDED_BODY_MAX = -0.7
ONE_SIDED_UP_BODY_MIN = 0.7
ONE_SIDED_UP_CLV_MIN = 0.8
RALLY_MULTIPLIER = 1.5
PRICE_FIELDS = ("open", "openw", "close", "closew", "highw", "loww")


def blended_vol_scale(open_to_open: pd.Series) -> pd.Series:
    done = open_to_open.shift(1)
    annual = np.sqrt(C.TRADING_DAYS_PER_YEAR)
    rolling = done.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * annual
    ewma = np.sqrt(
        done.pow(2).ewm(
            alpha=1.0 - EWMA_LAMBDA,
            adjust=False,
            min_periods=EWMA_MINIMUM,
        ).mean()
    ) * annual
    vol = pd.concat([rolling, ewma], axis=1).max(axis=1, skipna=False)
    return (VOL_TARGET / vol.where(vol > 0)).clip(upper=1.0).fillna(1.0)


def one_sided_down(
    bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> pd.Series:
    width = bars["highw"] - bars["loww"]
    body = (bars["closew"] - bars["openw"]) / width.where(width > 0)
    return body.mean(axis=1, skipna=False).reindex(index).le(ONE_SIDED_BODY_MAX)


def one_sided_up(
    bars: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    market_clv: pd.Series,
    body_min: float = ONE_SIDED_UP_BODY_MIN,
    clv_min: float = ONE_SIDED_UP_CLV_MIN,
) -> pd.Series:
    width = bars["highw"] - bars["loww"]
    body = (bars["closew"] - bars["openw"]) / width.where(width > 0)
    body_ratio = body.mean(axis=1, skipna=False).reindex(index)
    return body_ratio.gt(body_min) & market_clv.ge(clv_min)


def adaptive_targets(
    market_clv: pd.Series,
    bars: dict[str, pd.DataFrame],
    *,
    rally_z: pd.Series | None = None,
    one_sided_body_min: float = ONE_SIDED_UP_BODY_MIN,
    one_sided_clv_min: float = ONE_SIDED_UP_CLV_MIN,
) -> pd.DataFrame:
    index = market_clv.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("股指CLV索引必须为DatetimeIndex")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("股指信号日期必须唯一升序")
    if index.empty or index.min() < pd.Timestamp("2015-04-16"):
        raise C.HoldoutViolation("历史自适应诊断仅适用于2015年后的研究期")
    C.assert_no_holdout_dates(index, "历史自适应诊断")
    if market_clv.isna().any() or not market_clv.between(0, 1).all():
        raise ValueError("股指CLV缺失或越界")

    frames = {
        field: bars[field].reindex(index=index, columns=list(SYMBOLS))
        for field in PRICE_FIELDS
    }
    if any(frame.isna().any().any() for frame in frames.values()) or (
        frames["open"].le(0).any().any()
    ):
        raise ValueError("股指自适应诊断行情未完全对齐")
    if rally_z is None:
        rally_z = high_open_z(bars)
    if not rally_z.index.equals(index):
        raise ValueError("冲高因子必须与诊断日期对齐")

    open_to_open = (
        (frames["openw"].shift(-1) - frames["openw"]) / frames["open"]
    ).mean(axis=1, skipna=False)
    scale = blended_vol_scale(open_to_open)
    multipliers = allocation_multipliers(frames, "relative_return_rank_20d")
    tilt = pd.Series(
        np.where(rally_z.gt(0), RALLY_MULTIPLIER, 1.0), index=index
    )
    weak_close_long = market_clv.lt(THRESHOLD) & ~one_sided_down(frames, index)
    up_long = one_sided_up(
        frames, index, market_clv, one_sided_body_min, one_sided_clv_min
    )
    gross = (weak_close_long | up_long).astype(float) * scale * tilt
    return multipliers.shift(1).mul(gross.shift(1), axis=0).fillna(0.0)
