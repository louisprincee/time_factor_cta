"""股指动态品种预算计算。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.index_close_location import SYMBOLS, close_vol_scale
from tfcta import config as C

LOOKBACK = 20
WEIGHT_SUM = 3.0
MAX_MULTIPLIER = 2.0
METHODS = ("relative_return_rank_20d", "inverse_volatility_20d")


def _cap_weights(raw: np.ndarray) -> np.ndarray:
    if raw.shape != (len(SYMBOLS),) or not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("动态股指预算必须是三个正的有限分值")
    weights = raw / raw.sum() * WEIGHT_SUM
    capped = weights > MAX_MULTIPLIER
    if capped.any():
        weights[capped] = MAX_MULTIPLIER
        remaining = ~capped
        residual = WEIGHT_SUM - weights[capped].sum()
        if not remaining.any() or residual <= 0:
            raise ValueError("动态股指预算无法在集中度上限内归一化")
        weights[remaining] = raw[remaining] / raw[remaining].sum() * residual
    if not np.isclose(weights.sum(), WEIGHT_SUM) or (
        weights > MAX_MULTIPLIER + 1e-12
    ).any():
        raise ValueError("动态股指预算归一化或集中度限制失败")
    return weights


def allocation_multipliers(
    bars: dict[str, pd.DataFrame], method: str
) -> pd.DataFrame:
    """用收盘时已知的20日品种收益或波动分配总乘数3。"""
    if method not in METHODS:
        raise ValueError(f"未知动态品种分配方法：{method}")
    index = bars["close"].index
    close = bars["close"].reindex(index=index, columns=SYMBOLS)
    closew = bars["closew"].reindex(index=index, columns=SYMBOLS)
    if (
        close.isna().any().any()
        or closew.isna().any().any()
        or close.le(0).any().any()
        or not np.isfinite(closew.to_numpy(dtype="float64")).all()
    ):
        raise ValueError("动态股指预算缺少有效收盘行情")

    equal = pd.DataFrame(1.0, index=index, columns=SYMBOLS)
    if method == "relative_return_rank_20d":
        relative_return = (closew - closew.shift(LOOKBACK)) / close.shift(LOOKBACK)
        ranks = relative_return.rank(axis=1, method="average")
        weights = ranks.mul(WEIGHT_SUM, axis=0).div(
            ranks.sum(axis=1).replace(0, np.nan), axis=0
        )
        ready = relative_return.notna().all(axis=1)
        result = weights.where(ready, equal)
    else:
        close_returns = (closew - closew.shift(1)) / close.shift(1)
        realized_vol = close_returns.rolling(
            LOOKBACK, min_periods=LOOKBACK
        ).std()
        ready = realized_vol.notna().all(axis=1)
        if (realized_vol.loc[ready] <= 0).any().any():
            raise ValueError("20日股指已实现波动率必须严格为正")
        inverse_vol = 1.0 / realized_vol
        result = equal.copy()
        for day in index[ready]:
            result.loc[day] = _cap_weights(
                inverse_vol.loc[day, list(SYMBOLS)].to_numpy(dtype="float64")
            )

    if (
        result.isna().any().any()
        or not np.isfinite(result.to_numpy(dtype="float64")).all()
        or not np.allclose(result.sum(axis=1).to_numpy(), WEIGHT_SUM)
        or (result > MAX_MULTIPLIER + 1e-12).any().any()
    ):
        raise ValueError(f"{method}股指品种预算不满足总和或集中度约束")
    return result


def target_weights(
    method: str,
    market_clv: pd.Series,
    bars: dict[str, pd.DataFrame],
    *,
    rally_z: pd.Series,
) -> pd.DataFrame:
    """把动态品种预算应用于原冻结策略的共同信号和总风险预算。"""
    index = market_clv.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("股指CLV索引必须为DatetimeIndex")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("股指信号日期必须唯一升序")
    oos_mask = index >= pd.Timestamp(C.OOS_START)
    if oos_mask.any() and index[oos_mask].max() > pd.Timestamp(C.DEFAULT_OOS_END):
        raise C.HoldoutViolation("股指动态预算不读 2026")
    if market_clv.isna().any() or not market_clv.between(0, 1).all():
        raise ValueError("股指CLV缺失或越界")

    all_bars = {
        field: bars[field].reindex(index=index, columns=SYMBOLS)
        for field in ("open", "openw", "close", "closew", "highw", "loww")
    }
    if any(frame.isna().any().any() for frame in all_bars.values()):
        raise ValueError("股指动态预算行情未完全对齐")
    if not rally_z.index.equals(index):
        raise ValueError("冲高因子必须与动态预算日期对齐")

    if oos_mask.any():
        research_index, oos_index = index[~oos_mask], index[oos_mask]
        if research_index.empty or oos_index.empty:
            raise C.HoldoutViolation("OOS动态预算必须包含研究期预热和样本外行情")
        research_bars = {
            field: frame.reindex(research_index) for field, frame in all_bars.items()
        }
        oos_bars = {field: frame.reindex(oos_index) for field, frame in all_bars.items()}
        multipliers = pd.concat(
            [
                allocation_multipliers(research_bars, method),
                allocation_multipliers(oos_bars, method),
            ]
        )
        # 复权价基准在分区间独立；动态分配预热只用 OOS 自身历史。
        multipliers.loc[research_index[-1]] = 1.0
        research_returns = (
            (research_bars["openw"].shift(-1) - research_bars["openw"])
            / research_bars["open"]
        ).mean(axis=1, skipna=False)
        oos_returns = (
            (oos_bars["openw"].shift(-1) - oos_bars["openw"])
            / oos_bars["open"]
        ).mean(axis=1, skipna=False)
        scale = pd.concat(
            [
                close_vol_scale(research_returns).fillna(1.0),
                close_vol_scale(oos_returns).fillna(1.0),
            ]
        )
        # 最后一研究日的波动缩放只用研究期已经完成的开盘段，不跨复权断点。
        # 它决定下一开盘（第一个样本外交易日）的预算，不能改写成 1。
    else:
        multipliers = allocation_multipliers(all_bars, method)
        open_returns = (
            (all_bars["openw"].shift(-1) - all_bars["openw"])
            / all_bars["open"]
        ).mean(axis=1, skipna=False)
        scale = close_vol_scale(open_returns).fillna(1.0)

    gross = (
        market_clv.lt(0.4).astype(float)
        * scale
        * rally_z.gt(0).map({True: 1.5, False: 1.0})
    )
    # 乘数合计为 3，当日预算是信号、冲高和波动缩放的乘积。
    # 账本按 目标权重 × 敞口 × 权益 / 3 分配名义金额，这里不能再除以 3。
    budget = gross.shift(1)
    targets = multipliers.shift(1).mul(budget, axis=0)
    if not np.allclose(
        targets.sum(axis=1, skipna=False).to_numpy(dtype="float64"),
        (budget * WEIGHT_SUM).to_numpy(dtype="float64"),
        equal_nan=True,
    ):
        raise ValueError("动态股指目标权重合计必须等于当日预算乘以 3")
    return targets.fillna(0.0)
