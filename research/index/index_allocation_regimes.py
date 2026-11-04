"""分析股指信号分歧、近期相对收益与既有等权账本差异。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import (
    INDEX_RESEARCH_START,
    INDEX_RESEARCH_END,
    INDEX_SYMBOLS,
    _load_index_research_inputs,
    _simulate_index_targets,
    max_drawdown,
    newey_west_t,
    performance,
    sharpe_ratio,
)
from research.time_factor_screen import ols_hac
from research.index.index_execution_10m import opening_quotes, l1_slippage
from strategies.index_common import (
    SYMBOLS,
    close_vol_scale,
    high_open_z,
    market_close_location,
)
from strategies.index_allocation import allocation_multipliers
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_allocation_regimes.json"
OUT = C.study_dir("index", "index_allocation_regimes")
BOOK_IDS = (
    "index_clv_rally_fixed_long",
    "index_clv_pooled_equal_long",
    "index_clv_individual_equal_long",
)
YEARS = (2018, 2021)
CAPITAL = 10_000_000.0
FIXED_MULTIPLIERS = (0.891744548286604, 0.623831775700935, 1.484423676012461)
GROSS = 1.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
MOMENTUM_LOOKBACK = 20
CORRELATION_LOOKBACK = 60
CORRELATION_MINIMUM = 45
HAC_LAGS = 5
SHRINKAGE_LEVELS = (0.0, 0.5, 1.0)
ALLOCATION_LOOKBACK = 20
ALLOCATION_WEIGHT_SUM = 3.0
ALLOCATION_MAX_MULTIPLIER = 2.0
ALLOCATION_METHODS = (
    "fixed_current",
    "equal",
    "relative_return_rank_20d",
    "inverse_volatility_20d",
)


def _individual_close_locations(bars, index):
    close = bars["closew"].reindex(index=index, columns=SYMBOLS)
    high = bars["highw"].reindex(index=index, columns=SYMBOLS)
    low = bars["loww"].reindex(index=index, columns=SYMBOLS)
    width = high - low
    if close.isna().any().any() or width.isna().any().any() or width.le(0).any().any():
        raise ValueError("股指单品种收盘位置行情未完全对齐")
    values = (close - low) / width
    if not np.isfinite(values.to_numpy()).all() or values.lt(0).any().any() or values.gt(1).any().any():
        raise ValueError("股指单品种收盘位置无效")
    return values


def _legacy_targets(market_clv, bars, rally_z, multipliers):
    """为历史配置研究复现已退役书的固定信号与风险预算。"""
    index = market_clv.index
    opened = bars["open"].reindex(index=index, columns=SYMBOLS)
    adjusted = bars["openw"].reindex(index=index, columns=SYMBOLS)
    open_returns = ((adjusted.shift(-1) - adjusted) / opened).mean(axis=1, skipna=False)
    scale = close_vol_scale(open_returns).fillna(1.0)
    tilt = pd.Series(np.where(rally_z.reindex(index).gt(0), 1.5, 1.0), index=index)
    budget = market_clv.lt(0.4).astype(float) * scale * tilt
    return pd.DataFrame(
        {symbol: budget * multiplier for symbol, multiplier in zip(SYMBOLS, multipliers)},
        index=index,
    ).shift(1).fillna(0.0)


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    parameters = protocol.get("parameters", {})
    shrinkage = parameters.get("allocation_shrinkage", {})
    allocation = parameters.get("allocation_backtests", {})
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or parameters.get("symbols") != list(SYMBOLS)
        or parameters.get("study_years") != list(YEARS)
        or parameters.get("strategy_ids") != list(BOOK_IDS)
        or parameters.get("clv_threshold") != 0.4
        or parameters.get("relative_momentum_lookback") != MOMENTUM_LOOKBACK
        or parameters.get("pairwise_return_correlation_lookback") != CORRELATION_LOOKBACK
        or parameters.get("pairwise_return_correlation_minimum") != CORRELATION_MINIMUM
        or parameters.get("hac_lags") != HAC_LAGS
        or shrinkage.get("levels") != list(SHRINKAGE_LEVELS)
        or shrinkage.get("preserve_total_multiplier") != 3.0
        or shrinkage.get("fixed_multipliers")
        != dict(zip(SYMBOLS, FIXED_MULTIPLIERS))
        or shrinkage.get("equal_multipliers") != dict.fromkeys(SYMBOLS, 1.0)
        or allocation.get("lookback_sessions") != ALLOCATION_LOOKBACK
        or allocation.get("weight_sum") != ALLOCATION_WEIGHT_SUM
        or allocation.get("max_multiplier") != ALLOCATION_MAX_MULTIPLIER
        or allocation.get("execution_lag_sessions") != 1
        or allocation.get("methods") != list(ALLOCATION_METHODS)
        or allocation.get("signal_book")
        != "index_clv_rally_fixed_long pooled CLV<0.4 signal"
        or allocation.get("relative_return_definition")
        != "(closew_t - closew_(t-20)) / raw_close_(t-20), ranked cross-sectionally; "
        "multipliers are 3*rank/sum(rank), average ranks for ties"
        or allocation.get("volatility_definition")
        != "sample standard deviation of the last 20 close-to-close returns through signal "
        "close t; multipliers proportional to inverse volatility"
        or allocation.get("concentration_cap_rule")
        != "cap each multiplier at 2 and redistribute excess proportionally across uncapped "
        "symbols while preserving total weight 3"
        or allocation.get("warmup_rule")
        != "use equal multipliers until the 20-session allocation estimator is available"
        or any(phrase not in sample for phrase in ("不读 2022", "不读 2023–2025", "不读 2026"))
    ):
        raise ValueError("股指跨品种配置与样本边界不匹配")
    return protocol


def _read_book(book_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = C.backtest_dir("index", book_id, "research")
    daily = pd.read_csv(root / "daily.csv", parse_dates=["交易日"])
    positions = pd.read_csv(root / "positions.csv", parse_dates=["交易日"])
    annual = pd.read_csv(root / "annual.csv")
    if daily["交易日"].dt.year.max() > 2021 or positions["交易日"].dt.year.max() > 2021:
        raise C.HoldoutViolation(f"{book_id}账本包含研究期外日期")
    return daily.set_index("交易日"), positions, annual


def _forward_returns(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    segment = (bars["openw"].shift(-1) - bars["openw"]) / bars["open"]
    return segment.shift(-1)


def _features(bars: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = bars["close"].index
    clv = _individual_close_locations(bars, index)
    pooled_clv = market_close_location(bars, index)
    close_returns = (bars["closew"] - bars["closew"].shift(1)) / bars["close"].shift(1)
    rolling_corr = []
    for left, right in ((SYMBOLS[0], SYMBOLS[1]), (SYMBOLS[0], SYMBOLS[2]), (SYMBOLS[1], SYMBOLS[2])):
        rolling_corr.append(
            close_returns[left]
            .rolling(CORRELATION_LOOKBACK, min_periods=CORRELATION_MINIMUM)
            .corr(close_returns[right])
        )
    forward = _forward_returns(bars)
    momentum = (
        bars["closew"] - bars["closew"].shift(MOMENTUM_LOOKBACK)
    ) / bars["close"].shift(MOMENTUM_LOOKBACK)
    features = pd.DataFrame(
        {
            "pooled_clv": pooled_clv,
            "weak_symbol_count": clv.lt(0.4).sum(axis=1),
            "clv_dispersion": clv.std(axis=1, ddof=0),
            "mean_pairwise_corr_60d": pd.concat(rolling_corr, axis=1).mean(axis=1),
            "relative_momentum_range_20d": momentum.max(axis=1) - momentum.min(axis=1),
            "pooled_signal": pooled_clv.lt(0.4),
        },
        index=index,
    )
    return features, forward


def _annual_strategy_returns(book_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for book_id, (daily, _positions, annual) in book_data.items():
        annual_pnl = daily.groupby(daily.index.year)[["毛盈亏", "交易费用"]].sum()
        annual_start_equity = daily.groupby(daily.index.year)["期初权益"].first()
        for row in annual.to_dict("records"):
            year = int(row["年份"])
            denominator = float(annual_start_equity.at[year])
            rows.append(
                {
                    "策略": book_id,
                    "年份": year,
                    "净收益": float(row["净收益"]),
                    "Sharpe": float(row["Sharpe"]),
                    "年度毛盈亏_年度初权益比例": float(annual_pnl.at[year, "毛盈亏"]) / denominator,
                    "年度交易成本_年度初权益比例": float(annual_pnl.at[year, "交易费用"]) / denominator,
                }
            )
        if daily.index.year.max() > 2021:
            raise C.HoldoutViolation(f"{book_id}逐日收益含研究期外日期")
    return pd.DataFrame(rows)


def _annual_features(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in range(2015, 2022):
        sample = features.loc[features.index.year == year]
        rows.append(
            {
                "年份": year,
                "交易日数": len(sample),
                "pooled_CLV小于0.4比例": sample["pooled_signal"].mean(),
                "单品种弱收盘数均值": sample["weak_symbol_count"].mean(),
                "CLV横截面标准差均值": sample["clv_dispersion"].mean(),
                "滚动60日平均两两相关": sample["mean_pairwise_corr_60d"].mean(),
                "过去20日相对收益区间均值": sample["relative_momentum_range_20d"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _product_market_returns(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    returns = (bars["closew"] - bars["closew"].shift(1)) / bars["close"].shift(1)
    rows = []
    for year in range(2015, 2022):
        sample = returns.loc[returns.index.year == year]
        for symbol in SYMBOLS:
            values = sample[symbol].dropna()
            rows.append(
                {
                    "年份": year,
                    "品种": symbol,
                    "交易日数": len(values),
                    "指数期货连续价格净变动": (1.0 + values).prod() - 1.0,
                }
            )
    return pd.DataFrame(rows)


def _momentum_predictiveness(features: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    rank_ic = pd.Series(index=features.index, dtype="float64")
    pooled_days = features["pooled_signal"]
    # 每日仅三个品种；秩相关作为方向性描述，不据此选品种或调仓。
    for day in features.index[pooled_days]:
        momentum = features.loc[day, [f"{symbol}_momentum_20d" for symbol in SYMBOLS]]
        outcome = forward.loc[day, list(SYMBOLS)]
        if momentum.notna().all() and outcome.notna().all() and momentum.nunique() > 1:
            rank_ic.loc[day] = np.corrcoef(
                momentum.rank().to_numpy(),
                outcome.rank().to_numpy(),
            )[0, 1]
    valid = rank_ic.dropna()
    periods = [("2015–2021", valid)] + [
        (str(year), valid.loc[valid.index.year == year]) for year in YEARS
    ]
    return pd.DataFrame(
        [
            {
                "样本": period,
                "检验": "pooled CLV信号日：20日相对收益秩与下一开盘段收益的每日Spearman相关",
                "样本数": len(sample),
                "平均秩相关": sample.mean(),
                "NW_t": newey_west_t(sample.to_numpy(), HAC_LAGS),
            }
            for period, sample in periods
        ]
    )


def _book_spread_regressions(
    book_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    features: pd.DataFrame,
) -> pd.DataFrame:
    pooled = book_data["index_clv_pooled_equal_long"][0]["净收益"].rename("pooled")
    individual = book_data["index_clv_individual_equal_long"][0]["净收益"].rename("individual")
    panel = pd.concat([pooled, individual], axis=1).dropna()
    panel["pooled_minus_individual"] = panel["pooled"] - panel["individual"]
    decision_features = features[
        ["mean_pairwise_corr_60d", "clv_dispersion", "weak_symbol_count"]
    ].shift(1)
    panel = panel.join(decision_features).dropna()
    rows = []
    for feature in decision_features.columns:
        x = panel[feature]
        scale = x.std(ddof=0)
        if not np.isfinite(scale) or scale == 0:
            continue
        coef, t_value = ols_hac(
            panel["pooled_minus_individual"].to_numpy(),
            [((x - x.mean()) / scale).to_numpy()],
            HAC_LAGS,
        )
        rows.append(
            {
                "差值": "汇总等权净收益 - 单品种活跃等权净收益",
                "已知时点指标": feature,
                "每1标准差指标变化的日净收益差": coef,
                "NW_t": t_value,
                "样本数": len(panel),
            }
        )
    return pd.DataFrame(rows)


def _product_contributions(
    book_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    bars: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    open_delta = bars["openw"].shift(-1) - bars["openw"]
    rows = []
    for book_id in BOOK_IDS:
        positions = book_data[book_id][1].copy()
        positions["价格变动"] = [
            open_delta.at[day, symbol]
            for day, symbol in zip(positions["交易日"], positions["品种"])
        ]
        positions["毛盈亏贡献"] = (
            positions["手数"] * positions["合约乘数"] * positions["价格变动"]
        )
        attributed = positions.groupby("交易日")["毛盈亏贡献"].sum().reindex(
            book_data[book_id][0].index, fill_value=0.0
        )
        actual = book_data[book_id][0]["毛盈亏"]
        if not np.allclose(attributed.to_numpy(), actual.to_numpy(), rtol=1e-8, atol=0.01):
            raise ValueError(f"{book_id}按品种重建的毛盈亏与账本不一致")
        positions = positions[positions["交易日"].dt.year.isin(YEARS)]
        if positions.empty:
            continue
        grouped = positions.groupby(["交易日", "品种"], as_index=False)["毛盈亏贡献"].sum()
        grouped["年份"] = grouped["交易日"].dt.year
        annual = grouped.groupby(["年份", "品种"], as_index=False)["毛盈亏贡献"].sum()
        for row in annual.to_dict("records"):
            rows.append(
                {
                    "策略": book_id,
                    "年份": int(row["年份"]),
                    "品种": row["品种"],
                    "毛盈亏": float(row["毛盈亏贡献"]),
                    "占初始资金": float(row["毛盈亏贡献"]) / CAPITAL,
                }
            )
    return pd.DataFrame(rows)


def _allocation_shrinkage(
    index: pd.DatetimeIndex,
    bars: dict[str, pd.DataFrame],
    contracts: pd.DataFrame,
    rally_z: pd.Series,
    ticks: pd.DataFrame,
    book_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from research.index.index_execution_10m import load_opening_quotes, l1_slippage

    returns = ((bars["openw"].shift(-1) - bars["openw"]) / bars["open"]).dropna(how="any")
    scored = returns.loc[
        (returns.index.year >= INDEX_RESEARCH_START.year)
        & (returns.index.year <= INDEX_RESEARCH_END.year)
    ].index
    pooled_clv = market_close_location(bars, index)
    fixed_targets = _legacy_targets(
        pooled_clv, bars, rally_z, FIXED_MULTIPLIERS
    ).reindex(scored)
    equal_targets = _legacy_targets(
        pooled_clv, bars, rally_z, (1.0, 1.0, 1.0)
    ).reindex(scored)
    if not np.allclose(
        fixed_targets.sum(axis=1).to_numpy(),
        equal_targets.sum(axis=1).to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("固定权重与等权目标的总预算不一致")

    quotes = load_opening_quotes()
    slippage, _, _ = l1_slippage(quotes, bars, contracts, scored)
    targets_by_level = {
        level: fixed_targets.mul(1.0 - level).add(equal_targets.mul(level))
        for level in SHRINKAGE_LEVELS
    }
    daily_by_level: dict[float, pd.DataFrame] = {}
    summary_rows, annual_rows, paired_rows, weight_rows = [], [], [], []

    for level, target in targets_by_level.items():
        daily, _positions, _events = _simulate_index_targets(
            target,
            scored,
            bars,
            contracts,
            ticks,
            INDEX_RESEARCH_END.year,
            INDEX_SYMBOLS,
            exposure_limit=GROSS,
            drawdown_limit=DRAWDOWN_LIMIT,
            halt_days=HALT_DAYS,
            initial_capital=CAPITAL,
            open_limits=True,
            open_slippage=slippage,
        )
        daily_by_level[level] = daily
        returns_daily = daily["净收益"].fillna(0.0)
        metrics = performance(returns_daily)
        annual_group = daily.groupby(daily.index.year, sort=True)
        annual_returns = {}
        for year, part in annual_group:
            opening = float(part["期初权益"].iloc[0])
            annual_return = float(part["权益"].iloc[-1] / opening - 1.0)
            annual_returns[int(year)] = annual_return
            annual_rows.append(
                {
                    "等权收缩比例": level,
                    "年份": int(year),
                    "净收益": annual_return,
                    "Sharpe": sharpe_ratio(part["净收益"].fillna(0.0)),
                    "年度内最大回撤": max_drawdown(part["权益"], opening),
                    "交易费用_年度初权益比例": float(part["交易费用"].sum()) / opening,
                    "开仓笔数": int(part["开仓笔数"].sum()),
                    "平仓笔数": int(part["平仓笔数"].sum()),
                }
            )

        weights = FIXED_MULTIPLIERS
        equal_weights = (1.0, 1.0, 1.0)
        shrunk_weights = tuple(
            fixed * (1.0 - level) + equal * level
            for fixed, equal in zip(weights, equal_weights)
        )
        weight_rows.append(
            {
                "等权收缩比例": level,
                **dict(zip(SYMBOLS, shrunk_weights)),
                "乘数合计": sum(shrunk_weights),
            }
        )
        annual_values = np.asarray(list(annual_returns.values()), dtype="float64")
        summary_rows.append(
            {
                "等权收缩比例": level,
                "样本开始": daily.index.min().date().isoformat(),
                "样本结束": daily.index.max().date().isoformat(),
                "交易日数": len(daily),
                "累计净收益": float(daily["权益"].iloc[-1] / CAPITAL - 1.0),
                "年化收益": metrics["ann_return"],
                "年化波动": metrics["ann_vol"],
                "Sharpe": sharpe_ratio(returns_daily),
                "最大回撤": max_drawdown(daily["权益"], CAPITAL),
                "Calmar": metrics["calmar"],
                "正收益年份数": int((annual_values > 0).sum()),
                "年度净收益标准差": float(annual_values.std(ddof=1)),
                "最差年度净收益": float(annual_values.min()),
                "交易费用_初始资金比例": float(daily["交易费用"].sum()) / CAPITAL,
                "开仓笔数": int(daily["开仓笔数"].sum()),
                "平仓笔数": int(daily["平仓笔数"].sum()),
                "平均敞口倍数": float(daily["敞口倍数"].mean()),
            }
        )

    for level, reference_id in (
        (0.0, "index_clv_rally_fixed_long"),
        (1.0, "index_clv_pooled_equal_long"),
    ):
        actual = book_data[reference_id][0]
        reconstructed = daily_by_level[level]
        if not reconstructed.index.equals(actual.index) or not np.allclose(
            reconstructed["净收益"].to_numpy(),
            actual["净收益"].to_numpy(),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(f"lambda={level:g}端点未能逐日复现既有账本 {reference_id}")

    baseline = daily_by_level[0.0]["净收益"].fillna(0.0)
    for level in SHRINKAGE_LEVELS[1:]:
        difference = daily_by_level[level]["净收益"].fillna(0.0) - baseline
        paired_rows.append(
            {
                "比较": f"lambda={level:g}减lambda=0",
                "平均日净收益差_bp": float(difference.mean() * 10_000.0),
                "近似年化均值差_pct": float(difference.mean() * C.TRADING_DAYS_PER_YEAR * 100.0),
                "NW_t": newey_west_t(difference.to_numpy(), HAC_LAGS),
                "配对交易日数": len(difference),
            }
        )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(annual_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(weight_rows),
    )


def _allocation_multipliers(
    bars: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    index = bars["close"].index
    equal = pd.DataFrame(1.0, index=index, columns=SYMBOLS)
    weights = {
        "fixed_current": pd.DataFrame(
            np.tile(FIXED_MULTIPLIERS, (len(index), 1)),
            index=index,
            columns=SYMBOLS,
        ),
        "equal": equal.copy(),
    }

    for method in ALLOCATION_METHODS[2:]:
        weights[method] = allocation_multipliers(bars, method)

    for method, frame in weights.items():
        if (
            frame.isna().any().any()
            or not np.isfinite(frame.to_numpy(dtype="float64")).all()
            or not np.allclose(frame.sum(axis=1).to_numpy(), ALLOCATION_WEIGHT_SUM)
            or (frame > ALLOCATION_MAX_MULTIPLIER + 1e-12).any().any()
        ):
            raise ValueError(f"{method}股指品种预算不满足总和或集中度约束")

    daily = pd.concat(
        [
            frame.rename_axis(index="交易日")
            .rename_axis(columns="品种")
            .stack()
            .rename("目标乘数")
            .to_frame()
            .assign(分配方法=method)
            .reset_index()
            for method, frame in weights.items()
        ],
        ignore_index=True,
    )
    return weights, daily


def _allocation_backtests(
    index: pd.DatetimeIndex,
    bars: dict[str, pd.DataFrame],
    contracts: pd.DataFrame,
    rally_z: pd.Series,
    ticks: pd.DataFrame,
    book_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from research.index.index_execution_10m import load_opening_quotes, l1_slippage

    returns = ((bars["openw"].shift(-1) - bars["openw"]) / bars["open"]).dropna(how="any")
    scored = returns.loc[
        (returns.index.year >= INDEX_RESEARCH_START.year)
        & (returns.index.year <= INDEX_RESEARCH_END.year)
    ].index
    pooled_clv = market_close_location(bars, index)
    fixed_targets = _legacy_targets(
        pooled_clv, bars, rally_z, FIXED_MULTIPLIERS
    ).reindex(scored)
    base_budget = fixed_targets.sum(axis=1) / ALLOCATION_WEIGHT_SUM
    if not base_budget.between(0.0, 1.5, inclusive="both").all():
        raise ValueError("固定股指信号总预算超出预期范围")

    multipliers, weights_daily = _allocation_multipliers(bars)
    targets = {"fixed_current": fixed_targets}
    for method in ALLOCATION_METHODS[1:]:
        targets[method] = (
            multipliers[method].shift(1).reindex(scored).fillna(1.0)
            .mul(base_budget, axis=0)
        )

    quotes = load_opening_quotes()
    slippage, _, _ = l1_slippage(quotes, bars, contracts, scored)
    daily_by_method: dict[str, pd.DataFrame] = {}
    summary_rows, annual_rows, paired_rows, daily_rows = [], [], [], []

    for method in ALLOCATION_METHODS:
        target = targets[method]
        if target.isna().any().any():
            raise ValueError(f"{method}目标仓位包含缺失值")
        daily, positions, _events = _simulate_index_targets(
            target,
            scored,
            bars,
            contracts,
            ticks,
            INDEX_RESEARCH_END.year,
            INDEX_SYMBOLS,
            exposure_limit=GROSS,
            drawdown_limit=DRAWDOWN_LIMIT,
            halt_days=HALT_DAYS,
            initial_capital=CAPITAL,
            open_limits=True,
            open_slippage=slippage,
        )
        daily_by_method[method] = daily
        if method in ("fixed_current", "equal"):
            reference_id = (
                "index_clv_rally_fixed_long"
                if method == "fixed_current"
                else "index_clv_pooled_equal_long"
            )
            actual = book_data[reference_id][0]
            if not daily.index.equals(actual.index) or not np.allclose(
                daily["净收益"].to_numpy(),
                actual["净收益"].to_numpy(),
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(f"{method}端点未能逐日复现既有账本 {reference_id}")

        pnl_by_symbol = positions.copy()
        open_delta = bars["openw"].shift(-1) - bars["openw"]
        pnl_by_symbol["价格变动"] = [
            open_delta.at[day, symbol]
            for day, symbol in zip(pnl_by_symbol["交易日"], pnl_by_symbol["品种"])
        ]
        pnl_by_symbol["毛盈亏贡献"] = (
            pnl_by_symbol["手数"]
            * pnl_by_symbol["合约乘数"]
            * pnl_by_symbol["价格变动"]
        )
        attributed = pnl_by_symbol.groupby("交易日")["毛盈亏贡献"].sum().reindex(
            daily.index, fill_value=0.0
        )
        if not np.allclose(
            attributed.to_numpy(), daily["毛盈亏"].to_numpy(), rtol=1e-8, atol=0.01
        ):
            raise ValueError(f"{method}按品种重建的毛盈亏与账本不一致")

        net_returns = daily["净收益"].fillna(0.0)
        metrics = performance(net_returns)
        annual = daily.groupby(daily.index.year, sort=True)
        annual_returns = []
        for year, part in annual:
            opening = float(part["期初权益"].iloc[0])
            net_return = float(part["权益"].iloc[-1] / opening - 1.0)
            annual_returns.append(net_return)
            annual_rows.append(
                {
                    "分配方法": method,
                    "年份": int(year),
                    "净收益": net_return,
                    "Sharpe": sharpe_ratio(part["净收益"].fillna(0.0)),
                    "年度内最大回撤": max_drawdown(part["权益"], opening),
                    "交易费用_年度初权益比例": float(part["交易费用"].sum()) / opening,
                    "开仓笔数": int(part["开仓笔数"].sum()),
                    "平仓笔数": int(part["平仓笔数"].sum()),
                }
            )
        weights_at_signal = multipliers[method].reindex(scored).where(
            fixed_targets.sum(axis=1).ne(0), np.nan
        )
        summary_rows.append(
            {
                "分配方法": method,
                "样本开始": daily.index.min().date().isoformat(),
                "样本结束": daily.index.max().date().isoformat(),
                "交易日数": len(daily),
                "累计净收益": float(daily["权益"].iloc[-1] / CAPITAL - 1.0),
                "年化收益": metrics["ann_return"],
                "年化波动": metrics["ann_vol"],
                "Sharpe": sharpe_ratio(net_returns),
                "最大回撤": max_drawdown(daily["权益"], CAPITAL),
                "Calmar": metrics["calmar"],
                "正收益年份数": int((np.asarray(annual_returns) > 0).sum()),
                "年度净收益标准差": float(np.std(annual_returns, ddof=1)),
                "最差年度净收益": float(np.min(annual_returns)),
                "交易费用_初始资金比例": float(daily["交易费用"].sum()) / CAPITAL,
                "开仓笔数": int(daily["开仓笔数"].sum()),
                "平仓笔数": int(daily["平仓笔数"].sum()),
                "平均敞口倍数": float(daily["敞口倍数"].mean()),
                "信号日平均IF乘数": float(weights_at_signal["IF"].mean()),
                "信号日平均IH乘数": float(weights_at_signal["IH"].mean()),
                "信号日平均IC乘数": float(weights_at_signal["IC"].mean()),
            }
        )
        daily_rows.append(
            daily[["净收益", "毛盈亏", "交易费用", "权益", "敞口倍数"]]
            .rename(
                columns={
                    "净收益": f"{method}_净收益",
                    "毛盈亏": f"{method}_毛盈亏",
                    "交易费用": f"{method}_交易费用",
                    "权益": f"{method}_权益",
                    "敞口倍数": f"{method}_敞口倍数",
                }
            )
        )

    baseline = daily_by_method["fixed_current"]["净收益"].fillna(0.0)
    for method in ALLOCATION_METHODS[1:]:
        difference = daily_by_method[method]["净收益"].fillna(0.0) - baseline
        paired_rows.append(
            {
                "比较": f"{method}减fixed_current",
                "平均日净收益差_bp": float(difference.mean() * 10_000.0),
                "近似年化均值差_pct": float(
                    difference.mean() * C.TRADING_DAYS_PER_YEAR * 100.0
                ),
                "NW_t": newey_west_t(difference.to_numpy(), HAC_LAGS),
                "配对交易日数": len(difference),
            }
        )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(annual_rows),
        pd.DataFrame(paired_rows),
        pd.concat(daily_rows, axis=1).rename_axis("交易日").reset_index(),
        weights_daily,
    )


def run() -> None:
    protocol = load_protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    index, bars, contracts, rally_z, ticks = _load_index_research_inputs()
    C.assert_no_holdout_dates(index, what="股指跨品种研究")
    if index.min() < pd.Timestamp("2015-04-16") or index.max() > pd.Timestamp("2021-12-31"):
        raise C.HoldoutViolation("股指跨品种研究行情超出约定研究期")

    features, forward = _features(bars)
    for symbol in SYMBOLS:
        momentum = (
            bars["closew"][symbol]
            - bars["closew"][symbol].shift(MOMENTUM_LOOKBACK)
        ) / bars["close"][symbol].shift(MOMENTUM_LOOKBACK)
        features[f"{symbol}_momentum_20d"] = momentum

    book_data = {book_id: _read_book(book_id) for book_id in BOOK_IDS}
    _annual_strategy_returns(book_data).to_csv(
        OUT / "annual_strategy_returns.csv", index=False, encoding="utf-8-sig"
    )
    _annual_features(features).to_csv(
        OUT / "annual_regime_features.csv", index=False, encoding="utf-8-sig"
    )
    _product_market_returns(bars).to_csv(
        OUT / "product_market_returns.csv", index=False, encoding="utf-8-sig"
    )
    _momentum_predictiveness(features, forward).to_csv(
        OUT / "momentum_rank_predictiveness.csv", index=False, encoding="utf-8-sig"
    )
    _book_spread_regressions(book_data, features).to_csv(
        OUT / "regime_return_spreads.csv", index=False, encoding="utf-8-sig"
    )
    _product_contributions(book_data, bars).to_csv(
        OUT / "product_gross_contributions.csv", index=False, encoding="utf-8-sig"
    )
    shrinkage_summary, shrinkage_annual, shrinkage_paired, shrinkage_weights = (
        _allocation_shrinkage(index, bars, contracts, rally_z, ticks, book_data)
    )
    shrinkage_summary.to_csv(
        OUT / "allocation_shrinkage_summary.csv", index=False, encoding="utf-8-sig"
    )
    shrinkage_annual.to_csv(
        OUT / "allocation_shrinkage_annual.csv", index=False, encoding="utf-8-sig"
    )
    shrinkage_paired.to_csv(
        OUT / "allocation_shrinkage_paired_differences.csv", index=False, encoding="utf-8-sig"
    )
    shrinkage_weights.to_csv(
        OUT / "allocation_shrinkage_weights.csv", index=False, encoding="utf-8-sig"
    )
    allocation_summary, allocation_annual, allocation_paired, allocation_daily, weights_daily = (
        _allocation_backtests(index, bars, contracts, rally_z, ticks, book_data)
    )
    allocation_summary.to_csv(
        OUT / "allocation_backtest_summary.csv", index=False, encoding="utf-8-sig"
    )
    allocation_annual.to_csv(
        OUT / "allocation_backtest_annual.csv", index=False, encoding="utf-8-sig"
    )
    allocation_paired.to_csv(
        OUT / "allocation_backtest_paired_differences.csv", index=False, encoding="utf-8-sig"
    )
    allocation_daily.to_csv(
        OUT / "allocation_backtest_daily.csv", index=False, encoding="utf-8-sig"
    )
    weights_daily.to_csv(
        OUT / "allocation_weights_daily.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"研究结果写入 {OUT}", flush=True)


if __name__ == "__main__":
    run()
