"""仅在非弱收盘时开仓的固定 Donchian 趋势袖套研究试验。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest import engine
from research.index.strategy_studies import index_clv_breakout_sleeve as base
from strategies.index_common import SYMBOLS, close_vol_scale
from strategies.index_tactical_books import TACTICAL_ONLY_BOOK
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_clv_nonweak_breakout_sleeve.json"
OUT = C.study_dir("index", "index_clv_nonweak_breakout_sleeve")
ID = "index_clv_nonweak_breakout_sleeve"
NAME = "非弱收盘开仓的55/20突破趋势袖套"
DETAIL = "pooled CLV>=0.4 时允许55日收盘突破开仓；持仓按20日低点退出，趋势目标敞口上限0.5倍"
SCORE_START = pd.Timestamp("2015-04-16")
SCORE_END = pd.Timestamp("2021-12-30")
TREND_GROSS_EXPOSURE_LIMIT = 0.5


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parameters = protocol.get("parameters", {})
    sample = protocol.get("sample", "")
    if (
        protocol.get("id") != ID
        or         protocol.get("status") != "frozen_research_strategies"
        or protocol.get("frozen_at") != "2026-11-07"
        or protocol.get("chosen") != parameters
        or parameters.get("instruments") != list(SYMBOLS)
        or parameters.get("entry_lookback") != base.ENTRY_LOOKBACK
        or parameters.get("entry_price") != "closew > prior_55_session_max_highw"
        or parameters.get("exit_lookback") != base.EXIT_LOOKBACK
        or parameters.get("exit_price") != "closew < prior_20_session_min_loww"
        or parameters.get("entry_filter") != "pooled_clv >= 0.4 on signal close; filters entries only"
        or parameters.get("trend_gross_exposure_limit") != TREND_GROSS_EXPOSURE_LIMIT
        or parameters.get("trend_target_matrix_sum_limit")
        != TREND_GROSS_EXPOSURE_LIMIT * len(SYMBOLS)
        or parameters.get("per_symbol_target_limit") != engine.TARGET_CAP
        or parameters.get("clv_strategy") != TACTICAL_ONLY_BOOK.ID
        or parameters.get("clv_rally_multiplier") != 2.0
        or parameters.get("vol_window") != 20
        or parameters.get("vol_target") != 0.2
        or parameters.get("drawdown_breaker") != 0.1
        or parameters.get("halt_days") != 10
        or parameters.get("initial_capital") != 10_000_000
        or parameters.get("execution") != "l1_30m"
        or parameters.get("score_start") != str(SCORE_START.date())
        or parameters.get("score_end") != str(SCORE_END.date())
        or parameters.get("paired_difference_hac_lags") != 5
        or "不读取 2022–2025" not in sample
        or "任何 oos/目录" not in sample
        or "后验探索" not in protocol.get("limitations", "")
    ):
        raise ValueError("非弱收盘突破趋势袖套协议与实现或样本边界不一致")
    return protocol


def combine_targets(
    clv: pd.DataFrame,
    raw_trend: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert sleeve budget to account gross exposure and respect per-product limits."""
    if (
        not clv.index.equals(raw_trend.index)
        or tuple(clv.columns) != SYMBOLS
        or tuple(raw_trend.columns) != SYMBOLS
        or clv.empty
        or not np.isfinite(clv.to_numpy(dtype=float)).all()
        or not np.isfinite(raw_trend.to_numpy(dtype=float)).all()
        or clv.lt(0).any().any()
        or raw_trend.lt(0).any().any()
    ):
        raise ValueError("CLV 与趋势目标必须为对齐、非负、有限的多头目标")
    raw_gross = raw_trend.sum(axis=1) / len(SYMBOLS)
    if raw_gross.gt(base.TREND_BUDGET + 1e-12).any():
        raise ValueError("趋势目标超过登记的实际名义敞口预算")

    trend = raw_trend * len(SYMBOLS)
    available = (engine.TARGET_CAP - clv).clip(lower=0.0)
    trend = pd.DataFrame(
        np.minimum(trend.to_numpy(dtype=float), available.to_numpy(dtype=float)),
        index=trend.index,
        columns=trend.columns,
    )
    combined = clv + trend
    trend_gross = trend.sum(axis=1) / len(SYMBOLS)
    combined_gross = combined.sum(axis=1) / len(SYMBOLS)
    if (
        trend_gross.gt(TREND_GROSS_EXPOSURE_LIMIT + 1e-12).any()
        or combined.gt(engine.TARGET_CAP + 1e-12).any().any()
        or combined_gross.gt(2.0 + TREND_GROSS_EXPOSURE_LIMIT + 1e-12).any()
    ):
        raise ArithmeticError("趋势或组合目标超过登记的名义敞口上限")
    return trend, combined


def run(*, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = load_protocol()
    trial_result_dir = C.backtest_dir("index", ID, "research")
    if (OUT / "summary.csv").exists() and not overwrite:
        raise FileExistsError(f"非弱收盘突破研究结果已存在；使用 --overwrite 刷新：{OUT}")
    if trial_result_dir.exists() and not overwrite:
        raise FileExistsError(f"非弱收盘突破账本已存在；使用 --overwrite 刷新：{trial_result_dir}")

    C.assert_research_only(C.INDEX_RESEARCH_DIR)
    index, bars, contracts, rally_z, ticks = engine._load_index_research_inputs()
    C.assert_no_holdout_dates(index, what="非弱收盘突破袖套研究行情")
    if index.min() < SCORE_START or index.max() > pd.Timestamp("2021-12-31"):
        raise C.HoldoutViolation("非弱收盘突破袖套行情越过登记的研究期")

    open_to_open = (
        (bars["openw"].shift(-1) - bars["openw"]) / bars["open"]
    ).mean(axis=1, skipna=False)
    scale = close_vol_scale(open_to_open).fillna(1.0).reindex(index)
    if scale.isna().any():
        raise ValueError("非弱收盘突破袖套波动缩放覆盖不完整")

    market_clv, scores = TACTICAL_ONLY_BOOK.signal_inputs(bars, index)
    if market_clv.isna().any() or not np.isfinite(market_clv.to_numpy(dtype=float)).all():
        raise ValueError("非弱收盘开仓过滤器的 pooled CLV 覆盖不完整")
    states = base.breakout_states(
        bars,
        index,
        entry_allowed=market_clv.ge(0.4),
    )
    raw_trend = base.trend_targets(states, scale)
    clv = TACTICAL_ONLY_BOOK.targets((market_clv, scores), bars, rally_z=rally_z)
    trend, combined = combine_targets(clv, raw_trend)

    open_returns = (
        (bars["openw"].shift(-1) - bars["openw"]) / bars["open"]
    ).dropna(how="any")
    scored = open_returns.index[
        (open_returns.index >= SCORE_START) & (open_returns.index <= SCORE_END)
    ]
    if scored.empty or scored.max() > SCORE_END:
        raise ValueError("非弱收盘突破袖套计分日期为空或越过研究期")

    from research.index.shared.index_execution_10m import load_opening_quotes, l1_slippage

    quotes = load_opening_quotes()
    slippage, _, _ = l1_slippage(quotes, bars, contracts, scored)
    baseline_daily, _, _ = base._simulate(clv, scored, bars, contracts, ticks, slippage)
    trial_daily, trial_positions, trial_events = base._simulate(
        combined, scored, bars, contracts, ticks, slippage
    )
    benchmark_targets = pd.DataFrame(1.0, index=index, columns=SYMBOLS)
    benchmark_daily, _, _ = base._simulate(
        benchmark_targets,
        scored,
        bars,
        contracts,
        ticks,
        slippage,
        drawdown_limit=0.99,
    )

    trial = SimpleNamespace(ID=ID, NAME=NAME, DETAIL=DETAIL, GROSS=1.0)
    engine.write_results(
        "index_research",
        trial,
        trial_daily,
        trial_positions,
        trial_events,
        list(SYMBOLS),
        initial_capital=base.INITIAL_CAPITAL,
        market="index",
        benchmark=benchmark_daily["净收益"],
    )
    baseline_summary, baseline_annual = base._performance_rows(baseline_daily, "纯CLV基线")
    trial_summary, trial_annual = base._performance_rows(trial_daily, NAME)
    benchmark_summary, benchmark_annual = base._performance_rows(benchmark_daily, "等权买入持有")
    summaries = pd.DataFrame([baseline_summary, trial_summary, benchmark_summary])
    annual = pd.DataFrame([*baseline_annual, *trial_annual, *benchmark_annual])

    paired = trial_daily["净收益"].sub(baseline_daily["净收益"])
    comparison = pd.DataFrame(
        {
            "交易日": paired.index,
            "纯CLV净收益": baseline_daily["净收益"].to_numpy(dtype=float),
            "CLV加趋势净收益": trial_daily["净收益"].to_numpy(dtype=float),
            "日净收益差_bp": paired.to_numpy(dtype=float) * 1e4,
        }
    )
    paired_rows = []
    for year, part in paired.groupby(paired.index.year, sort=True):
        paired_rows.append(
            {
                "年份": int(year),
                "平均日净收益差_bp": float(part.mean() * 1e4),
                "平均年化净收益差_pct": float(
                    part.mean() * C.TRADING_DAYS_PER_YEAR * 100.0
                ),
                "NW_t_lag5": engine.newey_west_t(part.to_numpy(dtype=float), lags=5),
                "交易日数": len(part),
            }
        )
    paired_rows.append(
        {
            "年份": "合并",
            "平均日净收益差_bp": float(paired.mean() * 1e4),
            "平均年化净收益差_pct": float(
                paired.mean() * C.TRADING_DAYS_PER_YEAR * 100.0
            ),
            "NW_t_lag5": engine.newey_west_t(paired.to_numpy(dtype=float), lags=5),
            "交易日数": len(paired),
        }
    )
    execution_active = states.astype("boolean").shift(1).fillna(False).astype(bool)
    trend_report = pd.DataFrame(
        {
            "交易日": index,
            "pooled_CLV": market_clv.to_numpy(dtype=float),
            "IF收盘信号状态": states["IF"].to_numpy(),
            "IH收盘信号状态": states["IH"].to_numpy(),
            "IC收盘信号状态": states["IC"].to_numpy(),
            "开盘执行活跃品种数": execution_active.sum(axis=1).to_numpy(),
            "趋势袖套敞口倍数": (trend.sum(axis=1) / len(SYMBOLS)).to_numpy(),
            "CLV袖套敞口倍数": (clv.sum(axis=1) / len(SYMBOLS)).to_numpy(),
            "组合目标敞口倍数": (combined.sum(axis=1) / len(SYMBOLS)).to_numpy(),
        }
    )

    OUT.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "daily_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_rows).to_csv(
        OUT / "paired_difference_summary.csv", index=False, encoding="utf-8-sig"
    )
    trend_report.to_csv(
        OUT / "trend_states_and_budgets.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summaries.round(4).to_string(index=False), flush=True)
    print(annual.round(4).to_string(index=False), flush=True)
    print(f"非弱收盘突破袖套研究结果：{OUT}", flush=True)
    return summaries, annual


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="非弱收盘开仓的 Donchian 趋势袖套研究")
    parser.add_argument("--overwrite", action="store_true", help="覆盖同一研究协议生成的输出")
    args = parser.parse_args(argv)
    run(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
