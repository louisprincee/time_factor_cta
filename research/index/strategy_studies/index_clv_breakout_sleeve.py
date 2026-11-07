"""纯 CLV 之外的固定 Donchian 趋势袖套研究试验。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest import engine
from strategies.index_breakout import (
    ENTRY_LOOKBACK,
    EXIT_LOOKBACK,
    TREND_BUDGET,
    breakout_states,
    trend_targets,
)
from strategies.index_common import SYMBOLS, close_vol_scale, high_open_z
from strategies.index_tactical_books import TACTICAL_ONLY_BOOK
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_clv_breakout_sleeve.json"
OUT = C.study_dir("index", "index_clv_breakout_sleeve")
ID = "index_clv_breakout_sleeve"
NAME = "股指纯CLV加55/20突破趋势袖套"
DETAIL = "纯CLV规则不变；逐品种收盘突破此前55日最高价做多，收盘跌破此前20日最低价退出，趋势总预算0.5等权分配"
INITIAL_CAPITAL = 10_000_000.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
SCORE_START = pd.Timestamp("2015-04-16")
SCORE_END = pd.Timestamp("2021-12-30")


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
        or parameters.get("entry_lookback") != ENTRY_LOOKBACK
        or parameters.get("exit_lookback") != EXIT_LOOKBACK
        or parameters.get("trend_budget") != TREND_BUDGET
        or parameters.get("score_start") != str(SCORE_START.date())
        or parameters.get("score_end") != str(SCORE_END.date())
        or parameters.get("paired_difference_hac_lags") != 5
        or "不读取 2022–2025" not in sample
        or "任何 oos/目录" not in sample
    ):
        raise ValueError("突破趋势袖套协议与实现或样本边界不一致")
    return protocol


def _performance_rows(daily: pd.DataFrame, label: str) -> tuple[dict, list[dict]]:
    returns = daily["净收益"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    metrics = engine.performance(returns)
    summary = {
        "策略": label,
        "累计净收益": float(daily["权益"].iloc[-1] / INITIAL_CAPITAL - 1.0),
        "年化收益": metrics["ann_return"],
        "Sharpe": engine.sharpe_ratio(returns),
        "最大回撤": engine.max_drawdown(daily["权益"], INITIAL_CAPITAL),
        "最大名义敞口倍数": float(daily["敞口倍数"].max()),
        "最大保证金权益占用": float(daily["保证金权益占用"].max()),
        "开仓笔数": int(daily["开仓笔数"].sum()),
        "平仓笔数": int(daily["平仓笔数"].sum()),
    }
    annual = []
    for year, part in daily.groupby(daily.index.year, sort=True):
        year_returns = part["净收益"].fillna(0.0)
        opening = float(part["期初权益"].iloc[0])
        annual.append(
            {
                "策略": label,
                "年份": int(year),
                "净收益": float(part["权益"].iloc[-1] / opening - 1.0),
                "Sharpe": engine.sharpe_ratio(year_returns),
                "年度内最大回撤": engine.max_drawdown(part["权益"], opening),
            }
        )
    return summary, annual


def _simulate(
    target: pd.DataFrame,
    scored: pd.DatetimeIndex,
    bars,
    contracts,
    ticks,
    slippage,
    *,
    drawdown_limit: float = DRAWDOWN_LIMIT,
):
    return engine._simulate_index_targets(
        target.reindex(index=scored, columns=SYMBOLS).fillna(0.0),
        scored,
        bars,
        contracts,
        ticks,
        SCORE_END.year,
        SYMBOLS,
        exposure_limit=1.0,
        drawdown_limit=drawdown_limit,
        halt_days=HALT_DAYS,
        initial_capital=INITIAL_CAPITAL,
        open_limits=True,
        open_slippage=slippage,
    )


def run(*, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = load_protocol()
    trial_result_dir = C.backtest_dir("index", ID, "research")
    if (OUT / "summary.csv").exists() and not overwrite:
        raise FileExistsError(f"突破趋势研究结果已存在；使用 --overwrite 刷新：{OUT}")
    if trial_result_dir.exists() and not overwrite:
        raise FileExistsError(f"突破趋势账本已存在；使用 --overwrite 刷新：{trial_result_dir}")

    C.assert_research_only(C.INDEX_RESEARCH_DIR)
    index, bars, contracts, rally_z, ticks = engine._load_index_research_inputs()
    C.assert_no_holdout_dates(index, what="突破趋势袖套研究行情")
    if index.min() < SCORE_START or index.max() > pd.Timestamp("2021-12-31"):
        raise C.HoldoutViolation("突破趋势袖套行情越过登记的研究期")

    open_to_open = ((bars["openw"].shift(-1) - bars["openw"]) / bars["open"]).mean(axis=1, skipna=False)
    scale = close_vol_scale(open_to_open).fillna(1.0).reindex(index)
    if scale.isna().any():
        raise ValueError("突破趋势袖套波动缩放覆盖不完整")

    states = breakout_states(bars, index)
    trend = trend_targets(states, scale)
    market_clv, scores = TACTICAL_ONLY_BOOK.signal_inputs(bars, index)
    clv = TACTICAL_ONLY_BOOK.targets((market_clv, scores), bars, rally_z=rally_z)
    combined = clv.add(trend, fill_value=0.0)
    if not np.isfinite(combined.to_numpy(dtype=float)).all() or combined.abs().gt(2.5 + 1e-12).any().any():
        raise ValueError("CLV 与趋势合并目标超出登记的最大预算")

    open_returns = ((bars["openw"].shift(-1) - bars["openw"]) / bars["open"]).dropna(how="any")
    scored = open_returns.index[
        (open_returns.index >= SCORE_START) & (open_returns.index <= SCORE_END)
    ]
    if scored.empty or scored.max() > SCORE_END:
        raise ValueError("突破趋势袖套计分日期为空或越过研究期")

    from research.index.shared.index_execution_10m import load_opening_quotes, l1_slippage

    quotes = load_opening_quotes()
    slippage, _, _ = l1_slippage(quotes, bars, contracts, scored)
    baseline_daily, _, _ = _simulate(clv, scored, bars, contracts, ticks, slippage)
    trial_daily, trial_positions, trial_events = _simulate(combined, scored, bars, contracts, ticks, slippage)
    benchmark_targets = pd.DataFrame(1.0, index=index, columns=SYMBOLS)
    benchmark_daily, _, _ = _simulate(
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
        initial_capital=INITIAL_CAPITAL,
        market="index",
        benchmark=benchmark_daily["净收益"],
    )
    baseline_summary, baseline_annual = _performance_rows(baseline_daily, "纯CLV基线")
    trial_summary, trial_annual = _performance_rows(trial_daily, NAME)
    benchmark_summary, benchmark_annual = _performance_rows(benchmark_daily, "等权买入持有")
    summaries = pd.DataFrame([baseline_summary, trial_summary, benchmark_summary])
    annual = pd.DataFrame([*baseline_annual, *trial_annual, *benchmark_annual])
    paired = trial_daily["净收益"].sub(baseline_daily["净收益"]).rename("趋势袖套减纯CLV_日净收益差")
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
                "平均年化净收益差_pct": float(part.mean() * C.TRADING_DAYS_PER_YEAR * 100.0),
                "NW_t_lag5": engine.newey_west_t(part.to_numpy(dtype=float), lags=5),
                "交易日数": len(part),
            }
        )
    paired_rows.append(
        {
            "年份": "合并",
            "平均日净收益差_bp": float(paired.mean() * 1e4),
            "平均年化净收益差_pct": float(paired.mean() * C.TRADING_DAYS_PER_YEAR * 100.0),
            "NW_t_lag5": engine.newey_west_t(paired.to_numpy(dtype=float), lags=5),
            "交易日数": len(paired),
        }
    )
    active = states.astype(bool)
    execution_active = active.astype("boolean").shift(1).fillna(False).astype(bool)
    trend_report = pd.DataFrame(
        {
            "交易日": index,
            "IF收盘信号状态": active["IF"].to_numpy(),
            "IH收盘信号状态": active["IH"].to_numpy(),
            "IC收盘信号状态": active["IC"].to_numpy(),
            "开盘执行活跃品种数": execution_active.sum(axis=1).to_numpy(),
            "趋势目标总预算": trend.sum(axis=1).to_numpy(),
            "CLV目标总预算": clv.sum(axis=1).to_numpy(),
            "组合目标总预算": combined.sum(axis=1).to_numpy(),
        }
    )

    OUT.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "daily_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_rows).to_csv(
        OUT / "paired_difference_summary.csv", index=False, encoding="utf-8-sig"
    )
    trend_report.to_csv(OUT / "trend_states_and_budgets.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summaries.round(4).to_string(index=False), flush=True)
    print(annual.round(4).to_string(index=False), flush=True)
    print(f"突破趋势袖套研究结果：{OUT}", flush=True)
    return summaries, annual


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="纯 CLV 与 Donchian 突破趋势袖套研究")
    parser.add_argument("--overwrite", action="store_true", help="覆盖同一研究协议生成的输出")
    args = parser.parse_args(argv)
    run(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
