"""运行预登记的单边上涨信号门槛网格。"""
from __future__ import annotations

import json

import pandas as pd

from backtest.engine import (
    INDEX_SYMBOLS,
    _load_index_research_inputs,
    _simulate_index_targets,
    max_drawdown,
    performance,
    sharpe_ratio,
)
from research.index.index_execution_10m import load_opening_quotes, l1_slippage
from research.index._adaptive_signal_helpers import (
    DRAWDOWN_LIMIT,
    GROSS,
    HALT_DAYS,
    adaptive_targets,
    one_sided_down,
    one_sided_up,
)
from strategies.index_common import market_close_location
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_adaptive_signal_grid.json"
OUT = C.study_dir("index", "index_adaptive_signal_grid")
BODY_THRESHOLDS = (0.6, 0.7, 0.8)
CLV_THRESHOLDS = (0.8, 0.9, 1.0)


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parameters = protocol.get("parameters", {})
    if (
        protocol.get("status") != "declared_parameter_comparison"
        or parameters.get("body_ratio_thresholds") != list(BODY_THRESHOLDS)
        or parameters.get("pooled_clv_thresholds") != list(CLV_THRESHOLDS)
        or parameters.get("variants") != len(BODY_THRESHOLDS) * len(CLV_THRESHOLDS)
        or "Sharpe" not in protocol.get("selection", {}).get("primary_metric", "")
        or not protocol.get("selection", {}).get("report_all_variants")
        or any(
            value not in protocol.get("sample", "")
            for value in ("2015-04-16", "2021-12-30", "2022-01-01", "2025-12-31", "2026")
        )
    ):
        raise ValueError("单边上涨信号网格与预登记协议不一致")
    return protocol


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    load_protocol()
    index, bars, contracts, rally_z, ticks = _load_index_research_inputs()
    next_open_returns = (
        (bars["openw"].shift(-1) - bars["openw"]) / bars["open"]
    ).dropna(how="any")
    scored = next_open_returns.loc[
        (next_open_returns.index >= pd.Timestamp("2015-04-16"))
        & (next_open_returns.index <= pd.Timestamp("2021-12-30"))
    ].index
    pooled_clv = market_close_location(bars, index)
    quotes = load_opening_quotes()
    slippage, _, _ = l1_slippage(quotes, bars, contracts, scored)
    down_signal = one_sided_down(bars, index)
    weak_close = pooled_clv.lt(0.4) & ~down_signal
    summary_rows, annual_rows = [], []

    for body_min in BODY_THRESHOLDS:
        for clv_min in CLV_THRESHOLDS:
            targets = adaptive_targets(
                pooled_clv,
                bars,
                rally_z=rally_z,
                one_sided_body_min=body_min,
                one_sided_clv_min=clv_min,
            ).reindex(scored)
            daily, _, _ = _simulate_index_targets(
                targets,
                scored,
                bars,
                contracts,
                ticks,
                2021,
                INDEX_SYMBOLS,
                exposure_limit=GROSS,
                drawdown_limit=DRAWDOWN_LIMIT,
                halt_days=HALT_DAYS,
                initial_capital=10_000_000.0,
                open_limits=True,
                open_slippage=slippage,
            )
            returns = daily["净收益"].fillna(0.0)
            metrics = performance(returns)
            variant = f"body_{body_min:.1f}_clv_{clv_min:.1f}"
            summary_rows.append(
                {
                    "变体": variant,
                    "实体占比门槛": body_min,
                    "pooled_CLV门槛": clv_min,
                    "样本开始": daily.index.min().date().isoformat(),
                    "样本结束": daily.index.max().date().isoformat(),
                    "计分日数": len(daily),
                    "累计净收益": float(daily["权益"].iloc[-1] / 10_000_000.0 - 1.0),
                    "年化收益": metrics["ann_return"],
                    "Sharpe": sharpe_ratio(returns),
                    "最大回撤": max_drawdown(daily["权益"], 10_000_000.0),
                    "新增单边上涨信号日数": int(
                        (one_sided_up(bars, index, pooled_clv, body_min, clv_min) & ~weak_close)
                        .reindex(scored)
                        .sum()
                    ),
                }
            )
            for year, part in daily.groupby(daily.index.year, sort=True):
                opening = float(part["期初权益"].iloc[0])
                annual_rows.append(
                    {
                        "变体": variant,
                        "实体占比门槛": body_min,
                        "pooled_CLV门槛": clv_min,
                        "年份": int(year),
                        "净收益": float(part["权益"].iloc[-1] / opening - 1.0),
                        "Sharpe": sharpe_ratio(part["净收益"].fillna(0.0)),
                        "年度内最大回撤": max_drawdown(part["权益"], opening),
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["Sharpe", "最大回撤", "年化收益", "实体占比门槛", "pooled_CLV门槛"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    annual = pd.DataFrame(annual_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    return summary, annual


if __name__ == "__main__":
    result, _ = run()
    print(result.to_string(index=False), flush=True)
