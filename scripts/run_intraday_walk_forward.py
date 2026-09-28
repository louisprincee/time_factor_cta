"""运行经典日内策略的真实 walk-forward。

参数只在训练窗口内选择，随后测试窗口冻结；不要把输出的训练 Sharpe 当作样本外结果。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.data import shard_io  # noqa: E402
from tfcta.research.intraday import (  # noqa: E402
    BacktestConfig,
    WalkForwardConfig,
    classic_candidate_grid,
    run_walk_forward,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--strategies", nargs="+",
                        choices=("dual_thrust", "atr", "rbreaker"),
                        default=["dual_thrust", "atr", "rbreaker"])
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--min-train-trades", type=int, default=20)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--fee-rate", type=float, default=0.00025)
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--margin-rate", type=float, default=0.30)
    parser.add_argument("--output", default="runs/intraday_walk_forward")
    args = parser.parse_args()
    if not 0 < args.margin_rate <= 1:
        parser.error("--margin-rate 必须在 (0, 1] 内")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    columns = ["open", "openw", "close", "closew", "highw", "loww", "trading_date"]
    rows = []
    for symbol in [s.upper() for s in args.symbols]:
        minute = shard_io.load_shard(symbol, C.RESEARCH_DIR, columns=columns)
        if args.start:
            minute = minute[minute["trading_date"] >= pd.Timestamp(args.start)]
        if args.end:
            minute = minute[minute["trading_date"] <= pd.Timestamp(args.end)]
        result = run_walk_forward(
            minute,
            classic_candidate_grid(tuple(args.strategies)),
            WalkForwardConfig(args.train_years, args.test_years, args.min_train_trades),
            BacktestConfig(args.fee_rate, args.slippage_points, args.margin_rate),
            symbol,
        )
        result.folds.insert(0, "symbol", symbol)
        result.folds.to_csv(output / f"{symbol}_folds.csv", index=False)
        result.daily.to_csv(output / f"{symbol}_daily.csv")
        result.trades.to_csv(output / f"{symbol}_trades.csv", index=False)
        (output / f"{symbol}_params.json").write_text(json.dumps({
            "walk_forward": asdict(WalkForwardConfig(
                args.train_years, args.test_years, args.min_train_trades)),
            "backtest": asdict(BacktestConfig(
                args.fee_rate, args.slippage_points, args.margin_rate)),
            "strategies": args.strategies,
            "candidate_count": len(classic_candidate_grid(tuple(args.strategies))),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"symbol": symbol, **result.metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False) if not summary.empty else "没有生成结果")
    return 0 if not summary.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())