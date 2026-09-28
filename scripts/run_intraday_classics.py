"""运行隔离的 1 分钟经典日内 CTA 策略。

示例（使用 gu 环境）：
    python scripts/run_intraday_classics.py --symbols RB RU SR --strategy all

``opening_range_assumption`` 是没有拿到原报告公式时的开盘区间突破基线，
不是报告原策略的确认复刻。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.data import shard_io  # noqa: E402
from tfcta.research.intraday import (  # noqa: E402
    BacktestConfig,
    StrategyConfig,
    available_strategies,
    run_intraday,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--strategy", choices=[*available_strategies(), "all"], default="all")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="runs/intraday_classics")
    parser.add_argument("--fee-rate", type=float, default=0.00025)
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--margin-rate", type=float, default=0.30)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--atr-window", type=int, default=20)
    parser.add_argument("--atr-multiple", type=float, default=1.0)
    parser.add_argument("--stop-atr", type=float, default=None)
    parser.add_argument("--target-atr", type=float, default=None)
    parser.add_argument("--opening-range-minutes", type=int, default=30)
    parser.add_argument("--max-entries-per-day", type=int, default=1)
    args = parser.parse_args()

    if not 0 < args.margin_rate <= 1:
        parser.error("--margin-rate 必须在 (0, 1] 内")
    names = available_strategies() if args.strategy == "all" else (args.strategy,)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    columns = ["open", "openw", "close", "closew", "highw", "loww", "trading_date"]
    rows = []
    for symbol in [s.upper() for s in args.symbols]:
        try:
            minute = shard_io.load_shard(symbol, C.RESEARCH_DIR, columns=columns)
        except FileNotFoundError as exc:
            print(str(exc))
            continue
        if args.start:
            minute = minute[minute["trading_date"] >= pd.Timestamp(args.start)]
        if args.end:
            minute = minute[minute["trading_date"] <= pd.Timestamp(args.end)]
        for name in names:
            strategy = StrategyConfig(
                name=name, lookback=args.lookback, atr_window=args.atr_window,
                atr_multiple=args.atr_multiple,
                opening_range_minutes=args.opening_range_minutes,
                max_entries_per_day=args.max_entries_per_day,
                stop_atr_multiple=args.stop_atr,
                target_atr_multiple=args.target_atr,
            )
            result = run_intraday(
                minute, strategy,
                BacktestConfig(
                    fee_rate=args.fee_rate,
                    slippage_points=args.slippage_points,
                    margin_rate=args.margin_rate,
                ), symbol)
            rows.append({"symbol": symbol, "strategy": name, **result.metrics})
            result.trades.to_csv(output / f"{symbol}_{name}_trades.csv", index=False)
            result.daily.to_csv(output / f"{symbol}_{name}_daily.csv")
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False) if not summary.empty else "没有生成结果")
    return 0 if not summary.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())