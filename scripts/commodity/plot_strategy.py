"""重画已有结果的净值和回撤图（各策略脚本回测完已自动出图，这里用于挑序列重画）。

    python scripts/commodity/plot_strategy.py morning validation-2022 --series 主策略·亏损反手·30笔
    python scripts/commodity/plot_strategy.py oos 2024-2025
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tfcta.research import plot_performance

RESULTS = {
    "morning": "runs/commodity/morning/meta",
    "morning_v1": "runs/commodity/morning/morning_v1",
    "morning_v2": "runs/commodity/morning/morning_v2",
    "morning_benchmark": "runs/commodity/morning/morning_benchmark",
    "morning_compare": "runs/commodity/morning/comparison",
    "oos": "runs/commodity/oos",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("strategy", choices=RESULTS)
    parser.add_argument("phase", help="research、validation-2022；oos 时写年份，如 2024-2025")
    parser.add_argument("--series", nargs="+", help="只绘制指定的收益列")
    args = parser.parse_args()
    if args.strategy != "oos" and args.phase not in ("research", "validation-2022"):
        parser.error("阶段只能是 research 或 validation-2022")
    phase_dir = ROOT / RESULTS[args.strategy] / (args.phase if args.strategy == "oos" else args.phase.replace("-", "_"))
    returns = pd.read_csv(phase_dir / "daily.csv", index_col=0, parse_dates=True, encoding="utf-8-sig")
    try:
        path = plot_performance(returns, phase_dir / "plots" / args.phase, f"{args.strategy} {args.phase}",
                                        args.series)
    except KeyError as error:
        parser.error(str(error))
    print("图表已写入", path)


if __name__ == "__main__":
    main()
