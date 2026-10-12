"""统一运行策略：python scripts/run_strategy.py <strategy> <phase>。"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STRATEGIES = {
    "morning_v1": "--v1",
    "morning_v2": "--v2",
    "morning_benchmark": "--benchmark",
    "morning_compare": "--compare",
}
PHASES = ("research", "validation-2022", "oos")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=STRATEGIES)
    parser.add_argument("phase", choices=PHASES)
    args = parser.parse_args()

    if args.phase == "oos":
        parser.error("样本外最终测试请直接运行 scripts/oos_final.py（先 --check）")

    command = [sys.executable, str(ROOT / "scripts" / "morning.py"), STRATEGIES[args.strategy]]
    if args.phase == "validation-2022":
        command.append("--validation-2022")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()