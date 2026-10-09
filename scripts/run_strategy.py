"""统一运行策略：python scripts/run_strategy.py <strategy> <phase>。"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STRATEGIES = {
    "morning-oor": ("research_morning_oor.py", "flag"),
    "morning-meta": ("research_morning_meta.py", "flag"),
}
PHASES = ("research", "validation-2022", "oos")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=STRATEGIES)
    parser.add_argument("phase", choices=PHASES)
    args = parser.parse_args()

    if args.phase == "oos":
        parser.error("样本外最终测试请直接运行 scripts/oos_final.py --years 2024-2025（先 --check）")

    script, phase_mode = STRATEGIES[args.strategy]

    command = [sys.executable, str(ROOT / "scripts" / script)]
    if phase_mode == "flag" and args.phase == "validation-2022":
        command.append("--validation-2022")
    elif phase_mode == "option":
        command.append("--phase")
        command.append(args.phase)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()