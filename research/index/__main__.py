"""股指研究统一命令入口；各研究仍由独立协议和实现负责。"""
from __future__ import annotations

import argparse
import runpy
import sys

ROUTES = {
    ("mechanism", "clv"): (
        "research.index.mechanisms.index_close_location_mechanism"
    ),
    ("mechanism", "weak-close"): "research.index.mechanisms.weak_close_diagnostics",
    ("mechanism", "weak-close-classifier"): (
        "research.index.mechanisms.index_weak_close_mechanism_classifier"
    ),
    ("mechanism", "market"): "research.index.mechanisms.market_mechanisms",
    ("mechanism", "spot"): "research.index.mechanisms.spot_reversal",
    ("factor", "time-increment"): (
        "research.index.factors.index_time_factor_increment"
    ),
    ("factor", "cross-product"): (
        "research.index.factors.index_clv_cross_product_factors"
    ),
    ("factor", "strategy-candidates"): (
        "research.index.factors.index_tactical_factor_candidates"
    ),
    ("strategy", "allocation-regimes"): (
        "research.index.strategy_studies.index_allocation_regimes"
    ),
    ("strategy", "signal-grid"): (
        "research.index.strategy_studies.index_adaptive_signal_grid"
    ),
    ("strategy", "attribution"): (
        "research.index.strategy_studies.index_adaptive_attribution"
    ),
    ("strategy", "breakout"): (
        "research.index.strategy_studies.index_clv_breakout_sleeve"
    ),
    ("strategy", "nonweak-breakout"): (
        "research.index.strategy_studies.index_clv_nonweak_breakout_sleeve"
    ),
    ("strategy", "comparison"): (
        "research.index.strategy_studies.index_selected_strategy_comparison"
    ),
}
NO_ARGUMENT_STUDIES = {
    ("factor", "time-increment"),
    ("factor", "cross-product"),
    ("factor", "strategy-candidates"),
    ("strategy", "allocation-regimes"),
    ("strategy", "signal-grid"),
    ("strategy", "comparison"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="按研究方向运行股指研究；具体参数由各研究脚本解析。",
        usage="%(prog)s <方向> <研究名> [研究参数]",
    )
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        areas = tuple(dict.fromkeys(area for area, _ in ROUTES))
        groups = [
            f"{area}: "
            + ", ".join(name for route_area, name in ROUTES if route_area == area)
            for area in areas
        ]
        parser.epilog = "研究方向：" + "；".join(groups)
        parser.print_help()
        return 0

    areas = tuple(dict.fromkeys(area for area, _ in ROUTES))
    area = args[0]
    if area not in areas:
        parser.error(f"未知研究方向 {area!r}；可选项：{', '.join(areas)}")
    studies = [name for route_area, name in ROUTES if route_area == area]
    if len(args) == 1 or args[1] in ("-h", "--help"):
        print(f"{area} 下的研究：{', '.join(studies)}")
        print(f"用法：python -m research.index {area} <研究名> [研究参数]")
        return 0

    study = args[1]
    module = ROUTES.get((area, study))
    if module is None:
        choices = ", ".join(
            name for route_area, name in ROUTES if route_area == area
        )
        parser.error(f"{area} 下未知研究 {study!r}；可选项：{choices}")

    forwarded = args[2:]
    if (area, study) in NO_ARGUMENT_STUDIES and forwarded:
        if forwarded in (["-h"], ["--help"]):
            print(f"{area} {study} 不接受额外参数")
            return 0
        parser.error(f"{area} {study} 不接受额外参数：{' '.join(forwarded)}")

    original_argv = sys.argv
    try:
        sys.argv = [module, *forwarded]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
