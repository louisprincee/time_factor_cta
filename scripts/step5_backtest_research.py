"""第 5 步：研究期回测。自选因子、自选板块，等权周频，只读 2016–2021。

默认是四个时间因子的周频书：ts_high×-1、ts_low、dfp_max、dfp_top3 等权，不分板块。
各因子做一次 252 日事前 z 分数后等权，截到 [-1, 1]。time_combo 已经是这四者的 z 分数平均，
入书时不再标准化第二次。每周最后一个交易日更新，次日开盘成交，手续费加 tick 滑点。
研究期扣费后 Sharpe 与年化没过门槛的池子，不能进第 6 步。

    python scripts/step5_backtest_research.py --list-pools
    python scripts/step5_backtest_research.py --list-factors
    python scripts/step5_backtest_research.py --factors time_combo neg_clv neg_ret_day \\
        --pools 黑色金属 农产品 --merge-pools
    python scripts/step5_backtest_research.py --config config/strategy_example.json

结果只是研究期证据，不消耗 2022 验证期。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.research.backtest import strategy  # noqa: E402
from tfcta.research.workflow import context  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    strategy.add_book_args(parser, with_criteria=True)
    args = parser.parse_args()
    listed = strategy.handle_listing(args)
    if listed is not None:
        return listed
    try:
        cfg = strategy.config_from_args(args)
    except ValueError as exc:
        print(exc)
        return 2

    try:
        table, cost_note, cases = strategy.screen_research(cfg, strategy.case_labels(cfg))
    except FileNotFoundError as exc:
        print(str(exc).splitlines()[0])
        return 2
    except KeyError as exc:
        print(f"{str(exc).splitlines()[0]}\n请先运行 step3_build_factors.py，或用 --list-factors 查看可选因子。")
        return 2
    overall = strategy.research_period_label()
    table.insert(0, "fingerprint", table["universe"].map(
        lambda case: strategy.fingerprint(cfg, case)))
    summary = table[table["period"] == overall]
    passed_cases = [row.universe for row in summary.itertuples()
                    if strategy.passes(row._asdict(), cfg)]
    failed_cases = [row.universe for row in summary.itertuples()
                    if row.universe not in passed_cases]

    C.ensure_dirs()
    run = context.run_dir("step5_backtest")
    out = C.RESEARCH_OUT_DIR / "backtest_research.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    table.to_csv(run / out.name, index=False, encoding="utf-8-sig")
    context.dump_json(run / "params.json", {
        "config": cfg.to_dict(),
        "cases": cases,
        "years": sorted(int(p) for p in table["period"].unique() if str(p).isdigit()),
        "rebalance": strategy.REBALANCE,
        "execution": strategy.EXECUTION,
        "slippage_note": cost_note,
        "sharpe": "日均值 / 日标准差 × sqrt(252)，无风险利率 0",
    })

    empty = [case for case, members in cases.items() if not members]
    note = f"因子 {strategy.factor_label(cfg.factors)}；{cost_note}"
    if empty:
        note += f" 这些池子在研究期没有品种: {empty}"
    if passed_cases:
        note += f" 研究期通过、可以进入第 6 步的池子: {passed_cases}。"
    if failed_cases:
        note += (f" 未过研究期门槛（扣费后 Sharpe ≥ {cfg.min_net_sharpe:g} 且年化 > "
                 f"{cfg.min_net_ann_return:g}）的池子: {failed_cases}。")
    C.report_step(5, passed=bool(passed_cases), next_step=6 if passed_cases else None, paths=[
        (out, '研究期回测：整段与逐年的毛/净年化、Sharpe、回撤、换手、时序 IC'),
        (run / 'params.json', '本次因子、板块、费率与滑点说明'),
        (run, '上述结果的本次快照'),
    ], note=note)
    return 0 if passed_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
