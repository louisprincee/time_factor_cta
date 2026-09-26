"""第 5 步：研究期回测。自选因子、自选板块，等权周频，只读 2016–2021。

默认是四个时间因子的周频书：ts_high×-1、ts_low、dfp_max、dfp_top3 等权，不分板块。
252 日事前 z 分数等权后截到 [-1, 1]，每周最后一个交易日更新，次日开盘成交，
手续费加 tick 滑点。拼接 walk-forward 的六个测试年。

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
from tfcta.factors import library  # noqa: E402
from tfcta.research.backtest import costs, engine  # noqa: E402
from tfcta.research.backtest import strategy  # noqa: E402
from tfcta.research.workflow import context  # noqa: E402

OVERALL = f"{C.WF_TEST_YEARS_LIST[0]}-{C.WF_TEST_YEARS_LIST[-1]}"
SUMMARY_COLUMNS = ["universe", "n_symbols", "gross_ann_return", "gross_sharpe",
                   "net_ann_return", "net_sharpe", "net_max_drawdown", "turnover",
                   "ic_ts", "ic_t"]
YEAR_COLUMNS = ["universe", "period", "n_symbols", "net_ann_return", "net_sharpe",
                "net_max_drawdown"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    strategy.add_book_args(parser)
    args = parser.parse_args()
    listed = strategy.handle_listing(args)
    if listed is not None:
        return listed
    try:
        cfg = strategy.config_from_args(args)
    except ValueError as exc:
        print(exc)
        return 2

    reason = context.not_ready_reason()
    if reason:
        print(reason)
        return 2
    universe, symbols, day_ret = context.load_context(None)
    if not symbols or day_ret.empty:
        print("研究期品种池或收益为空。")
        return 2
    years = [fold["test_year"] for fold in engine.walk_forward_folds()]

    try:
        signal_set = library.load(symbols)
        signal = strategy.equal_weight_signal(signal_set, cfg.factors)
    except (FileNotFoundError, KeyError) as exc:
        print(f"{str(exc).splitlines()[0]}\n请先运行 step3_build_factors.py，或用 --list-factors 查看可选因子。")
        return 2
    slippage, cost_note = costs.research_slippage(symbols, day_ret.index, cfg.slippage_ticks)
    cases = strategy.resolve_cases(cfg, symbols)
    table = strategy.evaluate_cases(signal, day_ret, universe, cases, years,
                                 cfg.fee_rate, slippage, OVERALL)
    table.insert(0, "fingerprint", table["universe"].map(
        lambda case: strategy.fingerprint(cfg, case)))

    C.ensure_dirs()
    run = context.run_dir("step5_backtest")
    out = C.RESEARCH_OUT_DIR / "backtest_research.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    table.to_csv(run / out.name, index=False, encoding="utf-8-sig")
    context.dump_json(run / "params.json", {
        "config": cfg.to_dict(),
        "cases": cases,
        "years": years,
        "rebalance": strategy.REBALANCE,
        "execution": strategy.EXECUTION,
        "slippage_note": cost_note,
        "sharpe": "日均值 / 日标准差 × sqrt(252)，无风险利率 0",
    })

    print(f"因子: {strategy.factor_label(cfg.factors)}（符号乘在原始值上）")
    print(f"成本: 手续费 {cfg.fee_rate:.5f}；{cost_note}")
    overall = table[table["period"] == OVERALL]
    print(f"\n== 研究期 {OVERALL}（拼接测试年）")
    strategy.print_table(overall, SUMMARY_COLUMNS)
    yearly = table[table["period"] != OVERALL]
    if not yearly.empty:
        print("\n== 逐年")
        strategy.print_table(yearly, YEAR_COLUMNS)
    empty = [case for case, members in cases.items() if not members]
    if empty:
        print(f"\n这些池子在研究期品种池里没有品种: {empty}")
    print(f"\n结果: {out}\n留痕: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
