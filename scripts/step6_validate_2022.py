"""第 6 步：2022 验证期测试。自选因子、自选板块，与第 5 步同一套回测口径。

只读研究期分片和 2022 验证分片，2023 年及以后不在输入路径上。
每本书（因子+符号、板块、品种过滤、费率、滑点共同决定的指纹）在 2022 上只测一次，
结果与是否通过一起写进 data/validation_2022/ledger.jsonl。第 7 步只放行这里通过的书。

通过条件（默认）：2022 扣费后 Sharpe >= 0.5 且扣费后年化 > 0。
门槛可以用 --min-sharpe / --min-ann-return 或配置文件 pass_criteria 调整，但门槛不进指纹，
测过一次之后改门槛也不能重测同一本书。

    python scripts/step6_validate_2022.py --factors time_combo neg_clv neg_ret_day \\
        --pools 黑色金属 农产品
    python scripts/step6_validate_2022.py --config config/strategy_example.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.data import shard_io, universe as U  # noqa: E402
from tfcta.factors import external as EXT  # noqa: E402
from tfcta.research.backtest import costs, strategy  # noqa: E402
from tfcta.research.workflow import context, history, ledger  # noqa: E402

YEAR = C.VALIDATION_YEAR
PARTITIONS = ["research", "validation_2022"]
COLUMNS = ["universe", "n_symbols", "gross_ann_return", "gross_sharpe", "net_ann_return",
           "net_sharpe", "net_max_drawdown", "turnover", "ic_ts", "ic_t", "passed"]


def split_consumed(cfg: strategy.BookConfig) -> tuple[list[str], list[tuple[str, dict]]]:
    entries = ledger.validation_entries()
    todo, consumed = [], []
    for label in strategy.case_labels(cfg):
        hits = ledger.lookup(entries, strategy.fingerprint(cfg, label))
        if hits:
            consumed.append((label, hits[-1]))
        else:
            todo.append(label)
    return todo, consumed


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
    if not shard_io.list_shards(C.VALIDATION_DIR):
        print(f"缺少 2022 验证分片: {C.VALIDATION_DIR}")
        return 2

    todo, consumed = split_consumed(cfg)
    for label, entry in consumed:
        verdict = "通过" if entry.get("passed") else "未通过"
        source = entry.get("note") or entry.get("run_at")
        print(f"[{label}] 这本书已经在 2022 上测过（{source}，{verdict}），不再重测。")
    if not todo:
        print("没有尚未验证的书。")
        return 1

    try:
        EXT.require_partitions(cfg.factors, PARTITIONS)
        pool_2022, screen = U.validation_universe()
    except FileNotFoundError as exc:
        print(exc)
        return 2
    available = set(shard_io.list_shards(C.VALIDATION_DIR))
    candidates = [s for s in strategy.candidate_symbols(cfg) if s in pool_2022 and s in available]
    if not candidates:
        print("2022 时点品种池与所选板块、验证分片没有交集。")
        return 2

    print(f"2022 品种池 {len(pool_2022)} 个，本次加载 {len(candidates)} 个：{' '.join(candidates)}")
    hist = history.load_history(candidates, include_validation=True)
    for symbol, why in hist.skipped:
        print(f"  跳过 {symbol}: {why}")
    loaded = sorted(hist.bars)
    universe = {**U.load_universe(), YEAR: [s for s in pool_2022 if s in loaded]}
    try:
        signal_set = history.signal_set(hist, universe, PARTITIONS)
        signal = strategy.equal_weight_signal(signal_set, cfg.factors)
    except KeyError as exc:
        print(str(exc).strip("'\""))
        return 2
    day_ret = history.day_returns(hist)
    if day_ret.index.max() >= pd.Timestamp(C.STRICT_OOS_START):
        raise C.HoldoutViolation("验证面板混入了 2023 年及以后的数据")
    C.assert_validation_2022_dates(day_ret.index[day_ret.index.year == YEAR], what="验证收益")
    slippage = costs.slippage_wide(hist.ticks, day_ret.index, loaded, cfg.slippage_ticks)

    cases = strategy.resolve_cases(cfg, universe[YEAR], labels=todo)
    table = strategy.evaluate_cases(signal, day_ret, universe, cases, [YEAR],
                                 cfg.fee_rate, slippage, str(YEAR))
    table["passed"] = [strategy.passes(row, cfg) for _, row in table.iterrows()]
    table.insert(0, "fingerprint", table["universe"].map(
        lambda case: strategy.fingerprint(cfg, case)))

    run = context.run_dir("step6_validation2022")
    stamp = ledger.stamp()
    entries = []
    for _, row in table.iterrows():
        entries.append({
            "fingerprint": row["fingerprint"],
            "case": row["universe"],
            "factors": strategy.factor_label(cfg.factors),
            "config": cfg.to_dict(),
            "legacy": False,
            "run_at": stamp,
            "run_dir": str(run),
            "criteria": {"min_net_sharpe": cfg.min_net_sharpe,
                         "min_net_ann_return": cfg.min_net_ann_return},
            "passed": bool(row["passed"]),
            **{k: row.get(k) for k in ("n_symbols", "net_ann_return", "net_sharpe",
                                       "net_max_drawdown", "turnover", "ic_ts", "ic_t",
                                       "members")},
        })
    ledger.append(ledger.validation_path(), entries)

    table.to_csv(run / "performance.csv", index=False, encoding="utf-8-sig")
    screen.to_csv(run / "universe_2022_screen.csv", encoding="utf-8-sig")
    hist.ticks.to_csv(run / "tick_table.csv", index=False, encoding="utf-8-sig")
    context.dump_json(run / "params.json", {
        "config": cfg.to_dict(), "cases": cases, "loaded": loaded,
        "skipped": hist.skipped, "external_partitions": PARTITIONS,
        "rebalance": strategy.REBALANCE, "execution": strategy.EXECUTION,
        "slippage": "按加载的分钟数据逐年估 tick，含 2022",
    })

    print(f"\n因子: {strategy.factor_label(cfg.factors)}")
    print(f"通过条件: 扣费后 Sharpe >= {cfg.min_net_sharpe:g} 且扣费后年化 > "
          f"{cfg.min_net_ann_return:g}")
    strategy.print_table(table, COLUMNS)
    passed = table.loc[table["passed"], "universe"].tolist()
    print(f"\n已登记验证台账: {ledger.validation_path()}")
    print(f"留痕: {run}")
    if passed:
        print(f"通过 2022 验证、可以进入第 7 步样本外的池子: {passed}")
        return 0
    print("没有池子通过 2022 验证，第 7 步不会放行这些书。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
