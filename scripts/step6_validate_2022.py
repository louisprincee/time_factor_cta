"""第 6 步：2022 验证期测试。自选因子、自选板块，与第 5 步同一套回测口径。

研究期整段扣费后没过 Sharpe 与年化门槛的池子直接拒绝，不读 2022、不记账。
通过的池子才读研究期分片和 2022 验证分片，2023 年及以后不在输入路径上。
每本书（因子+符号、板块、品种过滤、费率、滑点、执行口径共同决定的指纹）在 2022 上只测一次，
而且同一组等效因子权重 + 同一个池子（``book_key``）只要在 2022 上看过一次，换了成本或
执行口径也不再放行。
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


def split_consumed(cfg: strategy.BookConfig) -> tuple[list[str], list[tuple[str, dict]]]:
    entries = ledger.validation_entries()
    todo, consumed = [], []
    for label in strategy.case_labels(cfg):
        hits = ledger.lookup(entries, strategy.fingerprint(cfg, label))
        if not hits:
            key = strategy.book_key(cfg, label)
            hits = [e for e in entries if e.get("book_key") == key]
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
    ledger.write_validation_log()
    for label, entry in consumed:
        verdict = "通过" if entry.get("passed") else "未通过"
        source = entry.get("note") or entry.get("run_at")
        print(f"[{label}] 这本书已经在 2022 上测过（{source}，{verdict}），不再重测。")
    if todo:
        try:
            research, _, _ = strategy.screen_research(cfg, todo)
        except (FileNotFoundError, KeyError) as exc:
            print(str(exc).splitlines()[0])
            return 2
        period = strategy.research_period_label()
        blocked = []
        for case in todo:
            rows = research[(research["universe"] == case) & (research["period"] == period)]
            row = rows.iloc[0] if len(rows) else {}
            if strategy.passes(row, cfg):
                continue
            sharpe = float(row.get("net_sharpe", float("nan"))) if len(rows) else float("nan")
            ann = float(row.get("net_ann_return", float("nan"))) if len(rows) else float("nan")
            print(f"[{case}] 研究期未过门槛（扣费后 Sharpe {sharpe:.3f}，年化 {ann:.2%}），不进入 2022。")
            blocked.append(case)
        todo = [case for case in todo if case not in blocked]
    if not todo:
        print("没有尚未验证、且研究期已通过的书。")
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

    hist = history.load_history(candidates, include_validation=True)
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
                                 cfg, slippage, str(YEAR), vol=signal_set.vol)
    table["passed"] = [strategy.passes(row, cfg) for _, row in table.iterrows()]
    table.insert(0, "fingerprint", table["universe"].map(
        lambda case: strategy.fingerprint(cfg, case)))

    run = context.run_dir("step6_validation2022")
    stamp = ledger.stamp()
    entries = []
    for _, row in table.iterrows():
        entries.append({
            "fingerprint": row["fingerprint"],
            "book_key": strategy.book_key(cfg, row["universe"]),
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
    validation_log = ledger.write_validation_log()
    context.dump_json(run / "params.json", {
        "config": cfg.to_dict(), "cases": cases, "loaded": loaded,
        "skipped": hist.skipped, "external_partitions": PARTITIONS,
        "rebalance": strategy.rebalance_label(cfg.tranches),
        "vol_target": cfg.vol_target, "execution": strategy.EXECUTION,
        "slippage": "按加载的分钟数据逐年估 tick，含 2022",
    })

    passed = table.loc[table["passed"], "universe"].tolist()
    skipped = [f"{s}: {why}" for s, why in hist.skipped]
    note = f"因子 {strategy.factor_label(cfg.factors)}。"
    if skipped:
        note += f" 跳过 {skipped}。"
    paths = [
        (ledger.validation_path(), '2022 验证台账（指纹、是否通过；每本书只记一次）'),
        (validation_log, '2022 验证结果登记簿（因子、池、成本与绩效；自动同步）'),
        (run / "performance.csv", '本次验证的毛/净绩效与是否通过'),
        (run / "universe_2022_screen.csv", '2022 时点品种池筛选明细'),
        (run / "params.json", '本次配置、加载品种与跳过原因'),
        (run, '上述文件所在的本次留痕目录'),
    ]
    if passed:
        C.report_step(6, passed=True, next_step=7, paths=paths,
                      note=note + f" 通过的池子: {passed}")
        return 0
    C.report_step(6, passed=False, paths=paths,
                  note=note + "没有池子通过 2022 验证，第 7 步不会放行这些书。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
