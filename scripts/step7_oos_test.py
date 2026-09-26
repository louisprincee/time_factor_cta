"""第 7 步：严格样本外测试（2023-01-01 起）。

两道闸门，任何一道不过都不读样本外数据：
1. 测试期必须已经结束：--oos-end 必须早于今天；
2. 这本书（与第 6 步同一个指纹）必须在 2022 验证台账里记为通过。
   没验证过、验证没通过的池子一律拒绝；全部被拒时直接退出。
同一本书在同一个样本外窗口上只测一次，结果写进 data/oos/ledger.jsonl。

样本外各年的品种池按第 2 步口径逐年重筛（第 y 年只用第 y-1 年的统计量）。
选了外部因子时，先用 step3_build_factors.py --external-partitions holdout_locked --oos-end 构造。

    python scripts/step7_oos_test.py --factors time_combo neg_clv neg_ret_day \\
        --pools 农产品 --oos-end 2025-12-31
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

PARTITIONS = ["research", "validation_2022", "holdout_locked"]
SUMMARY = ["universe", "n_symbols", "gross_ann_return", "gross_sharpe", "net_ann_return",
           "net_sharpe", "net_max_drawdown", "turnover", "ic_ts", "ic_t"]
YEARLY = ["universe", "period", "n_symbols", "net_ann_return", "net_sharpe", "net_max_drawdown"]


def gate(cfg: strategy.BookConfig, end) -> tuple[list[str], list[str]]:
    """返回 (放行的池子, 拒绝理由)。只读台账，不碰样本外数据。"""
    validation = ledger.validation_entries()
    done = ledger.read(ledger.oos_path())
    allowed, refused = [], []
    for label in strategy.case_labels(cfg):
        fp = strategy.fingerprint(cfg, label)
        hits = ledger.lookup(validation, fp)
        if not hits:
            refused.append(f"[{label}] 没有做过 2022 验证，先跑 step6_validate_2022.py")
        elif not hits[-1].get("passed"):
            refused.append(f"[{label}] 2022 验证未通过（{hits[-1].get('note') or hits[-1].get('run_at')}）")
        elif ledger.lookup(done, fp, oos_end=end.isoformat()):
            refused.append(f"[{label}] 已在截至 {end} 的样本外窗口上测过，不再重测")
        else:
            allowed.append(label)
    return allowed, refused


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    strategy.add_book_args(parser)
    parser.add_argument("--oos-end", default=C.DEFAULT_OOS_END,
                        help=f"样本外截止日期，必须已经过去，默认 {C.DEFAULT_OOS_END}")
    args = parser.parse_args()
    listed = strategy.handle_listing(args)
    if listed is not None:
        return listed
    try:
        cfg = strategy.config_from_args(args)
        end = C.to_date(args.oos_end)
    except ValueError as exc:
        print(exc)
        return 2
    if end < C.STRICT_OOS_START:
        print(f"样本外截止日期必须不早于 {C.STRICT_OOS_START}，收到 {end}")
        return 2

    try:
        C.assert_test_window_closed(end)
    except C.HoldoutViolation as exc:
        print(exc)
        return 1
    allowed, refused = gate(cfg, end)
    for line in refused:
        print(line)
    if not allowed:
        print("没有通过 2022 验证的书，不允许执行样本外测试。")
        return 1

    try:
        EXT.require_partitions(cfg.factors, PARTITIONS)
    except FileNotFoundError as exc:
        print(exc)
        return 2
    scoped_cfg = strategy.BookConfig(**{**cfg.to_dict(), "pools": [
        p for p in cfg.pools if any(p in label.split("+") for label in allowed)]})
    candidates = [s for s in strategy.candidate_symbols(scoped_cfg)
                  if shard_io.find_shard(C.HOLDOUT_DIR, s) is not None]
    oos_universe, screen = U.oos_universe(candidates, end)
    members = sorted({s for symbols in oos_universe.values() for s in symbols})
    if not members:
        print("样本外各年的时点品种池与所选板块没有交集。")
        return 2
    pool_2022, _ = U.validation_universe()

    print(f"样本外 {C.STRICT_OOS_START} .. {end}，放行 {allowed}")
    for year, symbols in oos_universe.items():
        print(f"  {year} 年品种池 {len(symbols)} 个: {' '.join(symbols)}")
    hist = history.load_history(members, include_validation=True, oos_end=end)
    for symbol, why in hist.skipped:
        print(f"  跳过 {symbol}: {why}")
    loaded = sorted(hist.bars)
    universe = {**U.load_universe(),
                C.VALIDATION_YEAR: [s for s in pool_2022 if s in loaded],
                **{y: [s for s in symbols if s in loaded] for y, symbols in oos_universe.items()}}
    try:
        signal_set = history.signal_set(hist, universe, PARTITIONS)
        signal = strategy.equal_weight_signal(signal_set, cfg.factors)
    except KeyError as exc:
        print(str(exc).strip("'\""))
        return 2
    day_ret = history.day_returns(hist)
    if day_ret.index.max() > pd.Timestamp(end):
        raise C.HoldoutViolation(f"样本外面板越过了截止日期 {end}")
    slippage = costs.slippage_wide(hist.ticks, day_ret.index, loaded, cfg.slippage_ticks)

    years = U.oos_years(end)
    label = f"{C.STRICT_OOS_START}..{end}"
    cases = strategy.resolve_cases(cfg, loaded, labels=allowed)
    table = strategy.evaluate_cases(
        signal, day_ret, universe, cases, years, cfg.fee_rate, slippage, label,
        window=(pd.Timestamp(C.STRICT_OOS_START), pd.Timestamp(end)))
    table.insert(0, "fingerprint", table["universe"].map(
        lambda case: strategy.fingerprint(cfg, case)))

    run = context.run_dir("step7_oos")
    stamp = ledger.stamp()
    overall = table[table["period"] == label]
    ledger.append(ledger.oos_path(), [{
        "fingerprint": row["fingerprint"],
        "case": row["universe"],
        "factors": strategy.factor_label(cfg.factors),
        "config": cfg.to_dict(),
        "oos_start": C.STRICT_OOS_START.isoformat(),
        "oos_end": end.isoformat(),
        "run_at": stamp,
        "run_dir": str(run),
        **{k: row.get(k) for k in ("n_symbols", "net_ann_return", "net_sharpe",
                                   "net_max_drawdown", "turnover", "ic_ts", "ic_t")},
    } for _, row in overall.iterrows()])

    ledger.OOS_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(run / "performance.csv", index=False, encoding="utf-8-sig")
    table.to_csv(ledger.OOS_ROOT / "latest_performance.csv", index=False, encoding="utf-8-sig")
    if not screen.empty:
        screen.to_csv(run / "universe_oos_screen.csv", index=False, encoding="utf-8-sig")
    context.dump_json(run / "params.json", {
        "config": cfg.to_dict(), "allowed": allowed, "refused": refused,
        "oos_end": end.isoformat(), "universe_oos": oos_universe,
        "loaded": loaded, "skipped": hist.skipped, "external_partitions": PARTITIONS,
    })

    print(f"\n因子: {strategy.factor_label(cfg.factors)}")
    print(f"== 样本外 {label}")
    strategy.print_table(overall, SUMMARY)
    yearly = table[table["period"] != label]
    if not yearly.empty:
        print("\n== 逐年")
        strategy.print_table(yearly, YEARLY)
    print(f"\n已登记样本外台账: {ledger.oos_path()}\n留痕: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
