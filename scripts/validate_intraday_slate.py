"""一次性检验：多策略清单。冻结配置里的每本书 2022 只测一次，过了的才读 2023–2025。

事先写定，看到结果前不改：
- 书与参数格来自 config/intraday_slate.json（research_intraday_slate.py 用 2019–2021 选定后冻结）；
- 组合书 slate_combo 的成员与权重同样冻结；
- 主口径 = 历史手续费 + 每边 1 tick；池 = 研究期条件池 + 2022 池；
- 2022 净 Sharpe ≥ 0.5 且几何净年化 > 0 才读 2023–2025；样本外沿用同一格、同一权重。

每本书的指纹由规格串算出，测过的书在台账里，脚本拒绝重测。输入数据有缺陷的书先用 ``--void``
作废（写明原因），再不带参数跑一次：只重跑作废的书，规格不变。

用法：
  python scripts/validate_intraday_slate.py          # 全部冻结书一次性测 2022（或只重跑作废的书）
  python scripts/validate_intraday_slate.py --void carry_daily --reason "..."   # 作废 2022 记录
  python scripts/validate_intraday_slate.py --oos    # 2022 通过的书读 2023–2025
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfcta import config as C  # noqa: E402
from tfcta.data import shard_io, universe as U  # noqa: E402
from tfcta.research.backtest import costs  # noqa: E402
from tfcta.research.intraday import walk_forward as W  # noqa: E402
from tfcta.research.workflow import context, ledger  # noqa: E402
import research_intraday_slate as R  # noqa: E402
import validate_orb_ml as V  # noqa: E402

OOS_END = C.DEFAULT_OOS_END
MIN_SHARPE = V.MIN_SHARPE
PARTITIONS_2022 = ("research", "validation_2022")
PARTITIONS_OOS = ("research", "validation_2022", "holdout_locked")
TRAIN_TOL = 0.02


def load_frozen() -> dict:
    if not R.FROZEN.exists():
        raise FileNotFoundError(f"没有 {R.FROZEN}，先跑 research_intraday_slate.py")
    frozen = json.loads(R.FROZEN.read_text(encoding="utf-8"))
    if list(frozen["train"]) != W.train_years(C.VALIDATION_YEAR):
        raise RuntimeError("冻结训练年份和走步窗口不一致")
    return frozen


def specs(frozen) -> dict[str, str]:
    out = {}
    for name, b in frozen["books"].items():
        params = json.dumps(b["params"], sort_keys=True)
        out[name] = f"slate:{name}:{b['variant']}:{params}:train2019-2021:test2022:hist+1tick"
    w = frozen["combo"]["weights"]
    if w:
        members = ",".join(f"{n}={frozen['books'][n]['variant']}@{w[n]:.6f}" for n in sorted(w))
        out[R.COMBO] = f"slate:{R.COMBO}:{members}:train2019-2021:test2022:hist+1tick"
    return out


def fingerprint(spec: str) -> str:
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


def build(frozen, names: list[str], universe: dict, partitions, oos_end=None):
    """冻结书的逐腿、面板与池级序列。``oos_end`` 为空时不打开锁定分片。"""
    symbols = sorted({s for v in universe.values() for s in v})
    wanted = {n for n in names if n != R.COMBO}
    if R.COMBO in names:
        wanted |= set(frozen["combo"]["weights"])
    books = {n: [frozen["books"][n]["variant"]] for n in sorted(wanted)}
    need_carry = bool(wanted & W.CARRY_STRATEGIES)
    costs.load_fee_history([s for s in symbols if s in costs.MULTIPLIER])
    end = oos_end or f"{C.VALIDATION_YEAR}-12-31"
    carry = R.carry_frame(symbols, partitions, end) if need_carry else pd.DataFrame()
    frames, ctxs, skipped = [], {}, []
    for symbol in symbols:
        minute = V.minute_frame(symbol, oos_end)
        if minute is None:
            skipped.append(symbol)
            continue
        tick = W.tick_dict(pd.DataFrame(costs.tick_rows(
            symbol, minute["close"], minute["trading_date"])))[symbol]
        col = carry[symbol] if symbol in carry and carry[symbol].notna().any() else None
        si = W.slate_input(symbol, minute, tick, col)
        frames.append(W.slate_legs(si, books))
        ctxs[symbol] = si.ctx
        del minute, si
    legs = pd.concat(frames, ignore_index=True)
    last = pd.DatetimeIndex(legs["date"]).max()
    if last > pd.Timestamp(end):
        raise RuntimeError(f"腿含 {last.date()}，超过 {end}")
    scoped = {y: [s for s in v if s in ctxs] for y, v in universe.items()}
    panel = W.slate_panel(ctxs)
    member = panel.members(scoped)
    series = W.slate_series(legs, panel, member, R.ROBUST)
    return legs, series, scoped, skipped


def book_series(frozen, series, name, scenario) -> pd.Series:
    if name == R.COMBO:
        w = frozen["combo"]["weights"]
        return R.combo({n: book_series(frozen, series, n, scenario) for n in w}, w)
    return series[(name, frozen["books"][name]["variant"], scenario)]


def check_train(frozen, series, names) -> None:
    """重算的 2019–2021 主口径 Sharpe 必须和冻结时一致，否则数据变了，不测。"""
    for n in names:
        if n == R.COMBO:
            continue
        s = book_series(frozen, series, n, "main")
        got = R.sharpe(s[s.index.year.isin(frozen["train"])])
        want = frozen["books"][n]["train_sharpe"]
        if not (np.isfinite(got) and abs(got - want) <= TRAIN_TOL):
            raise RuntimeError(f"{n} 训练期 Sharpe 重算 {got:.3f}，冻结时 {want:.3f}")


def ledger_row(fp, spec, frozen, name, run, summary, n_symbols, passed, note, **extra) -> dict:
    book = frozen["books"].get(name, {})
    return {
        "fingerprint": fp, "book_key": fp, "case": "全部", "factors": spec.rsplit(":train", 1)[0],
        "config": {
            "fee_rate": None, "slippage_ticks": 1.0, "tranches": 0, "vol_target": 0.0,
            "execution": book.get("kind", "combo"), "fee_schedule": "hist", "train": "2019-2021",
            "strategy": name, "variant": book.get("variant"), "params": book.get("params"),
            "weights": frozen["combo"]["weights"] if name == R.COMBO else None,
        },
        "run_at": ledger.stamp(), "run_dir": str(run),
        "criteria": {"min_net_sharpe": MIN_SHARPE, "min_net_ann_return": 0.0},
        "passed": passed, "n_symbols": n_symbols,
        "net_ann_return": summary["net_ann"], "net_sharpe": summary["net_sharpe"],
        "net_max_drawdown": summary["net_mdd"], "note": note, **extra,
    }


def report(frozen, series, names, years) -> pd.DataFrame:
    rows = []
    for n in names:
        row = {"strategy": n, "variant": frozen["books"].get(n, {}).get("variant", "combo")}
        main = book_series(frozen, series, n, "main")
        main = main[main.index.year.isin(years)]
        row.update(W.summarize(main))
        for sc in R.ROBUST:
            if sc != "main":
                s = book_series(frozen, series, n, sc)
                row[f"sr_{sc}"] = R.sharpe(s[s.index.year.isin(years)])
        row["passed"] = V.passes(row)
        rows.append(row)
    return pd.DataFrame(rows)


def run_2022(frozen, spec_map, rerun: bool = False) -> int:
    names = list(spec_map)
    universe = {**V.research_universe(), C.VALIDATION_YEAR: V.validation_members()}
    print(f"2022 检验：{len(names)} 本书，不打开 2023 及以后", flush=True)
    legs, series, scoped, skipped = build(frozen, names, universe, PARTITIONS_2022)
    check_train(frozen, series, names)
    table = report(frozen, series, names, (C.VALIDATION_YEAR,))
    run = context.run_dir("intraday_slate_validation2022")
    table.to_csv(run / "performance.csv", index=False, encoding="utf-8-sig")
    pd.concat({n: book_series(frozen, series, n, "main") for n in names}, axis=1).loc[
        str(C.VALIDATION_YEAR)].to_csv(run / "daily_net.csv", encoding="utf-8-sig")
    context.dump_json(run / "params.json", {
        "specs": spec_map, "frozen": frozen, "universe_2022": scoped.get(C.VALIDATION_YEAR, []),
        "skipped": skipped})
    n_symbols = len(scoped.get(C.VALIDATION_YEAR, []))
    rows = []
    for _, r in table.iterrows():
        n = r["strategy"]
        if r["n_days"] < 200:
            raise RuntimeError(f"{n} 2022 只有 {r['n_days']:.0f} 天")
        rows.append(ledger_row(fingerprint(spec_map[n]), spec_map[n], frozen, n, run, r.to_dict(),
                               n_symbols, bool(r["passed"]),
                               f"多策略清单 {n}，2019–2021 冻结格，2022 一次性，主口径 1 tick"
                               + ("；作废后按同一规格重跑" if rerun else "")))
    ledger.append(ledger.validation_path(), rows)
    ledger.write_validation_log()
    print_table(table, "2022")
    print(f"2022 通过 {int(table['passed'].sum())}/{len(table)}，留痕 {run}", flush=True)
    return 0


def print_table(table, label) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["strategy", "variant", "net_sharpe", "net_ann", "net_vol", "net_mdd", "sr_gross",
            "sr_slip2", "sr_fee_x2", "sr_fee_2026", "sr_close_yday", "passed"]
    cols += [c for c in table if c.startswith("sr_20")]
    print(f"\n{label}（主口径）")
    print(table[[c for c in cols if c in table]].round(3).to_string(index=False))


def run_oos(frozen, spec_map, entries) -> int:
    C.assert_test_window_closed(OOS_END)
    names = list(entries)
    names_hold = sorted(set(V.tradable(shard_io.list_shards(C.HOLDOUT_DIR))))
    oos_universe, screen = U.oos_universe(names_hold, OOS_END)
    oos_universe = {y: V.tradable(v) for y, v in oos_universe.items()}
    universe = {**V.research_universe(), C.VALIDATION_YEAR: V.validation_members(), **oos_universe}
    years = tuple(U.oos_years(OOS_END))
    print(f"样本外：{names}，至 {OOS_END}", flush=True)
    legs, series, scoped, skipped = build(frozen, names, universe, PARTITIONS_OOS, OOS_END)
    check_train(frozen, series, names)
    for n in names:                       # 2022 必须复现台账里的结果
        s = book_series(frozen, series, n, "main")
        got = R.sharpe(s[s.index.year == C.VALIDATION_YEAR])
        if abs(got - float(entries[n]["net_sharpe"])) > TRAIN_TOL:
            raise RuntimeError(f"{n} 2022 Sharpe 重算 {got:.3f}，台账 {entries[n]['net_sharpe']:.3f}")
    table = report(frozen, series, names, years)
    run = context.run_dir("intraday_slate_oos")
    table.to_csv(run / "performance.csv", index=False, encoding="utf-8-sig")
    daily = pd.concat({n: book_series(frozen, series, n, "main") for n in names}, axis=1)
    daily[daily.index.year.isin(years)].to_csv(run / "daily_net.csv", encoding="utf-8-sig")
    if not screen.empty:
        screen.to_csv(run / "universe_oos_screen.csv", index=False, encoding="utf-8-sig")
    context.dump_json(run / "params.json", {
        "specs": {n: spec_map[n] for n in names}, "oos_end": OOS_END.isoformat(),
        "universe_oos": {str(k): v for k, v in oos_universe.items()}, "skipped": skipped})
    n_symbols = len({s for y in years for s in scoped.get(y, [])})
    rows = []
    for _, r in table.iterrows():
        n = r["strategy"]
        entry = ledger_row(fingerprint(spec_map[n]), spec_map[n], frozen, n, run, r.to_dict(),
                           n_symbols, False, f"多策略清单 {n}，格冻结在 2019–2021，主口径 1 tick",
                           oos_start=C.STRICT_OOS_START.isoformat(), oos_end=OOS_END.isoformat())
        entry.pop("passed", None)
        entry.pop("criteria", None)
        rows.append(entry)
    ledger.append(ledger.oos_path(), rows)
    ledger.write_validation_log()
    print_table(table, f"{C.STRICT_OOS_START}..{OOS_END}")
    print(f"样本外留痕 {run}", flush=True)
    return 0


def void(spec_map, done, done_oos, names, reason) -> int:
    """作废 2022 记录。只能作废测过、还没读样本外的书，规格不变。"""
    if not reason:
        raise SystemExit("--void 需要 --reason")
    rows = []
    for n in names:
        fp = fingerprint(spec_map[n])
        hit = ledger.lookup(done, fp)
        if not hit:
            raise SystemExit(f"{n} 没有有效的 2022 记录可作废")
        if ledger.lookup(done_oos, fp):
            raise SystemExit(f"{n} 已读样本外，不能作废 2022")
        rows.append(ledger.void_entry(fp, f"{n}：{reason}", hit[-1].get("run_dir")))
    ledger.append(ledger.validation_path(), rows)
    ledger.write_validation_log()
    print(f"作废 {names}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="多策略清单：2022 一次性检验，可选再读 2023–2025")
    ap.add_argument("--oos", action="store_true", help="2022 通过的书读 2023–2025")
    ap.add_argument("--void", nargs="+", metavar="BOOK", help="作废这些书的 2022 记录（数据缺陷）")
    ap.add_argument("--reason", help="作废原因，和 --void 一起用")
    args = ap.parse_args()
    frozen = load_frozen()
    spec_map = specs(frozen)
    done_oos = ledger.read(ledger.oos_path())
    done = ledger.validation_entries()
    for n, spec in spec_map.items():
        print(f"{n:22s} {fingerprint(spec)}  {spec}", flush=True)
    if args.void:
        return void(spec_map, done, done_oos, args.void, args.reason)
    if not args.oos:
        pending = {n: s for n, s in spec_map.items() if not ledger.lookup(done, fingerprint(s))}
        if not pending:
            print("全部书的 2022 已经测过，不再重测。")
            return 2
        voided = {e["fingerprint"] for e in done if e.get("void")}
        rerun = len(pending) < len(spec_map)
        if rerun and any(fingerprint(s) not in voided for s in pending.values()):
            print(f"部分书已测、部分未测且没有作废记录，拒绝：{sorted(pending)}")
            return 2
        if rerun:
            print(f"只重跑作废的书：{sorted(pending)}", flush=True)
        return run_2022(frozen, pending, rerun)
    entries = {}
    for n, s in spec_map.items():
        hit = ledger.lookup(done, fingerprint(s))
        if not hit:
            print(f"{n} 还没测 2022，先不带 --oos 跑一次。")
            return 1
        if ledger.lookup(done_oos, fingerprint(s)):
            continue
        e = hit[-1]
        sr, ann = e.get("net_sharpe"), e.get("net_ann_return")
        if np.isfinite(sr) and np.isfinite(ann) and float(sr) >= MIN_SHARPE and float(ann) > 0:
            entries[n] = e
    if not entries:
        print("没有 2022 通过且未做样本外的书。")
        return 1
    return run_oos(frozen, spec_map, entries)


if __name__ == "__main__":
    raise SystemExit(main())
