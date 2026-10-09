"""早盘元策略（主策略·亏损反手·30笔）：严格样本外的最终测试。

    python scripts/oos_final.py --check               # 只重算研究期和 2022 并核对，不读样本外
    python scripts/oos_final.py --years 2023
    python scripts/oos_final.py --years 2024
    python scripts/oos_final.py --years 2023-2025

口径见 config/oos_protocol.json，策略冻结在 config/morning_meta.json。
读样本外之前先写台账 data/oos/ledger.jsonl；同一配置和年份已有结果时须加 --rerun。
未选择计分的年份只用于连续历史上下文；明确包含在 --years 中的年份计入绩效。输出在 runs/oos/<年份>/。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tfcta import config as C
from tfcta.data import universe as U
from tfcta.research import context, costs, stats
import morning_features
import research_morning_meta as meta
import research_morning_oor as oor

PROTOCOL = ROOT / "config" / "oos_protocol.json"
OUT = ROOT / "runs" / "oos"
HASHED = ["config/oos_protocol.json", "config/morning_meta.json", "config/morning_oor.json",
          "scripts/oos_final.py", "scripts/research_morning_meta.py", "scripts/research_morning_oor.py",
          "scripts/morning_features.py"]
TICKS = {"主口径": 1.0, "滑点2跳": 2.0}
SAVED = {"早盘元策略": (meta.OUT, "主策略·亏损反手·30笔"), "主策略·原样": (meta.OUT, "主策略·原样")}


def parse_years(text, allowed):
    """'2024' / '2024-2025' / '2024,2025' → 升序年份列表，只允许协议里登记的年份。"""
    years = set()
    for piece in str(text).replace("，", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = (int(x) for x in piece.split("-", 1))
            if a > b:
                raise ValueError(f"年份区间写反了: {piece}")
            years.update(range(a, b + 1))
        else:
            years.add(int(piece))
    if not years:
        raise ValueError("没有给年份")
    bad = sorted(years - set(allowed))
    if bad:
        raise ValueError(f"只能选 {allowed} 中的年份，收到 {bad}")
    return sorted(years)


def label_of(years):
    if years == list(range(years[0], years[-1] + 1)):
        return str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
    return "_".join(map(str, years))


def fingerprint(hashes, years):
    text = json.dumps({"sha256": hashes, "years": years}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def check_ledger(ledger, strategy, finger, hashes, rerun, previous=()):
    """同一指纹已有结果：须显式 --rerun；此前用别的配置跑过：返回改动过的配置，记入台账。
    previous 是这一系列测试在台账里用过的旧名字。"""
    names = {strategy, *previous}
    rows = [r for r in ledger if r.get("kind") == "final_evaluation" and r.get("strategy") in names]
    done = [r for r in rows if r.get("status") == "result" and r.get("fingerprint") == finger]
    if done and not rerun:
        raise SystemExit(f"这组配置和年份已在 {done[-1]['run_at']} 跑过，结果在 {done[-1].get('out')}；"
                         "确需重跑加 --rerun")
    return sorted({k for r in rows if r.get("status") == "result"
                   for k, v in r.get("sha256", {}).items()
                   if k.endswith(".json") and hashes.get(k) not in (None, v)})


def load_specs(protocol):
    candidate = protocol["candidates"]["早盘元策略"]
    spec = json.loads((ROOT / candidate["config"]).read_text(encoding="utf-8"))
    if spec.get("chosen") != "主策略·亏损反手·30笔" or not spec.get("frozen_at"):
        raise SystemExit(f"{candidate['config']} 没有冻结为 主策略·亏损反手·30笔")
    base = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    return candidate, base


# ---------------------------------------------------------------- 早盘元策略

def morning_books(table, base, candidate):
    """元策略和参照（不切换的主策略）；table 是研究期接 2022（接样本外）的连续特征表。"""
    side = oor.base_books(table, base)[candidate["book"]]
    score = meta.trailing_mean(table, side, candidate["window_trades"])
    return {"早盘元策略": meta.switched(side, score, candidate["mode"]), "主策略·原样": side}


def morning_returns(table, books, base):
    calendar = pd.DatetimeIndex(sorted(table.trading_date.unique()))
    out = {}
    for scenario in TICKS:
        for name, side in books.items():
            weight = oor.weights(table, side, base["sizing"])
            out[(scenario, name)] = oor.daily(table, weight, scenario, calendar)
    return out


def morning_trades(table, books, years):
    """每年的笔数、顺势笔数和主口径单笔净收益（只统计 years 里的年份）。"""
    rows = []
    year = table.trading_date.dt.year
    for name, side in books.items():
        net = side * table.gross_1130 - table.cost_1130
        follow = side == np.sign(table.dev15)
        for y in years:
            m = (side != 0) & (year == y)
            rows.append({"候选": name, "年份": y, "笔数": int(m.sum()), "顺势笔数": int((m & follow).sum()),
                         "单笔净bp": net[m].mean() * 1e4 if m.any() else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 汇总

def consistency(series):
    """研究期和 2022 部分必须与已落盘的研究结果逐日一致，否则说明数据或代码变了。"""
    notes = []
    for name, ret in series.items():
        folder, column = SAVED[name]
        for partition in ("research", "validation_2022"):
            path = folder / partition / "daily.csv"
            saved = pd.read_csv(path, index_col=0, parse_dates=True, encoding="utf-8-sig")[column].fillna(0.0)  # 无收益日记 0
            again = ret.reindex(saved.index).fillna(0.0)
            if not np.allclose(again, saved, atol=1e-10):
                worst = float((again - saved).abs().max())
                raise ValueError(f"{name} 的 {partition} 部分与已落盘结果不一致（最大差 {worst:.2e}）：{path}")
            notes.append(f"{name} {partition}: 与已落盘结果一致（{len(saved)} 日）")
    return notes


def describe(ret, label, scenario):
    perf = stats.performance(ret)
    return {"候选": label, "口径": scenario, "夏普": stats.sharpe_ratio(ret), "累计收益": float((1 + ret).prod() - 1),
            "年化收益": perf["ann_return"], "年化波动": perf["ann_vol"], "最大回撤": perf["max_drawdown"],
            "交易日": int(len(ret))}


def yearly(series, years, scored):
    rows = []
    for name, ret in series.items():
        for y in years:
            part = ret[ret.index.year == y]
            rows.append({"候选": name, "年份": y, "计入绩效": y in scored, "夏普": stats.sharpe_ratio(part),
                         "收益": float((1 + part).prod() - 1),
                         "年内回撤": float(((1 + part).cumprod() / (1 + part).cumprod().cummax().clip(lower=1.0) - 1).min())})
    return pd.DataFrame(rows)


def rebuild(base, candidate, oos_table=None):
    """研究期接 2022（再接样本外）的连续特征表上的元策略和参照。"""
    table = oor.prepare_through_2022(base, 2016)
    if oos_table is not None:
        table = pd.concat([table, oos_table], ignore_index=True)
    books = morning_books(table, base, candidate)
    morning = morning_returns(table, books, base)
    series = {scenario: {name: morning[(scenario, name)] for name in books} for scenario in TICKS}
    return table, books, series


def preflight(protocol):
    missing = [str(folder / p / "daily.csv") for folder, _ in SAVED.values() for p in ("research", "validation_2022")
               if not (folder / p / "daily.csv").exists()]
    if missing:
        raise SystemExit(f"先跑研究期和 2022：python scripts/research_morning_meta.py [--validation-2022]，缺 {missing}")
    if not costs.fee_path("oos").exists():
        raise SystemExit("先下载样本外手续费：python download/getFeeHistory.py --partition oos")
    if not (oor.FEATURES / "validation_2022.parquet").exists():
        raise SystemExit("先生成早盘特征：python scripts/morning_features.py")


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", help="协议 years_allowed 里的年份，如 2023-2025")
    parser.add_argument("--check", action="store_true", help="只重算研究期和 2022 并核对，不读样本外")
    parser.add_argument("--rerun", action="store_true", help="同一配置和年份已有结果时仍重跑")
    args = parser.parse_args()
    candidate, base = load_specs(protocol)
    preflight(protocol)
    if args.check or not args.years:
        _, _, series = rebuild(base, candidate)
        print("\n".join(consistency(series["主口径"])))
        print("检查通过，没有读样本外。")
        return
    years = parse_years(args.years, protocol["years_allowed"])
    end = C.to_date(f"{years[-1]}-12-31")
    files = [ROOT / f for f in HASHED]
    hashes = {Path(f).name: C._sha256(f) for f in files}
    finger = fingerprint(hashes, years)
    strategy = protocol["name"]
    changed = check_ledger(C.read_ledger(), strategy, finger, hashes, args.rerun, protocol.get("previous_names", []))
    note = f"years={years}；按所选年份计分" + (f"；此前样本外结果对应的配置已改动: {changed}" if changed else "")
    try:
        with C.final_evaluation(protocol, files, end, strategy, note=note) as ticket:
            oos_years = list(range(C.STRICT_OOS_START.year, end.year + 1))
            universe, _ = U.oos_universe([s for s in C.COMMODITY_SYMBOLS if s in costs.MULTIPLIER], end)
            universe = {y: [s for s in universe.get(y, []) if s in costs.MULTIPLIER] for y in oos_years}
            if not all(universe.values()):
                raise ValueError(f"样本外有年份品种池为空: {[y for y, v in universe.items() if not v]}")
            names = sorted({s for group in universe.values() for s in group})
            morning_features.build("oos", end=end, names=names)
            oos_table = oor.prepare("oos", base, years=(oos_years[0], oos_years[-1]), pools=universe)
            table, books, series = rebuild(base, candidate, oos_table)
            notes = consistency(series["主口径"])
    except Exception as error:  # 中途失败也留痕：台账里 opened 之后跟一条 aborted
        C.append_ledger({"kind": "final_evaluation", "status": "aborted", "strategy": strategy,
                         "run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "fingerprint": finger,
                         "years": years, "error": f"{type(error).__name__}: {error}"[:500]})
        raise
    write_outputs(ticket, years, oos_years, end, finger, changed, table, books, series, notes)


def write_outputs(ticket, years, oos_years, end, finger, changed, table, books, series, notes):
    label = label_of(years)
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    scored = {s: {k: v[v.index.year.isin(years)] for k, v in group.items()} for s, group in series.items()}
    summary = pd.DataFrame([describe(ret, name, scenario) for scenario, group in scored.items()
                            for name, ret in group.items()])
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    year_table = yearly(series["主口径"], oos_years, years)
    year_table.to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")
    trades = morning_trades(table, books, oos_years)
    trades.to_csv(out / "morning_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scored["主口径"]).to_csv(out / "daily.csv", encoding="utf-8-sig")
    print("图表：", context.plot_performance(scored["主口径"], out / "plots" / label, f"样本外 {label}（主口径，扣费后）"))

    def pick(name, scenario="主口径"):
        row = summary[(summary.候选 == name) & (summary.口径 == scenario)].iloc[0]
        return {k: float(row[k]) for k in ("夏普", "累计收益", "年化收益", "年化波动", "最大回撤")}

    C.append_ledger({**{k: ticket[k] for k in ("kind", "strategy", "chosen", "frozen_at", "sha256")},
                     "status": "result", "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                     "fingerprint": finger, "years": years, "oos_end": str(end), "out": str(out),
                     "早盘元策略": pick("早盘元策略"), "主策略·原样": pick("主策略·原样"),
                     "早盘元策略·滑点2跳": pick("早盘元策略", "滑点2跳"), "config_changed": changed})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n".join(notes))
    print(f"\n样本外 {label}（所选年份计入绩效）")
    print(summary.round(4).to_string(index=False))
    print("\n逐年（主口径）：")
    print(year_table.round(4).to_string(index=False))
    print("\n早盘逐年笔数：")
    print(trades.round(2).to_string(index=False))
    if changed:
        print(f"\n注意：此前样本外结果对应的配置已改动 {changed}，已记入台账。")
    print("\n结果：", out)


if __name__ == "__main__":
    main()
