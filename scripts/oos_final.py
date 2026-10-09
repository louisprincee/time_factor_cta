"""早盘元策略 + 四条日频腿：2024–2025 严格样本外的最终测试。

    python scripts/oos_final.py --check               # 只重算研究期和 2022 并核对，不读样本外
    python scripts/oos_final.py --years 2024
    python scripts/oos_final.py --years 2024-2025

口径见 config/oos_protocol.json，两个候选冻结在 config/morning_meta.json、config/multi_leg.json。
读样本外之前先写台账 data/oos/ledger.jsonl；同一配置和年份已有结果时须加 --rerun。
2023 只作预热（信号、最近 30 笔、波动乘数、品种池），不计入绩效。输出在 runs/oos/<年份>/。
"""
import argparse
import dataclasses
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
from tfcta.data import bars as B, universe as U, validation_import as VI
from tfcta.factors import daily as D, external, library
from tfcta.research import context, costs, stats, study
import morning_features
import research_morning_meta as meta
import research_morning_oor as oor
import research_multi_leg as ml

PROTOCOL = ROOT / "config" / "oos_protocol.json"
OUT = ROOT / "runs" / "oos"
HASHED = ["config/oos_protocol.json", "config/morning_meta.json", "config/multi_leg.json", "config/morning_oor.json",
          "scripts/oos_final.py", "scripts/research_morning_meta.py", "scripts/research_multi_leg.py",
          "scripts/research_morning_oor.py", "scripts/morning_features.py"]
TICKS = {"主口径": 1.0, "滑点2跳": 2.0}
SAVED = {"早盘元策略": (meta.OUT, "主策略·亏损反手·30笔"), "四条日频腿": (ml.OUT, "四条日频腿")}


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


def check_ledger(ledger, strategy, finger, hashes, rerun):
    """同一指纹已有结果：须显式 --rerun；此前用别的配置跑过：返回改动过的配置，记入台账。"""
    rows = [r for r in ledger if r.get("kind") == "final_evaluation" and r.get("strategy") == strategy]
    done = [r for r in rows if r.get("status") == "result" and r.get("fingerprint") == finger]
    if done and not rerun:
        raise SystemExit(f"这组配置和年份已在 {done[-1]['run_at']} 跑过，结果在 {done[-1].get('out')}；"
                         "确需重跑加 --rerun")
    return sorted({k for r in rows if r.get("status") == "result"
                   for k, v in r.get("sha256", {}).items()
                   if k.endswith(".json") and hashes.get(k) not in (None, v)})


def load_specs(protocol):
    specs = {name: json.loads((ROOT / c["config"]).read_text(encoding="utf-8"))
             for name, c in protocol["candidates"].items()}
    expected = {"早盘元策略": "主策略·亏损反手·30笔", "四条日频腿": "四条日频腿"}
    for name, spec in specs.items():
        if spec.get("chosen") != expected[name] or not spec.get("frozen_at"):
            raise SystemExit(f"{protocol['candidates'][name]['config']} 没有冻结为 {expected[name]}")
    base = json.loads((ROOT / specs["早盘元策略"]["base_config"]).read_text(encoding="utf-8"))
    return specs["早盘元策略"], specs["四条日频腿"], base


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


# ---------------------------------------------------------------- 四条日频腿

def daily_oos(universe, end):
    """样本外日频数据：行情从 2018 年接起做信号预热，收益、成本只取样本外。"""
    names = sorted({s for group in universe.values() for s in group})
    per, statuses = {}, {}
    for symbol in names:
        minutes = B.load_minutes(symbol, include_validation=True, oos_end=end,
                                 columns=list(B.DAILY_FIELDS) + ["trading_date"])
        minutes["trading_date"] = pd.to_datetime(minutes.trading_date)
        before = minutes[minutes.trading_date < pd.Timestamp(C.STRICT_OOS_START)]
        after = minutes[minutes.trading_date >= pd.Timestamp(C.STRICT_OOS_START)]
        statuses[symbol], _ = VI.boundary_check(before[before.trading_date.dt.year == 2022], after)
        day = B.daily_bars(minutes)
        per[symbol] = day[day.index >= "2018-01-01"]
    ok, note = VI.boundary_verdict(statuses)
    if not ok:
        raise ValueError(f"2022 → 样本外复权基准不衔接：{note}")
    print("复权衔接：", note, flush=True)
    bars = B.wide_by_field(per)
    close, adjusted = bars["close"], bars["closew"]
    pools = {**U.load_universe(), 2022: list(U.validation_universe()[0]), **universe}
    signals = library.SignalSet(bars=bars)
    signals.signed["tsmom"] = D.tsmom(close, adjusted)
    signals.signed["ma_break_20"] = D.ma_breakout(adjusted, D.REVIEW_VOL_WINDOW)
    momentum = D.momentum_components(close, adjusted, windows=(250,))["tsmom_ra_250"]
    signals.signed["cs_mom_ra_250"] = D.cross_sectional_rank(momentum, pools)
    cached = C.FACTOR_DAILY_DIR / "external" / "validation_2022"
    missing = [s for s in names if not any(cached.glob(f"{s}.*"))]
    if missing:
        external.build_partition("validation_2022", missing)
    external.build_partition("oos", names)  # 每次按本次截止日重建
    source = external.load_wide(names, ("research", "validation_2022", "oos"), close.index)
    signals.signed["carry_ms"] = source["carry_main_sub_annualized"]
    signals.vol = D.daily_vol(close, adjusted)
    prices = bars["open"]
    returns = (bars["openw"].shift(-1) - bars["openw"]) / prices
    index = returns.index[(returns.index >= pd.Timestamp(C.STRICT_OOS_START))
                          & (returns.index <= pd.Timestamp(end)) & returns.notna().any(axis=1)]
    C.assert_strict_oos_dates(index, "日频腿样本外")
    rolls = pd.DataFrame(False, index=index, columns=names)
    for symbol, frame in B.load_roll_calendar(names, end).items():
        rolls.loc[index.intersection(frame.trading_date), symbol] = True
    fees = pd.concat([costs.load_fees(p) for p in ("research", "validation_2022", "oos")], ignore_index=True)
    opened, closed, roll_closed = costs.fee_tables(fees, prices)
    slip = costs.slippage_tables(costs.load_ticks(), prices.loc[index], 1.0)
    return study.StudyData(signals, returns.loc[index], universe, rolls,
                           opened.loc[index], closed.loc[index], roll_closed.loc[index], slip)


def daily_frames(parts, leg_spec):
    """各口径下四条日频腿的缩放后日收益；parts 按时间排好（研究期、2022[、样本外]），1 跳。"""
    members = leg_spec["portfolios"][leg_spec["chosen"]]
    out = {}
    for scenario, n_ticks in TICKS.items():
        scaled = [dataclasses.replace(d, slippage=d.slippage * n_ticks) for d in parts]
        positions = {leg: [ml.base_position(d, leg_spec["legs"][leg]) for d in scaled] for leg in members}
        raw = {leg: pd.concat([ml.net_of(d, p) for d, p in zip(scaled, positions[leg])]) for leg in members}
        out[scenario] = ml.build(scaled, raw, positions, members, leg_spec)
    return out


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


def rebuild(base, meta_spec, leg_spec, candidate, oos=None):
    """研究期接 2022（再接样本外）的两个候选。oos = (样本外特征表, 样本外日频数据)。"""
    table = oor.prepare_through_2022(base, 2016)
    parts = [study.load_research(1.0), ml.daily_2022(1.0)]
    partitions = ["research", "validation_2022"]
    if oos is not None:
        table = pd.concat([table, oos[0]], ignore_index=True)
        parts.append(oos[1])
        partitions.append("oos")
    for data, partition in zip(parts, partitions):
        for leg in leg_spec["portfolios"][leg_spec["chosen"]]:
            ml.check_coverage(data, leg_spec["legs"][leg]["factors"], partition)
    books = morning_books(table, base, candidate)
    morning = morning_returns(table, books, base)
    legs = daily_frames(parts, leg_spec)
    series = {scenario: {"早盘元策略": morning[(scenario, "早盘元策略")], "主策略·原样": morning[(scenario, "主策略·原样")],
                         "四条日频腿": legs[scenario].组合} for scenario in TICKS}
    return table, books, series, legs


def preflight(protocol):
    missing = [str(folder / p / "daily.csv") for folder, _ in SAVED.values() for p in ("research", "validation_2022")
               if not (folder / p / "daily.csv").exists()]
    if missing:
        raise SystemExit("先跑研究期和 2022：python scripts/research_morning_meta.py [--validation-2022]、"
                         f"python scripts/research_multi_leg.py [--validation-2022]，缺 {missing}")
    if not costs.fee_path("oos").exists():
        raise SystemExit("先下载样本外手续费：python download/getFeeHistory.py --partition oos")
    if not (oor.FEATURES / "validation_2022.parquet").exists():
        raise SystemExit("先生成早盘特征：python scripts/morning_features.py")


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", help="2024、2025 或 2024-2025")
    parser.add_argument("--check", action="store_true", help="只重算研究期和 2022 并核对，不读样本外")
    parser.add_argument("--rerun", action="store_true", help="同一配置和年份已有结果时仍重跑")
    args = parser.parse_args()
    meta_spec, leg_spec, base = load_specs(protocol)
    candidate = protocol["candidates"]["早盘元策略"]
    preflight(protocol)
    if args.check or not args.years:
        _, _, series, _ = rebuild(base, meta_spec, leg_spec, candidate)
        print("\n".join(consistency({k: v for k, v in series["主口径"].items() if k in SAVED})))
        print("检查通过，没有读样本外。正式运行：python scripts/oos_final.py --years 2024-2025")
        return
    years = parse_years(args.years, protocol["years_allowed"])
    end = C.to_date(f"{years[-1]}-12-31")
    files = [ROOT / f for f in HASHED]
    hashes = {Path(f).name: C._sha256(f) for f in files}
    finger = fingerprint(hashes, years)
    strategy = protocol["name"]
    changed = check_ledger(C.read_ledger(), strategy, finger, hashes, args.rerun)
    note = f"years={years}；2023 只作预热" + (f"；此前样本外结果对应的配置已改动: {changed}" if changed else "")
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
            oos_data = daily_oos(universe, end)
            table, books, series, legs = rebuild(base, meta_spec, leg_spec, candidate, (oos_table, oos_data))
            notes = consistency({k: v for k, v in series["主口径"].items() if k in SAVED})
    except Exception as error:  # 中途失败也留痕：台账里 opened 之后跟一条 aborted
        C.append_ledger({"kind": "final_evaluation", "status": "aborted", "strategy": strategy,
                         "run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "fingerprint": finger,
                         "years": years, "error": f"{type(error).__name__}: {error}"[:500]})
        raise
    write_outputs(ticket, protocol, years, oos_years, end, finger, changed, table, books, series, legs, notes)


def write_outputs(ticket, protocol, years, oos_years, end, finger, changed, table, books, series, legs, notes):
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
    leg_frame = legs["主口径"]
    leg_years = pd.DataFrame([{"腿": leg, "年份": y, "收益": float((1 + leg_frame.loc[leg_frame.index.year == y, leg]).prod() - 1),
                               "夏普": stats.sharpe_ratio(leg_frame.loc[leg_frame.index.year == y, leg])}
                              for leg in leg_frame.columns if leg != "组合" for y in oos_years])
    leg_years.to_csv(out / "daily_legs_yearly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scored["主口径"]).to_csv(out / "daily.csv", encoding="utf-8-sig")
    print("图表：", context.plot_performance(scored["主口径"], out / "plots" / label, f"样本外 {label}（主口径，扣费后）"))
    print("图表：", context.plot_performance(leg_frame[leg_frame.index.year.isin(years)].drop(columns="组合"),
                                           out / "plots" / "legs", f"四条日频腿各腿 样本外 {label}"))

    def pick(name, scenario="主口径"):
        row = summary[(summary.候选 == name) & (summary.口径 == scenario)].iloc[0]
        return {k: float(row[k]) for k in ("夏普", "累计收益", "年化收益", "年化波动", "最大回撤")}

    C.append_ledger({**{k: ticket[k] for k in ("kind", "strategy", "chosen", "frozen_at", "sha256")},
                     "status": "result", "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                     "fingerprint": finger, "years": years, "oos_end": str(end), "out": str(out),
                     "早盘元策略": pick("早盘元策略"), "四条日频腿": pick("四条日频腿"),
                     "主策略·原样": pick("主策略·原样"), "早盘元策略·滑点2跳": pick("早盘元策略", "滑点2跳"),
                     "四条日频腿·滑点2跳": pick("四条日频腿", "滑点2跳"), "config_changed": changed})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n".join(notes))
    print(f"\n样本外 {label}（2023 只作预热，不计入）")
    print(summary.round(4).to_string(index=False))
    print("\n逐年（主口径）：")
    print(year_table.round(4).to_string(index=False))
    print("\n早盘逐年笔数：")
    print(trades.round(2).to_string(index=False))
    print("\n日频各腿逐年：")
    print(leg_years.round(4).to_string(index=False))
    if changed:
        print(f"\n注意：此前样本外结果对应的配置已改动 {changed}，已记入台账。")
    print("\n结果：", out)


if __name__ == "__main__":
    main()
