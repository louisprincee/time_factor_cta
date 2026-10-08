"""日频核心 + 早盘卫星：2023–2025 严格样本外的最终测试，可选年份。

    python scripts/oos_portfolio.py --years 2023
    python scripts/oos_portfolio.py --years 2023-2025
    python scripts/oos_portfolio.py --years 2023,2025

只跑 config/portfolio.json 里冻结的方案，口径见 config/oos_protocol.json。读数据前先写台账
data/oos/ledger.jsonl；同一配置和年份已有结果时须加 --rerun。所选年份之前的样本外年份也会被读入，
用于信号、乘数和品种池的连续历史（不计入绩效）。
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
from tfcta.research import costs, stats, study
import morning_features
import research_morning_oor as oor
import research_portfolio as rp

PROTOCOL = ROOT / "config" / "oos_protocol.json"
OUT = ROOT / "runs" / "oos"
HASHED = ["config/portfolio.json", "config/morning_oor.json", "config/oos_protocol.json",
          "scripts/oos_portfolio.py", "scripts/research_portfolio.py", "scripts/research_morning_oor.py",
          "scripts/morning_features.py"]


def parse_years(text, allowed):
    """'2024' / '2023-2025' / '2023,2025' → 升序年份列表，只允许协议里登记的年份。"""
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


def block_starts(years):
    """每段连续年份的第一年 1 月 1 日：熔断重置口径在这些日期重新部署。"""
    return [pd.Timestamp(f"{y}-01-01") for i, y in enumerate(years) if i == 0 or years[i - 1] != y - 1]


def label_of(years):
    if years == list(range(years[0], years[-1] + 1)):
        return str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
    return "_".join(map(str, years))


def fingerprint(hashes, years):
    text = json.dumps({"sha256": hashes, "years": years}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def check_ledger(ledger, strategy, finger, hashes, rerun):
    """同一指纹已有结果：须显式 --rerun；同一策略此前用别的配置跑过样本外：提示并记入台账。"""
    rows = [r for r in ledger if r.get("kind") == "final_evaluation" and r.get("strategy") == strategy]
    done = [r for r in rows if r.get("status") == "result" and r.get("fingerprint") == finger]
    if done and not rerun:
        raise SystemExit(f"这组配置和年份已在 {done[-1]['run_at']} 跑过，结果在 {done[-1].get('out')}；"
                         "确需重跑加 --rerun")
    changed = sorted({k for r in rows if r.get("status") == "result"
                      for k, v in r.get("sha256", {}).items()
                      if k.endswith(".json") and hashes.get(k) not in (None, v)})
    return changed


def load_core_oos(universe, end):
    """样本外日频数据：行情从 2018 年接起做信号预热（研究期 + 2022 + 样本外），收益、成本只取样本外。"""
    names = sorted({s for group in universe.values() for s in group})
    per, statuses = {}, {}
    for s in names:
        m = B.load_minutes(s, include_validation=True, oos_end=end,
                           columns=list(B.DAILY_FIELDS) + ["trading_date"])
        m["trading_date"] = pd.to_datetime(m.trading_date)
        before = m[m.trading_date < pd.Timestamp(C.STRICT_OOS_START)]
        after = m[m.trading_date >= pd.Timestamp(C.STRICT_OOS_START)]
        statuses[s], _ = VI.boundary_check(before[before.trading_date.dt.year == 2022], after)
        day = B.daily_bars(m)
        per[s] = day[day.index >= "2018-01-01"]
    ok, note = VI.boundary_verdict(statuses)
    if not ok:
        raise ValueError(f"2022 → 样本外复权基准不衔接：{note}")
    print("复权衔接：", note, flush=True)
    bars = B.wide_by_field(per)
    close, adjusted = bars["close"], bars["closew"]
    signals = library.SignalSet(bars=bars)
    signals.signed["tsmom"] = D.tsmom(close, adjusted)
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
    C.assert_strict_oos_dates(index, "日频核心样本外")
    rolls = pd.DataFrame(False, index=index, columns=names)
    for s, frame in B.load_roll_calendar(names, end).items():
        rolls.loc[index.intersection(frame.trading_date), s] = True
    fees = pd.concat([costs.load_fees(p) for p in ("research", "validation_2022", "oos")], ignore_index=True)
    o, c, r = costs.fee_tables(fees, prices)
    slip = costs.slippage_tables(costs.load_ticks(), prices.loc[index], 1.0)
    return study.StudyData(signals, returns.loc[index], universe, rolls,
                           o.loc[index], c.loc[index], r.loc[index], slip)


def with_ticks(data, n_ticks):
    """滑点表与跳数成正比，2 跳直接翻倍，不用重读行情。"""
    return dataclasses.replace(data, slippage=data.slippage * n_ticks)


def load_all(spec, oor_spec, universe, end, oos_years):
    """研究期、2022、样本外三段：核心 (数据, 基础仓位) 和卫星特征表，两个滑点口径共用。"""
    core = []
    for partition in ("research", "validation_2022", "oos"):
        if partition == "research":
            data = study.load_research(1.0)
        elif partition == "validation_2022":
            data = rp.load_core_2022(1.0)
        else:
            data = load_core_oos(universe, end)
        rp.check_carry(data, partition)
        core.append((data, rp.core_position(data, spec["core"])))
    names = sorted({s for group in universe.values() for s in group})
    morning_features.build("oos", end=end, names=names)
    tables = [oor.prepare("research", oor_spec), oor.prepare("validation_2022", oor_spec),
              oor.prepare("oos", oor_spec, years=(oos_years[0], oos_years[-1]), pools=universe)]
    return core, tables


def scenario_parts(core, tables, spec, oor_spec, scenario):
    n_ticks, oor_scenario = rp.SCENARIOS[scenario]
    candidate = oor_spec["candidates"][spec["satellite"]["candidate"]]
    parts = [(with_ticks(d, n_ticks), p) for d, p in core]
    satellite = pd.concat([oor.evaluate(t, oor_spec, candidate, oor_scenario)[0] for t in tables])
    return parts, satellite


def yearly_table(frame, satellite_paper, halt_on):
    rows = []
    for year, f in frame.groupby(frame.index.year):
        nav = (1 + f.组合).cumprod()
        rows.append({"年份": year, "核心": float((1 + f.核心).prod() - 1), "卫星": float((1 + f.卫星).prod() - 1),
                     "组合": float(nav.iloc[-1] - 1), "组合夏普": stats.sharpe_ratio(f.组合),
                     "组合年内回撤": float((nav / nav.cummax().clip(lower=1.0) - 1).min()),
                     "卫星出手日": int((satellite_paper.reindex(f.index).fillna(0) != 0).sum()),
                     "卫星熔断停手日": int((~halt_on.reindex(f.index).fillna(True)
                                       & (satellite_paper.reindex(f.index).fillna(0) != 0)).sum()),
                     "核心乘数均值": float(f.核心乘数.mean())})
    return pd.DataFrame(rows)


def consistency(frame):
    """续接口径下研究期和 2022 部分必须与已落盘结果一致，否则说明数据或代码变了。"""
    notes = []
    for partition in ("research", "validation_2022"):
        path = rp.OUT / partition / "daily.csv"
        if not path.exists():
            notes.append(f"{partition}: 无已落盘结果，未核对")
            continue
        saved = pd.read_csv(path, index_col=0, parse_dates=True)
        again = frame.组合.reindex(saved.index)
        if not np.allclose(again, saved.组合, atol=1e-10):
            raise ValueError(f"样本外运行中的 {partition} 部分与已落盘结果不一致：{path}")
        notes.append(f"{partition}: 与已落盘结果一致")
    return notes


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="冻结组合的样本外测试")
    parser.add_argument("--years", required=True, help="如 2024、2023-2025、2023,2025")
    parser.add_argument("--rerun", action="store_true", help="同一配置和年份已有结果时仍重跑")
    args = parser.parse_args()
    years = parse_years(args.years, protocol["years_allowed"])
    end = C.to_date(f"{years[-1]}-12-31")
    spec = json.loads((ROOT / protocol["strategy_config"]).read_text(encoding="utf-8"))
    oor_spec = json.loads((ROOT / spec["satellite"]["config"]).read_text(encoding="utf-8"))
    if not spec.get("chosen") or not spec.get("frozen_at"):
        raise SystemExit("config/portfolio.json 没有冻结的 chosen / frozen_at，不能跑样本外")
    if not (rp.OUT / "research" / "daily.csv").exists():
        raise SystemExit("先跑 scripts/research_portfolio.py（研究期）和 --validation-2022，样本外要核对这两段")
    if not costs.fee_path("oos").exists():
        raise SystemExit("先下载样本外手续费：python download/getFeeHistory.py --partition oos")
    halt = oor_spec["breaker"]["halt_drawdown"]
    files = [ROOT / f for f in HASHED]
    hashes = {Path(f).name: C._sha256(f) for f in files}
    finger = fingerprint(hashes, years)
    strategy = spec["name"]
    changed = check_ledger(C.read_ledger(), strategy, finger, hashes, args.rerun)
    label = label_of(years)
    out = OUT / label
    resets = block_starts(years)

    meta_note = f"years={years}" + (f"；此前样本外结果对应的配置已改动: {changed}" if changed else "")
    try:
        ticket, rows, frames, papers, variants, notes = evaluate(spec, oor_spec, end, years, resets, strategy,
                                                                 files, halt, meta_note)
    except Exception as error:  # 中途失败也留痕：台账里 opened 之后跟一条 aborted
        C.append_ledger({"kind": "final_evaluation", "status": "aborted", "strategy": strategy,
                         "run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "fingerprint": finger,
                         "years": years, "error": f"{type(error).__name__}: {error}"[:500]})
        raise
    write_outputs(protocol, spec, years, label, out, halt, finger, end, changed,
                  ticket, rows, frames, papers, variants, notes)


def evaluate(spec, oor_spec, end, years, resets, strategy, files, halt, note):
    oos_years = list(range(C.STRICT_OOS_START.year, end.year + 1))
    with C.final_evaluation(spec, files, end, strategy, note=note) as ticket:
        universe, _ = U.oos_universe([s for s in C.COMMODITY_SYMBOLS if s in costs.MULTIPLIER], end)
        universe = {y: [s for s in universe.get(y, []) if s in costs.MULTIPLIER] for y in oos_years}
        if not all(universe.values()):
            raise ValueError(f"样本外有年份品种池为空: {[y for y, v in universe.items() if not v]}")
        core, tables = load_all(spec, oor_spec, universe, end, oos_years)
        labels = {spec["chosen"]: spec["schemes"][spec["chosen"]], **spec["references"]}
        variants = {"熔断重置": resets, "熔断续接": []}
        rows, frames, papers = [], {}, {}
        for scenario in rp.SCENARIOS:
            parts, satellite = scenario_parts(core, tables, spec, oor_spec, scenario)
            papers[scenario] = satellite
            for variant, cuts in variants.items():
                for name, share in labels.items():
                    if share == 1.0 and variant != "熔断重置":
                        continue  # 只核心与熔断无关
                    frame = rp.build(parts, satellite, share, spec, halt, cuts)
                    frames[(scenario, variant, name)] = frame
                    window = frame[frame.index.year.isin(years)]
                    rows.append({"口径": scenario, "熔断": variant, **rp.describe(window, name)})
        notes = consistency(frames[("主口径", "熔断续接", spec["chosen"])])
    return ticket, rows, frames, papers, variants, notes


def write_outputs(protocol, spec, years, label, out, halt, finger, end, changed,
                  ticket, rows, frames, papers, variants, notes):
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    main_key = (protocol["main"]["scenario"], protocol["main"]["breaker"], spec["chosen"])
    yearly = []
    for variant, cuts in variants.items():
        frame = frames[("主口径", variant, spec["chosen"])]
        paper = papers["主口径"]
        on = rp.guard(paper, halt, cuts).ne(0) | paper.eq(0)
        window = frame[frame.index.year.isin(years)]
        yearly.append(yearly_table(window, paper, on).assign(熔断=variant))
        window.to_csv(out / f"daily_{variant}.csv", encoding="utf-8-sig")
    yearly = pd.concat(yearly, ignore_index=True)
    yearly.to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")
    main_window = frames[main_key][frames[main_key].index.year.isin(years)]
    rp.plot(main_window, out / "nav.png",
            f"日频核心 + 早盘卫星（{spec['chosen']}，{protocol['main']['breaker']}）样本外 {label}，扣费后")

    def pick(scenario, variant, name=spec["chosen"]):
        row = table[(table.口径 == scenario) & (table.熔断 == variant) & (table.方案 == name)].iloc[0]
        return {k: float(row[k]) for k in ("夏普", "年化收益", "年化波动", "最大回撤", "夏普95%下限")}

    C.append_ledger({**{k: ticket[k] for k in ("kind", "strategy", "chosen", "frozen_at", "sha256")},
                     "status": "result", "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                     "fingerprint": finger, "years": years, "oos_end": str(end), "out": str(out),
                     "main": pick(*main_key[:2]), "熔断续接": pick("主口径", "熔断续接"),
                     "滑点2跳": pick("滑点2跳", protocol["main"]["breaker"]), "config_changed": changed})

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n".join(notes))
    print(f"\n样本外 {label}，冻结方案 {spec['chosen']}；主口径 = {protocol['main']['scenario']} + "
          f"{protocol['main']['breaker']}")
    print(table.round(3).T.to_string())
    print("\n逐年（主口径）：")
    print(yearly.round(4).to_string(index=False))
    if changed:
        print(f"\n注意：此前样本外结果对应的配置已改动 {changed}，已记入台账。")
    print("\n结果：", out)


if __name__ == "__main__":
    main()
