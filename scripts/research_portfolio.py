"""日频核心（趋势 + 期限结构）与早盘卫星（开盘过度反应回归）的组合。"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tfcta import config as C
from tfcta.data import bars as B, universe as U
from tfcta.factors import daily as D, external, library
from tfcta.research import costs, engine, stats, study
import research_morning_oor as oor

CONFIG = ROOT / "config" / "portfolio.json"
OUT = ROOT / "runs" / "portfolio"
SCENARIOS = {"主口径": (1.0, "主口径"), "滑点2跳": (2.0, "滑点2跳")}


def load_core_2022(n_ticks):
    """2022 的日频数据：行情从 2018 年接起做信号预热，收益、成本只取 2022。"""
    names = [n for n in U.validation_universe()[0] if n in costs.MULTIPLIER]
    per = {}
    for s in names:
        m = B.load_minutes(s, include_validation=True, columns=list(B.DAILY_FIELDS) + ["trading_date"])
        day = B.daily_bars(m)
        per[s] = day[day.index >= "2018-01-01"]
    bars = B.wide_by_field(per)
    close, adjusted = bars["close"], bars["closew"]
    signals = library.SignalSet(bars=bars)
    signals.signed["tsmom"] = D.tsmom(close, adjusted)
    if not (C.FACTOR_DAILY_DIR / "external" / "validation_2022").exists():
        external.build_partition("validation_2022", names)
    source = external.load_wide(names, ("research", "validation_2022"), close.index)
    signals.signed["carry_ms"] = source["carry_main_sub_annualized"]
    signals.vol = D.daily_vol(close, adjusted)
    prices = bars["open"]
    returns = (bars["openw"].shift(-1) - bars["openw"]) / prices
    index = returns.index[(returns.index.year == 2022) & returns.notna().any(axis=1)]
    C.assert_validation_2022_dates(index, "日频核心 2022")
    rolls = pd.DataFrame(False, index=index, columns=names)
    for s, frame in B.load_roll_calendar(names, "2022-12-31").items():
        rolls.loc[index.intersection(frame.trading_date), s] = True
    fees = pd.concat([costs.load_fees("research"), costs.load_fees("validation_2022")], ignore_index=True)
    o, c, r = costs.fee_tables(fees, prices)
    slip = costs.slippage_tables(costs.load_ticks(), prices.loc[index], n_ticks)
    return study.StudyData(signals, returns.loc[index], {2022: names}, rolls,
                           o.loc[index], c.loc[index], r.loc[index], slip)


def check_carry(data, what, minimum=0.9):
    """期限结构缺数据会悄悄变成空仓，先确认覆盖。"""
    carry = data.factors.raw("carry_ms").reindex(data.returns.index)
    worst = {}
    for year, members in data.universe.items():
        rows = carry.index.year == int(year)
        if rows.any():
            cover = carry.loc[rows, [m for m in members if m in carry]].notna().mean()
            worst[year] = float(cover.min())
    if min(worst.values()) < minimum:
        raise ValueError(f"{what} 期限结构覆盖不足: {worst}")


def core_position(data, spec):
    signal = study.signal_for(data, spec)
    return engine.position(signal, data.factors.vol, spec.get("vol_target", .20), spec.get("cap", 1.),
                           spec["days"], spec["mode"], spec.get("phase", 0)).reindex_like(data.returns)


def core_net(data, position):
    return engine.backtest(position, data.returns, data.universe, data.open_fee, data.close_fee,
                           data.slippage, data.rolls, data.roll_close_fee).net


def multiplier(ret, target, window, min_periods, lag, cap):
    """目标波动 / 近期已实现波动，滞后 lag 日；波动未知时为 0（空仓）。"""
    vol = ret.rolling(window, min_periods=min_periods).std() * np.sqrt(252)
    m = (target / vol.where(vol > 0)).shift(lag).clip(upper=cap)
    return m.fillna(0.0)


def leg_targets(share, target_vol):
    norm = np.hypot(share, 1 - share)
    return target_vol * share / norm, target_vol * (1 - share) / norm


def guard(satellite, halt, resets=()):
    """熔断；resets 里的日期起账面高点重新从本金算（新部署），乘数仍用连续的历史。"""
    cuts = sorted(pd.Timestamp(r) for r in resets)
    edges = [satellite.index.min(), *cuts]
    pieces = [satellite[(satellite.index >= a) & (satellite.index < b)] for a, b in zip(edges, cuts)]
    pieces.append(satellite[satellite.index >= edges[-1]])
    return pd.concat([oor.breaker(p, halt) for p in pieces if len(p)])


def build(parts, satellite, share, spec, halt, resets=()):
    """parts：按时间排好的 (日频数据, 基础仓位) 列表；satellite：连续的卫星账面日收益。"""
    sc = spec["scaling"]
    core_target, sat_target = leg_targets(share, sc["target_vol"])
    base = pd.concat([core_net(d, p) for d, p in parts])
    m_core = multiplier(base, core_target, sc["vol_window"], sc["vol_min_periods"], 2, sc["max_core_multiplier"])
    core = pd.concat([core_net(d, p.mul(m_core.reindex(p.index), axis=0)) for d, p in parts])
    guarded = guard(satellite, halt, resets) if spec["satellite"]["breaker"] else satellite
    m_sat = multiplier(satellite, sat_target, sc["vol_window"], sc["vol_min_periods"], 1,
                       sc["max_satellite_multiplier"])
    sat = guarded * m_sat
    frame = pd.DataFrame({"核心": core, "卫星": sat}).fillna(0.0)
    frame["组合"] = frame.核心 + frame.卫星
    frame["核心乘数"] = m_core.reindex(frame.index).fillna(0.0)
    frame["卫星乘数"] = m_sat.reindex(frame.index).fillna(0.0)
    return frame


def describe(frame, label):
    ret = frame.组合
    perf = stats.performance(ret)
    low, high = oor.bootstrap_sharpe(ret)
    row = {"方案": label, "夏普": stats.sharpe_ratio(ret), "年化收益": perf["ann_return"],
           "年化波动": perf["ann_vol"], "最大回撤": perf["max_drawdown"], "夏普95%下限": low, "夏普95%上限": high,
           "核心夏普": stats.sharpe_ratio(frame.核心), "卫星夏普": stats.sharpe_ratio(frame.卫星),
           "两腿相关": frame.核心.corr(frame.卫星), "核心乘数均值": frame.核心乘数.mean(),
           "卫星乘数均值": frame.卫星乘数.mean()}
    for year, r in ret.groupby(ret.index.year):
        row[f"{year}夏普"] = stats.sharpe_ratio(r)
        row[f"{year}收益"] = float((1 + r).prod() - 1)
    return row


def select(table, schemes):
    eligible = table[table.方案.isin(schemes)]
    return eligible.loc[eligible["夏普95%下限"].idxmax(), "方案"]


def load_parts(partitions, spec, oor_spec, scenario):
    n_ticks, oor_scenario = SCENARIOS[scenario]
    candidate = oor_spec["candidates"][spec["satellite"]["candidate"]]
    parts, sats = [], []
    for partition in partitions:
        data = study.load_research(n_ticks) if partition == "research" else load_core_2022(n_ticks)
        check_carry(data, partition)
        parts.append((data, core_position(data, spec["core"])))
        ret, _ = oor.evaluate(oor.prepare(partition, oor_spec), oor_spec, candidate, oor_scenario)
        sats.append(ret)
    return parts, pd.concat(sats)


def plot(frame, path, title):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    for name, width in (("组合", 1.8), ("核心", 0.9), ("卫星", 0.9)):
        top.plot(frame.index, (1 + frame[name]).cumprod(), label=name, lw=width)
    nav = (1 + frame.组合).cumprod()
    bottom.fill_between(frame.index, nav / nav.cummax() - 1, 0, color="tab:red", alpha=0.5)
    top.set_title(title)
    top.legend(loc="upper left")
    top.grid(alpha=0.3)
    bottom.set_ylabel("组合回撤")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-2022", action="store_true")
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    oor_spec = json.loads((ROOT / spec["satellite"]["config"]).read_text(encoding="utf-8"))
    halt = oor_spec["breaker"]["halt_drawdown"]
    if args.validation_2022:
        if not spec["chosen"]:
            raise SystemExit("先跑研究期并把选中的方案写进 config/portfolio.json 的 chosen")
        partition, partitions = "validation_2022", ["research", "validation_2022"]
        labels = {spec["chosen"]: spec["schemes"][spec["chosen"]], **spec["references"]}
    else:
        partition, partitions = "research", ["research"]
        labels = {**spec["schemes"], **spec["references"]}
    first = 2022 if args.validation_2022 else 2016
    last = 2022 if args.validation_2022 else 2021
    out = OUT / partition
    out.mkdir(parents=True, exist_ok=True)

    rows, frames = [], {}
    for scenario in SCENARIOS:
        parts, satellite = load_parts(partitions, spec, oor_spec, scenario)
        for label, share in labels.items():
            frame = build(parts, satellite, share, spec, halt)
            frames[(scenario, label)] = frame
            window = frame[(frame.index.year >= first) & (frame.index.year <= last)]
            rows.append({"口径": scenario, **describe(window, label)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "schemes.csv", index=False, encoding="utf-8-sig")

    main_name = spec["chosen"] if args.validation_2022 else select(table[table.口径 == "主口径"], spec["schemes"])
    full = frames[("主口径", main_name)]
    if args.validation_2022:
        saved = pd.read_csv(OUT / "research" / "daily.csv", index_col=0, parse_dates=True)
        again = full.loc[saved.index, "组合"]
        if not np.allclose(again, saved.组合, atol=1e-10):
            raise ValueError("2022 运行中的研究期部分与研究期结果不一致")
    window = full[(full.index.year >= first) & (full.index.year <= last)]
    window.to_csv(out / "daily.csv", encoding="utf-8-sig")
    span = str(first) if first == last else f"{first}–{last}"
    plot(window, out / "nav.png", f"日频核心 + 早盘卫星（{main_name}）{span}，扣费后")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(table.round(3).T.to_string())
    print("按标准选中：" if not args.validation_2022 else "冻结方案：", main_name)


if __name__ == "__main__":
    main()
