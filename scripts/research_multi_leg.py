"""多腿等风险组合：早盘回吐 + 日频趋势、期限结构、截面动量、短周期趋势。研究期，以及 2022 的一次性诊断。

    python scripts/research_multi_leg.py
    python scripts/research_multi_leg.py --validation-2022

腿、缩放和组合写在 config/multi_leg.json。输出在 runs/multi_leg/<分区>/。
2022 的缩放用研究期接 2022 的连续收益，日频信号从 2018 年起预热。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tfcta import config as C
from tfcta.data import bars as B, universe as U
from tfcta.factors import daily as D, external, library
from tfcta.research import context, costs, engine, stats, study
import research_morning_oor as oor

CONFIG = ROOT / "config" / "multi_leg.json"
OUT = ROOT / "runs" / "multi_leg"
SCENARIOS = {"主口径": 1.0, "滑点2跳": 2.0}


def daily_2022(n_ticks):
    """2022 的日频数据：行情从 2018 年接起做信号预热，收益、成本只取 2022。"""
    names = [n for n in U.validation_universe()[0] if n in costs.MULTIPLIER]
    per = {}
    for symbol in names:
        minutes = B.load_minutes(symbol, include_validation=True, columns=list(B.DAILY_FIELDS) + ["trading_date"])
        day = B.daily_bars(minutes)
        per[symbol] = day[day.index >= "2018-01-01"]
    bars = B.wide_by_field(per)
    close, adjusted = bars["close"], bars["closew"]
    pools = {**{y: s for y, s in U.load_universe().items()}, 2022: names}
    signals = library.SignalSet(bars=bars)
    signals.signed["tsmom"] = D.tsmom(close, adjusted)
    signals.signed["ma_break_20"] = D.ma_breakout(adjusted, D.REVIEW_VOL_WINDOW)
    momentum = D.momentum_components(close, adjusted, windows=(250,))["tsmom_ra_250"]
    signals.signed["cs_mom_ra_250"] = D.cross_sectional_rank(momentum, pools)
    if not (C.FACTOR_DAILY_DIR / "external" / "validation_2022").exists():
        external.build_partition("validation_2022", names)
    source = external.load_wide(names, ("research", "validation_2022"), close.index)
    signals.signed["carry_ms"] = source["carry_main_sub_annualized"]
    signals.vol = D.daily_vol(close, adjusted)
    prices = bars["open"]
    returns = (bars["openw"].shift(-1) - bars["openw"]) / prices
    index = returns.index[(returns.index.year == 2022) & returns.notna().any(axis=1)]
    C.assert_validation_2022_dates(index, "日频腿 2022")
    rolls = pd.DataFrame(False, index=index, columns=names)
    for symbol, frame in B.load_roll_calendar(names, "2022-12-31").items():
        rolls.loc[index.intersection(frame.trading_date), symbol] = True
    fees = pd.concat([costs.load_fees("research"), costs.load_fees("validation_2022")], ignore_index=True)
    opened, closed, roll_closed = costs.fee_tables(fees, prices)
    slip = costs.slippage_tables(costs.load_ticks(), prices.loc[index], n_ticks)
    return study.StudyData(signals, returns.loc[index], {2022: names}, rolls,
                           opened.loc[index], closed.loc[index], roll_closed.loc[index], slip)


def check_coverage(data, factors, what, minimum=0.9):
    """因子缺数据会悄悄变成空仓，先确认池内覆盖。"""
    for name in factors:
        raw = data.factors.raw(name).reindex(data.returns.index)
        for year, members in data.universe.items():
            rows = raw.index.year == int(year)
            present = [m for m in members if m in raw]
            if rows.any() and present:
                cover = float(raw.loc[rows, present].notna().mean().min())
                if cover < minimum:
                    raise ValueError(f"{what} {name} {year} 年覆盖不足 {cover:.2f}")


def base_position(data, leg):
    signal = study.signal_for(data, {"factors": leg["factors"]})
    return engine.position(signal, data.factors.vol, leg.get("vol_target", .20), leg.get("cap", 1.),
                           leg["days"], leg["mode"]).reindex_like(data.returns)


def net_of(data, position):
    return engine.backtest(position, data.returns, data.universe, data.open_fee, data.close_fee,
                           data.slippage, data.rolls, data.roll_close_fee).net


def multiplier(ret, target, window, min_periods, lag, cap):
    """目标波动 / 近期已实现波动，滞后 lag 日；波动未知时为 0（空仓）。"""
    vol = ret.rolling(window, min_periods=min_periods).std() * np.sqrt(252)
    return (target / vol.where(vol > 0)).shift(lag).clip(upper=cap).fillna(0.0)


def morning_leg(partitions, spec, scenario):
    """早盘腿的账面日收益（未缩放），按分区接起来。"""
    morning = spec["legs"]["早盘回吐"]
    base = json.loads((ROOT / morning["config"]).read_text(encoding="utf-8"))
    out = []
    for partition in partitions:
        table = oor.prepare(partition, base)
        side = oor.base_books(table, base)[morning["book"]]
        weight = oor.weights(table, side, base["sizing"])
        calendar = pd.DatetimeIndex(sorted(table.trading_date.unique()))
        out.append(oor.daily(table, weight, "主口径" if scenario == "主口径" else "滑点2跳", calendar))
    return pd.concat(out)


def legs_returns(partitions, spec, scenario):
    """每条腿缩放后的日收益（研究期接 2022 时连续计算缩放）。"""
    sc = spec["scaling"]
    target = sc["target_vol"]
    n_ticks = SCENARIOS[scenario]
    parts = []
    for partition in partitions:
        data = study.load_research(n_ticks) if partition == "research" else daily_2022(n_ticks)
        parts.append(data)
    out, raw = {}, {}
    names = list(spec["legs"])
    for name, leg in spec["legs"].items():
        if leg["kind"] == "morning":
            series = morning_leg(partitions, spec, scenario)
            raw[name] = series
            out[name] = series  # 缩放在组合里按腿数做
            continue
        for data, partition in zip(parts, partitions):
            check_coverage(data, leg["factors"], partition)
        positions = [base_position(d, leg) for d in parts]
        raw[name] = pd.concat([net_of(d, p) for d, p in zip(parts, positions)])
        out[name] = positions
    return parts, raw, out


def build(parts, raw, positions, members, spec):
    """组合内每条腿缩放到 目标波动 / sqrt(腿数)，日频腿按缩放后的仓位重算成本。"""
    sc = spec["scaling"]
    leg_target = sc["target_vol"] / np.sqrt(len(members))
    frame = {}
    for name in members:
        leg = spec["legs"][name]
        if leg["kind"] == "morning":
            m = multiplier(raw[name], leg_target, sc["vol_window"], sc["vol_min_periods"], 1, sc["max_morning_multiplier"])
            frame[name] = raw[name] * m
        else:
            m = multiplier(raw[name], leg_target, sc["vol_window"], sc["vol_min_periods"], 2, sc["max_daily_multiplier"])
            frame[name] = pd.concat([net_of(d, p.mul(m.reindex(p.index), axis=0)) for d, p in zip(parts, positions[name])])
    frame = pd.DataFrame(frame).fillna(0.0)
    frame["组合"] = frame[members].sum(axis=1)
    return frame


def describe(ret, label, scenario):
    perf = stats.performance(ret)
    row = {"组合": label, "口径": scenario, "夏普": stats.sharpe_ratio(ret), "年化收益": perf["ann_return"],
           "年化波动": perf["ann_vol"], "最大回撤": perf["max_drawdown"]}
    if len(ret) > 300:
        row["夏普95%下限"] = oor.bootstrap_sharpe(ret)[0]
    for year, part in ret.groupby(ret.index.year):
        row[f"{year}夏普"] = stats.sharpe_ratio(part)
        row[f"{year}收益"] = float((1 + part).prod() - 1)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-2022", action="store_true")
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    partition = "validation_2022" if args.validation_2022 else "research"
    partitions = ["research", "validation_2022"] if args.validation_2022 else ["research"]
    first, last = (2022, 2022) if args.validation_2022 else (2016, 2021)
    out = OUT / partition
    out.mkdir(parents=True, exist_ok=True)
    rows, legs_rows, corr, curves, legs_curves = [], [], None, {}, {}
    for scenario in SCENARIOS:
        parts, raw, positions = legs_returns(partitions, spec, scenario)
        for label, members in spec["portfolios"].items():
            frame = build(parts, raw, positions, members, spec)
            window = frame[(frame.index.year >= first) & (frame.index.year <= last)]
            if partition == "research":
                C.assert_no_holdout_dates(window.index, "多腿组合")
            else:
                C.assert_validation_2022_dates(window.index, "多腿组合 2022")
            rows.append(describe(window.组合, label, scenario))
            if scenario == "主口径":
                curves[label] = window.组合
                if label == "五腿等风险":
                    corr = window[members].corr()
                    legs_curves = window[members]
                    for name in members:
                        legs_rows.append(describe(window[name], name, scenario))
    table = pd.DataFrame(rows)
    if partition == "research":
        table["保留"] = (table.口径 == "主口径") & (table.夏普 > 0)  # 保留只看研究期
    table.to_csv(out / "portfolios.csv", index=False, encoding="utf-8-sig")
    legs_table = pd.DataFrame(legs_rows)
    legs_table.to_csv(out / "legs.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(out / "correlation.csv", encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(out / "daily.csv", encoding="utf-8-sig")
    print("图表：", context.plot_performance(curves, out / "plots" / ("research" if partition == "research" else "validation-2022"), f"多腿等风险组合 {partition}（扣费后）"))
    print("图表：", context.plot_performance(legs_curves, out / "plots" / "legs", f"五腿各腿 {partition}（缩放后，扣费后）"))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(legs_table.round(3).to_string(index=False))
    print(corr.round(2).to_string())
    print(table.round(3).to_string(index=False))
    print("结果：", out)


if __name__ == "__main__":
    main()
