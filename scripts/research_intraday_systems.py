"""文档里的日内系统：只在 2016–2021 上各算一遍，不读 2022。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tfcta import config as C
from tfcta.data import shard_io, universe as U
from tfcta.factors import raw_price_path
from tfcta.research import costs, stats
import intraday_systems as rules
import morning_features as mf

CONFIG = ROOT / "config" / "intraday_systems.json"
OUT = ROOT / "runs" / "intraday_systems"


def load_spec():
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    if set(spec["rules"]) != set(rules.RULES):
        raise ValueError("配置里的系统和代码不一致")
    shared = spec["shared"]
    if [shared["top_n"], shared["range_days"], shared["min_range"], shared["max_stop"],
            shared["risk_per_trade"]] != [3, 5, 0.015, 0.02, 0.02]:
        raise ValueError("共同的筛选和止损被改过")
    return spec


def sessions_of(frame):
    """按交易日切出 09:00–15:00。夜盘不进这些系统。"""
    minute = frame.index.hour.to_numpy() * 60 + frame.index.minute.to_numpy()
    keep = (minute >= 9 * 60) & (minute <= 15 * 60)
    view = frame.loc[keep]
    clock = minute[keep]
    raw_close = raw_price_path(view)
    raw_open = raw_price_path(pd.DataFrame({"close": view.open.to_numpy()}, index=view.index))
    dates = pd.to_datetime(view.trading_date).dt.normalize().to_numpy()
    order = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    out = []
    for a, b in zip(order[:-1], order[1:]):
        if b - a < 2:
            continue
        out.append((pd.Timestamp(dates[a]), rules.Session(
            clock[a:b], view.openw.to_numpy()[a:b], view.highw.to_numpy()[a:b],
            view.loww.to_numpy()[a:b], view.closew.to_numpy()[a:b],
            raw_open[a:b], raw_close[a:b], view.volume.to_numpy()[a:b])))
    return out


def range_table(symbols):
    """每个品种已完成日盘的振幅，分母是当日开盘价。"""
    frames = {}
    for symbol in symbols:
        path = shard_io.find_shard(C.RESEARCH_DIR, symbol)
        if path is None:
            print(symbol, "无分片，跳过", flush=True)
            continue
        days = sessions_of(shard_io.load_shard(symbol, directory=C.RESEARCH_DIR, columns=mf.COLUMNS))
        if not days:
            continue
        ranges = [(day, (bars.high_w.max() - bars.low_w.min()) / bars.open_raw[0])
                  for day, bars in days if bars.open_raw[0] > 0]
        frames[symbol] = pd.Series({day: value for day, value in ranges}).sort_index()
        print("振幅", symbol, len(frames[symbol]), flush=True)
    panel = pd.DataFrame(frames).sort_index()
    C.assert_no_holdout_dates(panel.index, "日内振幅")
    return panel


def eligible(ranges, pools, shared):
    """前五日平均振幅的前三名；所有品种都不到 1.5% 的日子不做。天胶系统另看自己是否达到 1.5%。"""
    avg5 = ranges.shift(1).rolling(shared["range_days"], min_periods=shared["range_days"]).mean()
    chosen, rubber = {}, {}
    for day, row in avg5.iterrows():
        if not 2016 <= day.year <= 2021:
            continue
        members = [s for s in pools.get(day.year, ()) if s in row.index]
        vals = row[members].dropna()
        if vals.empty or float(vals.max()) < shared["min_range"]:
            continue
        chosen[day] = set(vals.nlargest(shared["top_n"]).index)
        if "RU" in vals.index and float(vals["RU"]) >= shared["min_range"]:
            rubber[day] = True
    return chosen, rubber


def run_symbol(symbol, ranges, chosen, rubber_days, fees, ticks, spec):
    days = sessions_of(shard_io.load_shard(symbol, directory=C.RESEARCH_DIR, columns=mf.COLUMNS))
    prior = {}
    last_close = None
    for day, bars in days:
        prior[day] = last_close
        last_close = bars.close_raw[-1]
    avg3 = ranges[symbol].shift(1).rolling(3, min_periods=3).mean()
    min3 = ranges[symbol].shift(1).rolling(3, min_periods=3).min()
    fee_rows = fees[fees.symbol.eq(symbol)].set_index("trading_date").sort_index()
    fee_rows = fee_rows.reindex(fee_rows.index.union(ranges.index), method="ffill")
    out = {name: [] for name in spec["rules"]}
    shared = spec["shared"]
    period = spec["rules"]["首轮波动共振"]["rsi_period"]
    previous_close = None
    for day, bars in days:
        warmed = bars.close_w if previous_close is None else np.r_[previous_close, bars.close_w]
        warmed = rules.rsi_wilder(warmed, period)[-len(bars.close_w):]
        previous_close = bars.close_w
        if day not in chosen and not (symbol == "RU" and day in rubber_days):
            continue
        if not 2016 <= day.year <= 2021:
            continue
        hist = {"avg_range_3": avg3.get(day, np.nan), "min_range_3": min3.get(day, np.nan),
                "prior_close_raw": prior.get(day), "symbol": symbol}
        fee = fee_rows.loc[day] if day in fee_rows.index else None
        tick = ticks.get(day.year, np.nan)
        if fee is None or not isinstance(getattr(fee, "commission_type", None), str) or not np.isfinite(tick):
            continue
        for name, rule in spec["rules"].items():
            if name == "天胶开盘反向" and (symbol != "RU" or day not in rubber_days):
                continue
            if name != "天胶开盘反向" and symbol not in chosen.get(day, ()):
                continue
            found = (rules.resonance(bars, warmed, hist, rule, shared) if name == "首轮波动共振"
                     else rules.RULES[name](bars, hist, rule, shared))
            for trade in found:
                cost = (mf.fee_rate(fee.commission_type, fee.open_commission, trade["entry_raw"],
                                    trade["entry_raw"], symbol)
                        + mf.fee_rate(fee.commission_type, fee.close_commission_today, trade["exit_raw"],
                                      trade["entry_raw"], symbol)
                        + tick / trade["entry_raw"] + tick / trade["exit_raw"])
                if not np.isfinite(cost):
                    continue
                weight = min(shared["max_weight"], shared["risk_per_trade"] / trade["stop_pct"])
                out[name].append((day, weight * (trade["gross"] - cost)))
    return out


def summarize(name, pairs, calendar):
    daily = pd.Series(0.0, index=calendar)
    if pairs:
        frame = pd.DataFrame(pairs, columns=["day", "ret"])
        daily = daily.add(frame.groupby("day").ret.sum(), fill_value=0.0)
    daily = daily.sort_index()
    perf = stats.performance(daily)
    row = {"系统": name, "夏普": stats.sharpe_ratio(daily), "年化收益": perf["ann_return"],
           "年化波动": perf["ann_vol"], "最大回撤": perf["max_drawdown"], "交易笔数": len(pairs),
           "有仓日": int((daily != 0).sum()), "交易日": int(len(daily))}
    yearly = {int(year): stats.sharpe_ratio(part) for year, part in daily.groupby(daily.index.year)}
    return row, yearly, daily


def main():
    spec = load_spec()
    pools = {year: names for year, names in U.load_universe().items() if 2016 <= year <= 2021}
    symbols = sorted({s for names in pools.values() for s in names if s in costs.MULTIPLIER})
    ranges = range_table(symbols)
    chosen, rubber_days = eligible(ranges, pools, spec["shared"])
    fees = mf.fee_history("research")
    tick_table = costs.load_ticks()
    books = {name: [] for name in spec["rules"]}
    for i, symbol in enumerate(ranges.columns, 1):
        print(f"信号 {i}/{ranges.shape[1]} {symbol}", flush=True)
        symbol_ticks = mf.prior_ticks(tick_table, symbol)
        found = run_symbol(symbol, ranges, chosen, rubber_days, fees, symbol_ticks, spec)
        for name, pairs in found.items():
            books[name].extend(pairs)
    calendar = ranges.index[(ranges.index.year >= 2016) & (ranges.index.year <= 2021)]
    C.assert_no_holdout_dates(calendar, "日内系统")
    rows, yearly_rows, daily = [], [], {}
    for name in spec["rules"]:
        row, yearly, series = summarize(name, books[name], calendar)
        rows.append(row)
        daily[name] = series
        for year, value in yearly.items():
            yearly_rows.append({"系统": name, "年": year, "夏普": value})
        print(f"{name}  夏普 {row['夏普']:.2f}  年化 {row['年化收益']:.2%}  "
              f"波动 {row['年化波动']:.2%}  回撤 {row['最大回撤']:.2%}  {row['交易笔数']} 笔", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(OUT / "yearly.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT / "daily.csv")
    print("消息驱动没有事件日历，没有构建。", flush=True)
    print(f"写出 {OUT}", flush=True)


if __name__ == "__main__":
    main()
