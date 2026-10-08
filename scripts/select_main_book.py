"""在研究期里选定一条主策略，然后只把这一条送到 2022。

比较是事先写死的，不是先看 2022 再挑。入选必须同时满足：
2016–2021 每年净收益为正，全期夏普不低于 0.8，最大回撤不深于 12%。
满足的里面，取最差年份夏普最高的；相同则取全期夏普更高、回撤更浅的。

规则都是同一条经济逻辑：决策时价格相对前 15 根连续分钟收盘偏离超过 0.2%，
并且与截至决策时的高点时钟同向，预估来回成本低于 6 个基点。时段按交易所小节定，
不按收益挑钟点。2022 曾被旧流程使用过，这里的结果不能当作从未看过的验证。
2023–2025 不读。
"""
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
from tfcta.data import shard_io, universe as U
from tfcta.factors.intraday import day_codes_of, raw_price_path
from tfcta.research import costs, stats
import plot_morning_rule as morning

SLOTS = {
    "早盘持有到11:30": ((9, 16), (9, 17), (11, 30)),
    "早盘只到10:15": ((9, 16), (9, 17), (10, 15)),
    "休市后到11:30": ((10, 46), (10, 47), (11, 30)),
    "下午到收盘": ((13, 46), (13, 47), (15, 0)),
}

# 名称, 使用的时段, 单品种上限, 该时段资金占账户的比例
BOOKS = [
    ("早盘20%上限", ["早盘持有到11:30"], 0.20, 1.0),
    ("早盘当天铺满", ["早盘持有到11:30"], None, 1.0),
    ("早盘与下午，各段20%上限", ["早盘持有到11:30", "下午到收盘"], 0.20, 1.0),
    ("三段各20%上限", ["早盘只到10:15", "休市后到11:30", "下午到收盘"], 0.20, 1.0),
    ("三段各用三分之一", ["早盘只到10:15", "休市后到11:30", "下午到收盘"], None, 1.0 / 3.0),
    ("三段各自铺满", ["早盘只到10:15", "休市后到11:30", "下午到收盘"], None, 1.0),
]


def fees_for(partition):
    research = costs.load_fees("research")
    if partition == "research":
        return research
    validation = costs.load_fees("validation_2022")
    return pd.concat([research, validation], ignore_index=True)


def calendar_and_members(partition):
    if partition == "research":
        universe = {year: names for year, names in U.load_universe().items() if 2016 <= year <= 2021}
        anchor = shard_io.load_shard("RB", directory=C.RESEARCH_DIR, columns=["trading_date"])
        master = pd.DatetimeIndex(sorted(set(pd.to_datetime(anchor.trading_date).dt.normalize())))
        master = master[(master >= "2016-01-01") & (master <= "2021-12-31")]
        C.assert_no_holdout_dates(master, "主策略研究日历")
        names = sorted({name for group in universe.values() for name in group})
    else:
        names, _ = U.validation_universe()
        universe = {2022: list(names)}
        anchor = shard_io.load_validation_shard("RB", columns=["trading_date"])
        master = pd.DatetimeIndex(sorted(set(pd.to_datetime(anchor.trading_date).dt.normalize())))
        C.assert_validation_2022_dates(master, "主策略验证日历")
    tradable = [name for name in names if name in costs.MULTIPLIER]
    skipped = [name for name in names if name not in costs.MULTIPLIER]
    return master, universe, tradable, skipped


def load_minutes(symbol, partition):
    columns = ["open", "close", "highw", "trading_date"]
    if partition == "research":
        frame = shard_io.load_shard(symbol, directory=C.RESEARCH_DIR, columns=columns)
    else:
        frame = shard_io.load_validation_shard(symbol, columns=columns)
    frame = frame.sort_index(kind="mergesort")
    frame["trading_date"] = pd.to_datetime(frame.trading_date).dt.normalize()
    return frame


def build(partition, needed_slots):
    master, universe, symbols, skipped = calendar_and_members(partition)
    print(f"{partition} 品种 {len(symbols)}，无乘数未计入 {skipped}", flush=True)
    loc = {day: i for i, day in enumerate(master)}
    shape = (len(master), len(symbols))
    gross = {name: np.full(shape, np.nan) for name in needed_slots}
    cost = {name: np.full(shape, np.nan) for name in needed_slots}
    signal = {name: np.full(shape, np.nan) for name in needed_slots}
    fee_table, tick_table = fees_for(partition), costs.load_ticks()
    missing = []
    for symbol in symbols:
        try:
            frame = load_minutes(symbol, partition)
        except FileNotFoundError:
            missing.append(symbol)
            continue
        raw_close = raw_price_path(frame)
        raw_open = raw_price_path(pd.DataFrame({"close": frame.open.to_numpy()}, index=frame.index))
        high = frame.highw.to_numpy(float)
        idx_ns = frame.index.asi8
        codes, days = day_codes_of(frame)
        bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
        pos = {ts: i for i, ts in enumerate(frame.index.to_numpy())}
        history = fee_table[fee_table.symbol.eq(symbol)].copy()
        history["trading_date"] = pd.to_datetime(history.trading_date)
        history = history.set_index("trading_date").sort_index()
        ticks = tick_table[tick_table.symbol.eq(symbol)].set_index("year").tick.sort_index().round(2)
        prior = ticks.rename(index=lambda year: int(year) + 1)
        prior = prior.reindex(sorted(set(prior.index) | set(range(2015, 2023)))).ffill()
        column = symbols.index(symbol)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            day = pd.Timestamp(days[int(codes[start])])
            tloc = loc.get(day)
            if tloc is None or history.empty:
                continue
            tick = float(prior.loc[day.year]) if day.year in prior.index and np.isfinite(prior.loc[day.year]) else np.nan
            row = history.reindex([day], method="ffill").iloc[0]
            if not np.isfinite(tick):
                continue
            for slot, (decision_at, entry_at, exit_at) in needed_slots.items():
                decision = pos.get(morning.stamp(day, decision_at))
                entry = pos.get(morning.stamp(day, entry_at))
                exit_bar = pos.get(morning.stamp(day, exit_at))
                if decision is None or not start <= decision < stop:
                    continue
                dev = morning.ma_dev(raw_close, idx_ns, decision, 15)
                if not np.isfinite(dev):
                    continue
                ref = float(raw_close[decision])
                if not (ref > 0 and np.isfinite(ref)):
                    continue
                kind = row.commission_type
                estimated = (
                    morning.one_rate(kind, row.open_commission, ref, ref, symbol)
                    + morning.one_rate(kind, row.close_commission_today, ref, ref, symbol)
                    + 2 * tick / ref
                )
                signal[slot][tloc, column] = morning.side_of(
                    dev, morning.clock_of(high, start, decision + 1), estimated
                )
                required = signal[slot][tloc, column] != 0 and symbol in universe.get(day.year, [])
                prices = morning.execution_prices(raw_open, raw_close, decision, entry, exit_bar,
                                                  start, stop, required)
                if prices is not None:
                    entry_px, exit_px = prices
                    gross[slot][tloc, column] = exit_px / entry_px - 1.0
                    cost[slot][tloc, column] = (
                        morning.one_rate(kind, row.open_commission, entry_px, entry_px, symbol)
                        + morning.one_rate(kind, row.close_commission_today, exit_px, entry_px, symbol)
                        + 2 * tick / entry_px
                    )
        print(symbol, flush=True)
    if missing:
        print(f"缺分片 {missing}", flush=True)
    member = np.zeros(shape, dtype=bool)
    for t, day in enumerate(master):
        for name in universe.get(int(day.year), []):
            if name in symbols:
                member[t, symbols.index(name)] = True
    for slot in signal:
        signal[slot] = np.where(member, np.nan_to_num(signal[slot]), 0.0)
    return master, signal, gross, cost


def combine(master, signal, gross, cost, spec):
    name, slots, cap, scale = spec
    daily = np.zeros(len(master))
    for slot in slots:
        weight = morning.allocate(signal[slot], cap) * scale
        daily = daily + morning.daily_pnl(weight, gross[slot], cost[slot])
    return pd.Series(daily, index=master, name=name)


def judge(daily):
    years = range(int(daily.index.year.min()), int(daily.index.year.max()) + 1)
    sharpes, returns = [], []
    for year in years:
        part = daily[daily.index.year == year]
        sharpes.append(float(stats.sharpe_ratio(part)))
        returns.append(float(stats.performance(part)["ann_return"]))
    full = stats.performance(daily)
    full_sr = float(stats.sharpe_ratio(daily))
    admitted = (
        all(value > 0 for value in returns)
        and full_sr >= 0.8
        and full["max_drawdown"] >= -0.12
        and all(np.isfinite(sharpes))
    )
    return {
        "admitted": admitted,
        "min_year_sharpe": min(sharpes),
        "full_sharpe": full_sr,
        "max_drawdown": full["max_drawdown"],
        "ann_return": full["ann_return"],
        "year_sharpe": sharpes,
        "year_return": returns,
    }


def choose(books):
    ranked = []
    for name, daily in books.items():
        score = judge(daily)
        score["name"] = name
        ranked.append(score)
        flag = "入选比较" if score["admitted"] else "不进比较"
        print(
            f"{name}  {flag}  全期夏普 {score['full_sharpe']:.2f}  年化 {score['ann_return']*100:.2f}%"
            f"  回撤 {score['max_drawdown']*100:.2f}%  最差年夏普 {score['min_year_sharpe']:.2f}",
            flush=True,
        )
        print(
            "  年化 "
            + "  ".join(f"{2016+i}:{score['year_return'][i]*100:.2f}%" for i in range(6)),
            flush=True,
        )
        print(
            "  夏普 "
            + "  ".join(f"{2016+i}:{score['year_sharpe'][i]:.2f}" for i in range(6)),
            flush=True,
        )
    admitted = [row for row in ranked if row["admitted"]]
    if not admitted:
        raise RuntimeError("没有组合通过事先写定的稳定性条件")
    admitted.sort(key=lambda row: (row["min_year_sharpe"], row["full_sharpe"], row["max_drawdown"]), reverse=True)
    return admitted[0]["name"]


def draw(daily, path):
    plt.rcParams["font.sans-serif"] = ["Heiti SC", "Songti SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    nav = (1.0 + daily).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, nav.to_numpy()])[1:]
    drawdown = nav.to_numpy() / peak - 1.0
    figure, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 7.2), gridspec_kw={"height_ratios": [2.1, 1]}
    )
    top.plot(nav.index, nav.to_numpy(), color="#1f4e79", linewidth=1.3)
    bottom.fill_between(nav.index, drawdown * 100, 0, color="#1f4e79", alpha=0.25)
    bottom.plot(nav.index, drawdown * 100, color="#1f4e79", linewidth=1.0)
    cut = pd.Timestamp("2022-01-01")
    top.axvline(cut, color="#888888", linewidth=0.8, linestyle="--")
    bottom.axvline(cut, color="#888888", linewidth=0.8, linestyle="--")
    top.axhline(1.0, color="#bbbbbb", linewidth=0.6)
    top.set_ylabel("净值")
    top.set_title("主策略净值与回撤。竖线右侧为 2022")
    bottom.axhline(0.0, color="#bbbbbb", linewidth=0.6)
    bottom.set_ylabel("回撤 (%)")
    bottom.set_xlabel("交易日")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    needed = {name: SLOTS[name] for name in {slot for _, slots, _, _ in BOOKS for slot in slots}}
    master, signal, gross, cost = build("research", needed)
    for slot, values in signal.items():
        fired = int((values != 0).any(axis=1).sum())
        print(f"时段 {slot} 有信号的交易日 {fired}", flush=True)
    books = {spec[0]: combine(master, signal, gross, cost, spec) for spec in BOOKS}
    winner = choose(books)
    print(f"主策略：{winner}", flush=True)
    spec = next(item for item in BOOKS if item[0] == winner)
    val_slots = {name: SLOTS[name] for name in spec[1]}
    val_master, val_signal, val_gross, val_cost = build("validation_2022", val_slots)
    val_daily = combine(val_master, val_signal, val_gross, val_cost, spec)
    val_perf = stats.performance(val_daily)
    val_total = float((1.0 + val_daily).prod() - 1.0)
    print(
        f"2022 净年化 {val_perf['ann_return']*100:.2f}%  当年累计 {val_total*100:.2f}%"
        f"  夏普 {stats.sharpe_ratio(val_daily):.2f}  波动 {val_perf['ann_vol']*100:.2f}%"
        f"  回撤 {val_perf['max_drawdown']*100:.2f}%  天数 {val_perf['n_days']}",
        flush=True,
    )
    print("2022 曾被旧流程使用过，上面这个数不是从未看过的验证。2023–2025 没有打开。", flush=True)
    stitched = pd.concat([books[winner], val_daily])
    out = ROOT / "runs" / "morning_rule" / "main_strategy.png"
    draw(stitched, out)
    record = {
        "winner": winner,
        "slots": spec[1],
        "cap": spec[2],
        "sleeve_scale": spec[3],
        "research": {key: judge(books[winner])[key] for key in ("ann_return", "full_sharpe", "max_drawdown", "min_year_sharpe")},
        "validation_2022": {
            "ann_return": val_perf["ann_return"],
            "total_return": val_total,
            "sharpe": float(stats.sharpe_ratio(val_daily)),
            "ann_vol": val_perf["ann_vol"],
            "max_drawdown": val_perf["max_drawdown"],
            "prior_use_disclosed": True,
        },
    }
    out.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    stitched.to_csv(out.with_suffix(".csv"))
    print(f"图：{out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
