"""早盘规则的净值与回撤，画在同一张图里。

研究期 2016–2021。信号不变：09:16 相对前 15 根连续分钟收盘偏离超过 0.2%，
且与全天高点时钟同向，预估来回成本低于 6 个基点。09:17 开盘进，11:30 收盘出。

同一条信号画两种资金用法：
- 有信号的品种平分，单品种不超过账户的 20%，剩下的留现金。
- 有信号的品种平分当天的全部资金，权重加总为 100%。没有信号的日子仍是现金。
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C
from tfcta.data import shard_io, universe as U
from tfcta.factors.intraday import day_codes_of, raw_price_path
from tfcta.research import costs, stats

DEC, ENTRY, EX = (9, 16), (9, 17), (11, 30)


def stamp(day, hm):
    ts = pd.Timestamp(day).replace(hour=int(hm[0]), minute=int(hm[1]), second=0, microsecond=0, nanosecond=0)
    return np.datetime64(ts, "ns")


def one_rate(kind, value, execution, denominator, symbol):
    if kind not in ("by_money", "by_volume") or not np.isfinite(execution) or execution <= 0:
        return np.nan
    if kind == "by_volume":
        return (value + 0.01) / (denominator * costs.MULTIPLIER[symbol])
    if value == 0:
        return 0.01 / (denominator * costs.MULTIPLIER[symbol])
    return value * 1.01 * execution / denominator


def ma_dev(raw, idx_ns, di, window=15):
    if di < window or idx_ns[di] - idx_ns[di - 1] != 60_000_000_000:
        return np.nan
    closes = raw[di - window:di]
    steps = np.diff(idx_ns[di - window:di + 1])
    if not (np.isfinite(closes).all() and (closes > 0).all() and np.all(steps == 60_000_000_000)):
        return np.nan
    price = raw[di]
    if not np.isfinite(price) or price <= 0:
        return np.nan
    return price / closes.mean() - 1.0


def clock_of(high, start, stop):
    segment = high[start:stop]
    if segment.size < 2 or not np.isfinite(segment).all():
        return np.nan
    return float(np.argmax(segment) / (segment.size - 1))


def side_of(dev, high_clock, estimated):
    if not (np.isfinite(estimated) and estimated < 6e-4 and np.isfinite(high_clock) and np.isfinite(dev)):
        return 0.0
    if abs(dev) <= 0.002:
        return 0.0
    momentum = -1.0 if dev > 0 else 1.0
    if high_clock >= 2 / 3:
        clock = -1.0
    elif high_clock <= 1 / 3:
        clock = 1.0
    else:
        return 0.0
    return momentum if momentum == clock else 0.0


def allocate(signal, cap):
    """有信号的品种平分。cap 为单品种上限；None 表示当天有信号就把资金铺满。"""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or np.isinf(signal).any():
        raise ValueError('信号须为二维有限值或 NaN')
    if cap is not None and (not np.isfinite(cap) or not 0 < cap <= 1):
        raise ValueError('单品种上限须在 (0, 1]')
    signal = np.where(np.isnan(signal), 0., signal)
    active = signal != 0
    count = active.sum(axis=1)
    share = np.zeros(signal.shape)
    ok = count > 0
    share[ok] = 1.0 / count[ok, None]
    if cap is not None:
        share = np.minimum(share, cap)
    return np.where(active, np.sign(signal) * share, 0.0)


def daily_pnl(weight, gross, cost):
    weight, gross, cost = [np.asarray(x, dtype=float) for x in (weight, gross, cost)]
    if weight.ndim != 2 or weight.shape != gross.shape or weight.shape != cost.shape:
        raise ValueError('仓位、收益和费用须为相同形状的二维数组')
    if not np.isfinite(weight).all():
        raise ValueError('持仓权重须为有限值')
    active = weight != 0
    if not (np.isfinite(gross[active]).all() and np.isfinite(cost[active]).all()):
        raise ValueError('实际持仓对应的收益或费用缺失/非有限')
    if (cost[active] < 0).any():
        raise ValueError('实际持仓费用不能为负')
    pnl = np.zeros_like(weight)
    pnl[active] = weight[active] * gross[active] - np.abs(weight[active]) * cost[active]
    daily = pnl.sum(axis=1)
    if not np.isfinite(daily).all():
        raise ValueError('持仓组合收益非有限')
    return daily


def execution_prices(raw_open, raw_close, decision, entry, exit_at, start, stop, required):
    """Called after the signal: missing future execution must never erase it."""
    valid = (entry is not None and exit_at is not None
             and start <= decision < entry <= exit_at < stop)
    if valid:
        prices = float(raw_open[entry]), float(raw_close[exit_at])
        valid = np.isfinite(prices).all() and min(prices) > 0
    if not valid:
        if required:
            raise ValueError('已发出信号，但成交或退出行情缺失/非法')
        return None
    return prices


def build(master, universe, symbols):
    loc = {d: i for i, d in enumerate(master)}
    days_n, names_n = len(master), len(symbols)
    gross = np.full((days_n, names_n), np.nan)
    cost = np.full((days_n, names_n), np.nan)
    signal = np.full((days_n, names_n), np.nan)
    fee_table, tick_table = costs.load_fees(), costs.load_ticks()
    for symbol in symbols:
        frame = shard_io.load_shard(
            symbol, directory=C.RESEARCH_DIR, columns=["open", "close", "highw", "trading_date"]
        )
        frame = frame.sort_index(kind="mergesort")
        frame["trading_date"] = pd.to_datetime(frame.trading_date).dt.normalize()
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
        prior = prior.reindex(sorted(set(prior.index) | set(range(2015, 2022)))).ffill()
        name_at = symbols.index(symbol)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            day = pd.Timestamp(days[int(codes[start])])
            tloc = loc.get(day)
            if tloc is None:
                continue
            decision = pos.get(stamp(day, DEC))
            entry = pos.get(stamp(day, ENTRY))
            exit_at = pos.get(stamp(day, EX))
            if decision is None or not start <= decision < stop or history.empty:
                continue
            dev = ma_dev(raw_close, idx_ns, decision, 15)
            if not np.isfinite(dev):
                continue
            ref = float(raw_close[decision])
            tick = float(prior.loc[day.year]) if day.year in prior.index and np.isfinite(prior.loc[day.year]) else np.nan
            row = history.reindex([day], method="ffill").iloc[0]
            if not (ref > 0 and np.isfinite([ref, tick]).all()):
                continue
            kind = row.commission_type
            estimated = (
                one_rate(kind, row.open_commission, ref, ref, symbol)
                + one_rate(kind, row.close_commission_today, ref, ref, symbol)
                + 2 * tick / ref
            )
            signal[tloc, name_at] = side_of(dev, clock_of(high, start, decision + 1), estimated)
            required = signal[tloc, name_at] != 0 and symbol in universe.get(day.year, [])
            prices = execution_prices(raw_open, raw_close, decision, entry, exit_at,
                                      start, stop, required)
            if prices is not None:
                entry_px, exit_px = prices
                gross[tloc, name_at] = exit_px / entry_px - 1.0
                cost[tloc, name_at] = (
                    one_rate(kind, row.open_commission, entry_px, entry_px, symbol)
                    + one_rate(kind, row.close_commission_today, exit_px, entry_px, symbol)
                    + 2 * tick / entry_px
                )
        print(symbol, flush=True)
    member = np.zeros((days_n, names_n), dtype=bool)
    for t, day in enumerate(master):
        for name in universe.get(int(day.year), []):
            if name in symbols:
                member[t, symbols.index(name)] = True
    signal = np.where(member, np.nan_to_num(signal), 0.0)
    books = {
        "单品种上限 20%": allocate(signal, 0.20),
        "当天资金铺满": allocate(signal, None),
    }
    out = pd.DataFrame({name: daily_pnl(weight, gross, cost) for name, weight in books.items()}, index=master)
    out.index.name = "trading_date"
    return out, signal


def describe(name, daily):
    perf = stats.performance(daily)
    print(
        f"{name}  净年化 {perf['ann_return']*100:.2f}%  夏普 {stats.sharpe_ratio(daily):.2f}"
        f"  波动 {perf['ann_vol']*100:.2f}%  最大回撤 {perf['max_drawdown']*100:.2f}%",
        flush=True,
    )
    for year in range(2016, 2022):
        part = daily[daily.index.year == year]
        year_perf = stats.performance(part)
        print(
            f"  {year}  年化 {year_perf['ann_return']*100:6.2f}%  夏普 {stats.sharpe_ratio(part):5.2f}",
            flush=True,
        )


def drawdown_of(daily):
    nav = (1.0 + daily).cumprod()
    return nav / nav.cummax().clip(lower=1.0) - 1.0


def draw(daily, path):
    plt.rcParams["font.sans-serif"] = ["Heiti SC", "Songti SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    nav = (1.0 + daily).cumprod()
    drawdown = drawdown_of(daily)
    colors = {"单品种上限 20%": "#1f4e79", "当天资金铺满": "#b85c38"}
    figure, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 7.2), gridspec_kw={"height_ratios": [2.1, 1]}
    )
    for name in daily.columns:
        top.plot(nav.index, nav[name], color=colors[name], linewidth=1.4, label=name)
        bottom.fill_between(drawdown.index, drawdown[name] * 100, 0, color=colors[name], alpha=0.25)
        bottom.plot(drawdown.index, drawdown[name] * 100, color=colors[name], linewidth=1.0)
    top.axhline(1.0, color="#888888", linewidth=0.6)
    top.set_ylabel("净值")
    top.legend(frameon=False, loc="upper left")
    top.set_title("早盘规则，2016–2021")
    bottom.axhline(0.0, color="#888888", linewidth=0.6)
    bottom.set_ylabel("回撤 (%)")
    bottom.set_xlabel("交易日")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/morning_rule/nav_drawdown.png"))
    args = parser.parse_args()
    universe = {year: names for year, names in U.load_universe().items() if 2016 <= year <= 2021}
    symbols = sorted({name for names in universe.values() for name in names if name in costs.MULTIPLIER})
    anchor = shard_io.load_shard("RB", directory=C.RESEARCH_DIR, columns=["trading_date"])
    master = pd.DatetimeIndex(sorted(set(pd.to_datetime(anchor.trading_date).dt.normalize())))
    master = master[(master >= "2016-01-01") & (master <= "2021-12-31")]
    C.assert_no_holdout_dates(master, "早盘净值图")
    daily, signal = build(master, universe, symbols)
    active = (signal != 0).sum(axis=1)
    fired = active[active > 0]
    print(
        f"有信号的交易日 {len(fired)}/{len(active)}。"
        f"这些日子里信号个数的中位数是 {np.median(fired):.0f}，"
        f"少于 5 个、因而 20% 上限会留下现金的日子占 {np.mean(fired < 5)*100:.0f}%。",
        flush=True,
    )
    for name in daily.columns:
        describe(name, daily[name])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    daily.to_csv(csv_path)
    draw(daily, args.output)
    print(f"图：{args.output.resolve()}", flush=True)
    print(f"日收益：{csv_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
