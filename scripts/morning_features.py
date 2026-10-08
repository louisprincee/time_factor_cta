"""早盘研究的逐日特征表：每个品种每个交易日一行，决策时点 09:16 收盘。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tfcta import config as C
from tfcta.data import shard_io, universe as U
from tfcta.factors import day_codes_of, raw_price_path
from tfcta.research import costs

OUT = ROOT / "runs" / "morning_features"
DECISION, ENTRY, LATE_ENTRY = (9, 16), (9, 17), (9, 18)
EXITS = {"1015": (10, 15), "1130": (11, 30), "1500": (15, 0)}
COLUMNS = ["open", "close", "closew", "openw", "highw", "loww", "volume", "trading_date"]
MINUTE = 60_000_000_000


def stamp(day, hm):
    return pd.Timestamp(day).value + (hm[0] * 60 + hm[1]) * MINUTE


def fee_rate(kind, value, price, base, symbol):
    """单边手续费占名义金额比例：按金额 ×1.01，按手数加 0.01 元，零费率按 0.01 元。"""
    if kind not in ("by_money", "by_volume") or not np.isfinite(price) or price <= 0:
        return np.nan
    if kind == "by_volume":
        return (value + 0.01) / (base * costs.MULTIPLIER[symbol])
    if value == 0:
        return 0.01 / (base * costs.MULTIPLIER[symbol])
    return value * 1.01 * price / base


def clock(values, fn):
    """极值在区间里出现的相对位置，0 为开头、1 为结尾；同值取最早一次。"""
    if values.size < 2 or not np.isfinite(values).all():
        return np.nan
    return float(fn(values) / (values.size - 1))


def load_minutes(symbol, partition, end=None):
    if partition == "research":
        frame = shard_io.load_shard(symbol, directory=C.RESEARCH_DIR, columns=COLUMNS)
    elif partition == "validation_2022":
        tail = shard_io.load_shard(symbol, directory=C.RESEARCH_DIR, columns=COLUMNS)
        tail = tail[pd.to_datetime(tail.trading_date) >= "2021-01-01"]
        frame = pd.concat([tail, shard_io.load_validation_shard(symbol, columns=COLUMNS)])
    else:  # 样本外：2022 做预热，只能在 C.final_evaluation() 块内读
        parts = []
        if shard_io.find_shard(C.VALIDATION_DIR, symbol) is not None:
            parts.append(shard_io.load_validation_shard(symbol, columns=COLUMNS))
        parts.append(shard_io.load_oos_shard(symbol, end=end, columns=COLUMNS))
        frame = pd.concat(parts)
    frame = frame.sort_index(kind="mergesort")
    frame["trading_date"] = pd.to_datetime(frame.trading_date).dt.normalize()
    return frame[frame.trading_date >= "2014-01-01"]


def fee_history(partition):
    table = costs.load_fees("research")
    if partition != "research":
        table = pd.concat([table, costs.load_fees("validation_2022")], ignore_index=True)
    if partition == "oos":
        table = pd.concat([table, costs.load_fees("oos")], ignore_index=True)
    table["trading_date"] = pd.to_datetime(table.trading_date)
    return table


def prior_ticks(tick_table, symbol, last_year=2022):
    """当年用上一年观测到的最小价位变动；首年没有就留空，之后没有新观测就沿用最近一年。"""
    ticks = tick_table[tick_table.symbol.eq(symbol)].set_index("year").tick.sort_index().round(2)
    prior = ticks.rename(index=lambda year: int(year) + 1)
    return prior.reindex(range(2014, last_year + 1)).ffill()


def symbol_rows(symbol, partition, fees, tick_table, end=None):
    frame = load_minutes(symbol, partition, end)
    raw = raw_price_path(frame)
    raw_open = raw_price_path(pd.DataFrame({"close": frame.open.to_numpy()}, index=frame.index))
    cw, ow = frame.closew.to_numpy(float), frame.openw.to_numpy(float)
    hw, lw = frame.highw.to_numpy(float), frame.loww.to_numpy(float)
    vol = frame.volume.to_numpy(float)
    idx = frame.index.asi8
    hour = frame.index.hour.to_numpy()
    night = (hour >= 20) | (hour <= 4)
    codes, days = day_codes_of(frame)
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    pos = {int(ts): i for i, ts in enumerate(idx)}
    history = fees[fees.symbol.eq(symbol)].set_index("trading_date").sort_index()
    fee_rows = history.reindex(days, method="ffill")  # 只向前填充
    ticks = prior_ticks(tick_table, symbol, max(2022, pd.Timestamp(end).year if end else 2022))
    rows = []
    for k, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        day = days[int(codes[s])]
        row = {"symbol": symbol, "trading_date": day}
        # 全日特征：只用于次日及以后
        if s > 0:
            base = raw[s - 1]
            row["day_ret"] = (cw[e - 1] - cw[s - 1]) / base
        hi, lo = hw[s:e].max(), lw[s:e].min()
        row["ts_high"] = clock(hw[s:e], np.argmax)
        row["ts_low"] = clock(lw[s:e], np.argmin)
        row["clv"] = (cw[e - 1] - lo) / (hi - lo) if hi > lo else np.nan
        row["day_range"] = (hi - lo) / raw[e - 1]
        row["day_volume"] = vol[s:e].sum()
        pm = np.flatnonzero(hour[s:e] >= 13) + s
        row["ts_high_pm"] = clock(hw[pm], np.argmax) if pm.size else np.nan
        row["ts_low_pm"] = clock(lw[pm], np.argmin) if pm.size else np.nan
        d = pos.get(stamp(day, DECISION))
        day_bars = np.flatnonzero(~night[s:e]) + s
        if d is None or not s <= d < e or s == 0 or not day_bars.size or day_bars[0] > d:
            rows.append(row)
            continue
        f = int(day_bars[0])
        prev = raw[s - 1]
        row["first_bar"] = pd.Timestamp(idx[f]).strftime("%H:%M")
        row["ref"] = raw[d]
        row["ret_on"] = (cw[d] - cw[s - 1]) / prev
        row["ret_night"] = (cw[f - 1] - cw[s - 1]) / prev if f > s else np.nan
        row["night_volume"] = vol[s:f].sum() if f > s else np.nan
        row["gap"] = (ow[f] - cw[f - 1]) / raw[f - 1]
        row["ret_pre"] = (cw[d] - cw[f - 1]) / raw[f - 1]
        row["rest_ret"] = (cw[e - 1] - cw[d]) / raw[d]
        window = raw[d - 15:d] if d >= 15 else np.array([])
        steps = np.diff(idx[d - 15:d + 1]) if d >= 15 else np.array([])
        if window.size == 15 and np.all(steps == MINUTE) and (window > 0).all():
            row["dev15"] = raw[d] / window.mean() - 1.0
        v = vol[f:d + 1]
        row["pre_volume"] = v.sum()
        if v.sum() > 0:
            row["vwap_dev"] = raw[d] / (np.dot(raw[f:d + 1], v) / v.sum()) - 1.0
        path = np.abs(np.diff(cw[f - 1:d + 1])).sum()
        row["eff_pre"] = abs(cw[d] - cw[f - 1]) / path if path > 0 else 0.0
        row["range_pre"] = (hw[f:d + 1].max() - lw[f:d + 1].min()) / raw[d]
        row["hclock_td"] = clock(hw[s:d + 1], np.argmax)
        row["lclock_td"] = clock(lw[s:d + 1], np.argmin)
        row["hclock_pre"] = clock(hw[f:d + 1], np.argmax)
        row["lclock_pre"] = clock(lw[f:d + 1], np.argmin)
        # 结果列：下一根开盘进场
        entry = pos.get(stamp(day, ENTRY))
        fee = fee_rows.iloc[k]
        tick = ticks.get(day.year, np.nan)
        kind = fee.commission_type if isinstance(fee.commission_type, str) else None
        row["est_cost"] = (fee_rate(kind, fee.open_commission, raw[d], raw[d], symbol)
                           + fee_rate(kind, fee.close_commission_today, raw[d], raw[d], symbol)
                           + 2 * tick / raw[d])
        if entry is not None and d < entry < e and raw_open[entry] > 0:
            px = raw_open[entry]
            row["entry_flat"] = bool(hw[entry] == lw[entry])
            for name, hm in EXITS.items():
                x = pos.get(stamp(day, hm))
                if x is None or not entry <= x < e or not raw[x] > 0:
                    continue
                row[f"gross_{name}"] = raw[x] / px - 1.0
                row[f"cost_{name}"] = (fee_rate(kind, fee.open_commission, px, px, symbol)
                                       + fee_rate(kind, fee.close_commission_today, raw[x], px, symbol)
                                       + 2 * tick / px)
            # 延迟一分钟进场的压力测试
            late = pos.get(stamp(day, LATE_ENTRY))
            x = pos.get(stamp(day, EXITS["1130"]))
            if late is not None and x is not None and entry < late < x < e and raw_open[late] > 0:
                row["gross_late_1130"] = raw[x] / raw_open[late] - 1.0
            row["tick_frac"] = tick / px
        rows.append(row)
    out = pd.DataFrame(rows)
    return out


def build(partition, end=None, names=None):
    """样本外（partition="oos"）由 scripts/oos_portfolio.py 在打开的最终评估里调用，传入截止日和品种。"""
    if partition == "oos":
        if end is None or names is None:
            raise ValueError("样本外特征需要截止日和品种池")
        end = C.to_date(end)
    elif partition == "research":
        names = sorted({n for group in U.load_universe().values() for n in group})
    else:
        names, _ = U.validation_universe()
    names = [n for n in names if n in costs.MULTIPLIER]
    fees, ticks = fee_history(partition), costs.load_ticks()
    parts = []
    for symbol in names:
        try:
            parts.append(symbol_rows(symbol, partition, fees, ticks, end))
        except FileNotFoundError:
            print(symbol, "无分片，跳过", flush=True)
            continue
        print(symbol, len(parts[-1]), flush=True)
    table = pd.concat(parts, ignore_index=True)
    if partition == "research":
        C.assert_no_holdout_dates(table.trading_date, "早盘特征")
    elif partition == "oos":
        table = table[(table.trading_date >= pd.Timestamp(C.HOLDOUT_START))
                      & (table.trading_date <= pd.Timestamp(end))]
        late = table.trading_date >= pd.Timestamp(C.STRICT_OOS_START)
        C.assert_validation_2022_dates(table.trading_date[~late], "早盘特征预热")
        C.assert_strict_oos_dates(table.trading_date[late], "早盘特征")
    else:
        table = table[table.trading_date >= "2021-01-01"]
        late = table.trading_date >= pd.Timestamp(C.HOLDOUT_START)
        C.assert_no_holdout_dates(table.trading_date[~late], "早盘特征预热")
        C.assert_validation_2022_dates(table.trading_date[late], "早盘特征")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT / f"{partition}.parquet", index=False)
    return table


if __name__ == "__main__":
    for part in (sys.argv[1:] or ["research", "validation_2022"]):
        if part not in ("research", "validation_2022"):
            raise SystemExit("样本外特征只能由 scripts/oos_portfolio.py 生成")
        build(part)
