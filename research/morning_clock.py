"""早盘时钟因子研究。只使用 2016–2021，不构造策略，不读 2022 及以后。

    python -m research.morning_clock features
    python -m research.morning_clock return
    python -m research.morning_clock return --exit 1500
    python -m research.morning_clock return --balance
    python -m research.morning_clock cases
    python -m research.morning_clock path
    python -m research.morning_clock flow
    python -m research.morning_clock state
"""
from __future__ import annotations

import argparse
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from tfcta import config as C
from tfcta.costs import (
    MULTIPLIER, commission_rate, estimate_ticks, load_fees, load_ticks, require_multipliers,
)
from tfcta.data import (
    find_shard, load_oos_shard, load_shard, load_universe, load_validation_shard,
    validation_universe,
)
from tfcta.factors import align_asof, day_codes_of, raw_price_path, _series

RUNS = C.RUNS_DIR / "commodity" / "morning"
FEATURES = RUNS / "features"
RETURN_OUT = RUNS / "clock_return"
CASES_OUT = RUNS / "clock_cases"
PATH_OUT = RUNS / "clock_path"
FLOW_OUT = RUNS / "clock_flow"
STATE_OUT = RUNS / "clock_state"
PATH_CONFIG = C.PROJECT_ROOT / "config" / "research" / "clock_path.json"
FLOW_CONFIG = C.PROJECT_ROOT / "config" / "research" / "clock_flow.json"
STATE_CONFIG = C.PROJECT_ROOT / "config" / "research" / "clock_state.json"

SECTORS = {
    "农产品": "A B C CS M Y P JD LH RR AP CF CY SR OI RM RS PK CJ WH PM RI LR JR".split(),
    "黑色": "I J JM RB HC WR SS SF SM ZC".split(),
    "有色": "CU AL ZN PB NI SN BC AO AD SI LC PS".split(),
    "贵金属": "AU AG".split(),
    "能源": "BU FU LU SC PG".split(),
    "化工": "L V PP EG EB RU NR BR SP TA MA FG SA UR PF PX PR SH".split(),
    "建材": "BB FB LG".split(),
    "航运": "EC".split(),
}
SECTOR_OF = {symbol: name for name, symbols in SECTORS.items() for symbol in symbols}
DECISION, ENTRY, EXIT = (9, 16), (9, 17), (11, 30)
COLUMNS = ["open", "close", "closew", "highw", "loww", "volume", "trading_date"]
MINUTE = 60_000_000_000


# --- 特征表：决策时点 09:16 ---

def stamp(day, hm):
    return pd.Timestamp(day).value + (hm[0] * 60 + hm[1]) * MINUTE


def fee_rate(kind, value, price, base, symbol):
    """单边手续费占名义金额比例，公式在 commission_rate。"""
    return commission_rate(kind, value, price, base, MULTIPLIER[symbol])


def session_clock(values, fn):
    """极值在区间里出现的相对位置，0 为开头、1 为结尾；同值取最早一次。"""
    if values.size < 2 or not np.isfinite(values).all():
        return np.nan
    return float(fn(values) / (values.size - 1))


def load_minutes(symbol, partition, end=None):
    if partition == "research":
        frame = load_shard(symbol, directory=C.RESEARCH_DIR, columns=COLUMNS)
    elif partition == "validation_2022":
        tail = load_shard(symbol, directory=C.RESEARCH_DIR, columns=COLUMNS)
        tail = tail[pd.to_datetime(tail.trading_date) >= "2021-01-01"]
        frame = pd.concat([tail, load_validation_shard(symbol, columns=COLUMNS)])
    else:  # 样本外：2022 做预热，只能在 C.final_evaluation() 块内读
        parts = []
        if find_shard(C.VALIDATION_DIR, symbol) is not None:
            parts.append(load_validation_shard(symbol, columns=COLUMNS))
        parts.append(load_oos_shard(symbol, end=end, columns=COLUMNS))
        frame = pd.concat(parts)
    frame = frame.sort_index(kind="mergesort")
    frame["trading_date"] = pd.to_datetime(frame.trading_date).dt.normalize()
    return frame[frame.trading_date >= "2014-01-01"]


def fee_history(partition):
    table = load_fees("research")
    if partition != "research":
        table = pd.concat([table, load_fees("validation_2022")], ignore_index=True)
    if partition == "oos":
        table = pd.concat([table, load_fees("oos")], ignore_index=True)
    table["trading_date"] = pd.to_datetime(table.trading_date)
    return fill_close_today(table)


CZCE_CLOSE_TODAY_FROM = pd.Timestamp("2017-06-23")  # 米筐郑商所平今费从这天起才有


def fill_close_today(table):
    """米筐的郑商所平今费率到 2017-06-22 才有。缺的日子取（平仓费，该品种后来第一次出现的平今费）中较高的一个，
    宁可把成本估高；同一品种的费率类型必须一致才这样填。"""
    table = table.sort_values(["symbol", "trading_date"]).copy()
    later = table.groupby("symbol").close_commission_today.transform(lambda s: s.bfill())
    source_kind = table.commission_type.where(table.close_commission_today.notna())
    same_kind = table.commission_type.eq(source_kind.groupby(table.symbol).transform(lambda s: s.bfill()))
    known_gap = table.trading_date < CZCE_CLOSE_TODAY_FROM  # 只补这段已知的数据缺口，别处缺了照样报错
    gap = table.close_commission_today.isna() & later.notna() & same_kind & known_gap
    table.loc[gap, "close_commission_today"] = np.maximum(table.close_commission[gap], later[gap])
    table["close_today_filled"] = gap
    return table


def prior_ticks(tick_table, symbol, last_year=2022):
    """当年用上一年观测到的最小价位变动；首年没有就留空，之后没有新观测就沿用最近一年。"""
    ticks = tick_table[tick_table.symbol.eq(symbol)].set_index("year").tick.sort_index().round(2)
    prior = ticks.rename(index=lambda year: int(year) + 1)
    return prior.reindex(range(2014, last_year + 1)).ffill()


def holding_gross(symbol, partition, exit_hm, end=None):
    """09:17 开盘到 exit_hm 收盘的多头收益。进场条件与 gross_1130 相同，没有对应 K 线时为空。"""
    frame = load_minutes(symbol, partition, end)
    raw = raw_price_path(frame)
    raw_open = raw_price_path(pd.DataFrame({"close": frame.open.to_numpy()}, index=frame.index))
    idx = frame.index.asi8
    hour = frame.index.hour.to_numpy()
    night = (hour >= 20) | (hour <= 4)
    codes, days = day_codes_of(frame)
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    pos = {int(ts): i for i, ts in enumerate(idx)}
    rows = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        day = days[int(codes[s])]
        d = pos.get(stamp(day, DECISION))
        day_bars = np.flatnonzero(~night[s:e]) + s
        gross = np.nan
        if d is not None and s <= d < e and day_bars.size and day_bars[0] <= d:
            entry = pos.get(stamp(day, ENTRY))
            x = pos.get(stamp(day, exit_hm))
            if (entry is not None and x is not None and d < entry < e and entry <= x < e
                    and raw_open[entry] > 0 and raw[x] > 0):
                gross = raw[x] / raw_open[entry] - 1.0
        rows.append({"symbol": symbol, "trading_date": day, "gross": gross})
    return pd.DataFrame(rows)


def symbol_rows(symbol, partition, fees, tick_table, end=None):
    frame = load_minutes(symbol, partition, end)
    raw = raw_price_path(frame)
    raw_open = raw_price_path(pd.DataFrame({"close": frame.open.to_numpy()}, index=frame.index))
    cw = frame.closew.to_numpy(float)
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
    if partition == "oos":  # 2023 起的年份没有落盘的 tick 表，用已读入的分钟价现算；第 y 年的值只用于 y+1 年
        seen = estimate_ticks(raw, frame.trading_date)
        seen = seen[seen.year >= C.STRICT_OOS_START.year].assign(symbol=symbol)
        tick_table = pd.concat([tick_table, seen], ignore_index=True)
    ticks = prior_ticks(tick_table, symbol, max(2022, pd.Timestamp(end).year if end else 2022))
    rows = []
    for k, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        day = days[int(codes[s])]
        row = {"symbol": symbol, "trading_date": day}
        d = pos.get(stamp(day, DECISION))
        day_bars = np.flatnonzero(~night[s:e]) + s
        if d is None or not s <= d < e or s == 0 or not day_bars.size or day_bars[0] > d:
            rows.append(row)
            continue
        f = int(day_bars[0])
        prev = raw[s - 1]
        row["ret_night"] = (cw[f - 1] - cw[s - 1]) / prev if f > s else np.nan
        window = raw[d - 15:d] if d >= 15 else np.array([])
        steps = np.diff(idx[d - 15:d + 1]) if d >= 15 else np.array([])
        if window.size == 15 and np.all(steps == MINUTE) and (window > 0).all():
            row["dev15"] = raw[d] / window.mean() - 1.0
        row["pre_volume"] = vol[f:d + 1].sum()
        row["hclock_td"] = session_clock(hw[s:d + 1], np.argmax)
        row["lclock_td"] = session_clock(lw[s:d + 1], np.argmin)
        entry = pos.get(stamp(day, ENTRY))
        fee = fee_rows.iloc[k]
        tick = ticks.get(day.year, np.nan)
        kind = fee.commission_type if isinstance(fee.commission_type, str) else None
        row["est_cost"] = (fee_rate(kind, fee.open_commission, raw[d], raw[d], symbol)
                           + fee_rate(kind, fee.close_commission_today, raw[d], raw[d], symbol)
                           + 2 * tick / raw[d])
        if entry is not None and d < entry < e and raw_open[entry] > 0:
            px = raw_open[entry]
            x = pos.get(stamp(day, EXIT))
            if x is not None and entry <= x < e and raw[x] > 0:
                row["gross_1130"] = raw[x] / px - 1.0
                row["cost_1130"] = (fee_rate(kind, fee.open_commission, px, px, symbol)
                                    + fee_rate(kind, fee.close_commission_today, raw[x], px, symbol)
                                    + 2 * tick / px)
            row["tick_frac"] = tick / px
        rows.append(row)
    out = pd.DataFrame(rows)
    return out


def check_costs(table, pools):
    """池内品种在池内年份必须算得出成本（手续费和上一年的最小变动价位）；缺了就停，不悄悄不交易。"""
    year = table.trading_date.dt.year
    member = pd.Series([s in pools.get(y, ()) for s, y in zip(table.symbol, year)], index=table.index)
    ready = table.dev15.notna() & member
    missing = table[ready & table.est_cost.isna()]
    if len(missing):
        share = missing.groupby([missing.symbol, missing.trading_date.dt.year]).size() /             table[ready].groupby([table.symbol, year]).size()
        bad = share[share > 0.05]
        if len(bad):
            raise ValueError(f"池内品种缺成本（手续费或 tick），按品种、年的缺失比例: {bad.round(3).to_dict()}")


def build_feature_table(partition, end=None, names=None, pools=None):
    """样本外（partition="oos"）只能在打开的 C.final_evaluation(...) 里调用，传入截止日和各年品种池。"""
    if partition == "oos":
        if end is None or pools is None:
            raise ValueError("样本外特征需要截止日和各年品种池")
        end = C.to_date(end)
        names = sorted({n for group in pools.values() for n in group})
    elif partition == "research":
        pools = load_universe()
        names = sorted({n for group in pools.values() for n in group})
    else:
        names, _ = validation_universe()
        pools = {C.VALIDATION_YEAR: list(names)}
    names = require_multipliers(names)
    fees, ticks = fee_history(partition), load_ticks("research" if partition == "research" else "validation_2022")
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
    check_costs(table[table.trading_date.dt.year.isin(pools)], pools)
    FEATURES.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FEATURES / f"{partition}.parquet", index=False)
    return table

# --- 时钟位置与持仓收益 ---

CLOCK = 2.0 / 3.0
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"]
DIFF_BIN_EDGES = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
DIFF_BIN_LABELS = ["-1 to -0.6", "-0.6 to -0.2", "-0.2 to 0.2", "0.2 to 0.6", "0.6 to 1"]
SEGMENT_LABELS = ["最早\n第1段", "偏早\n第2段", "中间\n第3段", "偏晚\n第4段", "最晚\n第5段"]
OLD_PLOTS = ("bins.png", "yearly.png", "samples.png", "year_sector.png", "by_symbol.png")


def cluster_slope(day, x, y):
    """y 对 x 的斜率。标准误按交易日聚类；交易日少于 3 天时不给 t。"""
    day = np.asarray(day)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    day, x, y = day[keep], x[keep], y[keep]
    empty = {"笔数": int(len(x)), "天数": int(pd.unique(day).size) if len(day) else 0,
             "斜率": np.nan, "标准误": np.nan, "t": np.nan, "双尾p": np.nan}
    if len(x) < 3 or np.unique(x).size < 2:
        return empty
    design = np.column_stack([np.ones(len(x)), x])
    try:
        bread = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return empty
    beta = bread @ (design.T @ y)
    resid = y - design @ beta
    order = np.argsort(day, kind="mergesort")
    day, design, resid = day[order], design[order], resid[order]
    _, starts, counts = np.unique(day, return_index=True, return_counts=True)
    meat = np.zeros((2, 2))
    for start, count in zip(starts, counts):
        score = design[start:start + count].T @ resid[start:start + count]
        meat += np.outer(score, score)
    groups = int(len(starts))
    if groups < 3:
        empty.update({"斜率": float(beta[1]), "天数": groups})
        return empty
    scale = (groups / (groups - 1)) * ((len(x) - 1) / (len(x) - 2))
    variance = float((bread @ meat @ bread * scale)[1, 1])
    se = np.sqrt(variance) if variance > 0 else np.nan
    t_stat = beta[1] / se if np.isfinite(se) and se > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(t_stat), groups - 1)) if np.isfinite(t_stat) else np.nan
    return {"笔数": int(len(x)), "天数": groups, "斜率": float(beta[1]), "标准误": float(se),
            "t": float(t_stat), "双尾p": p_value}


def association(day, x, y):
    """Spearman，以及原始收益对时钟的聚类斜率。斜率的单位随后再换成基点。"""
    day = np.asarray(day)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    day, x, y = day[keep], x[keep], y[keep]
    spearman = stats.spearmanr(x, y).correlation if len(x) >= 3 and np.unique(x).size > 1 else np.nan
    ranked = cluster_slope(day, stats.rankdata(x), stats.rankdata(y)) if len(x) else cluster_slope(day, x, y)
    raw = cluster_slope(day, x, y)
    return {"笔数": raw["笔数"], "天数": raw["天数"], "Spearman": float(spearman) if np.isfinite(spearman) else np.nan,
            "Spearman_t": ranked["t"], "Spearman_p": ranked["双尾p"],
            "斜率": raw["斜率"], "标准误": raw["标准误"], "斜率_t": raw["t"], "斜率_p": raw["双尾p"]}


def late_minus_early(day, clock, y, low=0.0, high=1.0):
    """取值区间 [low, high] 的上三分之一减下三分之一的平均收益，中间三分之一不进入对比。
    时钟是 [0, 1]：≥ 2/3 减 ≤ 1/3；低点时钟减高点时钟是 [−1, 1]：≥ 1/3 减 ≤ −1/3。"""
    clock = np.asarray(clock, dtype=float)
    y = np.asarray(y, dtype=float)
    day = np.asarray(day)
    lower = low + (high - low) * (1 - CLOCK)
    upper = low + (high - low) * CLOCK
    keep = np.isfinite(clock) & np.isfinite(y) & ((clock <= lower) | (clock >= upper))
    return cluster_slope(day[keep], (clock[keep] >= upper).astype(float), y[keep])


def describe(sample, scope, item):
    link = association(item["day"], item["x"], item["y"])
    span = item.get("range", (0.0, 1.0))
    gap = late_minus_early(item["day"], item["x"], item["y"], *span)
    crit = stats.t.ppf(0.975, link["天数"] - 1) if link["天数"] >= 3 else np.nan
    return {
        "样本": sample, "范围": scope, "关系": item["关系"], "预期方向": item["预期"],
        "笔数": link["笔数"], "天数": link["天数"],
        "Spearman": link["Spearman"], "Spearman聚类t": link["Spearman_t"], "Spearman双尾p": link["Spearman_p"],
        "时钟0到1的收益bp": link["斜率"] * 1e4, "斜率聚类t": link["斜率_t"], "斜率双尾p": link["斜率_p"],
        "斜率bp下限": (link["斜率"] - crit * link["标准误"]) * 1e4,
        "斜率bp上限": (link["斜率"] + crit * link["标准误"]) * 1e4,
        "晚减早bp": gap["斜率"] * 1e4, "晚减早t": gap["t"], "晚减早p": gap["双尾p"],
    }


def bin_means(sample, scope, item):
    clock = pd.Series(np.asarray(item["x"], dtype=float))
    group = pd.cut(clock, BIN_EDGES, include_lowest=True, labels=BIN_LABELS)
    day = np.asarray(item["day"])
    y = np.asarray(item["y"], dtype=float)
    return [bin_row(sample, scope, item["关系"], label, day[(group == label).to_numpy()], y[(group == label).to_numpy()])
            for label in BIN_LABELS]


def difference_bin_means(sample, scope, item):
    difference = pd.Series(np.asarray(item["x"], dtype=float))
    group = pd.cut(difference, DIFF_BIN_EDGES, include_lowest=True, labels=DIFF_BIN_LABELS)
    day = np.asarray(item["day"])
    y = np.asarray(item["y"], dtype=float)
    return [bin_row(sample, scope, item["关系"], label, day[(group == label).to_numpy()], y[(group == label).to_numpy()])
            for label in DIFF_BIN_LABELS]


def bin_row(sample, scope, name, label, day, y):
    """一个时钟区间里，收益均值和按日聚类的 95% 区间。"""
    day = np.asarray(day)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(y)
    day, y = day[keep], y[keep]
    if len(y) == 0:
        return {"样本": sample, "范围": scope, "关系": name, "时钟区间": label, "笔数": 0, "天数": 0,
                "均值bp": np.nan, "下限bp": np.nan, "上限bp": np.nan}
    order = np.argsort(day, kind="mergesort")
    day, y = day[order], y[order]
    _, starts, counts = np.unique(day, return_index=True, return_counts=True)
    groups = int(len(starts))
    mean = float(y.mean())
    if groups < 3:
        return {"样本": sample, "范围": scope, "关系": name, "时钟区间": label,
                "笔数": int(len(y)), "天数": groups, "均值bp": mean * 1e4, "下限bp": np.nan, "上限bp": np.nan}
    meat = 0.0
    for start, count in zip(starts, counts):
        meat += float(y[start:start + count].sum() - mean * count) ** 2
    se = np.sqrt(meat / len(y) ** 2 * (groups / (groups - 1)))
    crit = stats.t.ppf(0.975, groups - 1)
    return {"样本": sample, "范围": scope, "关系": name, "时钟区间": label, "笔数": int(len(y)), "天数": groups,
            "均值bp": mean * 1e4, "下限bp": (mean - crit * se) * 1e4, "上限bp": (mean + crit * se) * 1e4}


def attach_holding_gross(table, exit_hm, column):
    """按品种补上 09:17 开盘到指定收盘的多头收益。"""
    parts = []
    for symbol in sorted(set(table.symbol)):
        gross = holding_gross(symbol, "research", exit_hm)
        gross["trading_date"] = pd.to_datetime(gross.trading_date)
        parts.append(gross)
        print(symbol, int(gross.gross.notna().sum()), flush=True)
    held = pd.concat(parts, ignore_index=True)
    merged = table.merge(held.rename(columns={"gross": column}), on=["symbol", "trading_date"], how="left")
    if merged.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("持仓收益并回特征表后品种/日期重复")
    return merged


def load_research(return_col="gross_1130", exit_hm=None):
    """研究期、当年品种池、有夜盘，并且 09:16 时钟和指定持仓收益都在。"""
    table = pd.read_parquet(FEATURES / "research.parquet")
    table["trading_date"] = pd.to_datetime(table.trading_date)
    C.assert_no_holdout_dates(table.trading_date, "高低点与收益")
    table = table[table.trading_date.dt.year.between(2016, 2021)].copy()
    if return_col not in table.columns:
        if exit_hm is None:
            raise ValueError(f"特征表没有 {return_col}")
        table = attach_holding_gross(table, exit_hm, return_col)
    pools = load_universe()
    table["member"] = [symbol in pools.get(day.year, ()) for symbol, day in zip(table.symbol, table.trading_date)]
    unknown = set(table.symbol) - set(SECTOR_OF)
    if unknown:
        raise ValueError(f"未登记板块: {sorted(unknown)}")
    table["sector"] = table.symbol.map(SECTOR_OF)
    ready = (table.member & table.ret_night.notna() & table.hclock_td.notna() & table.lclock_td.notna()
             & table[return_col].notna())
    table = table.loc[ready].reset_index(drop=True)
    if table.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("特征表品种/日期重复")
    return table


def samples_of(table):
    return {"有夜盘": table}


def collect(table, return_col="gross_1130"):
    relations_rows, bin_rows, year_rows = [], [], []
    for sample, frame in samples_of(table).items():
        groups = [("全部", frame)] + [(name, frame[frame.sector == name]) for name in SECTORS]
        for scope, part in groups:
            for item in study_items(part, return_col):
                relations_rows.append(describe(sample, scope, item))
                if scope == "全部" and item["关系"] != "低点时钟-高点时钟 → 多头收益":
                    bin_rows.extend(bin_means(sample, scope, item))
            year_rows.extend(yearly_pooled_summary(part, sample, scope, return_col).to_dict("records"))
    return pd.DataFrame(relations_rows), pd.DataFrame(bin_rows), pd.DataFrame(year_rows)


def position_items(frame, return_col="gross_1130"):
    """每个有夜盘的交易日：高点、低点各自在夜盘开盘到 09:16 的位置，对持仓做多收益。"""
    day = frame.trading_date.to_numpy()
    gross = frame[return_col].to_numpy(dtype=float)
    return [
        {"关系": "高点", "day": day, "x": frame.hclock_td.to_numpy(dtype=float), "y": gross},
        {"关系": "低点", "day": day, "x": frame.lclock_td.to_numpy(dtype=float), "y": gross},
    ]


def study_items(frame, return_col="gross_1130"):
    """位置变量与原始多头收益的关系，不按偏离方向拆样。"""
    high, low = position_items(frame, return_col)
    day = frame.trading_date.to_numpy()
    gross = frame[return_col].to_numpy(dtype=float)
    return [
        {**high, "关系": "高点时钟 → 多头收益", "预期": "负"},
        {**low, "关系": "低点时钟 → 多头收益", "预期": "正"},
        {"关系": "低点时钟-高点时钟 → 多头收益", "预期": "正", "day": day,
         "x": low["x"] - high["x"], "y": gross, "range": (-1.0, 1.0)},
    ]


def yearly_pooled_summary(frame, sample, scope, return_col="gross_1130"):
    """每个板块、每年合并所有品种日后计算 Spearman。"""
    rows = []
    for year, annual in frame.groupby(frame.trading_date.dt.year, sort=True):
        for item in study_items(annual, return_col):
            link = association(item["day"], item["x"], item["y"])
            rows.append({"样本": sample, "范围": scope, "年份": int(year), "关系": item["关系"],
                         "Spearman": link["Spearman"], "Spearman双尾p": link["Spearman_p"],
                         "笔数": link["笔数"], "天数": link["天数"]})
    return pd.DataFrame(rows)


def segment_table(frame, scope="全部", return_col="gross_1130"):
    rows = []
    summaries = []
    items = position_items(frame, return_col)
    items.append(study_items(frame, return_col)[2])
    for item in items:
        if item["关系"] == "低点时钟-高点时钟 → 多头收益":
            rows.extend(difference_bin_means("有夜盘", scope, item))
        else:
            rows.extend(bin_means("有夜盘", scope, item))
        link = association(item["day"], item["x"], item["y"])
        summaries.append({"关系": item["关系"], **link, "时钟0到1的收益bp": link["斜率"] * 1e4})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def sector_table(frame, return_col="gross_1130"):
    rows = []
    for sector in SECTORS:
        part = frame[frame.sector == sector]
        if part.empty:
            continue
        for item in position_items(part, return_col):
            link = association(item["day"], item["x"], item["y"])
            rows.append({"板块": sector, "品种数": int(part.symbol.nunique()), "关系": item["关系"],
                         "Spearman": link["Spearman"], "Spearman双尾p": link["Spearman_p"],
                         "笔数": link["笔数"], "天数": link["天数"],
                         "时钟0到1的收益bp": link["斜率"] * 1e4, "斜率双尾p": link["斜率_p"]})
    return pd.DataFrame(rows)


def use_chinese_font():
    from matplotlib import font_manager
    have = {font.name for font in font_manager.fontManager.ttflist}
    chosen = [name for name in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "SimHei", "Noto Sans CJK SC")
              if name in have]
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def apply_style():
    use_chinese_font()
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E4E7EB",
        "axes.labelcolor": "#243B53",
        "axes.titlecolor": "#102A43",
        "xtick.color": "#52606D",
        "ytick.color": "#52606D",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#E4E7EB",
        "grid.linewidth": 0.7,
    })


def _p_text(p_value):
    if not np.isfinite(p_value):
        return "p —"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def plot_segments(segments, summary, path, scope="全部", hold_label="09:17 → 11:30"):
    """高点、低点位置及两者时钟差分档后的持仓做多收益。"""
    apply_style()
    names = ("高点", "低点", "低点时钟-高点时钟 → 多头收益")
    labels = {"高点": BIN_LABELS, "低点": BIN_LABELS, "低点时钟-高点时钟 → 多头收益": DIFF_BIN_LABELS}
    colors = {"高点": "#2F5D7C", "低点": "#C46B4A", "低点时钟-高点时钟 → 多头收益": "#1F7A4D"}
    titles = {"高点": "高点出现的位置", "低点": "低点出现的位置",
              "低点时钟-高点时钟 → 多头收益": "低点时钟 - 高点时钟"}
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.7), sharey=True)
    for ax, name in zip(axes, names):
        bin_labels = labels[name]
        part = segments[segments.关系 == name].set_index("时钟区间").loc[bin_labels]
        values = part.均值bp.to_numpy()
        low = part.下限bp.to_numpy()
        high = part.上限bp.to_numpy()
        positions = np.arange(len(bin_labels))
        ax.bar(positions, values, color=colors[name], width=0.66, zorder=2)
        ax.errorbar(positions, values, yerr=[values - low, high - values], fmt="none",
                    ecolor=colors[name], elinewidth=1.0, capsize=3.5, capthick=1.0, zorder=3)
        ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
        if name == names[2]:
            segment_names = ["差值最小", "偏小", "接近 0", "偏大", "差值最大"]
            ticks = [f"{segment}\n{label}\n{int(count):,}"
                     for segment, label, count in zip(segment_names, bin_labels, part.笔数)]
        else:
            ticks = [f"{label}\n{int(count):,}" for label, count in zip(SEGMENT_LABELS, part.笔数)]
        ax.set_xticks(positions, ticks)
        ax.grid(axis="y", zorder=0)
        ax.tick_params(length=0, labelsize=9)
        ax.set_axisbelow(True)
        link = summary[summary.关系 == name].iloc[0]
        ax.set_title(f"{titles[name]}\nSpearman {link.Spearman:+.3f}    {_p_text(link.Spearman_p)}",
                     loc="left", fontsize=12, color="#102A43", pad=8)
    axes[0].set_ylabel(f"{hold_label} 做多收益（bp）")
    scope_label = "全板块" if scope == "全部" else scope
    fig.suptitle(f"2016–2021  {scope_label}  每个交易日  {hold_label}", fontsize=15,
                 color="#102A43", x=0.07, ha="left", y=0.98)
    fig.text(0.07, 0.92,
             "竖线是按交易日聚类的 95% 区间。横轴数字是该段的品种日数量。位置从夜盘第一根到 09:16，含 09:16 这根已知 K 线。",
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.16, wspace=0.22)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def plot_sectors(sectors, path, hold_label="09:17 → 11:30"):
    """每个板块自己的 Spearman。两栏板块顺序相同。"""
    apply_style()
    order = [name for name in SECTORS if name in set(sectors.板块)]
    ranked = sectors.set_index(["关系", "板块"])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.8), sharex=True)
    titles = {"高点": "高点位置  →  做多收益", "低点": "低点位置  →  做多收益"}
    y = np.arange(len(order))[::-1]
    reach = max(abs(float(sectors.Spearman.min())), abs(float(sectors.Spearman.max())), 0.02)
    for ax, name in zip(axes, ("高点", "低点")):
        values = np.array([ranked.loc[(name, sector), "Spearman"] for sector in order], dtype=float)
        p_values = np.array([ranked.loc[(name, sector), "Spearman双尾p"] for sector in order], dtype=float)
        counts = np.array([ranked.loc[(name, sector), "笔数"] for sector in order], dtype=int)
        significant = np.isfinite(p_values) & (p_values < 0.05)
        colors = np.where(~significant, "#C5CED6", np.where(values >= 0, "#1F7A4D", "#B23A48"))
        ax.hlines(y, 0, np.nan_to_num(values), color="#9FB3C8", lw=1.6, zorder=1)
        ax.scatter(values, y, c=colors, s=72, zorder=2, linewidths=0)
        ax.axvline(0, color="#9AA5B1", lw=0.8, zorder=1)
        tick_labels = [f"{sector}    {value:+.3f}\n{count:,} 日    {_p_text(p_value)}"
                       for sector, value, count, p_value in zip(order, values, counts, p_values)]
        ax.set_yticks(y, tick_labels, fontsize=9)
        ax.set_xlabel("Spearman 相关")
        ax.set_xlim(-reach * 1.35, reach * 1.35)
        ax.set_ylim(-0.65, len(order) - 0.35)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=4)
        n_sig = int(significant.sum())
        ax.set_title(f"{titles[name]}\n{n_sig} / {len(order)} 个板块  p < 0.05",
                     loc="left", fontsize=12, color="#102A43", pad=8)
    fig.suptitle(f"2016–2021  分板块  {hold_label}", fontsize=15, color="#102A43", x=0.18, ha="left", y=0.98)
    fig.text(0.18, 0.915, "同一板块左右对齐。绿为显著正相关，红为显著负相关，灰为不显著。每个点用该板块研究期的全部交易日。",
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.18, right=0.98, top=0.78, bottom=0.12, wspace=0.42)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def plot_yearly(yearly, path, scope="全部", hold_label="09:17 → 11:30"):
    """年度 pooled Spearman；三条线都只使用极值位置和多头收益。"""
    apply_style()
    frame = yearly[(yearly.样本 == "有夜盘") & (yearly.范围 == scope)]
    years = sorted(frame.年份.unique())
    fig, ax = plt.subplots(figsize=(11.8, 5.7))
    series = [
        ("高点时钟 → 多头收益", "高点时钟 → 多头收益", "#2F5D7C"),
        ("低点时钟 → 多头收益", "低点时钟 → 多头收益", "#C46B4A"),
        ("低点时钟-高点时钟 → 多头收益", "低点时钟 - 高点时钟 → 多头收益", "#1F7A4D"),
    ]
    for relation, label, color in series:
        if relation not in set(frame.关系):
            continue
        part = frame[frame.关系 == relation].sort_values("年份")
        ax.plot(part.年份.to_numpy(dtype=int), part.Spearman, marker="o", markersize=5, linewidth=2,
                color=color, label=label)
    ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
    ax.set_xticks(years)
    if years:
        ax.set_xlim(years[0] - 0.2, years[-1] + 0.2)
    ax.set_ylabel("年度 pooled Spearman")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="best", fontsize=9)
    values = frame.Spearman.dropna().to_numpy()
    reach = max(float(np.abs(values).max()), 0.02) * 1.15
    ax.set_ylim(-reach, reach)
    title = "全部品种" if scope == "全部" else scope
    fig.suptitle(f"2016–2021  夜盘样本分年 pooled Spearman · {title}  {hold_label}", fontsize=15,
                 color="#102A43", x=0.07, ha="left", y=0.98)
    note = "每年合并所有品种日计算；不按 dev15 分组，也不反转收益。"
    if scope != "全部":
        note += " 板块图纵轴单独缩放，不宜比较不同板块的幅度。"
    fig.text(0.07, 0.92, note,
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.13)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


MIN_CROSS_SECTION = 5  # 一天在范围内至少这么多个品种才算当天的截面 IC


def daily_ic(frame, return_col="gross_1130", min_names=MIN_CROSS_SECTION):
    """每个交易日在品种之间算一次 Spearman：当天的时钟排名对当天的收益排名。
    剔除了“这一天整个市场一起涨跌”的成分，只回答“同一天里，时钟靠后的品种是否收益更高”。"""
    rows = []
    for item in study_items(frame, return_col):
        x = pd.Series(item["x"], dtype=float)
        y = pd.Series(item["y"], dtype=float)
        day = pd.Series(item["day"])
        ok = x.notna() & y.notna()
        part = pd.DataFrame({"day": day[ok].to_numpy(), "x": x[ok].to_numpy(), "y": y[ok].to_numpy()})
        count = part.groupby("day").size()
        part = part[part.day.map(count) >= min_names]
        if part.empty:
            continue
        rank = part.groupby("day")[["x", "y"]].rank()
        part = part.assign(rx=rank.x, ry=rank.y)
        ic = part.groupby("day").apply(lambda d: d.rx.corr(d.ry), include_groups=False).dropna()
        rows.append(pd.DataFrame({"关系": item["关系"], "trading_date": pd.to_datetime(ic.index), "IC": ic.to_numpy(),
                                  "品种数": count.reindex(ic.index).to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["关系", "trading_date", "IC", "品种数"])


def ic_summary(ic, scope, group_year=True):
    """逐年（和全期）的截面 IC 均值、ICIR、Newey-West t 与双尾 p。"""
    from tfcta.ic import timeseries_t
    rows = []
    keys = [("全期", ic)]
    if group_year:
        keys += [(int(y), g) for y, g in ic.groupby(ic.trading_date.dt.year)]
    for period, part in keys:
        for relation, g in part.groupby("关系", sort=False):
            series = g.sort_values("trading_date").IC
            test = timeseries_t(series)
            dof = max(int(test["n_periods"]) - 1, 1)
            p = float(2 * stats.t.sf(abs(test["t"]), dof)) if np.isfinite(test["t"]) else np.nan
            rows.append({"范围": scope, "年份": period, "关系": relation, "IC均值": float(series.mean()),
                         "IC标准差": float(series.std(ddof=1)), "ICIR": test["ic_ir"], "NW_t": test["t"], "双尾p": p,
                         "IC为正天数占比": float((series > 0).mean()), "天数": int(len(series)),
                         "平均品种数": float(g.品种数.mean())})
    return pd.DataFrame(rows)


def cross_section_tables(table, return_col="gross_1130"):
    """全部品种和各板块的逐日截面 IC 与逐年汇总。板块内每天至少 MIN_CROSS_SECTION 个品种。"""
    daily, summary = [], []
    scopes = [("全部", table)] + [(name, table[table.sector == name]) for name in SECTORS]
    for scope, part in scopes:
        if part.empty:
            continue
        ic = daily_ic(part, return_col)
        if ic.empty:
            continue
        daily.append(ic.assign(范围=scope))
        summary.append(ic_summary(ic, scope))
    return pd.concat(daily, ignore_index=True), pd.concat(summary, ignore_index=True)


IC_SERIES = [
    ("高点时钟 → 多头收益", "高点时钟 → 多头收益", "#2F5D7C"),
    ("低点时钟 → 多头收益", "低点时钟 → 多头收益", "#C46B4A"),
    ("低点时钟-高点时钟 → 多头收益", "低点时钟 - 高点时钟 → 多头收益", "#1F7A4D"),
]


def plot_cross_section_yearly(summary, path, scope="全部", hold_label="09:17 → 11:30"):
    """逐年截面 IC 均值，竖线是 Newey-West 95% 区间；实心点 p < 0.05，空心点不显著。"""
    apply_style()
    frame = summary[(summary.范围 == scope) & (summary.年份 != "全期")].copy()
    frame["年份"] = frame.年份.astype(int)
    whole = summary[(summary.范围 == scope) & (summary.年份 == "全期")].set_index("关系")
    years = sorted(frame.年份.unique())
    fig, ax = plt.subplots(figsize=(11.8, 5.9))
    offsets = {name: shift for (name, _, _), shift in zip(IC_SERIES, (-0.12, 0.0, 0.12))}
    for relation, label, color in IC_SERIES:
        part = frame[frame.关系 == relation].sort_values("年份")
        if part.empty:
            continue
        x = part.年份.to_numpy(dtype=float) + offsets[relation]
        mean = part.IC均值.to_numpy()
        half = np.where(np.isfinite(part.NW_t) & (part.NW_t != 0), np.abs(mean / part.NW_t) * 1.96, np.nan)
        ax.errorbar(x, mean, yerr=half, fmt="none", ecolor=color, elinewidth=1.0, capsize=3, alpha=0.7, zorder=2)
        sig = part.双尾p.to_numpy() < 0.05
        ax.plot(x, mean, color=color, lw=1.6, zorder=3,
                label=f"{label}   全期 IC {whole.loc[relation, 'IC均值']:+.3f}  t {whole.loc[relation, 'NW_t']:+.2f}")
        ax.scatter(x[sig], mean[sig], color=color, s=46, zorder=4)
        ax.scatter(x[~sig], mean[~sig], facecolor="white", edgecolor=color, s=46, linewidths=1.4, zorder=4)
    ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
    ax.set_xticks(years)
    if years:
        ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.set_ylabel("年内日截面 IC 均值")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="best", fontsize=9)
    title = "全部品种" if scope == "全部" else scope
    fig.suptitle(f"2016–2021  逐日截面 IC 分年 · {title}  {hold_label}", fontsize=15,
                 color="#102A43", x=0.07, ha="left", y=0.98)
    names = int(round(float(frame.平均品种数.mean()))) if len(frame) else 0
    fig.text(0.07, 0.92,
             f"每个交易日在品种之间算一次 Spearman（每天至少 {MIN_CROSS_SECTION} 个品种，平均约 {names} 个），再按年平均。"
             "竖线为 Newey-West 95% 区间；实心 p < 0.05，空心不显著。",
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.11)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def plot_cross_section_cumulative(daily, path, scope="全部", hold_label="09:17 → 11:30"):
    """逐日截面 IC 的累计和：斜率稳定向上（或向下）说明关系持续存在，走平或拐头说明那段时间失效。"""
    apply_style()
    frame = daily[daily.范围 == scope]
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    for relation, label, color in IC_SERIES:
        part = frame[frame.关系 == relation].sort_values("trading_date")
        if part.empty:
            continue
        ax.plot(part.trading_date, part.IC.cumsum(), color=color, lw=1.6, label=label)
    ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
    ax.set_ylabel("截面 IC 累计和")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    title = "全部品种" if scope == "全部" else scope
    fig.suptitle(f"2016–2021  截面 IC 累计 · {title}  {hold_label}", fontsize=15,
                 color="#102A43", x=0.07, ha="left", y=0.98)
    fig.text(0.07, 0.92, "线越陡关系越强；某段走平或反向，说明那段时间关系消失或反转。高点时钟预期向下，另两条预期向上。",
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.1)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def clear_old_plots(plot_dir):
    for name in OLD_PLOTS:
        path = plot_dir / name
        if path.exists():
            path.unlink()


EXITS = {
    "1130": {"hm": (11, 30), "column": "gross_1130", "label": "09:17 → 11:30", "out": RUNS / "clock_return"},
    "1500": {"hm": (15, 0), "column": "gross_1500", "label": "09:17 → 15:00", "out": RUNS / "clock_return_1500"},
}
BALANCE_SEED = 20261013
BALANCE_DRAWS = 200


def balance_up_down(table, return_col="gross_1130", seed=BALANCE_SEED):
    """涨跌各半重采样：去掉收益恰为 0 的品种日；在每个“年份 × 板块”里，把涨、跌中多的一边随机抽到和少的一边一样多。
    这样全样本、每一年、每个板块的涨跌笔数都相等，相关性不会来自样本里涨多或跌多。"""
    rng = np.random.default_rng(seed)
    sign = np.sign(table[return_col])
    frame = table[sign != 0]
    keep = []
    for _, cell in frame.groupby([frame.trading_date.dt.year, frame.sector], sort=True):
        up = cell.index[cell[return_col] > 0].to_numpy()
        down = cell.index[cell[return_col] < 0].to_numpy()
        n = min(len(up), len(down))
        keep += [rng.choice(up, n, replace=False), rng.choice(down, n, replace=False)]
    chosen = np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)
    return table.loc[chosen].reset_index(drop=True)


def balance_check(original, balanced, return_col="gross_1130"):
    """每个“年份 × 板块”重采样前后的涨、跌、平笔数。"""
    def counts(frame, tag):
        sign = np.sign(frame[return_col]).map({1.0: "涨", -1.0: "跌", 0.0: "平"})
        out = sign.groupby([frame.trading_date.dt.year.rename("年份"), frame.sector.rename("板块")]).value_counts()
        return out.unstack(fill_value=0).add_prefix(tag)
    return counts(original, "原样本").join(counts(balanced, "重采样后"), how="left").fillna(0).astype(int).reset_index()


def balance_draws(table, return_col="gross_1130", draws=BALANCE_DRAWS):
    """换 draws 个随机种子各重采样一次，看全样本 Spearman、聚类 p 和日截面 IC 的分布，确认结论不依赖某一次抽样。"""
    rows = []
    for k in range(draws):
        sample = balance_up_down(table, return_col, BALANCE_SEED + k)
        ic = daily_ic(sample, return_col)
        ic_mean = ic.groupby("关系").IC.mean()
        for item in study_items(sample, return_col):
            link = association(item["day"], item["x"], item["y"])
            rows.append({"抽样": k, "关系": item["关系"], "笔数": link["笔数"], "Spearman": link["Spearman"],
                         "Spearman双尾p": link["Spearman_p"], "日截面IC均值": ic_mean.get(item["关系"], np.nan)})
    return pd.DataFrame(rows)


def run_return(exit_name="1130", balance=False):
    spec = EXITS[exit_name]
    out = spec["out"]
    column, label = spec["column"], spec["label"]
    table = load_research(column, None if column == "gross_1130" else spec["hm"])
    if balance:
        out = out.with_name(out.name + "_balanced")
        out.mkdir(parents=True, exist_ok=True)
        draws = balance_draws(table, column)
        draws.to_csv(out / "balance_draws.csv", index=False, encoding="utf-8-sig")
        spread = draws.groupby("关系")[["Spearman", "Spearman双尾p", "日截面IC均值"]].quantile([.05, .5, .95]).unstack()
        print(f"{BALANCE_DRAWS} 次重采样的 5% / 50% / 95% 分位：")
        print(spread.round(4).to_string())
        balanced = balance_up_down(table, column)
        balance_check(table, balanced, column).to_csv(out / "balance_check.csv", index=False, encoding="utf-8-sig")
        table = balanced
        label = f"{label}（涨跌各半重采样）"
    result, bins, yearly = collect(table, column)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "relation.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(out / "bins.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")
    night = samples_of(table)["有夜盘"]
    segments, segment_summary = segment_table(night, return_col=column)
    sectors = sector_table(night, column)
    segments.to_csv(out / "segments.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(out / "position.csv", index=False, encoding="utf-8-sig")
    sectors.to_csv(out / "by_sector.csv", index=False, encoding="utf-8-sig")
    leftover = out / "by_symbol.csv"
    if leftover.exists():
        leftover.unlink()
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    clear_old_plots(plot_dir)
    plot_segments(segments, segment_summary, plot_dir / "segments.png", hold_label=label)
    plot_sectors(sectors, plot_dir / "sectors.png", label)
    plot_yearly(yearly, plot_dir / "yearly.png", hold_label=label)
    yearly_sector_dir = plot_dir / "sectors"
    yearly_sector_dir.mkdir(parents=True, exist_ok=True)
    available_scopes = set(yearly.范围.unique())
    for sector in SECTORS:
        if sector in available_scopes:
            plot_yearly(yearly, yearly_sector_dir / f"yearly_{sector}.png", scope=sector, hold_label=label)
    ic_daily, ic_yearly = cross_section_tables(night, column)
    ic_daily.to_csv(out / "ic_daily.csv", index=False, encoding="utf-8-sig")
    ic_yearly.to_csv(out / "ic_yearly.csv", index=False, encoding="utf-8-sig")
    plot_cross_section_yearly(ic_yearly, plot_dir / "ic_yearly.png", hold_label=label)
    plot_cross_section_cumulative(ic_daily, plot_dir / "ic_cumulative.png", hold_label=label)
    ic_sector_dir = plot_dir / "ic_sectors"
    ic_sector_dir.mkdir(parents=True, exist_ok=True)
    for sector in SECTORS:
        if sector in set(ic_yearly.范围):
            plot_cross_section_yearly(ic_yearly, ic_sector_dir / f"ic_yearly_{sector}.png", scope=sector, hold_label=label)
    show = result[result.范围 == "全部"][
        ["样本", "关系", "笔数", "Spearman", "Spearman双尾p", "时钟0到1的收益bp", "斜率双尾p", "晚减早bp", "晚减早p"]]
    (out / "summary.json").write_text(show.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(show.round(4).to_string(index=False))
    print("\n分板块（夜盘样本，Spearman 与双尾 p）：")
    sector = result[result.样本 == "有夜盘"]
    wide = sector.pivot(index="范围", columns="关系", values=["Spearman", "Spearman双尾p"])
    print(wide.round(4).to_string())
    print("结果：", out)

# --- 平均路径与个例 ---

SEED = 20261014
WIDE = 0.6      # 平均路径、正例占比用的 |低减高| 门槛，与 clock_return 的分档一致
STRONG = 0.8    # 个例只取路径更单边的日子
CASE_BAND = (0.5, 0.9)  # 个例的 |收益| 落在本类本年的这个分位区间：清楚但不是极端值
CASES_PER_YEAR = 2
PRE_POINTS = 21
PERIODS = {"2016–2017": (2016, 2017), "2018–2020": (2018, 2020), "2021": (2021, 2021)}
PERIOD_COLORS = {"2016–2017": "#2F5D7C", "2018–2020": "#C46B4A", "2021": "#1F7A4D"}
UP, DOWN = "#C0392B", "#1F7A4D"  # 国内习惯：红涨绿跌
CATEGORIES = {
    "下杀_正例": {"side": 1, "win": True, "title": "正例：夜盘一路下杀到 09:16，上午反弹"},
    "冲高_正例": {"side": -1, "win": True, "title": "正例：夜盘一路冲高到 09:16，上午回落"},
    "下杀_反例": {"side": 1, "win": False, "title": "反例：夜盘一路下杀到 09:16，上午继续跌"},
    "冲高_反例": {"side": -1, "win": False, "title": "反例：夜盘一路冲高到 09:16，上午继续涨"},
}


def session_minutes():
    """日盘 09:17 到 15:00 的分钟标签（K 线按结束时刻记）。"""
    spans = [((9, 17), (10, 15)), ((10, 31), (11, 30)), ((13, 31), (15, 0))]
    out = []
    for (h0, m0), (h1, m1) in spans:
        out += [(t // 60, t % 60) for t in range(h0 * 60 + m0, h1 * 60 + m1 + 1)]
    return out


DAY_GRID = session_minutes()
TICKS = ((9, 17), (10, 31), (11, 30), (14, 0), (15, 0))  # 11:30 和 13:31 相邻，只标一个


def pre_path_points(start, decision):
    """夜盘第一根到 09:16 按时钟等分取 PRE_POINTS 个点的位置，0 是第一根，最后一个是 09:16。"""
    return start + np.rint(np.linspace(0, 1, PRE_POINTS) * (decision - start)).astype(int)


def symbol_paths(symbol, wanted):
    """一个品种在 wanted 交易日上的路径和位置特征。价格都相对 09:16 收盘。
    复权价（closew/highw/loww）是加法复权，只能做差；比例一律除以原始价。"""
    frame = load_minutes(symbol, "research")
    raw = raw_price_path(frame)
    cw = frame.closew.to_numpy(float)
    hw, lw = frame.highw.to_numpy(float), frame.loww.to_numpy(float)
    idx = frame.index.asi8
    hour = frame.index.hour.to_numpy()
    night = (hour >= 20) | (hour <= 4)
    codes, days = day_codes_of(frame)
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    pos = {int(ts): i for i, ts in enumerate(idx)}
    rows, pre, post = [], [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        day = days[int(codes[s])]
        if day not in wanted or s == 0:
            continue
        d = pos.get(stamp(day, DECISION))
        day_bars = np.flatnonzero(~night[s:e]) + s
        if d is None or not s <= d < e or not day_bars.size or raw[d] <= 0:
            continue
        f = int(day_bars[0])
        high, low = hw[s:d + 1].max(), lw[s:d + 1].min()
        base = raw[d]
        rows.append({
            "symbol": symbol, "trading_date": day,
            "r_pre": (cw[d] - cw[s - 1]) / raw[s - 1],        # 昨收 → 09:16
            "r_am16": (cw[d] - cw[f - 1]) / raw[s - 1],       # 夜盘收盘 → 09:16
            "loc": (cw[d] - low) / (high - low) if high > low else np.nan,  # 09:16 在夜盘以来区间里的位置
            "rng": (high - low) / raw[s - 1],
        })
        line = (cw[pre_path_points(s, d)] - cw[d]) / base
        pre.append(np.r_[(cw[s - 1] - cw[d]) / base, line])  # 第一个点是昨收
        later = [pos.get(stamp(day, hm)) for hm in DAY_GRID]
        post.append(np.array([(cw[i] - cw[d]) / base if i is not None and d < i < e else np.nan for i in later]))
    return pd.DataFrame(rows), np.array(pre).reshape(-1, PRE_POINTS + 1), np.array(post).reshape(-1, len(DAY_GRID))


def build_paths(table):
    parts, pre, post = [], [], []
    for symbol in sorted(set(table.symbol)):
        wanted = set(table.loc[table.symbol == symbol, "trading_date"])
        rows, a, b = symbol_paths(symbol, wanted)
        parts.append(rows)
        pre.append(a)
        post.append(b)
        print(symbol, len(rows), flush=True)
    rows = pd.concat(parts, ignore_index=True)
    rows["row"] = np.arange(len(rows))
    merged = table.merge(rows, on=["symbol", "trading_date"], how="inner", validate="one_to_one")
    merged["diff"] = merged.lclock_td - merged.hclock_td
    merged["year"] = merged.trading_date.dt.year
    merged["period"] = merged.year.map(period_of)
    return merged, np.vstack(pre), np.vstack(post)


def period_of(year):
    return next(name for name, (a, b) in PERIODS.items() if a <= year <= b)


def label_cases(table, threshold=WIDE):
    """side=+1 为下杀（低减高 ≥ 门槛），−1 为冲高；signed 是按预期方向的收益，>0 为正例。收益为 0 的日子不算。"""
    side = np.where(table["diff"] >= threshold, 1, np.where(table["diff"] <= -threshold, -1, 0))
    out = table.assign(side=side, signed=side * table.gross_1130)
    return out[(out.side != 0) & (out.gross_1130 != 0)]


def pick_cases(table, seed=SEED, threshold=STRONG, band=CASE_BAND):
    """四类个例，每类每年最多两天：|低减高| ≥ threshold，|收益| 在本类本年的 band 分位之间，固定种子随机抽。"""
    rng = np.random.default_rng(seed)
    labelled = label_cases(table, threshold)
    chosen = []
    for name, spec in CATEGORIES.items():
        part = labelled[(labelled.side == spec["side"]) & ((labelled.signed > 0) == spec["win"])]
        for year, cell in part.groupby("year", sort=True):
            size = cell.gross_1130.abs()
            lo, hi = size.quantile(band[0]), size.quantile(band[1])
            pool = cell[(size >= lo) & (size <= hi)]
            if pool.empty:
                continue
            indices = rng.choice(len(pool), size=min(CASES_PER_YEAR, len(pool)), replace=False)
            chosen.append(pool.iloc[indices].assign(类别=name))
    return pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()


def hit_rate_table(table):
    """|低减高| ≥ WIDE 的日子里，各年、各板块的正例占比和按方向收益（按交易日聚类的 95% 区间）。"""
    labelled = label_cases(table)
    rows = []
    for (year, sector), part in [((y, "全部"), g) for y, g in labelled.groupby("year")] + \
            [((y, s), g) for (y, s), g in labelled.groupby(["year", "sector"])]:
        mean = bin_row("", "", "", "", part.trading_date.to_numpy(), part.signed.to_numpy())
        rows.append({"年份": int(year), "板块": sector, "笔数": len(part), "正例占比": float((part.signed > 0).mean()),
                     "下杀笔数": int((part.side == 1).sum()), "冲高笔数": int((part.side == -1).sum()),
                     "按方向收益bp": mean["均值bp"], "下限bp": mean["下限bp"], "上限bp": mean["上限bp"]})
    return pd.DataFrame(rows)


def split_table(table):
    """|低减高| ≥ WIDE 的日子按三种方式拆开看按方向收益：时期×方向；09:16 前 15 分钟是否仍在顺着极值方向走；
    夜盘收益是否和这段走势同向。dev15 < 0 在下杀日表示 09:16 仍在创新低（仍在延伸），> 0 表示已经回头。"""
    labelled = label_cases(table).dropna(subset=["dev15"])
    labelled = labelled.assign(
        时期=labelled.year.map(period_of),
        方向=np.where(labelled.side == 1, "下杀", "冲高"),
        到0916=np.where(-labelled.side * labelled.dev15 > 0, "仍在延伸", "已经回头"),
        夜盘=np.where(-labelled.side * labelled.ret_night > 0, "同向", "反向"),
        夜盘幅度=labelled.groupby(["symbol", "year"]).ret_night.transform(  # 品种内、年内按 |夜盘收益| 三分，避免变成品种波动差异
            lambda s: pd.qcut(s.abs().rank(method="first"), 3, labels=["小", "中", "大"])).astype(str))
    rows = []
    for split in (["时期", "方向"], ["到0916"], ["到0916", "year"], ["夜盘"], ["夜盘", "year"], ["夜盘幅度"], ["夜盘幅度", "year"], ["sector"], ["sector", "year"]):
        for key, part in labelled.groupby(split):
            key = key if isinstance(key, tuple) else (key,)
            mean = bin_row("", "", "", "", part.trading_date.to_numpy(), part.signed.to_numpy())
            rows.append({"拆分": "×".join(split), "组": " ".join(str(k) for k in key), "笔数": len(part),
                         "正例占比": float((part.signed > 0).mean()), "按方向收益bp": mean["均值bp"],
                         "下限bp": mean["下限bp"], "上限bp": mean["上限bp"]})
    return pd.DataFrame(rows)


def xs_rank(frame, column):
    """当天截面百分位排名减 0.5，去掉“这一天整体涨跌”的成分。"""
    return frame.groupby("trading_date")[column].rank(pct=True) - 0.5


def rank_regression(frame, columns, target="gross_1130"):
    """截面排名回归，按交易日聚类的标准误。返回每个变量的系数和 t。"""
    data = frame.dropna(subset=list(columns) + [target])
    y = xs_rank(data, target).to_numpy()
    x = np.column_stack([np.ones(len(data))] + [xs_rank(data, c).to_numpy() for c in columns])
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    score = pd.DataFrame(x * (y - x @ beta)[:, None]).groupby(pd.factorize(data.trading_date)[0]).sum().to_numpy()
    groups = len(score)
    cov = bread @ (score.T @ score) @ bread * groups / (groups - 1)
    se = np.sqrt(np.diag(cov))
    return {c: (float(beta[i + 1]), float(beta[i + 1] / se[i + 1])) for i, c in enumerate(columns)}


DECOMPOSE_MODELS = [
    ("低减高", ["diff"]),
    ("09:16 区间位置", ["loc"]),
    ("昨收→09:16", ["r_pre"]),
    ("夜盘 + 早盘前16分钟", ["ret_night", "r_am16"]),
    ("低减高 + 区间位置", ["diff", "loc"]),
    ("低减高 + 昨收→09:16", ["diff", "r_pre"]),
    ("全部", ["diff", "ret_night", "r_am16", "loc"]),
]
NAMES = {"diff": "低减高", "loc": "09:16 区间位置", "r_pre": "昨收→09:16", "ret_night": "夜盘收益",
         "r_am16": "夜盘收盘→09:16"}


def decompose(table):
    rows = []
    for model, columns in DECOMPOSE_MODELS:
        for column, (coef, t_stat) in rank_regression(table, columns).items():
            rows.append({"模型": model, "变量": NAMES[column], "系数": coef, "聚类t": t_stat})
    corr = table[list(NAMES)].rename(columns=NAMES).corr("spearman")
    return pd.DataFrame(rows), corr


def spread_ic(frame):
    ic = daily_ic(frame)
    return ic[ic.关系 == "低点时钟-高点时钟 → 多头收益"].set_index("trading_date").IC


def heterogeneity(daily):
    """各年 IC 均值是否相同：逆方差加权的 Cochran Q。"""
    g = daily.groupby(daily.index.year).agg(["mean", "std", "count"])
    weight = g["count"] / g["std"] ** 2
    pooled = (weight * g["mean"]).sum() / weight.sum()
    q = float((weight * (g["mean"] - pooled) ** 2).sum())
    return q, float(stats.chi2.sf(q, len(g) - 1))


def yearly_ic_table(table):
    full = spread_ic(table)
    years = table.groupby("symbol").year.nunique()
    panel = years[years == table.year.nunique()].index
    fixed = spread_ic(table[table.symbol.isin(panel)])
    rows = []
    for name, series in (("全部品种", full), (f"固定面板{len(panel)}个品种", fixed)):
        for year, part in series.groupby(series.index.year):
            rows.append({"样本": name, "年份": int(year), "IC均值": part.mean(),
                         "t": part.mean() / part.std() * np.sqrt(len(part)), "天数": len(part)})
        q, p = heterogeneity(series)
        rows.append({"样本": name, "年份": "Q检验", "IC均值": q, "t": p, "天数": len(series)})
    sector = (table.groupby(["sector", "year"])
              .apply(lambda d: stats.spearmanr(d["diff"], d.gross_1130).correlation, include_groups=False)
              .unstack())
    return pd.DataFrame(rows), sector, full, sorted(panel)


def state_table(table, daily):
    """事前状态：过去 20 个交易日（不含当天）的平均 |昨收→09:16| 和共振度 |均值|/均值|·|，三分组看当天 IC。"""
    day = table.groupby("trading_date").r_pre.agg(
        开盘移动幅度=lambda s: s.abs().mean(), 共振度=lambda s: abs(s.mean()) / s.abs().mean())
    day = day.join(daily.rename("IC"), how="inner")
    rows = []
    for column in ("开盘移动幅度", "共振度"):
        state = day[column].rolling(20, min_periods=15).mean().shift(1)
        for scheme, group in (("全期三分", pd.qcut(state, 3, labels=False)),
                              ("年内三分", state.groupby(state.index.year).transform(
                                  lambda s: pd.qcut(s, 3, labels=False)))):
            frame = day.assign(档=group).dropna(subset=["档"])
            for level, part in frame.groupby("档"):
                rows.append({"状态": column, "分组": scheme, "档": ["低", "中", "高"][int(level)], "年份": "全期",
                             "IC均值": part.IC.mean(), "t": part.IC.mean() / part.IC.std() * np.sqrt(len(part)),
                             "天数": len(part)})
                for year, cell in part.groupby(part.index.year):
                    rows.append({"状态": column, "分组": scheme, "档": ["低", "中", "高"][int(level)],
                                 "年份": int(year), "IC均值": cell.IC.mean(), "t": np.nan, "天数": len(cell)})
    monthly = daily.resample("ME").mean().dropna()
    return pd.DataFrame(rows), float(monthly.autocorr(1))


def plot_paths(table, pre, post, path):
    """两行：下杀日、冲高日。左栏是昨收和夜盘到 09:16（按时钟），右栏是 09:16 之后的日盘（按分钟）。"""
    apply_style()
    labelled = label_cases(table)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2), gridspec_kw={"width_ratios": [1, 2.3]}, sharey="row")
    x_pre = np.r_[-0.18, np.linspace(0, 1, PRE_POINTS)]
    x_day = np.arange(len(DAY_GRID))
    ticks = [i for i, hm in enumerate(DAY_GRID) if hm in TICKS]
    exit_at = DAY_GRID.index((11, 30))
    for row, (side, name) in enumerate(((1, "低减高 ≥ 0.6：夜盘一路下杀到 09:16"), (-1, "低减高 ≤ -0.6：夜盘一路冲高到 09:16"))):
        left, right = axes[row]
        for period, color in PERIOD_COLORS.items():
            part = labelled[(labelled.side == side) & (labelled.period == period)]
            rows = part.row.to_numpy()
            win = float((part.signed > 0).mean())
            label = f"{period}  {len(part):,} 日  正例 {win:.1%}"
            left.plot(x_pre, np.nanmean(pre[rows], axis=0) * 1e4, color=color, lw=2, label=label)
            right.plot(x_day, np.nanmean(post[rows], axis=0) * 1e4, color=color, lw=2, label=label)
        for ax in (left, right):
            ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
            ax.grid(axis="y", zorder=0)
            ax.tick_params(length=0, labelsize=9)
        left.set_xticks([-0.18, 0, 0.5, 1], ["昨收", "夜盘开", "时钟 0.5", "09:16"])
        right.set_xticks(ticks, [f"{DAY_GRID[i][0]:02d}:{DAY_GRID[i][1]:02d}" for i in ticks])
        right.axvspan(0, exit_at, color="#F0F4F8", zorder=0)
        right.text(exit_at / 2, 0.02, "持仓 09:17 → 11:30", transform=right.get_xaxis_transform(),
                   ha="center", fontsize=9, color="#7B8794")
        left.set_ylabel("相对 09:16 收盘（bp）")
        left.set_title(name, loc="left", fontsize=12, color="#102A43")
        right.legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle("2016–2021  平均路径：极值出得晚的日子，09:16 之后怎么走", fontsize=15, color="#102A43",
                 x=0.06, ha="left", y=0.985)
    fig.text(0.06, 0.94, "所有品种日等权平均，未扣成本；每个品种日以自己 09:16 收盘为 0。正例指 09:17→11:30 收益与预期方向一致"
             "（下杀后涨、冲高后跌）。", fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.07, wspace=0.05, hspace=0.3)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_day_paths(table, post, path):
    """只看 09:16 之后：按预期方向翻转后的平均收益（下杀日不变，冲高日取负），三个时期对比。"""
    apply_style()
    labelled = label_cases(table)
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    x_day = np.arange(len(DAY_GRID))
    ticks = [i for i, hm in enumerate(DAY_GRID) if hm in TICKS]
    exit_at = DAY_GRID.index((11, 30))
    ax.axvspan(0, exit_at, color="#F0F4F8", zorder=0)
    for period, color in PERIOD_COLORS.items():
        part = labelled[labelled.period == period]
        line = np.nanmean(post[part.row.to_numpy()] * part.side.to_numpy()[:, None], axis=0) * 1e4
        ax.plot(x_day, line, color=color, lw=2,
                label=f"{period}  {len(part):,} 日  11:30 {line[exit_at]:+.1f} bp  15:00 {line[-1]:+.1f} bp")
    ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
    ax.set_xticks(ticks, [f"{DAY_GRID[i][0]:02d}:{DAY_GRID[i][1]:02d}" for i in ticks])
    ax.set_ylabel("按预期方向的累计收益（bp）")
    ax.grid(axis="y", zorder=0)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle("2016–2021  |低减高| ≥ 0.6 的日子，09:16 之后按预期方向的平均收益", fontsize=14, color="#102A43",
                 x=0.07, ha="left", y=0.98)
    fig.text(0.07, 0.915, "下杀日做多、冲高日做空，以 09:16 收盘为 0，未扣成本。向上表示反转，向下表示延续。阴影是 09:17→11:30 持仓段。",
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.1)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_hit_rate(hits, path):
    apply_style()
    whole = hits[hits.板块 == "全部"].sort_values("年份")
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = whole.年份.to_numpy()
    left.bar(x, whole.正例占比 * 100 - 50, bottom=50, color="#2F5D7C", width=0.6, zorder=2)
    left.axhline(50, color="#9AA5B1", lw=0.8)
    for xi, value, n in zip(x, whole.正例占比, whole.笔数):
        left.text(xi, value * 100 + (0.15 if value >= 0.5 else -0.15), f"{value:.1%}\n{n:,} 日",
                  ha="center", va="bottom" if value >= 0.5 else "top", fontsize=8.5, color="#243B53")
    reach = max(abs(whole.正例占比 * 100 - 50).max() + 1.5, 3)
    left.set_ylim(50 - reach, 50 + reach)
    left.set_ylabel("正例占比（%）")
    left.set_title("正例占比", loc="left", fontsize=12, color="#102A43")
    mean = whole.按方向收益bp.to_numpy()
    right.bar(x, mean, color=np.where(mean >= 0, "#1F7A4D", "#B23A48"), width=0.6, zorder=2)
    right.errorbar(x, mean, yerr=[mean - whole.下限bp, whole.上限bp - mean], fmt="none", ecolor="#52606D",
                   elinewidth=1, capsize=3, zorder=3)
    right.axhline(0, color="#9AA5B1", lw=0.8)
    right.set_ylabel("按方向的 09:17→11:30 收益（bp）")
    right.set_title("按方向收益，竖线为按交易日聚类的 95% 区间", loc="left", fontsize=12, color="#102A43")
    for ax in (left, right):
        ax.set_xticks(x)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    fig.suptitle("2016–2021  |低减高| ≥ 0.6：下杀后做多、冲高后做空", fontsize=14, color="#102A43",
                 x=0.07, ha="left", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.1, wspace=0.25)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def candles(frame, start, end):
    """原始价 5 分钟 K 线，切点对齐使 09:16 恰好是一根的结束。原始高低 = 复权高低 + (原始收 − 复权收)。"""
    part = frame.iloc[start:end]
    raw = raw_price_path(part)
    shift = raw - part.closew.to_numpy(float)
    bars = pd.DataFrame({"open": part.open.to_numpy(float), "high": part.highw.to_numpy(float) + shift,
                         "low": part.loww.to_numpy(float) + shift, "close": raw,
                         "volume": part.volume.to_numpy(float)}, index=part.index)
    out = bars.resample("5min", closed="right", label="right", origin="epoch", offset="1min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna()


def draw_case(ax, volume_ax, frame, case):
    raw_close = raw_price_path(frame)
    hw, lw = frame.highw.to_numpy(float), frame.loww.to_numpy(float)
    shift = raw_close - frame.closew.to_numpy(float)
    codes, days = day_codes_of(frame)
    day = case.trading_date
    rows = np.flatnonzero(days[codes] == day)
    s, e = int(rows[0]), int(rows[-1]) + 1
    d = int(frame.index.get_loc(pd.Timestamp(stamp(day, DECISION))))
    bars = candles(frame, s, e)
    x = np.arange(len(bars))
    color = np.where(bars.close >= bars.open, UP, DOWN)
    ax.vlines(x, bars.low, bars.high, color=color, lw=0.7, zorder=2)
    body_low = np.minimum(bars.open, bars.close)
    body = np.maximum(np.abs(bars.close - bars.open), (bars.high - bars.low).max() * 0.004)
    ax.bar(x, body, bottom=body_low, width=0.7, color=color, linewidth=0, zorder=3)
    times = bars.index
    cut = int(np.searchsorted(times, frame.index[d]))       # 09:16 所在那根
    exit_at = int(np.searchsorted(times, pd.Timestamp(stamp(day, EXIT))))
    ax.axvspan(cut + 0.5, min(exit_at, len(x) - 1) + 0.5, color="#F0F4F8", zorder=0)
    ax.axvline(cut + 0.5, color="#52606D", lw=0.9, ls="--", zorder=1)
    volume_ax.axvspan(cut + 0.5, min(exit_at, len(x) - 1) + 0.5, color="#F0F4F8", zorder=0)
    volume_ax.axvline(cut + 0.5, color="#52606D", lw=0.9, ls="--", zorder=1)
    hi = s + int(np.argmax(hw[s:d + 1]))
    lo = s + int(np.argmin(lw[s:d + 1]))
    span = float(bars.high.max() - bars.low.min()) or 1.0
    for at, price, marker, offset in ((hi, hw[hi] + shift[hi], "v", 0.04), (lo, lw[lo] + shift[lo], "^", -0.04)):
        k = int(np.searchsorted(times, frame.index[at]))
        ax.scatter([k], [price + offset * span], marker=marker, s=46, color="#102A43", zorder=4)
    gaps = np.flatnonzero(np.diff(times.asi8) > 5 * MINUTE) + 1
    ticks = sorted(set([0, *gaps.tolist()]))
    volume_ax.bar(x, bars.volume, width=0.7, color=color, linewidth=0, zorder=2)
    ax.set_xticks(ticks)
    ax.tick_params(labelbottom=False)
    volume_ax.set_xticks(ticks, [times[i].strftime("%H:%M") for i in ticks], fontsize=7)
    ax.set_xlim(-1, len(x))
    volume_ax.set_xlim(-1, len(x))
    ax.tick_params(length=0, labelsize=7.5)
    volume_ax.tick_params(length=0, labelsize=7)
    ax.grid(axis="y", zorder=0)
    volume_ax.grid(axis="y", zorder=0)
    volume_ax.set_ylabel("量", fontsize=7, rotation=0, labelpad=8)
    volume_ax.set_ylim(bottom=0)
    ax.set_title(f"{case.year}  {case.symbol}（{case.sector}）{day:%Y-%m-%d}\n"
                 f"低减高 {case['diff']:+.2f}   夜盘 {case.ret_night * 1e4:+.0f}bp   "
                 f"dev15 {case.dev15 * 1e4:+.0f}bp   上午 {case.gross_1130 * 1e4:+.0f}bp",
                 loc="left", fontsize=9.5, color="#102A43")


def plot_cases(cases, path_of):
    apply_style()
    frames = {symbol: load_minutes(symbol, "research") for symbol in sorted(set(cases.symbol))}
    for name, spec in CATEGORIES.items():
        part = cases[cases.类别 == name].sort_values("year")
        if part.empty:
            continue
        fig, axes = plt.subplots(8, 3, figsize=(18, 15),
                                 gridspec_kw={"height_ratios": [3.2, 0.9] * 4})
        for index, (_, case) in enumerate(part.iterrows()):
            row, column = divmod(index, 3)
            draw_case(axes[row * 2, column], axes[row * 2 + 1, column], frames[case.symbol], case)
        for index in range(len(part), 12):
            row, column = divmod(index, 3)
            axes[row * 2, column].axis("off")
            axes[row * 2 + 1, column].axis("off")
        fig.suptitle(f"2016–2021  {spec['title']}", fontsize=15, color="#102A43", x=0.05, ha="left", y=0.985)
        fig.text(0.05, 0.945,
                 f"每类每年最多抽 {CASES_PER_YEAR} 天：|低减高| ≥ {STRONG}，上午收益位于本类本年 {CASE_BAND[0]:.0%}–{CASE_BAND[1]:.0%} 分位；固定种子。"
                 "原始价 5 分钟 K 线及成交量，红涨绿跌；▼ 夜盘以来高点，▲ 低点（到 09:16 为止）；虚线是 09:16，阴影是 09:17→11:30 持仓。",
                 fontsize=9, color="#7B8794")
        fig.subplots_adjust(left=0.05, right=0.99, top=0.87, bottom=0.05, hspace=0.6, wspace=0.18)
        fig.savefig(path_of(name), dpi=150, facecolor="white")
        plt.close(fig)


def run_cases():
    table = load_research()
    merged, pre, post = build_paths(table)
    if len(merged) != len(table):
        print(f"路径缺 {len(table) - len(merged)} 个品种日（09:16 或昨收缺失）")
    CASES_OUT.mkdir(parents=True, exist_ok=True)
    plots = CASES_OUT / "plots"
    plots.mkdir(exist_ok=True)

    hits = hit_rate_table(merged)
    hits.to_csv(CASES_OUT / "hit_rate.csv", index=False, encoding="utf-8-sig")
    splits = split_table(merged)
    splits.to_csv(CASES_OUT / "splits.csv", index=False, encoding="utf-8-sig")
    plot_paths(merged, pre, post, plots / "paths.png")
    plot_day_paths(merged, post, plots / "paths_day.png")
    plot_hit_rate(hits, plots / "hit_rate.png")
    cases = pick_cases(merged)
    cases[["类别", "year", "symbol", "sector", "trading_date", "diff", "ret_night", "dev15", "gross_1130", "loc"]].to_csv(
        CASES_OUT / "cases.csv", index=False, encoding="utf-8-sig")
    plot_cases(cases, lambda name: plots / f"cases_{name}.png")

    reg, corr = decompose(merged)
    reg.to_csv(CASES_OUT / "decompose.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(CASES_OUT / "decompose_corr.csv", encoding="utf-8-sig")
    yearly, sector, daily, panel = yearly_ic_table(merged)
    yearly.to_csv(CASES_OUT / "yearly_ic.csv", index=False, encoding="utf-8-sig")
    sector.to_csv(CASES_OUT / "sector_year_spearman.csv", encoding="utf-8-sig")
    states, ar1 = state_table(merged, daily)
    states.to_csv(CASES_OUT / "states.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(corr.round(3).to_string())
    print(reg.round(4).to_string(index=False))
    print(yearly.round(4).to_string(index=False))
    print("固定面板:", " ".join(panel))
    print(sector.round(3).to_string())
    print(states[states.年份 == "全期"].round(4).to_string(index=False))
    print(f"月度 IC 一阶自相关 {ar1:.3f}")
    print(hits[hits.板块 == "全部"].round(4).to_string(index=False))
    print(splits.round(4).to_string(index=False))

# --- 尾段冲击与路径效率 ---

FIVE = 5 * MINUTE
OFFSET = MINUTE
SIGN = {"正": 1, "负": -1}
TERCILE_LABELS = ("低", "中", "高")


def load_path_config(path=PATH_CONFIG):
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("declared_at", "decision", "variables", "sample") if key not in config]
    if missing:
        raise ValueError(f"配置缺字段: {missing}")
    if "2022" in config["sample"] and "不读 2022" not in config["sample"]:
        raise ValueError("样本口径必须写明不读 2022")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return config, digest


def five_minute_last(close, timestamps):
    """每个 5 分钟桶的最后一笔收盘，按时间排序。切法与个例 K 线一致。"""
    close = np.asarray(close, dtype=float)
    timestamps = np.asarray(timestamps)
    if len(close) == 0:
        return close
    bucket = (timestamps - OFFSET - 1) // FIVE + 1
    order = np.argsort(bucket, kind="mergesort")
    bucket = bucket[order]
    close = close[order]
    change = np.r_[True, bucket[1:] != bucket[:-1]]
    last = np.r_[np.flatnonzero(change)[1:] - 1, len(bucket) - 1]
    return close[last]


def path_efficiency_from_closes(closes):
    """|净变动| / 相邻变动绝对值之和。不足两根或分母为 0 时留空。"""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < 2 or not np.isfinite(closes).all():
        return np.nan
    steps = np.diff(closes)
    length = float(np.abs(steps).sum())
    if length <= 0:
        return np.nan
    return float(abs(closes[-1] - closes[0]) / length)


def tail_impulse(move, width):
    """尾段绝对幅度除以区间幅度。区间不是正数时留空。"""
    if not np.isfinite(move) or not np.isfinite(width) or width <= 0:
        return np.nan
    return float(abs(move) / width)


def still_extending(diff, dev15):
    """低减高为正的下杀日，dev15 < 0 表示 09:16 仍在创新低。"""
    diff = np.asarray(diff, dtype=float)
    dev15 = np.asarray(dev15, dtype=float)
    out = np.full(len(diff), np.nan)
    ok = np.isfinite(diff) & np.isfinite(dev15) & (diff != 0)
    out[ok] = (-np.sign(diff[ok]) * dev15[ok] > 0).astype(float)
    return out


def aligned_dev(diff, dev15):
    """−sign(低减高)×dev15。正值表示最后 15 分钟仍朝较晚的极值走。"""
    diff = np.asarray(diff, dtype=float)
    dev15 = np.asarray(dev15, dtype=float)
    out = np.full(len(diff), np.nan)
    ok = np.isfinite(diff) & np.isfinite(dev15) & (diff != 0)
    out[ok] = -np.sign(diff[ok]) * dev15[ok]
    return out


def assign_tercile(values):
    """一个品种、一年内部按秩三分。不足 3 个有效值则整组留空。"""
    values = pd.Series(values)
    out = pd.Series(np.nan, index=values.index, dtype=object)
    valid = values.dropna()
    if len(valid) < 3:
        return out
    out.loc[valid.index] = pd.qcut(valid.rank(method="first"), 3, labels=list(TERCILE_LABELS)).astype(str)
    return out


def centered_rank(frame, column):
    return frame.groupby("trading_date")[column].rank(pct=True) - 0.5


def clustered_coefficients(y, columns, day):
    """列已经是回归量。按交易日聚类，返回每个列名的系数和 t。"""
    y = np.asarray(y, dtype=float)
    day = np.asarray(day)
    names = list(columns)
    x = np.column_stack([np.ones(len(y))] + [np.asarray(columns[name], dtype=float) for name in names])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("回归量含非有限值")
    xtx = x.T @ x
    try:
        bread = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return {name: (np.nan, np.nan) for name in names}
    beta = bread @ x.T @ y
    score = pd.DataFrame(x * (y - x @ beta)[:, None]).groupby(pd.factorize(day)[0]).sum().to_numpy()
    groups = len(score)
    if groups < 3:
        return {name: (float(beta[i + 1]), np.nan) for i, name in enumerate(names)}
    cov = bread @ (score.T @ score) @ bread * groups / (groups - 1)
    se = np.sqrt(np.diag(cov))
    out = {}
    for i, name in enumerate(names):
        scale = se[i + 1]
        out[name] = (float(beta[i + 1]), float(beta[i + 1] / scale) if scale > 0 else np.nan)
    return out


def interaction_terms(frame, state, control):
    """截面居中排名，以及低减高×状态、低减高×控制。乘积不再排名。"""
    needed = ["gross_1130", "diff", state, control]
    data = frame.dropna(subset=needed).copy()
    data = data[data["diff"] != 0]
    diff_r = centered_rank(data, "diff")
    state_r = centered_rank(data, state)
    control_r = centered_rank(data, control)
    y = centered_rank(data, "gross_1130").to_numpy()
    columns = {
        "低减高": diff_r.to_numpy(),
        "状态": state_r.to_numpy(),
        "低减高×状态": (diff_r * state_r).to_numpy(),
        "控制": control_r.to_numpy(),
        "低减高×控制": (diff_r * control_r).to_numpy(),
    }
    return y, columns, data.trading_date.to_numpy(), len(data)


def _empty_ic():
    return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="trading_date"))


def daily_spearman(frame, x="diff", y="gross_1130", min_names=5):
    data = frame.loc[frame[x].notna() & frame[y].notna(), ["trading_date", x, y]].copy()
    if data.empty:
        return _empty_ic()
    sizes = data.groupby("trading_date")[x].transform("size")
    data = data.loc[sizes >= min_names]
    if data.empty:
        return _empty_ic()
    data["rx"] = data.groupby("trading_date")[x].rank()
    data["ry"] = data.groupby("trading_date")[y].rank()
    rows = []
    for day, group in data.groupby("trading_date", sort=True):
        if group.rx.nunique() < 2 or group.ry.nunique() < 2:
            continue
        rows.append((pd.Timestamp(day), float(group.rx.corr(group.ry))))
    if not rows:
        return _empty_ic()
    index, values = zip(*rows)
    return pd.Series(values, index=pd.DatetimeIndex(index, name="trading_date"), dtype=float)


def same_sign(high_minus_low, expected_sign):
    if not np.isfinite(high_minus_low) or high_minus_low == 0:
        return "不足"
    return "同号" if np.sign(high_minus_low) == expected_sign else "反号"


def judge(primary_t, robust_t, year_same, sector_same, expected_sign, decision):
    """主回归、稳健性、年份和板块四项都过，才算这个变量通过。"""
    threshold = decision["interaction_abs_t"]
    primary_ok = bool(np.isfinite(primary_t) and abs(primary_t) > threshold and np.sign(primary_t) == expected_sign)
    robust_ok = bool(np.isfinite(robust_t) and abs(robust_t) > threshold and np.sign(robust_t) == expected_sign)
    year_ok = int(year_same) >= decision["min_years_same_sign"]
    sector_ok = int(sector_same) >= decision["min_sectors_same_sign"]
    return {
        "主回归过线": primary_ok,
        "稳健性过线": robust_ok,
        "年份过线": year_ok,
        "板块过线": sector_ok,
        "通过": bool(primary_ok and robust_ok and year_ok and sector_ok),
    }


def session_rows(symbol, wanted):
    """一个品种在 wanted 交易日上的尾段冲击占比和路径效率。"""
    frame = load_minutes(symbol, "research")
    raw = raw_price_path(frame)
    cw = frame.closew.to_numpy(float)
    hw, lw = frame.highw.to_numpy(float), frame.loww.to_numpy(float)
    idx = frame.index.asi8
    hour = frame.index.hour.to_numpy()
    night = (hour >= 20) | (hour <= 4)
    codes, days = day_codes_of(frame)
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    pos = {int(ts): i for i, ts in enumerate(idx)}
    rows = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        day = days[int(codes[s])]
        if day not in wanted or s == 0:
            continue
        d = pos.get(stamp(day, DECISION))
        day_bars = np.flatnonzero(~night[s:e]) + s
        if d is None or not s <= d < e or not day_bars.size or raw[d] <= 0 or raw[s - 1] <= 0:
            continue
        f = int(day_bars[0])
        if f <= s:
            continue
        high, low = float(hw[s:d + 1].max()), float(lw[s:d + 1].min())
        width = (high - low) / raw[s - 1]
        move = (cw[d] - cw[f - 1]) / raw[s - 1]
        closes = five_minute_last(cw[s:d + 1], idx[s:d + 1])
        rows.append({
            "symbol": symbol, "trading_date": day,
            "impulse": tail_impulse(move, width),
            "efficiency": path_efficiency_from_closes(closes),
        })
    return rows


def build_path_features(table):
    rows = []
    for symbol in sorted(set(table.symbol)):
        wanted = set(table.loc[table.symbol == symbol, "trading_date"])
        part = session_rows(symbol, wanted)
        rows.extend(part)
        print(symbol, len(part), flush=True)
    features_ = pd.DataFrame(rows)
    merged = table.merge(features_, on=["symbol", "trading_date"], how="inner", validate="one_to_one")
    merged["diff"] = merged.lclock_td - merged.hclock_td
    merged["year"] = merged.trading_date.dt.year
    merged["extend"] = still_extending(merged["diff"], merged.dev15)
    merged["push"] = aligned_dev(merged["diff"], merged.dev15)
    return merged


def add_terciles(frame, variables):
    out = frame.copy()
    for spec in variables:
        column = spec["id"]
        parts = [assign_tercile(group[column]) for _, group in out.groupby(["symbol", "year"], sort=False)]
        out[column + "_bin"] = pd.concat(parts).sort_index()
    return out


def ic_rows(frame, spec, decision):
    """一个变量的低中高三档：全期、分年、分板块的逐日截面 IC。"""
    column = spec["id"]
    rows = []
    scopes = [("全部", frame)] + [(sector, frame[frame.sector == sector]) for sector in SECTORS]
    for scope, part in scopes:
        for label in TERCILE_LABELS:
            cell = part[part[column + "_bin"] == label]
            ic = daily_spearman(cell, min_names=decision["min_names"])
            test = ic_summary(
                pd.DataFrame({"关系": column, "trading_date": ic.index, "IC": ic.to_numpy(),
                              "品种数": np.ones(len(ic))}),
                scope, group_year=False) if len(ic) else pd.DataFrame()
            mean = float(ic.mean()) if len(ic) else np.nan
            t_stat = float(test.NW_t.iloc[0]) if len(test) else np.nan
            rows.append({"变量": spec["name"], "档": label, "范围": scope, "年份": "全期",
                         "IC均值": mean, "NW_t": t_stat, "天数": int(len(ic))})
            for year, values in ic.groupby(ic.index.year):
                rows.append({"变量": spec["name"], "档": label, "范围": scope, "年份": int(year),
                             "IC均值": float(values.mean()), "NW_t": np.nan, "天数": int(len(values))})
    return pd.DataFrame(rows)


def signed_rows(frame, spec):
    """|低减高| ≥ 0.6 时，各档的按方向收益。"""
    column = spec["id"]
    labelled = label_cases(frame.dropna(subset=[column + "_bin"]))
    rows = []
    groups = [("全部", "全期", labelled)]
    groups += [(str(year), "年份", part) for year, part in labelled.groupby(labelled.trading_date.dt.year)]
    groups += [(sector, "板块", part) for sector, part in labelled.groupby("sector")]
    for scope, kind, part in groups:
        for label in TERCILE_LABELS:
            cell = part[part[column + "_bin"] == label]
            mean = bin_row("", "", "", "", cell.trading_date.to_numpy(), cell.signed.to_numpy())
            rows.append({"变量": spec["name"], "档": label, "口径": kind, "范围": scope,
                         "笔数": mean["笔数"], "天数": mean["天数"], "按方向收益bp": mean["均值bp"],
                         "下限bp": mean["下限bp"], "上限bp": mean["上限bp"]})
    return pd.DataFrame(rows)


def fit_record(spec, model, control, fitted, interaction, n):
    coef, t_stat = fitted[interaction]
    return {"模型": model, "变量": spec["name"], "状态列": spec["id"], "控制列": control,
            "交互系数": coef, "交互t": t_stat, "低减高系数": fitted["低减高"][0], "低减高t": fitted["低减高"][1],
            "笔数": n, "预期": spec["expected_interaction"]}


def fit_one(frame, spec, control, model):
    y, columns, day, n = interaction_terms(frame, spec["id"], control)
    if model == "对照":
        columns = {name: columns[name] for name in ("低减高", "状态", "低减高×状态")}
    fitted = clustered_coefficients(y, columns, day)
    return fit_record(spec, model, "" if model == "对照" else control, fitted, "低减高×状态", n)


def fit_path_joint(frame):
    needed = ["gross_1130", "diff", "impulse", "efficiency", "extend"]
    data = frame.dropna(subset=needed).copy()
    data = data[data["diff"] != 0]
    diff_r = centered_rank(data, "diff")
    impulse_r = centered_rank(data, "impulse")
    efficiency_r = centered_rank(data, "efficiency")
    extend_r = centered_rank(data, "extend")
    y = centered_rank(data, "gross_1130").to_numpy()
    columns = {
        "低减高": diff_r.to_numpy(),
        "尾段冲击": impulse_r.to_numpy(),
        "路径效率": efficiency_r.to_numpy(),
        "低减高×尾段冲击": (diff_r * impulse_r).to_numpy(),
        "低减高×路径效率": (diff_r * efficiency_r).to_numpy(),
        "仍在延伸": extend_r.to_numpy(),
        "低减高×仍在延伸": (diff_r * extend_r).to_numpy(),
    }
    fitted = clustered_coefficients(y, columns, data.trading_date.to_numpy())
    rows = []
    for name in ("低减高×尾段冲击", "低减高×路径效率"):
        coef, t_stat = fitted[name]
        rows.append({"模型": "联合附录", "变量": name, "状态列": "", "控制列": "extend",
                     "交互系数": coef, "交互t": t_stat, "低减高系数": fitted["低减高"][0],
                     "低减高t": fitted["低减高"][1], "笔数": len(data), "预期": ""})
    return pd.DataFrame(rows)


def sign_rows(ic, signed, spec, decision):
    """高档减低档。截面 IC 的同号个数用于判定；按方向收益只报告。"""
    expected = SIGN[spec["expected_interaction"]]
    rows = []
    ic_var = ic[ic.变量 == spec["name"]]
    for scope in ["全部", *SECTORS]:
        for year in (["全期"] if scope != "全部" else ["全期", *range(2016, 2022)]):
            high = ic_var[(ic_var.范围 == scope) & (ic_var.年份 == year) & (ic_var.档 == "高")]
            low = ic_var[(ic_var.范围 == scope) & (ic_var.年份 == year) & (ic_var.档 == "低")]
            if high.empty or low.empty:
                continue
            gap = float(high.IC均值.iloc[0] - low.IC均值.iloc[0])
            days_high, days_low = int(high.天数.iloc[0]), int(low.天数.iloc[0])
            minimum = decision["min_days_per_year_tail"] if scope == "全部" and year != "全期" else (
                decision["min_days_per_sector_tail"] if year == "全期" and scope != "全部" else 1)
            status = same_sign(gap, expected) if min(days_high, days_low) >= minimum else "不足"
            if year == "全期" and scope == "全部":
                status = same_sign(gap, expected)
            rows.append({"变量": spec["name"], "口径": "截面IC", "范围": scope, "年份": year,
                         "高减低": gap, "高档天数": days_high, "低档天数": days_low, "状态": status})
    signed_var = signed[signed.变量 == spec["name"]]
    for kind, minimum in (("年份", decision["min_days_per_year_tail"]), ("板块", decision["min_days_per_sector_tail"])):
        part = signed_var[signed_var.口径 == kind]
        for scope, cell in part.groupby("范围"):
            high = cell[cell.档 == "高"]
            low = cell[cell.档 == "低"]
            if high.empty or low.empty:
                continue
            gap = float(high.按方向收益bp.iloc[0] - low.按方向收益bp.iloc[0])
            days_high, days_low = int(high.天数.iloc[0]), int(low.天数.iloc[0])
            status = same_sign(gap, expected) if min(days_high, days_low) >= minimum else "不足"
            rows.append({"变量": spec["name"], "口径": "按方向收益", "范围": scope, "年份": "",
                         "高减低": gap, "高档天数": days_high, "低档天数": days_low, "状态": status})
    return pd.DataFrame(rows)


def count_same(signs, spec, kind, key):
    part = signs[(signs.变量 == spec["name"]) & (signs.口径 == "截面IC")]
    if key == "年份":
        part = part[(part.范围 == "全部") & (part.年份 != "全期")]
    else:
        part = part[(part.年份 == "全期") & (part.范围 != "全部")]
    return int((part.状态 == "同号").sum()), int((part.状态 != "不足").sum())


def run_path():
    config, digest = load_path_config()
    print(f"声明日期 {config['declared_at']}  配置 sha256 {digest[:12]}", flush=True)
    table = load_research()
    merged = add_terciles(build_path_features(table), config["variables"])
    if len(merged) != len(table):
        print(f"路径缺 {len(table) - len(merged)} 个品种日")
    PATH_OUT.mkdir(parents=True, exist_ok=True)
    keep = ["symbol", "trading_date", "sector", "year", "diff", "gross_1130", "dev15",
            "impulse", "efficiency", "extend", "push", "impulse_bin", "efficiency_bin"]
    merged[keep].to_parquet(PATH_OUT / "features.parquet", index=False)

    decision = config["decision"]
    ic = pd.concat([ic_rows(merged, spec, decision) for spec in config["variables"]], ignore_index=True)
    signed = pd.concat([signed_rows(merged, spec) for spec in config["variables"]], ignore_index=True)
    fits = []
    for spec in config["variables"]:
        fits.append(fit_one(merged, spec, "extend", "对照"))
        fits.append(fit_one(merged, spec, "extend", "主回归"))
        fits.append(fit_one(merged, spec, "push", "稳健性"))
    fit_table = pd.concat([pd.DataFrame(fits), fit_path_joint(merged)], ignore_index=True)
    signs = pd.concat([sign_rows(ic, signed, spec, decision) for spec in config["variables"]], ignore_index=True)

    verdicts = []
    for spec in config["variables"]:
        primary = fit_table[(fit_table.模型 == "主回归") & (fit_table.变量 == spec["name"])].iloc[0]
        robust = fit_table[(fit_table.模型 == "稳健性") & (fit_table.变量 == spec["name"])].iloc[0]
        year_same, year_known = count_same(signs, spec, "截面IC", "年份")
        sector_same, sector_known = count_same(signs, spec, "截面IC", "板块")
        result = judge(primary.交互t, robust.交互t, year_same, sector_same,
                       SIGN[spec["expected_interaction"]], decision)
        verdicts.append({"变量": spec["name"], "预期": spec["expected_interaction"],
                         "主回归t": primary.交互t, "稳健性t": robust.交互t,
                         "年份同号": year_same, "年份可判": year_known,
                         "板块同号": sector_same, "板块可判": sector_known,
                         **result, "声明日期": config["declared_at"], "配置sha256": digest})
    verdict = pd.DataFrame(verdicts)
    ic.to_csv(PATH_OUT / "bins_ic.csv", index=False, encoding="utf-8-sig")
    signed.to_csv(PATH_OUT / "bins_signed.csv", index=False, encoding="utf-8-sig")
    fit_table.to_csv(PATH_OUT / "interaction.csv", index=False, encoding="utf-8-sig")
    signs.to_csv(PATH_OUT / "signs.csv", index=False, encoding="utf-8-sig")
    verdict.to_csv(PATH_OUT / "verdict.csv", index=False, encoding="utf-8-sig")
    corr = merged[["impulse", "efficiency", "extend", "push", "diff"]].corr("spearman")
    corr.to_csv(PATH_OUT / "corr.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(corr.round(3).to_string())
    print(fit_table.round(4).to_string(index=False))
    print(verdict.to_string(index=False))
    print(signs[signs.口径 == "截面IC"].round(4).to_string(index=False))
    print(signed[signed.口径.isin(["全期", "年份"])].round(4).to_string(index=False))

# --- 开盘量能 ---

WINDOW_START, WINDOW_END = (9, 1), DECISION
BASELINE_WINDOW, BASELINE_MIN = 20, 15


def volume_surprise(current, baseline):
    if not np.isfinite(current) or not np.isfinite(baseline) or baseline <= 0:
        return np.nan
    return float(current / baseline)


def impact_per_volume(open_move, surprise):
    if not np.isfinite(open_move) or not np.isfinite(surprise) or surprise <= 0:
        return np.nan
    return float(abs(open_move) / np.sqrt(surprise))


def volume_peak_clock(volumes):
    values = np.asarray(volumes, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < 2:
        return np.nan
    peak = valid[np.argmax(values[valid])]
    return float(peak / (len(values) - 1))


def load_flow_config(config_path=FLOW_CONFIG):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"declared_at", "sample", "decision", "models", "variables"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"配置缺字段: {missing}")
    if "不读 2022" not in config["sample"]:
        raise ValueError("配置必须明确不读 2022")
    if len(config["variables"]) != 3:
        raise ValueError("本轮只允许三个预注册量能候选")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return config, digest


def symbol_flow_rows(symbol, wanted):
    frame = load_minutes(symbol, "research")
    C.assert_no_holdout_dates(frame.trading_date, what=f"{symbol} 开盘量能")
    raw = raw_price_path(frame)
    raw_open = raw_price_path(pd.DataFrame({"close": frame.open.to_numpy(float)}, index=frame.index))
    volume = frame.volume.to_numpy(float)
    codes, days = day_codes_of(frame)
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    positions = {int(ts): i for i, ts in enumerate(frame.index.asi8)}
    window_start_offset = WINDOW_START[0] * 60 + WINDOW_START[1]
    window_end_offset = WINDOW_END[0] * 60 + WINDOW_END[1]
    rows = []

    for start, end in zip(bounds[:-1], bounds[1:]):
        day = pd.Timestamp(days[int(codes[start])]).normalize()
        if day.year > C.RESEARCH_END.year:
            continue
        decision = positions.get(stamp(day, WINDOW_END))
        first_window = positions.get(stamp(day, WINDOW_START))
        if decision is None or first_window is None or not start <= first_window <= decision < end:
            continue
        window = np.arange(first_window, decision + 1)
        times = frame.index[window]
        minute_offsets = times.hour.to_numpy() * 60 + times.minute.to_numpy()
        if not np.array_equal(minute_offsets, np.arange(window_start_offset, window_end_offset + 1)):
            continue
        if not np.isfinite(raw_open[first_window]) or raw_open[first_window] <= 0:
            continue
        window_volume = volume[window]
        rows.append({
            "symbol": symbol,
            "trading_date": day,
            "open_volume": float(np.nansum(window_volume)),
            "open_move": float(raw[decision] / raw_open[first_window] - 1.0),
            "volume_peak_clock": volume_peak_clock(window_volume),
        })

    if not rows:
        return pd.DataFrame(columns=["symbol", "trading_date", *[s["id"] for s in load_flow_config()[0]["variables"]]])
    daily = pd.DataFrame(rows).sort_values("trading_date").drop_duplicates("trading_date", keep="last")
    baseline = daily.open_volume.rolling(BASELINE_WINDOW, min_periods=BASELINE_MIN).median().shift(1)
    daily["open_volume_surprise"] = [volume_surprise(value, prior)
                                     for value, prior in zip(daily.open_volume, baseline)]
    daily["impact_per_volume"] = [impact_per_volume(move, surprise)
                                  for move, surprise in zip(daily.open_move, daily.open_volume_surprise)]
    return daily[daily.trading_date.isin(wanted)][
        ["symbol", "trading_date", "open_volume_surprise", "impact_per_volume", "volume_peak_clock"]]


def build_flow_features(table):
    rows = []
    for symbol in sorted(set(table.symbol)):
        wanted = set(pd.to_datetime(table.loc[table.symbol == symbol, "trading_date"]).dt.normalize())
        frame = symbol_flow_rows(symbol, wanted)
        rows.append(frame)
        print(symbol, len(frame), flush=True)
    flow = pd.concat(rows, ignore_index=True)
    merged = table.merge(flow, on=["symbol", "trading_date"], how="left", validate="one_to_one")
    merged["diff"] = merged.lclock_td - merged.hclock_td
    merged["year"] = merged.trading_date.dt.year
    merged["extend"] = still_extending(merged["diff"], merged.dev15)
    merged["push"] = aligned_dev(merged["diff"], merged.dev15)
    return merged


def fit_flow_joint(frame, specs):
    needed = ["gross_1130", "diff", "extend", *[spec["id"] for spec in specs]]
    data = frame.dropna(subset=needed).copy()
    data = data[data["diff"] != 0]
    diff_rank = centered_rank(data, "diff")
    extend_rank = centered_rank(data, "extend")
    y = centered_rank(data, "gross_1130").to_numpy()
    columns = {"低减高": diff_rank.to_numpy(), "仍在延伸": extend_rank.to_numpy(),
               "低减高×仍在延伸": (diff_rank * extend_rank).to_numpy()}
    for spec in specs:
        state_rank = centered_rank(data, spec["id"])
        columns[spec["name"]] = state_rank.to_numpy()
        columns[f"低减高×{spec['name']}"] = (diff_rank * state_rank).to_numpy()
    fitted = clustered_coefficients(y, columns, data.trading_date.to_numpy())
    return pd.DataFrame([
        {"模型": "联合附录", "变量": f"低减高×{spec['name']}", "状态列": spec["id"],
         "控制列": "其他状态+extend", "交互系数": fitted[f"低减高×{spec['name']}"][0],
         "交互t": fitted[f"低减高×{spec['name']}"][1], "笔数": len(data),
         "预期": spec["expected_interaction"]}
        for spec in specs
    ])


def run_flow():
    config, digest = load_flow_config()
    print(f"声明日期 {config['declared_at']}  配置 sha256 {digest[:12]}", flush=True)
    merged = build_flow_features(load_research())
    merged = add_terciles(merged, config["variables"])
    FLOW_OUT.mkdir(parents=True, exist_ok=True)
    decision = config["decision"]
    ic = pd.concat([ic_rows(merged, spec, decision) for spec in config["variables"]], ignore_index=True)
    signed = pd.concat([signed_rows(merged, spec) for spec in config["variables"]], ignore_index=True)
    fits = []
    for spec in config["variables"]:
        fits.extend(fit_one(merged, spec, control, model)
                    for model, control in (("对照", "extend"), ("主回归", "extend"), ("稳健性", "push")))
    fit_table = pd.concat([pd.DataFrame(fits), fit_flow_joint(merged, config["variables"])], ignore_index=True)
    signs = pd.concat([sign_rows(ic, signed, spec, decision) for spec in config["variables"]], ignore_index=True)

    verdicts = []
    for spec in config["variables"]:
        primary = fit_table[(fit_table.模型 == "主回归") & (fit_table.变量 == spec["name"])].iloc[0]
        robust = fit_table[(fit_table.模型 == "稳健性") & (fit_table.变量 == spec["name"])].iloc[0]
        year_same, year_known = count_same(signs, spec, "截面IC", "年份")
        sector_same, sector_known = count_same(signs, spec, "截面IC", "板块")
        result = judge(primary.交互t, robust.交互t, year_same, sector_same,
                            SIGN[spec["expected_interaction"]], decision)
        verdicts.append({"变量": spec["name"], "预期": spec["expected_interaction"],
                         "主回归t": primary.交互t, "稳健性t": robust.交互t,
                         "年份同号": year_same, "年份可判": year_known,
                         "板块同号": sector_same, "板块可判": sector_known,
                         **result, "声明日期": config["declared_at"], "配置sha256": digest})
    verdict = pd.DataFrame(verdicts)
    merged.to_parquet(FLOW_OUT / "features.parquet", index=False)
    ic.to_csv(FLOW_OUT / "bins_ic.csv", index=False, encoding="utf-8-sig")
    signed.to_csv(FLOW_OUT / "bins_signed.csv", index=False, encoding="utf-8-sig")
    fit_table.to_csv(FLOW_OUT / "interaction.csv", index=False, encoding="utf-8-sig")
    signs.to_csv(FLOW_OUT / "signs.csv", index=False, encoding="utf-8-sig")
    verdict.to_csv(FLOW_OUT / "verdict.csv", index=False, encoding="utf-8-sig")
    print(fit_table.round(4).to_string(index=False))
    print(verdict.to_string(index=False))

# --- 期限结构、仓单、持仓量 ---

EXTERNAL = C.COMMODITY_DATA / "external_rqdata"
OI_DIR = C.RESEARCH_DIR / "open_interest"
Z_WINDOW, Z_MIN = 250, 60
WARRANT_LAG, OI_BASELINE, OI_MIN = 20, 20, 10
CARRY_LAG, FILL_LIMIT = 1, 5


def load_state_config(path=STATE_CONFIG):
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("declared_at", "decision", "variables", "sample", "sign_convention") if key not in config]
    if missing:
        raise ValueError(f"配置缺字段: {missing}")
    if "不读 2022" not in config["sample"]:
        raise ValueError("样本口径必须写明不读 2022")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return config, digest


def zscore_known(series, window=Z_WINDOW, min_periods=Z_MIN):
    """序列里的值本身必须已经是决策前可知的。z 用含当期的滚动窗，不使用更晚的点。"""
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=min_periods).mean()
    scale = values.rolling(window, min_periods=min_periods).std()
    return (values - mean) / scale.where(scale > 0)


def warrant_change(level, periods=WARRANT_LAG):
    """20 个已公布仓单记录的变化率。当期或基期不是正数时留空。"""
    level = pd.to_numeric(level, errors="coerce")
    base = level.shift(periods)
    change = level / base - 1
    return change.where((level > 0) & (base > 0))


def move_sign(ret):
    """涨 +1，跌 −1，持平或缺失留空。"""
    values = pd.to_numeric(ret, errors="coerce")
    sign = np.sign(values)
    return pd.Series(sign, index=getattr(values, "index", None)).where(values != 0)


def curve_alignment(carry_z, move):
    """正：上涨且 back，或下跌且 contango。"""
    return pd.Series(carry_z, index=getattr(move, "index", None)).mul(move).where(move != 0)


def stock_alignment(stock_z, move):
    """正：去库且上涨，或累库且下跌。"""
    return pd.Series(stock_z, index=getattr(move, "index", None)).mul(-1).mul(move).where(move != 0)


def oi_impulse(change, baseline, same_contract):
    """持仓变化除以过去的持仓水平。换月或基线不是正数时留空。"""
    change = np.asarray(change, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    same = np.asarray(same_contract, dtype=bool)
    out = np.full(len(change), np.nan)
    ok = same & np.isfinite(change) & np.isfinite(baseline) & (baseline > 0)
    out[ok] = change[ok] / baseline[ok]
    return out


def _research_series(series):
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.loc[series.index < pd.Timestamp("2022-01-01")]


def carry_known(symbol, calendar):
    path = EXTERNAL / "roll_yield" / f"{symbol}_main_sub.pkl"
    if not path.exists():
        return pd.Series(np.nan, index=calendar)
    raw = _research_series(_series(pd.read_pickle(path), "annualized_yield"))
    return align_asof(raw, calendar, lag=CARRY_LAG, fill_limit=FILL_LIMIT)


def stock_known(symbol, calendar):
    path = EXTERNAL / "warehouse" / f"{symbol}.pkl"
    if not path.exists():
        return pd.Series(np.nan, index=calendar)
    frame = pd.read_pickle(path)
    dates = pd.to_datetime(frame.index.get_level_values("date").astype(str))
    level = pd.Series(pd.to_numeric(frame["on_warrant"], errors="coerce").to_numpy(), index=dates)
    change = warrant_change(_research_series(level))
    return align_asof(change, calendar, lag=CARRY_LAG, fill_limit=FILL_LIMIT)


def iter_day_bounds(frame):
    idx = frame.index.asi8
    hour = frame.index.hour.to_numpy()
    night = (hour >= 20) | (hour <= 4)
    codes, days = day_codes_of(frame)
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    pos = {int(ts): i for i, ts in enumerate(idx)}
    for s, e in zip(bounds[:-1], bounds[1:]):
        day = pd.Timestamp(days[int(codes[s])]).normalize()
        if day.year > 2021 or s == 0:
            continue
        d = pos.get(stamp(day, DECISION))
        day_bars = np.flatnonzero(~night[s:e]) + s
        if d is None or not s <= d < e or not day_bars.size:
            continue
        yield day, s, e, d, int(day_bars[0])


def price_moves(symbol, wanted):
    frame = load_minutes(symbol, "research")
    raw = raw_price_path(frame)
    cw = frame.closew.to_numpy(float)
    rows = []
    for day, s, e, d, f in iter_day_bounds(frame):
        if day not in wanted or f <= s or raw[d] <= 0 or raw[s - 1] <= 0:
            continue
        rows.append({"symbol": symbol, "trading_date": day, "r_pre": (cw[d] - cw[s - 1]) / raw[s - 1]})
    return pd.DataFrame(rows)


def oi_states(symbol):
    if not (OI_DIR / f"{symbol}.parquet").exists():
        return pd.DataFrame(columns=["symbol", "trading_date", "oi_state"])
    frame = load_shard(symbol, directory=OI_DIR, columns=["open_interest", "dominant_id", "trading_date"])
    frame = frame.loc[pd.to_datetime(frame["trading_date"]).dt.year <= 2021]
    oi = frame.open_interest.to_numpy(float)
    contract = frame.dominant_id.astype(str).to_numpy()
    rows = []
    for day, s, e, d, f in iter_day_bounds(frame):
        if f <= s or not np.isfinite(oi[d]) or not np.isfinite(oi[s]):
            continue
        rows.append({
            "trading_date": day,
            "oi_0916": float(oi[d]),
            "change": float(oi[d] - oi[s]),
            "same_contract": contract[s] == contract[d],
        })
    if not rows:
        return pd.DataFrame(columns=["symbol", "trading_date", "oi_state"])
    daily = pd.DataFrame(rows).drop_duplicates("trading_date", keep="last").set_index("trading_date").sort_index()
    baseline = daily.oi_0916.rolling(OI_BASELINE, min_periods=OI_MIN).mean().shift(1)
    daily["oi_state"] = oi_impulse(daily.change, baseline, daily.same_contract)
    out = daily[["oi_state"]].reset_index()
    out.insert(0, "symbol", symbol)
    return out


def symbol_states(symbol, wanted, calendar):
    carry_z = zscore_known(carry_known(symbol, calendar))
    stock_z = zscore_known(stock_known(symbol, calendar))
    moves = price_moves(symbol, wanted)
    oi = oi_states(symbol)
    print(symbol, len(moves), int(oi.oi_state.notna().sum()) if len(oi) else 0, flush=True)
    if moves.empty:
        return pd.DataFrame()
    moves["move"] = move_sign(moves.r_pre)
    moves["carry_z"] = moves.trading_date.map(carry_z)
    moves["stock_z"] = moves.trading_date.map(stock_z)
    moves["carry_align"] = curve_alignment(moves.carry_z, moves.move)
    moves["stock_align"] = stock_alignment(moves.stock_z, moves.move)
    return moves.merge(oi, on=["symbol", "trading_date"], how="left")


def build_states(table):
    rows = []
    for symbol in sorted(set(table.symbol)):
        part = table.loc[table.symbol == symbol, "trading_date"]
        wanted = set(pd.to_datetime(part).dt.normalize())
        calendar = pd.DatetimeIndex(load_shard(symbol, columns=["trading_date"]).trading_date.unique())
        calendar = calendar[calendar < pd.Timestamp("2022-01-01")]
        built = symbol_states(symbol, wanted, calendar)
        if len(built):
            rows.append(built)
    features_ = pd.concat(rows, ignore_index=True)
    merged = table.merge(features_, on=["symbol", "trading_date"], how="left", validate="one_to_one")
    merged["diff"] = merged.lclock_td - merged.hclock_td
    merged["year"] = merged.trading_date.dt.year
    merged["extend"] = still_extending(merged["diff"], merged.dev15)
    merged["push"] = aligned_dev(merged["diff"], merged.dev15)
    if merged.trading_date.dt.year.gt(2021).any():
        raise C.HoldoutViolation("状态表进入了 2022 或更晚")
    return merged


def judge_state(primary_t, year_same, sector_same, reversal_bp, expected_sign, decision):
    """主回归、年份、板块、反转档毛收益四项都过，才算通过。"""
    threshold = decision["interaction_abs_t"]
    primary_ok = bool(np.isfinite(primary_t) and abs(primary_t) > threshold and np.sign(primary_t) == expected_sign)
    year_ok = int(year_same) >= decision["min_years_same_sign"]
    sector_ok = int(sector_same) >= decision["min_sectors_same_sign"]
    payoff_ok = bool(np.isfinite(reversal_bp) and reversal_bp >= decision["min_reversal_bp"])
    return {
        "主回归过线": primary_ok,
        "年份过线": year_ok,
        "板块过线": sector_ok,
        "收益过线": payoff_ok,
        "通过": bool(primary_ok and year_ok and sector_ok and payoff_ok),
    }


def fit_state_joint(frame, specs):
    needed = ["gross_1130", "diff", "extend", *[spec["id"] for spec in specs]]
    data = frame.dropna(subset=needed).copy()
    data = data[data["diff"] != 0]
    diff_r = centered_rank(data, "diff")
    extend_r = centered_rank(data, "extend")
    y = centered_rank(data, "gross_1130").to_numpy()
    columns = {"低减高": diff_r.to_numpy()}
    for spec in specs:
        rank = centered_rank(data, spec["id"])
        columns[spec["name"]] = rank.to_numpy()
        columns[f"低减高×{spec['name']}"] = (diff_r * rank).to_numpy()
    columns["仍在延伸"] = extend_r.to_numpy()
    columns["低减高×仍在延伸"] = (diff_r * extend_r).to_numpy()
    fitted = clustered_coefficients(y, columns, data.trading_date.to_numpy())
    rows = []
    for spec in specs:
        name = f"低减高×{spec['name']}"
        coef, t_stat = fitted[name]
        rows.append({"模型": "联合附录", "变量": name, "状态列": spec["id"], "控制列": "extend",
                     "交互系数": coef, "交互t": t_stat, "低减高系数": fitted["低减高"][0],
                     "低减高t": fitted["低减高"][1], "笔数": len(data), "预期": spec["expected_interaction"]})
    return pd.DataFrame(rows)


def reversal_bp(signed, spec, decision):
    cell = signed[(signed.变量 == spec["name"]) & (signed.口径 == "全期") & (signed.范围 == "全部")
                  & (signed.档 == decision["reversal_bin"])]
    if cell.empty:
        return np.nan
    return float(cell.按方向收益bp.iloc[0])


def run_state():
    config, digest = load_state_config()
    print(f"声明日期 {config['declared_at']}  配置 sha256 {digest[:12]}", flush=True)
    merged = add_terciles(build_states(load_research()), config["variables"])
    STATE_OUT.mkdir(parents=True, exist_ok=True)
    keep = ["symbol", "trading_date", "sector", "year", "diff", "gross_1130", "dev15", "r_pre", "move",
            "carry_z", "stock_z", "carry_align", "stock_align", "oi_state", "extend", "push",
            "carry_align_bin", "stock_align_bin", "oi_state_bin"]
    merged[keep].to_parquet(STATE_OUT / "features.parquet", index=False)

    decision = config["decision"]
    ic = pd.concat([ic_rows(merged, spec, decision) for spec in config["variables"]], ignore_index=True)
    signed = pd.concat([signed_rows(merged, spec) for spec in config["variables"]], ignore_index=True)
    fits = [fit_one(merged, spec, control, model)
            for spec in config["variables"]
            for model, control in (("对照", "extend"), ("主回归", "extend"), ("稳健性", "push"))]
    fit_table = pd.concat([pd.DataFrame(fits), fit_state_joint(merged, config["variables"])], ignore_index=True)
    signs = pd.concat([sign_rows(ic, signed, spec, decision) for spec in config["variables"]], ignore_index=True)

    verdicts = []
    for spec in config["variables"]:
        primary = fit_table[(fit_table.模型 == "主回归") & (fit_table.变量 == spec["name"])].iloc[0]
        robust = fit_table[(fit_table.模型 == "稳健性") & (fit_table.变量 == spec["name"])].iloc[0]
        year_same, year_known = count_same(signs, spec, "截面IC", "年份")
        sector_same, sector_known = count_same(signs, spec, "截面IC", "板块")
        payoff = reversal_bp(signed, spec, decision)
        result = judge_state(primary.交互t, year_same, sector_same, payoff,
                             SIGN[spec["expected_interaction"]], decision)
        verdicts.append({"变量": spec["name"], "预期": spec["expected_interaction"],
                         "主回归t": primary.交互t, "稳健性t": robust.交互t,
                         "年份同号": year_same, "年份可判": year_known,
                         "板块同号": sector_same, "板块可判": sector_known,
                         "反转档bp": payoff, **result,
                         "声明日期": config["declared_at"], "配置sha256": digest})
    verdict = pd.DataFrame(verdicts)
    ic.to_csv(STATE_OUT / "bins_ic.csv", index=False, encoding="utf-8-sig")
    signed.to_csv(STATE_OUT / "bins_signed.csv", index=False, encoding="utf-8-sig")
    fit_table.to_csv(STATE_OUT / "interaction.csv", index=False, encoding="utf-8-sig")
    signs.to_csv(STATE_OUT / "signs.csv", index=False, encoding="utf-8-sig")
    verdict.to_csv(STATE_OUT / "verdict.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)
    print(fit_table.round(4).to_string(index=False))
    print(verdict.to_string(index=False))
    print(signed[signed.口径 == "全期"].round(3).to_string(index=False))

def main(argv=None):
    """特征、相关、个例、路径、量能、状态。样本外特征不从这里生成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    features_cmd = sub.add_parser("features", help="生成 09:16 特征表")
    features_cmd.add_argument("partition", nargs="*", default=["research", "validation_2022"])
    return_cmd = sub.add_parser("return", help="时钟位置与 09:17 之后的收益")
    return_cmd.add_argument("--exit", choices=tuple(EXITS), default="1130")
    return_cmd.add_argument("--balance", action="store_true")
    sub.add_parser("cases", help="平均路径、个例和分解")
    sub.add_parser("path", help="尾段冲击占比与路径效率")
    sub.add_parser("flow", help="开盘量能")
    sub.add_parser("state", help="期限结构、仓单、持仓量")
    args = parser.parse_args(argv)
    if args.step == "features":
        for part in args.partition:
            if part not in ("research", "validation_2022"):
                raise SystemExit("样本外特征只能在 C.final_evaluation(...) 里生成")
            build_feature_table(part)
        return
    {"return": run_return, "cases": run_cases, "path": run_path, "flow": run_flow, "state": run_state}[args.step](
        *(([args.exit, args.balance]) if args.step == "return" else ())
    )


if __name__ == "__main__":
    main()
