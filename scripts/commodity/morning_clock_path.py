"""高低点时钟的路径形态：尾段冲击占比、路径效率。只使用 2016–2021。

假设、分档和判定标准在跑之前写在 config/commodity/clock_path.json。
不读 2022，也不读 2023 年以后。这不是交易规则。

    python scripts/commodity/morning_clock_path.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "commodity"))
from tfcta.factors import day_codes_of, raw_price_path
import morning
import morning_clock_return as clock
import morning_clock_cases as cases
import morning_features as features

CONFIG_PATH = ROOT / "config" / "commodity" / "clock_path.json"
OUT = morning.RUNS / "clock_path"
FIVE = 5 * features.MINUTE
OFFSET = features.MINUTE
SIGN = {"正": 1, "负": -1}
BIN_LABELS = ("低", "中", "高")


def load_config(path=CONFIG_PATH):
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
    out.loc[valid.index] = pd.qcut(valid.rank(method="first"), 3, labels=list(BIN_LABELS)).astype(str)
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
    frame = features.load_minutes(symbol, "research")
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
        d = pos.get(features.stamp(day, features.DECISION))
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


def build_features(table):
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
    scopes = [("全部", frame)] + [(sector, frame[frame.sector == sector]) for sector in morning.SECTORS]
    for scope, part in scopes:
        for label in BIN_LABELS:
            cell = part[part[column + "_bin"] == label]
            ic = daily_spearman(cell, min_names=decision["min_names"])
            test = clock.ic_summary(
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
    labelled = cases.label_cases(frame.dropna(subset=[column + "_bin"]))
    rows = []
    groups = [("全部", "全期", labelled)]
    groups += [(str(year), "年份", part) for year, part in labelled.groupby(labelled.trading_date.dt.year)]
    groups += [(sector, "板块", part) for sector, part in labelled.groupby("sector")]
    for scope, kind, part in groups:
        for label in BIN_LABELS:
            cell = part[part[column + "_bin"] == label]
            mean = clock.bin_row("", "", "", "", cell.trading_date.to_numpy(), cell.signed.to_numpy())
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


def fit_joint(frame):
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
    for scope in ["全部", *morning.SECTORS]:
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


def main():
    config, digest = load_config()
    print(f"声明日期 {config['declared_at']}  配置 sha256 {digest[:12]}", flush=True)
    table = clock.load_research()
    merged = add_terciles(build_features(table), config["variables"])
    if len(merged) != len(table):
        print(f"路径缺 {len(table) - len(merged)} 个品种日")
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["symbol", "trading_date", "sector", "year", "diff", "gross_1130", "dev15",
            "impulse", "efficiency", "extend", "push", "impulse_bin", "efficiency_bin"]
    merged[keep].to_parquet(OUT / "features.parquet", index=False)

    decision = config["decision"]
    ic = pd.concat([ic_rows(merged, spec, decision) for spec in config["variables"]], ignore_index=True)
    signed = pd.concat([signed_rows(merged, spec) for spec in config["variables"]], ignore_index=True)
    fits = []
    for spec in config["variables"]:
        fits.append(fit_one(merged, spec, "extend", "对照"))
        fits.append(fit_one(merged, spec, "extend", "主回归"))
        fits.append(fit_one(merged, spec, "push", "稳健性"))
    fit_table = pd.concat([pd.DataFrame(fits), fit_joint(merged)], ignore_index=True)
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
    ic.to_csv(OUT / "bins_ic.csv", index=False, encoding="utf-8-sig")
    signed.to_csv(OUT / "bins_signed.csv", index=False, encoding="utf-8-sig")
    fit_table.to_csv(OUT / "interaction.csv", index=False, encoding="utf-8-sig")
    signs.to_csv(OUT / "signs.csv", index=False, encoding="utf-8-sig")
    verdict.to_csv(OUT / "verdict.csv", index=False, encoding="utf-8-sig")
    corr = merged[["impulse", "efficiency", "extend", "push", "diff"]].corr("spearman")
    corr.to_csv(OUT / "corr.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(corr.round(3).to_string())
    print(fit_table.round(4).to_string(index=False))
    print(verdict.to_string(index=False))
    print(signs[signs.口径 == "截面IC"].round(4).to_string(index=False))
    print(signed[signed.口径.isin(["全期", "年份"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
