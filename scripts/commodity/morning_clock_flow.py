"""检验开盘异常量、单位成交量冲击和量峰时间能否区分回吐与延续。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "commodity"))
from tfcta import config as C
from tfcta.factors import day_codes_of, raw_price_path
import morning
import morning_clock_path as path
import morning_clock_return as clock
import morning_features as features

CONFIG_PATH = ROOT / "config" / "commodity" / "clock_flow.json"
OUT = morning.RUNS / "clock_flow"
WINDOW_START, WINDOW_END = (9, 1), features.DECISION
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


def load_config(config_path=CONFIG_PATH):
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
    frame = features.load_minutes(symbol, "research")
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
        decision = positions.get(features.stamp(day, WINDOW_END))
        first_window = positions.get(features.stamp(day, WINDOW_START))
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
        return pd.DataFrame(columns=["symbol", "trading_date", *[s["id"] for s in load_config()[0]["variables"]]])
    daily = pd.DataFrame(rows).sort_values("trading_date").drop_duplicates("trading_date", keep="last")
    baseline = daily.open_volume.rolling(BASELINE_WINDOW, min_periods=BASELINE_MIN).median().shift(1)
    daily["open_volume_surprise"] = [volume_surprise(value, prior)
                                     for value, prior in zip(daily.open_volume, baseline)]
    daily["impact_per_volume"] = [impact_per_volume(move, surprise)
                                  for move, surprise in zip(daily.open_move, daily.open_volume_surprise)]
    return daily[daily.trading_date.isin(wanted)][
        ["symbol", "trading_date", "open_volume_surprise", "impact_per_volume", "volume_peak_clock"]]


def build_features(table):
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
    merged["extend"] = path.still_extending(merged["diff"], merged.dev15)
    merged["push"] = path.aligned_dev(merged["diff"], merged.dev15)
    return merged


def fit_joint(frame, specs):
    needed = ["gross_1130", "diff", "extend", *[spec["id"] for spec in specs]]
    data = frame.dropna(subset=needed).copy()
    data = data[data["diff"] != 0]
    diff_rank = path.centered_rank(data, "diff")
    extend_rank = path.centered_rank(data, "extend")
    y = path.centered_rank(data, "gross_1130").to_numpy()
    columns = {"低减高": diff_rank.to_numpy(), "仍在延伸": extend_rank.to_numpy(),
               "低减高×仍在延伸": (diff_rank * extend_rank).to_numpy()}
    for spec in specs:
        state_rank = path.centered_rank(data, spec["id"])
        columns[spec["name"]] = state_rank.to_numpy()
        columns[f"低减高×{spec['name']}"] = (diff_rank * state_rank).to_numpy()
    fitted = path.clustered_coefficients(y, columns, data.trading_date.to_numpy())
    return pd.DataFrame([
        {"模型": "联合附录", "变量": f"低减高×{spec['name']}", "状态列": spec["id"],
         "控制列": "其他状态+extend", "交互系数": fitted[f"低减高×{spec['name']}"][0],
         "交互t": fitted[f"低减高×{spec['name']}"][1], "笔数": len(data),
         "预期": spec["expected_interaction"]}
        for spec in specs
    ])


def main():
    config, digest = load_config()
    print(f"声明日期 {config['declared_at']}  配置 sha256 {digest[:12]}", flush=True)
    merged = build_features(clock.load_research())
    merged = path.add_terciles(merged, config["variables"])
    OUT.mkdir(parents=True, exist_ok=True)
    decision = config["decision"]
    ic = pd.concat([path.ic_rows(merged, spec, decision) for spec in config["variables"]], ignore_index=True)
    signed = pd.concat([path.signed_rows(merged, spec) for spec in config["variables"]], ignore_index=True)
    fits = []
    for spec in config["variables"]:
        fits.extend(path.fit_one(merged, spec, control, model)
                    for model, control in (("对照", "extend"), ("主回归", "extend"), ("稳健性", "push")))
    fit_table = pd.concat([pd.DataFrame(fits), fit_joint(merged, config["variables"])], ignore_index=True)
    signs = pd.concat([path.sign_rows(ic, signed, spec, decision) for spec in config["variables"]], ignore_index=True)

    verdicts = []
    for spec in config["variables"]:
        primary = fit_table[(fit_table.模型 == "主回归") & (fit_table.变量 == spec["name"])].iloc[0]
        robust = fit_table[(fit_table.模型 == "稳健性") & (fit_table.变量 == spec["name"])].iloc[0]
        year_same, year_known = path.count_same(signs, spec, "截面IC", "年份")
        sector_same, sector_known = path.count_same(signs, spec, "截面IC", "板块")
        result = path.judge(primary.交互t, robust.交互t, year_same, sector_same,
                            path.SIGN[spec["expected_interaction"]], decision)
        verdicts.append({"变量": spec["name"], "预期": spec["expected_interaction"],
                         "主回归t": primary.交互t, "稳健性t": robust.交互t,
                         "年份同号": year_same, "年份可判": year_known,
                         "板块同号": sector_same, "板块可判": sector_known,
                         **result, "声明日期": config["declared_at"], "配置sha256": digest})
    verdict = pd.DataFrame(verdicts)
    merged.to_parquet(OUT / "features.parquet", index=False)
    ic.to_csv(OUT / "bins_ic.csv", index=False, encoding="utf-8-sig")
    signed.to_csv(OUT / "bins_signed.csv", index=False, encoding="utf-8-sig")
    fit_table.to_csv(OUT / "interaction.csv", index=False, encoding="utf-8-sig")
    signs.to_csv(OUT / "signs.csv", index=False, encoding="utf-8-sig")
    verdict.to_csv(OUT / "verdict.csv", index=False, encoding="utf-8-sig")
    print(fit_table.round(4).to_string(index=False))
    print(verdict.to_string(index=False))


if __name__ == "__main__":
    main()