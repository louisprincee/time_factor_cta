"""高低点时钟的供需状态：期限结构、仓单、持仓量。只使用 2016–2021。

假设、符号和判定标准在跑之前写在 config/commodity/clock_state.json。
不读 2022，也不读 2023 年以后。这不是交易规则。

    python scripts/commodity/morning_clock_state.py
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
from tfcta import config as C
from tfcta.data import load_shard
from tfcta.factors import align_asof, day_codes_of, raw_price_path, _series
import morning
import morning_clock_return as clock
import morning_clock_path as path
import morning_features as features

CONFIG_PATH = ROOT / "config" / "commodity" / "clock_state.json"
OUT = morning.RUNS / "clock_state"
EXTERNAL = C.COMMODITY_DATA / "external_rqdata"
OI_DIR = C.RESEARCH_DIR / "open_interest"
Z_WINDOW, Z_MIN = 250, 60
WARRANT_LAG, OI_BASELINE, OI_MIN = 20, 20, 10
CARRY_LAG, FILL_LIMIT = 1, 5


def load_config(path=CONFIG_PATH):
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
        d = pos.get(features.stamp(day, features.DECISION))
        day_bars = np.flatnonzero(~night[s:e]) + s
        if d is None or not s <= d < e or not day_bars.size:
            continue
        yield day, s, e, d, int(day_bars[0])


def price_moves(symbol, wanted):
    frame = features.load_minutes(symbol, "research")
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
    merged["extend"] = path.still_extending(merged["diff"], merged.dev15)
    merged["push"] = path.aligned_dev(merged["diff"], merged.dev15)
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


def fit_joint(frame, specs):
    needed = ["gross_1130", "diff", "extend", *[spec["id"] for spec in specs]]
    data = frame.dropna(subset=needed).copy()
    data = data[data["diff"] != 0]
    diff_r = path.centered_rank(data, "diff")
    extend_r = path.centered_rank(data, "extend")
    y = path.centered_rank(data, "gross_1130").to_numpy()
    columns = {"低减高": diff_r.to_numpy()}
    for spec in specs:
        rank = path.centered_rank(data, spec["id"])
        columns[spec["name"]] = rank.to_numpy()
        columns[f"低减高×{spec['name']}"] = (diff_r * rank).to_numpy()
    columns["仍在延伸"] = extend_r.to_numpy()
    columns["低减高×仍在延伸"] = (diff_r * extend_r).to_numpy()
    fitted = path.clustered_coefficients(y, columns, data.trading_date.to_numpy())
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


def main():
    config, digest = load_config()
    print(f"声明日期 {config['declared_at']}  配置 sha256 {digest[:12]}", flush=True)
    merged = path.add_terciles(build_states(clock.load_research()), config["variables"])
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["symbol", "trading_date", "sector", "year", "diff", "gross_1130", "dev15", "r_pre", "move",
            "carry_z", "stock_z", "carry_align", "stock_align", "oi_state", "extend", "push",
            "carry_align_bin", "stock_align_bin", "oi_state_bin"]
    merged[keep].to_parquet(OUT / "features.parquet", index=False)

    decision = config["decision"]
    ic = pd.concat([path.ic_rows(merged, spec, decision) for spec in config["variables"]], ignore_index=True)
    signed = pd.concat([path.signed_rows(merged, spec) for spec in config["variables"]], ignore_index=True)
    fits = [path.fit_one(merged, spec, control, model)
            for spec in config["variables"]
            for model, control in (("对照", "extend"), ("主回归", "extend"), ("稳健性", "push"))]
    fit_table = pd.concat([pd.DataFrame(fits), fit_joint(merged, config["variables"])], ignore_index=True)
    signs = pd.concat([path.sign_rows(ic, signed, spec, decision) for spec in config["variables"]], ignore_index=True)

    verdicts = []
    for spec in config["variables"]:
        primary = fit_table[(fit_table.模型 == "主回归") & (fit_table.变量 == spec["name"])].iloc[0]
        robust = fit_table[(fit_table.模型 == "稳健性") & (fit_table.变量 == spec["name"])].iloc[0]
        year_same, year_known = path.count_same(signs, spec, "截面IC", "年份")
        sector_same, sector_known = path.count_same(signs, spec, "截面IC", "板块")
        payoff = reversal_bp(signed, spec, decision)
        result = judge_state(primary.交互t, year_same, sector_same, payoff,
                             path.SIGN[spec["expected_interaction"]], decision)
        verdicts.append({"变量": spec["name"], "预期": spec["expected_interaction"],
                         "主回归t": primary.交互t, "稳健性t": robust.交互t,
                         "年份同号": year_same, "年份可判": year_known,
                         "板块同号": sector_same, "板块可判": sector_known,
                         "反转档bp": payoff, **result,
                         "声明日期": config["declared_at"], "配置sha256": digest})
    verdict = pd.DataFrame(verdicts)
    ic.to_csv(OUT / "bins_ic.csv", index=False, encoding="utf-8-sig")
    signed.to_csv(OUT / "bins_signed.csv", index=False, encoding="utf-8-sig")
    fit_table.to_csv(OUT / "interaction.csv", index=False, encoding="utf-8-sig")
    signs.to_csv(OUT / "signs.csv", index=False, encoding="utf-8-sig")
    verdict.to_csv(OUT / "verdict.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)
    print(fit_table.round(4).to_string(index=False))
    print(verdict.to_string(index=False))
    print(signed[signed.口径 == "全期"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
