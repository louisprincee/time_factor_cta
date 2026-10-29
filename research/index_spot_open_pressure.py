"""现货 T+1 开盘卖压：弱收盘的次日，开盘半小时是否继续下跌、其后是否反弹。

    python -m research.index_spot_open_pressure

规则在 config/research/index_spot_open_pressure.json，先登记后运行。
分钟线来自 download.index.getIndexSpot --minute，只读到 2021-12-31。
结果在 runs/index/index_spot_open_pressure/。不读 2022 及以后。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.time_factor_screen import ols_hac
from strategies.index_close_location import market_close_location
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_spot_open_pressure.json"
OUT = C.RUNS_DIR / "index" / "index_spot_open_pressure"
MINUTE_PATH = C.INDEX_DATA_ROOT / "external_rqdata" / "spot_minute.parquet"
DAILY_PATH = C.INDEX_DATA_ROOT / "external_rqdata" / "spot_daily.csv"
SYMBOLS = ("IF", "IH", "IC")
TAGS = ("P1", "P2", "P3", "P4a", "P4b", "P4", "P5", "D_IF", "D_IH", "D_IC", "R0")
OUTCOME_FIELDS = ("open_30", "rest", "vol_share", "open_30_IF", "open_30_IH", "open_30_IC", "gap")


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    parameters = protocol.get("parameters") or {}
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022" not in sample
        or "不读 2026" not in sample
        or list(protocol.get("predictions", {})) != [tag for tag in TAGS if tag != "P4"]
        or parameters.get("instruments") != list(SYMBOLS)
        or parameters.get("open_bars") != 30
        or parameters.get("min_morning_bars") != 120
        or parameters.get("min_day_bars") != 230
        or parameters.get("hac_lags") != 5
        or parameters.get("primary_outcome_start") != "2016-01-04"
        or parameters.get("outcome_end") != "2021-12-31"
        or parameters.get("mediation_shrink") != 0.5
        or parameters.get("bar30_minute_min") != "09:55"
        or parameters.get("bar30_minute_max") != "10:05"
        or parameters.get("open_match_tolerance") != 0.0001
        or parameters.get("max_mismatch_share") != 0.01
    ):
        raise ValueError("现货开盘卖压登记与实现或样本边界不一致")
    return protocol


def _clock_minute(stamp: pd.Series) -> pd.Series:
    stamp = pd.to_datetime(stamp)
    return stamp.dt.hour * 60 + stamp.dt.minute


def session_paths(minutes: pd.DataFrame, open_bars: int = 30, min_morning: int = 120, min_day: int = 230) -> pd.DataFrame:
    """每个指数的完整交易日：开盘前 30 根的收益、其后收益、这 30 根的成交量占比。"""
    frame = minutes.copy()
    if "datetime" not in frame.columns or "symbol" not in frame.columns:
        raise KeyError("现货分钟需要 symbol 和 datetime")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    C.assert_no_holdout_dates(frame["datetime"], "现货指数分钟")
    frame["trading_date"] = frame["datetime"].dt.normalize()
    rows = []
    for (symbol, day), group in frame.groupby(["symbol", "trading_date"], sort=True):
        group = group.sort_values("datetime")
        morning = group[group["datetime"].dt.hour < 12]
        if len(morning) < min_morning or len(group) < min_day:
            continue
        window = morning.iloc[:open_bars]
        open_px = float(window["open"].iloc[0])
        px30 = float(window["close"].iloc[-1])
        close_px = float(group["close"].iloc[-1])
        volume = float(group["volume"].sum())
        if open_px <= 0 or px30 <= 0 or close_px <= 0 or volume <= 0:
            continue
        rows.append({
            "symbol": symbol,
            "trading_date": pd.Timestamp(day),
            "open_30": px30 / open_px - 1.0,
            "rest": close_px / px30 - 1.0,
            "vol_share": float(window["volume"].sum()) / volume,
            "day_open": open_px,
            "bar30_time": window["datetime"].iloc[-1],
        })
    if not rows:
        raise ValueError("现货分钟没有完整交易日")
    out = pd.DataFrame(rows)
    if out.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("现货分钟路径有重复的品种-日期")
    return out


def market_outcomes(paths: pd.DataFrame) -> pd.DataFrame:
    """三指数都有完整路径的日子，等权合成；任一指数缺失则该日不用。"""
    fields = ("open_30", "rest", "vol_share", "day_open")
    wide = {field: paths.pivot(index="trading_date", columns="symbol", values=field).reindex(columns=list(SYMBOLS))
            for field in fields}
    out = pd.DataFrame({field: frame.mean(axis=1, skipna=False) for field, frame in wide.items()})
    opens = wide["open_30"]
    out["open_30_IF"] = opens["IF"]
    out["open_30_IH"] = opens["IH"]
    out["open_30_IC"] = opens["IC"]
    out["ic_minus_ih"] = opens["IC"] - opens["IH"]
    return out.dropna(how="any").sort_index()


def with_prior_clv(outcomes: pd.DataFrame, clv: pd.Series, gap: pd.Series) -> pd.DataFrame:
    """结果日用期货交易日序列的前一天收盘位置。当天收盘位置不进回归。

    前一天可以没有完整的现货分钟路径（例如熔断提前收盘），只要期货收盘位置有值。
    """
    if clv.index.has_duplicates:
        raise ValueError("期货收盘位置日期必须唯一")
    clv = clv.sort_index()
    previous = pd.Series(clv.index[:-1].to_numpy(), index=clv.index[1:])
    out = outcomes.loc[outcomes.index.isin(previous.index)].copy()
    if len(out) < 2:
        raise ValueError("现货路径与期货收盘位置的重叠不足")
    out["clv_prior"] = clv.reindex(previous.reindex(out.index).to_numpy()).to_numpy()
    out["gap"] = gap.reindex(out.index)
    C.assert_no_holdout_dates(out.index, "现货开盘卖压结果日")
    return out.sort_index().dropna(subset=["clv_prior"])


def check_bar_clock(paths: pd.DataFrame, low: str = "09:55", high: str = "10:05") -> dict:
    """第 30 根若不是 10:00 前后，这 30 根就不是开盘半小时。"""
    low_m = int(low[:2]) * 60 + int(low[3:])
    high_m = int(high[:2]) * 60 + int(high[3:])
    summary = {}
    for symbol, group in paths.groupby("symbol"):
        minute = int(_clock_minute(group["bar30_time"]).median())
        summary[symbol] = f"{minute // 60:02d}:{minute % 60:02d}"
        if not low_m <= minute <= high_m:
            raise ValueError(f"{symbol} 第 30 根时刻中位数是 {summary[symbol]}，不在 {low}–{high}")
    return summary


def mismatched_opens(paths: pd.DataFrame, daily: pd.DataFrame, tolerance: float, max_share: float) -> pd.DataFrame:
    """日线开盘对不上的品种日。中位数或占比超标说明窗口定义错了，直接拒绝。"""
    day = daily.rename(columns={"date": "trading_date"})
    day["trading_date"] = pd.to_datetime(day["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(day["trading_date"], "现货指数日线")
    merged = paths.merge(day[["symbol", "trading_date", "open"]], on=["symbol", "trading_date"], how="inner")
    if len(merged) != len(paths):
        raise ValueError("现货分钟路径有对不上现货日线的品种日")
    gap = (merged["day_open"] - merged["open"]).abs() / merged["open"]
    bad = merged.loc[gap > tolerance, ["symbol", "trading_date"]].assign(相对偏差=gap[gap > tolerance].to_numpy())
    if float(gap.median()) > tolerance or float((gap > tolerance).mean()) > max_share:
        raise ValueError(
            f"分钟开盘与日线开盘中位相对偏差 {float(gap.median()):.6g}，"
            f"超标占比 {float((gap > tolerance).mean()):.4f}，窗口定义不对"
        )
    return bad.reset_index(drop=True)


def fit(y: pd.Series, xs: list[pd.Series], lags: int) -> tuple[float, float, float, int]:
    frame = pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 60:
        return np.nan, np.nan, np.nan, len(frame)
    coef, tstat = ols_hac(frame.iloc[:, 0].to_numpy(), [frame.iloc[:, i].to_numpy() for i in range(1, frame.shape[1])], lags)
    se = abs(coef / tstat) if tstat else np.nan
    return coef, tstat, se, len(frame)


def verdict(value, tstat, expected: str) -> str:
    if pd.isna(tstat):
        return "不确定"
    right = value > 0 if expected == ">0" else value < 0
    if abs(tstat) >= 2:
        return "支持" if right else "否定"
    return "不确定"


def yearly_signs(y: pd.Series, x: pd.Series, lags: int) -> str:
    signs = []
    for year in sorted(set(y.dropna().index.year)):
        mask = y.index.year == year
        coef = fit(y[mask], [x[mask]], lags)[0]
        signs.append(f"{year}:{'+' if coef > 0 else '−' if coef < 0 else '?'}")
    return " ".join(signs)


def channel_verdict(rest_on_open: float, rest_on_open_t: float, controlled: float, alone: float, shrink: float) -> str:
    """开盘卖压解释后段：后段顺着开盘收益走，且收盘位置斜率至少掉一半。"""
    if pd.isna(rest_on_open_t) or pd.isna(alone) or alone == 0 or pd.isna(controlled):
        return "不确定"
    open_supports = rest_on_open < 0 and rest_on_open_t <= -2
    shrunk = abs(controlled) <= shrink * abs(alone) and alone < 0
    if open_supports and shrunk:
        return "支持"
    if (rest_on_open > 0 and rest_on_open_t >= 2) or (alone < 0 and abs(controlled) > abs(alone)):
        return "否定"
    return "不确定"


def tests(panel: pd.DataFrame, lags: int, shrink: float) -> pd.DataFrame:
    clv = panel["clv_prior"]
    specs = [
        ("P1", "开盘半小时收益 ~ 前一日收盘位置", "open_30", [clv], ">0", True),
        ("P2", "其后收益 ~ 前一日收盘位置", "rest", [clv], "<0", True),
        ("P3", "开盘成交量占比 ~ 前一日收盘位置", "vol_share", [clv], "<0", True),
        ("P4a", "其后收益 ~ 开盘半小时收益 + 前一日收盘位置", "rest", [panel["open_30"], clv], "<0", False),
        ("P4b", "其后收益 ~ 前一日收盘位置 + 开盘半小时收益", "rest", [clv, panel["open_30"]], "<0", False),
        ("P5", "中证500减上证50的开盘半小时 ~ 前一日收盘位置", "ic_minus_ih", [clv], ">0", True),
        ("D_IF", "沪深300开盘半小时 ~ 前一日收盘位置", "open_30_IF", [clv], ">0", False),
        ("D_IH", "上证50开盘半小时 ~ 前一日收盘位置", "open_30_IH", [clv], ">0", False),
        ("D_IC", "中证500开盘半小时 ~ 前一日收盘位置", "open_30_IC", [clv], ">0", False),
        ("R0", "隔夜跳空 ~ 前一日收盘位置（已知复述）", "gap", [clv], ">0", True),
    ]
    rows = []
    fitted = {}
    for tag, label, target, xs, expected, signs in specs:
        coef, tstat, se, nobs = fit(panel[target], xs, lags)
        fitted[tag] = (coef, tstat)
        rows.append({
            "预测": tag, "内容": label, "系数": coef, "NW_t": tstat, "天数": nobs, "预期": expected,
            "判断": verdict(coef, tstat, expected),
            "逐年符号": yearly_signs(panel[target], xs[0], lags) if signs else "",
        })
    p2, p4a, p4b = fitted["P2"], fitted["P4a"], fitted["P4b"]
    ratio = abs(p4b[0]) / abs(p2[0]) if p2[0] else np.nan
    rows.append({
        "预测": "P4", "内容": "控制开盘半小时后，收盘位置斜率绝对值相对 P2 的比例",
        "系数": ratio, "NW_t": np.nan, "天数": int(panel[["rest", "open_30", "clv_prior"]].dropna().shape[0]),
        "预期": f"<={shrink} 且 P4a 支持", "判断": channel_verdict(p4a[0], p4a[1], p4b[0], p2[0], shrink),
        "逐年符号": "",
    })
    return pd.DataFrame(rows)


def quintiles(panel: pd.DataFrame) -> pd.DataFrame:
    group = pd.qcut(panel["clv_prior"], 5, labels=False) + 1
    rows = []
    for name, part in panel.groupby(group):
        rows.append({
            "分组": int(name), "天数": len(part), "前一日收盘位置": float(part["clv_prior"].mean()),
            "开盘半小时bp": float(part["open_30"].mean() * 1e4), "其后bp": float(part["rest"].mean() * 1e4),
            "成交量占比": float(part["vol_share"].mean()), "隔夜bp": float(part["gap"].mean() * 1e4),
        })
    return pd.DataFrame(rows)


def futures_clv() -> pd.Series:
    from tfcta.data import load_daily_bars

    bars = load_daily_bars(list(SYMBOLS), directory=C.INDEX_RESEARCH_DIR)
    width = bars["highw"] - bars["loww"]
    ok = (width.gt(0) & bars["closew"].notna() & bars["highw"].notna() & bars["loww"].notna()).all(axis=1)
    index = ok.index[ok]
    index = index[(index >= pd.Timestamp("2015-04-16")) & (index <= pd.Timestamp("2021-12-31"))]
    return market_close_location(bars, index)


def spot_gap(daily: pd.DataFrame) -> pd.Series:
    day = daily.copy()
    day["date"] = pd.to_datetime(day["date"]).dt.normalize()
    C.assert_no_holdout_dates(day["date"], "现货指数日线")
    wide_open = day.pivot(index="date", columns="symbol", values="open").reindex(columns=list(SYMBOLS))
    wide_close = day.pivot(index="date", columns="symbol", values="close").reindex(columns=list(SYMBOLS))
    gap = wide_open / wide_close.shift(1) - 1.0
    return gap.mean(axis=1, skipna=False).rename("gap")


def run() -> None:
    protocol = load_protocol()
    parameters = protocol["parameters"]
    if not MINUTE_PATH.is_file():
        raise FileNotFoundError("缺少现货分钟线，先运行 python -m download.index.getIndexSpot --minute")
    minutes = pd.read_parquet(MINUTE_PATH)
    C.assert_no_holdout_dates(pd.to_datetime(minutes["datetime"]), "现货指数分钟")
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    paths = session_paths(
        minutes, parameters["open_bars"], parameters["min_morning_bars"], parameters["min_day_bars"],
    )
    clocks = check_bar_clock(paths, parameters["bar30_minute_min"], parameters["bar30_minute_max"])
    bad = mismatched_opens(paths, daily, parameters["open_match_tolerance"], parameters["max_mismatch_share"])
    if len(bad):
        marked = paths.merge(bad.assign(_bad=1), on=["symbol", "trading_date"], how="left")
        paths = marked.loc[marked["_bad"].isna()].drop(columns=["_bad", "相对偏差"])
    outcomes = market_outcomes(paths)
    panel = with_prior_clv(outcomes, futures_clv(), spot_gap(daily))
    panel = panel[panel.index <= pd.Timestamp(parameters["outcome_end"])]
    lags, shrink = parameters["hac_lags"], parameters["mediation_shrink"]
    full = tests(panel, lags, shrink)
    primary = tests(panel[panel.index >= pd.Timestamp(parameters["primary_outcome_start"])], lags, shrink)
    full.insert(0, "样本", f"{panel.index.min().date()}..{panel.index.max().date()}")
    primary.insert(0, "样本", f"{parameters['primary_outcome_start']}..{panel.index.max().date()}")
    table = pd.concat([primary, full], ignore_index=True)
    q_primary = quintiles(panel[panel.index >= parameters["primary_outcome_start"]])
    q_primary.insert(0, "样本", "2016-01-04 起")
    q_full = quintiles(panel)
    q_full.insert(0, "样本", "全样本")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat([q_primary, q_full], ignore_index=True).to_csv(OUT / "quintiles.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(OUT / "daily_panel.csv", encoding="utf-8-sig")
    kept = paths.groupby("symbol")["trading_date"].agg(["min", "max", "count"])
    desc = {
        "第30根时刻中位数": clocks,
        "开盘价对不上日线的品种日": [
            {"symbol": row.symbol, "trading_date": str(pd.Timestamp(row.trading_date).date()), "相对偏差": float(row.相对偏差)}
            for row in bad.itertuples(index=False)
        ],
        "完整交易日": {symbol: {"min": str(row["min"].date()), "max": str(row["max"].date()), "days": int(row["count"])}
                    for symbol, row in kept.iterrows()},
        "进入回归的结果日": int(len(panel)),
        "主样本结果日": int((panel.index >= parameters["primary_outcome_start"]).sum()),
    }
    (OUT / "protocol.json").write_text(
        json.dumps({"protocol": protocol, "descriptive": desc}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(table.round(4).to_string(index=False), flush=True)
    print(pd.concat([q_primary, q_full], ignore_index=True).round(2).to_string(index=False), flush=True)
    print(json.dumps(desc, ensure_ascii=False, indent=2), flush=True)
    print(f"结果：{OUT}", flush=True)


if __name__ == "__main__":
    run()
