"""股指收盘位置反转的经济机制：期现套利限制、库存（流动性提供）和过度反应，用 L1 逐笔和现货指数检验。

    python -m research.index_liquidity_mechanism

规则在 config/research/index_liquidity_mechanism.json，先登记后运行。只读股指研究期：
L1 逐笔 data/index/external_rqdata/l1_ticks/、现货日线 spot_daily.csv（download.index.getIndexSpot）、研究分钟分片。
结果在 runs/index/index_liquidity_mechanism/。不读 2022 及以后，holdout_locked 不读。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import newey_west_t
from research import index_market_clock as I
from research.time_factor_screen import ols_hac
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_liquidity_mechanism.json"
OUT = C.RUNS_DIR / "index" / "index_liquidity_mechanism"
TICK_ROOT = C.INDEX_DATA_ROOT / "external_rqdata" / "l1_ticks"
SPOT_PATH = C.INDEX_DATA_ROOT / "external_rqdata" / "spot_daily.csv"
SYMBOLS = ("IF", "IH", "IC")
START, END = pd.Timestamp("2015-04-16"), pd.Timestamp(C.RESEARCH_END)
SUB_START = pd.Timestamp("2016-01-04")
LAGS = 5
TAIL = pd.Timedelta(minutes=30)
MIN_MINUTES = 60
IMPACT_WINDOW = 250


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "declared_test_not_a_strategy":
        raise ValueError("检验登记必须声明这不是策略")
    if "不读 2022" not in protocol.get("sample", "") or "不读 2026" not in protocol.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    return protocol


# --------------------------------------------------------------------------
# 逐笔 → 逐合约日特征（收盘 t 已知）
# --------------------------------------------------------------------------
def tick_day_features(ticks: pd.DataFrame) -> dict:
    """一个合约日的日盘逐笔（volume 为当日累计）→ 中间价、成交流量的价格冲击、尾盘和开盘深度。"""
    t = ticks.sort_values("datetime")
    ok = (t["a1"] > 0) & (t["b1"] > 0) & (t["a1"] >= t["b1"])
    t = t.assign(mid=((t["a1"] + t["b1"]) / 2).where(ok), depth=(t["a1_v"] + t["b1_v"]).where(ok))
    t["dvol"] = t["volume"].diff().fillna(t["volume"]).clip(lower=0)
    prev_mid = t["mid"].ffill().shift(1)
    direction = np.sign(t["last"] - prev_mid)
    tick_rule = np.sign(t["last"].diff()).replace(0, np.nan).ffill()
    t["sv"] = direction.where(direction != 0, tick_rule).fillna(0.0) * t["dvol"]
    quotes = t[t["mid"].notna()]
    out = {"n_ticks": len(t)}
    if len(quotes) < 2:
        return out
    stamp = quotes["datetime"]
    first, last = stamp.iloc[0], stamp.iloc[-1]
    day = pd.Timestamp(first).normalize()
    mid_open, mid_close = float(quotes["mid"].iloc[0]), float(quotes["mid"].iloc[-1])
    at_1500 = quotes[stamp <= day + pd.Timedelta(hours=15)]
    in_first30 = quotes[stamp <= first + TAIL]
    tail = quotes[stamp >= last - TAIL]
    head = quotes[stamp <= first + TAIL]
    out.update({
        "mid_open": mid_open, "mid_close": mid_close,
        "mid_1500": float(at_1500["mid"].iloc[-1]) if len(at_1500) else np.nan,
        "mid_first30": float(in_first30["mid"].iloc[-1]),
        "depth_head": float(head["depth"].mean()), "depth_tail": float(tail["depth"].mean()),
    })
    minute = t.set_index("datetime").resample("1min")
    sv = minute["sv"].sum()
    mid_end = minute["mid"].last()
    frame = pd.DataFrame({"sv": sv, "mid": mid_end}).dropna(subset=["mid"])
    frame["dm"] = np.log(frame["mid"]).diff()
    frame = frame.dropna(subset=["dm"])
    if len(frame) < MIN_MINUTES or frame["sv"].std() == 0:
        return out
    lam = float(np.polyfit(frame["sv"].to_numpy(), frame["dm"].to_numpy(), 1)[0])
    tail_minutes = frame.index >= (last - TAIL).floor("min")
    total = float(np.log(mid_close / mid_open))
    flow = lam * float(frame["sv"].sum())
    out.update({
        "lambda": lam, "flow": flow, "flow_tail": lam * float(frame.loc[tail_minutes, "sv"].sum()),
        "resid": total - flow, "intraday_mid": total,
        "impact": lam * float(frame["sv"].abs().median()),
    })
    return out


def build_tick_features(symbols=SYMBOLS) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        root = TICK_ROOT / symbol
        C.assert_research_only(root)
        for path in sorted(root.glob("*/*.parquet")):
            frame = pd.read_parquet(path, columns=[
                "datetime", "trading_date", "last", "volume", "a1", "b1", "a1_v", "b1_v", "contract"])
            frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
            C.assert_no_holdout_dates(frame["trading_date"], f"{symbol} L1 逐笔")
            for day, part in frame.groupby("trading_date", sort=True):
                if not START <= day <= END:
                    continue
                rows.append({"symbol": symbol, "trading_date": day, "contract": part["contract"].iloc[-1],
                             **tick_day_features(part)})
    out = pd.DataFrame(rows)
    if out.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("L1 逐笔特征出现重复的品种-日期")
    return out


# --------------------------------------------------------------------------
# 日度面板
# --------------------------------------------------------------------------
def market_panel(features: pd.DataFrame, spot: pd.DataFrame, bars: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """逐品种量，三品种等权（任一缺失则缺失）。次日量都对齐到 t。"""
    def wide(frame, field):
        return frame.pivot(index="trading_date", columns="symbol", values=field).reindex(index=dates, columns=list(SYMBOLS))

    contract = wide(features, "contract")
    same_next = contract.eq(contract.shift(-1))
    m = {f: wide(features, f) for f in ("mid_open", "mid_close", "mid_1500", "mid_first30", "depth_head",
                                       "depth_tail", "flow", "flow_tail", "resid", "intraday_mid", "impact")}
    s_close, s_high, s_low = (wide(spot, k) for k in ("close", "high", "low"))
    basis = (m["mid_1500"] - s_close) / s_close
    close, high, low = (bars[k].reindex(index=dates, columns=list(SYMBOLS)) for k in ("closew", "highw", "loww"))
    width = high - low
    per = {
        "clv_futures": (close - low) / width.where(width > 0),
        "clv_spot": (s_close - s_low) / (s_high - s_low).where(s_high > s_low),
        "basis": basis,
        "d_basis_next": (basis.shift(-1) - basis).where(same_next),
        "next_spot": s_close.shift(-1) / s_close - 1,
        "next_futures_cc": (m["mid_1500"].shift(-1) / m["mid_1500"] - 1).where(same_next),
        "next_intraday": np.log(m["mid_close"].shift(-1) / m["mid_open"].shift(-1)),
        "next_first30": np.log(m["mid_first30"].shift(-1) / m["mid_open"].shift(-1)),
        "flow": m["flow"], "flow_tail": m["flow_tail"], "resid": m["resid"], "impact": m["impact"],
        "depth_recovery": m["depth_head"].shift(-1) / m["depth_tail"],
        "basis_dev20": basis - basis.shift(1).rolling(20, min_periods=20).mean(),
    }
    out = pd.DataFrame({k: v.mean(axis=1, skipna=False) for k, v in per.items()}, index=dates)
    out["next_rest"] = out["next_intraday"] - out["next_first30"]
    impact_median = out["impact"].shift(1).rolling(IMPACT_WINDOW, min_periods=IMPACT_WINDOW // 2).median()
    out["high_impact"] = (out["impact"] > impact_median).astype(float).where(impact_median.notna())
    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------
# 检验
# --------------------------------------------------------------------------
def fit(y: pd.Series, xs: list[pd.Series]) -> tuple[float, float, float, int]:
    frame = pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 60:
        return np.nan, np.nan, np.nan, len(frame)
    b, t = ols_hac(frame.iloc[:, 0].to_numpy(), [frame.iloc[:, i].to_numpy() for i in range(1, frame.shape[1])], LAGS)
    return b, t, abs(b / t) if t else np.nan, len(frame)


def verdict(value, t, expected) -> str:
    if pd.isna(t):
        return "不确定"
    right = value < 0 if expected == "<0" else value > 0
    if abs(t) >= 2:
        return "支持" if right else "否定"
    return "不确定"


def row(tag, label, value, t, se, n, expected) -> dict:
    return {"预测": tag, "内容": label, "系数": value, "NW_t": t, "天数": n,
            "95%下限": value - 1.96 * se if pd.notna(se) else np.nan,
            "95%上限": value + 1.96 * se if pd.notna(se) else np.nan,
            "80%功效最小可检出": 2.8 * se if pd.notna(se) else np.nan,
            "预期": expected, "判断": verdict(value, t, expected)}


def yearly_signs(y, x) -> str:
    signs = []
    for year in sorted(set(y.dropna().index.year)):
        b = fit(y[y.index.year == year], [x[x.index.year == year]])[0]
        signs.append(f"{year}:{'+' if b > 0 else '−' if b < 0 else '?'}")
    return " ".join(signs)


def tests(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clv, y = p["clv_futures"], p["next_intraday"]

    def add(tag, label, target, xs, expected):
        b, t, se, n = fit(target, xs)
        rows.append(row(tag, label, b, t, se, n, expected))
        return b, t, se

    b_cc = add("A0", "期货次日收盘到收盘 ~ 期货收盘位置", p["next_futures_cc"], [clv], "<0")[0]
    add("A1", "次日基差变化 ~ 期货收盘位置", p["d_basis_next"], [clv], "<0")
    b_spot = add("A2", "现货次日收益 ~ 期货收盘位置", p["next_spot"], [clv], "<0")[0]
    rows.append({"预测": "A2b", "内容": "现货斜率 / 期货收盘到收盘斜率", "系数": b_spot / b_cc if b_cc else np.nan,
                 "预期": "远小于 1 支持套利限制", "判断": "见数值"})
    add("A3", "期货次日收盘到收盘 ~ 期货收盘位置 + 现货收盘位置：期货项", p["next_futures_cc"], [clv, p["clv_spot"]], "<0")
    add("A3b", "同上：现货收盘位置项", p["next_futures_cc"], [p["clv_spot"], clv], "<0")
    add("A3c", "现货次日收益 ~ 现货收盘位置", p["next_spot"], [p["clv_spot"]], "<0")
    low = (clv <= clv.quantile(0.2)).astype(float).where(clv.notna())
    add("A4", "收盘基差偏离 20 日均值 ~ 1{低位收盘}", p["basis_dev20"], [low], "<0")

    add("B0", "次日盘中 ~ 期货收盘位置", y, [clv], "<0")
    add("B1", "次日盘中 ~ 流量部分 F", y, [p["flow"]], "<0")
    bf, tf, sef = add("B2", "次日盘中 ~ F + R：F 项", y, [p["flow"], p["resid"]], "<0")
    br, tr, ser = add("O1", "次日盘中 ~ F + R：R 项（非流量部分）", y, [p["resid"], p["flow"]], "<0")
    bd, td, sed, nd = fit(y, [p["flow"], p["flow"] + p["resid"]])          # F 系数 = F 与 R 的系数差
    rows.append(row("B2b", "F 系数 − R 系数（<0 表示流量部分回复更多）", bd, td, sed, nd, "<0"))
    add("B2c", "次日盘中 ~ 尾盘流量部分", y, [p["flow_tail"]], "<0")
    add("B2d", "次日盘中 ~ 收盘位置 + F + R：收盘位置项", y, [clv, p["flow"], p["resid"]], "<0")
    b30 = add("B3", "次日开盘后 30 分钟 ~ 期货收盘位置", p["next_first30"], [clv], "<0")[0]
    brest = add("B3b", "次日其余时段 ~ 期货收盘位置", p["next_rest"], [clv], "<0")[0]
    rows.append({"预测": "B3c", "内容": "开盘后 30 分钟占次日盘中斜率的比例", "系数": b30 / (b30 + brest) if (b30 + brest) else np.nan,
                 "预期": "> 0.5", "判断": "支持" if b30 / (b30 + brest) > 0.5 else "不支持"})
    hi = p["high_impact"]
    add("B4", "次日盘中 ~ 收盘位置 × 高冲击日（交乘）", y, [clv * hi, clv, hi], "<0")
    add("B5", "次日深度恢复 ~ 1{低位收盘}", p["depth_recovery"], [low], ">0")
    lows = low.eq(1)
    corr = p.loc[lows, ["depth_recovery", "next_intraday"]].dropna()
    rows.append({"预测": "B5b", "内容": "低位收盘日：深度恢复与次日盘中收益的相关", "系数": float(corr.corr().iloc[0, 1]),
                 "天数": len(corr), "预期": ">0", "判断": "见数值"})
    out = pd.DataFrame(rows)
    out.loc[out["预测"].isin(["A0", "A1", "A2", "B0", "B1", "B3", "O1"]), "逐年符号"] = [
        yearly_signs(target, x) for target, x in (
            (p["next_futures_cc"], clv), (p["d_basis_next"], clv), (p["next_spot"], clv), (y, clv),
            (y, p["flow"]), (p["next_first30"], clv), (y, p["resid"]))]
    return out


def post_hoc(p: pd.DataFrame, features: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """看过预登记结果后加的拆分（登记文件 post_hoc 一节）：基差回归发生在隔夜还是次日盘中，现货自己怎么走。"""
    def wide(frame, field):
        return frame.pivot(index="trading_date", columns="symbol", values=field).reindex(index=p.index, columns=list(SYMBOLS))

    contract = wide(features, "contract")
    same_next = contract.eq(contract.shift(-1))
    s_open, s_close = wide(spot, "open"), wide(spot, "close")
    b_close = (wide(features, "mid_1500") - s_close) / s_close
    b_open = (wide(features, "mid_open") - s_open) / s_open
    series = {
        "基差 收盘→次日开盘": (b_open.shift(-1) - b_close).where(same_next),
        "基差 次日开盘→次日收盘": b_close.shift(-1) - b_open.shift(-1),
        "现货 收盘→次日开盘": s_open.shift(-1) / s_close - 1,
        "现货 次日开盘→收盘": s_close.shift(-1) / s_open.shift(-1) - 1,
    }
    series = {k: v.mean(axis=1, skipna=False) for k, v in series.items()}
    series["期货 次日盘中（中间价）"] = p["next_intraday"]
    series["期货 次日开盘后30分钟"] = p["next_first30"]
    rows = []
    for label, start in (("全样本", p.index.min()), ("2016+", SUB_START)):
        m = p.index >= start
        for xname in ("clv_futures", "clv_spot"):
            x = p.loc[m, xname]
            q5 = pd.qcut(x, 5, labels=False) + 1
            for name, y in series.items():
                b, t, se, n = fit(y[m], [x])
                rows.append({"样本": label, "预测变量": xname, "结果": name, "斜率": b, "NW_t": t, "天数": n,
                             **{f"Q{k}均值bp": float(y[m].groupby(q5).mean().get(k, np.nan) * 1e4) for k in range(1, 6)}})
    return pd.DataFrame(rows)


def run() -> None:
    protocol = load_protocol()
    dates, bars, *_ = I._load_index_inputs()
    dates = dates[(dates >= START) & (dates <= END)]
    C.assert_no_holdout_dates(dates, "股指机制日期")
    spot = pd.read_csv(SPOT_PATH, parse_dates=["date"]).rename(columns={"date": "trading_date"})
    C.assert_no_holdout_dates(spot["trading_date"], "现货指数日线")
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "tick_features.csv"
    if cache.is_file():
        features = pd.read_csv(cache, parse_dates=["trading_date"])
    else:
        features = build_tick_features()
        features.to_csv(cache, index=False, encoding="utf-8-sig")
    C.assert_no_holdout_dates(features["trading_date"], "L1 逐笔特征")
    coverage = features.groupby("symbol")["trading_date"].agg(["min", "max", "count"])
    panel = market_panel(features, spot, bars, dates)
    full = tests(panel)
    sub = tests(panel[panel.index >= SUB_START])
    full.insert(0, "样本", f"{dates.min().date()}..{dates.max().date()}")
    sub.insert(0, "样本", f"{SUB_START.date()}..{dates.max().date()}")
    table = pd.concat([full, sub], ignore_index=True)
    table.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    extra = post_hoc(panel, features, spot)
    extra.to_csv(OUT / "post_hoc_split.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(OUT / "daily_panel.csv", encoding="utf-8-sig")
    desc = {
        "逐笔覆盖": {k: {"min": str(v["min"].date()), "max": str(v["max"].date()), "days": int(v["count"])}
                    for k, v in coverage.iterrows()},
        "lambda 有值的品种日占比": float(features["lambda"].notna().mean()),
        "收盘基差均值": float(panel["basis"].mean()), "收盘基差中位": float(panel["basis"].median()),
        "期货收盘位置与现货收盘位置相关": float(panel["clv_futures"].corr(panel["clv_spot"])),
        "F 与当日收益相关": float(panel["flow"].corr(panel["flow"] + panel["resid"])),
        "F 与收盘位置相关": float(panel["flow"].corr(panel["clv_futures"])),
        "R 与收盘位置相关": float(panel["resid"].corr(panel["clv_futures"])),
        "高冲击日占比": float(panel["high_impact"].mean()),
    }
    (OUT / "protocol.json").write_text(json.dumps({"protocol": protocol, "descriptive": desc},
                                                  ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 44)
    print(table.drop(columns=["逐年符号"]).round(5).to_string(index=False))
    print(full[["预测", "逐年符号"]].dropna().to_string(index=False))
    print(extra.round(5).to_string(index=False))
    print(json.dumps(desc, ensure_ascii=False, indent=1, default=float))
    print(f"结果：{OUT}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run()
