"""股指收盘位置反转：前 20 名会员持仓和持仓量变化，区分套保集中加空和多头被迫平仓。只用股指期货研究期数据。

    python -m research.index_positioning_mechanism
    python -m research.index_positioning_mechanism --unabsorbed   # 没被大会员接住的卖压（登记在 index_unabsorbed_pressure.json）

规则在 config/research/index_positioning_mechanism.json，先登记后运行。
结果在 runs/index/index_positioning_mechanism/。不读 2022 及以后。
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from research import index_market_clock as I
from research.time_factor_screen import ols_hac
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_positioning_mechanism.json"
UNABSORBED_PATH = C.CONFIG_DIR / "research" / "index_unabsorbed_pressure.json"
UNABSORBED_OUT = C.RUNS_DIR / "index" / "index_unabsorbed_pressure"
OUT = C.RUNS_DIR / "index" / "index_positioning_mechanism"
EXTERNAL = C.INDEX_DATA_ROOT / "external_rqdata"
SYMBOLS = ("IF", "IH", "IC")
START, END = pd.Timestamp("2015-04-16"), pd.Timestamp(C.RESEARCH_END)
SUB_START = pd.Timestamp("2016-01-04")
LAGS = 5
STATE_WINDOW = 250


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "declared_test_not_a_strategy":
        raise ValueError("检验登记必须声明这不是策略")
    if "不读 2022" not in protocol.get("sample", "") or "不读 2026" not in protocol.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    return protocol


def above_own_median(series: pd.Series, window: int = STATE_WINDOW) -> pd.Series:
    """高于自身过去 window 天（不含当天，至少一半）的中位数为 1，否则 0；历史不足为缺失。"""
    median = series.shift(1).rolling(window, min_periods=window // 2).median()
    return (series > median).astype(float).where(median.notna() & series.notna())


def positioning_panel(rank: pd.DataFrame, contract_daily: dict, dates: pd.DatetimeIndex) -> dict:
    """逐品种：总持仓、持仓对数变化、前 20 名多空当日变化占前一日总持仓的比例。"""
    oi, d_oi, short_flow, long_flow = {}, {}, {}, {}
    for symbol in SYMBOLS:
        daily = contract_daily[symbol]
        total = daily.groupby("date")["open_interest"].sum().reindex(dates)
        prev = total.shift(1)
        oi[symbol] = total
        d_oi[symbol] = np.log(total / prev)
        part = rank[rank["symbol"] == symbol]
        flows = part.groupby(["side", "trading_date"])["volume_change"].sum().unstack("side")
        flows = flows.reindex(dates)
        short_flow[symbol] = flows["short"] / prev
        long_flow[symbol] = flows["long"] / prev
    frame = lambda d: pd.DataFrame(d, index=dates)[list(SYMBOLS)]   # noqa: E731
    return {"oi": frame(oi), "d_oi": frame(d_oi), "short_flow": frame(short_flow), "long_flow": frame(long_flow)}


def expiry_mask(dates: pd.DatetimeIndex, maturity: pd.Series) -> pd.Series:
    """当天或下一交易日是股指合约到期日为真。"""
    expiries = set(pd.DatetimeIndex(maturity).normalize())
    today = pd.Series(dates.isin(list(expiries)), index=dates)
    return today | today.shift(-1, fill_value=False)


def fit(y: pd.Series, xs: list[pd.Series]):
    frame = pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 60:
        return np.nan, np.nan, np.nan, len(frame)
    b, t = ols_hac(frame.iloc[:, 0].to_numpy(), [frame.iloc[:, i].to_numpy() for i in range(1, frame.shape[1])], LAGS)
    return b, t, abs(b / t) if t else np.nan, len(frame)


def row(tag, label, y, xs, expected, scale=1.0):
    b, t, se, n = fit(y, xs)
    right = (b < 0) if expected == "<0" else (b > 0)
    verdict = "不确定" if pd.isna(t) or abs(t) < 2 else ("支持" if right else "否定")
    years = []
    for year in sorted(set(y.dropna().index.year)):
        m = y.index.year == year
        by = fit(y[m], [x[m] for x in xs])[0]
        years.append("+" if by > 0 else "−" if by < 0 else "?")
    return {"预测": tag, "内容": label, "系数": b * scale, "NW_t": t, "天数": n,
            "95%下限": (b - 1.96 * se) * scale if pd.notna(se) else np.nan,
            "95%上限": (b + 1.96 * se) * scale if pd.notna(se) else np.nan,
            "80%功效最小可检出": 2.8 * se * scale if pd.notna(se) else np.nan,
            "预期": expected, "判断": verdict, "逐年符号": "".join(years)}


def tests(p: pd.DataFrame) -> pd.DataFrame:
    clv, y = p["clv"], p["next_intraday"]
    hi_net_short = above_own_median(-p["net_flow"])          # 大会员当天净加空高于自身中位
    oi_up = (p["d_oi"] > 0).astype(float).where(p["d_oi"].notna())
    oi_down = 1.0 - oi_up
    rows = [
        row("基准", "次日盘中 ~ 收盘位置（bp/单位）", y, [clv], "<0", 1e4),
        row("基准", "账本下一段 ~ 收盘位置（bp/单位）", p["fwd"], [clv], "<0", 1e4),
        row("H1", "当天前20空头变化/持仓 ~ 收盘位置（bp）", p["short_flow"], [clv], "<0", 1e4),
        row("H1", "当天前20净多变化/持仓 ~ 收盘位置（bp）", p["net_flow"], [clv], ">0", 1e4),
        row("H2", "次日前20空头变化 ~ 当天收盘位置（bp）", p["short_flow_next"], [clv], ">0", 1e4),
        row("H2", "次日前20净多变化 ~ 当天收盘位置（bp）", p["net_flow_next"], [clv], "<0", 1e4),
        row("H3", "次日盘中 ~ 净多变化 + 收盘位置：净多变化项", y, [p["net_flow"], clv], "<0"),
        row("H3", "次日盘中 ~ 净多变化 + 收盘位置：收盘位置项（bp）", y, [clv, p["net_flow"]], "<0", 1e4),
        row("H3", "次日盘中 ~ 收盘位置 × 1{大会员净加空高}（交乘，bp）", y, [clv * hi_net_short, clv, hi_net_short], "<0", 1e4),
        row("H4", "次日盘中 ~ 收盘位置 × 1{增仓}（交乘，bp）", y, [clv * oi_up, clv, oi_up], "<0", 1e4),
        row("L1", "次日盘中 ~ 收盘位置 × 1{减仓}（交乘，bp）", y, [clv * oi_down, clv, oi_down], "<0", 1e4),
        row("L2", "当天前20多头变化/持仓 ~ 收盘位置（bp）", p["long_flow"], [clv], ">0", 1e4),
        row("描述", "当天持仓对数变化 ~ 收盘位置（bp）", p["d_oi"], [clv], "<0", 1e4),
    ]
    for state, mask in (("增仓日", oi_up == 1), ("减仓日", oi_down == 1),
                        ("大会员净加空高", hi_net_short == 1), ("大会员净加空低", hi_net_short == 0)):
        rows.append(row("分组", f"{state}：次日盘中 ~ 收盘位置（bp）", y[mask], [clv[mask]], "<0", 1e4))
    return pd.DataFrame(rows)


def unabsorbed_terms(clv: pd.Series, absorption: pd.Series) -> pd.DataFrame:
    """P = max(0, 0.5 − clv)；U = P × 1{A ≤ 0}（大会员也在卖），B = P × 1{A > 0}（大会员在接）。"""
    pressure = (0.5 - clv).clip(lower=0.0)
    taking = absorption > 0
    valid = clv.notna() & absorption.notna()
    return pd.DataFrame({"P": pressure, "U": pressure.where(~taking, 0.0), "B": pressure.where(taking, 0.0)}).where(valid)


def unabsorbed_tests(p: pd.DataFrame) -> pd.DataFrame:
    terms = unabsorbed_terms(p["clv"], p["net_flow"])
    rows = []
    for outcome, label in (("next_intraday", "次日盘中"), ("fwd", "账本下一段")):
        y = p[outcome]
        rows += [
            row("U1", f"{label} ~ U（bp/单位）", y, [terms["U"]], ">0", 1e4),
            row("U2", f"{label} ~ U + B：U 项（bp）", y, [terms["U"], terms["B"]], ">0", 1e4),
            row("U2", f"{label} ~ U + B：B 项（bp）", y, [terms["B"], terms["U"]], ">0", 1e4),
            row("U2b", f"{label}：U 系数 − B 系数（bp）", y, [terms["U"], terms["U"] + terms["B"]], ">0", 1e4),
            row("U3", f"{label} ~ U + clv：U 项（bp）", y, [terms["U"], p["clv"]], ">0", 1e4),
            row("U3", f"{label} ~ U + clv：clv 项（bp）", y, [p["clv"], terms["U"]], "<0", 1e4),
            row("参照", f"{label} ~ P（bp）", y, [terms["P"]], ">0", 1e4),
        ]
    return pd.DataFrame(rows)


def run(unabsorbed: bool = False) -> None:
    protocol = load_protocol()
    if unabsorbed:
        extra = json.loads(UNABSORBED_PATH.read_text(encoding="utf-8"))
        if "不读 2022" not in extra.get("sample", "") or "不读 2026" not in extra.get("sample", ""):
            raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    dates, bars, *_ = I._load_index_inputs()
    dates = dates[(dates >= START) & (dates <= END)]
    C.assert_no_holdout_dates(dates, "股指持仓检验日期")
    rank = pd.read_csv(EXTERNAL / "member_rank.csv", parse_dates=["trading_date"])
    C.assert_no_holdout_dates(rank["trading_date"], "会员持仓")
    contract_daily = {}
    for symbol in SYMBOLS:
        daily = pd.read_pickle(EXTERNAL / f"{symbol}_contract_daily.pkl")
        daily["date"] = pd.to_datetime(daily["date"])
        C.assert_no_holdout_dates(daily["date"], f"{symbol} 合约日线")
        contract_daily[symbol] = daily
    maturity = pd.read_csv(EXTERNAL / "contract_maturity.csv", parse_dates=["maturity_date"])["maturity_date"]
    pos = positioning_panel(rank, contract_daily, dates)

    close, high, low = (bars[k].reindex(index=dates, columns=list(SYMBOLS)) for k in ("closew", "highw", "loww"))
    width = high - low
    o, ow, cw = (bars[k].reindex(index=dates, columns=list(SYMBOLS)) for k in ("open", "openw", "closew"))
    per = {
        "clv": (close - low) / width.where(width > 0),
        "next_intraday": ((cw - ow) / o.where(o > 0)).shift(-1),
        "fwd": ((ow.shift(-1) - ow) / o.where(o > 0)).shift(-1),
        "d_oi": pos["d_oi"], "short_flow": pos["short_flow"], "long_flow": pos["long_flow"],
        "net_flow": pos["long_flow"] - pos["short_flow"],
    }
    panel = pd.DataFrame({k: v.mean(axis=1, skipna=False) for k, v in per.items()}, index=dates)
    panel["short_flow_next"] = panel["short_flow"].shift(-1)
    panel["net_flow_next"] = panel["net_flow"].shift(-1)
    excluded = expiry_mask(dates, maturity)
    panel = panel[~excluded.to_numpy()].replace([np.inf, -np.inf], np.nan)

    if unabsorbed:
        full, sub = unabsorbed_tests(panel), unabsorbed_tests(panel[panel.index >= SUB_START])
        full.insert(0, "样本", f"{panel.index.min().date()}..{panel.index.max().date()}")
        sub.insert(0, "样本", f"{SUB_START.date()}..{panel.index.max().date()}")
        table = pd.concat([full, sub], ignore_index=True)
        UNABSORBED_OUT.mkdir(parents=True, exist_ok=True)
        table.to_csv(UNABSORBED_OUT / "predictions.csv", index=False, encoding="utf-8-sig")
        terms = unabsorbed_terms(panel["clv"], panel["net_flow"])
        desc = {"收在中点以下的日子": int((terms["P"] > 0).sum()),
                "其中大会员也在卖（U>0）": int((terms["U"] > 0).sum()),
                "其中大会员在接（B>0）": int((terms["B"] > 0).sum())}
        (UNABSORBED_OUT / "protocol.json").write_text(json.dumps({"protocol": extra, "descriptive": desc},
                                                                ensure_ascii=False, indent=2), encoding="utf-8")
        pd.set_option("display.width", 260)
        pd.set_option("display.max_colwidth", 40)
        print(table[["样本", "预测", "内容", "系数", "NW_t", "天数", "95%下限", "95%上限", "判断", "逐年符号"]]
              .round(3).to_string(index=False))
        print(desc)
        return
    full = tests(panel)
    sub = tests(panel[panel.index >= SUB_START])
    full.insert(0, "样本", f"{panel.index.min().date()}..{panel.index.max().date()}")
    sub.insert(0, "样本", f"{SUB_START.date()}..{panel.index.max().date()}")
    table = pd.concat([full, sub], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(OUT / "daily_panel.csv", encoding="utf-8-sig")
    top20_share = {s: float((rank[(rank.symbol == s) & (rank.side == "short")].groupby("trading_date")["volume"].sum()
                             / contract_daily[s].groupby("date")["open_interest"].sum()).median()) for s in SYMBOLS}
    desc = {"排除的到期相关日数": int(excluded.sum()), "可用天数": len(panel),
            "前20空头占总持仓中位": top20_share,
            "收盘位置与当天持仓变化相关": float(panel["clv"].corr(panel["d_oi"])),
            "收盘位置与前20净多变化相关": float(panel["clv"].corr(panel["net_flow"]))}
    (OUT / "protocol.json").write_text(json.dumps({"protocol": protocol, "descriptive": desc},
                                                  ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 40)
    print(table[["样本", "预测", "内容", "系数", "NW_t", "天数", "95%下限", "95%上限", "判断", "逐年符号"]]
          .round(3).to_string(index=False))
    print(json.dumps(desc, ensure_ascii=False, indent=1, default=float))
    print(f"结果：{OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="股指会员持仓与收盘位置反转")
    parser.add_argument("--unabsorbed", action="store_true", help="没被大会员接住的卖压检验")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run(unabsorbed=parser.parse_args().unabsorbed)
