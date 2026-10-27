"""下载商品主力合约的期货 tick（L1：买一卖一价和量），只落盘每个交易日的收盘段汇总。

    python -m download.commodity.getTickL1Close                                  # 默认：RB、CU、M、TA、I，2016–2021
    python -m download.commodity.getTickL1Close --pool                           # 研究期品种池，2011–2021，每个品种只下它在池里的年份
    python -m download.commodity.getTickL1Close --pool --symbols A AG AL         # 池里的这几个品种，可以分几个终端并行

主力 = 米筐 get_dominant(rule=0)，与研究分片一致。每个合约日算日盘和最后 30 分钟（14:30–15:00）的相对价差、
买一卖一深度、Lee–Ready 方向的成交不平衡度，以及收盘价相对中间价的位置。原始 tick 不落盘。
可中断续跑：已经写过的交易日跳过。最早 2011-01-01，最晚 2021-12-31，不读 2022 及以后。
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.rq_auth import rqdata_credentials  # noqa: E402

SYMBOLS = ("RB", "CU", "M", "TA", "I")
START, END = date(2016, 1, 1), C.RESEARCH_END
OUT = C.COMMODITY_DATA / "external_rqdata" / "l1_close"
DAY_OPEN, LAST_WINDOW, DAY_CLOSE = "09:00:00", "14:30:00", "15:00:00.999"


def _flow_imbalance(frame: pd.DataFrame) -> float:
    """Lee–Ready：成交价高于前一笔中间价算买、低于算卖，等于中间价时用最近一次价格变动的方向。"""
    if frame.empty:
        return np.nan
    direction = np.sign(frame["last"] - frame["prev_mid"])
    tick = np.sign(frame["last"].diff()).replace(0, np.nan).ffill()
    direction = direction.where(direction != 0, tick).fillna(0.0)
    volume = frame["dvol"]
    total = volume.sum()
    return float((direction * volume).sum() / total) if total > 0 else np.nan


def l1_day_stats(ticks: pd.DataFrame, trading_date) -> dict:
    """一个合约日的 tick（含夜盘）→ 日盘和尾盘的流动性汇总。volume 是交易日内累计量。"""
    day = pd.Timestamp(trading_date).normalize()
    t = ticks.sort_values("datetime").copy()
    t["dvol"] = t["volume"].diff().fillna(t["volume"]).clip(lower=0)
    quote_ok = (t["a1"] > 0) & (t["b1"] > 0) & (t["a1"] >= t["b1"])
    t["mid"] = ((t["a1"] + t["b1"]) / 2).where(quote_ok)
    t["rel_spread"] = ((t["a1"] - t["b1"]) / t["mid"]).where(quote_ok)
    t["depth"] = (t["a1_v"] + t["b1_v"]).where(quote_ok)
    t["prev_mid"] = t["mid"].shift(1)
    stamp = pd.to_datetime(t["datetime"])
    day_session = (stamp >= pd.Timestamp(f"{day.date()} {DAY_OPEN}")) & (stamp <= pd.Timestamp(f"{day.date()} {DAY_CLOSE}"))
    last = day_session & (stamp >= pd.Timestamp(f"{day.date()} {LAST_WINDOW}"))
    d, tail = t[day_session], t[last]
    out = {"trading_date": day, "n_ticks": int(day_session.sum())}
    if d.empty:
        return out
    final = d.iloc[-1]
    final_quote = d[d["mid"].notna()]
    mid_final = float(final_quote["mid"].iloc[-1]) if len(final_quote) else np.nan
    half = float(final_quote["rel_spread"].iloc[-1]) / 2 if len(final_quote) else np.nan
    high, low = float(final["high"]), float(final["low"])
    width = high - low
    out.update({
        "spread_day": float(d["rel_spread"].mean()), "spread_last30": float(tail["rel_spread"].mean()),
        "depth_day": float(d["depth"].mean()), "depth_last30": float(tail["depth"].mean()),
        "imb_day": _flow_imbalance(d), "imb_last30": _flow_imbalance(tail),
        "close_last": float(final["last"]), "mid_final": mid_final,
        "close_vs_mid": (float(final["last"]) - mid_final) / mid_final if mid_final else np.nan,
        "half_spread_final": half,
        "clv_last": (float(final["last"]) - low) / width if width > 0 else np.nan,
        "clv_mid": (mid_final - low) / width if width > 0 else np.nan,
    })
    return out


def research_pool(first_year: int, last_year: int) -> dict[int, list[str]]:
    """研究期逐年品种池：2016 起用落盘的，更早的年份按同一门槛用 yearly_stats 重建（第 y 年只看第 y−1 年）。"""
    from tfcta.data import build_universe, load_universe

    stored = {int(y): sorted(v) for y, v in load_universe().items()}
    years = list(range(first_year, last_year + 1))
    early = [y for y in years if y < C.STUDY_START.year]
    universe = {y: stored[y] for y in years if y in stored}
    if early:
        stats = pd.read_csv(C.UNIVERSE_DIR / "yearly_stats.csv")
        rebuilt, _ = build_universe(stats, years=early)
        universe.update({y: rebuilt[y] for y in early})
    return universe


def wanted_days(dominant: pd.Series, symbol: str, universe: dict | None) -> pd.Series:
    """主力表里要下的日子：给了品种池就只留该品种在池里的年份。"""
    if universe is None:
        return dominant
    years = [y for y, members in universe.items() if symbol in members]
    return dominant[dominant.index.year.isin(years)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="商品主力合约 L1 收盘段汇总下载")
    parser.add_argument("--pool", action="store_true", help="按研究期品种池逐年选品种，默认区间改为 2011–2021")
    parser.add_argument("--symbols", nargs="*", default=None, help="只下这些品种（与 --pool 合用时取交集）")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD，不早于 2011-01-01")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD，不晚于 2021-12-31")
    args = parser.parse_args(argv)
    start = pd.Timestamp(args.start or (C.EARLY_START if args.pool else START)).date()
    end = pd.Timestamp(args.end or END).date()
    if start < C.EARLY_START or end > C.RESEARCH_END or start > end:
        raise C.HoldoutViolation(f"L1 只下 {C.EARLY_START}..{C.RESEARCH_END}，收到 {start}..{end}")
    universe = research_pool(start.year, end.year) if args.pool else None
    if universe is not None:
        names = sorted({s for members in universe.values() for s in members})
        if args.symbols:
            names = [s for s in names if s in {x.upper() for x in args.symbols}]
    else:
        names = [s.upper() for s in args.symbols] if args.symbols else list(SYMBOLS)
    return download(names, start, end, universe)


def download(symbols, start, end, universe) -> int:
    C.assert_no_holdout_dates(pd.DatetimeIndex([pd.Timestamp(end)]), "L1 下载截止日")
    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证；请检查 config/rqdata.env")
    import rqdatac as rq

    rq.init(*credentials)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(symbols)} 个品种，{start}..{end}", flush=True)
    for symbol in symbols:
        path = OUT / f"{symbol}.csv"
        done = pd.read_csv(path, parse_dates=["trading_date"]) if path.is_file() else pd.DataFrame()
        seen = set(done["trading_date"].dt.normalize()) if len(done) else set()
        dom = rq.futures.get_dominant(symbol, start, end, rule=0)
        if dom is None or dom.empty:
            print(f"{symbol}: 区间内没有主力表，跳过", flush=True)
            continue
        dom.index = pd.to_datetime(dom.index)
        C.assert_no_holdout_dates(dom.index, f"{symbol} 主力表")
        dom = wanted_days(dom, symbol, universe)
        rows, failures = [], []
        for day, contract in dom.items():
            if day in seen:
                continue
            try:
                ticks = rq.get_price(contract, start_date=day, end_date=day, frequency="tick")
                if ticks is None or ticks.empty:
                    failures.append(f"{day.date()} {contract} 空")
                    continue
                stats = l1_day_stats(ticks.reset_index(), day)
                rows.append({"symbol": symbol, "contract": contract, **stats})
            except Exception as exc:                       # 单日失败记下来继续，下次续跑补
                failures.append(f"{day.date()} {contract} {type(exc).__name__}")
            if len(rows) >= 40:
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                done.to_csv(path, index=False, encoding="utf-8-sig")
                rows = []
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{symbol}: {len(done)} 天，失败 {len(failures)}" + (f"：{failures[:5]}" if failures else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
