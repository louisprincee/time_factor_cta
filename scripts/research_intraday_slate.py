"""研究期走步：日内 / 隔夜 / 多日持仓的多策略清单。只读 2016–2021，不读 2022 及以后。

口径：
- 书 = ``walk_forward.SLATE`` 里的一条规则（池级、多空对称、方向先验事先写定），每条只有一小格参数；
- 第 y 年（2019–2021）用 train_years(y) 的主口径池级 Sharpe 选格，再看 y 年，三年接起来是走步结果；
- 主口径 = 历史手续费 + 每边 1 tick；另报 slip2 / fee_x2 / fee_2026 / close_yday；
- 池级组合：当年条件池内品种风险缩放后等权；
- 冻结：用 2019–2021 选格（即预测 2022 的规则），连同组合书的权重写进 config/intraday_slate.json。
  组合书 = 走步主口径 Sharpe > 0 的书，按 2019–2021 主口径波动倒数定权。冻结后再跑
  scripts/validate_intraday_slate.py 一次性看 2022。

用法：python scripts/research_intraday_slate.py [--rebuild] [--no-freeze]
逐腿缓存在 RESEARCH_OUT_DIR/slate_legs.pkl。
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.data import universe as U  # noqa: E402
from tfcta.factors import external as EXT  # noqa: E402
from tfcta.research.backtest import costs  # noqa: E402
from tfcta.research.intraday import walk_forward as W  # noqa: E402
from tfcta.research.workflow import context  # noqa: E402

YEARS = tuple(range(2016, 2022))
TEST_YEARS = (2019, 2020, 2021)
FREEZE_TRAIN = tuple(W.train_years(2022))
ROBUST = ("gross", "main", "slip2", "fee_x2", "fee_2026", "close_yday")
CARRY = "carry_main_sub_annualized"
FROZEN = C.CONFIG_DIR / "intraday_slate.json"
COMBO = "slate_combo"


def tradable(names: list[str]) -> list[str]:
    known = set(pd.read_csv(costs.fee_history_path(), usecols=["symbol"])["symbol"])
    return [s for s in names if s in costs.MULTIPLIER and s in known]


def cache_path() -> Path:
    return C.RESEARCH_OUT_DIR / "slate_legs.pkl"


def carry_frame(symbols, partitions, end) -> pd.DataFrame:
    EXT.require_partitions(["carry_ms"], partitions)
    index = pd.bdate_range("2014-07-01", end)
    return EXT.load_wide(symbols, partitions, index)[CARRY].astype("float64")


def symbol_legs(symbol, minute, tick, carry: pd.DataFrame):
    col = carry[symbol] if symbol in carry and carry[symbol].notna().any() else None
    si = W.slate_input(symbol, minute, tick, col)
    return W.slate_legs(si), si.ctx, col is not None


def build(rebuild: bool):
    if cache_path().exists() and not rebuild:
        with open(cache_path(), "rb") as f:
            return pickle.load(f)
    raw = U.load_universe()
    universe = {y: tradable(raw.get(y, [])) for y in YEARS}
    symbols = sorted({s for y in YEARS for s in universe[y]})
    print(f"研究期池 2016–2021，加载 {len(symbols)} 个品种", flush=True)
    costs.load_fee_history(symbols)
    ticks = W.tick_by_year(symbols)
    carry = carry_frame(symbols, ("research",), "2021-12-31")
    frames, ctxs, no_carry = [], {}, []
    for symbol in symbols:
        if symbol not in ticks:
            print(f"  {symbol} 没有 tick 表，跳过", flush=True)
            continue
        legs, ctx, has_carry = symbol_legs(symbol, W.load_minutes(symbol, start="2014-07-01"),
                                           ticks[symbol], carry)
        frames.append(legs)
        ctxs[symbol] = ctx
        if not has_carry:
            no_carry.append(symbol)
    legs = pd.concat(frames, ignore_index=True)
    if pd.DatetimeIndex(legs["date"]).max() >= pd.Timestamp("2022-01-01"):
        raise RuntimeError("腿含 2022 及以后")
    universe = {y: [s for s in v if s in ctxs] for y, v in universe.items()}
    out = {"legs": legs, "ctxs": ctxs, "universe": universe, "no_carry": no_carry}
    C.RESEARCH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path(), "wb") as f:
        pickle.dump(out, f)
    return out


def walk_forward(series, name) -> tuple[dict[str, pd.Series], dict[int, str]]:
    """{情景: 2019–2021 走步序列}，{年: 选中的格}。"""
    variants = list(W.SLATE[name][2])
    picks, parts = {}, {sc: [] for sc in ROBUST}
    for y in TEST_YEARS:
        v = W.select_variant({v: series[(name, v, "main")] for v in variants
                              if (name, v, "main") in series}, W.train_years(y))
        picks[y] = v
        for sc in ROBUST:
            s = series[(name, v, sc)]
            parts[sc].append(s[s.index.year == y])
    return {sc: pd.concat(p).sort_index() for sc, p in parts.items()}, picks


def sharpe(s: pd.Series) -> float:
    return W.performance(s)["sharpe"]


def combo(books: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    frame = pd.concat({k: books[k] for k in weights}, axis=1).fillna(0.0)
    return frame.mul(pd.Series(weights)).sum(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description="多策略清单研究期走步（2016–2021）")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-freeze", action="store_true", help="只报告，不写冻结配置")
    args = ap.parse_args()
    t0 = time.time()
    data = build(args.rebuild)
    legs, ctxs, universe = data["legs"], data["ctxs"], data["universe"]
    print(f"腿 {len(legs):,} 行，{len(ctxs)} 个品种，无展期数据 {data['no_carry']}  "
          f"{time.time() - t0:.0f}s", flush=True)
    panel = W.slate_panel(ctxs)
    member = panel.members(universe)
    series = W.slate_series(legs, panel, member, ROBUST)

    grid_rows, wf_rows, wf_main, frozen = [], [], {}, {}
    for name, (kind, _fn, grid) in W.SLATE.items():
        if (name, next(iter(grid)), "main") not in series:
            continue
        for v in grid:
            part = legs[(legs["strategy"] == name) & (legs["variant"] == v)]
            row = {"strategy": name, "variant": v, "kind": kind,
                   "trades_per_sym_year": part["n"].sum() / max(part.groupby(
                       ["symbol", pd.DatetimeIndex(part["date"]).year]).ngroups, 1),
                   "cost_over_gross": (part["slip"].sum() + part["fee"].sum())
                   / part["gross"].sum() if part["gross"].sum() else np.nan}
            for sc in ("gross", "main"):
                s = series[(name, v, sc)]
                row[f"sr_{sc}_1621"] = sharpe(s)
                row.update({f"{sc}_{y}": sharpe(s[s.index.year == y]) for y in YEARS})
            grid_rows.append(row)
        wf, picks = walk_forward(series, name)
        wf_main[name] = wf["main"]
        row = {"strategy": name, "kind": kind,
               "picks": " ".join(f"{y}:{v}" for y, v in picks.items())}
        row.update(W.summarize(wf["main"]))
        row.update({f"sr_{sc}": sharpe(wf[sc]) for sc in ROBUST if sc != "main"})
        wf_rows.append(row)
        pick = W.select_variant({v: series[(name, v, "main")] for v in grid}, FREEZE_TRAIN)
        s = series[(name, pick, "main")]
        s = s[s.index.year.isin(FREEZE_TRAIN)]
        frozen[name] = {"kind": kind, "variant": pick, "params": grid[pick],
                        "train_sharpe": sharpe(s), "train_vol": W.performance(s)["ann_vol"],
                        "wf_sharpe": row["net_sharpe"]}

    grid = pd.DataFrame(grid_rows)
    wf = pd.DataFrame(wf_rows).sort_values("net_sharpe", ascending=False)
    chosen = [n for n in wf["strategy"] if np.isfinite(wf.set_index("strategy").loc[n, "net_sharpe"])
              and wf.set_index("strategy").loc[n, "net_sharpe"] > 0]
    inv = {n: 1.0 / frozen[n]["train_vol"] for n in chosen if frozen[n]["train_vol"] > 0}
    weights = {n: v / sum(inv.values()) for n, v in inv.items()}
    if weights:
        cs = combo(wf_main, weights)
        row = {"strategy": COMBO, "kind": "combo", "picks": f"{len(weights)} 本"}
        row.update(W.summarize(cs))
        wf = pd.concat([wf, pd.DataFrame([row])], ignore_index=True)
    corr = pd.concat(wf_main, axis=1).fillna(0.0).corr()

    run = context.run_dir("intraday_slate_research")
    grid.to_csv(run / "grid.csv", index=False, encoding="utf-8-sig")
    wf.to_csv(run / "walk_forward.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(run / "correlation.csv", encoding="utf-8-sig")
    pd.concat(wf_main, axis=1).to_csv(run / "wf_daily_main.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["strategy", "picks", "net_sharpe", "net_ann", "net_mdd", "sr_2019", "sr_2020",
            "sr_2021", "sr_gross", "sr_slip2", "sr_fee_x2", "sr_fee_2026", "sr_close_yday"]
    print("\n走步 2019–2021（主口径，按年选格）")
    print(wf[[c for c in cols if c in wf]].round(2).to_string(index=False))
    print("\n参数格 2016–2021（样本内，仅供参考）")
    print(grid[["strategy", "variant", "trades_per_sym_year", "cost_over_gross", "sr_gross_1621",
                "sr_main_1621"] + [f"main_{y}" for y in YEARS]].round(2).to_string(index=False))
    print(f"\n留痕 {run}")

    if args.no_freeze:
        return 0
    if FROZEN.exists():
        print(f"{FROZEN} 已存在，冻结后不再改写（2022 可能已按它检验）。")
        return 0
    context.dump_json(FROZEN, {
        "train": list(FREEZE_TRAIN), "test": C.VALIDATION_YEAR, "scenario": "main",
        "rule": "每本 2022 净 Sharpe >= 0.5 且净年化 > 0 才读 2023–2025",
        "books": frozen,
        "combo": {"name": COMBO, "rule": "走步主口径 Sharpe > 0，按 2019–2021 波动倒数定权",
                  "weights": weights},
        "research_run": str(run), "frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    })
    print(f"冻结 {FROZEN}：{len(frozen)} 本 + 组合书（{len(weights)} 本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
