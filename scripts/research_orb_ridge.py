"""研究期走步：开盘区间突破 + 岭回归。只读 2016–2021，不读 2022 及以后。

事先写定，看到结果前不改：
- 模型 ridge10，特征是区间本身 + 动量 + 期限结构，预测值 > 0 才做；
- 开盘定义有两种：交易日第一根起的 30 根（含夜盘），以及日盘第一根起的 30 根；
- 预测值在每年测试集里分成五档，看越高的档之后净收益是否也越高。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta.data import universe as U  # noqa: E402
from tfcta.research.backtest import costs  # noqa: E402
from tfcta.research.intraday import walk_forward as W  # noqa: E402
from tfcta.research.workflow import context, history  # noqa: E402

BASES = ("orb30|eod", "orb30day|eod")
GROUPS = "core+trend+carry"
MODEL = "ridge10"
TEST_YEARS = (2019, 2020, 2021)


def tradable(names: list[str]) -> list[str]:
    known = set(pd.read_csv(costs.fee_history_path(), usecols=["symbol"])["symbol"])
    return [s for s in names if s in costs.MULTIPLIER and s in known]


def stitched(parts: list[pd.Series]) -> pd.Series:
    out = pd.concat(parts).sort_index()
    if out.index.max() >= pd.Timestamp("2022-01-01"):
        raise RuntimeError("研究期结果含 2022 及以后的日期")
    return out


def one_year(panel, member, nets, pred, test, year) -> pd.Series:
    keep = test & np.isfinite(pred) & (pred > 0)
    net = panel.pool(panel.book(nets, keep, panel.di, panel.si), member)
    return net[net.index.year == year]


def main() -> int:
    universe = U.load_universe()
    for year in range(2016, 2022):
        universe[year] = tradable(universe.get(year, []))
    symbols = sorted({s for year in range(2016, 2022) for s in universe[year]})
    print(f"研究期池 2016–2021，加载 {len(symbols)} 个品种")
    hist = history.load_history(symbols, include_validation=False)
    loaded = list(hist.bars)
    if hist.skipped:
        print("跳过 " + "; ".join(f"{s}" for s, _ in hist.skipped))
    for year in range(2016, 2022):
        universe[year] = [s for s in universe[year] if s in loaded]
    sig = history.signal_set(hist, universe, ("research",))
    factors = {}
    for name in W.FACTORS:
        src = sig.signed if name in sig.signed else sig.unsigned
        factors[name] = src[name]

    costs.load_fee_history(loaded)
    ticks = W.tick_by_year(loaded)
    frames, ctxs = [], {}
    for symbol in loaded:
        minute = W.load_minutes(symbol, start="2014-07-01")
        trades, ctx = W.orb_candidates(symbol, minute, ticks[symbol], bases=BASES)
        if len(trades):
            frames.append(trades)
        ctxs[symbol] = ctx
    trades = pd.concat(frames, ignore_index=True)
    print(f"单边候选 {len(trades):,} 笔")
    books, bins = [], []
    for base in BASES:
        part = trades[trades["base"] == base].reset_index(drop=True)
        panel = W.Panel.build(part, ctxs)
        member = panel.members({y: universe[y] for y in range(2016, 2022)})
        nets = W.scenario_net(part, "main")
        sub = W.attach_features(part, ctxs, factors)
        target = W.target(nets, sub["atr_pct"].to_numpy())
        year = pd.DatetimeIndex(sub["date"]).year.to_numpy()
        on_member = member[panel.di, panel.si]
        finite = np.isfinite(target)
        X = W.ml_matrix(sub, GROUPS)
        pred = np.full(len(sub), np.nan)
        test = np.zeros(len(sub), bool)
        parts, takes, tested = [], [], []
        for test_year in TEST_YEARS:
            train = finite & on_member & np.isin(year, W.train_years(test_year))
            hold = on_member & (year == test_year)
            test |= hold
            if train.sum() > 1000 and hold.any():
                got, _, _ = W.fit_predict(MODEL, X, target, train, hold)
                pred[hold] = got
            parts.append(one_year(panel, member, nets, pred, hold, test_year))
            takes.append(int((hold & (pred > 0)).sum()))
            tested.append(int(hold.sum()))
        net = stitched(parts)
        row = {"base": base, "groups": GROUPS, "model": MODEL,
               "n_test": int(sum(tested)), "n_taken": int(sum(takes)),
               "take_rate": float(sum(takes) / sum(tested)) if sum(tested) else np.nan}
        row.update(W.summarize(net))
        books.append(row)
        print(f"{base:14} 净 Sharpe {row['net_sharpe']:7.3f}  净年化 {row['net_ann']:7.2%}  "
              + "  ".join(f"{y}:{row.get(f'sr_{y}', np.nan):5.2f}" for y in TEST_YEARS))

        quintile = np.full(len(pred), -1, int)
        for test_year in TEST_YEARS:
            mask = test & (year == test_year) & np.isfinite(pred)
            if mask.sum() < 25:
                continue
            quintile[mask] = np.asarray(
                pd.qcut(pred[mask], 5, labels=False, duplicates="drop"), dtype=int)
        for k in range(5):
            chosen = quintile == k
            series = panel.pool(panel.book(nets, chosen, panel.di, panel.si), member)
            series = series[series.index.year.isin(TEST_YEARS)]
            summary = W.summarize(series) if len(series) else {}
            trade_net = nets[chosen]
            item = {"base": base, "quintile": k + 1, "n": int(chosen.sum()),
                    "mean_pred": float(pred[chosen].mean()) if chosen.any() else np.nan,
                    "mean_trade_net": float(trade_net.mean()) if chosen.any() else np.nan,
                    "net_sharpe": summary.get("net_sharpe", np.nan),
                    "net_ann": summary.get("net_ann", np.nan)}
            bins.append(item)
            print(f"  第 {k + 1} 档  笔数 {item['n']:5d}  预测 {item['mean_pred']:7.3f}  "
                  f"单笔净收益 {item['mean_trade_net']:8.5f}  组合 Sharpe {item['net_sharpe']:7.3f}")

    run = context.run_dir("orb_ridge_research")
    pd.DataFrame(books).to_csv(run / "anchors.csv", index=False)
    pd.DataFrame(bins).to_csv(run / "quintiles.csv", index=False)
    print(f"留痕 {run}")
    print("只使用了 2016–2021。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
