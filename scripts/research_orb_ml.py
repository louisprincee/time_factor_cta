"""研究期走步：开盘区间突破 + 元标签模型。只读 2016–2021，不读 2022 及以后。

口径与之前的 ridge 表一致：
- 候选：``orb30|eod``（交易日第一根起 30 根，含夜盘）与 ``orb30day|eod``（日盘第一根起 30 根），
  单边撮合，尾盘平；主口径 = 历史手续费 + 每边 1 tick；
- 预测第 y 年只用此前三年（不早于 2016）的池内交易训练，预测值 > 0 才做；
- 池级组合：当年条件池内品种风险缩放后等权。

网格：特征组 × 模型（岭回归各档、Huber、逻辑回归、HGB、随机森林、ExtraTrees）。
网格里 2019–2021 最好的那格有选择偏差，所以另做嵌套选择：预测 T 年时只按 2018..T-1
的走步结果挑配置，再看 T 年，三年接起来才是"事先按规则选"能拿到的结果。

用法：python scripts/research_orb_ml.py [--models ...] [--groups ...]
候选交易与特征缓存在 RESEARCH_OUT_DIR/orb_candidates.pkl，加 --rebuild 重建。
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.data import universe as U  # noqa: E402
from tfcta.research.backtest import costs  # noqa: E402
from tfcta.research.intraday import walk_forward as W  # noqa: E402
from tfcta.research.workflow import context, history  # noqa: E402

BASES = ("orb30|eod", "orb30day|eod")
YEARS = tuple(range(2016, 2022))
VAL_YEARS = (2018, 2019, 2020, 2021)     # 有走步预测的年份（2018 只用于嵌套选择）
TEST_YEARS = (2019, 2020, 2021)
MIN_TRAIN = 1000
ROBUST = ("gross", "main", "slip2", "fee_x2", "fee_2026", "close_yday")


def tradable(names: list[str]) -> list[str]:
    known = set(pd.read_csv(costs.fee_history_path(), usecols=["symbol"])["symbol"])
    return [s for s in names if s in costs.MULTIPLIER and s in known]


def cache_path() -> Path:
    return C.RESEARCH_OUT_DIR / "orb_candidates.pkl"


def build(rebuild: bool):
    if cache_path().exists() and not rebuild:
        with open(cache_path(), "rb") as f:
            return pickle.load(f)
    universe = U.load_universe()
    for year in YEARS:
        universe[year] = tradable(universe.get(year, []))
    symbols = sorted({s for year in YEARS for s in universe[year]})
    print(f"研究期池 2016–2021，加载 {len(symbols)} 个品种", flush=True)
    hist = history.load_history(symbols, include_validation=False)
    loaded = list(hist.bars)
    for year in YEARS:
        universe[year] = [s for s in universe[year] if s in loaded]
    sig = history.signal_set(hist, universe, ("research",))
    factors = {n: (sig.signed if n in sig.signed else sig.unsigned)[n] for n in W.FACTORS}
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
    feats = W.attach_features(trades, ctxs, factors)
    if pd.DatetimeIndex(feats["date"]).max() >= pd.Timestamp("2022-01-01"):
        raise RuntimeError("候选交易含 2022 及以后")
    out = {"feats": feats, "ctxs": ctxs, "universe": {y: universe[y] for y in YEARS}}
    C.ensure_dirs()
    with open(cache_path(), "wb") as f:
        pickle.dump(out, f)
    return out


class Book:
    """一个开盘定义下的全部候选交易、池子与各情景净收益。"""

    def __init__(self, feats, ctxs, universe, base):
        self.sub = feats[feats["base"] == base].reset_index(drop=True)
        self.panel = W.Panel.build(self.sub, ctxs)
        self.member = self.panel.members(universe)
        self.nets = {s: W.scenario_net(self.sub, s) for s in ROBUST}
        self.y = W.target(self.nets["main"], self.sub["atr_pct"].to_numpy())
        self.year = pd.DatetimeIndex(self.sub["date"]).year.to_numpy()
        self.on_member = self.member[self.panel.di, self.panel.si]

    def series(self, keep, scenario="main") -> pd.Series:
        return self.panel.pool(self.panel.book(self.nets[scenario], keep), self.member)

    def walk(self, groups, model) -> np.ndarray:
        X = W.ml_matrix(self.sub, groups)
        pred = np.full(len(self.sub), np.nan)
        finite = np.isfinite(self.y)
        for v in VAL_YEARS:
            train = finite & self.on_member & np.isin(self.year, W.train_years(v))
            hold = self.on_member & (self.year == v)
            if train.sum() > MIN_TRAIN and hold.any():
                pred[hold] = W.fit_predict(model, X, self.y, train, hold)[0]
        return pred


def sharpe(ret: pd.Series) -> float:
    return W.performance(ret)["sharpe"] if len(ret) > 20 else np.nan


def years_of(ret: pd.Series, years) -> pd.Series:
    return ret[ret.index.year.isin(list(years))]


def quintile_ic(book: Book, pred: np.ndarray) -> float:
    """每年测试集内按预测值分五档，档号与档内单笔净收益均值的秩相关（三年平均）。"""
    out = []
    for v in TEST_YEARS:
        m = book.on_member & (book.year == v) & np.isfinite(pred)
        if m.sum() < 100:
            continue
        q = pd.qcut(pred[m], 5, labels=False, duplicates="drop")
        means = pd.Series(book.nets["main"][m]).groupby(np.asarray(q)).mean()
        out.append(spearmanr(means.index, means.to_numpy())[0])
    return float(np.nanmean(out)) if out else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(W.MODELS))
    ap.add_argument("--groups", nargs="*", default=list(W.GROUP_SETS))
    ap.add_argument("--bases", nargs="*", default=list(BASES), choices=BASES)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    data = build(args.rebuild)
    print(f"候选 {len(data['feats']):,} 笔（两种开盘定义合计）  {time.time() - t0:.0f}s", flush=True)

    rows, yearly, preds = [], {}, {}
    for base in args.bases:
        book = Book(data["feats"], data["ctxs"], data["universe"], base)
        base_all = book.on_member & np.isin(book.year, TEST_YEARS)
        raw = years_of(book.series(base_all), TEST_YEARS)
        rows.append({"base": base, "groups": "-", "model": "全做", "take_rate": 1.0,
                     "sr_2018": np.nan, **W.summarize(raw)})
        for groups in args.groups:
            for model in args.models:
                t1 = time.time()
                pred = book.walk(groups, model)
                keep = np.isfinite(pred) & (pred > 0)
                ret = book.series(keep)
                key = (base, groups, model)
                yearly[key] = {v: sharpe(years_of(ret, [v])) for v in VAL_YEARS}
                preds[key] = pred
                test = book.on_member & np.isin(book.year, TEST_YEARS)
                row = {"base": base, "groups": groups, "model": model,
                       "take_rate": float((keep & test).sum() / max(test.sum(), 1)),
                       "sr_2018": yearly[key][2018], **W.summarize(years_of(ret, TEST_YEARS)),
                       "quintile_ic": quintile_ic(book, pred)}
                rows.append(row)
                print(f"{base:13} {groups:17} {model:9} SR {row['net_sharpe']:5.2f}  "
                      + " ".join(f"{v}:{yearly[key][v]:5.2f}" for v in VAL_YEARS)
                      + f"  取 {row['take_rate']:.0%}  {time.time() - t1:.0f}s", flush=True)
        # 候选的全部情景（网格跑完再算，只对这本书）
        for key in [k for k in preds if k[0] == base]:
            keep = np.isfinite(preds[key]) & (preds[key] > 0)
            for s in ROBUST:
                if s != "main":
                    r = next(r for r in rows if (r["base"], r["groups"], r["model"]) == key)
                    r[f"sr_{s}"] = sharpe(years_of(book.series(keep, s), TEST_YEARS))
        data.setdefault("books", {})[base] = book

    grid = pd.DataFrame(rows)

    # 嵌套选择：预测 T 年时按 2018..T-1 的逐年 Sharpe 均值挑配置
    def nested(keys, label):
        parts, picks = [], []
        for T in TEST_YEARS:
            prior = range(2018, T)
            score = {k: np.nanmean([yearly[k][v] for v in prior]) for k in keys}
            best = max(score, key=lambda k: -np.inf if np.isnan(score[k]) else score[k])
            book = data["books"][best[0]]
            keep = np.isfinite(preds[best]) & (preds[best] > 0)
            parts.append(years_of(book.series(keep), [T]))
            picks.append(f"{T}:{best[1]}/{best[2]}" + ("" if best[0] == BASES[0] else "(day)"))
        return {"rule": label, "picks": "; ".join(picks),
                **W.summarize(pd.concat(parts).sort_index())}

    keys = list(preds)
    nest = [nested(keys, "全部配置")]
    nest.append(nested([k for k in keys if k[0] == BASES[0]], "只 orb30"))
    for model in args.models:
        nest.append(nested([k for k in keys if k[2] == model and k[0] == BASES[0]],
                           f"orb30 · {model} · 选特征组"))
    for groups in args.groups:
        nest.append(nested([k for k in keys if k[1] == groups and k[0] == BASES[0]],
                           f"orb30 · {groups} · 选模型"))
    nest = pd.DataFrame(nest)

    run = context.run_dir("orb_ml_research")
    grid.to_csv(run / "grid.csv", index=False, encoding="utf-8-sig")
    nest.to_csv(run / "nested.csv", index=False, encoding="utf-8-sig")
    top = grid[grid["model"] != "全做"].sort_values("net_sharpe", ascending=False).head(12)
    daily = {}
    for _, r in top.iterrows():
        key = (r["base"], r["groups"], r["model"])
        keep = np.isfinite(preds[key]) & (preds[key] > 0)
        daily["/".join(key)] = years_of(data["books"][key[0]].series(keep), TEST_YEARS)
    pd.DataFrame(daily).to_csv(run / "daily_net_top.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 300)
    cols = ["base", "groups", "model", "net_sharpe", "net_ann", "sr_2018", "sr_2019", "sr_2020",
            "sr_2021", "take_rate", "quintile_ic", "sr_slip2", "sr_fee_x2", "sr_fee_2026"]
    print(grid[cols].round(3).to_string(index=False))
    print(nest[["rule", "net_sharpe", "net_ann", "sr_2019", "sr_2020", "sr_2021", "picks"]]
          .round(3).to_string(index=False))
    ml = grid[grid["model"] != "全做"]
    for base in args.bases:
        g = ml[ml["base"] == base]
        print(f"{base}: {len(g)} 格，净 Sharpe 中位 {g['net_sharpe'].median():.2f}，"
              f"> 0 占 {(g['net_sharpe'] > 0).mean():.0%}；三年都正 {(g['pos_years'] == 3).mean():.0%}")
        by_model = g.groupby("model")["net_sharpe"].median().sort_values(ascending=False)
        print("  按模型中位：" + "  ".join(f"{k} {v:.2f}" for k, v in by_model.items()))
        by_group = g.groupby("groups")["net_sharpe"].median().sort_values(ascending=False)
        print("  按特征组中位：" + "  ".join(f"{k} {v:.2f}" for k, v in by_group.items()))
    print(f"留痕 {run}  用时 {time.time() - t0:.0f}s")
    print("只使用了 2016–2021。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
