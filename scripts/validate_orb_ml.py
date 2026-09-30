"""一次性检验：开盘区间突破 + 元标签模型。每本书 2022 只测一次，过了才读 2023–2025。

事先写定，看到结果前不改：
- 只做 ``orb30|eod``：交易日第一根起 30 根（含夜盘），尾盘平；
- 特征组（``walk_forward.GROUP_SETS``）与模型（``walk_forward.MODELS``）由命令行给定，预测值 > 0 才做；
- 训练只用 2019–2021 的池内交易，标签不包含 2022 及以后；
- 主口径 = 历史手续费 + 每边 1 tick；
- 2022 净 Sharpe ≥ 0.5 且几何净年化 > 0 才读 2023–2025；
- 样本外沿用 2022 检验时存下的模型，不用 2022 及以后的标签重训。

一本书 = 模型 + 特征组，指纹由规格串算出，测过的书在台账里，脚本拒绝重测。

用法：
  python scripts/validate_orb_ml.py --model ridge10 --groups core          # 只测 2022
  python scripts/validate_orb_ml.py --model ridge10 --groups core --oos    # 2022 已通过后读 2023–2025
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.data import bars as B  # noqa: E402
from tfcta.data import sessions, shard_io, universe as U  # noqa: E402
from tfcta.factors import external as EXT  # noqa: E402
from tfcta.factors import intraday  # noqa: E402
from tfcta.research.backtest import costs  # noqa: E402
from tfcta.research.intraday import walk_forward as W  # noqa: E402
from tfcta.research.workflow import context, history, ledger  # noqa: E402

BASE = "orb30|eod"
TRAIN = (2019, 2020, 2021)
OOS_END = C.DEFAULT_OOS_END
MIN_SHARPE = 0.5
PARTITIONS_2022 = ("research", "validation_2022")
PARTITIONS_OOS = ("research", "validation_2022", "holdout_locked")
MODEL = GROUPS = SPEC = FACTORS = SIDE_SCOPE = None  # main() 按命令行设定
LONG_ONLY = False


def select(model: str, groups: str, long_only: bool = False) -> None:
    global MODEL, GROUPS, SPEC, FACTORS, SIDE_SCOPE, LONG_ONLY
    MODEL, GROUPS = model, groups
    LONG_ONLY = bool(long_only)
    SIDE_SCOPE = "long_only" if LONG_ONLY else "both_sides"
    FACTORS = f"ml:{MODEL}:{BASE}:{GROUPS}" + (":long_only" if LONG_ONLY else "")
    SPEC = f"{FACTORS}:pred>0:train2019-2021:test2022:hist+1tick"
    if LONG_ONLY:
        SPEC += ":side=long"


def fingerprint() -> str:
    return hashlib.sha256(SPEC.encode()).hexdigest()[:16]


def tradable(names: list[str]) -> list[str]:
    known = set(pd.read_csv(costs.fee_history_path(), usecols=["symbol"])["symbol"])
    return [s for s in names if s in costs.MULTIPLIER and s in known]


def minute_frame(symbol: str, oos_end=None):
    """研究期必读。验证期在有分片时读。样本外分片只在传入截止日期后才打开。"""
    if oos_end is not None and shard_io.find_shard(C.HOLDOUT_DIR, symbol) is not None:
        return B.load_minutes(symbol, include_validation=True, oos_end=oos_end)
    if shard_io.find_shard(C.VALIDATION_DIR, symbol) is not None:
        return B.load_minutes(symbol, include_validation=True)
    if shard_io.find_shard(C.RESEARCH_DIR, symbol) is not None:
        return B.load_minutes(symbol, include_validation=False)
    return None


def build(symbols: list[str], universe: dict, partitions, oos_end=None):
    """分钟、日频因子、ORB 候选。``oos_end`` 为空时不打开锁定分片。"""
    EXT.require_partitions(["carry_ms"], partitions)
    hist = history.History()
    frames, ctxs, skipped = [], {}, []
    costs.load_fee_history([s for s in symbols if s in costs.MULTIPLIER])
    for symbol in symbols:
        minute = minute_frame(symbol, oos_end)
        if minute is None:
            skipped.append(symbol)
            continue
        coords = sessions.add_intraday_coords(minute)
        hist.factors[symbol] = intraday.symbol_daily_factors(
            coords, C.IC_REFERENCE_LOOKBACK, C.IC_REFERENCE_PCT, with_coords=True)
        hist.bars[symbol] = B.daily_bars(minute)
        tick = W.tick_dict(pd.DataFrame(costs.tick_rows(
            symbol, minute["close"], minute["trading_date"])))[symbol]
        trades, ctx = W.orb_candidates(symbol, minute, tick, bases=(BASE,))
        if len(trades):
            frames.append(trades)
        ctxs[symbol] = ctx
        del minute, coords
    loaded = list(hist.bars)
    scoped = {y: [s for s in names if s in loaded] for y, names in universe.items()}
    sig = history.signal_set(hist, scoped, partitions)
    factors = {}
    for name in W.FACTORS:
        src = sig.signed if name in sig.signed else sig.unsigned
        factors[name] = src[name]
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trades, ctxs, factors, scoped, skipped


def prepare(trades, ctxs, factors):
    part = trades[trades["base"] == BASE].reset_index(drop=True)
    panel = W.Panel.build(part, ctxs)
    sub = W.attach_features(part, ctxs, factors)
    nets = W.scenario_net(part, "main")
    target = W.target(nets, sub["atr_pct"].to_numpy())
    year = pd.DatetimeIndex(sub["date"]).year.to_numpy()
    X = W.ml_matrix(sub, GROUPS)
    return part, panel, sub, nets, target, year, X


def assert_train_dates(dates, mask) -> None:
    used = pd.DatetimeIndex(dates)[mask]
    if len(used) and used.max() >= pd.Timestamp("2022-01-01"):
        raise RuntimeError(f"训练标签含 {used.max().date()}")


def digest(sub, X, mask, columns) -> tuple[int, str]:
    keys = sub.loc[mask, ["symbol", "date", "side"]].copy()
    keys["date"] = pd.DatetimeIndex(keys["date"]).strftime("%Y-%m-%d")
    order = np.lexsort((keys["side"].to_numpy(), keys["date"].to_numpy(),
                        keys["symbol"].astype(str).to_numpy()))
    values = X.loc[mask, columns].to_numpy(dtype="float64")[order]
    values = np.round(np.nan_to_num(values, nan=-999.0), 6)
    token = hashlib.sha256(values.tobytes()).hexdigest()
    return int(mask.sum()), token


def score(panel, member, nets, pred, year, years: tuple[int, ...]) -> pd.Series:
    keep = np.isin(year, years) & member[panel.di, panel.si] & np.isfinite(pred) & (pred > 0)
    series = panel.pool(panel.book(nets, keep, panel.di, panel.si), member)
    return series[series.index.year.isin(years)]


def passes(summary: dict) -> bool:
    sharpe, ann = summary.get("net_sharpe"), summary.get("net_ann")
    return bool(np.isfinite(sharpe) and np.isfinite(ann) and sharpe >= MIN_SHARPE and ann > 0)


def rank_ic(pred, target, mask) -> float:
    if mask.sum() < 20:
        return float("nan")
    a, b = pd.Series(pred[mask]), pd.Series(target[mask])
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return float("nan")
    return float(a.corr(b, method="spearman"))


def performance_table(net: pd.Series, pred, target, hold, n_symbols: int, period: str) -> pd.DataFrame:
    rows = []
    windows = [(period, net)]
    if net.index.year.nunique() > 1:
        windows += [(str(y), net[net.index.year == y]) for y in sorted(set(net.index.year))]
    for label, series in windows:
        if not len(series):
            continue
        row = {"period": label, "n_symbols": n_symbols}
        row.update(W.summarize(series))
        if label == period:
            row["rank_ic"] = rank_ic(pred, target, hold & np.isfinite(pred))
            row["n_test"] = int(hold.sum())
            row["n_taken"] = int((hold & np.isfinite(pred) & (pred > 0)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def ledger_row(fp, run, summary, n_symbols, passed, note, **extra) -> dict:
    return {
        "fingerprint": fp,
        "book_key": fp,
        "case": "全部",
        "factors": FACTORS,
        "config": {
            "fee_rate": None, "slippage_ticks": 1.0, "tranches": 0, "vol_target": 0.0,
            "execution": "ml_meta_label", "fee_schedule": "hist", "train": "2019-2021",
            "groups": GROUPS, "model": MODEL, "base": BASE, "side_scope": SIDE_SCOPE,
            "rule": "pred>0 and side=long" if LONG_ONLY else "pred>0",
        },
        "run_at": ledger.stamp(),
        "run_dir": str(run),
        "criteria": {"min_net_sharpe": MIN_SHARPE, "min_net_ann_return": 0.0},
        "passed": passed,
        "n_symbols": n_symbols,
        "net_ann_return": summary["net_ann"],
        "net_sharpe": summary["net_sharpe"],
        "net_max_drawdown": summary["net_mdd"],
        "note": note,
        **extra,
    }


def research_universe() -> dict[int, list[str]]:
    raw = U.load_universe()
    return {y: tradable(raw.get(y, [])) for y in range(2016, 2022)}


def validation_members() -> list[str]:
    names, _ = U.validation_universe()
    return tradable(names)


def fit_book(dataset) -> dict:
    trades, ctxs, factors, universe, skipped = dataset
    part, panel, sub, nets, target, year, X = prepare(trades, ctxs, factors)
    member = panel.members(universe)
    on_member = member[panel.di, panel.si]
    train = np.isfinite(target) & on_member & np.isin(year, TRAIN)
    assert_train_dates(sub["date"], train)
    if train.sum() <= 1000:
        raise RuntimeError(f"训练样本只有 {int(train.sum())} 笔")
    _, fitted, use = W.fit_predict(MODEL, X, target, train, train)
    n_train, token = digest(sub, X, train, use)
    print(f"训练 {n_train} 笔，特征 {len(use)} 列，标签最后一天 "
          f"{pd.DatetimeIndex(sub.loc[train, 'date']).max().date()}", flush=True)
    return {"part": part, "panel": panel, "sub": sub, "nets": nets, "target": target,
            "year": year, "X": X, "member": member, "on_member": on_member,
            "train": train, "fitted": fitted, "use": use, "n_train": n_train,
            "token": token, "universe": universe, "skipped": skipped}


def predict(state, years) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    hold = state["on_member"] & np.isin(state["year"], years)
    pred = np.full(len(state["sub"]), np.nan)
    if hold.any():
        pred[hold] = state["fitted"].predict(state["X"].loc[hold, state["use"]])
    pred = W.scope_predictions(pred, state["sub"], SIDE_SCOPE)
    net = score(state["panel"], state["member"], state["nets"], pred, state["year"], years)
    return pred, hold, net


def run_2022(fp: str):
    pool_2022 = validation_members()
    universe = {**research_universe(), C.VALIDATION_YEAR: pool_2022}
    symbols = sorted({s for names in universe.values() for s in names})
    print(f"2022 检验：加载 {len(symbols)} 个品种，不打开 2023 及以后", flush=True)
    state = fit_book(build(symbols, universe, PARTITIONS_2022, oos_end=None))
    pred, hold, net = predict(state, (2022,))
    if net.empty or net.index.min() < pd.Timestamp("2022-01-01") or net.index.max() > pd.Timestamp("2022-12-31"):
        raise RuntimeError("2022 净值窗口不正确")
    summary = W.summarize(net)
    n_symbols = len(state["universe"].get(2022, []))
    taken = int((hold & np.isfinite(pred) & (pred > 0)).sum())
    print(f"2022 净 Sharpe {summary['net_sharpe']:.3f}  净年化 {summary['net_ann']:.2%}  "
          f"波动 {summary['net_vol']:.2%}  回撤 {summary['net_mdd']:.2%}  "
          f"天数 {summary['n_days']:.0f}  保留 {taken}/{int(hold.sum())}", flush=True)
    run = context.run_dir("orb_ml_validation2022")
    table = performance_table(net, pred, state["target"], hold, n_symbols, "2022")
    table.to_csv(run / "performance.csv", index=False, encoding="utf-8-sig")
    net.rename("net").to_csv(run / "daily_net.csv", header=True, encoding="utf-8-sig")
    with (run / "model.pkl").open("wb") as fh:
        pickle.dump({"fitted": state["fitted"], "use": state["use"],
                     "n_train": state["n_train"], "token": state["token"]}, fh)
    context.dump_json(run / "params.json", {
        "spec": SPEC, "fingerprint": fp, "train": list(TRAIN), "groups": GROUPS,
        "side_scope": SIDE_SCOPE,
        "group_parts": list(W.GROUP_SETS[GROUPS]),
        "model": MODEL, "base": BASE, "n_train": state["n_train"], "token": state["token"],
        "use": state["use"], "universe_2022": state["universe"].get(2022, []),
        "skipped": state["skipped"],
    })
    ok = passes(summary)
    ledger.append(ledger.validation_path(), [ledger_row(
        fp, run, summary, n_symbols, ok,
        f"开盘区间突破元标签，{MODEL}，{GROUPS}，"
        f"{'只做多探索性' if LONG_ONLY else '双向'}，2022 一次性，主口径 1 tick")])
    ledger.write_validation_log()
    print(f"2022 {'通过' if ok else '未通过'}，留痕 {run}", flush=True)
    return ok, run, state


def run_oos(fp: str, fitted, use, n_train: int, token: str):
    C.assert_test_window_closed(OOS_END)
    if ledger.lookup(ledger.read(ledger.oos_path()), fp):
        raise RuntimeError("这本已经做过样本外测试")
    pool_2022 = validation_members()
    names = sorted(set(shard_io.list_shards(C.HOLDOUT_DIR)) & set(tradable(
        shard_io.list_shards(C.HOLDOUT_DIR))))
    oos_universe, screen = U.oos_universe(names, OOS_END)
    oos_universe = {y: tradable(v) for y, v in oos_universe.items()}
    universe = {**research_universe(), C.VALIDATION_YEAR: pool_2022, **oos_universe}
    symbols = sorted({s for v in universe.values() for s in v})
    print(f"样本外：2022 已通过，加载 {len(symbols)} 个品种至 {OOS_END}", flush=True)
    state = fit_book(build(symbols, universe, PARTITIONS_OOS, oos_end=OOS_END))
    # 不用这次拟合的模型。训练矩阵必须和 2022 检验时一致，预测用当时存下的模型。
    if list(state["use"]) != list(use) or (state["n_train"], state["token"]) != (n_train, token):
        raise RuntimeError(
            f"训练矩阵变了（{state['n_train']} 笔），样本外不使用这份数据，也不重训")
    state["fitted"], state["use"] = fitted, use
    years = tuple(U.oos_years(OOS_END))
    pred, hold, net = predict(state, years)
    if net.empty or net.index.min() < pd.Timestamp("2023-01-01") or net.index.max() > pd.Timestamp(OOS_END):
        raise RuntimeError("样本外净值窗口不正确")
    summary = W.summarize(net)
    n_symbols = len({s for y in years for s in state["universe"].get(y, [])})
    taken = int((hold & np.isfinite(pred) & (pred > 0)).sum())
    print(f"2023–2025 净 Sharpe {summary['net_sharpe']:.3f}  净年化 {summary['net_ann']:.2%}  "
          f"波动 {summary['net_vol']:.2%}  回撤 {summary['net_mdd']:.2%}  "
          f"天数 {summary['n_days']:.0f}  保留 {taken}/{int(hold.sum())}", flush=True)
    for y in years:
        one = W.summarize(net[net.index.year == y])
        print(f"  {y} 净 Sharpe {one['net_sharpe']:.3f}  净年化 {one['net_ann']:.2%}  "
              f"回撤 {one['net_mdd']:.2%}  天数 {one['n_days']:.0f}", flush=True)
    run = context.run_dir("orb_ml_oos")
    table = performance_table(
        net, pred, state["target"], hold, n_symbols, f"{C.STRICT_OOS_START}..{OOS_END}")
    table.to_csv(run / "performance.csv", index=False, encoding="utf-8-sig")
    net.rename("net").to_csv(run / "daily_net.csv", header=True, encoding="utf-8-sig")
    if not screen.empty:
        screen.to_csv(run / "universe_oos_screen.csv", index=False, encoding="utf-8-sig")
    context.dump_json(run / "params.json", {
        "spec": SPEC, "fingerprint": fp, "train": list(TRAIN), "frozen_model": True,
        "side_scope": SIDE_SCOPE,
        "n_train": n_train, "oos_end": OOS_END.isoformat(),
        "universe_oos": {str(k): v for k, v in oos_universe.items()},
        "skipped": state["skipped"],
    })
    entry = ledger_row(
        fp, run, summary, n_symbols, False,
        "开盘区间突破元标签，模型冻结在 2019–2021，主口径 1 tick",
        oos_start=C.STRICT_OOS_START.isoformat(), oos_end=OOS_END.isoformat())
    entry.pop("passed", None)
    entry.pop("criteria", None)
    ledger.append(ledger.oos_path(), [entry])
    ledger.write_validation_log()
    print(f"样本外留痕 {run}", flush=True)


def saved_model(entry: dict):
    path = Path(entry["run_dir"]) / "model.pkl"
    with path.open("rb") as fh:
        blob = pickle.load(fh)
    return blob["fitted"], blob["use"], int(blob["n_train"]), blob["token"]


def main() -> int:
    ap = argparse.ArgumentParser(description="开盘区间元标签书：2022 一次性检验，可选再读 2023–2025")
    ap.add_argument("--model", required=True, choices=W.MODELS)
    ap.add_argument("--groups", required=True, choices=list(W.GROUP_SETS))
    ap.add_argument("--long-only", action="store_true",
                    help="仅保留多头预测；探索性 2022 检查，不允许继续打开 OOS")
    ap.add_argument("--oos", action="store_true", help="2022 已测过且通过后读 2023–2025")
    args = ap.parse_args()
    select(args.model, args.groups, args.long_only)
    if LONG_ONLY and args.oos:
        ap.error("--long-only 是事后提出的一次性探索变体，不允许继续读取 2023 及以后")
    if W.train_years(2022) != list(TRAIN):
        raise RuntimeError("训练年份和走步窗口不一致")
    if W.BASES[BASE].open_anchor != "first":
        raise RuntimeError("开盘定义不是交易日第一根")
    fp = fingerprint()
    print(f"规格 {SPEC}  指纹 {fp}", flush=True)
    if ledger.lookup(ledger.read(ledger.oos_path()), fp):
        print("这本的样本外已经测过，不再重测。")
        return 2
    done = ledger.lookup(ledger.validation_entries(), fp)
    if not args.oos:
        if done:
            print("这本的 2022 已经测过，不再重测。要读 2023–2025 加 --oos。")
            return 2
        ok, _run, _state = run_2022(fp)
        return 0 if ok else 1
    if not done:
        print("这本还没测 2022，先不带 --oos 跑一次。")
        return 1
    entry = done[-1]
    sharpe, ann = entry.get("net_sharpe"), entry.get("net_ann_return")
    ok = np.isfinite(sharpe) and np.isfinite(ann) and float(sharpe) >= MIN_SHARPE and float(ann) > 0
    if not ok:
        print("2022 未通过，不读 2023–2025。")
        return 1
    fitted, use, n_train, token = saved_model(entry)
    run_oos(fp, fitted, use, n_train, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
