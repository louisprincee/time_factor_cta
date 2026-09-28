"""研究期候选书对比（只读 2016–2021，不碰 2022 与样本外，不记台账）。

对比四类书，以及执行口径本身的影响：

1. 基线：四个时间因子等权；分别在「周五单批 / 五批错开」×「不缩放 / 波动率目标」下跑；
2. 快慢分书 + 50/50 风险平价：快书 = time_combo、neg_clv、neg_ret_day；
   慢书 = tsmom、cs_mom_ra_250、er_signed、po。两本书各自成书，净收益按事前 60 日波动倒数配权；
3. 趋势交互：``time_combo × 1{sign(time_combo) = sign(tsmom)}``，与趋势同向才持有；
4. 软状态书：``w = clip(ER 分位（滞后一日）, 0.2, 0.8)``，
   信号 = w × 趋势块 + (1 − w) × 反转块；另跑 w ≡ 0.5 的对照，看状态权重本身有没有贡献。

另附两张诊断表：单批每周调仓按「星期几」与按「交易日相位」取信号的对比，
以及基线信号对未来第 1–5 日收益的分星期几 t 值，用来判断旧周五口径的优势是不是日历相位。

风险平价在收益层合成，两本书之间的对冲不省成本，净值偏保守。
这里的结果只用来挑书；挑中的书再按 step5 → step6 的流程走，2022 只测一次。

    python scripts/research_compare_books.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C                                 # noqa: E402
from tfcta.data import sectors                                # noqa: E402
from tfcta.factors import library                             # noqa: E402
from tfcta.research.analysis import stats                     # noqa: E402
from tfcta.research.backtest import costs, engine, strategy   # noqa: E402
from tfcta.research.workflow import context                   # noqa: E402

FAST = {"time_combo": 1.0, "neg_clv": 1.0, "neg_ret_day": 1.0}
SLOW = {"tsmom": 1.0, "cs_mom_ra_250": 1.0, "er_signed": 1.0, "po": 1.0}
BASELINE = dict(C.FACTOR_SIGNS)
RP_WINDOW, RP_MIN = 60, 20
ER_WINDOW, ER_MIN = 252, 120
W_LO, W_HI = 0.2, 0.8
EXECUTIONS = {                      # 名字: (批数, 波动率目标)
    "周五单批/不缩放": (0, 0.0),
    "五批错开/不缩放": (C.REBALANCE_TRANCHES, 0.0),
    "周五单批/波动目标": (0, C.VOL_TARGET),
    "五批错开/波动目标": (C.REBALANCE_TRANCHES, C.VOL_TARGET),
}
NEW = "五批错开/波动目标"


def block(signal_set: library.SignalSet, factors: dict[str, float]) -> pd.DataFrame:
    """与 ``strategy.equal_weight_signal`` 相同，但不截断，供再合成用。"""
    frames = []
    for name, sign in factors.items():
        signed = signal_set.raw(name) * sign
        if name not in library.STANDARDIZED_FACTORS:
            signed = library.trail_z(signed)
        frames.append(signed)
    return library.average_signals(frames)


def er_weight(signal_set: library.SignalSet) -> pd.DataFrame:
    """ER（趋势效率）在自身过去一年里的分位，滞后一日，截到 [0.2, 0.8]。"""
    er = signal_set.raw("er")
    pct = er.rolling(ER_WINDOW, min_periods=ER_MIN).rank(pct=True)
    return pct.shift(1).clip(W_LO, W_HI)


def blend(w: pd.DataFrame, trend: pd.DataFrame, rev: pd.DataFrame) -> pd.DataFrame:
    """w × 趋势 + (1 − w) × 反转；某一块缺失时用另一块，两块都缺才是 NaN。"""
    idx = trend.index.union(rev.index)
    cols = trend.columns.union(rev.columns)
    t, r = trend.reindex(index=idx, columns=cols), rev.reindex(index=idx, columns=cols)
    w = w.reindex(index=idx, columns=cols)
    both = w * t + (1 - w) * r
    return both.where(t.notna() & r.notna() & w.notna(),
                      t.where(r.isna(), r.where(t.isna())))


def _hold(signal: pd.DataFrame, is_reb: np.ndarray) -> pd.DataFrame:
    """只在 ``is_reb`` 那几天取值、持有到下一次调仓；调仓日缺值则整段空仓（同 ``engine.weekly``）。"""
    reb = pd.Series(is_reb, index=signal.index)
    valid = signal.notna().astype("float64").where(reb, axis=0).ffill().eq(1.0)
    return signal.where(reb, axis=0).ffill().where(valid)


def phase_table(signal, day_ret, universe, years, slip, vol) -> list[dict]:
    """单批每周调仓按星期几取信号 vs 按交易日序号取相位：看周频结果是不是押在某个星期几上。"""
    scoped = {int(y): [s for s in v if s in day_ret.columns] for y, v in universe.items()}
    idx = pd.DatetimeIndex(signal.index)
    target = engine.vol_target(signal, vol, C.VOL_TARGET)
    order = np.arange(len(idx)) % 5
    phases = {f"星期{'一二三四五'[d]}": idx.weekday == d for d in range(5)}
    phases.update({f"交易日相位{k}": order == k for k in range(5)})
    rows = []
    for name, mask in phases.items():
        pos = engine.execute_position(_hold(target, mask))
        gross = engine.stitch_test_years(engine.run_book(pos, day_ret, scoped, 0.0), years)
        net = engine.stitch_test_years(
            engine.run_book(pos, day_ret, scoped, C.FEE_BASE, slippage=slip), years)
        rows.append({"phase": name, "gross_sharpe": stats.sharpe_ratio(gross),
                     "net_sharpe": stats.sharpe_ratio(net),
                     "turnover": engine.annual_turnover(pos, scoped, years)})
    return rows


def lag_table(signal: pd.DataFrame, day_ret: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """t 日收盘信号 × 第 t+j 日持有收益的池内均值，按信号所在星期几分组的 t 值。"""
    s = signal.loc[str(years[0]):str(years[-1])]
    weekday = pd.Series(s.index.weekday, index=s.index)
    out = {}
    for j in range(1, 6):
        pnl = (s * day_ret.shift(-j).reindex_like(s)).mean(axis=1)
        g = pnl.groupby(weekday)
        out[f"lag{j}"] = g.mean() / (g.std() / np.sqrt(g.count()))
    table = pd.DataFrame(out).T
    table.columns = [f"星期{'一二三四五六日'[d]}" for d in table.columns]
    return table


def row_of(table: pd.DataFrame, book: str, execution: str, overall: str) -> tuple[dict, dict]:
    head = table[table["period"] == overall].iloc[0].to_dict()
    by_year = {f"sharpe_{p}": v for p, v in
               table[table["period"] != overall].set_index("period")["net_sharpe"].items()}
    head.update(book=book, execution=execution)
    return head, by_year


def main() -> int:
    reason = context.not_ready_reason()
    if reason:
        print(reason)
        return 2
    universe, symbols, day_ret = context.load_context(None)
    years = [f["test_year"] for f in engine.walk_forward_folds()]
    overall = strategy.research_period_label()
    sig = library.load(symbols)
    slip, cost_note = costs.research_slippage(symbols, day_ret.index, C.SLIPPAGE_TICKS)
    cases = {sectors.ALL_POOL: sorted(symbols)}

    fast, slow = block(sig, FAST), block(sig, SLOW)
    interaction = sig.signed["time_combo_trend"]
    w_er = er_weight(sig)
    signals = {
        "基线（四个时间因子）": block(sig, BASELINE),
        "快书": fast,
        "慢书": slow,
        "趋势交互 time_combo×同号(tsmom)": interaction,
        "软状态书 w=ER分位": blend(w_er, slow, fast),
        "对照 w≡0.5": blend(w_er * 0 + 0.5, slow, fast),
    }

    rows, years_rows, nets = [], [], {}
    for book, raw in signals.items():
        signal = raw.clip(-1, 1)
        plans = EXECUTIONS if book.startswith("基线") else {NEW: EXECUTIONS[NEW]}
        for execution, (tranches, target) in plans.items():
            cfg = strategy.BookConfig(factors={"tsmom": 1.0}, tranches=tranches,
                                      vol_target=target)
            table = strategy.evaluate_cases(signal, day_ret, universe, cases, years,
                                            cfg, slip, overall, vol=sig.vol)
            head, by_year = row_of(table, book, execution, overall)
            rows.append(head)
            years_rows.append({"book": book, "execution": execution, **by_year})
            if execution == NEW and book in ("快书", "慢书"):
                _, _, _, _, net = strategy.portfolio(
                    signal, day_ret, universe, cases[sectors.ALL_POOL], cfg, slip, sig.vol)
                nets[book] = (engine.stitch_test_years(net, years), head["turnover"])

    # 快慢两本书的风险平价：事前 60 日波动倒数配权，权重只用到昨天
    (r_fast, to_fast), (r_slow, to_slow) = nets["快书"], nets["慢书"]
    both = pd.concat({"fast": r_fast, "slow": r_slow}, axis=1).dropna()
    inv = 1.0 / both.rolling(RP_WINDOW, min_periods=RP_MIN).std().shift(1)
    w = inv.div(inv.sum(axis=1), axis=0).fillna(0.5)
    rp = (w * both).sum(axis=1)
    perf = stats.performance(rp)
    rows.append({
        "book": "快慢分书 50/50 风险平价", "execution": NEW, "universe": sectors.ALL_POOL,
        "period": overall, "n_symbols": len(symbols),
        "net_ann_return": perf["ann_return"], "net_ann_vol": perf["ann_vol"],
        "net_sharpe": stats.sharpe_ratio(rp), "net_max_drawdown": perf["max_drawdown"],
        "net_calmar": perf["calmar"],
        "turnover": float((w["fast"] * to_fast + w["slow"] * to_slow).mean()),
        "corr_fast_slow": float(both.corr().iloc[0, 1]),
        "avg_w_fast": float(w["fast"].mean()),
    })
    years_rows.append({"book": "快慢分书 50/50 风险平价", "execution": NEW,
                       **{f"sharpe_{y}": stats.sharpe_ratio(rp[rp.index.year == y])
                          for y in years}})

    phases = pd.DataFrame(
        [{"book": book, **row} for book in ("基线（四个时间因子）", "趋势交互 time_combo×同号(tsmom)")
         for row in phase_table(signals[book].clip(-1, 1), day_ret, universe, years, slip, sig.vol)])
    lags = lag_table(signals["基线（四个时间因子）"].clip(-1, 1), day_ret, years)

    cols = ["book", "execution", "gross_ann_return", "gross_sharpe", "net_ann_return",
            "net_ann_vol", "net_sharpe", "net_max_drawdown", "net_calmar", "turnover",
            "ic_ts", "ic_t", "ic_ts_1d", "ic_t_1d", "corr_fast_slow", "avg_w_fast"]
    out = pd.DataFrame(rows)
    out = out[[c for c in cols if c in out.columns]]
    by_year = pd.DataFrame(years_rows)
    C.ensure_dirs()
    run = context.run_dir("research_compare_books")
    for tab, name in ((out, "book_compare.csv"), (by_year, "book_compare_by_year.csv"),
                      (phases, "book_compare_phase.csv"),
                      (lags.reset_index(names="lag"), "baseline_lag_by_weekday.csv")):
        tab.to_csv(C.RESEARCH_OUT_DIR / name, index=False, encoding="utf-8-sig")
        tab.to_csv(run / name, index=False, encoding="utf-8-sig")
    context.dump_json(run / "params.json", {
        "fast": FAST, "slow": SLOW, "baseline": BASELINE, "executions": EXECUTIONS,
        "rp_window": RP_WINDOW, "er_window": ER_WINDOW, "w_bounds": [W_LO, W_HI],
        "vol_cap": C.VOL_TARGET_CAP, "ic_horizon": C.IC_HORIZON, "slippage": cost_note,
    })
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:+.3f}"):
        print(out.to_string(index=False))
        print(by_year.to_string(index=False))
        print("\n单批每周调仓：按星期几 vs 按交易日相位（波动目标开）")
        print(phases.to_string(index=False))
        print("\n基线信号 × 第 t+j 日收益的 t 值，按信号所在星期几")
        print(lags.to_string())
    print(f"\n{cost_note}\n结果: {C.RESEARCH_OUT_DIR / 'book_compare.csv'}\n快照: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
