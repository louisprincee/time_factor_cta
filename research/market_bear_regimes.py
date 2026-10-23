"""熊市识别规则比较：五个候选对照事后标注的熊市区间，再看空头腿和账本。只读研究分片 2010–2021。

    python -m research.market_bear_regimes

规则在 config/research/market_bear_regimes.json，先登记后运行。只报告，不改冻结策略。
结果在 runs/research/market_bear_regimes/。不读 2022 及以后。
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from backtest.engine import ledger_inputs, load_research, newey_west_t
from research.market_clock_carry import pool_open_return, sectors, simulate, summarize
from research.slow_short_leg import early_universe
from research.stress_2014_2015 import load_protocol as stress_protocol, load_stress_data
from strategies import market_clock_carry as S
from strategies import market_clock_slow_short as T
from tfcta import config as C
from tfcta.data import load_universe
from tfcta.factors import load

CONFIG_PATH = C.CONFIG_DIR / "research" / "market_bear_regimes.json"
OUT = C.RUNS_DIR / "research" / "market_bear_regimes"
CANDIDATES = ("base", "fast4", "drawdown", "breadth_trend", "breadth_carry")
FAST_WINDOWS = (20, 60, 120, 250)
DRAWDOWN_LIMIT, DRAWDOWN_WINDOW, DRAWDOWN_MIN = 0.10, 250, 60
BREADTH_SHARE, BREADTH_MIN = 0.5, 5
SCORE_START, SCORE_END = pd.Timestamp("2011-01-01"), pd.Timestamp(C.RESEARCH_END)
PERIODS = {"2011–2015": range(2011, 2016), "2016–2021": range(2016, 2022), "2011–2021": range(2011, 2022)}
LAGS = 5


def load_protocol(path=CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "declared_test_not_a_strategy":
        raise ValueError("检验登记必须声明这不是策略")
    if "不读 2022" not in protocol.get("sample", "") or "不读 2026" not in protocol.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    if [c["id"] for c in protocol["candidates"]] != list(CANDIDATES):
        raise ValueError("登记的候选规则与实现不一致")
    return protocol


def window_trend(daily: pd.Series, windows) -> pd.Series:
    """池子日收益累计各窗口后的符号平均；任一窗口不足为缺失。"""
    signs = [np.sign(daily.rolling(w, min_periods=w).sum()) for w in windows]
    return sum(signs) / len(signs)


def breadth(flags: pd.DataFrame, valid: pd.DataFrame, minimum: int = BREADTH_MIN) -> pd.Series:
    """有值的池内品种里 flags 为真的占比；有值的不足 minimum 个时为缺失。"""
    count = valid.sum(axis=1)
    return ((flags & valid).sum(axis=1) / count.where(count > 0)).where(count >= minimum)


def candidate_bears(daily: pd.Series, trend: pd.DataFrame, lagged: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """收盘 t 已知的五个熊市判断；缺失一律当作不是熊市。"""
    level = (1 + daily.fillna(0.0)).cumprod()
    peak = level.rolling(DRAWDOWN_WINDOW, min_periods=DRAWDOWN_MIN).max()
    raw = {
        "base": window_trend(daily, T.TREND_WINDOWS) < 0,
        "fast4": window_trend(daily, FAST_WINDOWS) < 0,
        "drawdown": (level / peak - 1) <= -DRAWDOWN_LIMIT,
        "breadth_trend": breadth(trend.lt(0), trend.notna() & pool) > BREADTH_SHARE,
        "breadth_carry": breadth(lagged.lt(0), lagged.notna() & pool) > BREADTH_SHARE,
    }
    return {name: flag.fillna(False).astype(bool) for name, flag in raw.items()}


def bear_episodes(level: pd.Series, down: float, up: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Lunde–Timmermann：从运行高点跌 down 进熊市（起点记在高点），从运行低点涨 up 出熊市（终点记在低点）。"""
    values, dates = level.to_numpy(dtype="float64"), level.index
    episodes, bear, top, bottom, start = [], False, 0, 0, 0
    for i in range(1, len(values)):
        if not bear:
            if values[i] > values[top]:
                top = i
            if values[i] <= values[top] * (1 - down):
                bear, start, bottom = True, top, i
        else:
            if values[i] < values[bottom]:
                bottom = i
            if values[i] >= values[bottom] * (1 + up):
                episodes.append((dates[start], dates[bottom]))
                bear, top = False, i
    if bear:
        episodes.append((dates[start], dates[bottom]))
    return episodes


def label_days(index: pd.DatetimeIndex, episodes) -> pd.Series:
    out = pd.Series(False, index=index)
    for peak, trough in episodes:
        out[(index > peak) & (index <= trough)] = True
    return out


def detection_rows(name: str, flag: pd.Series, level: pd.Series, episodes, label: str) -> list[dict]:
    index = flag.index
    rows = []
    for peak, trough in episodes:
        if trough < SCORE_START:
            continue
        inside = index[(index > peak) & (index <= trough)]
        hits = inside[flag.reindex(inside).to_numpy()]
        after = index[index > trough]
        off = after[~flag.reindex(after).to_numpy()]
        exit_lag = (int(index.get_loc(off[0]) - index.get_loc(trough)) if len(off) else np.nan) \
            if flag.loc[trough] else 0
        rows.append({
            "规则": name, "标注": label, "高点": peak.date(), "低点": trough.date(),
            "段内交易日": len(inside), "跌幅": float(level.loc[trough] / level.loc[peak] - 1),
            "检测延迟": int(index.get_loc(hits[0]) - index.get_loc(peak)) if len(hits) else np.nan,
            "检测前已跌": float(level.loc[hits[0]] / level.loc[peak] - 1) if len(hits) else np.nan,
            "段内覆盖": float(flag.reindex(inside).mean()) if len(inside) else np.nan,
            "退出延迟": exit_lag,
        })
    return rows


def summary_row(name: str, flag: pd.Series, labels: pd.Series, pool_fwd: pd.Series, leg: pd.Series,
                years) -> dict:
    days = flag.index[flag.index.year.isin(list(years))]
    f, lab = flag.reindex(days), labels.reindex(days)
    on, off = pool_fwd.reindex(days)[f].dropna(), pool_fwd.reindex(days)[~f].dropna()
    leg = leg.reindex(days).dropna()
    switches = int(f.astype(int).diff().abs().sum())
    return {
        "规则": name, "熊市日占比": float(f.mean()),
        "标注熊市命中率": float(f[lab].mean()) if lab.any() else np.nan,
        "判熊市里误判占比": float((~lab[f]).mean()) if f.any() else np.nan,
        "每年切换次数": switches / max(len(set(days.year)), 1),
        "熊市日池子下一段bp": float(on.mean() * 1e4) if len(on) else np.nan, "其t": newey_west_t(on, LAGS),
        "其余日池子下一段bp": float(off.mean() * 1e4) if len(off) else np.nan,
        "空头腿bp": float(leg.mean() * 1e4), "空头腿t": newey_west_t(leg, LAGS),
        "空头腿累计": float((1 + leg).prod() - 1),
    }


def book_targets(data, bear: pd.Series) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """多头腿同冻结策略，空头腿用给定的熊市序列（从 2010 起连续算好，这里按日期取）。"""
    index = data.returns.index
    columns = sorted({s for members in data.universe.values() for s in members})
    pool = S.pool_mask(data.universe, index, columns)
    carry = S.roll_yield(index, columns)
    S.require_carry_coverage(carry, pool)
    lagged = carry.shift(1)
    score = S.market_score(data.factors.signed, data.universe, index)
    trend = T.symbol_trend(data.factors.bars, columns).reindex(index)
    decision = T.decision_weights(score, lagged, trend, bear.reindex(index).eq(True))
    return decision.shift(1).fillna(0.0).where(pool, 0.0), index


def book_rows(data, bears: dict, years: tuple, label: str) -> list[dict]:
    inputs = ledger_inputs(data, years, pd.Timestamp(f"{years[-1]}-12-31"))
    market = pool_open_return(data, inputs["universe"], inputs["index"])
    rows = []
    for name, bear in bears.items():
        target, _ = book_targets(data, bear)
        target = target.reindex(index=inputs["index"], columns=inputs["columns"]).fillna(0.0)
        daily, positions, _ = simulate(target, inputs, T.BOOK)
        row, _ = summarize(name, daily, positions, market, inputs["adjusted_open_delta"], sectors())
        short_days = int((positions.groupby("交易日")["手数"].min() < 0).sum()) if not positions.empty else 0
        rows.append({"账本": label, "规则": name, **{k: row[k] for k in (
            "净Sharpe", "年化净收益", "最大回撤", "相对池子beta", "相对池子年化alpha", "alpha_t",
            "年化成本占权益", "下跌月捕获")}, "空头持仓日": short_days})
    return rows


def run(protocol=None) -> None:
    protocol = protocol or load_protocol()
    stats = pd.read_csv(C.UNIVERSE_DIR / "yearly_stats.csv")
    early = early_universe(stats, protocol)
    stored = load_universe()
    if early[2015] != sorted(stored[2015]):
        raise ValueError("重建的 2015 品种池与落盘的不一致")
    universe = {**early, **{y: sorted(stored[y]) for y in range(2016, 2022)}}
    warm = {2010: early[2011], **universe}
    symbols = sorted({s for members in universe.values() for s in members})

    bars = {name: frame.loc[frame.index <= SCORE_END].reindex(columns=symbols)
            for name, frame in load(symbols).bars.items()}
    index = bars["close"].index
    C.assert_no_holdout_dates(index, "熊市识别比较行情")
    carry = S.roll_yield(index, symbols)
    S.require_carry_coverage(carry, S.pool_mask(universe, index, symbols))
    pool = S.pool_mask(warm, index, symbols)
    daily = T.close_return(bars, symbols).reindex(index).where(pool).mean(axis=1)
    level = (1 + daily.fillna(0.0)).cumprod()
    lagged = carry.shift(1)
    trend = T.symbol_trend(bars, symbols).reindex(index)
    bears = candidate_bears(daily, trend, lagged, pool)

    strict = S.pool_mask(universe, index, symbols)
    open_, openw = bars["open"], bars["openw"]
    forward = ((openw.shift(-1) - openw) / open_.where(open_ > 0)).shift(-1).where(strict)
    pool_fwd = forward.mean(axis=1)
    basket = strict & trend.lt(0) & lagged.lt(0)
    width = strict.sum(axis=1).replace(0, np.nan)
    leg_all = -forward.where(basket).sum(axis=1) / width

    scored = index[(index >= SCORE_START) & pool_fwd.notna()]
    labels = {key: bear_episodes(level, **protocol["ex_post_labels"][key]) for key in ("primary", "sensitivity")}
    detect, summary = [], []
    for name, flag in bears.items():
        for key, episodes in labels.items():
            detect.extend(detection_rows(name, flag, level, episodes, key))
        lab = label_days(index, labels["primary"])
        leg = leg_all.where(flag, 0.0).reindex(scored)
        for period, years in PERIODS.items():
            summary.append({"区间": period, **summary_row(name, flag.reindex(scored), lab, pool_fwd, leg, years)})
    detect, summary = pd.DataFrame(detect), pd.DataFrame(summary)

    base = summary.query("区间 == '2011–2021'").set_index("规则")
    lag = detect.query("标注 == 'primary'").groupby("规则")["检测延迟"].median()
    dominance = {
        name: bool(lag[name] < lag["base"]
                   and base.loc[name, "判熊市里误判占比"] <= base.loc["base", "判熊市里误判占比"]
                   and base.loc[name, "空头腿bp"] >= base.loc["base", "空头腿bp"])
        for name in CANDIDATES if name != "base"
    }

    books = book_rows(load_stress_data(stress_protocol()), bears, (2014, 2015), "2014–2015")
    books += book_rows(load_research(1.0), bears, tuple(range(2016, 2022)), "2016–2021")
    books = pd.DataFrame(books)

    OUT.mkdir(parents=True, exist_ok=True)
    detect.to_csv(OUT / "detection.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    books.to_csv(OUT / "books.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bears).assign(池子指数=level).to_csv(OUT / "flags.csv", encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps({
        "protocol": protocol, "run_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": {k: [[str(a.date()), str(b.date())] for a, b in v] for k, v in labels.items()},
        "dominates_base": dominance, "median_detection_lag_primary": lag.to_dict(),
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("事后标注熊市（主口径 15%/15%）：", [(str(a.date()), str(b.date())) for a, b in labels["primary"]])
    print(detect.query("标注 == 'primary'").round(3).to_string(index=False))
    print(summary.round(3).to_string(index=False))
    print(books.round(3).to_string(index=False))
    print("相对 base 是否同时更快、误判不更多、空头腿不更差：", dominance)
    print(f"结果：{OUT}")


if __name__ == "__main__":
    run()
