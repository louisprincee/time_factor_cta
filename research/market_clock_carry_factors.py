"""当日分数且展期为正的信号层检验：分数择时、展期截面 IC、两者叠加、账本分解和多重检验。只读研究期。

    python -m research.market_clock_carry_factors

规则在 config/research/market_clock_carry_factors.json，先登记后运行。不读 2022 及以后。
结果写到 runs/research/market_clock_carry_factors/。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backtest.engine import (
    PERIODS, deflated_sharpe, ledger_inputs, load_research, newey_west_t, pbo_cscv,
)
from research.market_clock_carry import pool_open_return, sectors, simulate, summarize
from research.time_factor_screen import ols_hac
from strategies import market_clock_carry as S
from strategies.catalog import FROZEN_BOOKS
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "market_clock_carry_factors.json"
OUT = C.RUNS_DIR / "research" / "market_clock_carry_factors"
YEARS = tuple(range(C.STUDY_START.year, C.RESEARCH_END.year + 1))


def load_protocol(path=CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "declared_test_not_a_strategy":
        raise ValueError("检验登记必须声明这不是策略")
    if "不读 2022" not in protocol.get("sample", "") or "不读 2026" not in protocol.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    return protocol


def _same_sign_years(values: pd.Series, sign: int) -> int:
    by_year = values.groupby(values.index.year).mean()
    return int((np.sign(by_year) == sign).sum())


def signal_panels(data, protocol):
    """收盘 t 已知的分数、滞后一日的展期收益、池子掩码，和 t 决策对应的下一段收益。"""
    index = data.returns.index
    columns = sorted({s for members in data.universe.values() for s in members})
    pool = S.pool_mask(data.universe, index, columns)
    carry = S.roll_yield(index, columns)
    S.require_carry_coverage(carry, pool)
    lagged = carry.shift(1)
    score = S.market_score(data.factors.signed, data.universe, index)
    forward = data.returns.reindex(index=index, columns=columns).shift(-1).where(pool)
    keep = index[(index >= pd.Timestamp(C.STUDY_START)) & (index <= pd.Timestamp(C.RESEARCH_END))]
    return score.reindex(keep), lagged.reindex(keep), pool.reindex(keep), forward.reindex(keep)


def timing_test(score, forward, lags):
    market = forward.mean(axis=1)
    ok = score.notna() & market.notna()
    s, y = score[ok], market[ok]
    bullish = s.gt(0).astype(float)
    diff, t_diff = ols_hac(y.to_numpy(), [bullish.to_numpy()], lags)
    slope, t_slope = ols_hac(y.to_numpy(), [s.to_numpy()], lags)
    yearly = []
    for year in YEARS:
        ys, ss = y[y.index.year == year], s[s.index.year == year]
        up, down = ys[ss > 0], ys[ss <= 0]
        yearly.append({"年份": year, "看多日数": len(up), "看多日下一段收益": up.mean(),
                       "其余日下一段收益": down.mean(), "差": up.mean() - down.mean()})
    yearly = pd.DataFrame(yearly)
    row = {
        "检验": "分数择时", "看多日占比": float(bullish.mean()),
        "看多日下一段收益": float(y[s > 0].mean()), "其余日下一段收益": float(y[s <= 0].mean()),
        "差": diff, "差_NW_t": t_diff, "连续分数斜率": slope, "斜率_NW_t": t_slope,
        "差同号年份": int((yearly["差"] > 0).sum()),
    }
    return row, yearly


def carry_tests(carry, forward, pool, score, sector_of, min_names, lags):
    valid = carry.notna() & forward.notna() & pool
    count = valid.sum(axis=1)
    ic = carry.where(valid).rank(axis=1).corrwith(forward.where(valid).rank(axis=1), axis=1)
    ic = ic[count >= min_names].dropna()
    positive, rest = carry.gt(0) & valid, carry.le(0) & valid
    spread = (forward.where(positive).mean(axis=1) - forward.where(rest).mean(axis=1)).dropna()
    yearly = pd.DataFrame({
        "年份": YEARS,
        "IC": [ic[ic.index.year == y].mean() for y in YEARS],
        "展期正减非正": [spread[spread.index.year == y].mean() for y in YEARS],
    })
    sector_rows = []
    for name in sorted(set(sector_of.values())):
        names = [c for c in carry.columns if sector_of.get(c) == name]
        if not names:
            continue
        part = (forward[names].where(positive[names]).mean(axis=1)
                - forward[names].where(rest[names]).mean(axis=1)).dropna()
        if len(part) >= 60:
            sector_rows.append({"板块": name, "天数": len(part), "展期正减非正": part.mean(),
                                "NW_t": newey_west_t(part, lags)})
    sectors_table = pd.DataFrame(sector_rows)
    row = {
        "检验": "展期截面", "IC均值": float(ic.mean()), "IC_NW_t": newey_west_t(ic, lags),
        "IC为正天数占比": float((ic > 0).mean()), "IC同号年份": int((yearly["IC"] > 0).sum()),
        "展期正减非正": float(spread.mean()), "收益差_NW_t": newey_west_t(spread, lags),
        "收益差同号年份": int((yearly["展期正减非正"] > 0).sum()),
        "板块同号": int((sectors_table["展期正减非正"] > 0).sum()) if len(sectors_table) else 0,
        "板块数": len(sectors_table),
    }
    s = score.reindex(spread.index)
    ok = s.notna()
    on, off = spread[ok & (s > 0)], spread[ok & (s <= 0)]
    diff, t_diff = ols_hac(spread[ok].to_numpy(), [s[ok].gt(0).astype(float).to_numpy()], lags)
    interaction = {
        "检验": "叠加", "分数为正日展期正减非正": float(on.mean()), "其t": newey_west_t(on, lags),
        "分数非正日展期正减非正": float(off.mean()), "其t_": newey_west_t(off, lags),
        "两组之差": diff, "差_NW_t": t_diff,
    }
    return row, yearly, sectors_table, interaction, ic


def decomposition(data, alpha_lags):
    """同一账本、同样成本，换条件看每条腿各贡献多少。"""
    inputs = ledger_inputs(data, YEARS, pd.Timestamp(C.RESEARCH_END))
    index, columns = data.returns.index, sorted({s for v in data.universe.values() for s in v})
    pool = S.pool_mask(data.universe, index, columns)
    carry = S.roll_yield(index, columns).shift(1)
    score = S.market_score(data.factors.signed, data.universe, index)
    ones = pd.DataFrame(1.0, index=index, columns=columns)
    cases = {
        "分数>0且展期>0（策略）": ones.mul(score.gt(0), axis=0).where(carry.gt(0), 0.0),
        "只看展期>0": ones.where(carry.gt(0), 0.0),
        "只看分数>0": ones.mul(score.gt(0), axis=0),
        "全池常驻": ones,
        "分数≤0且展期>0": ones.mul(score.le(0), axis=0).where(carry.gt(0), 0.0),
    }
    market = pool_open_return(data, inputs["universe"], inputs["index"])
    rows, returns = [], {}
    for case, decision in cases.items():
        target = decision.shift(1).fillna(0.0).where(pool, 0.0).reindex(
            index=inputs["index"], columns=inputs["columns"]).fillna(0.0)
        daily, positions, _ = simulate(target, inputs, FROZEN_BOOKS[0])   # 只借它的 1 倍敞口，目标是各条件自己的
        row, _ = summarize(case, daily, positions, market, inputs["adjusted_open_delta"], sectors())
        rows.append({"条件": case, **{k: row[k] for k in (
            "净Sharpe", "年化净收益", "最大回撤", "相对池子beta", "相对池子年化alpha", "alpha_t",
            "alpha为正年份", "年化成本占权益", "平均敞口倍数")}})
        returns[case] = daily["净收益"].fillna(0.0)
    return pd.DataFrame(rows), pd.DataFrame(returns)


def book_returns(data):
    out = {}
    for book in FROZEN_BOOKS:
        path = C.RUNS_DIR / "strategies" / book.ID / "commodity" / "2016-2021" / "daily.csv"   # 信号层检验只看 2016–2021
        if not path.is_file():
            raise FileNotFoundError(f"{book.NAME} 还没有 2016–2021 结果，先跑 python -m backtest.engine --start 2016 --end 2021")
        daily = pd.read_csv(path, parse_dates=["交易日"]).set_index("交易日")
        out[book.NAME] = daily["净收益"].fillna(0.0)
    return pd.DataFrame(out)


def main() -> None:
    protocol = load_protocol()
    data = load_research(1.0)
    C.assert_no_holdout_dates(data.returns.index, "信号层检验收益")
    lags, alpha_lags = int(protocol["hac_lags"]), int(protocol["alpha_hac_lags"])
    sector_of = sectors()
    score, carry, pool, forward = signal_panels(data, protocol)
    timing, timing_yearly = timing_test(score, forward, lags)
    carry_row, carry_yearly, carry_sectors, interaction, ic = carry_tests(
        carry, forward, pool, score, sector_of, int(protocol["min_names"]), lags)
    decomposition_table, case_returns = decomposition(data, alpha_lags)
    books = book_returns(data)
    trial_sharpes = [float(r.mean() / r.std() * np.sqrt(PERIODS)) for _, r in case_returns.items()]
    trial_counts = [len(trial_sharpes) + len(books.columns), *protocol["tests"]["multiple_testing"]["trials"]]
    dsr = {
        name: {str(n): deflated_sharpe(books[name], trial_sharpes, n) for n in trial_counts}
        for name in books.columns
    }
    matrix = pd.concat([case_returns, books], axis=1).dropna()
    pbo = pbo_cscv(matrix, blocks=16)

    gates = {
        "分数择时": timing["差_NW_t"] >= 2 and timing["差同号年份"] >= 5,
        "展期截面": carry_row["IC_NW_t"] >= 2 and carry_row["IC同号年份"] >= 5 and carry_row["板块同号"] >= 4,
        "叠加": interaction["差_NW_t"] >= 2,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([timing]).to_csv(OUT / "timing.csv", index=False, encoding="utf-8-sig")
    timing_yearly.to_csv(OUT / "timing_yearly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([carry_row]).to_csv(OUT / "carry.csv", index=False, encoding="utf-8-sig")
    carry_yearly.to_csv(OUT / "carry_yearly.csv", index=False, encoding="utf-8-sig")
    carry_sectors.to_csv(OUT / "carry_sectors.csv", index=False, encoding="utf-8-sig")
    ic.rename("IC").to_csv(OUT / "carry_ic_daily.csv", encoding="utf-8-sig")
    pd.DataFrame([interaction]).to_csv(OUT / "interaction.csv", index=False, encoding="utf-8-sig")
    decomposition_table.to_csv(OUT / "decomposition.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps({
        "protocol": protocol, "gates": gates, "deflated_sharpe": dsr, "pbo": pbo,
        "trial_sharpes": trial_sharpes, "window": "2016-2021",
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("分数择时：", {k: round(v, 5) if isinstance(v, float) else v for k, v in timing.items()})
    print(timing_yearly.round(5).to_string(index=False))
    print("展期截面：", {k: round(v, 5) if isinstance(v, float) else v for k, v in carry_row.items()})
    print(carry_yearly.round(5).to_string(index=False))
    print(carry_sectors.round(5).to_string(index=False))
    print("叠加：", {k: round(v, 5) if isinstance(v, float) else v for k, v in interaction.items()})
    print(decomposition_table.round(3).to_string(index=False))
    print("门槛：", gates)
    print("紧缩夏普：", {name: {n: round(v["dsr"], 3) for n, v in item.items()} for name, item in dsr.items()})
    print("过拟合概率：", round(pbo["pbo"], 3), f"（{matrix.shape[1]} 本账）")
    print(f"结果：{OUT}")


if __name__ == "__main__":
    main()
