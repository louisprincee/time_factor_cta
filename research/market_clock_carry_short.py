"""对偶空头检验：分数 < 0 且滞后展期 < 0 做空，和冻结多头对照，再看研究期池子大跌时各账本的表现。

    python -m research.market_clock_carry_short

规则在 config/research/market_clock_carry_short.json，先登记后运行。只读商品 2016–2021，不读 2022 及以后。
结果写到 runs/research/market_clock_carry_short/。冻结的四条不动。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest.engine import ledger_inputs, load_research, newey_west_t
from research.market_clock_carry import pool_open_return, sectors, simulate, summarize
from research.time_factor_screen import ols_hac
from strategies import market_clock_carry as S
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "market_clock_carry_short.json"
OUT = C.RUNS_DIR / "research" / "market_clock_carry_short"
YEARS = tuple(range(C.STUDY_START.year, C.RESEARCH_END.year + 1))
LEGS = ("long", "short")


def load_protocol(path=CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "declared_test_not_a_strategy":
        raise ValueError("检验登记必须声明这不是策略")
    if "不读 2022" not in protocol.get("sample", "") or "不读 2026" not in protocol.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    if any(set(v["legs"]) - set(LEGS) for v in protocol["variants"]):
        raise ValueError("腿只能是 long 或 short")
    return protocol


def dual_weights(score: pd.Series, carry: pd.DataFrame, legs) -> pd.DataFrame:
    """收盘决策。多头：分数 > 0 且展期 > 0 记 +1；空头：分数 < 0 且展期 < 0 记 −1。展期已经滞后一天。"""
    carry = carry.reindex(index=score.index)
    out = pd.DataFrame(0.0, index=score.index, columns=carry.columns)
    if "long" in legs:
        out = out.mask(carry.gt(0).mul(score.gt(0), axis=0), 1.0)
    if "short" in legs:
        out = out.mask(carry.lt(0).mul(score.lt(0), axis=0), -1.0)
    return out


def side_pnl(positions: pd.DataFrame, delta: pd.DataFrame) -> dict:
    """持仓按方向拆开的毛盈亏，元。"""
    if positions.empty:
        return {"多头毛盈亏": 0.0, "空头毛盈亏": 0.0}
    moves = delta.stack()
    moves.index.names = ["交易日", "品种"]
    held = positions.set_index(["交易日", "品种"])
    pnl = held["手数"] * held["合约乘数"] * moves.reindex(held.index).to_numpy()
    return {"多头毛盈亏": float(pnl[held["手数"] > 0].sum()), "空头毛盈亏": float(pnl[held["手数"] < 0].sum())}


def signal_tests(score, carry, forward, lags):
    market = forward.mean(axis=1)
    ok = score.notna() & market.notna()
    bad = score.lt(0) & ok
    diff, t_diff = ols_hac(market[ok].to_numpy(), [bad[ok].astype(float).to_numpy()], lags)
    basket_mask = carry.lt(0).mul(bad, axis=0) & forward.notna()
    basket = forward.where(basket_mask).mean(axis=1)[bad].dropna()
    excess = (basket - market.reindex(basket.index)).dropna()
    by_year = pd.DataFrame({
        "年份": YEARS,
        "分数<0日池子": [market[bad & (market.index.year == y)].mean() for y in YEARS],
        "空头篮子": [basket[basket.index.year == y].mean() for y in YEARS],
        "篮子减池子": [excess[excess.index.year == y].mean() for y in YEARS],
    })
    rows = [
        {"检验": "分数<0日减其余日的池子下一段收益", "均值": diff, "NW_t": t_diff, "天数": int(bad.sum()),
         "预期": "负"},
        {"检验": "分数<0日 展期<0品种减池子", "均值": float(excess.mean()), "NW_t": newey_west_t(excess, lags),
         "天数": len(excess), "预期": "负"},
        {"检验": "对偶空头篮子的下一段收益", "均值": float(basket.mean()), "NW_t": newey_west_t(basket, lags),
         "天数": len(basket), "预期": "负"},
        {"检验": "分数<0日池子下一段收益（做空池子的毛收益取负号）", "均值": float(market[bad].mean()),
         "NW_t": newey_west_t(market[bad], lags), "天数": int(bad.sum()), "预期": "负"},
    ]
    return pd.DataFrame(rows), by_year


def drawdown_episodes(market: pd.Series, count: int) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """池子净值最深的几段回撤。一段从一个新高开始，到下一个新高之前的最低点为止，各段互不重叠。"""
    nav = (1 + market.fillna(0.0)).cumprod()
    high = nav.cummax()
    segment = (nav >= high).cumsum()           # 每创一次新高开一段
    episodes = []
    for _, part in nav.groupby(segment):
        trough = part.idxmin()
        peak = part.index[0]
        depth = float(part[trough] / part[peak] - 1)
        if depth < 0:
            episodes.append((peak, trough, depth))
    return sorted(episodes, key=lambda item: item[2])[:count]


def main() -> None:
    protocol = load_protocol()
    data = load_research(1.0)
    C.assert_no_holdout_dates(data.returns.index, "对偶空头检验收益")
    index = data.returns.index
    columns = sorted({s for members in data.universe.values() for s in members})
    pool = S.pool_mask(data.universe, index, columns)
    carry = S.roll_yield(index, columns)
    S.require_carry_coverage(carry, pool)
    lagged = carry.shift(1)
    score = S.market_score(data.factors.signed, data.universe, index)

    keep = index[(index >= pd.Timestamp(C.STUDY_START)) & (index <= pd.Timestamp(C.RESEARCH_END))]
    forward = data.returns.reindex(index=index, columns=columns).shift(-1).where(pool).reindex(keep)
    tests, tests_yearly = signal_tests(score.reindex(keep), lagged.reindex(keep), forward, 5)

    inputs = ledger_inputs(data, YEARS, pd.Timestamp(C.RESEARCH_END))
    market = pool_open_return(data, inputs["universe"], inputs["index"])
    trend = market.rolling(60).sum().shift(1)
    terciles = pd.qcut(trend, 3, labels=["过去60日下跌", "中间", "过去60日上涨"])
    episodes = drawdown_episodes(market, 3)
    rows, annual_rows, regime_rows, episode_rows = [], [], [], []
    for variant in protocol["variants"]:
        decision = dual_weights(score, lagged, variant["legs"])
        target = decision.shift(1).fillna(0.0).where(pool, 0.0).reindex(
            index=inputs["index"], columns=inputs["columns"]).fillna(0.0)
        for sizing in protocol["sizing"]:
            book = SimpleNamespace(GROSS=sizing["gross"], GROSS_SPLIT=sizing["split"])
            label = f"{variant['id']}·{'按池宽' if sizing['split'] == 'pool' else '入选平分'}"
            daily, positions, _ = simulate(target, inputs, book)
            row, annual = summarize(label, daily, positions, market, inputs["adjusted_open_delta"], sectors())
            net = daily["净收益"].fillna(0.0)
            gross = daily["毛盈亏"] / daily["期初权益"]
            active = daily["持仓品种数"] > 0
            rows.append({
                "变体": variant["id"], "分仓": sizing["split"],
                **{k: row[k] for k in ("净Sharpe", "毛Sharpe", "年化净收益", "最大回撤", "相对池子beta",
                                       "相对池子年化alpha", "alpha_t", "为正年份", "年化成本占权益",
                                       "平均敞口倍数", "下跌月捕获", "熔断次数")},
                "持仓日占比": float(active.mean()),
                "持仓日毛收益_bp": float(gross[active].mean() * 1e4),
                "持仓日池子_bp": float(market[active].mean() * 1e4),
                **{k: v / 1e6 for k, v in side_pnl(positions, inputs["adjusted_open_delta"]).items()},
            })
            annual_rows.extend(annual)
            for level in terciles.cat.categories:
                days = terciles.index[terciles.eq(level)]
                regime_rows.append({"变体": label, "状态": level, "天数": len(days),
                                    "年化净收益": float(net.reindex(days).mean() * 242),
                                    "池子年化": float(market.reindex(days).mean() * 242)})
            for peak, trough, depth in episodes:
                part = net.loc[peak:trough]
                episode_rows.append({"变体": label, "峰": peak.date(), "谷": trough.date(), "池子": depth,
                                     "账本": float((1 + part).prod() - 1)})

    results = pd.DataFrame(rows)
    by = results.set_index(["变体", "分仓"])
    short_ok = all(
        by.loc[("short", split), "净Sharpe"] > 0 and by.loc[("short", split), "为正年份"] >= 4
        for split in ("pool", "selected")
    ) and tests.iloc[2]["NW_t"] <= -2
    ls_ok = all(
        by.loc[("long_short", split), "净Sharpe"] > by.loc[("long", split), "净Sharpe"]
        and by.loc[("long_short", split), "最大回撤"] >= by.loc[("long", split), "最大回撤"]
        for split in ("pool", "selected")
    )
    gates = {"空头": bool(short_ok), "多空": bool(ls_ok)}

    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "books.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual_rows).to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUT / "signal_tests.csv", index=False, encoding="utf-8-sig")
    tests_yearly.to_csv(OUT / "signal_tests_yearly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(regime_rows).to_csv(OUT / "regimes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(episode_rows).to_csv(OUT / "drawdowns.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps({"protocol": protocol, "gates": gates, "window": "2016-2021"},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    print(tests.round(5).to_string(index=False))
    print(tests_yearly.round(5).to_string(index=False))
    print(results.round(3).to_string(index=False))
    print(pd.DataFrame(annual_rows).pivot(index="变体", columns="年份", values="净收益").round(3).to_string())
    print(pd.DataFrame(regime_rows).round(3).to_string(index=False))
    print(pd.DataFrame(episode_rows).round(3).to_string(index=False))
    print("门槛：", gates)
    print(f"结果：{OUT}")


if __name__ == "__main__":
    main()
