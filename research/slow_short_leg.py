"""慢趋势空头腿的熊市信号检验：2011–2015 计分，2016–2021 参照。只读研究分片，不扣成本。

    python -m research.slow_short_leg

规则在 config/research/slow_short_leg_2011_2015.json，先登记后运行，只开一次。
结果在 runs/research/slow_short_leg_2011_2015/。不读 2022 及以后。
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from backtest.engine import newey_west_t
from strategies import market_clock_carry as S
from strategies import market_clock_slow_short as T
from tfcta import config as C
from tfcta.data import build_universe, load_universe
from tfcta.factors import load

CONFIG_PATH = C.CONFIG_DIR / "research" / "slow_short_leg_2011_2015.json"
OUT = C.RUNS_DIR / "research" / "slow_short_leg_2011_2015"
SCORED = (2011, 2012, 2013, 2014, 2015)
REFERENCE = (2016, 2017, 2018, 2019, 2020, 2021)
PERIODS = {
    **{str(y): (y,) for y in SCORED},
    "2011–2013": (2011, 2012, 2013), "2014–2015": (2014, 2015), "2016–2021": REFERENCE,
}
LAGS = 5
CRASH_SHARE = 0.20
QUANTITIES = ("池子下一段", "空头篮子下一段", "篮子减池子", "只看趋势<0减池子", "只看展期<0减池子", "空头腿贡献")


def load_protocol(path=CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "declared_test_not_a_strategy":
        raise ValueError("检验登记必须声明这不是策略")
    if "不读 2022" not in protocol.get("sample", "") or "不读 2026" not in protocol.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    if list(protocol["report"]["quantities"]) != list(QUANTITIES):
        raise ValueError("登记的报告量与实现不一致")
    return protocol


def early_universe(stats: pd.DataFrame, protocol: dict) -> dict[int, list[str]]:
    """2011–2015 的品种池：第 y 年只看第 y−1 年，宽度必须等于登记值。"""
    C.assert_no_holdout_dates(pd.to_datetime(stats["year"].astype(str) + "-01-01"), "品种池统计")
    screen = protocol["pools"]["screen"]
    universe, _ = build_universe(
        stats, years=list(SCORED), lookback_years=int(screen["lookback_years"]),
        min_turnover=float(screen["minimum_daily_turnover"]),
        min_valid_ratio=float(screen["minimum_valid_day_ratio"]),
        min_valid_days=int(screen["minimum_valid_days"]),
    )
    widths = {str(y): len(universe[y]) for y in SCORED}
    if widths != protocol["pools"]["expected_widths"]:
        raise ValueError(f"重建的品种池宽度与登记不符：{widths}")
    return {y: list(universe[y]) for y in SCORED}


def panels(bars: dict, universe: dict, carry: pd.DataFrame, warmup_universe: dict) -> dict:
    """逐日的篮子、熊市状态和下一段收益。收盘 t 决策，收益是 t+1 开盘到 t+2 开盘。"""
    index = bars["close"].index
    columns = sorted({s for members in universe.values() for s in members})
    pool = S.pool_mask(universe, index, columns)
    open_, openw = bars["open"].reindex(columns=columns), bars["openw"].reindex(columns=columns)
    forward = ((openw.shift(-1) - openw) / open_.where(open_ > 0)).shift(-1).where(pool)
    lagged = carry.reindex(index=index, columns=columns).shift(1)
    trend = T.symbol_trend(bars, columns).reindex(index)
    daily = T.close_return(bars, columns).reindex(index)

    def bear_of(members: dict) -> pd.Series:
        return T.market_trend(daily.where(S.pool_mask(members, index, columns)).mean(axis=1)).lt(0)

    basket = pool & trend.lt(0) & lagged.lt(0)
    pool_fwd = forward.mean(axis=1)
    width = pool.sum(axis=1).replace(0, np.nan)
    return {
        "pool_fwd": pool_fwd,
        "basket_fwd": forward.where(basket).mean(axis=1),
        "trend_fwd": forward.where(pool & trend.lt(0)).mean(axis=1),
        "carry_fwd": forward.where(pool & lagged.lt(0)).mean(axis=1),
        "basket_sum": forward.where(basket).sum(axis=1),
        "basket_n": basket.sum(axis=1),
        "width": width,
        "bear": {"严格": bear_of(universe), "2010 借用 2011 池预热": bear_of(warmup_universe)},
    }


def daily_quantities(p: dict, bear: pd.Series) -> pd.DataFrame:
    leg = (-p["basket_sum"] / p["width"]).where(bear.eq(True), 0.0)
    out = pd.DataFrame({
        "池子下一段": p["pool_fwd"],
        "空头篮子下一段": p["basket_fwd"],
        "篮子减池子": p["basket_fwd"] - p["pool_fwd"],
        "只看趋势<0减池子": p["trend_fwd"] - p["pool_fwd"],
        "只看展期<0减池子": p["carry_fwd"] - p["pool_fwd"],
        "空头腿贡献": leg,
    })
    return out[p["pool_fwd"].notna()]


def summary_rows(table: pd.DataFrame, bear: pd.Series, label: str) -> list[dict]:
    rows = []
    bear = bear.reindex(table.index).eq(True)
    for period, years in PERIODS.items():
        part = table.index.year.isin(years)
        on = part & bear
        worst = table["池子下一段"][on]
        crash = on & (table["池子下一段"] <= worst.quantile(CRASH_SHARE)) if len(worst) else on
        for condition, mask in (("全部日", part), ("熊市日", on), ("非熊市日", part & ~bear),
                                (f"熊市日里池子下一段最差 {CRASH_SHARE:.0%}", crash)):
            for quantity in QUANTITIES:
                values = table.loc[mask, quantity].dropna()
                rows.append({
                    "口径": label, "区间": period, "条件": condition, "量": quantity, "天数": len(values),
                    "均值bp": float(values.mean() * 1e4) if len(values) else np.nan,
                    "NW_t": newey_west_t(values, LAGS),
                })
    return rows


def yearly_rows(p: dict, table: pd.DataFrame, bears: dict) -> list[dict]:
    rows = []
    for year in (*SCORED, *REFERENCE):
        part = table.index.year == year
        days = table.index[part]
        strict = bears["严格"].reindex(days).eq(True)
        rows.append({
            "年份": year, "交易日": len(days), "池宽": float(p["width"].reindex(days).mean()),
            "熊市日（严格）": int(strict.sum()),
            "熊市日（借用预热）": int(bears["2010 借用 2011 池预热"].reindex(days).eq(True).sum()),
            "熊市日平均篮子品种数": float(p["basket_n"].reindex(days)[strict].mean()) if strict.any() else 0.0,
            "池子累计": float((1 + table.loc[part, "池子下一段"]).prod() - 1),
            "空头腿累计（未扣成本）": float((1 + table.loc[part, "空头腿贡献"]).prod() - 1),
        })
    return rows


def run(protocol=None) -> None:
    protocol = protocol or load_protocol()
    if (OUT / "summary.csv").exists():
        raise FileExistsError(f"这份检验只开一次，结果已在：{OUT}")
    stats = pd.read_csv(C.UNIVERSE_DIR / "yearly_stats.csv")
    early = early_universe(stats, protocol)
    stored = load_universe()
    if early[2015] != sorted(stored[2015]):
        raise ValueError("重建的 2015 品种池与落盘的不一致")
    universe = {**early, **{y: sorted(stored[y]) for y in REFERENCE}}
    warmup_universe = {2010: early[2011], **universe}
    symbols = sorted({s for members in universe.values() for s in members})

    bars = {name: frame.loc[frame.index <= pd.Timestamp(C.RESEARCH_END)].reindex(columns=symbols)
            for name, frame in load(symbols).bars.items()}
    index = bars["close"].index
    C.assert_no_holdout_dates(index, "空头腿检验行情")
    carry = S.roll_yield(index, symbols)
    S.require_carry_coverage(carry, S.pool_mask(universe, index, symbols))

    p = panels(bars, universe, carry, warmup_universe)
    rows, tables = [], {}
    for label, bear in p["bear"].items():
        tables[label] = daily_quantities(p, bear)
        rows.extend(summary_rows(tables[label], bear, label))
    summary = pd.DataFrame(rows)
    yearly = pd.DataFrame(yearly_rows(p, tables["严格"], p["bear"]))

    pick = summary.query("口径 == '严格' and 区间 == '2011–2013' and 条件 == '熊市日'").set_index("量")
    leg, rel = pick.loc["空头腿贡献"], pick.loc["篮子减池子"]
    gate = bool(leg["均值bp"] > 0 and leg["NW_t"] >= 2 and rel["均值bp"] < 0)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT / "yearly.csv", index=False, encoding="utf-8-sig")
    tables["严格"].to_csv(OUT / "daily.csv", encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps({
        "protocol": protocol, "run_at": datetime.now().isoformat(timespec="seconds"),
        "universe": {str(y): v for y, v in universe.items()}, "gate_2011_2013_bear": gate,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(yearly.round(4).to_string(index=False))
    view = summary.query("条件 != '全部日' or 量 in ['篮子减池子', '只看趋势<0减池子', '只看展期<0减池子']")
    print(view.pivot_table(index=["口径", "区间", "条件"], columns="量", values=["均值bp", "NW_t"], sort=False)
          .round(2).to_string())
    print("门槛（严格口径、2011–2013 熊市日：空头腿 > 0 且 t ≥ 2，篮子减池子 < 0）：", gate)
    print(f"结果：{OUT}")


if __name__ == "__main__":
    run()
