"""当天分数且展期收益为正。

    python -m research.market_clock_carry
    python -m backtest.engine --research
    python -m backtest.engine --validation
    python -m backtest.engine --oos

2022 及以后先记台账再读。这些年份以前看过，不是干净样本外。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import (
    PERIODS, _write_results, ledger_inputs, load_market_clock_year, load_research,
    sharpe_ratio, simulate_target_lots,
)
from strategies import market_clock_carry as S
from tfcta import config as C

OUT = C.RUNS_DIR / "research" / S.ID


def sectors() -> dict[str, str]:
    table = json.loads((C.CONFIG_DIR / "research" / "time_factor_screen.json").read_text(encoding="utf-8"))
    return {symbol: name for name, members in table["sectors"].items() for symbol in members}


def pool_open_return(data, universe: dict, index) -> pd.Series:
    """池内开盘到开盘收益的等权平均，与账本的逐日净收益同一口径。"""
    pool = S.pool_mask(universe, index, data.returns.columns)
    return data.returns.reindex(index).where(pool).mean(axis=1).fillna(0.0)


def summarize(label: str, daily: pd.DataFrame, positions: pd.DataFrame, market: pd.Series,
              delta: pd.DataFrame, sector_of: dict) -> tuple[dict, list[dict]]:
    net = daily["净收益"].fillna(0.0)
    gross = (daily["毛盈亏"] / daily["期初权益"]).fillna(0.0)
    market = market.reindex(net.index).fillna(0.0)
    beta = float(net.cov(market) / market.var())
    alpha = net - beta * market
    years = sorted(set(net.index.year))
    month = net.index.to_period("M")
    strat_m = (1 + net).groupby(month).prod() - 1
    pool_m = (1 + market).groupby(month).prod() - 1
    up, down = pool_m > 0, pool_m < 0
    equity = daily["权益"]
    peak = np.maximum.accumulate(np.r_[100_000_000.0, equity.to_numpy()])[1:]
    share = {}
    if not positions.empty:
        moves = delta.stack()
        moves.index.names = ["交易日", "品种"]
        held = positions.set_index(["交易日", "品种"])
        pnl = held["手数"] * held["合约乘数"] * moves.reindex(held.index).to_numpy()
        by_sector = pnl.groupby(pnl.index.get_level_values("品种").map(sector_of)).sum()
        share = (by_sector / by_sector.abs().sum()).to_dict()
    annual = []
    for year in years:
        part = net[net.index.year == year]
        annual.append({
            "变体": label, "年份": year, "净收益": float((1 + part).prod() - 1),
            "Sharpe": sharpe_ratio(part), "相对池子alpha": float(alpha[alpha.index.year == year].sum()),
        })
    row = {
        "变体": label,
        "净Sharpe": sharpe_ratio(net),
        "毛Sharpe": sharpe_ratio(gross),
        "年化净收益": float((equity.iloc[-1] / 1e8) ** (PERIODS / len(net)) - 1),
        "年化波动": float(net.std() * np.sqrt(PERIODS)),
        "最大回撤": float((equity.to_numpy() / peak - 1).min()),
        "年化成本占权益": float((daily["交易费用"] / daily["期初权益"]).mean() * PERIODS),
        "平均敞口倍数": float(daily["敞口倍数"].mean()),
        "最大敞口倍数": float(daily["敞口倍数"].max()),
        "为正年份": int(sum(item["净收益"] > 0 for item in annual)),
        "相对池子beta": beta,
        "相对池子年化alpha": float(alpha.mean() * PERIODS),
        "alpha为正年份": int(sum(item["相对池子alpha"] > 0 for item in annual)),
        "上涨月捕获": float(strat_m[up].mean() / pool_m[up].mean()),
        "下跌月捕获": float(strat_m[down].mean() / pool_m[down].mean()),
        "熔断次数": int((daily["停止新开仓"].astype(bool) & ~daily["停止新开仓"].astype(bool).shift(1, fill_value=False)).sum()),
        "最大板块毛盈亏占比": float(max(share.values())) if share else np.nan,
        "最大占比板块": max(share, key=share.get) if share else "",
    }
    std = float(alpha.std())
    row["alpha_t"] = float(alpha.mean() / std * np.sqrt(len(alpha))) if std > 0 else float("nan")
    return row, annual


def _simulate(target, inputs):
    keys = (
        "open_price", "close_price", "adjusted_open", "adjusted_open_delta", "adjusted_high",
        "adjusted_low", "rolls", "contract_codes",
    )
    return simulate_target_lots(
        target, *(inputs[key] for key in keys), inputs["universe"], inputs["multipliers"],
        inputs["open_fee"], inputs["close_fee"], inputs["roll_close_fee"], inputs["slippage"],
        initial_capital=100_000_000.0, margin_rate=0.20, exposure_limit=S.GROSS,
        drawdown_limit=S.DRAWDOWN_LIMIT, halt_days=S.HALT_DAYS,
    )


def _load(end: int):
    end = int(end)
    if end <= 2021:
        data = load_research(1.0)
        if data.returns.empty or data.returns.index.max() >= pd.Timestamp("2022-01-01"):
            raise C.HoldoutViolation("研究数据越过 2021")
        return data
    if end > 2025:
        raise C.HoldoutViolation("不读 2026")
    return load_market_clock_year(end)


def _book(data, years: tuple[int, ...], period) -> None:
    inputs = ledger_inputs(data, years, pd.Timestamp(f"{max(years)}-12-31"))
    every = sorted({symbol for members in data.universe.values() for symbol in members})
    target = S.targets(
        data.factors.signed, data.factors.bars, data.universe, data.returns.index, every,
    ).reindex(index=inputs["index"], columns=inputs["columns"]).fillna(0.0)
    daily, positions, events = _simulate(target, inputs)
    summary, _annual = _write_results(
        period, S, daily, positions, events, inputs["symbols"], 100_000_000.0, ("全品种",), "commodity",
    )
    market = pool_open_return(data, inputs["universe"], inputs["index"])
    row, annual = summarize(S.NAME, daily, positions, market, inputs["adjusted_open_delta"], sectors())
    print(
        f"{period}  累计 {summary['累计净收益']:.3%}  年化 {row['年化净收益']:.2%}"
        f"  净Sharpe {summary['Sharpe']:.2f}  回撤 {row['最大回撤']:.2%}"
        f"  beta {row['相对池子beta']:.2f}  年化alpha {row['相对池子年化alpha']:.2%}"
        f"  t {row['alpha_t']:.2f}  年化成本 {row['年化成本占权益']:.2%}",
        flush=True,
    )
    for item in annual:
        print(f"  {item['年份']}  净收益 {item['净收益']:.2%}  Sharpe {item['Sharpe']:.2f}", flush=True)


def _period_for(start: int, end: int):
    if (start, end) == (2016, 2021):
        return "research"
    if (start, end) == (2023, 2025):
        return "oos"
    if start == end:
        return start
    return (start, end)


def _window_name(start: int, end: int) -> str:
    return f"{S.ID}_{start}" if start == end else f"{S.ID}_{start}_{end}"


def _finished(name: str, ledger) -> bool:
    return any(
        entry.get("strategy") == name and entry.get("status") == "finished"
        for entry in C.read_ledger(ledger)
    )


def _show_saved(period) -> bool:
    folder = period if isinstance(period, str) else (
        f"{period[0]}-{period[1]}" if isinstance(period, tuple) else str(period)
    )
    summary = C.RUNS_DIR / "strategies" / S.ID / "commodity" / folder / "summary.csv"
    if not summary.is_file():
        return False
    print(pd.read_csv(summary).to_string(index=False), flush=True)
    print(f"这段已经跑过，结果在 {summary.parent}", flush=True)
    return True


def _oos_spec() -> dict:
    return {"id": S.ID, "frozen_at": "2026-10-20", "chosen": dict(S.load_config()["parameters"])}


def run_span(start: int, end: int, by_year: bool = False, ledger=None) -> None:
    """一段年份一本账。2022 及以后先记台账，已经跑完的只打印。"""
    start, end = int(start), int(end)
    if start > end or start < 2016 or end > 2025:
        raise C.HoldoutViolation(f"回测年份必须落在 2016–2025，收到 {start}–{end}")
    if by_year and start != end:
        for year in range(start, end + 1):
            run_span(year, year, ledger=ledger)
        return
    if end <= 2021:
        period = _period_for(start, end)
        _book(_load(end), tuple(range(start, end + 1)), period)
        if period == "research":
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "protocol.json").write_text(
                json.dumps(S.load_config(), ensure_ascii=False, indent=2), encoding="utf-8",
            )
        return
    name = _window_name(start, end)
    saved = _period_for(start, end)
    if _show_saved(saved):
        if not _finished(name, ledger):
            C.append_ledger({
                "kind": "year_test" if end < 2023 else "final_evaluation",
                "status": "finished", "run_at": datetime.now().isoformat(timespec="seconds"),
                "strategy": name, "scored_start": start, "scored_end": end,
                "note": "结果文件已经在，没有重读行情。",
            }, ledger)
        return
    note = (
        f"当日分数且展期为正 {start}–{end}。"
        "这些年份的行情以前看过，规则没有在其上拟合，不是干净样本外。"
    )

    def run():
        _book(_load(end), tuple(range(start, end + 1)), saved)

    if end >= 2023:
        files = [Path(S.__file__), S.CONFIG_PATH]
        with C.final_evaluation(_oos_spec(), files, f"{end}-12-31", strategy=name, note=note, ledger=ledger):
            run()
        kind = "final_evaluation"
    else:
        C.append_ledger({
            "kind": "year_test", "status": "opened",
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "strategy": name, "test_start": f"{start}-01-01", "test_end": f"{end}-12-31", "note": note,
        }, ledger)
        run()
        kind = "year_test"
    C.append_ledger({
        "kind": kind, "status": "finished",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": name, "scored_start": start, "scored_end": end, "note": note,
    }, ledger)


def run_research() -> None:
    run_span(2016, 2021)


def show_or_run_2022() -> None:
    run_span(2022, 2022)


if __name__ == "__main__":
    run_research()
