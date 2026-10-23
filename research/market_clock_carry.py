"""当日分数且展期为正：按区间跑两条冻结策略，算相对池子的 beta 和 alpha，管 2022 及以后的台账。

    python -m research.market_clock_carry          # 研究期 2011–2021
    python -m backtest.engine --research            # 同上，命令行入口

研究期不写台账。2022 及以后：策略没冻结直接拒绝；冻结了先记台账再读；已经打开过的窗口只打印旧结果，
结果文件丢了要先在台账里作废并写明原因才能重开。不读 2026。
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from backtest.engine import (
    INITIAL_CAPITAL, MARGIN_RATE, PERIODS, ledger_inputs, load_research, load_through_year,
    max_drawdown, newey_west_t, sharpe_ratio, simulate_target_lots, write_results,
)
from strategies import market_clock_carry as S
from strategies.catalog import BOOKS, book_named
from tfcta import config as C

NW_LAGS = 10


def sectors() -> dict[str, str]:
    table = json.loads((C.CONFIG_DIR / "research" / "time_factor_screen.json").read_text(encoding="utf-8"))
    return {symbol: name for name, members in table["sectors"].items() for symbol in members}


def pool_open_return(data, universe: dict, index) -> pd.Series:
    """池内开盘到开盘收益的等权平均，与账本的逐日净收益同一口径。"""
    pool = S.pool_mask(universe, index, data.returns.columns)
    return data.returns.reindex(index).where(pool).mean(axis=1).fillna(0.0)


def summarize(label: str, daily: pd.DataFrame, positions: pd.DataFrame, market: pd.Series,
              delta: pd.DataFrame, sector_of: dict) -> tuple[dict, list[dict]]:
    """净收益对池子收益回归：beta 是斜率，逐日 alpha 是残差加截距，年化 alpha 是逐日 alpha 均值乘 242。"""
    net = daily["净收益"].fillna(0.0)
    gross = (daily["毛盈亏"] / daily["期初权益"]).fillna(0.0)
    market = market.reindex(net.index).fillna(0.0)
    beta = float(net.cov(market) / market.var())
    alpha = net - beta * market
    month = net.index.to_period("M")
    strat_m = (1 + net).groupby(month).prod() - 1
    pool_m = (1 + market).groupby(month).prod() - 1
    up, down = pool_m > 0, pool_m < 0
    equity = daily["权益"]
    share = {}
    if not positions.empty:
        moves = delta.stack()
        moves.index.names = ["交易日", "品种"]
        held = positions.set_index(["交易日", "品种"])
        pnl = held["手数"] * held["合约乘数"] * moves.reindex(held.index).to_numpy()
        by_sector = pnl.groupby(pnl.index.get_level_values("品种").map(sector_of)).sum()
        share = (by_sector / by_sector.abs().sum()).to_dict()
    annual = []
    for year in sorted(set(net.index.year)):
        part = net[net.index.year == year]
        annual.append({
            "变体": label, "年份": year, "净收益": float((1 + part).prod() - 1),
            "Sharpe": sharpe_ratio(part), "相对池子alpha": float(alpha[alpha.index.year == year].sum()),
        })
    row = {
        "变体": label,
        "净Sharpe": sharpe_ratio(net),
        "毛Sharpe": sharpe_ratio(gross),
        "年化净收益": float((equity.iloc[-1] / INITIAL_CAPITAL) ** (PERIODS / len(net)) - 1),
        "年化波动": float(net.std() * np.sqrt(PERIODS)),
        "最大回撤": max_drawdown(equity, INITIAL_CAPITAL),
        "年化成本占权益": float((daily["交易费用"] / daily["期初权益"]).mean() * PERIODS),
        "平均敞口倍数": float(daily["敞口倍数"].mean()),
        "最大敞口倍数": float(daily["敞口倍数"].max()),
        "为正年份": int(sum(item["净收益"] > 0 for item in annual)),
        "相对池子beta": beta,
        "相对池子年化alpha": float(alpha.mean() * PERIODS),
        "alpha_t": newey_west_t(alpha, NW_LAGS),
        "alpha为正年份": int(sum(item["相对池子alpha"] > 0 for item in annual)),
        "上涨月捕获": float(strat_m[up].mean() / pool_m[up].mean()),
        "下跌月捕获": float(strat_m[down].mean() / pool_m[down].mean()),
        "熔断次数": int((daily["停止新开仓"].astype(bool) & ~daily["停止新开仓"].astype(bool).shift(1, fill_value=False)).sum()),
        "最大板块毛盈亏占比": float(max(share.values())) if share else np.nan,
        "最大占比板块": max(share, key=share.get) if share else "",
    }
    return row, annual


def simulate(target, inputs, book):
    keys = (
        "open_price", "close_price", "adjusted_open", "adjusted_open_delta", "adjusted_high",
        "adjusted_low", "rolls", "contract_codes",
    )
    return simulate_target_lots(
        target, *(inputs[key] for key in keys), inputs["universe"], inputs["multipliers"],
        inputs["open_fee"], inputs["close_fee"], inputs["roll_close_fee"], inputs["slippage"],
        initial_capital=INITIAL_CAPITAL, margin_rate=MARGIN_RATE, exposure_limit=book.GROSS,
        drawdown_limit=S.DRAWDOWN_LIMIT, halt_days=S.HALT_DAYS,
        contract_multipliers=inputs["contract_multipliers"],
    )


def _load(end: int, start: int | None = None):
    end = int(end)
    if end > C.DEFAULT_OOS_END.year:
        raise C.HoldoutViolation("不读 2026")
    early = start is not None and int(start) < C.STUDY_START.year
    if early and end > C.RESEARCH_END.year:
        raise ValueError("2016 年以前起算的区间只能在研究期里，不能跨到 2022 及以后")
    if end <= C.RESEARCH_END.year:
        data = load_research(1.0, first_year=int(start) if early else None)
        if data.returns.empty or data.returns.index.max() >= pd.Timestamp(C.HOLDOUT_START):
            raise C.HoldoutViolation("研究数据越过 2021")
        return data
    return load_through_year(end)


def _book(data, years: tuple[int, ...], period, book) -> dict:
    inputs = ledger_inputs(data, years, pd.Timestamp(f"{max(years)}-12-31"))
    every = sorted({symbol for members in data.universe.values() for symbol in members})
    target = book.targets(data, every).reindex(
        index=inputs["index"], columns=inputs["columns"]).fillna(0.0)
    daily, positions, events = simulate(target, inputs, book)
    summary, _ = write_results(period, book, daily, positions, events, inputs["symbols"])
    market = pool_open_return(data, inputs["universe"], inputs["index"])
    row, annual = summarize(book.NAME, daily, positions, market, inputs["adjusted_open_delta"], sectors())
    print(
        f"{book.NAME}  {period}  累计 {summary['累计净收益']:.3%}  年化 {row['年化净收益']:.2%}"
        f"  净Sharpe {summary['Sharpe']:.2f}  回撤 {row['最大回撤']:.2%}"
        f"  beta {row['相对池子beta']:.2f}  年化alpha {row['相对池子年化alpha']:.2%}"
        f"  NW t {row['alpha_t']:.2f}  年化成本 {row['年化成本占权益']:.2%}",
        flush=True,
    )
    for item in annual:
        print(f"  {item['年份']}  净收益 {item['净收益']:.2%}  Sharpe {item['Sharpe']:.2f}", flush=True)
    return row


def _period_for(start: int, end: int):
    if (start, end) == (C.EARLY_START.year, C.RESEARCH_END.year):
        return "research"
    if (start, end) == (C.STRICT_OOS_START.year, C.DEFAULT_OOS_END.year):
        return "oos"
    if start == end:
        return start
    return (start, end)


def _window_name(start: int, end: int, book) -> str:
    return f"{book.ID}_{start}" if start == end else f"{book.ID}_{start}_{end}"


def _window_state(name: str, ledger) -> str:
    """台账里这个窗口最后的状态：none、opened、finished 或 void。"""
    state = "none"
    for entry in C.read_ledger(ledger):
        if entry.get("strategy") == name:
            state = "void" if entry.get("kind") == "void" else entry.get("status", state)
    return state


def _show_saved(period, book) -> bool:
    folder = period if isinstance(period, str) else (
        f"{period[0]}-{period[1]}" if isinstance(period, tuple) else str(period)
    )
    summary = C.RUNS_DIR / "strategies" / book.ID / "commodity" / folder / "summary.csv"
    if not summary.is_file():
        return False
    print(pd.read_csv(summary).to_string(index=False), flush=True)
    print(f"这段已经跑过，结果在 {summary.parent}", flush=True)
    return True


def _chosen(book):
    """点了名就只跑那一条。不点名时返回 None，调用方跑全部。"""
    if book is None:
        return None
    if isinstance(book, str):
        return book_named(book)
    return book


def run_span(start: int, end: int, by_year: bool = False, ledger=None, book=None) -> None:
    """一段年份一本账。2022 及以后要先冻结，先记台账再读；打开过的窗口只打印。"""
    start, end = int(start), int(end)
    book = _chosen(book)
    if start > end or start < C.EARLY_START.year or end > C.DEFAULT_OOS_END.year:
        raise C.HoldoutViolation(f"回测年份必须落在 {C.EARLY_START.year}–2025，收到 {start}–{end}。不读 2026。")
    if by_year and start != end:
        for year in range(start, end + 1):
            run_span(year, year, ledger=ledger, book=book)
        return
    books = (book,) if book is not None else BOOKS
    period = _period_for(start, end)
    if end <= C.RESEARCH_END.year:
        data = _load(end, start)
        for item in books:
            _book(data, tuple(range(start, end + 1)), period, item)
            if period == "research":
                out = C.RUNS_DIR / "research" / item.ID
                out.mkdir(parents=True, exist_ok=True)
                (out / "protocol.json").write_text(
                    json.dumps(item.load_config(), ensure_ascii=False, indent=2), encoding="utf-8",
                )
        return
    configs = {item.ID: item.frozen() for item in books}   # 任何一条没冻结，整批都不开、不记台账
    for item in books:
        name = _window_name(start, end, item)
        state = _window_state(name, ledger)
        if state in ("opened", "finished") and _show_saved(period, item):
            if state == "opened":
                C.append_ledger({
                    "kind": "year_test" if end < C.STRICT_OOS_START.year else "final_evaluation",
                    "status": "finished", "run_at": datetime.now().isoformat(timespec="seconds"),
                    "strategy": name, "scored_start": start, "scored_end": end,
                    "note": "结果文件已经在，没有重读行情。",
                }, ledger)
            continue
        if state in ("opened", "finished"):
            raise C.HoldoutViolation(
                f"{name} 在台账里已经打开过，但结果文件不在。要重跑先在台账里作废（追加 kind=void）并写明原因。"
            )
        config = configs[item.ID]
        files = item.FILES
        note = "重开，之前的结果已作废" if state == "void" else ""

        def run(item=item):
            _book(_load(end), tuple(range(start, end + 1)), period, item)

        if end >= C.STRICT_OOS_START.year:
            spec = {"id": item.ID, "frozen_at": config["frozen_at"], "chosen": config["chosen"]}
            with C.final_evaluation(spec, files, f"{end}-12-31", strategy=name, note=note,
                                    ledger=ledger, oos_start=f"{max(start, C.STRICT_OOS_START.year)}-01-01"):
                run()
            kind = "final_evaluation"
        else:
            C.append_ledger({
                "kind": "year_test", "status": "opened",
                "run_at": datetime.now().isoformat(timespec="seconds"),
                "strategy": name, "frozen_at": config["frozen_at"], "chosen": config["chosen"],
                "test_start": f"{start}-01-01", "test_end": f"{end}-12-31",
                "sha256": C.file_hashes(files), "note": note,
            }, ledger)
            run()
            kind = "year_test"
        C.append_ledger({
            "kind": kind, "status": "finished",
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "strategy": name, "scored_start": start, "scored_end": end, "note": note,
        }, ledger)


if __name__ == "__main__":
    run_span(C.EARLY_START.year, C.RESEARCH_END.year)
