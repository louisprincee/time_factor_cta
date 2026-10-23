"""按冻结规则复现已看过的 2014–2015 商品压力窗口；不得据此调参。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backtest.engine import StudyData, ledger_inputs, load_roll_calendar, simulate_target_lots, write_results
from research.market_clock_slow_short import pool_open_return, sectors, summarize
from strategies.catalog import FROZEN_BOOKS, book_named
from tfcta import config as C
from tfcta.costs import fee_tables, load_fees, load_ticks, slippage_tables
from tfcta.data import build_universe, load_universe
from tfcta.factors import SignalSet, load

CONFIG_PATH = C.CONFIG_DIR / "research" / "stress_2014_2015_reproduction.json"
START = pd.Timestamp("2014-01-01")
END = pd.Timestamp("2015-12-31")
YEARS = (2014, 2015)
# 只重跑仍冻结的慢趋势策略；原多头书的历史复现结果不在本脚本重跑
REPRODUCED = ("market_clock_slow_short",)


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "declared_reproduction_no_tuning"
        or protocol.get("books") != list(REPRODUCED)
        or "not be used for tuning" not in protocol.get("purpose", "")
        or protocol.get("universe", {}).get("expected_pool_widths") != {"2014": 11, "2015": 17}
    ):
        raise ValueError("压力窗口复现规格与冻结策略不一致")
    return protocol


def stress_universe(stats: pd.DataFrame, protocol: dict) -> dict[int, list[str]]:
    """用前一年统计量重建 2014、2015 的品种池，并核对预登记池宽。"""
    C.assert_no_holdout_dates(pd.to_datetime(stats["year"].astype(str) + "-01-01"),
                              what="压力窗口品种池统计")
    params = protocol["universe"]
    universe, _ = build_universe(
        stats,
        years=[2013, *YEARS],
        lookback_years=int(params["lookback_years"]),
        min_turnover=float(params["minimum_daily_turnover"]),
        min_valid_ratio=float(params["minimum_valid_day_ratio"]),
        min_valid_days=int(params["minimum_valid_days"]),
    )
    expected = {int(year): int(count) for year, count in params["expected_pool_widths"].items()}
    actual = {year: len(universe[year]) for year in YEARS}
    if actual != expected:
        raise ValueError(f"重建的压力窗口品种池与预登记宽度不符：{actual} != {expected}")
    return universe


def load_stress_data(protocol: dict) -> StudyData:
    stats_path = C.UNIVERSE_DIR / "yearly_stats.csv"
    if not stats_path.is_file():
        raise FileNotFoundError(f"找不到品种池统计缓存：{stats_path}")
    stats = pd.read_csv(stats_path)
    universe = stress_universe(stats, protocol)
    cached = load_universe()
    if universe[2015] != cached.get(2015):
        raise ValueError("2015 压力池与现存研究期品种池不一致")

    symbols = sorted({symbol for members in universe.values() for symbol in members})
    full = load(symbols)
    bars = {
        name: frame.loc[frame.index <= END].copy()
        for name, frame in full.bars.items()
    }
    signed = {
        name: frame.loc[frame.index <= END].copy()
        for name, frame in full.signed.items()
    }
    factors = SignalSet(bars=bars, signed=signed)
    open_price = bars["open"]
    returns = (bars["openw"].shift(-1) - bars["openw"]) / open_price.where(open_price > 0)
    returns = returns.loc[(returns.index >= pd.Timestamp("2013-01-01")) & (returns.index <= END)]
    returns = returns.loc[returns.notna().any(axis=1)]
    C.assert_no_holdout_dates(returns.index, what="压力窗口回测收益")

    fees = load_fees()
    fees = fees[fees["trading_date"] <= END]
    ticks = load_ticks()
    ticks = ticks[ticks["year"] < END.year]
    traded_open = open_price.reindex(index=returns.index, columns=symbols)
    open_fee, close_fee, roll_fee = fee_tables(fees, traded_open)
    slippage = slippage_tables(ticks, traded_open)

    calendar = load_roll_calendar(symbols, END.date())
    rolls = pd.DataFrame(False, index=returns.index, columns=symbols)
    for symbol, frame in calendar.items():
        if not frame.empty:
            days = returns.index.intersection(pd.DatetimeIndex(frame["trading_date"]))
            rolls.loc[days, symbol] = True
    data = StudyData(factors, returns, universe, rolls, open_fee, close_fee, roll_fee, slippage)
    data.roll_calendar = calendar
    return data


def run() -> None:
    protocol = load_protocol()
    frozen = {book.ID for book in FROZEN_BOOKS}
    removed = [name for name in protocol["books"] if name not in frozen]
    if removed:
        raise ValueError(f"这些策略已删掉，压力窗口复现不能重跑：{removed}；已有结果在 runs/strategies/<id>/commodity/2014-2015/")
    books = [book_named(name) for name in protocol["books"]]
    for book in books:
        book.frozen()
        output = C.RUNS_DIR / "strategies" / book.ID / "commodity" / "2014-2015"
        if (output / "summary.csv").exists():
            raise FileExistsError(f"压力窗口结果已存在，不重复运行：{output}")
    data = load_stress_data(protocol)
    inputs = ledger_inputs(data, YEARS, END)
    market = pool_open_return(data, inputs["universe"], inputs["index"])

    for book in books:
        targets = book.targets(data, inputs["symbols"]).reindex(
            index=inputs["index"], columns=inputs["columns"]
        ).fillna(0.0)
        daily, positions, events = simulate_target_lots(
            targets,
            *(inputs[key] for key in (
                "open_price", "close_price", "adjusted_open", "adjusted_open_delta",
                "adjusted_high", "adjusted_low", "rolls", "contract_codes",
            )),
            inputs["universe"], inputs["multipliers"],
            inputs["open_fee"], inputs["close_fee"], inputs["roll_close_fee"], inputs["slippage"],
            initial_capital=100_000_000.0, margin_rate=0.20, exposure_limit=book.GROSS,
            drawdown_limit=0.10, halt_days=10, contract_multipliers=inputs["contract_multipliers"],
        )
        summary, _ = write_results((YEARS[0], YEARS[-1]), book, daily, positions, events, inputs["symbols"])
        market_series = market.reindex(daily.index).fillna(0.0)
        row, _ = summarize(
            book.NAME, daily, positions, market_series,
            inputs["adjusted_open_delta"], sectors(),
        )
        print(
            f"{book.NAME} 2014–2015：年化 {row['年化净收益']:.2%} "
            f"Sharpe {summary['Sharpe']:.2f} 回撤 {row['最大回撤']:.2%} "
            f"beta {row['相对池子beta']:.2f} alpha {row['相对池子年化alpha']:.2%}",
            flush=True,
        )
    print("此窗口已看过，本次只复现冻结规则，不得据此调参。", flush=True)


if __name__ == "__main__":
    run()
