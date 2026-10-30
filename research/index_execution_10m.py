"""冻结股指熊市空头的 1000 万容量与成交规则试算。

默认只读 2015–2021 的 IF/IH/IC。--early-if 只读 IF 上市至 IH/IC 上市前的专用数据，计分从 2011 年起。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import (
    INDEX_RESEARCH_END,
    INDEX_SYMBOLS,
    INITIAL_CAPITAL,
    _load_index_research_inputs,
    _simulate_index_targets,
    max_drawdown,
    performance,
    sharpe_ratio,
)
from research.index_rally_residual import load_opening_quotes, l1_slippage
from strategies.index_bear_reversal import BOOK
from strategies.index_close_location import market_close_location
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_execution_10m.json"
EARLY_CONFIG_PATH = C.CONFIG_DIR / "research" / "if_execution_10m_2011.json"
OUT = C.RUNS_DIR / "index" / "index_execution_10m"
EARLY_OUT = C.RUNS_DIR / "index" / "if_execution_10m_2011"
CAPITALS = (10_000_000.0, 100_000_000.0)
EARLY_CAPITAL = 10_000_000.0
MODELS = ("tick", "l1_30m", "auction_l1")


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022" not in sample
        or "不读 2023–2025" not in sample
        or "不读 2026" not in sample
        or protocol.get("capitals") != [int(value) for value in CAPITALS]
        or tuple(protocol.get("execution_models", {})) != MODELS
        or "当日实际集合竞价成交量" not in protocol.get("execution_models", {}).get("auction_l1", "")
        or "事后容量估算" not in protocol.get("limitations", "")
        or "index_close_location_bear_short" not in protocol.get("book", "")
    ):
        raise ValueError("1000 万成交规格与实现或样本边界不一致")
    BOOK.declared()
    return protocol


def auction_allowance(
    quotes: pd.DataFrame, scored: pd.DatetimeIndex, symbols=None,
) -> pd.DataFrame:
    if isinstance(symbols, str):
        raise TypeError("symbols 要用序列，不能是单个字符串")
    names = list(INDEX_SYMBOLS if symbols is None else symbols)
    frame = quotes.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(frame["trading_date"], what="集合竞价历史成交量")
    if frame.duplicated(["trading_date", "symbol"]).any():
        raise ValueError("集合竞价日成交量有重复品种日")
    wide = frame.pivot(index="trading_date", columns="symbol", values="auction_volume")
    wide = wide.reindex(columns=names).sort_index()
    if wide.isna().any().any() or not np.isfinite(wide.to_numpy(dtype="float64")).all():
        raise ValueError("集合竞价历史成交量覆盖不完整")
    if (wide < 0).any().any():
        raise ValueError("集合竞价成交量不能为负")
    if not scored.isin(wide.index).all():
        missing = scored.difference(wide.index)
        raise ValueError(f"集合竞价成交量缺少计分日：{missing[:5].tolist()}")
    allowance = np.floor(wide.reindex(index=scored) * 0.10)
    if allowance.isna().any().any() or (allowance < 0).any().any():
        raise ValueError("集合竞价限额计算不完整")
    return allowance.astype("float64")


def _metrics(daily: pd.DataFrame, capital: float, model: str, insufficient: int) -> dict:
    net = daily["净收益"].fillna(0.0)
    metrics = performance(net)
    tick_basis = float(daily["开盘一跳基准费用"].sum())
    slip_cost = float(daily["开盘滑点成本"].sum())
    opening_lots = int(daily["开盘成交手数"].sum())
    auction_lots = int(daily["集合竞价成交手数"].sum())
    return {
        "初始资金": capital,
        "成交口径": model,
        "起点": str(daily.index.min().date()),
        "终点": str(daily.index.max().date()),
        "年化收益": metrics["ann_return"],
        "年化波动": metrics["ann_vol"],
        "最大回撤": max_drawdown(daily["权益"], capital),
        "Sharpe": sharpe_ratio(net),
        "Calmar比率": metrics["calmar"],
        "平均敞口": float(daily["敞口倍数"].fillna(0.0).mean()),
        "年化成本占权益": float(daily["交易费用"].sum() / capital / (len(net) / C.TRADING_DAYS_PER_YEAR)),
        "开盘成交手数": opening_lots,
        "集合竞价成交手数": auction_lots,
        "集合竞价成交占比": auction_lots / opening_lots if opening_lots else np.nan,
        "开盘滑点加权跳数": slip_cost / tick_basis if tick_basis else np.nan,
        "不足一手事件数": insufficient,
        "开仓受限次数": int(daily["开仓受限数"].sum()),
    }


def _annual_rows(
    daily: pd.DataFrame, capital: float, model: str, events: pd.DataFrame
) -> list[dict]:
    rows = []
    for year, part in daily.groupby(daily.index.year, sort=True):
        opening = float(part["期初权益"].iloc[0])
        net = part["净收益"].fillna(0.0)
        opening_lots = int(part["开盘成交手数"].sum())
        auction_lots = int(part["集合竞价成交手数"].sum())
        tick_basis = float(part["开盘一跳基准费用"].sum())
        year_events = events.loc[pd.to_datetime(events["交易日"]).dt.year == year] if not events.empty else events
        rows.append({
            "初始资金": capital,
            "成交口径": model,
            "年份": int(year),
            "净收益": float(part["权益"].iloc[-1] / opening - 1.0) if opening else np.nan,
            "Sharpe": sharpe_ratio(net),
            "年度内最大回撤": max_drawdown(part["权益"], opening),
            "年度成本占年初权益": float(part["交易费用"].sum() / opening) if opening else np.nan,
            "开盘成交手数": opening_lots,
            "集合竞价成交手数": auction_lots,
            "集合竞价成交占比": auction_lots / opening_lots if opening_lots else np.nan,
            "开盘滑点加权跳数": (
                float(part["开盘滑点成本"].sum()) / tick_basis if tick_basis else np.nan
            ),
            "不足一手事件数": int(year_events["事件"].eq("不足一手").sum()) if not year_events.empty else 0,
            "开仓受限次数": int(part["开仓受限数"].sum()),
        })
    return rows


def _assert_official_tick_match(daily: pd.DataFrame) -> None:
    path = C.RUNS_DIR / "strategies" / BOOK.ID / "research" / "daily.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结策略正式研究账本，无法做逐日核对：{path}")
    official = pd.read_csv(path, parse_dates=["交易日"]).set_index("交易日").sort_index()
    if not official.index.equals(daily.index):
        raise AssertionError("100M tick 账本日期与冻结策略正式研究账本不一致")
    for field in ("净收益", "权益", "交易费用", "毛盈亏"):
        if field not in official or not np.allclose(
            official[field].to_numpy(dtype="float64"),
            daily[field].to_numpy(dtype="float64"),
            rtol=1e-11,
            atol=1e-6,
        ):
            raise AssertionError(f"100M tick 账本与正式研究账本的 {field} 不逐日一致")


def run() -> None:
    protocol = load_protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    index, bars, contracts, _carry, ticks = _load_index_research_inputs()
    open_to_open = (bars["openw"].shift(-1) - bars["openw"]) / bars["open"]
    scored = open_to_open.dropna(how="any").index
    C.assert_no_holdout_dates(scored, what="1000 万股指成交试算计分日")
    if scored.min() < pd.Timestamp("2015-04-16") or scored.max() > INDEX_RESEARCH_END:
        raise C.HoldoutViolation("1000 万成交试算超出 IF/IH/IC 研究期")
    strategy_bars = {
        key: frame.reindex(index=index, columns=INDEX_SYMBOLS)
        for key, frame in bars.items()
    }
    market_clv = market_close_location(strategy_bars, index)
    targets = BOOK.targets(market_clv, strategy_bars)

    quotes = load_opening_quotes(OUT / "l1_opening_30m.parquet")
    quotes["trading_date"] = pd.to_datetime(quotes["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(quotes["trading_date"], what="股指开盘 L1 摘要")
    if quotes["trading_date"].max() > INDEX_RESEARCH_END:
        raise C.HoldoutViolation("L1 开盘摘要越过 2021-12-31")
    slip, bad_cells, _volume = l1_slippage(quotes, bars, contracts, scored)
    allowance = auction_allowance(quotes, scored)
    slip["auction_lots"] = allowance
    print(f"L1 合约或开盘价不匹配的品种日：{bad_cells}", flush=True)

    summary_rows, annual_rows = [], []
    for capital in CAPITALS:
        for model in MODELS:
            if model == "tick":
                open_slippage = None
            elif model == "l1_30m":
                open_slippage = {"buy": slip["buy"], "sell": slip["sell"]}
            else:
                open_slippage = slip
            daily, positions, events = _simulate_index_targets(
                targets,
                scored,
                bars,
                contracts,
                ticks,
                INDEX_RESEARCH_END.year,
                INDEX_SYMBOLS,
                exposure_limit=BOOK.GROSS,
                drawdown_limit=BOOK.DRAWDOWN_LIMIT,
                halt_days=BOOK.HALT_DAYS,
                initial_capital=capital,
                open_limits=True,
                open_slippage=open_slippage,
            )
            insufficient = int(events["事件"].eq("不足一手").sum()) if not events.empty else 0
            summary = _metrics(daily, capital, model, insufficient)
            summary_rows.append(summary)
            annual_rows.extend(_annual_rows(daily, capital, model, events))

            name = f"capital_{int(capital)}_{model}"
            directory = OUT / name
            directory.mkdir(parents=True, exist_ok=True)
            daily.to_csv(directory / "daily.csv", encoding="utf-8-sig")
            positions.to_csv(directory / "positions.csv", index=False, encoding="utf-8-sig")
            events.to_csv(directory / "events.csv", index=False, encoding="utf-8-sig")
            if capital == INITIAL_CAPITAL and model == "tick":
                _assert_official_tick_match(daily)
            print(
                f"{int(capital):,} / {model}: {summary['年化收益']:.2%} 年化, "
                f"Sharpe {summary['Sharpe']:.2f}, MDD {summary['最大回撤']:.2%}",
                flush=True,
            )

    pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual_rows).to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(f"结果：{OUT}", flush=True)


def load_early_protocol(path: Path = EARLY_CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022" not in sample
        or "不读 2023–2025" not in sample
        or "不读 2026" not in sample
        or "2011-01-01" not in sample
        or "2015-04-16" not in sample
        or "if_before_ih_ic" not in sample
        or protocol.get("capital") != int(EARLY_CAPITAL)
        or tuple(protocol.get("execution_models", {})) != MODELS
        or "当日实际集合竞价成交量" not in protocol.get("execution_models", {}).get("auction_l1", "")
        or "正好落在连续交易起点" not in protocol.get("execution_models", {}).get("auction_l1", "")
        or "开盘报价缺失" not in protocol.get("execution_models", {}).get("auction_l1", "")
        or "事后容量估算" not in protocol.get("limitations", "")
    ):
        raise ValueError("早期 IF 1000 万成交规格与实现或样本边界不一致")
    BOOK.declared()
    return protocol


def run_early_if() -> None:
    """1000 万、三种开盘成交口径，只做 IF，计分 2011 至 IH/IC 上市前。"""
    from research.if_before_ih_ic import (
        EXEC_START, IH_IC_LISTING, early_inputs, early_ledger, load_early_opening_quotes,
    )

    protocol = load_early_protocol()
    EARLY_OUT.mkdir(parents=True, exist_ok=True)
    (EARLY_OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    inputs = early_inputs()
    scored = inputs["scored"][inputs["scored"] >= EXEC_START]
    C.assert_no_holdout_dates(scored, what="早期 IF 1000 万成交试算")
    if scored.empty or scored.min() < EXEC_START or scored.max() >= IH_IC_LISTING:
        raise C.HoldoutViolation("早期 IF 成交试算越出 2011 至 IH/IC 上市前")
    held = (inputs["side"] * inputs["scale"]).shift(1).fillna(0.0)
    bundle = {**inputs, "scored": scored}
    quotes = load_early_opening_quotes(inputs["contracts"])
    quotes["trading_date"] = pd.to_datetime(quotes["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(quotes["trading_date"], what="早期 IF 开盘报价")
    if quotes["trading_date"].max() >= IH_IC_LISTING:
        raise C.HoldoutViolation("早期 IF 开盘报价越过 IH/IC 上市日")
    contracts = inputs["contracts"].reindex(scored).to_frame("IF")
    quote_slip, _bad, _volume = l1_slippage(
        quotes, inputs["bars"], contracts, scored, symbols=("IF",)
    )
    allowance = auction_allowance(quotes, scored, symbols=("IF",))
    gap_days = pd.DatetimeIndex(
        quotes.loc[quotes["open_gap"], "trading_date"]
    ).intersection(scored)
    intact = scored.difference(gap_days)
    for side in ("buy", "sell"):
        values = quote_slip[side].reindex(intact)["IF"].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError("报价完整日的合约、开盘价或 L1 成本对不上")
    if len(gap_days):
        from tfcta.costs import slippage_tables

        prices = inputs["bars"]["open"].reindex(index=scored, columns=["IF"])
        tick_rate = slippage_tables(
            inputs["ticks"][inputs["ticks"].year < IH_IC_LISTING.year], prices
        )
        rate = tick_rate.reindex(gap_days)["IF"]
        if not np.isfinite(rate.to_numpy(dtype="float64")).all():
            raise ValueError("开盘报价缺失日没有 1 跳成本")
        for side in ("buy", "sell"):
            quote_slip[side].loc[gap_days, "IF"] = rate.to_numpy()
    print(f"开盘报价缺失 {len(gap_days)} / {len(scored)} 天", flush=True)
    slips = {
        "tick": None,
        "l1_30m": {"buy": quote_slip["buy"], "sell": quote_slip["sell"]},
        "auction_l1": {
            "buy": quote_slip["buy"], "sell": quote_slip["sell"], "auction_lots": allowance,
        },
    }
    summary_rows, annual_rows = [], []
    for model in MODELS:
        daily, positions, events = early_ledger(
            held, bundle, initial_capital=EARLY_CAPITAL, open_slippage=slips[model],
        )
        insufficient = int(events["事件"].eq("不足一手").sum()) if not events.empty else 0
        summary = _metrics(daily, EARLY_CAPITAL, model, insufficient)
        intact = daily.index.difference(gap_days)
        opening = float(daily.loc[intact, "开盘成交手数"].sum())
        auction = float(daily.loc[intact, "集合竞价成交手数"].sum())
        summary["开盘报价缺失天数"] = int(len(gap_days))
        summary["报价缺失日开盘手数"] = int(daily.reindex(gap_days)["开盘成交手数"].fillna(0).sum())
        summary["报价完整日集合竞价占比"] = auction / opening if opening else np.nan
        summary_rows.append(summary)
        annual_rows.extend(_annual_rows(daily, EARLY_CAPITAL, model, events))
        directory = EARLY_OUT / f"capital_{int(EARLY_CAPITAL)}_{model}"
        directory.mkdir(parents=True, exist_ok=True)
        daily.to_csv(directory / "daily.csv", encoding="utf-8-sig")
        positions.to_csv(directory / "positions.csv", index=False, encoding="utf-8-sig")
        events.to_csv(directory / "events.csv", index=False, encoding="utf-8-sig")
        print(
            f"{int(EARLY_CAPITAL):,} / {model}: {summary['年化收益']:.2%} 年化, "
            f"Sharpe {summary['Sharpe']:.2f}, MDD {summary['最大回撤']:.2%}",
            flush=True,
        )
    pd.DataFrame(summary_rows).to_csv(EARLY_OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual_rows).to_csv(EARLY_OUT / "annual.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(f"结果：{EARLY_OUT}", flush=True)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="股指 1000 万成交口径试算")
    parser.add_argument("--early-if", action="store_true", help="只做 IF，计分 2011 至 IH/IC 上市前")
    args = parser.parse_args(argv)
    if args.early_if:
        run_early_if()
    else:
        run()


if __name__ == "__main__":
    main()
