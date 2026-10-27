"""按预登记口径把多头基线和慢趋势策略迁移到 IF、IH、IC 研究期检验。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from backtest.engine import (
    INITIAL_CAPITAL,
    MARGIN_RATE,
    max_drawdown,
    newey_west_t,
    performance,
    plot_performance,
    sharpe_ratio,
    simulate_target_lots,
)
from strategies import commodity_original as S
from tfcta import config as C
from tfcta.costs import (
    INDEX_MULTIPLIER,
    estimate_ticks,
    fee_tables,
    load_index_fees,
    slippage_tables,
)
from tfcta.data import add_intraday_coords, daily_bars, load_shard
from tfcta.factors import (
    day_codes_of,
    duration_factors,
    intraday_abs_diff,
    raw_price_path,
    report_factors,
    rolling_threshold,
    timestamp_factors,
)

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_market_clock_2015_2021.json"
# 登记时的只做多书 market_clock_pool 已在 2026-10-23 删掉；它就是两条空头策略共用的多头腿，这里按多头腿单独算
LONG_LEG = SimpleNamespace(ID="market_clock_pool", NAME="按池宽1倍多头腿", GROSS=S.GROSS)
OUT = C.RUNS_DIR / "index" / "market_clock_transfer_2015_2021"
SYMBOLS = ("IF", "IH", "IC")
FACTORS = ("ts_low", "ts_high", "dfp_top3", "ts_low_pm", "ts_high_pm")
SCORE_START = pd.Timestamp("2015-04-16")
SCORE_END = pd.Timestamp("2021-12-31")
WARMUP_START = pd.Timestamp("2015-01-01")
THRESHOLD_LOOKBACK = 250
THRESHOLD_PCT = 55.0


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected_signs = {name: C.FACTOR_SIGNS[name] for name in FACTORS}
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or protocol.get("instruments") != list(SYMBOLS)
        or protocol.get("books") != [LONG_LEG.ID, S.BOOK.ID]
        or "Do not read 2022" not in protocol.get("sample", "")
        or "or 2026" not in protocol.get("sample", "")
        or str(SCORE_START.date()) not in protocol.get("sample", "")
        or protocol.get("parameters", {}).get("factors") != list(FACTORS)
        or protocol.get("parameters", {}).get("factor_signs") != expected_signs
        or protocol.get("parameters", {}).get("short_trend_windows") != list(S.TREND_WINDOWS)
        or protocol.get("minimum_valid_instruments_for_market_score") != len(SYMBOLS)
    ):
        raise ValueError("股指迁移研究登记与因子、样本边界或当前策略不一致")
    return protocol


def _daily_factor_values(frame: pd.DataFrame) -> pd.DataFrame:
    """按冻结策略的 N250/M55 口径计算日内时戳与 DFP。"""
    frame = add_intraday_coords(frame)
    codes, _ = day_codes_of(frame)
    price = raw_price_path(frame)
    price_threshold = rolling_threshold(
        intraday_abs_diff(price, codes), codes, THRESHOLD_LOOKBACK, THRESHOLD_PCT
    )
    dfp = duration_factors(
        frame, THRESHOLD_LOOKBACK, THRESHOLD_PCT, thr_p=price_threshold
    )
    timestamps = timestamp_factors(frame)
    report = report_factors(
        frame,
        THRESHOLD_LOOKBACK,
        THRESHOLD_PCT,
        price_threshold=price_threshold,
    )
    return pd.concat(
        [timestamps, dfp[["dfp_top3"]], report[["ts_low_pm", "ts_high_pm"]]],
        axis=1,
    ).reindex(columns=FACTORS)


def _load_index_inputs():
    bars_by_symbol = {}
    contracts_by_symbol = {}
    factor_by_symbol = {}
    ticks = []

    for symbol in SYMBOLS:
        minute = load_shard(
            symbol,
            directory=C.INDEX_RESEARCH_DIR,
            columns=list(dict.fromkeys([*C.FACTOR_FIELDS, *C.PRICE_FIELDS])),
        )
        oi = load_shard(
            symbol,
            directory=C.INDEX_RESEARCH_DIR / "open_interest",
            columns=["dominant_id", "trading_date"],
        )
        if not minute.index.equals(oi.index):
            raise ValueError(f"{symbol} 分钟行情与主力合约代码时间索引不一致")
        minute = minute.copy()
        minute["dominant_id"] = oi["dominant_id"]
        dates = pd.to_datetime(minute["trading_date"]).dt.normalize()
        C.assert_no_holdout_dates(dates, what=f"{symbol} 股指分钟行情")

        factor_by_symbol[symbol] = _daily_factor_values(minute)
        bars_by_symbol[symbol] = daily_bars(minute)
        contracts = minute.groupby(dates, sort=True)["dominant_id"].first()
        contracts.index = pd.DatetimeIndex(contracts.index)
        contracts_by_symbol[symbol] = contracts

        tick_table = estimate_ticks(minute["close"], dates)
        tick_table.insert(0, "symbol", symbol)
        ticks.append(tick_table)

    index = pd.DatetimeIndex(sorted(set.union(*(
        set(frame.index) for frame in bars_by_symbol.values()
    ))))
    # IH、IC 2015-04-16 才上市：三品种比较从共同上市日开始，信号仍需滚动预热
    common = max(frame.index.min() for frame in bars_by_symbol.values())
    index = index[(index >= max(WARMUP_START, common)) & (index <= SCORE_END)]
    bars = {
        field: pd.DataFrame({
            symbol: frame[field].reindex(index)
            for symbol, frame in bars_by_symbol.items()
        }, index=index)
        for field in ("open", "openw", "close", "closew", "highw", "loww", "volume")
    }
    contracts = pd.DataFrame({
        symbol: series.reindex(index)
        for symbol, series in contracts_by_symbol.items()
    }, index=index)
    if contracts.isna().any().any():
        missing = contracts.columns[contracts.isna().any()].tolist()
        raise ValueError(f"主力合约代码覆盖不完整：{missing}")
    if bars["open"].isna().any().any() or bars["openw"].isna().any().any():
        raise ValueError("IF、IH、IC 的研究期开盘行情未完全对齐")

    raw = {
        factor: pd.DataFrame({
            symbol: panel[factor].reindex(index)
            for symbol, panel in factor_by_symbol.items()
        }, index=index)
        for factor in FACTORS
    }
    signed = {name: raw[name] * C.FACTOR_SIGNS[name] for name in FACTORS}
    return index, bars, contracts, signed, pd.concat(ticks, ignore_index=True)


def load_index_carry(index: pd.DatetimeIndex) -> pd.DataFrame:
    root = C.INDEX_DATA_ROOT / "external_rqdata" / "roll_yield"
    carry = {}
    for symbol in SYMBOLS:
        path = root / f"{symbol}_main_sub.pkl"
        if not path.is_file():
            raise FileNotFoundError(
                f"股指展期收益缓存缺失：{path}；先运行 python -m download.index.getIndexRollYield"
            )
        C.assert_research_only(path)
        frame = pd.read_pickle(path)
        if "annualized_yield" not in frame:
            raise KeyError(f"{path} 缺少 annualized_yield")
        dates = pd.DatetimeIndex(frame.index.get_level_values(-1))
        C.assert_no_holdout_dates(dates, what=f"{symbol} 股指展期收益")
        values = pd.Series(frame["annualized_yield"].to_numpy(), index=dates)
        if values.index.has_duplicates:
            raise ValueError(f"{symbol} 股指展期收益日期重复")
        carry[symbol] = values.sort_index()
    return pd.DataFrame(carry).reindex(index=index, columns=SYMBOLS)


def require_index_carry_coverage(carry: pd.DataFrame, scored_index: pd.DatetimeIndex) -> None:
    for year in sorted(set(scored_index.year)):
        days = scored_index[scored_index.year == year]
        if days.empty:
            raise ValueError(f"{year} 年没有可评分的股指交易日")
        missing = carry.reindex(days).isna().mean()
        bad = missing[missing > S.CARRY_MAX_MISSING]
        if not bad.empty:
            detail = "、".join(f"{symbol} 缺 {share:.0%}" for symbol, share in bad.items())
            raise ValueError(f"{year} 年股指展期收益覆盖不足（阈值 5%）：{detail}")


def _trim_to_first_target(
    returns: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Timestamp]:
    dates = [
        frame.index[frame.ne(0).any(axis=1)][0]
        for frame in targets.values()
        if frame.ne(0).any(axis=1).any()
    ]
    if not dates:
        raise ValueError("股指研究窗口没有任何非零目标，无法确定有效回测起点")
    first_signal_date = min(dates)
    trimmed = returns.loc[returns.index >= first_signal_date]
    if trimmed.empty:
        raise ValueError("首个非零目标日没有可用的下一开盘收益")
    return trimmed, first_signal_date


def _targets(index, bars, signed, carry):
    universe = {year: list(SYMBOLS) for year in range(SCORE_START.year, SCORE_END.year + 1)}
    pool = S.pool_mask(universe, index, SYMBOLS)
    # dfp_top3 要 250 日分钟历史，IF 约 2016-01、IH/IC 约 2016-04 才有；分数从五个因子都至少有一个品种有值的那天起算，
    # 之前缺失（不交易）。各因子的 z 仍用各自完整历史
    first = max(signed[name].dropna(how="all").index.min() for name in FACTORS)
    scored = index[index >= first]
    score = S.market_score(
        signed, universe, scored,
        min_symbols=len(SYMBOLS),
    ).reindex(index)
    lagged_carry = carry.reindex(index=index, columns=SYMBOLS).shift(1)
    long_decision = S.long_weights(score, lagged_carry)
    long_targets = long_decision.shift(1).fillna(0.0).where(pool, 0.0)

    symbol_trend = S.symbol_trend(bars, SYMBOLS).reindex(index)
    close_to_close = S.close_return(bars, SYMBOLS).reindex(index).where(pool)
    market_daily = close_to_close.mean(axis=1)
    bear = S.market_trend(market_daily).lt(0)
    slow_decision = S.decision_weights(score, lagged_carry, symbol_trend, bear)
    slow_targets = slow_decision.shift(1).fillna(0.0).where(pool, 0.0)
    return {
        LONG_LEG.ID: long_targets,
        S.BOOK.ID: slow_targets,
    }


def _summary_row(strategy_id, name, daily, market):
    net = daily["净收益"].fillna(0.0)
    beta = float(net.cov(market) / market.var())
    alpha = net - beta * market
    result = performance(net)
    return {
        "策略ID": strategy_id,
        "策略": name,
        "样本": f"{daily.index.min().date()}..{daily.index.max().date()}",
        "累计净收益": float(daily["权益"].iloc[-1] / INITIAL_CAPITAL - 1),
        "年化净收益": result["ann_return"],
        "年化波动": result["ann_vol"],
        "净Sharpe": sharpe_ratio(net),
        "最大回撤": max_drawdown(daily["权益"], INITIAL_CAPITAL),
        "相对等权股指beta": beta,
        "年化alpha": float(alpha.mean() * C.TRADING_DAYS_PER_YEAR),
        "alpha_NW_t": newey_west_t(alpha, 10),
        "年化交易成本": float(
            (daily["交易费用"] / daily["期初权益"]).mean() * C.TRADING_DAYS_PER_YEAR
        ),
        "熔断次数": int(
            (daily["停止新开仓"].astype(bool)
             & ~daily["停止新开仓"].astype(bool).shift(1, fill_value=False)).sum()
        ),
    }


def write_plots(book_returns: pd.DataFrame, market_returns: pd.Series, out: Path = OUT) -> None:
    """按商品账本的格式输出各策略净值/回撤图，并附股指基准对比图。"""
    labels = {
        LONG_LEG.ID: LONG_LEG.NAME,
        S.BOOK.ID: S.BOOK.NAME,
    }
    for strategy_id, label in labels.items():
        if strategy_id not in book_returns:
            raise KeyError(f"股指日收益缺少策略：{strategy_id}")
        plot_performance(
            book_returns[strategy_id].rename(label),
            out / "plots",
            title=f"整数手数：{label}，股指，1亿元、1倍敞口、20%保证金假设",
            filename=f"{strategy_id}.png",
        )
    comparison = book_returns.rename(columns=labels).copy()
    comparison["等权股指"] = market_returns.reindex(comparison.index)
    plot_performance(
        comparison,
        out / "plots",
        title="IF/IH/IC 股指迁移：策略与等权基准净值及回撤",
        filename="comparison.png",
    )


def _existing_output_matches(protocol: dict) -> bool:
    path = OUT / "protocol.json"
    if not path.is_file():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    prior = saved.get("protocol", {})
    return (
        prior.get("name") == protocol.get("name")
        and prior.get("instruments") == list(SYMBOLS)
        and prior.get("books") == protocol.get("books")
        and saved.get("calendar_start") == str(SCORE_START.date())
    )


def run(*, overwrite: bool = False) -> None:
    protocol = load_protocol()
    if (OUT / "summary.csv").exists():
        if not overwrite or not _existing_output_matches(protocol):
            raise FileExistsError(
                f"股指迁移结果已存在；仅可用 --overwrite 刷新同一研究协议的输出：{OUT}"
            )
    S.BOOK.frozen()

    daily_index, bars, contracts, signed, ticks = _load_index_inputs()
    all_returns = (
        bars["openw"].shift(-1) - bars["openw"]
    ) / bars["open"].where(bars["open"] > 0)
    scored_index = all_returns.index[
        (all_returns.index >= SCORE_START) & (all_returns.index <= SCORE_END)
    ]
    returns = all_returns.reindex(scored_index).dropna(how="any")
    if returns.empty:
        raise ValueError("股指研究窗口没有完整的下一开盘收益")
    C.assert_no_holdout_dates(returns.index, what="股指策略研究收益")

    carry = load_index_carry(daily_index)
    require_index_carry_coverage(carry, returns.index)
    targets = _targets(daily_index, bars, signed, carry)
    first_target_dates = {
        strategy_id: (
            str(target.index[target.ne(0).any(axis=1)][0].date())
            if target.ne(0).any(axis=1).any()
            else None
        )
        for strategy_id, target in targets.items()
    }
    returns, _ = _trim_to_first_target(returns, targets)
    C.assert_no_holdout_dates(returns.index, what="预热后股指策略收益")

    fee_table = load_index_fees("research")
    fee_table = fee_table[fee_table["trading_date"] <= SCORE_END]
    selected_open = bars["open"].reindex(index=returns.index, columns=SYMBOLS)
    open_fee, close_fee, roll_fee = fee_tables(fee_table, selected_open)
    selected_ticks = ticks[ticks["year"] < SCORE_END.year]
    slippage = slippage_tables(selected_ticks, selected_open)
    selected_contracts = contracts.reindex(index=returns.index, columns=SYMBOLS)
    rolls = selected_contracts.ne(selected_contracts.shift()).fillna(False)
    if not rolls.empty:
        rolls.iloc[0] = False

    index_results = {}
    summaries, annual, positions, events = [], [], [], []
    for book in (LONG_LEG, S.BOOK):
        target = targets[book.ID].reindex(index=returns.index, columns=SYMBOLS).fillna(0.0)
        daily, held, event_log = simulate_target_lots(
            target,
            selected_open,
            bars["close"].reindex(index=returns.index, columns=SYMBOLS),
            bars["openw"].reindex(index=returns.index, columns=SYMBOLS),
            (bars["openw"].shift(-1) - bars["openw"]).reindex(
                index=returns.index, columns=SYMBOLS
            ),
            bars["highw"].reindex(index=returns.index, columns=SYMBOLS),
            bars["loww"].reindex(index=returns.index, columns=SYMBOLS),
            rolls,
            selected_contracts,
            {year: list(SYMBOLS) for year in range(SCORE_START.year, SCORE_END.year + 1)},
            {symbol: INDEX_MULTIPLIER[symbol] for symbol in SYMBOLS},
            open_fee,
            close_fee,
            roll_fee,
            slippage,
            initial_capital=INITIAL_CAPITAL,
            margin_rate=MARGIN_RATE,
            exposure_limit=book.GROSS,
            drawdown_limit=S.DRAWDOWN_LIMIT,
            halt_days=S.HALT_DAYS,
        )
        market = returns.mean(axis=1).reindex(daily.index)
        summaries.append(_summary_row(book.ID, book.NAME, daily, market))
        index_results[book.ID] = daily["净收益"].rename(book.ID)
        for year, section in daily.groupby(daily.index.year):
            annual.append({
                "策略ID": book.ID,
                "策略": book.NAME,
                "年份": int(year),
                "净收益": float(section["权益"].iloc[-1] / section["期初权益"].iloc[0] - 1),
                "Sharpe": sharpe_ratio(section["净收益"].fillna(0.0)),
                "最大回撤": max_drawdown(
                    section["权益"], float(section["期初权益"].iloc[0])
                ),
            })
        held.insert(0, "策略ID", book.ID)
        event_log.insert(0, "策略ID", book.ID)
        positions.append(held)
        events.append(event_log)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(index_results).assign(等权股指收益=returns.mean(axis=1)).to_csv(
        OUT / "daily.csv", encoding="utf-8-sig"
    )
    pd.concat(positions, ignore_index=True).to_csv(
        OUT / "positions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(events, ignore_index=True).to_csv(
        OUT / "events.csv", index=False, encoding="utf-8-sig"
    )
    write_plots(pd.DataFrame(index_results), returns.mean(axis=1))
    (OUT / "protocol.json").write_text(
        json.dumps({
            "protocol": protocol,
            "calendar_start": str(SCORE_START.date()),
            "first_return_date": str(returns.index.min().date()),
            "last_return_date": str(returns.index.max().date()),
            "return_observations": len(returns),
            "first_nonzero_target_date": first_target_dates,
            "instruments": list(SYMBOLS),
            "carry_coverage_by_year": {
                str(year): {
                    symbol: float(carry.reindex(returns.index).loc[
                        returns.index.year == year, symbol
                    ].notna().mean())
                    for symbol in SYMBOLS
                }
            for year in sorted(set(returns.index.year))
            },
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(pd.DataFrame(annual).to_string(index=False), flush=True)
    print(f"股指迁移检验结果：{OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF/IH/IC 当前策略迁移回测")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="仅当结果目录属于同一研究协议时，刷新已有输出",
    )
    run(overwrite=parser.parse_args().overwrite)
