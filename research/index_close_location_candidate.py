"""IF/IH/IC 收盘位置长仓候选；只读取研究期数据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
from research import index_market_clock as M
from strategies import index_catalog
from strategies import index_close_location as S
from tfcta import config as C
from tfcta.costs import (
    INDEX_MULTIPLIER,
    fee_tables,
    load_index_fees,
    slippage_tables,
)

OUT = C.RUNS_DIR / "index" / "index_close_location_candidate"
SYMBOLS = S.SYMBOLS
SCORE_START = pd.Timestamp("2016-01-13")
SCORE_END = pd.Timestamp("2021-12-31")
CANDIDATE_ID = S.BOOK.ID
CANDIDATE_NAME = S.BOOK.NAME
BENCHMARK_ID = "index_equal_weight_risk_matched"
BENCHMARK_NAME = "等权长仓"


def load_protocol() -> dict:
    protocol = S.BOOK.declared()
    parameters = protocol.get("parameters", {})
    sample = protocol.get("sample", "")
    decision_rules = protocol.get("decision_rules", [])
    expected_close_location = (
        "For each index, (closew - loww) / (highw - loww); take the equal-weight mean "
        "across all three. closew/highw/loww are additive adjusted prices, so only "
        "their differences are used."
    )
    expected_target = (
        "At close t, if equal-weight market close location < 0.5, target +1 in each "
        "index; otherwise target 0. Weights are split equally by the shared ledger's "
        "pool-width accounting, producing 1x total gross exposure or cash."
    )
    expected_execution = (
        "Use close-t information to set the target for the next trading day's open; "
        "hold from that open to the following open."
    )
    expected_risk = (
        f"Use the shared {S.DRAWDOWN_LIMIT:.0%} peak drawdown breaker and halt new "
        f"positions for {S.HALT_DAYS} trading days."
    )
    expected_benchmark = (
        "Equal-weight 1x long IF/IH/IC on every scored day, run through the same "
        "integer-lot ledger, historical fees, 1 tick per-side slippage, and drawdown "
        "breaker. Also report the equal-weight gross index return for context."
    )
    expected_costs = (
        "Historical index exchange fees and one tick per side, using the prior year's "
        "estimated tick; same assumptions for candidate and risk-matched benchmark."
    )
    if (
        protocol.get("status") != S.STATUS
        or "Do not read 2022" not in sample
        or "2023-2025 locked partitions" not in sample
        or "or 2026" not in sample
        or parameters.get("instruments") != list(SYMBOLS)
        or parameters.get("score_start") != str(SCORE_START.date())
        or parameters.get("score_end") != str(SCORE_END.date())
        or parameters.get("close_location") != expected_close_location
        or parameters.get("target_threshold") != S.TARGET_THRESHOLD
        or parameters.get("target_rule") != expected_target
        or parameters.get("execution") != expected_execution
        or parameters.get("risk") != expected_risk
        or parameters.get("gross_exposure_limit") != S.GROSS
        or parameters.get("benchmark") != expected_benchmark
        or parameters.get("costs") != expected_costs
        or parameters.get("hac_lags") != 10
        or len(decision_rules) < 2
        or "paired daily excess-return Newey-West t" not in decision_rules[0]
        or ">= 2" not in decision_rules[1]
    ):
        raise ValueError("股指收盘位置候选登记与样本边界或规则不一致")
    return protocol


def build_targets(
    bars: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """从收盘位置生成次日目标，并返回当日等权市场 CLV。"""
    for field in ("closew", "highw", "loww"):
        if field not in bars:
            raise KeyError(f"缺少收盘位置字段：{field}")
        if not bars[field].reindex(index=index, columns=SYMBOLS).notna().all().all():
            raise ValueError(f"{field} 的 IF/IH/IC 行情未完全对齐")

    close = bars["closew"].reindex(index=index, columns=SYMBOLS)
    high = bars["highw"].reindex(index=index, columns=SYMBOLS)
    low = bars["loww"].reindex(index=index, columns=SYMBOLS)
    day_range = high - low
    if day_range.le(0).any().any():
        raise ValueError("股指日内最高价必须高于最低价")

    clv = (close - low) / day_range
    if not np.isfinite(clv.to_numpy()).all() or clv.lt(0).any().any() or clv.gt(1).any().any():
        raise ValueError("股指收盘位置超出 [0, 1] 或包含无效值")
    market_clv = clv.mean(axis=1, skipna=False).rename("market_clv")
    candidate = S.BOOK.targets(market_clv).reindex(index=index, columns=SYMBOLS)
    benchmark = pd.DataFrame(1.0, index=index, columns=SYMBOLS)
    return {CANDIDATE_ID: candidate, BENCHMARK_ID: benchmark}, market_clv


def simulate_book(target: pd.DataFrame, index: pd.DatetimeIndex, bars: dict, contracts: pd.DataFrame,
                  ticks: pd.DataFrame, symbols: tuple[str, ...] = SYMBOLS):
    """同一整数手账本：研究期股指手续费、上一年估计 tick 每边 1 跳、10% 熔断停 10 天。"""
    symbols = tuple(symbols)
    if not symbols or any(symbol not in SYMBOLS for symbol in symbols):
        raise ValueError(f"无效的股指回测品种：{symbols}")
    selected_open = bars["open"].reindex(index=index, columns=symbols)
    fee_table = load_index_fees("research")
    fee_table = fee_table[fee_table["trading_date"] <= SCORE_END]
    open_fee, close_fee, roll_fee = fee_tables(fee_table, selected_open)
    selected_ticks = ticks[ticks["symbol"].isin(symbols)]
    slippage = slippage_tables(selected_ticks[selected_ticks["year"] < SCORE_END.year], selected_open)
    selected_contracts = contracts.reindex(index=index, columns=symbols)
    rolls = selected_contracts.ne(selected_contracts.shift()).fillna(False)
    if not rolls.empty:
        rolls.iloc[0] = False
    universe = {year: list(symbols) for year in range(index.min().year, index.max().year + 1)}
    return simulate_target_lots(
        target.reindex(index=index, columns=symbols).fillna(0.0),
        selected_open,
        bars["close"].reindex(index=index, columns=symbols),
        bars["openw"].reindex(index=index, columns=symbols),
        (bars["openw"].shift(-1) - bars["openw"]).reindex(index=index, columns=symbols),
        bars["highw"].reindex(index=index, columns=symbols),
        bars["loww"].reindex(index=index, columns=symbols),
        rolls,
        selected_contracts,
        universe,
        {symbol: INDEX_MULTIPLIER[symbol] for symbol in symbols},
        open_fee,
        close_fee,
        roll_fee,
        slippage,
        initial_capital=INITIAL_CAPITAL,
        margin_rate=MARGIN_RATE,
        exposure_limit=1.0,
        drawdown_limit=S.DRAWDOWN_LIMIT,
        halt_days=S.HALT_DAYS,
    )


ROBUST_OUT = OUT / "robustness"
EXCHANGE_TICK = 0.2      # 中金所 IF/IH/IC 最小变动价位，上市以来一直是 0.2 点


def run_robustness() -> None:
    """登记文件 index_liquidity_mechanism.json 的 candidate_robustness：不改门槛结论，不调参数。"""
    load_protocol()
    daily_index, bars, contracts, _, ticks = M._load_index_inputs()
    all_returns = (bars["openw"].shift(-1) - bars["openw"]) / bars["open"].where(bars["open"] > 0)
    all_returns = all_returns.loc[all_returns.index <= SCORE_END].dropna(how="any")
    C.assert_no_holdout_dates(all_returns.index, what="股指候选稳健性收益")
    targets, _ = build_targets(bars, daily_index)
    first_year = ticks[ticks["year"] == 2015].set_index("symbol")["tick"].reindex(list(SYMBOLS))
    if not np.allclose(first_year.to_numpy(dtype=float), EXCHANGE_TICK):
        raise ValueError(f"2015 年估计的 tick 不等于 {EXCHANGE_TICK}：{first_year.to_dict()}")
    ticks_2015 = pd.concat([ticks[ticks["year"] != 2014],
                            ticks[ticks["year"] == 2015].assign(year=2014)], ignore_index=True)
    windows = {
        "登记口径 2016-01-13": (SCORE_START, ticks),
        "R1 2016-01-04": (pd.Timestamp("2016-01-04"), ticks),
        "R2 2015-04-16": (pd.Timestamp("2015-04-16"), ticks_2015),
    }
    rows, annual, capacity = [], [], []
    for label, (start, tick_table) in windows.items():
        index = all_returns.index[all_returns.index >= start]
        cand = targets[CANDIDATE_ID].reindex(index)
        average = float(cand.mean(axis=1).mean())
        books = {
            "候选": cand,
            "等权长仓": targets[BENCHMARK_ID].reindex(index),
            f"R3 常数敞口 {average:.2f}": pd.DataFrame(average, index=index, columns=list(SYMBOLS)),
        }
        nets = {}
        for name, target in books.items():
            daily, _, events = simulate_book(target, index, bars, contracts, tick_table)
            net = daily["净收益"].fillna(0.0)
            nets[name] = net
            row = M._summary_row(name, name, daily, all_returns.reindex(index).mean(axis=1))
            rows.append({"窗口": label, **row})
            for year, part in daily.groupby(daily.index.year):
                annual.append({"窗口": label, "策略": name, "年份": int(year),
                               "净收益": float(part["权益"].iloc[-1] / part["期初权益"].iloc[0] - 1)})
            if name == "候选" and label == "R2 2015-04-16":
                opened = events[events["事件"].isin(["开仓", "加仓"])].copy()
                opened["年份"] = pd.to_datetime(opened["交易日"]).dt.year
                for (symbol, year), part in opened.groupby(["品种", "年份"]):
                    daily_lots = part.groupby("交易日")["手数"].sum()
                    capacity.append({"品种": symbol, "年份": int(year), "开仓日数": len(daily_lots),
                                     "单日开仓最大手数": int(daily_lots.max()),
                                     "单日开仓95分位手数": float(daily_lots.quantile(0.95))})
        for other in [k for k in nets if k != "候选"]:
            rows.append({"窗口": label, "策略ID": f"候选 − {other}", "策略": "配对超额",
                         "年化净收益": float((nets["候选"] - nets[other]).mean() * C.TRADING_DAYS_PER_YEAR),
                         "alpha_NW_t": newey_west_t(nets["候选"] - nets[other], 10)})
    ROBUST_OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(ROBUST_OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(ROBUST_OUT / "annual.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(capacity).to_csv(ROBUST_OUT / "capacity.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    print(table[["窗口", "策略ID", "年化净收益", "净Sharpe", "最大回撤", "相对等权股指beta", "alpha_NW_t", "熔断次数"]]
          .round(3).to_string(index=False))
    print(pd.DataFrame(annual).pivot_table(index=["窗口", "策略"], columns="年份", values="净收益").round(3).to_string())
    print(pd.DataFrame(capacity).to_string(index=False))
    print(f"稳健性结果：{ROBUST_OUT}")


SHORT_LEG_PATH = C.CONFIG_DIR / "research" / "index_short_leg_2015_2021.json"
SHORT_LEG_OUT = C.RUNS_DIR / "index" / "index_short_leg_2015_2021"


def short_leg_targets(long_target: pd.DataFrame, bars: dict, carry: pd.DataFrame | None) -> pd.DataFrame:
    """熊市日（池子 20/60/120 日趋势 < 0）品种自身趋势 < 0（且给了 carry 时滞后展期 < 0）→ −1；冲突时以空头为准。
    收盘 t 已知的决策后移到下一开盘，与多头目标同口径。"""
    from strategies import commodity_original as T

    index = long_target.index
    symbols = list(SYMBOLS)
    trend = T.symbol_trend(bars, symbols).reindex(index)
    bear = T.market_trend(T.close_return(bars, symbols).reindex(index).mean(axis=1)).lt(0)
    short = trend.lt(0)
    if carry is not None:
        short &= carry.reindex(index=index, columns=symbols).shift(1).lt(0)
    short = short.mul(bear.astype(bool), axis=0).astype(bool).shift(1, fill_value=False)
    return long_target.where(~short, -1.0)


def run_short_leg() -> None:
    """登记在 index_short_leg_2015_2021.json：候选、候选加展期空头腿、候选加趋势空头腿，两个起点。"""
    load_protocol()
    spec = json.loads(SHORT_LEG_PATH.read_text(encoding="utf-8"))
    if "不读 2022" not in spec.get("sample", "") or "不读 2026" not in spec.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    daily_index, bars, contracts, _, ticks = M._load_index_inputs()
    carry = M.load_index_carry(daily_index)
    all_returns = (bars["openw"].shift(-1) - bars["openw"]) / bars["open"].where(bars["open"] > 0)
    all_returns = all_returns.loc[all_returns.index <= SCORE_END].dropna(how="any")
    C.assert_no_holdout_dates(all_returns.index, what="股指空头腿收益")
    targets, _ = build_targets(bars, daily_index)
    long_target = targets[CANDIDATE_ID]
    books = {
        "long": long_target,
        "long_plus_carry_short": short_leg_targets(long_target, bars, carry),
        "long_plus_trend_short": short_leg_targets(long_target, bars, None),
    }
    first_year = ticks[ticks["year"] == 2015].set_index("symbol")["tick"].reindex(list(SYMBOLS))
    if not np.allclose(first_year.to_numpy(dtype=float), EXCHANGE_TICK):
        raise ValueError(f"2015 年估计的 tick 不等于 {EXCHANGE_TICK}：{first_year.to_dict()}")
    ticks_2015 = pd.concat([ticks[ticks["year"] != 2014],
                            ticks[ticks["year"] == 2015].assign(year=2014)], ignore_index=True)
    rows, annual = [], []
    for label, start in (("2015-04-16 起", pd.Timestamp("2015-04-16")), ("2016-01-13 起", SCORE_START)):
        index = all_returns.index[all_returns.index >= start]
        market = all_returns.reindex(index).mean(axis=1)
        nets = {}
        for name, target in books.items():
            daily, held, _ = simulate_book(target.reindex(index), index, bars, contracts, ticks_2015)
            net = daily["净收益"].fillna(0.0)
            nets[name] = net
            summary = M._summary_row(name, name, daily, market)
            short_days = int((held.groupby("交易日")["手数"].min() < 0).sum()) if not held.empty else 0
            rows.append({"起点": label, **summary, "空头持仓日": short_days})
            moves = (bars["openw"].shift(-1) - bars["openw"]).reindex(index=index, columns=list(SYMBOLS)).stack()
            moves.index.names = ["交易日", "品种"]
            pos = held.set_index(["交易日", "品种"])
            pnl = pos["手数"] * pos["合约乘数"] * moves.reindex(pos.index).to_numpy()
            short_pnl = pnl[pos["手数"] < 0].groupby(level="交易日").sum()
            equity = daily["期初权益"]
            for year, part in daily.groupby(daily.index.year):
                sp = short_pnl[short_pnl.index.year == year]
                annual.append({"起点": label, "账本": name, "年份": int(year),
                               "净收益": float(part["权益"].iloc[-1] / part["期初权益"].iloc[0] - 1),
                               "空头腿毛盈亏占期初权益": float((sp / equity.reindex(sp.index)).sum())})
        for name in ("long_plus_carry_short", "long_plus_trend_short"):
            diff = nets[name] - nets["long"]
            rows.append({"起点": label, "策略ID": f"{name} − long", "策略": "配对超额",
                         "年化净收益": float(diff.mean() * C.TRADING_DAYS_PER_YEAR), "alpha_NW_t": newey_west_t(diff, 10)})
    SHORT_LEG_OUT.mkdir(parents=True, exist_ok=True)
    table, annual = pd.DataFrame(rows), pd.DataFrame(annual)
    table.to_csv(SHORT_LEG_OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(SHORT_LEG_OUT / "annual.csv", index=False, encoding="utf-8-sig")
    (SHORT_LEG_OUT / "protocol.json").write_text(json.dumps({"protocol": spec}, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    pd.set_option("display.width", 240)
    print(table[["起点", "策略ID", "年化净收益", "净Sharpe", "最大回撤", "相对等权股指beta", "alpha_NW_t",
                 "熔断次数", "空头持仓日"]].round(3).to_string(index=False))
    print(annual.pivot_table(index=["起点", "账本"], columns="年份", values="净收益").round(3).to_string())
    print(annual.pivot_table(index=["起点", "账本"], columns="年份", values="空头腿毛盈亏占期初权益").round(3).to_string())
    print(f"结果：{SHORT_LEG_OUT}")


def _existing_output_matches(protocol: dict) -> bool:
    path = OUT / "protocol.json"
    if not path.is_file():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    last_scored_date = saved.get("last_scored_date", "")
    saved_protocol = saved.get("protocol", {})
    return (
        (
            saved_protocol.get("id") == protocol.get("id")
            or saved_protocol.get("name") == "IF/IH/IC close-location reversal candidate"
        )
        and saved.get("calendar_start") == "2015-04-16"
        and saved.get("first_scored_date") == str(SCORE_START.date())
        and str(SCORE_START.date()) <= last_scored_date <= str(SCORE_END.date())
    )


def run(*, overwrite: bool = False) -> None:
    protocol = load_protocol()
    if index_catalog.book_named(CANDIDATE_ID) is not S.BOOK:
        raise ValueError("股指候选没有登记到股指策略目录")
    if (OUT / "summary.csv").exists() and (
        not overwrite or not _existing_output_matches(protocol)
    ):
        raise FileExistsError(
            f"股指候选结果已存在；仅可用 --overwrite 刷新同一研究协议的输出：{OUT}"
        )

    daily_index, bars, contracts, _, ticks = M._load_index_inputs()
    all_returns = (bars["openw"].shift(-1) - bars["openw"]) / bars["open"].where(
        bars["open"] > 0
    )
    returns = all_returns.loc[
        (all_returns.index >= SCORE_START) & (all_returns.index <= SCORE_END)
    ].dropna(how="any")
    if returns.empty:
        raise ValueError("股指收盘位置候选没有完整的下一开盘收益")
    C.assert_no_holdout_dates(returns.index, what="股指收盘位置候选收益")

    targets, market_clv = build_targets(bars, daily_index)
    first_targets = targets[CANDIDATE_ID].reindex(returns.index).ne(0).any(axis=1)
    if not first_targets.any():
        raise ValueError("股指候选在研究窗口没有非零目标")
    first_target_date = first_targets.index[first_targets][0]
    returns = returns.loc[returns.index >= first_target_date]
    C.assert_no_holdout_dates(returns.index, what="预热后股指候选收益")

    ledger_results = {}
    summaries = []
    annual = []
    positions = []
    events = []
    labels = {
        CANDIDATE_ID: CANDIDATE_NAME,
        BENCHMARK_ID: BENCHMARK_NAME,
    }
    for book_id, name in labels.items():
        daily, held, event_log = simulate_book(targets[book_id], returns.index, bars, contracts, ticks)
        summaries.append(M._summary_row(book_id, name, daily, returns.mean(axis=1)))
        ledger_results[book_id] = daily
        for year, section in daily.groupby(daily.index.year):
            annual.append({
                "策略ID": book_id,
                "策略": name,
                "年份": int(year),
                "净收益": float(section["权益"].iloc[-1] / section["期初权益"].iloc[0] - 1),
                "Sharpe": sharpe_ratio(section["净收益"].fillna(0.0)),
                "最大回撤": max_drawdown(
                    section["权益"], float(section["期初权益"].iloc[0])
                ),
            })
        held.insert(0, "策略ID", book_id)
        event_log.insert(0, "策略ID", book_id)
        positions.append(held)
        events.append(event_log)

    candidate_net = ledger_results[CANDIDATE_ID]["净收益"]
    benchmark_net = ledger_results[BENCHMARK_ID]["净收益"]
    excess = candidate_net - benchmark_net
    candidate_return = float(
        ledger_results[CANDIDATE_ID]["权益"].iloc[-1] / INITIAL_CAPITAL - 1
    )
    benchmark_return = float(
        ledger_results[BENCHMARK_ID]["权益"].iloc[-1] / INITIAL_CAPITAL - 1
    )
    candidate_dd = max_drawdown(
        ledger_results[CANDIDATE_ID]["权益"], INITIAL_CAPITAL
    )
    benchmark_dd = max_drawdown(
        ledger_results[BENCHMARK_ID]["权益"], INITIAL_CAPITAL
    )
    paired_t = newey_west_t(excess, 10)
    gate = {
        "candidate_net_return": candidate_return,
        "risk_matched_equal_weight_net_return": benchmark_return,
        "paired_daily_excess_NW_t": paired_t,
        "candidate_max_drawdown": candidate_dd,
        "risk_matched_equal_weight_max_drawdown": benchmark_dd,
        "pass": bool(
            candidate_return > benchmark_return
            and paired_t >= 2.0
            and candidate_dd > benchmark_dd
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    gross_market = returns.mean(axis=1)
    gross_metrics = performance(gross_market)
    pd.DataFrame([{
        "基准": "IF/IH/IC 等权毛收益",
        "口径": "未扣手续费和滑点；未施加回撤熔断",
        "样本": f"{returns.index.min().date()}..{returns.index.max().date()}",
        "累计收益": float((1.0 + gross_market).prod() - 1.0),
        "年化收益": gross_metrics["ann_return"],
        "年化波动": gross_metrics["ann_vol"],
        "Sharpe": sharpe_ratio(gross_market),
        "最大回撤": gross_metrics["max_drawdown"],
        "交易日数": gross_metrics["n_days"],
    }]).to_csv(OUT / "equal_weight_gross.csv", index=False, encoding="utf-8-sig")
    daily_output = pd.DataFrame({
        CANDIDATE_ID: candidate_net,
        BENCHMARK_ID: benchmark_net,
        "等权股指毛收益": gross_market,
        "市场收盘位置": market_clv.reindex(returns.index),
        "候选目标敞口": targets[CANDIDATE_ID].reindex(returns.index).mean(axis=1),
    })
    daily_output.to_csv(OUT / "daily.csv", encoding="utf-8-sig")
    pd.concat(positions, ignore_index=True).to_csv(
        OUT / "positions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(events, ignore_index=True).to_csv(
        OUT / "events.csv", index=False, encoding="utf-8-sig"
    )

    plot_returns = pd.DataFrame({
        labels[book_id]: daily["净收益"]
        for book_id, daily in ledger_results.items()
    })
    plot_returns["等权股指（毛）"] = returns.mean(axis=1)
    plot_performance(
        plot_returns,
        OUT / "plots",
        title="IF/IH/IC 收盘位置候选与等权基准：净值及回撤",
        filename="comparison.png",
    )
    for book_id, daily in ledger_results.items():
        plot_performance(
            daily["净收益"].rename(labels[book_id]),
            OUT / "plots",
            title=f"整数手数：{labels[book_id]}，1亿元、1倍敞口、研究期成本",
            filename=f"{book_id}.png",
        )

    (OUT / "protocol.json").write_text(
        json.dumps({
            "protocol": protocol,
            "calendar_start": "2015-04-16",
            "first_scored_date": str(SCORE_START.date()),
            "first_nonzero_target_date": str(first_target_date.date()),
            "last_scored_date": str(returns.index.max().date()),
            "return_observations": len(returns),
            "instruments": list(SYMBOLS),
            "gate": gate,
            "scope": "研究期探索性结果；不读取 2022 年及以后。",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    print(f"股指收盘位置候选结果：{OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF/IH/IC 收盘位置长仓候选研究回测")
    parser.add_argument("--overwrite", action="store_true", help="重写同一预登记方案的研究输出")
    parser.add_argument("--robustness", action="store_true", help="登记过的稳健性检验 R1–R4，写到 robustness/")
    parser.add_argument("--short-leg", action="store_true", help="候选加空头腿，2015–2021（登记在 index_short_leg_2015_2021.json）")
    cli = parser.parse_args()
    if cli.short_leg:
        run_short_leg()
    elif cli.robustness:
        run_robustness()
    else:
        run(overwrite=cli.overwrite)
