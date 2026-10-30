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
CANDIDATE_ID = "index_close_location_reversal"
CANDIDATE_NAME = "股指收盘位置候选策略"
BENCHMARK_ID = "index_equal_weight_risk_matched"
BENCHMARK_NAME = "等权长仓"


def _candidate_book():
    """只做多策略已于 2026-10-29 删除。候选主入口用到时才找它。"""
    book = getattr(S, "BOOK", None)
    if book is None:
        raise RuntimeError("只做多的股指收盘位置策略已在 2026-10-29 删除，这条候选主入口不能再跑。")
    return book


def load_protocol() -> dict:
    protocol = _candidate_book().declared()
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
        or protocol.get("chosen") != parameters
        or not protocol.get("frozen_at")
        or protocol.get("chosen") != parameters
        or not protocol.get("frozen_at")
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
    candidate = _candidate_book().targets(market_clv).reindex(index=index, columns=SYMBOLS)
    benchmark = pd.DataFrame(1.0, index=index, columns=SYMBOLS)
    return {_candidate_book().ID: candidate, BENCHMARK_ID: benchmark}, market_clv


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
    if index_catalog.book_named(CANDIDATE_ID) is not _candidate_book():
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


VARIANTS_PATH = C.CONFIG_DIR / "research" / "index_close_location_variants.json"
VARIANTS_OUT = C.RUNS_DIR / "index" / "index_close_location_variants"
VARIANT_KEYS = ("frozen", "clv_0.3", "clv_0.4", "clv_0.5", "clv_0.6", "clv_0.7", "graded", "range_gate")


def variant_decisions(market_clv: pd.Series, market_range: pd.Series, frozen: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """登记的变体在收盘 t 的决策（frozen 已经后移，其余未后移）。"""
    ones = lambda flag: pd.DataFrame({s: flag.astype(float) for s in SYMBOLS}, index=market_clv.index)  # noqa: E731
    out = {f"clv_{t}": ones(market_clv.lt(t)) for t in (0.3, 0.4, 0.5, 0.6, 0.7)}
    weight = (2 * (0.5 - market_clv)).clip(lower=0.0, upper=1.0)
    out["graded"] = pd.DataFrame({s: weight for s in SYMBOLS}, index=market_clv.index)
    median = market_range.shift(1).rolling(250, min_periods=120).median()
    out["range_gate"] = ones(market_clv.lt(0.5) & (market_range > median) & median.notna())
    out = {k: v.shift(1).fillna(0.0) for k, v in out.items()}
    return {"frozen": frozen, **out}


def run_variants() -> None:
    """登记在 index_close_location_variants.json：8 个变体，两个起点，全部报告。"""
    from backtest.engine import _load_index_research_inputs, _simulate_index_targets, deflated_sharpe
    from strategies.index_close_location import STRICT_SWAP_BOOK, market_close_location

    spec = json.loads(VARIANTS_PATH.read_text(encoding="utf-8"))
    if (list(spec.get("variants", {})) != list(VARIANT_KEYS) or "不读 2022" not in spec.get("sample", "")
            or "不读 2026" not in spec.get("sample", "")):
        raise ValueError("收盘位置变体登记与实现不一致")
    index, bars, contracts, carry, ticks = _load_index_research_inputs()
    returns = ((bars["openw"].shift(-1) - bars["openw"]) / bars["open"].where(bars["open"] > 0)).dropna(how="any")
    C.assert_no_holdout_dates(returns.index, what="股指收盘位置变体收益")
    market_clv = market_close_location(bars, index)
    market_range = ((bars["highw"] - bars["loww"]) / bars["close"]).mean(axis=1)
    frozen = STRICT_SWAP_BOOK.targets(market_clv, bars, carry)
    decisions = variant_decisions(market_clv, market_range, frozen)
    first = frozen.reindex(returns.index).ne(0).any(axis=1)
    rows, annual, nets_all = [], [], {}
    for label, start in (("2015-04-20 起", first.index[first][0]), ("2016-01-04 起", pd.Timestamp("2016-01-04"))):
        scored = returns.index[returns.index >= start]
        simulate = lambda target, limit: _simulate_index_targets(            # noqa: E731
            target.reindex(index=scored, columns=list(SYMBOLS)).fillna(0.0), scored, bars, contracts, ticks,
            SCORE_END.year, tuple(SYMBOLS), exposure_limit=1.0, drawdown_limit=limit, halt_days=S.HALT_DAYS)
        bench_daily, _, _ = simulate(pd.DataFrame(1.0, index=scored, columns=list(SYMBOLS)), 0.99)
        bench = bench_daily["净收益"].fillna(0.0)
        nets = {}
        for key in VARIANT_KEYS:
            daily, _, _ = simulate(decisions[key], S.DRAWDOWN_LIMIT)
            net = daily["净收益"].fillna(0.0)
            nets[key] = net
            rows.append({"起点": label, "变体": key, "年化净收益": performance(net)["ann_return"],
                         "净Sharpe": sharpe_ratio(net), "最大回撤": max_drawdown(daily["权益"], INITIAL_CAPITAL),
                         "相对买入持有beta": float(net.cov(bench) / bench.var()),
                         "平均敞口": float(daily["敞口倍数"].mean()),
                         "年化成本": float((daily["交易费用"] / daily["期初权益"]).mean() * C.TRADING_DAYS_PER_YEAR)})
            for year, part in net.groupby(net.index.year):
                annual.append({"起点": label, "变体": key, "年份": int(year), "净收益": float((1 + part).prod() - 1)})
        rows.append({"起点": label, "变体": "等权买入持有", "年化净收益": performance(bench)["ann_return"],
                     "净Sharpe": sharpe_ratio(bench), "最大回撤": max_drawdown(bench_daily["权益"], INITIAL_CAPITAL),
                     "相对买入持有beta": 1.0, "平均敞口": float(bench_daily["敞口倍数"].mean())})
        trials = [sharpe_ratio(n) for n in nets.values()]
        for row in rows:
            if row["起点"] == label and row["变体"] in nets:
                diff = nets[row["变体"]] - nets["clv_0.5"]
                row["相对clv_0.5配对t"] = newey_west_t(diff, 10) if diff.abs().sum() > 0 else np.nan
                row["DSR"] = deflated_sharpe(nets[row["变体"]], trials, n_trials=len(trials))["dsr"]
        nets_all[label] = nets
    VARIANTS_OUT.mkdir(parents=True, exist_ok=True)
    table, annual = pd.DataFrame(rows), pd.DataFrame(annual)
    table.to_csv(VARIANTS_OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(VARIANTS_OUT / "annual.csv", index=False, encoding="utf-8-sig")
    (VARIANTS_OUT / "protocol.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 240)
    print(table.round(3).to_string(index=False))
    print(annual.pivot_table(index=["起点", "变体"], columns="年份", values="净收益").round(3).to_string())
    print(f"结果：{VARIANTS_OUT}")


TILT_PATH = C.CONFIG_DIR / "research" / "index_product_tilt.json"
TILT_OUT = C.RUNS_DIR / "index" / "index_product_tilt"
TILT_KEYS = ("equal", "vol_parity", "rolling_strength", "static_tilt", "drop_ih", "ic_only")
STATIC_TILTS = {"static_tilt": (1.0, 0.5, 1.5), "drop_ih": (1.5, 0.0, 1.5), "ic_only": (0.0, 0.0, 3.0)}


def _normalize_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """每行按和为 3 归一；只在每月第一个交易日取值、四舍五入到 0.1，其余日子沿用。缺失或全 ≤ 0 时用等权。"""
    total = raw.clip(lower=0.0).sum(axis=1)
    weights = raw.clip(lower=0.0).div(total.where(total > 0), axis=0) * len(SYMBOLS)
    weights = weights.where(total > 0).fillna(1.0)
    months = pd.Series(raw.index.to_period("M"), index=raw.index)
    first = months.ne(months.shift(1))
    return weights.where(first, np.nan).ffill().round(1).fillna(1.0)


def tilt_multipliers(index: pd.DatetimeIndex, returns: pd.DataFrame, market_clv: pd.Series) -> dict[str, pd.DataFrame]:
    """收盘 t 决定的品种乘数（未后移）。returns[d] 是 d 开盘到 d+1 开盘，在 d+1 开盘实现。"""
    returns = returns.reindex(index=index, columns=list(SYMBOLS))
    out = {"equal": pd.DataFrame(1.0, index=index, columns=list(SYMBOLS))}
    vol = returns.shift(1).rolling(60, min_periods=40).std()          # 收盘 t 已实现到 t−1 开盘到 t 开盘
    out["vol_parity"] = _normalize_monthly(1.0 / vol)
    fwd = returns.shift(-1)                                            # 收盘 d 的信号对应 d+1 开盘到 d+2 开盘
    x = market_clv.reindex(index)
    slopes = {}
    for symbol in SYMBOLS:
        y = fwd[symbol]
        cov = (x * y).rolling(500, min_periods=250).mean() - x.rolling(500, min_periods=250).mean() * y.rolling(500, min_periods=250).mean()
        var = x.rolling(500, min_periods=250).var(ddof=0)
        slopes[symbol] = (cov / var).shift(2)                         # 配对的结果在 d+2 开盘才实现
    out["rolling_strength"] = _normalize_monthly(-pd.DataFrame(slopes, index=index))
    for key, (w_if, w_ih, w_ic) in STATIC_TILTS.items():
        out[key] = pd.DataFrame({"IF": w_if, "IH": w_ih, "IC": w_ic}, index=index)[list(SYMBOLS)]
    return out


def tilt_diagnostics(returns: pd.DataFrame, market_clv: pd.Series) -> pd.DataFrame:
    from research.time_factor_screen import ols_hac

    fwd = returns.shift(-1)
    scaled = fwd / returns.shift(1).rolling(20, min_periods=20).std()
    beta = (returns["IC"].rolling(250, min_periods=120).cov(returns["IH"])
            / returns["IH"].rolling(250, min_periods=120).var()).shift(1)
    cases = {**{f"D1 {s} 原始": fwd[s] for s in SYMBOLS}, **{f"D1 {s} 波动标准化": scaled[s] for s in SYMBOLS},
             "D2 IC − IH 原始": fwd["IC"] - fwd["IH"], "D2 IC − IH 波动标准化": scaled["IC"] - scaled["IH"],
             "D3 IC − β·IH": fwd["IC"] - beta * fwd["IH"]}
    rows = []
    for label, start in (("2015-04-16 起", pd.Timestamp("2015-04-16")), ("2016-01-04 起", pd.Timestamp("2016-01-04"))):
        for name, y in cases.items():
            frame = pd.concat([y, market_clv], axis=1).dropna()
            frame = frame[(frame.index >= start) & (frame.index <= SCORE_END)]
            b, t = ols_hac(frame.iloc[:, 0].to_numpy(), [frame.iloc[:, 1].to_numpy()], 5)
            unit = 1.0 if "标准化" in name else 1e4
            rows.append({"起点": label, "检验": name, "斜率": b * unit, "单位": "标准差" if unit == 1 else "bp",
                         "NW_t": t, "天数": len(frame)})
    return pd.DataFrame(rows)


def run_tilt() -> None:
    """登记在 index_product_tilt.json：6 种品种权重，两个起点，全部报告。"""
    from backtest.engine import _load_index_research_inputs, _simulate_index_targets, deflated_sharpe
    from strategies.index_close_location import STRICT_SWAP_BOOK, market_close_location

    spec = json.loads(TILT_PATH.read_text(encoding="utf-8"))
    if (list(spec.get("weights", {})) != list(TILT_KEYS) or "不读 2022" not in spec.get("sample", "")
            or "不读 2026" not in spec.get("sample", "")):
        raise ValueError("品种权重登记与实现不一致")
    index, bars, contracts, carry, ticks = _load_index_research_inputs()
    all_returns = (bars["openw"].shift(-1) - bars["openw"]) / bars["open"].where(bars["open"] > 0)
    returns = all_returns.dropna(how="any")
    C.assert_no_holdout_dates(returns.index, what="股指品种权重收益")
    market_clv = market_close_location(bars, index)
    frozen = STRICT_SWAP_BOOK.targets(market_clv, bars, carry)
    multipliers = tilt_multipliers(index, all_returns, market_clv)
    targets = {k: frozen * m.shift(1).fillna(1.0) for k, m in multipliers.items()}
    first = frozen.reindex(returns.index).ne(0).any(axis=1)
    rows, annual = [], []
    for label, start in (("2015-04-20 起", first.index[first][0]), ("2016-01-04 起", pd.Timestamp("2016-01-04"))):
        scored = returns.index[returns.index >= start]
        simulate = lambda target, limit: _simulate_index_targets(            # noqa: E731
            target.reindex(index=scored, columns=list(SYMBOLS)).fillna(0.0), scored, bars, contracts, ticks,
            SCORE_END.year, tuple(SYMBOLS), exposure_limit=1.0, drawdown_limit=limit, halt_days=S.HALT_DAYS)
        bench_daily, _, _ = simulate(pd.DataFrame(1.0, index=scored, columns=list(SYMBOLS)), 0.99)
        bench = bench_daily["净收益"].fillna(0.0)
        nets = {}
        for key in TILT_KEYS:
            daily, _, _ = simulate(targets[key], S.DRAWDOWN_LIMIT)
            net = daily["净收益"].fillna(0.0)
            nets[key] = net
            mean_w = multipliers[key].shift(1).reindex(scored).mean()
            rows.append({"起点": label, "权重": key, "年化净收益": performance(net)["ann_return"],
                         "净Sharpe": sharpe_ratio(net), "最大回撤": max_drawdown(daily["权益"], INITIAL_CAPITAL),
                         "相对买入持有beta": float(net.cov(bench) / bench.var()),
                         "平均敞口": float(daily["敞口倍数"].mean()),
                         **{f"平均乘数_{s}": float(mean_w[s]) for s in SYMBOLS}})
            for year, part in net.groupby(net.index.year):
                annual.append({"起点": label, "权重": key, "年份": int(year), "净收益": float((1 + part).prod() - 1)})
        trials = [sharpe_ratio(n) for n in nets.values()]
        for row in rows:
            if row["起点"] == label:
                diff = nets[row["权重"]] - nets["equal"]
                row["相对equal配对t"] = newey_west_t(diff, 10) if diff.abs().sum() > 0 else np.nan
                row["DSR"] = deflated_sharpe(nets[row["权重"]], trials, n_trials=len(trials))["dsr"]
    diagnostics = tilt_diagnostics(all_returns, market_clv)
    TILT_OUT.mkdir(parents=True, exist_ok=True)
    table, annual = pd.DataFrame(rows), pd.DataFrame(annual)
    table.to_csv(TILT_OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(TILT_OUT / "annual.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(TILT_OUT / "diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.concat({k: m for k, m in multipliers.items() if k in ("vol_parity", "rolling_strength")}, axis=1).to_csv(
        TILT_OUT / "multipliers.csv", encoding="utf-8-sig")
    (TILT_OUT / "protocol.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 260)
    print(table.round(3).to_string(index=False))
    print(annual.pivot_table(index=["起点", "权重"], columns="年份", values="净收益").round(3).to_string())
    print(diagnostics.round(3).to_string(index=False))
    print(f"结果：{TILT_OUT}")


CANDIDATES_PATH = C.CONFIG_DIR / "research" / "index_strategy_candidates.json"
CANDIDATES_OUT = C.RUNS_DIR / "index" / "index_strategy_candidates"
CANDIDATE_KEYS = ("A_frozen", "B_clv", "C_clv_short")


def candidate_targets(index, bars, carry) -> dict[str, pd.DataFrame]:
    """执行日目标（已后移一天）：冻结策略；只用等权收盘位置 < 0.5 做多；后者加冻结策略的空头腿。"""
    from strategies.index_close_location import STRICT_SWAP_BOOK, market_close_location

    clv = market_close_location(bars, index)
    plain = pd.DataFrame({s: clv.lt(0.5).astype(float) for s in SYMBOLS}, index=index)
    plain = plain.shift(1).fillna(0.0)
    return {"A_frozen": STRICT_SWAP_BOOK.targets(clv, bars, carry), "B_clv": plain,
            "C_clv_short": short_leg_targets(plain, bars, carry)}


def run_candidates() -> None:
    """登记在 index_strategy_candidates.json：冻结策略和 B，两个起点，五项指标全部报告。"""
    from backtest.engine import _load_index_research_inputs, _simulate_index_targets
    from research.time_factor_screen import ols_hac
    from strategies.index_close_location import market_close_location

    spec = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    if (list(spec.get("candidates", {})) != list(CANDIDATE_KEYS) or "不读 2022" not in spec.get("sample", "")
            or "不读 2026" not in spec.get("sample", "")):
        raise ValueError("股指策略候选登记与实现不一致")
    index, bars, contracts, carry, ticks = _load_index_research_inputs()
    targets = candidate_targets(index, bars, carry)
    all_returns = (bars["openw"].shift(-1) - bars["openw"]) / bars["open"].where(bars["open"] > 0)
    returns = all_returns.dropna(how="any")
    C.assert_no_holdout_dates(returns.index, what="股指策略候选收益")
    first = targets["A_frozen"].reindex(returns.index).ne(0).any(axis=1)
    rows, annual, correlations = [], [], {}
    for label, start in (("2015-04-20 起", first.index[first][0]), ("2016-01-04 起", pd.Timestamp("2016-01-04"))):
        scored = returns.index[returns.index >= start]
        simulate = lambda target, limit: _simulate_index_targets(            # noqa: E731
            target.reindex(index=scored, columns=list(SYMBOLS)).fillna(0.0), scored, bars, contracts, ticks,
            SCORE_END.year, tuple(SYMBOLS), exposure_limit=1.0, drawdown_limit=limit, halt_days=S.HALT_DAYS)
        bench_daily, _, _ = simulate(pd.DataFrame(1.0, index=scored, columns=list(SYMBOLS)), 0.99)
        bench = bench_daily["净收益"].fillna(0.0)
        nets = {}
        for key in CANDIDATE_KEYS:
            daily, _, _ = simulate(targets[key], S.DRAWDOWN_LIMIT)
            net = daily["净收益"].fillna(0.0)
            nets[key] = net
            stats = performance(net)
            years = (1 + net).groupby(net.index.year).prod() - 1
            rows.append({"起点": label, "候选": key, "年化净收益": stats["ann_return"], "年化波动": stats["ann_vol"],
                         "最大回撤": max_drawdown(daily["权益"], INITIAL_CAPITAL), "净Sharpe": sharpe_ratio(net),
                         "Calmar": stats["calmar"], "相对买入持有beta": float(net.cov(bench) / bench.var()),
                         "平均敞口": float(daily["敞口倍数"].mean()), "为正年份": f"{int((years > 0).sum())}/{len(years)}",
                         "年化成本": float((daily["交易费用"] / daily["期初权益"]).mean() * C.TRADING_DAYS_PER_YEAR),
                         "熔断次数": int((daily["停止新开仓"].astype(bool)
                                      & ~daily["停止新开仓"].astype(bool).shift(1, fill_value=False)).sum()),
                         "空头持仓日": int(targets[key].reindex(scored).lt(0).any(axis=1).sum())})
            for year, value in years.items():
                annual.append({"起点": label, "候选": key, "年份": int(year), "净收益": float(value)})
        b_stats = performance(bench)
        rows.append({"起点": label, "候选": "等权买入持有", "年化净收益": b_stats["ann_return"],
                     "年化波动": b_stats["ann_vol"], "最大回撤": max_drawdown(bench_daily["权益"], INITIAL_CAPITAL),
                     "净Sharpe": sharpe_ratio(bench), "Calmar": b_stats["calmar"], "相对买入持有beta": 1.0,
                     "平均敞口": float(bench_daily["敞口倍数"].mean())})
        for row in rows:
            if row["起点"] == label and row["候选"] in nets:
                diff = nets[row["候选"]] - nets["B_clv"]
                row["相对B配对t"] = newey_west_t(diff, 10) if diff.abs().sum() > 0 else np.nan
        correlations[label] = pd.DataFrame(nets).corr()
    clv = market_close_location(bars, index)
    horizon = []
    for lag in (1, 2, 3):
        y = all_returns.mean(axis=1, skipna=False).shift(-lag)
        for label, start in (("2015-04-16 起", pd.Timestamp("2015-04-16")), ("2016-01-04 起", pd.Timestamp("2016-01-04"))):
            frame = pd.concat([y, clv], axis=1).dropna()
            frame = frame[(frame.index >= start) & (frame.index <= SCORE_END)]
            b, t = ols_hac(frame.iloc[:, 0].to_numpy(), [frame.iloc[:, 1].to_numpy()], 5)
            horizon.append({"起点": label, "第几段": lag, "斜率_bp": b * 1e4, "NW_t": t, "天数": len(frame)})
    CANDIDATES_OUT.mkdir(parents=True, exist_ok=True)
    table, annual, horizon = pd.DataFrame(rows), pd.DataFrame(annual), pd.DataFrame(horizon)
    table.to_csv(CANDIDATES_OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(CANDIDATES_OUT / "annual.csv", index=False, encoding="utf-8-sig")
    horizon.to_csv(CANDIDATES_OUT / "horizon.csv", index=False, encoding="utf-8-sig")
    pd.concat(correlations).to_csv(CANDIDATES_OUT / "correlation.csv", encoding="utf-8-sig")
    (CANDIDATES_OUT / "protocol.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    pd.set_option("display.width", 280)
    print(table.round(3).to_string(index=False))
    print(annual.pivot_table(index=["起点", "候选"], columns="年份", values="净收益").round(3).to_string())
    print(horizon.round(2).to_string(index=False))
    for label, corr in correlations.items():
        print(label)
        print(corr.round(2).to_string())
    print(f"结果：{CANDIDATES_OUT}")


BEAR_OVERLAY_PATH = C.CONFIG_DIR / "research" / "index_bear_overlays.json"
BEAR_OVERLAY_OUT = C.RUNS_DIR / "index" / "index_bear_overlays"
BEAR_OVERLAY_KEYS = ("equal", "vol_parity", "rolling_strength", "vol_target")


def book_vol_scale(returns: pd.DataFrame) -> pd.Series:
    """收盘 t 已知的整本乘数：过去 20 段已实现的等权波动，不超过 1。"""
    from strategies.index_bear_reversal import close_vol_scale

    return close_vol_scale(returns.reindex(columns=list(SYMBOLS)).mean(axis=1))


def run_bear_overlays() -> None:
    """熊市空头的方向不动，只改品种权重或整本缩放。四档都报，不读 2022。"""
    from backtest.engine import _load_index_research_inputs, _simulate_index_targets, deflated_sharpe
    from strategies.index_bear_reversal import BOOK as BEAR_BOOK, unscaled_targets
    from strategies.index_close_location import market_close_location

    spec = json.loads(BEAR_OVERLAY_PATH.read_text(encoding="utf-8"))
    if (list(spec.get("weights", {})) != list(BEAR_OVERLAY_KEYS) or "不读 2022" not in spec.get("sample", "")
            or "不读 2026" not in spec.get("sample", "") or spec.get("base") != BEAR_BOOK.ID):
        raise ValueError("熊市空头权重登记与实现不一致")
    index, bars, contracts, _carry, ticks = _load_index_research_inputs()
    all_returns = (bars["openw"].shift(-1) - bars["openw"]) / bars["open"].where(bars["open"] > 0)
    returns = all_returns.dropna(how="any")
    C.assert_no_holdout_dates(returns.index, what="熊市空头权重收益")
    market_clv = market_close_location(bars, index)
    base = unscaled_targets(market_clv, bars)
    multipliers = tilt_multipliers(index, all_returns, market_clv)
    scale = book_vol_scale(all_returns).reindex(index)
    targets = {
        "equal": base,
        "vol_parity": base * multipliers["vol_parity"].shift(1).fillna(1.0),
        "rolling_strength": base * multipliers["rolling_strength"].shift(1).fillna(1.0),
        "vol_target": base.mul(scale.shift(1).fillna(1.0), axis=0),
    }
    rows, annual = [], []
    for label, start in (("2015-04-16 起", pd.Timestamp("2015-04-16")), ("2016-01-04 起", pd.Timestamp("2016-01-04"))):
        scored = returns.index[returns.index >= start]
        simulate = lambda target, limit: _simulate_index_targets(            # noqa: E731
            target.reindex(index=scored, columns=list(SYMBOLS)).fillna(0.0), scored, bars, contracts, ticks,
            SCORE_END.year, tuple(SYMBOLS), exposure_limit=BEAR_BOOK.GROSS,
            drawdown_limit=limit, halt_days=BEAR_BOOK.HALT_DAYS)
        nets = {}
        for key in BEAR_OVERLAY_KEYS:
            daily, _, _ = simulate(targets[key], BEAR_BOOK.DRAWDOWN_LIMIT)
            net = daily["净收益"].fillna(0.0)
            nets[key] = net
            stats = performance(net)
            rows.append({
                "起点": label, "权重": key, "年化净收益": stats["ann_return"], "年化波动": stats["ann_vol"],
                "净Sharpe": sharpe_ratio(net), "最大回撤": max_drawdown(daily["权益"], INITIAL_CAPITAL),
                "Calmar": stats["calmar"], "平均敞口": float(daily["敞口倍数"].mean()),
            })
            for year, part in net.groupby(net.index.year):
                annual.append({"起点": label, "权重": key, "年份": int(year), "净收益": float((1 + part).prod() - 1)})
        trials = [sharpe_ratio(n) for n in nets.values()]
        for row in rows:
            if row["起点"] == label:
                diff = nets[row["权重"]] - nets["equal"]
                row["相对equal配对t"] = newey_west_t(diff, 10) if diff.abs().sum() > 0 else np.nan
                row["DSR"] = deflated_sharpe(nets[row["权重"]], trials, n_trials=len(trials))["dsr"]
    BEAR_OVERLAY_OUT.mkdir(parents=True, exist_ok=True)
    table, annual = pd.DataFrame(rows), pd.DataFrame(annual)
    table.to_csv(BEAR_OVERLAY_OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(BEAR_OVERLAY_OUT / "annual.csv", index=False, encoding="utf-8-sig")
    (BEAR_OVERLAY_OUT / "protocol.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 240)
    print(table.round(3).to_string(index=False))
    print(annual.pivot_table(index=["起点", "权重"], columns="年份", values="净收益").round(3).to_string())
    print(f"结果：{BEAR_OVERLAY_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF/IH/IC 收盘位置长仓候选研究回测")
    parser.add_argument("--candidates", action="store_true", help="股指策略候选（index_strategy_candidates.json）")
    parser.add_argument("--tilt", action="store_true", help="IF/IH/IC 敞口分配（index_product_tilt.json）")
    parser.add_argument("--bear-overlays", action="store_true", help="熊市空头上的波动率和预测强弱权重")
    parser.add_argument("--variants", action="store_true", help="登记的收盘位置变体网格（index_close_location_variants.json）")
    parser.add_argument("--overwrite", action="store_true", help="重写同一预登记方案的研究输出")
    parser.add_argument("--robustness", action="store_true", help="登记过的稳健性检验 R1–R4，写到 robustness/")
    parser.add_argument("--short-leg", action="store_true", help="候选加空头腿，2015–2021（登记在 index_short_leg_2015_2021.json）")
    cli = parser.parse_args()
    if cli.candidates:
        run_candidates()
    elif cli.tilt:
        run_tilt()
    elif cli.bear_overlays:
        run_bear_overlays()
    elif cli.variants:
        run_variants()
    elif cli.short_leg:
        run_short_leg()
    elif cli.robustness:
        run_robustness()
    else:
        run(overwrite=cli.overwrite)
