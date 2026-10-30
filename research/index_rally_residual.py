"""冲高残差择多：研究期账本对照，开盘按 L1 前 30 分钟报价成交。

只读 IF/IH/IC 研究分片、研究期 L1（2015–2021）和 IH/IC 上市前的 IF 专用分片。不读 2022，不读 2026。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import (
    INDEX_EXCHANGE_TICK, INDEX_RESEARCH_END, INDEX_SYMBOLS, INITIAL_CAPITAL, _load_index_research_inputs,
    _simulate_index_targets, max_drawdown, newey_west_t, performance, sharpe_ratio,
)
from research.if_before_ih_ic import early_inputs, early_ledger
from strategies.index_bear_reversal import BOOK as FROZEN, THRESHOLD, _open_to_open, close_vol_scale, decision_side
from strategies.index_bear_short_rally_residual import (
    BOOK, day_features, long_weight, rally_residual, weighted_targets,
)
from strategies.index_close_location import market_close_location
from tfcta import config as C
from tfcta.costs import index_open_limits

L1_DIR = C.INDEX_DATA_ROOT / "external_rqdata" / "l1_ticks"
OUT = C.RUNS_DIR / "index" / "index_rally_residual"
START_2016 = pd.Timestamp("2016-01-04")
WINDOW = pd.Timedelta(30, unit="min")
PARTICIPATION_LIMIT = 0.10
BOOKS = ("frozen", "rally_residual", "constant_share")
LABELS = {"frozen": "冻结熊市空头", "rally_residual": "冲高残差择多", "constant_share": "固定比例多头"}
OPEN_KINDS = {"开仓": 1, "加仓": 1, "开空": -1, "加空": -1}
CLOSE_KINDS = {"平仓", "减仓", "清仓", "平空", "减空"}


def continuous_start(day: pd.Timestamp) -> pd.Timedelta:
    """中金所股指期货连续交易起点：2016 年起 09:30，此前 09:15。"""
    return pd.Timedelta(
        9 * 60 + (15 if day < pd.Timestamp("2016-01-01") else 30), unit="min"
    )


def opening_quotes(ticks: pd.DataFrame) -> pd.DataFrame:
    """每个交易日：开盘价、集合竞价量、连续交易前 30 分钟有效卖一/买一均值和成交量。"""
    rows = []
    for day, part in ticks.groupby("trading_date", sort=True):
        day = pd.Timestamp(day)
        contracts = part["contract"].dropna().unique()
        if len(contracts) != 1:
            raise ValueError(f"{day.date()} L1 当日不止一个合约：{contracts}")
        offset = part["datetime"] - day
        start = continuous_start(day)
        before = part[offset < start]
        window = part[(offset >= start) & (offset < start + WINDOW)]
        valid = window[(window["a1"] > 0) & (window["b1"] > 0) & (window["a1"] >= window["b1"])]
        auction = float(before["volume"].max()) if not before.empty else 0.0
        opens = part["open"].dropna()
        rows.append({
            "trading_date": day, "contract": contracts[0],
            "tick_open": float(opens.iloc[0]) if not opens.empty else np.nan,
            "auction_volume": auction,
            "ask": float(valid["a1"].mean()) if not valid.empty else np.nan,
            "bid": float(valid["b1"].mean()) if not valid.empty else np.nan,
            "volume": float(window["volume"].max() - auction) if not window.empty else np.nan,
            "quotes": int(len(valid)),
        })
    return pd.DataFrame(rows)


def load_opening_quotes(cache_path=None) -> pd.DataFrame:
    """读研究期 L1 开盘前后（10:05 以前）的报价，缓存每日摘要。"""
    cache = Path(cache_path) if cache_path is not None else OUT / "l1_opening_30m.parquet"
    if cache.is_file():
        quotes = pd.read_parquet(cache)
        C.assert_no_holdout_dates(quotes["trading_date"], what="L1 开盘缓存")
        return quotes
    parts = []
    for symbol in INDEX_SYMBOLS:
        for path in sorted((L1_DIR / symbol).glob("*/*.parquet")):
            C.assert_research_only(path)
            ticks = pd.read_parquet(path, columns=["datetime", "trading_date", "open", "volume", "a1", "b1", "contract"])
            ticks["trading_date"] = pd.to_datetime(ticks["trading_date"]).dt.normalize()
            C.assert_no_holdout_dates(ticks["trading_date"], what=f"{symbol} L1")
            ticks = ticks[ticks["datetime"] - ticks["trading_date"] < pd.Timedelta("10:05:00")]
            frame = opening_quotes(ticks)
            frame.insert(1, "symbol", symbol)
            parts.append(frame)
        print(f"{symbol} L1 开盘摘要完成", flush=True)
    quotes = pd.concat(parts, ignore_index=True)
    if quotes.duplicated(["trading_date", "symbol"]).any():
        raise ValueError("L1 开盘摘要日期重复")
    cache.parent.mkdir(parents=True, exist_ok=True)
    quotes.to_parquet(cache, index=False)
    return quotes


def l1_slippage(
    quotes: pd.DataFrame, bars: dict, contracts: pd.DataFrame, scored: pd.DatetimeIndex,
    symbols=None,
):
    """买入 (卖一均值 − 开盘)/开盘，卖出 (开盘 − 买一均值)/开盘。合约或开盘价对不上的格子留缺，用到时账本报错。"""
    if isinstance(symbols, str):
        raise TypeError("symbols 要用序列，不能是单个字符串")
    names = list(INDEX_SYMBOLS if symbols is None else symbols)
    opened = bars["open"].reindex(index=scored, columns=names)
    pick = lambda field: quotes.pivot(index="trading_date", columns="symbol", values=field).reindex(
        index=scored, columns=names)
    ok = pick("contract").eq(contracts.reindex(index=scored, columns=names)) & (
        (pick("tick_open") - opened).abs() < 1e-6)
    buy = ((pick("ask") - opened) / opened).where(ok)
    sell = ((opened - pick("bid")) / opened).where(ok)
    return {"buy": buy, "sell": sell}, int((~ok).sum().sum()), pick("volume")


def open_legs(events: pd.DataFrame, positions: pd.DataFrame, scored, limits) -> pd.DataFrame:
    """按开盘成交的每一腿：方向（+1 买、−1 卖）和手数。换月只取新合约那一腿。"""
    lots = positions.pivot(index="交易日", columns="品种", values="手数").reindex(scored).fillna(0.0)
    prior = lots.shift(1).fillna(0.0)
    rows = []
    for row in events.itertuples(index=False):
        kind, symbol, day = row.事件, row.品种, pd.Timestamp(row.交易日)
        if kind in OPEN_KINDS:
            side, quantity = OPEN_KINDS[kind], row.手数
        elif kind in CLOSE_KINDS:
            side, quantity = -int(np.sign(prior.at[day, symbol])), row.手数
        elif kind == "换月":
            side, quantity = int(np.sign(prior.at[day, symbol])), int(min(row.手数, limits.at[day, symbol]))
        else:
            continue
        if quantity:
            rows.append({"交易日": day, "品种": symbol, "方向": side, "手数": int(quantity)})
    return pd.DataFrame(rows, columns=["交易日", "品种", "方向", "手数"])


def metrics_row(daily: pd.DataFrame, **labels) -> dict:
    net = daily["净收益"].fillna(0.0)
    m = performance(net)
    row = {**labels, "起点": str(net.index[0].date()), "终点": str(net.index[-1].date()),
           "年化收益": m["ann_return"], "年化波动": m["ann_vol"],
           "最大回撤": max_drawdown(daily["权益"], INITIAL_CAPITAL), "Sharpe": sharpe_ratio(net),
           "Calmar比率": m["calmar"], "平均敞口": float(daily["敞口倍数"].fillna(0.0).mean()),
           "年化成本占权益": float((daily["交易费用"] / daily["期初权益"]).sum()
                             / (len(net) / C.TRADING_DAYS_PER_YEAR))}
    for year, part in daily.groupby(daily.index.year):
        row[f"{year}净收益"] = float(part["权益"].iloc[-1] / part["期初权益"].iloc[0] - 1.0)
    return row


def paired_rows(nets: dict[str, pd.Series], sample: str, model: str) -> list[dict]:
    rows = []
    for base in ("frozen", "constant_share"):
        diff = nets["rally_residual"] - nets[base]
        years = diff.groupby(diff.index.year).sum()
        rows.append({"样本": sample, "成交口径": model, "对照": LABELS[base],
                     "日均差_bp": float(diff.mean() * 1e4), "年化差": float(diff.mean() * C.TRADING_DAYS_PER_YEAR),
                     "NW_t_lag10": newey_west_t(diff, lags=10), "逐年为正": f"{int((years > 0).sum())}/{len(years)}"})
    return rows


def group_table(clv: pd.Series, residual: pd.Series, outcome: pd.Series, sample: str) -> pd.DataFrame:
    """弱收盘信号日按残差分组：下一开盘段（执行日开盘到再下一开盘）毛收益。"""
    frame = pd.DataFrame({"clv": clv, "residual": residual, "outcome": outcome}).dropna(subset=["clv", "outcome"])
    frame = frame[frame["clv"].lt(THRESHOLD)]
    frame["组"] = np.where(frame["residual"].isna(), "未知", np.where(frame["residual"] > 0, "残差>0", "残差≤0"))
    rows = []
    cells = [((int(year), group), part) for (year, group), part in frame.groupby([frame.index.year, "组"])]
    cells += [(("合计", group), part) for group, part in frame.groupby("组")]
    for (year, group), part in cells:
        rows.append({"样本": sample, "年份": year, "组": group, "天数": len(part),
                     "正收益比例": float(part["outcome"].gt(0).mean()),
                     "平均_bp": float(part["outcome"].mean() * 1e4), "中位_bp": float(part["outcome"].median() * 1e4)})
    return pd.DataFrame(rows)


def kept_share(side: pd.Series, weight: pd.Series, scored) -> float:
    """计分期内（按执行日）弱收盘做多日里新规则保留做多的比例。"""
    held_side = side.shift(1).reindex(scored)
    held_weight = weight.reindex(side.index).shift(1).reindex(scored)
    longs = held_side.gt(0)
    return float(held_weight[longs].mean())


def run() -> None:
    config = BOOK.declared()
    FROZEN.declared()
    OUT.mkdir(parents=True, exist_ok=True)
    index, bars, contracts, _carry, ticks = _load_index_research_inputs()
    open_to_open = (bars["openw"].shift(-1) - bars["openw"]) / bars["open"]
    scored = open_to_open.dropna(how="any").index
    C.assert_no_holdout_dates(scored, what="冲高残差研究期计分日")
    strategy_bars = {key: frame.reindex(index=index, columns=INDEX_SYMBOLS) for key, frame in bars.items()}
    clv = market_close_location(strategy_bars, index)
    side = decision_side(clv, strategy_bars)
    scale = close_vol_scale(_open_to_open(strategy_bars, side.index))
    residual = rally_residual(day_features(strategy_bars).reindex(index), clv)
    weight = long_weight(residual)
    outcome = open_to_open.mean(axis=1).shift(-1)       # 信号日 t：t+1 开盘到 t+2 开盘

    groups = [group_table(clv, residual, outcome, "共同")]
    summary, paired, shares = [], [], []
    limits = index_open_limits(scored, INDEX_SYMBOLS)
    quotes = load_opening_quotes()
    slip, bad_cells, volume = l1_slippage(quotes, bars, contracts, scored)
    print(f"L1 合约或开盘价与账本不符的品种日：{bad_cells}", flush=True)
    participation, leg_costs = [], []
    for sample, days in (("共同2015起", scored), ("共同2016起", scored[scored >= START_2016])):
        share = kept_share(side, weight, days)
        shares.append({"样本": sample, "保留做多比例": share})
        targets = {"frozen": weighted_targets(side, scale, 1.0),
                   "rally_residual": weighted_targets(side, scale, weight),
                   "constant_share": weighted_targets(side, scale, share)}
        if sample == "共同2015起":
            if not targets["frozen"].equals(FROZEN.targets(clv, strategy_bars)):
                raise AssertionError("固定权重 1 的对照与冻结规则不一致")
            if not targets["rally_residual"].equals(BOOK.targets(clv, strategy_bars)):
                raise AssertionError("冲高残差目标与策略模块不一致")
        models = ("tick", "l1_30m") if sample == "共同2015起" else ("l1_30m",)
        for model in models:
            nets = {}
            for name in BOOKS:
                daily, positions, events = _simulate_index_targets(
                    targets[name], days, bars, contracts, ticks, INDEX_RESEARCH_END.year, INDEX_SYMBOLS,
                    exposure_limit=BOOK.GROSS, drawdown_limit=BOOK.DRAWDOWN_LIMIT, halt_days=BOOK.HALT_DAYS,
                    initial_capital=INITIAL_CAPITAL, open_limits=True,
                    open_slippage=slip if model == "l1_30m" else None,
                )
                nets[name] = daily["净收益"].fillna(0.0)
                summary.append(metrics_row(daily, 样本=sample, 成交口径=model, 账本=LABELS[name]))
                if sample == "共同2015起" and model == "l1_30m":
                    daily.to_csv(OUT / f"daily_{name}.csv", encoding="utf-8-sig")
                    legs = open_legs(events, positions, days, limits)
                    rate = np.where(legs["方向"] > 0,
                                    [slip["buy"].at[d, s] for d, s in zip(legs["交易日"], legs["品种"])],
                                    [slip["sell"].at[d, s] for d, s in zip(legs["交易日"], legs["品种"])])
                    legs["成本_跳"] = rate * [bars["open"].at[d, s] for d, s in zip(legs["交易日"], legs["品种"])] / INDEX_EXCHANGE_TICK
                    legs["账本"] = LABELS[name]
                    leg_costs.append(legs)
                    per_day = legs.groupby(["交易日", "品种"])["手数"].sum().reset_index()
                    per_day["30分钟成交量"] = [volume.at[d, s] for d, s in zip(per_day["交易日"], per_day["品种"])]
                    per_day["参与率"] = per_day["手数"] / per_day["30分钟成交量"].where(per_day["30分钟成交量"] > 0)
                    for (symbol, year), part in per_day.groupby(["品种", per_day["交易日"].dt.year]):
                        participation.append({"账本": LABELS[name], "品种": symbol, "年份": int(year),
                                              "成交品种日": len(part), "中位手数": float(part["手数"].median()),
                                              "参与率中位": float(part["参与率"].median()),
                                              "参与率最大": float(part["参与率"].max()),
                                              "超过10%占比": float(part["参与率"].gt(PARTICIPATION_LIMIT).mean())})
            paired += paired_rows(nets, sample, model)
        print(f"{sample} 完成", flush=True)

    early = early_inputs()
    eb = early["bars"]
    early_clv = (eb["closew"]["IF"] - eb["loww"]["IF"]) / (eb["highw"]["IF"] - eb["loww"]["IF"])
    early_residual = rally_residual(day_features(eb, ("IF",)), early_clv)
    early_weight = long_weight(early_residual)
    early_oto = (eb["openw"]["IF"].shift(-1) - eb["openw"]["IF"]) / eb["open"]["IF"]
    next_day = pd.Series(early_oto.index, index=early_oto.index).shift(-1)
    early_outcome = early_oto.shift(-1).where(next_day.isin(early["scored"]))
    groups.append(group_table(early_clv, early_residual, early_outcome, "早期IF"))
    share = kept_share(early["side"], early_weight, early["scored"])
    shares.append({"样本": "早期IF", "保留做多比例": share})
    nets = {}
    for name, w in (("frozen", 1.0), ("rally_residual", early_weight), ("constant_share", share)):
        held = weighted_targets(early["side"], early["scale"], w, ("IF",))["IF"]
        daily, _, _ = early_ledger(held, early)
        nets[name] = daily["净收益"].fillna(0.0)
        summary.append(metrics_row(daily, 样本="早期IF", 成交口径="tick", 账本=LABELS[name]))
    paired += paired_rows(nets, "早期IF", "tick")

    legs = pd.concat(leg_costs, ignore_index=True)
    cost = legs.groupby(["账本", "方向"]).apply(lambda g: pd.Series({
        "腿数": len(g), "手数加权_跳": float((g["成本_跳"] * g["手数"]).sum() / g["手数"].sum()),
        "中位_跳": float(g["成本_跳"].median()), "90分位_跳": float(g["成本_跳"].quantile(0.9))})).reset_index()
    tables = {"summary": pd.DataFrame(summary), "paired": pd.DataFrame(paired), "groups": pd.concat(groups),
              "kept_share": pd.DataFrame(shares), "open_cost": cost, "participation": pd.DataFrame(participation)}
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    (OUT / "protocol.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        cols = ["样本", "成交口径", "账本", "年化收益", "年化波动", "最大回撤", "Sharpe", "Calmar比率", "平均敞口"]
        print(tables["summary"][cols].to_string(index=False))
        print(tables["paired"].to_string(index=False))
        print(tables["kept_share"].to_string(index=False))
        g = tables["groups"]
        print(g[g["年份"] == "合计"].to_string(index=False))
        print(cost.to_string(index=False))
    print(f"结果：{OUT}", flush=True)


if __name__ == "__main__":
    run()
