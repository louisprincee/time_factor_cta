"""IF 单独上市到 IH、IC 上市前，用 IF 自己的收盘位置跑熊市空头。

行情和手续费写到 data/index/if_before_ih_ic/，不覆盖研究分片。
不读 2022，不读 2026。不是新策略。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import (
    INITIAL_CAPITAL, MARGIN_RATE, max_drawdown, performance, sharpe_ratio, simulate_target_lots,
)
from download.commodity.getFeeHistory import one_symbol
from download.index.getRiceQuantIndexFutures import download_symbol
from strategies.commodity_original import market_trend
from strategies.index_bear_reversal import (
    DRAWDOWN_LIMIT, GROSS, HALT_DAYS, THRESHOLD, close_vol_scale,
)
from tfcta import config as C
from tfcta.costs import (
    INDEX_MULTIPLIER, fee_tables, fill_close_today, index_open_limits, slippage_tables,
)
from tfcta.costs import estimate_ticks
from tfcta.data import daily_bars, save_shard
from tfcta.rq_auth import rqdata_credentials

ROOT = C.INDEX_DATA_ROOT / "if_before_ih_ic"
MINUTE_DIR = ROOT / "minute_shards"
OUT = C.RUNS_DIR / "index" / "if_before_ih_ic"
SPEC = C.PROJECT_ROOT / "config" / "research" / "if_before_ih_ic.json"
# 沪深300股指期货 2010-04-16 上市；IH、IC 2015-04-16 上市。
LISTING = pd.Timestamp("2010-04-16")
IH_IC_LISTING = pd.Timestamp("2015-04-16")
EXCHANGE_TICK = 0.2


def _load_or_download():
    price_path = MINUTE_DIR / "IF.parquet"
    fee_path = ROOT / "fee_history.csv"
    if price_path.is_file() and fee_path.is_file():
        minute = pd.read_parquet(price_path)
        fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
        return minute, fees
    credentials = rqdata_credentials()
    if credentials is None:
        raise SystemExit("请在 config/rqdata.env 填写米筐账号")
    import rqdatac as rq

    rq.init(*credentials)
    frame = download_symbol(rq, "IF", LISTING.date(), IH_IC_LISTING.date())
    dates = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(dates, what="IF 上市至 IH/IC 上市前")
    if dates.min() > LISTING:
        raise ValueError(f"IF 行情从 {dates.min().date()} 才开始，不是上市日 {LISTING.date()}")
    if dates.max() < IH_IC_LISTING:
        raise ValueError("缺少 IH、IC 上市当日的开盘，前一交易日的收益算不完")
    from download.index.getRiceQuantIndexFutures import split_research_frames

    prices, oi = split_research_frames(frame)
    save_shard(prices, MINUTE_DIR, "IF")
    save_shard(oi, MINUTE_DIR / "open_interest", "IF")
    fees = one_symbol(rq, "IF", "20100416", "20150416")
    if fees is None or fees.empty:
        raise ValueError("IF 这段没有手续费")
    fees = fill_close_today(fees)
    C.assert_no_holdout_dates(fees["trading_date"], what="IF 早期手续费")
    fee_path.parent.mkdir(parents=True, exist_ok=True)
    fees.to_csv(fee_path, index=False, date_format="%Y-%m-%d")
    return pd.read_parquet(price_path), fees


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if "不读 2022" not in spec["sample"] or "不读 2026" not in spec["sample"]:
        raise ValueError("样本边界没写明")
    minute, fees = _load_or_download()
    minute["trading_date"] = pd.to_datetime(minute["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(minute["trading_date"], what="IF 早期分钟")
    daily = daily_bars(minute)
    oi = pd.read_parquet(MINUTE_DIR / "open_interest" / "IF.parquet")
    oi["trading_date"] = pd.to_datetime(oi["trading_date"]).dt.normalize()
    contracts = oi["dominant_id"].groupby(oi["trading_date"]).first()
    contracts.index = pd.DatetimeIndex(contracts.index)
    index = daily.index.intersection(contracts.index).sort_values()
    daily = daily.reindex(index)
    contracts = contracts.reindex(index)
    if daily[["open", "openw", "close", "closew", "highw", "loww"]].isna().any().any():
        raise ValueError("IF 日线不完整")
    if contracts.isna().any():
        raise ValueError("IF 主力合约代码不完整")
    bars = {field: daily[[field]].rename(columns={field: "IF"}) for field in
            ("open", "openw", "close", "closew", "highw", "loww")}
    width = bars["highw"]["IF"] - bars["loww"]["IF"]
    if width.le(0).any():
        raise ValueError("IF 日内区间无效")
    clv = (bars["closew"]["IF"] - bars["loww"]["IF"]) / width
    if clv.lt(0).any() or clv.gt(1).any():
        raise ValueError("IF 收盘位置超出 [0, 1]")
    own_return = ((bars["closew"]["IF"] - bars["closew"]["IF"].shift(1))
                  / bars["close"]["IF"].shift(1))
    bear = market_trend(own_return).lt(0)
    side = pd.Series(0.0, index=index)
    side = side.mask(clv.lt(THRESHOLD), 1.0).mask(bear & clv.ge(THRESHOLD), -1.0)
    open_to_open = (bars["openw"]["IF"].shift(-1) - bars["openw"]["IF"]) / bars["open"]["IF"]
    scale = close_vol_scale(open_to_open).reindex(index).fillna(1.0)
    held = (side * scale).shift(1).fillna(0.0)
    scored = open_to_open.loc[open_to_open.index < IH_IC_LISTING].dropna().index
    C.assert_no_holdout_dates(scored, what="IF 早期计分日")
    if scored.max() >= IH_IC_LISTING or scored.min() < LISTING:
        raise ValueError("计分日越出 IF 单独上市的窗口")
    target = pd.DataFrame({"IF": held.reindex(scored).fillna(0.0)})
    prices = bars["open"].reindex(scored)
    selected = contracts.reindex(scored).to_frame("IF")
    rolls = selected.ne(selected.shift()).fillna(False)
    rolls.iloc[0] = False
    open_fee, close_fee, roll_fee = fee_tables(fees, prices)
    ticks = estimate_ticks(minute["close"], minute["trading_date"])
    if ticks.empty or not np.isclose(float(ticks.loc[ticks.year == 2010, "tick"].iloc[0]), EXCHANGE_TICK):
        raise ValueError(f"2010 年估计 tick 不是中金所的 {EXCHANGE_TICK}：{ticks.to_dict('records')}")
    ticks.insert(0, "symbol", "IF")
    ticks = pd.concat(
        [ticks, pd.DataFrame({"symbol": ["IF"], "year": [2009], "tick": [EXCHANGE_TICK]})],
        ignore_index=True,
    )
    slippage = slippage_tables(ticks[ticks.year < IH_IC_LISTING.year], prices)
    universe = {int(year): ["IF"] for year in range(int(scored.min().year), int(scored.max().year) + 1)}
    daily_out, positions, events = simulate_target_lots(
        target, prices,
        bars["close"].reindex(scored), bars["openw"].reindex(scored),
        (bars["openw"]["IF"].shift(-1) - bars["openw"]["IF"]).reindex(scored).to_frame("IF"),
        bars["highw"].reindex(scored), bars["loww"].reindex(scored),
        rolls, selected, universe, {"IF": INDEX_MULTIPLIER["IF"]},
        open_fee, close_fee, roll_fee, slippage,
        initial_capital=INITIAL_CAPITAL, margin_rate=MARGIN_RATE,
        exposure_limit=GROSS, drawdown_limit=DRAWDOWN_LIMIT, halt_days=HALT_DAYS,
        open_limits=index_open_limits(scored, ["IF"]),
    )
    net = daily_out["净收益"].fillna(0.0)
    metrics = performance(net)
    summary = {
        "品种": "IF",
        "起点": str(scored.min().date()),
        "终点": str(scored.max().date()),
        "初始资金": INITIAL_CAPITAL,
        "年化收益": metrics["ann_return"],
        "年化波动": metrics["ann_vol"],
        "Sharpe": sharpe_ratio(net),
        "最大回撤": max_drawdown(daily_out["权益"], INITIAL_CAPITAL),
        "Calmar比率": metrics["calmar"],
        "期末权益": float(daily_out["权益"].iloc[-1]),
        "做多天数": int((held.reindex(scored) > 0).sum()),
        "做空天数": int((held.reindex(scored) < 0).sum()),
        "开仓受限数": int(daily_out["开仓受限数"].sum()),
        "最大持仓手数": int(daily_out["持仓总手数"].max()),
        "首个空头信号": str(side[side < 0].index.min().date()) if (side < 0).any() else "",
    }
    annual_rows = []
    for year, part in daily_out.groupby(daily_out.index.year):
        opening = float(part["期初权益"].iloc[0])
        year_net = part["净收益"].fillna(0.0)
        annual_rows.append({
            "年份": int(year),
            "净收益": float(part["权益"].iloc[-1] / opening - 1.0),
            "Sharpe": sharpe_ratio(year_net),
            "年度内最大回撤": max_drawdown(part["权益"], opening),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    pd.DataFrame(annual_rows).to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    daily_out.to_csv(OUT / "daily.csv", encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    print(pd.DataFrame(annual_rows).to_string(index=False), flush=True)
    print(f"结果：{OUT}", flush=True)


if __name__ == "__main__":
    main()
