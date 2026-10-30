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
OPENING_QUOTES = ROOT / "l1_opening_30m.parquet"
EXEC_START = pd.Timestamp("2011-01-01")
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


def early_inputs() -> dict:
    """IF 单独上市期的日线、主力合约、手续费、tick 和计分日。"""
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
    scored = open_to_open.loc[open_to_open.index < IH_IC_LISTING].dropna().index
    C.assert_no_holdout_dates(scored, what="IF 早期计分日")
    if scored.max() >= IH_IC_LISTING or scored.min() < LISTING:
        raise ValueError("计分日越出 IF 单独上市的窗口")
    ticks = estimate_ticks(minute["close"], minute["trading_date"])
    if ticks.empty or not np.isclose(float(ticks.loc[ticks.year == 2010, "tick"].iloc[0]), EXCHANGE_TICK):
        raise ValueError(f"2010 年估计 tick 不是中金所的 {EXCHANGE_TICK}：{ticks.to_dict('records')}")
    ticks.insert(0, "symbol", "IF")
    ticks = pd.concat(
        [ticks, pd.DataFrame({"symbol": ["IF"], "year": [2009], "tick": [EXCHANGE_TICK]})],
        ignore_index=True,
    )
    return {"bars": bars, "contracts": contracts, "fees": fees, "ticks": ticks,
            "side": side, "scale": scale, "scored": scored}


def _read_opening_cache() -> pd.DataFrame | None:
    if not OPENING_QUOTES.is_file():
        return None
    frame = pd.read_parquet(OPENING_QUOTES)
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(frame["trading_date"], what="IF 早期开盘报价缓存")
    if frame["trading_date"].min() < EXEC_START or frame["trading_date"].max() >= IH_IC_LISTING:
        raise ValueError("IF 早期开盘报价缓存越出 2011-01-01 至 IH/IC 上市前")
    if frame.duplicated("trading_date").any():
        raise ValueError("IF 早期开盘报价日期重复")
    return frame


def call_auction_volume(ticks: pd.DataFrame, day: pd.Timestamp) -> float:
    """集合竞价累计成交量。

    起点之前有 tick 时用那段，与 2015–2021 的开盘摘要相同。
    米筐在 2011–2014 经常把竞价成交记在 09:15:00 这一笔上；那种日子没有更早的 tick，就用这一笔。
    """
    from research.index_rally_residual import continuous_start

    start = continuous_start(pd.Timestamp(day))
    offset = pd.to_datetime(ticks["datetime"]) - pd.Timestamp(day).normalize()
    before = ticks[offset < start]
    if not before.empty:
        volume = float(before["volume"].max())
    else:
        exact = ticks[offset == start]
        if exact.empty or not np.isfinite(exact["volume"]).all():
            raise ValueError(f"{pd.Timestamp(day).date()} 没有集合竞价成交时点")
        volume = float(exact["volume"].max())
    if volume < 0 or not np.isfinite(volume):
        raise ValueError(f"{pd.Timestamp(day).date()} 的集合竞价成交量无效")
    return volume


def _opening_row(ticks: pd.DataFrame, day: pd.Timestamp, contract: str) -> dict:
    """一日 tick 只留 10:05 以前，再做成与研究期相同的开盘摘要。"""
    from research.index_rally_residual import opening_quotes

    frame = ticks.reset_index()
    if "datetime" not in frame.columns:
        raise ValueError(f"{day.date()} {contract} 的 tick 没有 datetime")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    if getattr(frame["datetime"].dt, "tz", None) is not None:
        frame["datetime"] = frame["datetime"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    if "trading_date" not in frame.columns:
        frame["trading_date"] = frame["datetime"].dt.normalize()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    if "order_book_id" in frame.columns and not frame["order_book_id"].astype(str).eq(contract).all():
        raise ValueError(f"{day.date()} 返回的合约不是 {contract}")
    C.assert_no_holdout_dates(frame["trading_date"], what=f"{contract} 开盘 tick")
    if frame["trading_date"].max() >= IH_IC_LISTING or not frame["trading_date"].eq(day).all():
        raise ValueError(f"{day.date()} {contract} 的 tick 日期不是这一天")
    morning = frame[frame["datetime"] - frame["trading_date"] < pd.Timedelta("10:05:00")].copy()
    missing = {"open", "volume", "a1", "b1"} - set(morning.columns)
    if missing:
        raise ValueError(f"{day.date()} {contract} 的 tick 缺少 {sorted(missing)}")
    morning["contract"] = contract
    try:
        auction = call_auction_volume(morning, day)
    except ValueError:
        return _gap_row(frame, day, contract)
    summary = opening_quotes(morning)
    if len(summary) != 1:
        raise ValueError(f"{day.date()} {contract} 没有可用的开盘 tick")
    row = summary.iloc[0].to_dict()
    row["volume"] = float(row["volume"]) + float(row["auction_volume"]) - auction
    row["auction_volume"] = auction
    row["open_gap"] = False
    if row["auction_volume"] < 0 or not np.isfinite(row["ask"]) or not np.isfinite(row["bid"]):
        raise ValueError(f"{day.date()} {contract} 的集合竞价量或开盘买卖报价无效")
    if row["volume"] < 0 or row["quotes"] <= 0:
        raise ValueError(f"{day.date()} {contract} 的开盘 30 分钟没有有效报价")
    row["symbol"] = "IF"
    return row


def _gap_row(frame: pd.DataFrame, day: pd.Timestamp, contract: str) -> dict:
    """开盘成交没被记下来。不把后面的报价当成竞价量或 30 分钟均价。"""
    opens = frame["open"].dropna() if "open" in frame.columns else pd.Series(dtype="float64")
    return {
        "trading_date": pd.Timestamp(day).normalize(),
        "contract": contract,
        "tick_open": float(opens.iloc[0]) if not opens.empty else np.nan,
        "auction_volume": 0.0,
        "ask": np.nan,
        "bid": np.nan,
        "volume": np.nan,
        "quotes": 0,
        "symbol": "IF",
        "open_gap": True,
    }


def load_early_opening_quotes(contracts: pd.Series) -> pd.DataFrame:
    """2011 年至 IH/IC 上市前，IF 主力开盘摘要。缺日从米筐补，只写这段专用目录。"""
    need = contracts[(contracts.index >= EXEC_START) & (contracts.index < IH_IC_LISTING)].sort_index()
    if need.empty or need.isna().any():
        raise ValueError("IF 早期开盘报价没有完整的主力合约")
    C.assert_no_holdout_dates(need.index, what="IF 早期开盘报价")
    cached = _read_opening_cache()
    have = set(cached["trading_date"]) if cached is not None else set()
    missing = need.index.difference(have)
    if len(missing):
        credentials = rqdata_credentials()
        if credentials is None:
            raise SystemExit("请在 config/rqdata.env 填写米筐账号")
        import rqdatac as rq

        rq.init(*credentials)
        rows = [] if cached is None else cached.to_dict("records")
        runs: list[list] = []
        previous = None
        for day in missing:
            contract = str(need.at[day])
            gap = None if previous is None else (pd.Timestamp(day) - pd.Timestamp(previous)).days
            if runs and runs[-1][0] == contract and gap is not None and gap <= 10:
                runs[-1][1].append(pd.Timestamp(day))
            else:
                runs.append([contract, [pd.Timestamp(day)]])
            previous = day
        done = len(rows)
        for contract, days in runs:
            ticks = rq.get_price(
                contract, start_date=days[0].date(), end_date=days[-1].date(), frequency="tick",
                fields=["trading_date", "open", "volume", "a1", "b1"],
            )
            if ticks is None or len(ticks) == 0:
                raise ValueError(f"{days[0].date()}–{days[-1].date()} {contract} 没有 tick 行情")
            traded = pd.to_datetime(ticks["trading_date"]).dt.normalize()
            for day in days:
                part = ticks[traded.eq(day).to_numpy()]
                if part.empty:
                    raise ValueError(f"{day.date()} {contract} 没有 tick 行情")
                rows.append(_opening_row(part, day, contract))
            done = len(rows)
            frame = pd.DataFrame(rows)
            frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
            OPENING_QUOTES.parent.mkdir(parents=True, exist_ok=True)
            temporary = OPENING_QUOTES.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(OPENING_QUOTES)
            print(f"IF 开盘报价 {done}/{len(need)} 天，至 {days[-1].date()}", flush=True)
        cached = _read_opening_cache()
    if cached is None or not set(need.index).issubset(set(cached["trading_date"])):
        raise ValueError("IF 早期开盘报价没有盖住全部计分所需交易日")
    out = cached[cached["trading_date"].isin(need.index)].sort_values("trading_date")
    if "open_gap" not in out.columns:
        out["open_gap"] = False
    out["open_gap"] = out["open_gap"].eq(True)
    want = need.reindex(pd.DatetimeIndex(out["trading_date"])).astype(str)
    if not out["contract"].astype(str).eq(want.to_numpy()).all():
        raise ValueError("IF 早期开盘报价的合约与分钟线主力不一致")
    return out.reset_index(drop=True)


def early_ledger(held: pd.Series, inputs: dict, *, initial_capital: float = INITIAL_CAPITAL,
                 open_slippage: dict | None = None):
    """把收盘后已后移的 IF 目标权重放进整数手账本，开仓按当日中金所限额。"""
    bars, contracts, fees, ticks, scored = (inputs[k] for k in ("bars", "contracts", "fees", "ticks", "scored"))
    target = pd.DataFrame({"IF": held.reindex(scored).fillna(0.0)})
    prices = bars["open"].reindex(scored)
    selected = contracts.reindex(scored).to_frame("IF")
    rolls = selected.ne(selected.shift()).fillna(False)
    rolls.iloc[0] = False
    open_fee, close_fee, roll_fee = fee_tables(fees, prices)
    slippage = slippage_tables(ticks[ticks.year < IH_IC_LISTING.year], prices)
    universe = {int(year): ["IF"] for year in range(int(scored.min().year), int(scored.max().year) + 1)}
    daily_out, positions, events = simulate_target_lots(
        target, prices,
        bars["close"].reindex(scored), bars["openw"].reindex(scored),
        (bars["openw"]["IF"].shift(-1) - bars["openw"]["IF"]).reindex(scored).to_frame("IF"),
        bars["highw"].reindex(scored), bars["loww"].reindex(scored),
        rolls, selected, universe, {"IF": INDEX_MULTIPLIER["IF"]},
        open_fee, close_fee, roll_fee, slippage,
        initial_capital=initial_capital, margin_rate=MARGIN_RATE,
        exposure_limit=GROSS, drawdown_limit=DRAWDOWN_LIMIT, halt_days=HALT_DAYS,
        open_limits=index_open_limits(scored, ["IF"]),
        open_slippage=None if open_slippage is None else {
            key: frame.reindex(index=scored, columns=["IF"]) for key, frame in open_slippage.items()
        },
    )
    return daily_out, positions, events


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if "不读 2022" not in spec["sample"] or "不读 2026" not in spec["sample"]:
        raise ValueError("样本边界没写明")
    inputs = early_inputs()
    side, scale, scored = inputs["side"], inputs["scale"], inputs["scored"]
    held = (side * scale).shift(1).fillna(0.0)
    daily_out, _positions, _events = early_ledger(held, inputs)
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
