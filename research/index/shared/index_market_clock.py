"""机制研究共用的研究期日线与时间因子装载；没有策略回测入口。"""
from __future__ import annotations
import pandas as pd
from tfcta import config as C
from tfcta.costs import estimate_ticks
from tfcta.data import add_intraday_coords, daily_bars, load_shard
from tfcta.factors import (day_codes_of, duration_factors, intraday_abs_diff, raw_price_path, report_factors, rolling_threshold, timestamp_factors)

SYMBOLS = ("IF", "IH", "IC")
FACTORS = ("ts_low", "ts_high", "dfp_top3", "ts_low_pm", "ts_high_pm")
SCORE_END = pd.Timestamp("2021-12-31")
WARMUP_START = pd.Timestamp("2015-01-01")
THRESHOLD_LOOKBACK = 250
THRESHOLD_PCT = 55.0


def _daily_factor_values(frame: pd.DataFrame) -> pd.DataFrame:
    """按历史研究因子的 N250/M55 口径计算日内时戳与 DFP。"""
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
            columns=list(dict.fromkeys([*C.FACTOR_FIELDS, *C.PRICE_FIELDS, "dominant_id"])),
        )
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
