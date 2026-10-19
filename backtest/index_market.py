"""股指期货的整数手数数据。日盘自己的分钟做成时钟，不套商品夜盘网格。

研究期只读 IF、IH、IC 的 2021 及以前分片。2022 与 2023–2025 都在锁定目录里，
验证和样本外各自按日期截开，截完再核对边界。
"""
from __future__ import annotations

import pandas as pd

from tfcta import config as C
from tfcta.costs import (
    estimate_ticks, fee_tables, index_fee_path, load_index_fees, slippage_tables,
)
from tfcta.data import (
    add_intraday_coords, build_universe, collect_stats, daily_bars, daily_stats,
    find_shard, load_shard, screen_year, stats_from_daily,
)
from tfcta.factors import SignalSet, timestamp_factors

RESEARCH_SYMBOLS = ("IF", "IH", "IC")
SYMBOLS = ("IF", "IH", "IC", "IM")
MINUTE_COLUMNS = [
    "trading_date", "open", "openw", "close", "closew", "highw", "loww",
    "volume", "total_turnover",
]
OI_COLUMNS = ["trading_date", "dominant_id"]


def assert_day_session(index, symbol: str) -> None:
    """股指没有夜盘。出现夜盘墙钟就停下，避免按商品夜盘网格解释时点。"""
    hours = pd.DatetimeIndex(index).hour
    if ((hours >= C.NIGHT_START_HOUR) | (hours <= C.NIGHT_END_HOUR)).any():
        raise ValueError(f"{symbol} 的分钟里有夜盘时钟，股指只使用日盘")


def dominant_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """主力代码。首日也写一行，后面的换月才能往前填充合约。"""
    if "dominant_id" not in frame.columns or "trading_date" not in frame.columns:
        raise KeyError("股指持仓分片缺少 dominant_id 或 trading_date")
    clean = frame.dropna(subset=["dominant_id"]).copy()
    clean["trading_date"] = pd.to_datetime(clean["trading_date"]).dt.normalize()
    daily = clean.groupby("trading_date", sort=True)["dominant_id"].last()
    if daily.empty:
        raise ValueError("股指持仓分片没有主力合约代码")
    rows = [{
        "trading_date": daily.index[0],
        "new_contract": daily.iloc[0],
        "prev_contract": daily.iloc[0],
    }]
    previous = daily.iloc[0]
    for day, contract in daily.iloc[1:].items():
        if contract != previous:
            rows.append({
                "trading_date": day,
                "new_contract": contract,
                "prev_contract": previous,
            })
            previous = contract
    out = pd.DataFrame(rows)
    out["trading_date"] = pd.to_datetime(out["trading_date"])
    return out


def _oi_dir(price_dir):
    return price_dir / "open_interest"


def _filter_dates(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    dates = pd.to_datetime(frame["trading_date"]).dt.normalize()
    kept = frame.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    kept["trading_date"] = dates.loc[kept.index]
    return kept.sort_index(kind="mergesort")


def _read_holdout(symbol: str, start, end, columns: list[str], kind: str) -> pd.DataFrame:
    directory = C.INDEX_HOLDOUT_DIR if kind == "price" else _oi_dir(C.INDEX_HOLDOUT_DIR)
    path = find_shard(directory, symbol)
    if path is None:
        raise FileNotFoundError(f"找不到 {symbol} 的股指锁定分片: {directory}")
    frame = pd.read_parquet(
        path, columns=columns,
        filters=[
            ("trading_date", ">=", pd.Timestamp(start)),
            ("trading_date", "<=", pd.Timestamp(end)),
        ],
    )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol} 的股指锁定分片没有时间索引")
    if "trading_date" not in frame.columns:
        raise KeyError(f"{symbol} 的股指锁定分片缺少 trading_date")
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    frame = _filter_dates(frame, start, end)
    if frame.empty:
        raise ValueError(f"{symbol} 在 {start} 至 {end} 没有股指锁定行情")
    return frame


def read_minutes(symbol: str, start, end) -> pd.DataFrame:
    """按窗口读分钟。研究期与锁定分区不在一次调用里跨过去。"""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= pd.Timestamp(C.RESEARCH_END):
        frame = load_shard(symbol, directory=C.INDEX_RESEARCH_DIR, columns=MINUTE_COLUMNS)
        frame = _filter_dates(frame, start, end)
        assert_day_session(frame.index, symbol)
        C.assert_no_holdout_dates(frame["trading_date"], f"{symbol} 股指研究分钟")
        return frame
    if pd.Timestamp(C.HOLDOUT_START) <= start and end <= pd.Timestamp(C.VALIDATION_END):
        frame = _read_holdout(symbol, start, end, MINUTE_COLUMNS, "price")
        assert_day_session(frame.index, symbol)
        C.assert_validation_2022_dates(frame["trading_date"], f"{symbol} 股指验证分钟")
        return frame
    if start >= pd.Timestamp(C.STRICT_OOS_START):
        C.assert_oos_research_locked(end.date())
        frame = _read_holdout(symbol, start, end, MINUTE_COLUMNS, "price")
        assert_day_session(frame.index, symbol)
        C.assert_strict_oos_dates(frame["trading_date"], f"{symbol} 股指样本外分钟")
        if frame["trading_date"].max() > pd.Timestamp(end):
            raise C.HoldoutViolation(f"{symbol} 股指样本外分钟越过 {end.date()}")
        return frame
    raise ValueError("股指分钟不能跨研究期和锁定分区一次读出")


def read_oi(symbol: str, start, end) -> pd.DataFrame:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= pd.Timestamp(C.RESEARCH_END):
        frame = load_shard(symbol, directory=_oi_dir(C.INDEX_RESEARCH_DIR), columns=OI_COLUMNS)
        frame = _filter_dates(frame, start, end)
        C.assert_no_holdout_dates(frame["trading_date"], f"{symbol} 股指研究主力")
        return frame
    if pd.Timestamp(C.HOLDOUT_START) <= start and end <= pd.Timestamp(C.VALIDATION_END):
        frame = _read_holdout(symbol, start, end, OI_COLUMNS, "oi")
        C.assert_validation_2022_dates(frame["trading_date"], f"{symbol} 股指验证主力")
        return frame
    if start >= pd.Timestamp(C.STRICT_OOS_START):
        C.assert_oos_research_locked(end.date())
        frame = _read_holdout(symbol, start, end, OI_COLUMNS, "oi")
        C.assert_strict_oos_dates(frame["trading_date"], f"{symbol} 股指样本外主力")
        return frame
    raise ValueError("股指主力合约不能跨研究期和锁定分区一次读出")


def _clock(minute: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    located = add_intraday_coords(minute)
    return timestamp_factors(located), daily_bars(located)


def _research_symbols() -> list[str]:
    missing = [symbol for symbol in RESEARCH_SYMBOLS if find_shard(C.INDEX_RESEARCH_DIR, symbol) is None]
    if missing:
        raise FileNotFoundError(f"股指研究分片缺少 {missing}")
    return list(RESEARCH_SYMBOLS)


def _panel(symbols, minutes_by_symbol):
    clocks, bars = {}, {}
    for symbol in symbols:
        clock, day = _clock(minutes_by_symbol[symbol])
        clocks[symbol] = clock
        bars[symbol] = day
    high = pd.DataFrame({symbol: clocks[symbol]["ts_high"] for symbol in symbols}).sort_index()
    low = pd.DataFrame({symbol: clocks[symbol]["ts_low"] for symbol in symbols}).sort_index()
    fields = {}
    for field in ("open", "openw", "close", "closew", "highw", "loww"):
        fields[field] = pd.DataFrame({
            symbol: bars[symbol][field] for symbol in symbols if field in bars[symbol].columns
        }).sort_index()
    return high, low, fields


def _keep_last(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _ticks(minutes_by_symbol) -> pd.DataFrame:
    parts = []
    for symbol, minute in minutes_by_symbol.items():
        estimated = estimate_ticks(minute["close"], minute["trading_date"])
        if estimated.empty:
            continue
        estimated.insert(0, "symbol", symbol)
        parts.append(estimated)
    if not parts:
        raise ValueError("股指分钟不足以估计最小变动价位")
    return pd.concat(parts, ignore_index=True)


def _require_fee_file(partition: str) -> None:
    path = index_fee_path(partition)
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少股指手续费缓存 {path}。"
            f"请先运行 python download/commodity/getFeeHistory.py --market index --partition {partition}"
        )


def require_oos_inputs() -> None:
    """只检查文件在不在，不读取锁定行情。"""
    for partition in ("research", "validation_2022", "oos"):
        _require_fee_file(partition)
    missing = [symbol for symbol in RESEARCH_SYMBOLS if find_shard(C.INDEX_HOLDOUT_DIR, symbol) is None]
    if missing:
        raise FileNotFoundError(f"股指锁定分片缺少 {missing}")


def _study(bars, high, low, returns, universe, rolls, calendar, fees, ticks, n_ticks):
    from backtest.engine import StudyData

    index = returns.index
    symbols = list(returns.columns)
    prices = bars["open"].reindex(index).reindex(columns=symbols)
    opened, closed, roll_closed = fee_tables(fees, prices)
    slip = slippage_tables(ticks, prices, n_ticks)
    return StudyData(
        SignalSet(bars=bars, signed={"ts_high": -high, "ts_low": low}),
        returns.reindex(columns=symbols),
        universe,
        rolls.reindex(index=index, columns=symbols).fillna(False),
        opened, closed, roll_closed, slip,
        roll_calendar=calendar,
    )


def _returns(open_px, open_w, start, end):
    returns = (open_w.shift(-1) - open_w) / open_px
    returns = returns.loc[
        (returns.index >= pd.Timestamp(start))
        & (returns.index <= pd.Timestamp(end))
        & returns.notna().any(axis=1)
    ]
    return returns


def load_research(n_ticks=1.0):
    """2016–2021 计分，2015 只做预热。不读锁定目录。"""
    symbols = _research_symbols()
    _require_fee_file("research")
    minutes, calendars = {}, {}
    for symbol in symbols:
        minute = read_minutes(symbol, "2015-01-01", C.RESEARCH_END)
        oi = read_oi(symbol, "2015-01-01", C.RESEARCH_END)
        minutes[symbol] = minute
        calendars[symbol] = dominant_calendar(oi)
    stats = collect_stats(symbols, directory=C.INDEX_RESEARCH_DIR)
    universe, _detail = build_universe(stats, years=list(range(2016, 2022)))
    universe = {year: names for year, names in universe.items() if names}
    if not universe:
        raise ValueError("股指研究期流动性池是空的")
    high, low, bars = _panel(symbols, minutes)
    returns = _returns(bars["open"], bars["openw"], C.STUDY_START, C.RESEARCH_END)
    C.assert_no_holdout_dates(returns.index, "股指研究收益")
    rolls = pd.DataFrame(False, index=returns.index, columns=symbols)
    for symbol, calendar in calendars.items():
        changed = calendar.loc[calendar.new_contract.ne(calendar.prev_contract), "trading_date"]
        rolls.loc[rolls.index.intersection(pd.DatetimeIndex(changed)), symbol] = True
    fees = load_index_fees("research")
    return _study(bars, high, low, returns, universe, rolls, calendars, fees, _ticks(minutes), n_ticks)


def _screened(stats, year: int) -> list[str]:
    detail = screen_year(stats, year)
    if detail.empty:
        return []
    return sorted(detail.index[detail["passed"]].tolist())


def load_validation(n_ticks=1.0):
    """研究期接到 2022。先确认手续费文件，再读锁定目录里的 2022。"""
    _require_fee_file("research")
    _require_fee_file("validation_2022")
    research = load_research(n_ticks)
    stats = collect_stats(list(RESEARCH_SYMBOLS), directory=C.INDEX_RESEARCH_DIR)
    names = [
        symbol for symbol in _screened(stats, 2022)
        if find_shard(C.INDEX_HOLDOUT_DIR, symbol) is not None
    ]
    if not names:
        raise ValueError("2022 股指池没有可用分片")
    minutes = {
        symbol: read_minutes(symbol, C.HOLDOUT_START, C.VALIDATION_END) for symbol in names
    }
    calendars = {}
    for symbol in names:
        oi = pd.concat([
            read_oi(symbol, "2015-01-01", C.RESEARCH_END),
            read_oi(symbol, C.HOLDOUT_START, C.VALIDATION_END),
        ], ignore_index=True)
        calendars[symbol] = dominant_calendar(oi)
    high, low, bars = _panel(names, minutes)
    research_high = -research.factors.signed["ts_high"]
    research_low = research.factors.signed["ts_low"]
    high = _keep_last([research_high.reindex(columns=names), high])
    low = _keep_last([research_low.reindex(columns=names), low])
    stitched = {
        field: _keep_last([research.factors.bars[field].reindex(columns=names), bars[field]])
        for field in bars
    }
    returns = _returns(stitched["open"], stitched["openw"], C.STUDY_START, C.VALIDATION_END)
    if returns.empty or returns.index.max() >= pd.Timestamp(C.STRICT_OOS_START):
        raise C.HoldoutViolation("股指验证收益越过 2022")
    symbols = list(returns.columns)
    rolls = pd.DataFrame(False, index=returns.index, columns=symbols)
    for symbol, calendar in calendars.items():
        if symbol not in rolls.columns:
            continue
        changed = calendar.loc[calendar.new_contract.ne(calendar.prev_contract), "trading_date"]
        rolls.loc[rolls.index.intersection(pd.DatetimeIndex(changed)), symbol] = True
    universe = {int(year): list(pool) for year, pool in research.universe.items()}
    universe[2022] = names
    fees = pd.concat([
        load_index_fees("research"), load_index_fees("validation_2022"),
    ], ignore_index=True)
    scored = returns.loc[returns.index.year == 2022]
    C.assert_validation_2022_dates(scored.index, "股指验证收益")
    data = _study(
        stitched, high, low, scored, {2022: names}, rolls, calendars, fees,
        _ticks({
            symbol: read_minutes(symbol, "2015-01-01", C.RESEARCH_END) for symbol in names
        }),
        n_ticks,
    )
    return data, (low - high).reindex(columns=symbols)


def load_oos(end=None, n_ticks=1.0):
    """只在最终评估打开后读取。收益只留 2023 至截止日，更早的 bar 只做预热。"""
    end = C.to_date(end or C.final_evaluation_end())
    C.assert_oos_research_locked(end)
    symbols = [symbol for symbol in SYMBOLS if find_shard(C.INDEX_HOLDOUT_DIR, symbol) is not None]
    if not symbols:
        raise ValueError("股指样本外没有锁定分片")
    minutes = {}
    for symbol in symbols:
        parts = []
        if find_shard(C.INDEX_RESEARCH_DIR, symbol) is not None:
            parts.append(read_minutes(symbol, "2015-01-01", C.RESEARCH_END))
        parts.append(read_minutes(symbol, C.HOLDOUT_START, C.VALIDATION_END))
        parts.append(read_minutes(symbol, C.STRICT_OOS_START, end))
        minute = pd.concat(parts).sort_index(kind="mergesort")
        if minute.index.has_duplicates:
            raise ValueError(f"{symbol} 股指分区有重复时间戳")
        minutes[symbol] = minute
    high, low, bars = _panel(symbols, minutes)
    returns = _returns(bars["open"], bars["openw"], C.STRICT_OOS_START, end)
    if returns.empty:
        raise ValueError(f"股指样本外窗口没有可计分的收益: {C.STRICT_OOS_START} 至 {end}")
    C.assert_strict_oos_dates(returns.index, "股指样本外收益")
    per_symbol = {}
    for symbol, minute in minutes.items():
        per_symbol[symbol] = daily_stats(minute)
    stats = stats_from_daily(per_symbol)
    universe = {}
    for year in range(C.STRICT_OOS_START.year, end.year + 1):
        names = [symbol for symbol in _screened(stats, year) if symbol in symbols]
        if names:
            universe[year] = names
    if not universe:
        raise ValueError("股指样本外流动性池是空的")
    calendars = {}
    rolls = pd.DataFrame(False, index=returns.index, columns=symbols)
    for symbol in symbols:
        oi_parts = []
        if find_shard(_oi_dir(C.INDEX_RESEARCH_DIR), symbol) is not None:
            oi_parts.append(read_oi(symbol, "2015-01-01", C.RESEARCH_END))
        oi_parts.append(read_oi(symbol, C.HOLDOUT_START, C.VALIDATION_END))
        oi_parts.append(read_oi(symbol, C.STRICT_OOS_START, end))
        calendar = dominant_calendar(pd.concat(oi_parts, ignore_index=True))
        calendars[symbol] = calendar
        changed = calendar.loc[calendar.new_contract.ne(calendar.prev_contract), "trading_date"]
        if symbol in rolls.columns:
            rolls.loc[rolls.index.intersection(pd.DatetimeIndex(changed)), symbol] = True
    fees = pd.concat([
        load_index_fees("research"),
        load_index_fees("validation_2022"),
        load_index_fees("oos"),
    ], ignore_index=True)
    data = _study(
        bars, high, low, returns, universe, rolls, calendars, fees,
        _ticks(minutes), n_ticks,
    )
    return data, (low - high).reindex(columns=returns.columns)
