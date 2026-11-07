"""股指共享L1报价、竞价容量与早期开盘摘要工具；没有策略回测入口。"""
from pathlib import Path
import numpy as np
import pandas as pd
from backtest.engine import INDEX_SYMBOLS
from tfcta import config as C
from tfcta.rq_auth import rqdata_credentials

L1_DIR = C.INDEX_L1_TICKS_DIR
OPENING_CACHE = C.study_dir("index", "index_execution_10m") / "l1_opening_30m.parquet"
BUY_AUCTION_OUT = C.study_dir("index", "index_execution_buy_auction")
ROOT = C.INDEX_DATA_ROOT / "if_before_ih_ic"
OPENING_QUOTES = ROOT / "l1_opening_30m.parquet"
EXEC_START = pd.Timestamp("2011-01-01")
IH_IC_LISTING = pd.Timestamp("2015-04-16")
WINDOW = pd.Timedelta(30, unit="min")
BEFORE_CONTINUOUS = pd.Timedelta(9 * 3600 + 30 * 60, unit="s")
BEFORE_1005 = pd.Timedelta(10 * 3600 + 5 * 60, unit="s")
AUCTION_SHARE, AUCTION_HISTORY, AUCTION_HISTORY_MIN = .10, 20, 10
OPEN_KINDS = {"开仓": 1, "加仓": 1, "开空": -1, "加空": -1}
CLOSE_KINDS = {"平仓", "减仓", "清仓", "平空", "减空"}


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



def auction_snapshot(ticks: pd.DataFrame) -> pd.DataFrame:
    """每个交易日连续交易起点以前最后一笔 tick：开盘价、竞价成交量、竞价后一档买卖价和量。"""
    rows = []
    for day, part in ticks.groupby("trading_date", sort=True):
        day = pd.Timestamp(day)
        before = part[part["datetime"] - day < continuous_start(day)]
        if before.empty:
            continue
        last = before.sort_values("datetime").iloc[-1]
        rows.append({
            "trading_date": day, "contract": last["contract"], "open": float(last["open"]),
            "auction_volume": float(last["volume"]),
            "a1": float(last["a1"]), "a1_v": float(last["a1_v"]),
            "b1": float(last["b1"]), "b1_v": float(last["b1_v"]),
        })
    return pd.DataFrame(rows)



def load_auction_snapshots() -> pd.DataFrame:
    """2015–2021 研究期 L1 的竞价快照，缓存在本检验的输出目录。"""
    cache = BUY_AUCTION_OUT / "auction_snapshot.parquet"
    if cache.is_file():
        frame = pd.read_parquet(cache)
        C.assert_no_holdout_dates(frame["trading_date"], what="竞价快照缓存")
        return frame
    parts = []
    columns = ["datetime", "trading_date", "open", "volume", "a1", "a1_v", "b1", "b1_v", "contract"]
    for symbol in INDEX_SYMBOLS:
        for path in sorted((L1_DIR / symbol).glob("*/*.parquet")):
            C.assert_research_only(path)
            ticks = pd.read_parquet(path, columns=columns)
            ticks["trading_date"] = pd.to_datetime(ticks["trading_date"]).dt.normalize()
            C.assert_no_holdout_dates(ticks["trading_date"], what=f"{symbol} L1")
            ticks = ticks[ticks["datetime"] - ticks["trading_date"] < BEFORE_CONTINUOUS]
            frame = auction_snapshot(ticks)
            frame.insert(1, "symbol", symbol)
            parts.append(frame)
        print(f"{symbol} 竞价快照完成", flush=True)
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(["trading_date", "symbol"]).any():
        raise ValueError("竞价快照日期重复")
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame



def buy_auction_tables(
    snapshots: pd.DataFrame, open_price: pd.DataFrame, contracts: pd.DataFrame,
    tick_rate: pd.DataFrame, scored: pd.DatetimeIndex,
):
    """买入竞价额度和成本率。额度只用执行日以前的竞价量；成本率 = max(一跳, 竞价后卖一相对开盘的冲击下界)。"""
    names = list(open_price.columns)
    pick = lambda field: snapshots.pivot(index="trading_date", columns="symbol", values=field).reindex(columns=names)
    volume = pick("auction_volume").sort_index()
    history = volume.rolling(AUCTION_HISTORY, min_periods=AUCTION_HISTORY_MIN).median().shift(1)
    allowance = np.floor(AUCTION_SHARE * history).reindex(index=scored).fillna(0.0)
    opened = open_price.reindex(index=scored, columns=names)
    a1, a1_v = pick("a1").reindex(scored), pick("a1_v").reindex(scored)
    b1, snap_open = pick("b1").reindex(scored), pick("open").reindex(scored)
    valid = (
        pick("contract").reindex(scored).eq(contracts.reindex(index=scored, columns=names))
        & (snap_open - opened).abs().lt(1e-6)
        & a1.gt(0) & b1.gt(0) & b1.le(opened + 1e-9) & a1.ge(opened - 1e-9)
    )
    allowance = allowance.where(valid, 0.0)
    covered = (a1 - opened).abs().lt(1e-9) & a1_v.ge(allowance)
    impact = ((a1 - opened) / opened).where(~covered, 0.0)
    rate = np.maximum(tick_rate.reindex(index=scored, columns=names), impact).where(valid)
    return allowance.astype("float64"), rate, int((~valid).sum().sum()), impact.where(valid)



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
    morning = frame[frame["datetime"] - frame["trading_date"] < BEFORE_1005].copy()
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
        part = part.sort_values("datetime")
        contracts = part["contract"].dropna().unique()
        if len(contracts) != 1:
            raise ValueError(f"{day.date()} L1 当日不止一个合约：{contracts}")
        offset = part["datetime"] - day
        start = continuous_start(day)
        before = part[offset < start]
        window = part[(offset >= start) & (offset < start + WINDOW)]
        valid = window[(window["a1"] > 0) & (window["b1"] > 0) & (window["a1"] >= window["b1"])]
        auction = float(before["volume"].max()) if not before.empty else 0.0
        opens = part.loc[part["open"] > 0, "open"]
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
    cache = Path(cache_path) if cache_path is not None else OPENING_CACHE
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
            ticks = ticks[ticks["datetime"] - ticks["trading_date"] < BEFORE_1005]
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
