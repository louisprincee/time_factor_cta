"""下载并预处理股指期货分钟行情，支持研究期与独立锁定 OOS 分区。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.data import save_shard  # noqa: E402
from tfcta.rq_auth import rqdata_credentials  # noqa: E402

INDEX_SYMBOLS = ("IF", "IH", "IC", "IM")
RESEARCH_SYMBOLS = ("IF", "IH", "IC")
INDEX_LISTED_FROM = {"IM": date(2022, 7, 22)}
SERIES = ("88", "889", "99")
DEFAULT_START = date(2015, 1, 1)
OOS_START = date(2022, 1, 1)
OOS_END = C.DEFAULT_OOS_END


def normalize_api_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Convert a one-instrument RiceQuant result to a sorted datetime index."""
    if frame is None or frame.empty:
        raise ValueError(f"RiceQuant returned no rows for {label}")
    if isinstance(frame.index, pd.MultiIndex):
        if "datetime" not in frame.index.names:
            raise ValueError(f"{label} has no datetime index level")
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(frame.index.get_level_values("datetime"))
    else:
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.has_duplicates:
        raise ValueError(f"{label} has duplicate minute timestamps")
    return frame.sort_index(kind="mergesort")


def assemble_symbol_frames(main: pd.DataFrame, next_main: pd.DataFrame,
                           aggregate: pd.DataFrame) -> pd.DataFrame:
    """Join 88 raw prices, 889 adjusted prices, and 99 volume/OI by minute."""
    frames = [
        normalize_api_frame(main, "88"),
        normalize_api_frame(next_main, "889"),
        normalize_api_frame(aggregate, "99"),
    ]
    if not all(frames[0].index.equals(frame.index) for frame in frames[1:]):
        raise ValueError("88/889/99 minute indexes differ; refusing an implicit partial join")

    required = {
        "88": {"trading_date", "open", "high", "low", "close", "total_turnover", "dominant_id"},
        "889": {"open", "high", "low", "close"},
        "99": {"volume", "open_interest"},
    }
    for label, frame, columns in zip(SERIES, frames, required.values()):
        missing = sorted(columns - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing required fields: {missing}")

    trading_date = pd.to_datetime(frames[0]["trading_date"]).dt.normalize()
    for frame in frames[1:]:
        other = pd.to_datetime(frame["trading_date"]).dt.normalize()
        if not trading_date.equals(other):
            raise ValueError("88/889/99 trading_date values differ")

    result = frames[0][["open", "high", "low", "close", "total_turnover", "dominant_id"]].copy()
    for source, target in (("open", "openw"), ("high", "highw"),
                           ("low", "loww"), ("close", "closew")):
        result[target] = frames[1][source]
    result["volume"] = frames[2]["volume"]
    result["open_interest"] = frames[2]["open_interest"]
    result["trading_date"] = trading_date
    return result


def split_research_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return factor/price and OI shards, rejecting any validation or OOS date."""
    if "trading_date" not in frame:
        raise KeyError("股指分钟面板缺少 trading_date")
    dates = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(dates, what="股指期货研究分片")
    fields = list(dict.fromkeys([*C.FACTOR_FIELDS, *C.PRICE_FIELDS]))
    missing = sorted(set(fields) - set(frame.columns))
    if missing:
        raise ValueError(f"股指分钟面板缺少分片字段: {missing}")
    prices = frame[fields].copy()
    prices["trading_date"] = dates
    oi = frame[["open_interest", "dominant_id"]].copy()
    oi["trading_date"] = dates
    return prices, oi


def split_oos_frames(frame: pd.DataFrame, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare aligned OOS shards without routing them through research readers."""
    if "trading_date" not in frame:
        raise KeyError("股指分钟面板缺少 trading_date")
    dates = pd.to_datetime(frame["trading_date"]).dt.normalize()
    if dates.empty or dates.min().date() < start or dates.max().date() > end:
        raise ValueError("RiceQuant 返回日期超出请求窗口")
    if dates.min().date() < OOS_START or dates.max().date() > OOS_END:
        raise C.HoldoutViolation("股指锁定分片日期超出登记的 OOS 窗口")
    fields = list(dict.fromkeys([*C.FACTOR_FIELDS, *C.PRICE_FIELDS]))
    missing = sorted(set(fields) - set(frame.columns))
    if missing:
        raise ValueError(f"股指分钟面板缺少分片字段: {missing}")
    prices = frame[fields].copy()
    prices["trading_date"] = dates
    oi = frame[["open_interest", "dominant_id"]].copy()
    oi["trading_date"] = dates
    return prices, oi


def download_symbol(rq, symbol: str, start: date, end: date) -> pd.DataFrame:
    fetched = {}
    for suffix in SERIES:
        fetched[suffix] = rq.get_price(
            f"{symbol}{suffix}", start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"), frequency="1m")
    frame = assemble_symbol_frames(fetched["88"], fetched["889"], fetched["99"])
    dates = pd.to_datetime(frame["trading_date"])
    frame = frame.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if frame.empty:
        raise ValueError(f"{symbol} has no rows in {start}..{end}")
    C.assert_no_holdout_dates(frame["trading_date"], what=f"{symbol} 股指期货研究分片")
    return frame


def download_oos_symbol(rq, symbol: str, start: date, end: date) -> pd.DataFrame:
    fetched = {}
    for suffix in SERIES:
        fetched[suffix] = rq.get_price(
            f"{symbol}{suffix}", start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"), frequency="1m")
    frame = assemble_symbol_frames(fetched["88"], fetched["889"], fetched["99"])
    dates = pd.to_datetime(frame["trading_date"]).dt.normalize()
    frame = frame.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if frame.empty:
        raise ValueError(f"No rows returned for requested locked symbol {symbol}")
    split_oos_frames(frame, start, end)
    return frame


def effective_start(symbol: str, start: date) -> date:
    return max(start, INDEX_LISTED_FROM.get(symbol, start))


def save_oos_locked(rq, symbols, start, end, root: Path) -> None:
    locked_dir = root / "minute_shards" / "holdout_locked"
    prices_dir, oi_dir = locked_dir, locked_dir / "open_interest"
    manifest_path = locked_dir / "oos_manifest.json"
    completed = []

    for symbol in symbols:
        symbol_start = effective_start(symbol, start)
        if symbol_start > end:
            raise ValueError(f"Requested OOS window predates {symbol} listing")
        price_path, oi_path = prices_dir / f"{symbol}.parquet", oi_dir / f"{symbol}.parquet"
        if price_path.exists() and oi_path.exists():
            completed.append(symbol)
            continue
        if price_path.exists() or oi_path.exists():
            raise FileExistsError(f"Incomplete locked shard pair for {symbol}; refusing overwrite")

        frame = download_oos_symbol(rq, symbol, symbol_start, end)
        prices, oi = split_oos_frames(frame, symbol_start, end)
        stage = locked_dir / f".staging_{symbol}"
        if stage.exists():
            raise FileExistsError(f"Locked staging path already exists: {stage}")
        stage_prices, stage_oi = stage, stage / "open_interest"
        save_shard(prices, stage_prices, symbol)
        save_shard(oi, stage_oi, symbol)

        locked_dir.mkdir(parents=True, exist_ok=True)
        oi_dir.mkdir(parents=True, exist_ok=True)
        (stage_prices / f"{symbol}.parquet").replace(price_path)
        (stage_oi / f"{symbol}.parquet").replace(oi_path)
        stage_oi.rmdir()
        stage.rmdir()
        completed.append(symbol)
        print(f"已写入锁定分区: {symbol}", flush=True)

        manifest = {
            "source": "RiceQuant",
            "partition": "holdout_locked",
            "requested_start": str(start),
            "requested_end": str(end),
            "symbols": sorted(completed),
            "im_listed_from": str(INDEX_LISTED_FROM["IM"]),
            "content_not_loaded_for_analysis": True,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    locked_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "RiceQuant",
        "partition": "holdout_locked",
        "requested_start": str(start),
        "requested_end": str(end),
        "symbols": sorted(completed),
        "im_listed_from": str(INDEX_LISTED_FROM["IM"]),
        "content_not_loaded_for_analysis": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("research", "oos_locked"), default="research")
    parser.add_argument("--symbols", nargs="+", choices=INDEX_SYMBOLS, default=None)
    parser.add_argument("--start-date", default=None, help="YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="YYYYMMDD")
    parser.add_argument("--datapath", default=str(C.INDEX_DATA_ROOT),
                        help="独立股指数据根目录")
    parser.add_argument("--force", action="store_true", help="仅研究分片允许显式覆盖")
    args = parser.parse_args()
    symbols = args.symbols or (list(RESEARCH_SYMBOLS) if args.partition == "research" else list(INDEX_SYMBOLS))
    default_start = DEFAULT_START if args.partition == "research" else OOS_START
    default_end = C.RESEARCH_END if args.partition == "research" else OOS_END
    try:
        start = pd.Timestamp(args.start_date).date() if args.start_date else default_start
        end = pd.Timestamp(args.end_date).date() if args.end_date else default_end
    except (TypeError, ValueError):
        parser.error("日期必须是有效的 YYYYMMDD")
    if start > end:
        parser.error("开始日期不能晚于结束日期")
    if args.partition == "research" and set(symbols) - set(RESEARCH_SYMBOLS):
        parser.error("研究期只有 IF/IH/IC；IM 在 2022 年上市，只能进入锁定 OOS 分区")
    if args.partition == "research" and end > C.RESEARCH_END:
        parser.error(f"本入口只处理研究期，结束日期不得晚于 {C.RESEARCH_END}")
    if args.partition == "oos_locked" and (start < OOS_START or end > OOS_END):
        parser.error(f"锁定 OOS 日期必须落在 {OOS_START}..{OOS_END}")
    if args.partition == "oos_locked" and args.force:
        parser.error("锁定 OOS 分片不允许覆盖")

    root = Path(args.datapath)
    if args.partition == "oos_locked":
        credentials = rqdata_credentials()
        if credentials is None:
            parser.error("请在 config/rqdata.env 配置 RiceQuant 凭证")
        import rqdatac as rq

        rq.init(*credentials)
        save_oos_locked(rq, symbols, start, end, root)
        print("OOS 下载完成，数据保存在锁定目录；未读取数据内容进行分析。", flush=True)
        return 0

    minute_dir = root / "minute_shards" / "research"
    oi_dir = minute_dir / "open_interest"
    targets = [path for symbol in symbols
               for path in (minute_dir / f"{symbol}.parquet", oi_dir / f"{symbol}.parquet")]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        parser.error(f"目标分片已存在，默认拒绝覆盖；需要显式 --force：{existing}")

    credentials = rqdata_credentials()
    if credentials is None:
        parser.error("请在 config/rqdata.env 配置 RiceQuant 凭证")
    import rqdatac as rq

    rq.init(*credentials)
    manifest = {
        "source": "RiceQuant",
        "sample": "research_only",
        "start_date": str(start),
        "end_date": str(end),
        "symbols": {},
    }
    for symbol in symbols:
        print(f"下载 {symbol} {start}..{end}", flush=True)
        frame = download_symbol(rq, symbol, start, end)
        prices, oi = split_research_frames(frame)
        save_shard(prices, minute_dir, symbol)
        save_shard(oi, oi_dir, symbol)
        manifest["symbols"][symbol] = {
            "bars": len(frame),
            "trading_days": int(frame.trading_date.nunique()),
            "first_trading_date": str(frame.trading_date.min().date()),
            "last_trading_date": str(frame.trading_date.max().date()),
            "price_fields": list(prices.columns),
            "oi_fields": list(oi.columns),
        }
        print(symbol, manifest["symbols"][symbol], flush=True)

    root.mkdir(parents=True, exist_ok=True)
    (root / "research_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"股指研究期分片已写入 {minute_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())