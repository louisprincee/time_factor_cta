"""下载 IF、IH、IC 对应现货指数（沪深300、上证50、中证500）研究期行情。

    python -m download.index.getIndexSpot            # 日线
    python -m download.index.getIndexSpot --minute   # 1 分钟线

只请求 2015-04-16 至 2021-12-31，不读 2022 及以后。日线写到
data/index/external_rqdata/spot_daily.csv，分钟线写到同目录 spot_minute.parquet。
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.rq_auth import rqdata_credentials  # noqa: E402

SPOT = {"IF": "000300.XSHG", "IH": "000016.XSHG", "IC": "000905.XSHG"}
START, END = date(2015, 4, 16), C.RESEARCH_END
OUT = C.INDEX_DATA_ROOT / "external_rqdata" / "spot_daily.csv"
MINUTE_OUT = C.INDEX_DATA_ROOT / "external_rqdata" / "spot_minute.parquet"
MINUTE_FIELDS = ("open", "high", "low", "close", "volume")


def normalize_spot_minutes(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """整理米筐分钟线。时间戳越过 2021-12-31 时直接拒绝，不写文件。"""
    frame = raw.copy()
    if isinstance(frame.index, pd.MultiIndex) or isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()
    if "datetime" not in frame.columns:
        raise KeyError(f"{symbol} 现货分钟缺少 datetime")
    missing = [name for name in MINUTE_FIELDS if name not in frame.columns]
    if missing:
        raise KeyError(f"{symbol} 现货分钟缺少字段：{missing}")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    C.assert_no_holdout_dates(frame["datetime"], f"{symbol} 现货分钟")
    out = frame[["datetime", *MINUTE_FIELDS]].copy()
    out.insert(0, "symbol", symbol)
    out = out.dropna(subset=["datetime", "open", "close", "volume"])
    if out.empty:
        raise ValueError(f"{symbol} 现货分钟为空")
    if out["datetime"].duplicated().any():
        raise ValueError(f"{symbol} 现货分钟时间戳重复")
    if out["datetime"].dt.date.min() < START or out["datetime"].dt.date.max() > END:
        raise ValueError(f"{symbol} 现货分钟超出 2015-04-16 至 2021-12-31")
    return out.sort_values("datetime")


def _fetch_year(rq, code: str, start: date, end: date) -> pd.DataFrame:
    px = rq.get_price(code, start_date=start, end_date=end, frequency="1m", fields=list(MINUTE_FIELDS))
    if px is None or len(px) == 0:
        raise RuntimeError(f"{code} {start}..{end} 没有分钟线")
    return px


def download_minutes(rq) -> pd.DataFrame:
    """按年拉取，避免一次请求整段研究期。三个指数都成功才返回。"""
    frames = []
    for symbol, code in SPOT.items():
        parts = []
        for year in range(START.year, END.year + 1):
            lo = max(START, date(year, 1, 1))
            hi = min(END, date(year, 12, 31))
            parts.append(_fetch_year(rq, code, lo, hi))
            print(f"{symbol} {lo}..{hi} 已取回", flush=True)
        frames.append(normalize_spot_minutes(pd.concat(parts), symbol))
    out = pd.concat(frames, ignore_index=True)
    C.assert_no_holdout_dates(out["datetime"], "现货指数分钟")
    return out


def _credentials_and_rq():
    C.assert_no_holdout_dates(pd.DatetimeIndex([pd.Timestamp(END)]), "现货指数下载截止日")
    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证；请检查 config/rqdata.env")
    import rqdatac as rq

    rq.init(*credentials)
    return rq


def download_daily(rq) -> None:
    frames = []
    for symbol, code in SPOT.items():
        px = rq.get_price(code, start_date=START, end_date=END, frequency="1d",
                          fields=["open", "high", "low", "close"])
        if px is None or px.empty:
            raise RuntimeError(f"{code} 没有日线")
        px = px.reset_index()
        px["date"] = pd.to_datetime(px["date"])
        C.assert_no_holdout_dates(px["date"], f"{code} 日线")
        px.insert(0, "symbol", symbol)
        frames.append(px[["symbol", "order_book_id", "date", "open", "high", "low", "close"]])
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(OUT, len(out), out.groupby("symbol")["date"].agg(["min", "max", "count"]).to_dict("index"), flush=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="下载沪深300、上证50、中证500研究期行情")
    parser.add_argument("--minute", action="store_true", help="1 分钟线，2015-04-16 至 2021-12-31")
    args = parser.parse_args(argv)
    rq = _credentials_and_rq()
    if args.minute:
        out = download_minutes(rq)
        MINUTE_OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(MINUTE_OUT, index=False)
        print(MINUTE_OUT, len(out), out.groupby("symbol")["datetime"].agg(["min", "max", "count"]).to_dict("index"),
              flush=True)
        return 0
    download_daily(rq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
