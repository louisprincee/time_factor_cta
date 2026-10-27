"""下载 IF、IH、IC 对应现货指数（沪深300、上证50、中证500）研究期日线。

    python -m download.index.getIndexSpot

只请求 2015-04-16 至 2021-12-31，不读 2022 及以后。写到 data/index/external_rqdata/spot_daily.csv，
供 research.index_liquidity_mechanism 算基差。
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


def main() -> int:
    C.assert_no_holdout_dates(pd.DatetimeIndex([pd.Timestamp(END)]), "现货指数下载截止日")
    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证；请检查 config/rqdata.env")
    import rqdatac as rq

    rq.init(*credentials)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
