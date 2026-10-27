"""下载 IF、IH、IC 研究期每日前 20 名会员多头和空头持仓（品种合计，中金所公布口径）。

    python -m download.index.getMemberRank

米筐 futures.get_member_rank(品种, start_date, end_date, rank_by="long"/"short")：品种层面等于各合约之和
（2026-10-27 用 IF 2018-03-15 核对）。只请求 2015-04-16 至 2021-12-31，不读 2022 及以后。
写到 data/index/external_rqdata/member_rank.csv。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.rq_auth import rqdata_credentials  # noqa: E402

SYMBOLS = ("IF", "IH", "IC")
START, END = pd.Timestamp("2015-04-16"), pd.Timestamp(C.RESEARCH_END)
OUT = C.INDEX_DATA_ROOT / "external_rqdata" / "member_rank.csv"


def main() -> int:
    C.assert_no_holdout_dates(pd.DatetimeIndex([END]), "会员持仓下载截止日")
    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证；请检查 config/rqdata.env")
    import rqdatac as rq

    rq.init(*credentials)
    frames = []
    for symbol in SYMBOLS:
        for side in ("long", "short"):
            for year in range(START.year, END.year + 1):
                left = max(START, pd.Timestamp(f"{year}-01-01"))
                right = min(END, pd.Timestamp(f"{year}-12-31"))
                part = rq.futures.get_member_rank(symbol, start_date=left.strftime("%Y%m%d"),
                                                  end_date=right.strftime("%Y%m%d"), rank_by=side)
                if part is None or part.empty:
                    raise RuntimeError(f"{symbol} {side} {year} 没有会员持仓")
                part = part.reset_index()
                part["trading_date"] = pd.to_datetime(part["trading_date"])
                C.assert_no_holdout_dates(part["trading_date"], f"{symbol} 会员持仓")
                frames.append(part.assign(symbol=symbol, side=side))
    out = pd.concat(frames, ignore_index=True)[
        ["symbol", "side", "trading_date", "rank", "member_name", "volume", "volume_change"]]
    if out.duplicated(["symbol", "side", "trading_date", "rank"]).any():
        raise ValueError("会员持仓出现重复的品种-方向-日期-名次")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    days = out.groupby(["symbol", "side"])["trading_date"].nunique()
    print(OUT, len(out), days.to_dict(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
