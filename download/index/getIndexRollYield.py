"""下载 IF、IH、IC 研究期主力对次主力年化展期收益。"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.rq_auth import rqdata_credentials  # noqa: E402

INDEX_SYMBOLS = ("IF", "IH", "IC")
START = date(2015, 1, 1)
END = C.RESEARCH_END


def main() -> int:
    C.assert_no_holdout_dates(pd.DatetimeIndex([pd.Timestamp(END)]), "股指展期收益下载截止日")
    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证；请检查 config/rqdata.env")

    import rqdatac as rq
    from download.commodity.getRiceQuantExternalData import CoverageCache

    rq.init(*credentials)
    cache = CoverageCache(C.INDEX_DATA_ROOT / "external_rqdata")
    try:
        for symbol in INDEX_SYMBOLS:
            cache.fetch(
                "roll_yield",
                f"{symbol}_main_sub",
                START,
                END,
                lambda left, right, symbol=symbol: rq.futures.get_roll_yield(
                    symbol, start_date=left, end_date=right, type="main_sub"
                ),
                require_nonempty=True,
            )
    finally:
        cache.flush()
    if cache.failures:
        raise RuntimeError("股指展期收益下载失败：" + "；".join(cache.failures))
    print(f"研究期展期收益保存在 {cache.root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
