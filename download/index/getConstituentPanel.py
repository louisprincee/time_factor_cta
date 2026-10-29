"""下载上证50、沪深300和中证500成分股的研究期日频面板，用于检验股指收盘位置反转里“谁在卖”。

    python -m download.index.getConstituentPanel

只请求 2015-04-01 至 2021-12-31，不读 2022 及以后。写到 data/index/external_rqdata/constituents/：
- membership.csv：每月第一个交易日三个指数的成分股（成分股按月沿用到下一次取样）；
- prices.parquet：不复权开高低收、涨跌停价、成交额（日内收益和收盘位置不受复权影响）；
- margin.parquet：个股融资余额、融资买入、融资偿还；
- shares.parquet：流通 A 股股数；
- northbound.parquet：北向持股数和持股比例（港交所 2017-03-17 起每日披露）。
可续跑：已存在的文件跳过。
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

INDEXES = {"SSE50": "000016.XSHG", "CSI300": "000300.XSHG", "CSI500": "000905.XSHG"}
START, END = date(2015, 4, 1), C.RESEARCH_END
OUT = C.INDEX_DATA_ROOT / "external_rqdata" / "constituents"
CHUNK = 100
PRICE_FIELDS = ["open", "high", "low", "close", "limit_up", "limit_down", "total_turnover"]


def _clip(frame: pd.DataFrame, date_level: str) -> pd.DataFrame:
    frame = frame.reset_index()
    frame[date_level] = pd.to_datetime(frame[date_level]).dt.normalize()
    C.assert_no_holdout_dates(frame[date_level], "成分股面板")
    if frame[date_level].max().date() > END:
        raise ValueError("成分股面板越过 2021-12-31")
    return frame


def membership(rq, indexes: dict[str, str] = INDEXES) -> pd.DataFrame:
    days = pd.DatetimeIndex(rq.get_trading_dates(START, END))
    firsts = days[days.to_period("M") != pd.Series(days.to_period("M")).shift(1).to_numpy()]
    rows = []
    for day in firsts:
        for name, code in indexes.items():
            for stock in rq.index_components(code, day):
                rows.append({"month_start": day, "index": name, "order_book_id": stock})
    out = pd.DataFrame(rows)
    C.assert_no_holdout_dates(out["month_start"], "成分股名单")
    return out


def _chunked(call, stocks: list[str], date_level: str) -> pd.DataFrame:
    parts = []
    for k in range(0, len(stocks), CHUNK):
        part = call(stocks[k:k + CHUNK])
        if part is not None and len(part):
            parts.append(_clip(part, date_level))
        print(f"  {min(k + CHUNK, len(stocks))}/{len(stocks)}", flush=True)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    import rqdatac as rq

    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证；请检查 config/rqdata.env")
    rq.init(*credentials)
    OUT.mkdir(parents=True, exist_ok=True)
    member_path = OUT / "membership.csv"
    if not member_path.is_file():
        membership(rq).to_csv(member_path, index=False, encoding="utf-8-sig")
    saved = pd.read_csv(member_path)
    missing = {k: v for k, v in INDEXES.items() if k not in set(saved["index"])}
    if missing:
        saved = pd.concat([saved, membership(rq, missing)], ignore_index=True)
        saved.to_csv(member_path, index=False, encoding="utf-8-sig")
    stocks = sorted(saved["order_book_id"].unique())
    have = set(pd.read_parquet(OUT / "prices.parquet", columns=["order_book_id"])["order_book_id"])         if (OUT / "prices.parquet").is_file() else set(stocks)
    if set(stocks) - have:
        raise RuntimeError(f"新成分股不在已下载面板里：{sorted(set(stocks) - have)[:10]}；删除 parquet 后重下")
    print(f"成分股并集 {len(stocks)} 只", flush=True)
    jobs = {
        "prices.parquet": (lambda s: rq.get_price(s, start_date=START, end_date=END, frequency="1d",
                                                  fields=PRICE_FIELDS, adjust_type="none"), "date"),
        "margin.parquet": (lambda s: rq.get_securities_margin(
            s, start_date=START, end_date=END,
            fields=["margin_balance", "buy_on_margin_value", "margin_repayment"]), "date"),
        "shares.parquet": (lambda s: rq.get_shares(s, start_date=START, end_date=END, fields=["circulation_a"]), "date"),
        "northbound.parquet": (lambda s: rq.get_stock_connect(s, start_date=date(2017, 3, 17), end_date=END,
                                                              fields=["shares_holding", "holding_ratio"]),
                               "trading_date"),
    }
    for name, (call, level) in jobs.items():
        path = OUT / name
        if path.is_file():
            print(f"{name} 已存在，跳过", flush=True)
            continue
        print(f"下载 {name}", flush=True)
        _chunked(call, stocks, level).to_parquet(path, index=False)
    print(f"完成：{OUT}", flush=True)


if __name__ == "__main__":
    main()
