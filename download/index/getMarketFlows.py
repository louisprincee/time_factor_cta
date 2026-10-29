"""下载 A 股市场层面的融资融券余额和北向资金成交，用于股指收盘位置反转的机制检验。

    python -m download.index.getMarketFlows

只请求 2015-01-01 至 2021-12-31，不读 2022 及以后。写到 data/index/external_rqdata/market_flows.csv：
沪深两市合计的融资余额、融资买入、融资偿还，以及北向（沪股通 + 深股通）买入、卖出成交额。
融资数据是交易所 T+1 早上公布的 T 日数据；北向成交是 T 日收盘后公布。只做机制检验时可以同日使用，
做交易规则必须至少滞后一天。
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

START, END = date(2015, 1, 1), C.RESEARCH_END
OUT = C.INDEX_DATA_ROOT / "external_rqdata" / "market_flows.csv"


def market_flows(margin: pd.DataFrame, connect: pd.DataFrame) -> pd.DataFrame:
    """两市融资合计和北向合计，按日期一行；越过 2021-12-31 直接拒绝。"""
    margin = margin.reset_index()
    margin["date"] = pd.to_datetime(margin["date"]).dt.normalize()
    summed = margin.groupby("date")[["margin_balance", "buy_on_margin_value", "margin_repayment"]].sum()
    if (margin.groupby("date")["order_book_id"].nunique() != 2).any():
        raise ValueError("融资数据有日期缺沪市或深市")
    connect = connect.reset_index()
    connect["datetime"] = pd.to_datetime(connect["datetime"]).dt.normalize()
    north = connect[connect["connect"].isin(["hk_to_sh", "hk_to_sz"])]
    flows = north.groupby("datetime")[["buy_turnover", "sell_turnover"]].sum()
    flows.columns = ["north_buy", "north_sell"]
    out = summed.join(flows, how="left")
    out.index.name = "date"
    C.assert_no_holdout_dates(out.index, "市场资金流")
    if out.index.min().date() < START or out.index.max().date() > END:
        raise ValueError("市场资金流超出 2015-01-01 至 2021-12-31")
    return out.sort_index()


def main() -> None:
    import rqdatac as rq

    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证；请检查 config/rqdata.env")
    rq.init(*credentials)
    margin = rq.get_securities_margin(["XSHG", "XSHE"], start_date=START, end_date=END)
    connect = rq.get_stock_connect_quota(start_date=START, end_date=END)
    if margin is None or connect is None or margin.empty or connect.empty:
        raise RuntimeError("融资或北向数据为空")
    out = market_flows(margin, connect)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, encoding="utf-8-sig")
    print(f"市场资金流 {out.index.min().date()}..{out.index.max().date()}，{len(out)} 天，"
          f"北向缺失 {int(out['north_buy'].isna().sum())} 天：{OUT}")


if __name__ == "__main__":
    main()
