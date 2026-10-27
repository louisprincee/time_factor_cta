"""用合约日线自算 IF、IH、IC 研究期主力对次主力年化展期收益（米筐 get_roll_yield 对股指返回空）。

    python -m download.index.getIndexRollYield

口径与米筐商品 main_sub 一致，2026-10-26 在 RB、CU、M 2018–2019 共 1461 天上逐日核对：
主力 = get_dominant(rule=0)；次主力 = 到期晚于主力的合约里当天持仓量最大的一张（与米筐 to_contract 1461/1461 一致）；
yield = (主力结算价 − 次主力结算价) / 次主力结算价；annualized_yield = yield × 365 / |两张到期日相差天数|（复现误差 < 2e-15）。
只请求 2015-01-01 至 2021-12-31，不读 2022 及以后。结果写成与商品同格式的 pickle，供 research.index_market_clock 读取。
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

INDEX_SYMBOLS = ("IF", "IH", "IC")
START = date(2015, 1, 1)
END = C.RESEARCH_END
OUT = C.INDEX_DATA_ROOT / "external_rqdata"


def main_sub_carry(daily: pd.DataFrame, maturity: pd.Series, main: pd.Series) -> pd.DataFrame:
    """daily：date × contract 的结算价和持仓（长表，列 date、contract、settlement、open_interest）；
    maturity：合约 → 到期日；main：date → 主力合约。返回逐日 from/to 合约、yield 和 annualized_yield。"""
    rows = []
    for day, g in daily.groupby("date", sort=True):
        g = g.set_index("contract")
        head = main.get(day)
        if not isinstance(head, str) or head not in g.index or head not in maturity.index:
            continue
        later = g[g.index.map(lambda c: c in maturity.index and maturity[c] > maturity[head])]
        later = later[later["open_interest"].notna() & later["settlement"].notna()]
        if later.empty or not pd.notna(g.loc[head, "settlement"]):
            continue
        sub = later["open_interest"].idxmax()
        main_px, sub_px = float(g.loc[head, "settlement"]), float(g.loc[sub, "settlement"])
        gap = (maturity[sub] - maturity[head]).days
        if sub_px <= 0 or gap == 0:
            continue
        y = (main_px - sub_px) / sub_px
        rows.append({"date": pd.Timestamp(day), "from_contract": head, "to_contract": sub,
                     "yield": y, "annualized_yield": y * 365 / abs(gap), "gap_days": gap})
    return pd.DataFrame(rows)


def main() -> int:
    C.assert_no_holdout_dates(pd.DatetimeIndex([pd.Timestamp(END)]), "股指展期收益截止日")
    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证；请检查 config/rqdata.env")
    import rqdatac as rq

    rq.init(*credentials)
    ins = rq.all_instruments(type="Future")
    ins = ins[ins["underlying_symbol"].isin(INDEX_SYMBOLS) & ins["order_book_id"].str.match(r"^I[FHC]\d{4}$")].copy()
    ins["maturity_date"] = pd.to_datetime(ins["maturity_date"])
    ins["listed_date"] = pd.to_datetime(ins["listed_date"])
    ins = ins[(ins["maturity_date"] >= pd.Timestamp(START)) & (ins["listed_date"] <= pd.Timestamp(END))]
    (OUT / "roll_yield").mkdir(parents=True, exist_ok=True)
    ins[["order_book_id", "underlying_symbol", "listed_date", "maturity_date", "contract_multiplier"]].to_csv(
        OUT / "contract_maturity.csv", index=False, encoding="utf-8-sig")
    maturity = ins.set_index("order_book_id")["maturity_date"]
    for symbol in INDEX_SYMBOLS:
        contracts = ins.loc[ins["underlying_symbol"] == symbol, "order_book_id"].tolist()
        px = rq.get_price(contracts, start_date=START, end_date=END, frequency="1d",
                          fields=["settlement", "open_interest"]).reset_index()
        px = px.rename(columns={"order_book_id": "contract"})
        px["date"] = pd.to_datetime(px["date"])
        C.assert_no_holdout_dates(px["date"], f"{symbol} 合约日线")
        dom = rq.futures.get_dominant(symbol, START, END, rule=0)
        dom.index = pd.to_datetime(dom.index)
        carry = main_sub_carry(px, maturity, dom)
        if carry.empty:
            raise RuntimeError(f"{symbol} 没有算出展期收益")
        carry.insert(0, "underlying_symbol", symbol)
        frame = carry.set_index(["underlying_symbol", "date"]).sort_index()
        frame.to_pickle(OUT / "roll_yield" / f"{symbol}_main_sub.pkl")
        px.to_pickle(OUT / f"{symbol}_contract_daily.pkl")
        print(f"{symbol}: {len(frame)} 天，{frame.index.get_level_values('date').min().date()}.."
              f"{frame.index.get_level_values('date').max().date()}，倒挂 {int((frame['gap_days'] < 0).sum())} 天",
              flush=True)
    print(f"研究期展期收益保存在 {OUT / 'roll_yield'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
