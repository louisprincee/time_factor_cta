"""按主力合约逐日下载交易所手续费，写成 fee_history.csv。"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfcta import config as C
from tfcta.data import universe as U
from tfcta.research import costs
from getRiceQuantExternalData import rqdata_credentials

FIELDS = ["commission_type", "open_commission", "close_commission", "close_commission_today"]
WINDOWS = {"research": ("20150101", "20211231"), "validation_2022": ("20220101", "20221231"),
           "oos": ("20230101", "20251231")}


def one_symbol(rq, symbol, start, end):
    dominant = rq.futures.get_dominant(symbol, start, end)
    if dominant is None or dominant.empty:
        return None
    parts = []
    for contract, days in dominant.groupby(dominant):
        days = pd.to_datetime(days.index)
        table = rq.futures.get_trading_parameters(contract, days.min(), days.max(), fields=FIELDS)
        if table is None or table.empty:
            continue
        table = table.reset_index()
        table["trading_date"] = pd.to_datetime(table.trading_date)
        parts.append(table[table.trading_date.isin(days)])
    if not parts:
        return None
    out = pd.concat(parts).rename(columns={"order_book_id": "contract"})
    out.insert(0, "symbol", symbol)
    return out.sort_values("trading_date")[["symbol", "trading_date", "contract", *FIELDS]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=list(WINDOWS), default="validation_2022")
    args = parser.parse_args()
    if args.partition == "research":
        names = sorted({n for group in U.load_universe().values() for n in group})
    elif args.partition == "validation_2022":
        names, _ = U.validation_universe()
    else:
        # 样本外品种池要读样本外行情才能定，这里不读：直接下载所有有合约乘数的品种（只是交易所费率）
        names = sorted(costs.MULTIPLIER)
    path = costs.fee_path(args.partition)
    names = [n for n in names if n in costs.MULTIPLIER]
    credentials = rqdata_credentials()
    if credentials is None:
        raise SystemExit("请在 config/rqdata.env 填写米筐账号")
    import rqdatac as rq
    rq.init(*credentials)
    start, end = WINDOWS[args.partition]
    parts = []
    for symbol in names:
        table = one_symbol(rq, symbol, start, end)
        print(symbol, 0 if table is None else len(table), flush=True)
        if table is not None:
            parts.append(table)
    table = pd.concat(parts, ignore_index=True)
    if args.partition == "oos":
        C.assert_strict_oos_dates(table.trading_date, "手续费下载")
        if table.trading_date.max() > pd.Timestamp(end):
            raise ValueError("手续费下载越过窗口")
    else:
        guard = C.assert_no_holdout_dates if args.partition == "research" else C.assert_validation_2022_dates
        guard(table.trading_date, "手续费下载")
    if table.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("手续费有重复品种/日期")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(path, table.shape, table.symbol.nunique())


if __name__ == "__main__":
    main()
