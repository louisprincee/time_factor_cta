"""按主力合约逐日下载交易所手续费，写成 fee_history.csv。"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tfcta import config as C
from tfcta.data import load_universe, validation_universe
from tfcta.costs import INDEX_MULTIPLIER, MULTIPLIER, fee_path, index_fee_path, require_multipliers
from tfcta.rq_auth import rqdata_credentials

FIELDS = ["commission_type", "open_commission", "close_commission", "close_commission_today"]
# 研究分区从 2013-10 起：米筐历史手续费大多从 2013-10 或 2014 年才有，2014–2015 熊市压力窗口要用
WINDOWS = {"research": ("20131001", "20211231"), "validation_2022": ("20220101", "20221231"),
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
    out = out.dropna(subset=["commission_type", "open_commission", "close_commission"])
    if out.empty:
        return None
    out.insert(0, "symbol", symbol)
    return out.sort_values("trading_date")[["symbol", "trading_date", "contract", *FIELDS]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("commodity", "index"), default="commodity")
    parser.add_argument("--partition", choices=list(WINDOWS), default="validation_2022")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="只下载这些品种。已有文件里，这次没下到的品种会保留")
    parser.add_argument("--start-date", default=None, help="YYYYMMDD，不早于分区起点；只替换这段日期")
    parser.add_argument("--end-date", default=None, help="YYYYMMDD，不晚于分区终点；只替换这段日期")
    args = parser.parse_args()
    if args.symbols:
        names = sorted({s.strip().upper() for s in args.symbols if s.strip()})
    elif args.market == "index" and args.partition == "research":
        names = ["IF", "IH", "IC"]
    elif args.market == "index":
        names = sorted(INDEX_MULTIPLIER)
    elif args.partition == "research":
        names = sorted({n for group in load_universe().values() for n in group})
    elif args.partition == "validation_2022":
        names, _ = validation_universe()
    else:
        # 样本外品种池要读样本外行情才能定，这里不读：直接下载所有有合约乘数的品种（只是交易所费率）
        names = sorted(MULTIPLIER)
    path = index_fee_path(args.partition) if args.market == "index" else fee_path(args.partition)
    names = require_multipliers(names)
    credentials = rqdata_credentials()
    if credentials is None:
        raise SystemExit("请在 config/rqdata.env 填写米筐账号")
    import rqdatac as rq
    rq.init(*credentials)
    start, end = WINDOWS[args.partition]
    if args.start_date or args.end_date:
        start = max(start, args.start_date or start)
        end = min(end, args.end_date or end)
        if start > end:
            raise SystemExit(f"日期窗口落在分区 {WINDOWS[args.partition]} 之外")
    parts = []
    for symbol in names:
        table = one_symbol(rq, symbol, start, end)
        print(symbol, 0 if table is None else len(table), flush=True)
        if table is not None:
            parts.append(table)
    if not parts:
        print("窗口内没有主力合约，未改文件", path)
        return
    table = pd.concat(parts, ignore_index=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["trading_date"])
        # 只替换这次下到的品种在这次窗口里的行，窗口外的旧行保留
        replaced = old.symbol.isin(set(table.symbol)) & old.trading_date.between(pd.Timestamp(start), pd.Timestamp(end))
        old = old[~replaced]
        if not old.empty:
            table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values(["symbol", "trading_date"])
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
