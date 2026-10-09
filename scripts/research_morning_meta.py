"""早盘元策略：按反向单自己近期的账面盈亏决定继续反向、停手还是改顺势。研究期，以及 2022 的一次性诊断。

    python scripts/research_morning_meta.py
    python scripts/research_morning_meta.py --validation-2022

规则写在 config/morning_meta.json。输出在 runs/morning_oor/meta/<分区>/。
2022 的账面从研究期末接着算（2021 的单进入 2022 年初的窗口）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tfcta import config as C
from tfcta.research import context
import research_morning_oor as oor

CONFIG = ROOT / "config" / "morning_meta.json"
OUT = oor.OUT / "meta"


def trailing_mean(table, side, window):
    """每个交易日可用的“最近 window 笔账面单笔均值”：只用前一交易日及以前的单，当天的单不算。"""
    active = side != 0
    trades = pd.DataFrame({"day": table.trading_date[active], "symbol": table.symbol[active],
                           "net": (side * table.gross_1130 - table.cost_1130)[active]})
    trades = trades.sort_values(["day", "symbol"], kind="mergesort")
    rolling = trades.net.rolling(window, min_periods=window).mean()
    at_close = rolling.groupby(trades.day).last()  # 当天收盘后已知
    calendar = pd.DatetimeIndex(sorted(table.trading_date.unique()))
    known = at_close.reindex(calendar).ffill().shift(1)  # 次日才用
    return table.trading_date.map(known)


def switched(side, score, mode):
    if mode == "原样":
        return side
    losing = score < 0
    if mode == "亏损停手":
        return side.where(~losing, 0.0)
    if mode == "亏损反手":
        return side.where(~losing, -side)
    raise ValueError(f"未知模式 {mode}")


def run(partition):
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    if partition == "research":
        table = oor.prepare("research", base)
        C.assert_no_holdout_dates(table.trading_date, "早盘元策略")
    else:
        table = oor.prepare_through_2022(base, 2016)
    books = oor.base_books(table, base)
    windows = [spec["window_trades"], *spec["neighborhood"]["window_trades"]]
    keep = table.trading_date.dt.year.between(*((2016, 2021) if partition == "research" else (2022, 2022)))
    view = table[keep].reset_index(drop=True)
    if partition != "research":
        C.assert_validation_2022_dates(view.trading_date, "早盘元策略 2022")
    calendar = pd.DatetimeIndex(sorted(view.trading_date.unique()))
    rows, curves = [], {}
    for book, side in books.items():
        for window in windows:
            score = trailing_mean(table, side, window)[keep].reset_index(drop=True)
            base_side = side[keep].reset_index(drop=True)
            for mode in spec["modes"]:
                if mode == "原样" and window != spec["window_trades"]:
                    continue
                new = switched(base_side, score, mode)
                row, ret = oor.book_row(view, new, base["sizing"], calendar)
                losing = (score < 0)[base_side != 0]
                name = f"{book}·{mode}" + ("" if mode == "原样" else f"·{window}笔")
                rows.append({"组合": name, "窗口": window, "切换比例": float(losing.mean()) if mode != "原样" else 0.0,
                             **row, "主设定": window == spec["window_trades"]})
                curves[name] = ret
    result = pd.DataFrame(rows).set_index("组合")
    if partition == "research":
        result["保留"] = result.主设定 & (result.夏普 > 0)  # 保留只看研究期
    out = OUT / partition
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "books.csv", encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(out / "daily.csv", encoding="utf-8-sig")
    main = [n for n in curves if result.loc[n, "主设定"]]
    print("图表：", context.plot_performance(curves, out / "plots" / ("research" if partition == "research" else "validation-2022"), f"早盘元策略 {partition}（30 笔主设定，扣费后）", main))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(result.round(3).to_string())
    print("结果：", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-2022", action="store_true")
    args = parser.parse_args()
    run("validation_2022" if args.validation_2022 else "research")


if __name__ == "__main__":
    main()
