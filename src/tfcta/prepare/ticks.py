"""按研究期分钟数据估计上一年度最小变动价位表。

    python src/tfcta/prepare/ticks.py

估计方法见 tfcta.costs.estimate_ticks。第 y 年的估计用于第 y+1 年的成本（上一年观测到的价位）。
样本外年份只在获准评估时现算。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tfcta import config as C
from tfcta.data import list_shards, load_shard
from tfcta.factors import raw_price_path
from tfcta.costs import MULTIPLIER, estimate_ticks, tick_path

COLUMNS = ["close", "trading_date"]


def table_for():
    rows = []
    for symbol in sorted(set(list_shards(C.RESEARCH_DIR)) & set(MULTIPLIER)):
        frame = load_shard(symbol, directory=C.RESEARCH_DIR, columns=COLUMNS)
        frame = frame.sort_index(kind="mergesort")
        estimate = estimate_ticks(raw_price_path(frame), frame.trading_date)
        rows.append(estimate.assign(symbol=symbol))
    table = pd.concat(rows, ignore_index=True)
    table["rate_per_tick"] = table.tick / table.median_close
    return table[["symbol", "year", "tick", "tick_mode", "n_diff", "median_close", "rate_per_tick"]]


def main():
    table = table_for()
    years = pd.to_datetime(table.year.astype(str) + "-01-01")
    C.assert_no_holdout_dates(years, "tick 表")
    path = tick_path("research")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(path, f"{table.symbol.nunique()} 个品种，{len(table)} 行")


if __name__ == "__main__":
    main()
