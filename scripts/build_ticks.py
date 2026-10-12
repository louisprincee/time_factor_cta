"""最小变动价位表：按年从分钟原始收盘价估计，写到 data/research/tick_size.csv 和 data/validation_2022/tick_size.csv。

    python scripts/build_ticks.py

估计方法见 tfcta.research.estimate_ticks。第 y 年的估计用于第 y+1 年的成本（上一年观测到的价位）。
样本外年份不落盘，由 C.final_evaluation(...) 里的特征构建现算。
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tfcta import config as C
from tfcta.data import list_shards, load_shard, load_validation_shard
from tfcta.factors import raw_price_path
from tfcta.research import MULTIPLIER, estimate_ticks, tick_path

COLUMNS = ["close", "trading_date"]


def table_for(partition):
    rows = []
    directory = C.RESEARCH_DIR if partition == "research" else C.VALIDATION_DIR
    for symbol in sorted(set(list_shards(directory)) & set(MULTIPLIER)):
        if partition == "research":
            frame = load_shard(symbol, directory=C.RESEARCH_DIR, columns=COLUMNS)
        else:
            frame = load_validation_shard(symbol, columns=COLUMNS)
        frame = frame.sort_index(kind="mergesort")
        estimate = estimate_ticks(raw_price_path(frame), frame.trading_date)
        if partition != "research":
            estimate = estimate[estimate.year == C.VALIDATION_YEAR]
        rows.append(estimate.assign(symbol=symbol))
    table = pd.concat(rows, ignore_index=True)
    table["rate_per_tick"] = table.tick / table.median_close
    return table[["symbol", "year", "tick", "tick_mode", "n_diff", "median_close", "rate_per_tick"]]


def main():
    for partition in ("research", "validation_2022"):
        table = table_for(partition)
        years = pd.to_datetime(table.year.astype(str) + "-01-01")
        if partition == "research":
            C.assert_no_holdout_dates(years, "tick 表")
        else:
            C.assert_validation_2022_dates(years, "2022 tick 表")
        path = tick_path(partition)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(path, f"{table.symbol.nunique()} 个品种，{len(table)} 行")


if __name__ == "__main__":
    main()
