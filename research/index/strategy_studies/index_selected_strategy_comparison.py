"""跑当前保留的三本股指策略研究期对比，并核对引擎写出的总对比图。

    python -m research.index strategy comparison

总图和逐年图由 backtest.engine 在全部策略账本结束后写入
runs/index/research/index_strategy_comparison/plots/。
各策略自己相对等权买入持有的图仍在各自账本的 plots/ 里。
"""
from __future__ import annotations

import json

import pandas as pd

from strategies.index_catalog import INDEX_BOOKS
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_selected_strategy_comparison.json"
OUT = C.study_dir("index", "index_strategy_comparison")
START_YEAR, END_YEAR = 2015, 2021


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    strategy_ids = [book.ID for book in INDEX_BOOKS]
    if (
        protocol.get("status") != "preregistered_selection_comparison"
        or protocol.get("strategies") != strategy_ids
        or "不读取2022–2025" not in protocol.get("sample", "")
        or "任何oos/" not in protocol.get("sample", "")
        or len(strategy_ids) != 3
    ):
        raise ValueError("当前三策略对比预登记或策略注册不一致")
    return protocol


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = load_protocol()
    C.assert_research_only(C.INDEX_RESEARCH_DIR)
    from backtest import engine

    engine.run_index(START_YEAR, END_YEAR)
    summary_rows, annual_rows = [], []
    for book in INDEX_BOOKS:
        result_dir = C.backtest_dir("index", book.ID, "research")
        summary = pd.read_csv(result_dir / "summary.csv")
        annual = pd.read_csv(result_dir / "annual.csv")
        summary.insert(0, "策略ID", book.ID)
        annual.insert(0, "策略ID", book.ID)
        summary_rows.append(summary)
        annual_rows.append(annual)
    summary_table = pd.concat(summary_rows, ignore_index=True)
    annual_table = pd.concat(annual_rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_table.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual_table.to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    performance = OUT / "plots" / "performance.png"
    yearly = [OUT / "plots" / f"{year}.png" for year in range(START_YEAR, END_YEAR + 1)]
    if not performance.is_file() or any(not path.is_file() for path in yearly):
        raise FileNotFoundError(f"引擎没有写全当前三策略对比图：{OUT / 'plots'}")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print("\n研究期汇总", flush=True)
        print(summary_table.round(4).to_string(index=False), flush=True)
        print("\n逐年结果", flush=True)
        print(annual_table.round(4).to_string(index=False), flush=True)
    print(f"当前三策略对比图：{OUT / 'plots'}", flush=True)
    return summary_table, annual_table


if __name__ == "__main__":
    run()
