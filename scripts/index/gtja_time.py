"""按国泰君安研报的规则，复现 IF、IH、IC 的日间择时。只读研究期。

    python scripts/index/gtja_time.py

绩效截止 2021-12-31。不读 2022 及以后，因此不能和研报 2016–2025 的夏普直接画等号。
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tfcta import config as C
from tfcta.data import load_shard
from tfcta.index.gtja_time import COLUMNS, backtest, daily_factors, open_to_open, pair_signal, performance, standardize

CONFIG_PATH = ROOT / "config" / "index" / "gtja_time.json"
OUT = C.RUNS_DIR / "index" / "gtja_time"
EVAL_START = pd.Timestamp("2016-01-04")


def load_config(path=CONFIG_PATH):
    config = json.loads(path.read_text(encoding="utf-8"))
    if "不读 2022" not in config["sample"]:
        raise ValueError("样本口径必须写明不读 2022")
    return config


def load_symbol(symbol):
    frame = load_shard(symbol, directory=C.INDEX_RESEARCH_DIR, columns=[
        "trading_date", "open", "openw", "close", "closew", "highw", "loww", "volume", "total_turnover"])
    if pd.to_datetime(frame.trading_date).dt.year.gt(2021).any():
        raise C.HoldoutViolation(f"{symbol} 研究分片进入了 2022")
    return frame


def main():
    config = load_config()
    built = {}
    forwards = {}
    for symbol in ("IF", "IH", "IC"):
        frame = load_symbol(symbol)
        factors = daily_factors(frame, config["price_threshold"], config["volume_threshold"])
        built[symbol] = standardize(factors[list(COLUMNS)], config["z_window"], config["z_min_periods"])
        forwards[symbol] = open_to_open(frame)
        print(symbol, int(factors.shape[0]), flush=True)

    rows = []
    nets = {}
    for rule in config["rules"]:
        scored = built[rule["symbol"]]
        result = backtest(pair_signal(scored, rule["long"]), forwards[rule["symbol"]], config["fee_per_side"])
        result = result.loc[result.index >= EVAL_START]
        stats = performance(result)
        report = rule["report"]
        rows.append({
            "规则": rule["id"], "品种": rule["symbol"], **stats,
            "研报年化": report["年化"], "研报夏普": report["夏普"], "研报最大回撤": report["最大回撤"],
        })
        nets[rule["id"]] = result.net
    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(nets).to_parquet(OUT / "daily_net.parquet")
    show = summary.copy()
    for col in ("持仓占比", "年化", "最大回撤", "研报年化", "研报最大回撤"):
        show[col] = (show[col] * 100).round(1)
    for col in ("夏普", "卡玛", "盈亏比", "研报夏普"):
        show[col] = show[col].round(2)
    print(show.to_string(index=False))
    print(f"结果：{OUT}")


if __name__ == "__main__":
    main()
