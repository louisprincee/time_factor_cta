"""10 日上涨，并且低点减高点的 3 日指数平均处在自身最高 20%。

参数在同目录的 trend_ema3.json。回测不在这里读行情。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from strategies.book import LOOKBACK, QUANTILE, TREND_WINDOW, accept_lock, upper_tail

CONFIG_PATH = Path(__file__).with_name("trend_ema3.json")
SPAN = 3
ID = "trend_ema3"
NAME = "趋势加三日平滑尾部"
DETAIL = "10日上涨/EMA3低点减高点/250日最高20%/持有5日"


def load_config(path=CONFIG_PATH) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    accept_lock(config)
    chosen = config["parameters"]
    if (
        config.get("id") != ID
        or chosen.get("direction") != "long_only"
        or chosen.get("trend_window") != TREND_WINDOW
        or chosen.get("smoothing_span") != SPAN
        or chosen.get("lookback") != LOOKBACK
        or chosen.get("quantile") != QUANTILE
        or chosen.get("comparison") != "strict"
        or chosen.get("holding_days") != 5
    ):
        raise ValueError("趋势加三日平滑尾部的参数与实现不一致")
    return config


def signal(
    trend: pd.DataFrame,
    micro: pd.DataFrame,
    high_vol: pd.DataFrame,
    ranging: pd.DataFrame,
    above_range: pd.DataFrame,
) -> pd.DataFrame:
    """收盘当日是否允许做多。波动和 ADX 不参与这条规则。"""
    del high_vol, ranging, above_range
    trend = trend.reindex(index=micro.index, columns=micro.columns)
    smoothed = micro.ewm(span=SPAN, min_periods=SPAN, adjust=False).mean()
    tail = pd.DataFrame(
        {symbol: upper_tail(smoothed[symbol]) for symbol in smoothed.columns},
        index=smoothed.index,
    )
    return trend.gt(0) & tail.gt(0)


load_config()
