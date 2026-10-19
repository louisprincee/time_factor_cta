"""10 日上涨、低点减高点处在自身最高 20%，并且波动高于自己的常态。

参数在同目录的 trend_high_atr.json。回测不在这里读行情。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from strategies.book import (
    ATR_PERIOD, ATR_REFERENCE, LOOKBACK, QUANTILE, TREND_WINDOW, accept_lock, upper_tail,
)

CONFIG_PATH = Path(__file__).with_name("trend_high_atr.json")
ID = "trend_high_atr"
NAME = "趋势加尾部且高波动"
DETAIL = "10日上涨/低点减高点最高20%/ATR高于自身250日中位数/持有5日"


def load_config(path=CONFIG_PATH) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    accept_lock(config)
    chosen = config["parameters"]
    if (
        config.get("id") != ID
        or chosen.get("direction") != "long_only"
        or chosen.get("trend_window") != TREND_WINDOW
        or chosen.get("lookback") != LOOKBACK
        or chosen.get("quantile") != QUANTILE
        or chosen.get("comparison") != "strict"
        or chosen.get("atr_period") != ATR_PERIOD
        or chosen.get("atr_reference") != ATR_REFERENCE
        or chosen.get("holding_days") != 5
    ):
        raise ValueError("趋势加尾部且高波动的参数与实现不一致")
    return config


def signal(
    trend: pd.DataFrame,
    micro: pd.DataFrame,
    high_vol: pd.DataFrame,
    ranging: pd.DataFrame,
    above_range: pd.DataFrame,
) -> pd.DataFrame:
    """收盘当日是否允许做多。ADX 不参与这条规则。"""
    del ranging, above_range
    trend = trend.reindex(index=micro.index, columns=micro.columns)
    high_vol = high_vol.reindex(index=micro.index, columns=micro.columns)
    tail = pd.DataFrame(
        {symbol: upper_tail(micro[symbol]) for symbol in micro.columns},
        index=micro.index,
    )
    return trend.gt(0) & tail.gt(0) & high_vol.fillna(False)


load_config()
