"""10 日上涨、低点减高点处在自身最高 20%，并且 ADX 处在震荡区间。

参数在同目录的 trend_adx_range.json。回测不在这里读行情。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from strategies.book import (
    ADX_PERIOD, ADX_RANGE_MAX, LOOKBACK, QUANTILE, TREND_WINDOW, accept_lock, upper_tail,
)

CONFIG_PATH = Path(__file__).with_name("trend_adx_range.json")
ID = "trend_adx_range"
NAME = "趋势加尾部且ADX震荡"
DETAIL = "10日上涨/低点减高点最高20%/ADX小于等于20/持有5日"


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
        or chosen.get("adx_period") != ADX_PERIOD
        or chosen.get("adx_range_max") != ADX_RANGE_MAX
        or chosen.get("holding_days") != 5
    ):
        raise ValueError("趋势加尾部且ADX震荡的参数与实现不一致")
    return config


def signal(
    trend: pd.DataFrame,
    micro: pd.DataFrame,
    high_vol: pd.DataFrame,
    ranging: pd.DataFrame,
    above_range: pd.DataFrame,
) -> pd.DataFrame:
    """收盘当日是否允许做多。高波动不参与这条规则。"""
    del high_vol, above_range
    trend = trend.reindex(index=micro.index, columns=micro.columns)
    ranging = ranging.reindex(index=micro.index, columns=micro.columns)
    tail = pd.DataFrame(
        {symbol: upper_tail(micro[symbol]) for symbol in micro.columns},
        index=micro.index,
    )
    return trend.gt(0) & tail.gt(0) & ranging.fillna(False)


load_config()
