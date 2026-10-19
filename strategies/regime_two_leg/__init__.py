"""震荡里做晚低点，ADX 离开震荡后只在上涨且高波动时做晚低点。

参数在同目录的 regime_two_leg.json。回测不在这里读行情。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from strategies.book import (
    ADX_PERIOD,
    ADX_RANGE_MAX,
    ATR_PERIOD,
    ATR_REFERENCE,
    LOOKBACK,
    QUANTILE,
    TREND_WINDOW,
    accept_lock,
    upper_tail,
)

CONFIG_PATH = Path(__file__).with_name("regime_two_leg.json")
ID = "regime_two_leg"
NAME = "趋势与震荡分腿"
DETAIL = "ADX小于等于20做尾部/ADX大于20则10日上涨且高波动才做尾部/持有5日"


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
        or chosen.get("atr_period") != ATR_PERIOD
        or chosen.get("atr_reference") != ATR_REFERENCE
        or chosen.get("holding_days") != 5
        or chosen.get("range_leg") != "raw_tail_without_trend"
        or chosen.get("trend_leg") != "trend_up_and_raw_tail_and_high_atr"
    ):
        raise ValueError("趋势与震荡分腿的参数与实现不一致")
    return config


def signal(
    trend: pd.DataFrame,
    micro: pd.DataFrame,
    high_vol: pd.DataFrame,
    ranging: pd.DataFrame,
    above_range: pd.DataFrame,
) -> pd.DataFrame:
    """收盘当日是否允许做多。同一品种同一天只会落在一条腿上。"""
    trend = trend.reindex(index=micro.index, columns=micro.columns)
    high_vol = high_vol.reindex(index=micro.index, columns=micro.columns)
    ranging = ranging.reindex(index=micro.index, columns=micro.columns).fillna(False)
    above_range = above_range.reindex(index=micro.index, columns=micro.columns).fillna(False)
    tail = pd.DataFrame(
        {symbol: upper_tail(micro[symbol]) for symbol in micro.columns},
        index=micro.index,
    ).gt(0)
    range_leg = ranging & tail
    trend_leg = above_range & trend.gt(0) & tail & high_vol.fillna(False)
    return range_leg | trend_leg


load_config()
