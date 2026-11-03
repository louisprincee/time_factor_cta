"""股指弱收盘策略：按过去20日相对收益排名分配品种预算。"""
from __future__ import annotations

import json
from pathlib import Path

from strategies.index_allocation import target_weights
from strategies.index_close_location import SYMBOLS

ID = "index_clv_return_rank_20d_long"
NAME = "股指弱收盘20日收益排名多头"
GROSS = 1.0
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
CONFIG_PATH = Path(__file__).with_name(f"{ID}.json")
PARAMETERS = {
    "instruments": list(SYMBOLS),
    "clv_threshold": 0.4,
    "signal_rule": "equal-weight mean of IF/IH/IC CLV < 0.4",
    "allocation_rule": "rank ((closew_t - closew_(t-20)) / raw_close_(t-20)); average ranks for ties; normalize to sum 3",
    "allocation_lookback_sessions": 20,
    "allocation_weight_sum": 3.0,
    "allocation_max_multiplier": 2.0,
    "warmup_rule": "equal multipliers until 20-session estimate is available; OOS warmup uses OOS observations only",
    "short_rule": "none",
    "rally_window": 60,
    "rally_minimum": 40,
    "rally_gate": 0.0,
    "rally_multiplier": 1.5,
    "vol_window": 20,
    "vol_target": 0.2,
    "vol_scale_cap": 1.0,
    "gross_exposure_limit": GROSS,
    "drawdown_breaker": DRAWDOWN_LIMIT,
    "halt_days": HALT_DAYS,
    "initial_capital": 10000000,
    "execution": "l1_30m",
    "open_limits": True,
    "signal_execution": "close t -> open t+1",
    "rebalance": "only target change; whole lots",
}


def declared_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config.get("id") != ID
        or config.get("status") != "frozen"
        or not config.get("frozen_at")
        or config.get("parameters") != PARAMETERS
        or config.get("chosen") != PARAMETERS
    ):
        raise ValueError("股指20日收益排名策略冻结规则与实现不一致")
    sample = config.get("sample", "")
    if "尚未回测" not in sample or "不读 2026" not in sample:
        raise ValueError("股指20日收益排名策略缺少样本边界")
    return config


def targets(market_clv, bars, *, rally_z=None):
    if rally_z is None:
        from strategies.index_close_location import high_open_z

        rally_z = high_open_z(bars)
    return target_weights(
        "relative_return_rank_20d", market_clv, bars, rally_z=rally_z
    )


class Book:
    ID, NAME, SYMBOLS = ID, NAME, SYMBOLS
    GROSS, DRAWDOWN_LIMIT, HALT_DAYS = GROSS, DRAWDOWN_LIMIT, HALT_DAYS
    STATUS, CONFIG_PATH = "frozen", CONFIG_PATH
    DETAIL = "共同CLV<0.4；按20日相对收益排名分配IF/IH/IC；保留冲高及波动缩放；无空头"
    declared = staticmethod(declared_config)
    frozen = staticmethod(declared_config)
    targets = staticmethod(targets)

    @property
    def FILES(self):
        return [
            Path(__file__),
            Path(__file__).parents[1] / "index_allocation.py",
            Path(__file__).parents[1] / "index_close_location" / "__init__.py",
            CONFIG_PATH,
        ]


BOOK = Book()
