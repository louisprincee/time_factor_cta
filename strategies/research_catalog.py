"""研究期探索策略；与冻结名单分开，不能通过样本外入口运行。"""
from __future__ import annotations

from strategies import market_clock_long

RESEARCH_EXPERIMENTS = (market_clock_long,)
