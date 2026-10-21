"""当前的四条，2026-10-21 冻结。没冻结的策略只能跑研究期；冻结后才能用 --validation、--year、--oos 打开 2022 及以后。"""
from __future__ import annotations

from strategies.market_clock_carry import BOOKS

RESEARCH_EXPERIMENTS = BOOKS
