"""登记策略目录；冻结策略与探索期候选统一供回测入口按名称选择。"""
from __future__ import annotations

from strategies import (
    market_clock_carry,
    market_clock_slow_short,
    market_clock_slow_short_15,
)

FROZEN_BOOKS = (
    market_clock_carry.BOOK,
    market_clock_slow_short.BOOK,
)

BOOKS = (
    *FROZEN_BOOKS,
    market_clock_slow_short_15.BOOK,
)


def book_named(name: str):
    """用策略 ID 或中文名取一条。"""
    key = str(name).strip()
    matches = [book for book in BOOKS if key in (book.ID, book.NAME)]
    if len(matches) != 1:
        known = "、".join(f"{book.NAME}（{book.ID}）" for book in BOOKS)
        raise ValueError(f"没有这个策略 {name}。可选：{known}")
    return matches[0]
