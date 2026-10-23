"""当前冻结策略的注册表。"""
from __future__ import annotations

from strategies import market_clock_slow_short_20_60_120

FROZEN_BOOKS = (market_clock_slow_short_20_60_120.BOOK,)
BOOKS = FROZEN_BOOKS


def book_named(name: str):
    """用策略 ID 或中文名取策略。"""
    key = str(name).strip()
    matches = [book for book in BOOKS if key in (book.ID, book.NAME)]
    if len(matches) != 1:
        known = "、".join(f"{book.NAME}（{book.ID}）" for book in BOOKS)
        raise ValueError(f"未知策略 {key!r}；当前策略：{known}")
    return matches[0]
