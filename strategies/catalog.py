"""现有的两条策略，都已冻结。冻结状态写在各自的 JSON 里；没冻结的策略只能跑研究期。"""
from __future__ import annotations

from strategies import market_clock_carry, market_clock_slow_short

BOOKS = (market_clock_carry.BOOK, market_clock_slow_short.BOOK)


def book_named(name: str):
    """用策略 ID 或中文名取一条。"""
    key = str(name).strip()
    matches = [book for book in BOOKS if key in (book.ID, book.NAME)]
    if len(matches) != 1:
        known = "、".join(f"{book.NAME}（{book.ID}）" for book in BOOKS)
        raise ValueError(f"没有这个策略 {name}。可选：{known}")
    return matches[0]
