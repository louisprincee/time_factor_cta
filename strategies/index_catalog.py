"""股指策略目录，与商品账本注册表分开。用户 2026-10-29：股指只保留熊市空头加波动率缩放。"""
from __future__ import annotations

from strategies.index_bear_reversal import BOOK as BEAR_BOOK

INDEX_FROZEN_BOOKS = (BEAR_BOOK,)
INDEX_BOOKS = INDEX_FROZEN_BOOKS
BOOKS = INDEX_BOOKS


def book_named(name: str):
    """按策略 ID 或中文名查找股指策略。"""
    key = str(name).strip()
    matches = [book for book in BOOKS if key in (book.ID, book.NAME)]
    if len(matches) != 1:
        known = "、".join(book.NAME for book in BOOKS)
        raise ValueError(f"未知股指策略 {key!r}；当前股指策略：{known}")
    return matches[0]
