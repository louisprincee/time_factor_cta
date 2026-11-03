"""商品策略注册表：冻结策略可开样本外，探索策略只能跑研究期。"""
from __future__ import annotations

from strategies import commodity_close_location, commodity_original

FROZEN_BOOKS = (
    commodity_original.BOOK,
    commodity_close_location.STRICT_SWAP_BOOK,
)
BOOKS = FROZEN_BOOKS


def book_named(name: str):
    """用策略 ID 或中文名取策略。"""
    key = str(name).strip()
    matches = [book for book in BOOKS if key in (book.ID, book.NAME)]
    if len(matches) != 1:
        known = "、".join(book.NAME for book in BOOKS)
        raise ValueError(f"未知策略 {key!r}；当前策略：{known}")
    return matches[0]
