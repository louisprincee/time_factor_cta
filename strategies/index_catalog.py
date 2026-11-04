"""股指策略注册表。"""
from strategies.index_clv_rally_base_long_2x import BOOK as RALLY_BASE_2X_BOOK

DEFAULT_BOOK = RALLY_BASE_2X_BOOK
INDEX_BOOKS = (RALLY_BASE_2X_BOOK,)
# 只有冻结的策略能跑样本外；冻结状态以各自 JSON 为准。
INDEX_FROZEN_BOOKS = tuple(book for book in INDEX_BOOKS if book.STATUS == "frozen")
BOOKS = INDEX_BOOKS


def book_named(name):
    requested = str(name).strip()
    for book in INDEX_BOOKS:
        if requested in (book.ID, book.NAME):
            return book
    names = ", ".join(book.ID for book in INDEX_BOOKS)
    raise ValueError(f"未知股指策略 {name!r}；可选策略：{names}")
