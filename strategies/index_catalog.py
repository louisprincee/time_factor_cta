"""股指策略注册表。"""
from strategies.index_clv_breakout_sleeve import BOOK as BREAKOUT_SLEEVE_BOOK
from strategies.index_tactical_books import FACTOR_TILT_BOOK, TACTICAL_ONLY_BOOK

# 顺序即默认研究对比图图例顺序：纯CLV、CLV因子倾斜、CLV加标准55/20突破。
INDEX_BOOKS = (
    TACTICAL_ONLY_BOOK,
    FACTOR_TILT_BOOK,
    BREAKOUT_SLEEVE_BOOK,
)
DEFAULT_BOOK = INDEX_BOOKS[0]


def book_named(name):
    requested = str(name).strip()
    for book in INDEX_BOOKS:
        if requested in (book.ID, book.NAME):
            return book
    names = ", ".join(book.ID for book in INDEX_BOOKS)
    raise ValueError(f"未知股指策略 {name!r}；可选策略：{names}")
