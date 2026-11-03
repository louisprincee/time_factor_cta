"""股指冻结策略注册表；默认仍使用原有固定品种权重策略。"""
from strategies.index_close_location import BOOK
from strategies.index_clv_individual_equal import BOOK as INDIVIDUAL_EQUAL_BOOK
from strategies.index_clv_pooled_equal import BOOK as POOLED_EQUAL_BOOK
from strategies.index_clv_return_rank_20d import BOOK as RETURN_RANK_BOOK
from strategies.index_clv_inverse_vol_20d import BOOK as INVERSE_VOL_BOOK

DEFAULT_BOOK = BOOK
INDEX_FROZEN_BOOKS = (
    BOOK,
    INDIVIDUAL_EQUAL_BOOK,
    POOLED_EQUAL_BOOK,
    RETURN_RANK_BOOK,
    INVERSE_VOL_BOOK,
)
INDEX_BOOKS = INDEX_FROZEN_BOOKS
BOOKS = INDEX_BOOKS


def book_named(name):
    requested = str(name).strip()
    for book in INDEX_BOOKS:
        if requested in (book.ID, book.NAME):
            return book
    names = ", ".join(book.ID for book in INDEX_BOOKS)
    raise ValueError(f"未知股指策略 {name!r}；可选策略：{names}")
