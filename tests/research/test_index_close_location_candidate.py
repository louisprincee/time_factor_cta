import pandas as pd
import pytest

from research.index_close_location_candidate import (
    CANDIDATE_ID,
    BENCHMARK_ID,
    SYMBOLS,
    build_targets,
    load_protocol,
)
from strategies.catalog import BOOKS as COMMODITY_BOOKS
from strategies.index_catalog import BOOKS as INDEX_BOOKS, book_named


def test_candidate_uses_prior_close_location_and_registered_binary_rule():
    index = pd.date_range("2016-01-13", periods=4, freq="B")
    low = pd.DataFrame(0.0, index=index, columns=SYMBOLS)
    high = pd.DataFrame(10.0, index=index, columns=SYMBOLS)
    close = pd.DataFrame(
        [[2.0] * 3, [8.0] * 3, [1.0] * 3, [9.0] * 3],
        index=index,
        columns=SYMBOLS,
    )
    bars = {"loww": low, "highw": high, "closew": close}

    targets, market_clv = build_targets(bars, index)

    protocol = load_protocol()
    assert protocol["status"] == "exploratory_predeclared"
    assert protocol["parameters"]["score_start"] == "2016-01-13"
    assert book_named(CANDIDATE_ID) is INDEX_BOOKS[0]
    assert all(book.ID != CANDIDATE_ID for book in COMMODITY_BOOKS)
    assert market_clv.tolist() == pytest.approx([0.2, 0.8, 0.1, 0.9])
    assert targets[CANDIDATE_ID].iloc[:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert targets[BENCHMARK_ID].eq(1.0).all().all()

    changed = close.copy()
    changed.iloc[-1] = 0.0
    changed_targets, _ = build_targets({**bars, "closew": changed}, index)
    pd.testing.assert_frame_equal(
        targets[CANDIDATE_ID].iloc[:-1],
        changed_targets[CANDIDATE_ID].iloc[:-1],
    )
