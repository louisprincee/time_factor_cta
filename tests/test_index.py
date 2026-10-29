"""Index feed, tick-summary, and mechanism contracts."""

# --- test_index_futures.py ---

import json
from datetime import date
import pandas as pd
import pytest
from tfcta import config as C
from download.index.getRiceQuantIndexFutures import (
    INDEX_LISTED_FROM, assemble_symbol_frames, effective_start, save_oos_locked,
    split_oos_frames, split_research_frames,
)

def api_frame(index, suffix):
    date = pd.Series(pd.to_datetime(index).normalize(), index=index)
    frame = pd.DataFrame({"trading_date": date}, index=index)
    if suffix in ("88", "889"):
        frame = frame.assign(open=100.0, high=101.0, low=99.0, close=100.5)
    if suffix == "88":
        frame["total_turnover"] = 1000.0
        frame["dominant_id"] = "IF2101"
    if suffix == "99":
        frame["volume"] = 10.0
        frame["open_interest"] = 100.0
    return frame

def test_assemble_symbol_frames_maps_raw_adjusted_volume_and_oi_fields():
    index = pd.date_range("2021-01-04 09:31", periods=2, freq="min")
    main = api_frame(index, "88")
    next_main = api_frame(index, "889").assign(close=101.0)
    aggregate = api_frame(index, "99")
    result = assemble_symbol_frames(main, next_main, aggregate)
    assert result.index.equals(index)
    assert result.close.tolist() == [100.5, 100.5]
    assert result.closew.tolist() == [101.0, 101.0]
    assert result.volume.tolist() == [10.0, 10.0]
    assert result.open_interest.tolist() == [100.0, 100.0]
    assert result.trading_date.tolist() == [pd.Timestamp("2021-01-04")] * 2

def test_assemble_symbol_frames_refuses_misaligned_series():
    index = pd.date_range("2021-01-04 09:31", periods=2, freq="min")
    with pytest.raises(ValueError, match="indexes differ"):
        assemble_symbol_frames(api_frame(index, "88"), api_frame(index[:1], "889"),
                               api_frame(index, "99"))

def test_research_split_rejects_holdout_dates():
    index = pd.date_range("2022-01-04 09:31", periods=2, freq="min")
    frame = assemble_symbol_frames(api_frame(index, "88"), api_frame(index, "889"),
                                   api_frame(index, "99"))
    with pytest.raises(C.HoldoutViolation, match="样本外日期"):
        split_research_frames(frame)

def test_research_split_preserves_factor_and_open_interest_fields():
    index = pd.date_range("2021-12-30 09:31", periods=2, freq="min")
    frame = assemble_symbol_frames(api_frame(index, "88"), api_frame(index, "889"),
                                   api_frame(index, "99"))
    prices, oi = split_research_frames(frame)
    assert {"closew", "close", "highw", "loww", "volume", "total_turnover", "open", "openw"} <= set(prices)
    assert set(oi) == {"open_interest", "dominant_id", "trading_date"}
    assert len(prices) == len(oi) == 2

def test_im_oos_window_begins_on_its_listing_date():
    assert effective_start("IM", pd.Timestamp("2022-01-01").date()) == INDEX_LISTED_FROM["IM"]
    assert effective_start("IF", pd.Timestamp("2022-01-01").date()) == pd.Timestamp("2022-01-01").date()

def test_oos_split_accepts_only_the_declared_2022_to_2025_window():
    index = pd.date_range("2022-07-22 09:31", periods=2, freq="min")
    frame = assemble_symbol_frames(api_frame(index, "88"), api_frame(index, "889"),
                                   api_frame(index, "99"))
    prices, oi = split_oos_frames(frame, pd.Timestamp("2022-07-22").date(),
                                  pd.Timestamp("2025-12-31").date())
    assert len(prices) == len(oi) == 2

def test_oos_download_starts_im_at_listing_and_only_writes_locked_paths(tmp_path, monkeypatch):
    index = pd.date_range("2022-07-22 09:31", periods=2, freq="min")
    class FakeRQ:
        def __init__(self):
            self.calls = []
        def get_price(self, instrument, start_date, end_date, frequency):
            self.calls.append((instrument, start_date, end_date, frequency))
            suffix = "889" if instrument.endswith("889") else instrument[-2:]
            return api_frame(index, suffix)
    rq = FakeRQ()
    save_oos_locked(rq, ["IM"], date(2022, 1, 1), date(2025, 12, 31), tmp_path)
    assert {call[1] for call in rq.calls} == {"20220722"}
    assert {call[2] for call in rq.calls} == {"20251231"}
    locked = tmp_path / "minute_shards" / "holdout_locked"
    assert (locked / "IM.parquet").exists()
    assert (locked / "open_interest" / "IM.parquet").exists()
    assert json.loads((locked / "oos_manifest.json").read_text())["symbols"] == ["IM"]
    monkeypatch.setattr(C, "INDEX_HOLDOUT_DIR", locked)
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(locked / "IM.parquet")


# --- test_index_liquidity_mechanism.py ---

import numpy as np
import pandas as pd
import pytest





# --- test_index_positioning_mechanism.py ---

import numpy as np
import pandas as pd
import pytest





# --- test_tick_l1.py ---

from datetime import date

import pandas as pd
import pytest

from tfcta.config import HoldoutViolation
import download.index.getTickL1Close as tick_download
from download.index.getTickL1Close import (
    l1_day_stats,
    prepare_raw_ticks,
    validate_window,
)


def test_index_l1_summary_uses_regular_session_and_last_30_minutes():
    day = pd.Timestamp("2016-01-04")
    ticks = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2016-01-04 09:29:59",
            "2016-01-04 09:30:00",
            "2016-01-04 14:29:59",
            "2016-01-04 14:30:00",
            "2016-01-04 15:00:00",
            "2016-01-04 15:00:01",
        ]),
        "volume": [1, 2, 3, 4, 5, 6],
        "a1": [100.1, 100.1, 100.1, 100.1, 100.1, 100.1],
        "b1": [99.9, 99.9, 99.9, 99.9, 99.9, 99.9],
        "a1_v": [10, 10, 10, 10, 10, 10],
        "b1_v": [12, 12, 12, 12, 12, 12],
        "last": [100.0, 100.0, 100.0, 100.0, 100.05, 100.0],
        "high": [100.0, 100.0, 100.0, 100.0, 100.1, 100.1],
        "low": [100.0, 100.0, 100.0, 100.0, 99.9, 99.9],
    })

    result = l1_day_stats(ticks, day)

    assert result["n_ticks"] == 4
    assert result["trading_date"] == day
    assert result["close_last"] == pytest.approx(100.05)
    assert result["mid_final"] == pytest.approx(100.0)
    assert result["depth_day"] == pytest.approx(22.0)
    assert result["depth_last30"] == pytest.approx(22.0)
    assert result["clv_last"] == pytest.approx(0.75)


def test_index_l1_summary_handles_no_session_ticks():
    ticks = pd.DataFrame({
        "datetime": pd.to_datetime(["2016-01-04 09:00:00"]),
        "volume": [1],
        "a1": [100.1],
        "b1": [99.9],
        "a1_v": [10],
        "b1_v": [12],
        "last": [100.0],
        "high": [100.0],
        "low": [100.0],
    })

    result = l1_day_stats(ticks, "2016-01-04")

    assert result["n_ticks"] == 0
    assert result["trading_date"] == pd.Timestamp("2016-01-04")


def test_raw_index_ticks_keep_all_fields_and_add_contract_metadata():
    ticks = pd.DataFrame(
        {
            "trading_date": [pd.Timestamp("2016-01-04")],
            "open": [100.0],
            "last": [100.1],
            "high": [100.2],
            "low": [99.9],
            "volume": [5],
            "a1": [100.2],
            "b1": [100.0],
            "a1_v": [10],
            "b1_v": [12],
            "a2": [0.0],
            "a2_v": [0],
        },
        index=pd.DatetimeIndex(["2016-01-04 09:30:00"], name="datetime"),
    )

    result = prepare_raw_ticks(ticks, "IF", "IF1601")

    assert result.loc[0, "symbol"] == "IF"
    assert result.loc[0, "contract"] == "IF1601"
    assert result.loc[0, "datetime"] == pd.Timestamp("2016-01-04 09:30:00")
    assert result.loc[0, "trading_date"] == pd.Timestamp("2016-01-04")
    assert result.loc[0, "a2"] == 0.0
    assert result.loc[0, "a2_v"] == 0


def test_raw_index_ticks_fail_if_l1_quotes_are_missing():
    ticks = pd.DataFrame(
        {
            "trading_date": [pd.Timestamp("2016-01-04")],
            "open": [100.0],
            "last": [100.1],
            "high": [100.2],
            "low": [99.9],
            "volume": [5],
        },
        index=pd.DatetimeIndex(["2016-01-04 09:30:00"], name="datetime"),
    )

    with pytest.raises(ValueError, match="a1"):
        prepare_raw_ticks(ticks, "IF", "IF1601")


def test_tick_downloader_separates_research_and_locked_oos_windows():
    validate_window(pd.Timestamp("2022-01-01").date(), pd.Timestamp("2025-12-31").date(), "oos_locked")

    with pytest.raises(HoldoutViolation, match="锁定股指 L1"):
        validate_window(pd.Timestamp("2021-12-31").date(), pd.Timestamp("2025-12-31").date(), "oos_locked")
    with pytest.raises(HoldoutViolation, match="锁定股指 L1"):
        validate_window(pd.Timestamp("2022-01-01").date(), pd.Timestamp("2026-01-01").date(), "oos_locked")
    with pytest.raises(HoldoutViolation, match="研究期股指 L1"):
        validate_window(pd.Timestamp("2022-01-01").date(), pd.Timestamp("2022-12-31").date(), "research")


def test_locked_download_writes_raw_ticks_without_research_summaries(tmp_path, monkeypatch):
    day = pd.Timestamp("2022-01-03")

    class FakeFutures:
        @staticmethod
        def get_dominant(symbol, start, end, rule):
            return pd.Series(["IF2201"], index=pd.DatetimeIndex([day]))

    class FakeRQ:
        futures = FakeFutures()

        @staticmethod
        def init(*credentials):
            return None

        @staticmethod
        def get_price(contract, start_date, end_date, frequency):
            return pd.DataFrame(
                {
                    "trading_date": [day, day],
                    "open": [100.0, 100.0],
                    "last": [100.1, 100.2],
                    "high": [100.2, 100.2],
                    "low": [99.9, 99.9],
                    "volume": [1, 2],
                    "a1": [100.2, 100.3],
                    "b1": [100.0, 100.1],
                    "a1_v": [10, 11],
                    "b1_v": [12, 13],
                },
                index=pd.DatetimeIndex(
                    ["2022-01-03 09:30:00", "2022-01-03 09:30:01"],
                    name="datetime",
                ),
            )

    monkeypatch.setattr(tick_download, "LOCKED_RAW_OUT", tmp_path / "locked" / "l1_ticks")
    monkeypatch.setattr(tick_download, "rqdata_credentials", lambda: ("user", "password"))
    monkeypatch.setitem(__import__("sys").modules, "rqdatac", FakeRQ)

    result = tick_download.download_locked(["IF"], date(2022, 1, 1), date(2022, 1, 4))

    symbol_root = tmp_path / "locked" / "l1_ticks" / "IF"
    parts = list(symbol_root.rglob("*.parquet"))
    assert result == 0
    assert len(parts) == 1
    assert not (tmp_path / "locked" / "l1_close" / "IF.csv").exists()
    metadata = (symbol_root / "download_manifest.json").read_text(encoding="utf-8")
    assert '"content_not_loaded_for_analysis": true' in metadata


from download.index.getIndexSpot import normalize_spot_minutes


def test_qvix_split_keeps_research_dates_and_same_day_close_only(tmp_path, monkeypatch):
    """2022 及以后只进锁定期；信号日用当天 QVIX，不用下一天。"""
    from download.index.getQvix import load_research_qvix, split_qvix

    raw = pd.DataFrame({
        "date": pd.to_datetime(["2015-02-09", "2021-12-31", "2022-01-04"]),
        "open": [10.0, 20.0, 30.0],
        "high": [11.0, 21.0, 31.0],
        "low": [9.0, 19.0, 29.0],
        "close": [10.5, 20.5, 30.5],
    })
    research, holdout = split_qvix(raw)
    assert research["date"].max() == pd.Timestamp("2021-12-31")
    assert set(holdout["date"]) == {pd.Timestamp("2022-01-04")}
    locked = tmp_path / "holdout_locked"
    locked.mkdir()
    target = locked / "qvix_50etf.csv"
    holdout.to_csv(target, index=False)
    monkeypatch.setattr(C, "INDEX_EXTERNAL_HOLDOUT_DIR", locked)
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(target)
    with pytest.raises(C.HoldoutViolation):
        load_research_qvix(target)
    research_path = tmp_path / "qvix_50etf.csv"
    research.to_csv(research_path, index=False)
    loaded = load_research_qvix(research_path)
    assert loaded.loc[loaded["date"] == pd.Timestamp("2021-12-31"), "close"].iloc[0] == pytest.approx(20.5)
    leaked = research.copy()
    leaked.loc[len(leaked), ["date", "open", "high", "low", "close"]] = [
        pd.Timestamp("2022-01-04"), 1, 1, 1, 1,
    ]
    leaked_path = tmp_path / "leaked.csv"
    leaked.to_csv(leaked_path, index=False)
    with pytest.raises(C.HoldoutViolation):
        load_research_qvix(leaked_path)




def test_spot_minute_normalize_rejects_2022():
    raw = pd.DataFrame({
        "datetime": pd.to_datetime(["2022-01-04 09:31:00"]),
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0],
    })
    with pytest.raises(C.HoldoutViolation):
        normalize_spot_minutes(raw, "IF")


def test_read_index_2022_minutes_drops_later_years_and_rejects_2026(tmp_path, monkeypatch):
    import backtest.engine as engine

    monkeypatch.setattr(C, "INDEX_HOLDOUT_DIR", tmp_path)
    rows = pd.DataFrame({
        "trading_date": pd.to_datetime(["2022-01-04", "2023-01-04"]),
        "open": [1.0, 9.0], "openw": [1.0, 9.0], "close": [1.0, 9.0],
        "closew": [1.0, 9.0], "highw": [2.0, 9.0], "loww": [0.5, 9.0],
    })
    rows.to_parquet(tmp_path / "IF.parquet", index=False)
    kept = engine.read_index_2022_minutes("IF")
    assert kept["trading_date"].dt.year.unique().tolist() == [2022]
    assert kept["close"].tolist() == [1.0]
    leaked = pd.concat([rows, pd.DataFrame({"trading_date": [pd.Timestamp("2026-01-05")]})], ignore_index=True)
    leaked.to_parquet(tmp_path / "IH.parquet", index=False)
    with pytest.raises(C.HoldoutViolation):
        engine.read_index_2022_minutes("IH")
    with pytest.raises(C.HoldoutViolation):
        engine.read_index_2022_minutes("IM")


def test_bear_2022_records_the_ledger_before_reading_holdout(tmp_path, monkeypatch):
    import backtest.engine as engine
    from strategies.index_bear_reversal import BOOK as BEAR_BOOK

    monkeypatch.setattr(C, "INDEX_OOS_LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(engine, "read_index_2022_minutes", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("holdout")))
    with pytest.raises(RuntimeError, match="holdout"):
        engine.run_index_bear_2022(BEAR_BOOK)
    rows = C.read_ledger(tmp_path / "ledger.jsonl")
    assert [row["status"] for row in rows] == ["opened"]
    assert rows[0]["strategy"] == "index_close_location_bear_short_2022"
    assert rows[0]["test_start"] == "2022-01-01"
    assert rows[0]["test_end"] == "2022-12-31"


def test_index_only_the_bear_short_book_exists_and_opens_only_2022(monkeypatch):
    import backtest.engine as engine

    for removed in ("index_close_location_reversal_strict_swap", "index_close_location_reversal"):
        with pytest.raises(SystemExit):
            engine.main([removed, "--research"])
    monkeypatch.setattr(
        engine, "run_index_bear_2022",
        lambda book: (_ for _ in ()).throw(AssertionError("不应打开熊市空头的 2023")),
    )
    with pytest.raises(SystemExit):
        engine.main(["index_close_location_bear_short", "--oos"])
    with pytest.raises(SystemExit):
        engine.main(["index_close_location_bear_short", "--year", "2023"])
    opened = {}
    monkeypatch.setattr(engine, "run_index_bear_2022", lambda book: opened.setdefault("id", book.ID))
    engine.main(["index_close_location_bear_short", "--validation"])
    assert opened["id"] == "index_close_location_bear_short"
