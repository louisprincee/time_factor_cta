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
