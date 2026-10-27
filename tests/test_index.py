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

from research.index_liquidity_mechanism import tick_day_features


def test_flow_and_residual_add_up_and_lambda_recovers_the_planted_impact():
    rng = np.random.default_rng(0)
    stamps = pd.date_range("2018-03-15 09:30:00", periods=7200, freq="1s")
    signs = rng.choice([-1.0, 1.0], size=len(stamps))
    mid = 4000 * np.exp(np.cumsum(2e-6 * signs))             # 每手买单推高中间价 2e-6（对数）
    ticks = pd.DataFrame({"datetime": stamps, "last": mid + 0.1 * signs, "a1": mid + 0.1, "b1": mid - 0.1,
                          "a1_v": 5.0, "b1_v": 5.0, "volume": np.arange(1, len(stamps) + 1, dtype=float)})
    out = tick_day_features(ticks)
    assert out["flow"] + out["resid"] == pytest.approx(out["intraday_mid"])
    assert out["lambda"] == pytest.approx(2e-6, rel=0.25)
    assert out["mid_first30"] == pytest.approx(
        ticks.loc[
            ticks["datetime"] <= stamps[0] + pd.Timedelta(30, unit="min"), "a1"
        ].iloc[-1]
        - 0.1
    )


# --- test_index_positioning_mechanism.py ---

import numpy as np
import pandas as pd
import pytest

from research import index_positioning_mechanism as P


def test_flows_are_scaled_by_prior_open_interest_and_expiry_days_are_flagged():
    dates = pd.bdate_range("2018-03-12", periods=4)
    rank = pd.concat([pd.DataFrame({"symbol": s, "side": side, "trading_date": dates, "volume_change": change})
                      for s in P.SYMBOLS for side, change in (("short", 100.0), ("long", -50.0))])
    daily = {s: pd.DataFrame({"date": np.repeat(dates, 2), "open_interest": 500.0}) for s in P.SYMBOLS}
    pos = P.positioning_panel(rank, daily, dates)
    assert pos["short_flow"]["IF"].iloc[1:].tolist() == pytest.approx([0.1] * 3)      # 100 / 前一日 1000
    assert pos["long_flow"]["IC"].iloc[0] != pos["long_flow"]["IC"].iloc[0]            # 第一天没有前一日，缺失
    flag = P.expiry_mask(dates, pd.Series(pd.to_datetime(["2018-03-14"])))
    assert flag.tolist() == [False, True, True, False]
    s = pd.Series(np.arange(300, dtype=float))
    moved = s.copy()
    moved.iloc[-1] = -1e9
    assert P.above_own_median(s).iloc[:-1].equals(P.above_own_median(moved).iloc[:-1])


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
