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