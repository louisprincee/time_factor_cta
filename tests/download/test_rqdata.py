import json
import os
from datetime import date

import pandas as pd

from download.getRiceQuantExternalData import (
    CoverageCache,
    merge_frames,
    missing_ranges,
    remove_unavailable_cache,
    split_period,
    trading_in,
)
from tfcta.rq_auth import load_rqdata_env, rqdata_credentials


def test_split_period_partitions_crossing_range_without_overlap():
    partitions = split_period(date(2021, 12, 30), date(2023, 1, 2))
    assert partitions == [
        ("research", date(2021, 12, 30), date(2021, 12, 31)),
        ("validation_2022", date(2022, 1, 1), date(2022, 12, 31)),
        ("holdout_locked", date(2023, 1, 1), date(2023, 1, 2)),
    ]


def test_split_period_research_only():
    assert split_period(date(2020, 1, 1), date(2020, 1, 2)) == [
        ("research", date(2020, 1, 1), date(2020, 1, 2))
    ]


def test_missing_ranges_only_returns_uncovered_dates():
    gaps = missing_ranges(
        date(2020, 1, 1), date(2020, 1, 10),
        [["20200103", "20200105"], ["20200108", "20200108"]],
    )
    assert gaps == [
        (date(2020, 1, 1), date(2020, 1, 2)),
        (date(2020, 1, 6), date(2020, 1, 7)),
        (date(2020, 1, 9), date(2020, 1, 10)),
    ]


def test_missing_ranges_merges_adjacent_coverage():
    gaps = missing_ranges(
        date(2020, 1, 1), date(2020, 1, 10),
        [["20200103", "20200105"], ["20200106", "20200108"]],
    )
    assert gaps == [(date(2020, 1, 1), date(2020, 1, 2)),
                    (date(2020, 1, 9), date(2020, 1, 10))]


def test_merge_frames_keeps_new_values_for_duplicate_index():
    old = pd.DataFrame({"value": [1, 2]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    new = pd.DataFrame({"value": [20, 3]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    merged = merge_frames(old, new)
    assert merged["value"].tolist() == [1, 20, 3]


def test_cache_requests_only_uncovered_ranges(tmp_path):
    cache = CoverageCache(tmp_path)
    calls = []

    def request(start, end):
        calls.append((start, end))
        return pd.DataFrame({"value": [len(calls)]}, index=[pd.Timestamp(start)])

    cache.fetch("roll_yield", "CU_main_sub", date(2020, 1, 1), date(2020, 1, 3), request)
    cache.fetch("roll_yield", "CU_main_sub", date(2020, 1, 2), date(2020, 1, 4), request)
    cache.flush()

    assert calls == [("20200101", "20200103"), ("20200104", "20200104")]
    assert (tmp_path / "roll_yield" / "CU_main_sub.pkl").exists()
    assert (tmp_path / "coverage.json").exists()


def test_failed_request_is_not_marked_as_covered(tmp_path):
    cache = CoverageCache(tmp_path)

    def request(_start, _end):
        raise RuntimeError("service unavailable")

    cache.fetch("warehouse", "CU", date(2020, 1, 1), date(2020, 1, 2), request)

    assert cache.failures
    assert cache.manifest["items"]["warehouse/CU"]["coverage"] == []


def test_none_response_is_not_marked_as_covered(tmp_path):
    cache = CoverageCache(tmp_path)
    cache.fetch("member_rank", "CU_long", date(2020, 1, 1), date(2020, 1, 2),
                lambda _start, _end: None, require_nonempty=True)

    assert cache.failures
    assert cache.manifest["items"]["member_rank/CU_long"]["coverage"] == []


def test_empty_member_rank_cache_is_invalidated(tmp_path):
    cache = CoverageCache(tmp_path)
    entry = cache._entry("member_rank", "CU_long")
    entry["coverage"] = [["20200101", "20200102"]]
    path = tmp_path / entry["file"]
    path.parent.mkdir(parents=True)
    pd.DataFrame().to_pickle(path)
    cache.flush()
    calls = []

    cache.fetch("member_rank", "CU_long", date(2020, 1, 1), date(2020, 1, 2),
                lambda start, end: calls.append((start, end)) or None,
                require_nonempty=True)

    assert calls == [("20200101", "20200102")]
    assert cache.manifest["items"]["member_rank/CU_long"]["coverage"] == []


def test_empty_dataset_cache_can_be_invalidated(tmp_path):
    cache = CoverageCache(tmp_path)
    entry = cache._entry("spot", "daily_AU9999.SGEX")
    entry["coverage"] = [["20200101", "20200102"]]
    path = tmp_path / entry["file"]
    path.parent.mkdir(parents=True)
    pd.DataFrame().to_pickle(path)
    cache.flush()

    cache.invalidate_empty("spot", key_prefix="daily_")

    assert cache.manifest["items"]["spot/daily_AU9999.SGEX"]["coverage"] == []
    assert not path.exists()


def test_dry_run_does_not_invalidate_or_delete_cache(tmp_path):
    cache = CoverageCache(tmp_path)
    entry = cache._entry("spot", "daily_AU9999.SGEX")
    entry["coverage"] = [["20200101", "20200102"]]
    path = tmp_path / entry["file"]
    path.parent.mkdir(parents=True)
    pd.DataFrame().to_pickle(path)
    cache.flush()
    dry_cache = CoverageCache(tmp_path, dry_run=True)

    dry_cache.invalidate_empty("spot", key_prefix="daily_")

    assert path.exists()
    assert dry_cache.manifest["items"]["spot/daily_AU9999.SGEX"]["coverage"]


def test_remove_unavailable_cache_entries(tmp_path):
    root = tmp_path
    manifest = {
        "version": 1,
        "items": {
            "member_rank/CU_long": {"file": "member_rank/CU_long.pkl", "coverage": []},
            "dominant/CU_front_month": {"file": "dominant/CU_front_month.pkl", "coverage": []},
            "roll_yield/CU_near_main": {"file": "roll_yield/CU_near_main.pkl", "coverage": []},
            "roll_yield/CU_main_sub": {"file": "roll_yield/CU_main_sub.pkl", "coverage": []},
        },
    }
    (root / "coverage.json").write_text(json.dumps(manifest))

    removed = remove_unavailable_cache(root)
    saved = json.loads((root / "coverage.json").read_text())

    assert removed == 3
    assert set(saved["items"]) == {"roll_yield/CU_main_sub"}

def test_coverage_without_file_is_downloaded_again(tmp_path):
    cache = CoverageCache(tmp_path)
    cache._entry("warehouse", "CU")["coverage"] = [["20200101", "20200110"]]
    calls = []

    def request(left, right):
        calls.append((left, right))
        return pd.DataFrame({"on_warrant": [1.0]}, index=pd.to_datetime(["2020-01-02"]))

    cache.fetch("warehouse", "CU", date(2020, 1, 1), date(2020, 1, 10), request)
    assert calls == [("20200101", "20200110")]
    assert (tmp_path / "warehouse" / "CU.pkl").exists()


def test_trading_in_keeps_contracts_overlapping_window():
    instruments = pd.DataFrame({
        "order_book_id": ["CU1501", "CU2203", "CU2401", "CU9999"],
        "listed_date": ["2014-01-16", "2021-03-16", "2023-01-16", "0000-00-00"],
        "de_listed_date": ["2015-01-15", "2022-03-15", "2024-01-15", "0000-00-00"],
    })
    kept = trading_in(instruments, date(2022, 1, 1), date(2022, 12, 31))
    assert kept["order_book_id"].tolist() == ["CU2203", "CU9999"]


def test_load_rqdata_env_fills_blank_names_without_override(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)
    monkeypatch.setenv("RQDATAC_USERNAME", "already-set")
    path = tmp_path / "rqdata.env"
    path.write_text(
        "\n".join([
            "# comment",
            "RQDATAC_LICENSE=",
            "export RQDATAC_USERNAME='from-file'",
            'RQDATAC_PASSWORD="secret"',
            "",
        ]),
        encoding="utf-8",
    )

    load_rqdata_env(path)

    assert os.environ["RQDATAC_USERNAME"] == "already-set"
    assert os.environ["RQDATAC_PASSWORD"] == "secret"
    assert "RQDATAC_LICENSE" not in os.environ


def test_rqdata_credentials_prefers_license(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)
    path = tmp_path / "rqdata.env"
    path.write_text(
        "RQDATAC_LICENSE=token\nRQDATAC_USERNAME=user\nRQDATAC_PASSWORD=pw\n",
        encoding="utf-8",
    )

    assert rqdata_credentials(path) == ("token",)


def test_rqdata_credentials_uses_password_when_license_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)
    path = tmp_path / "rqdata.env"
    path.write_text("RQDATAC_USERNAME=user\nRQDATAC_PASSWORD=pw\n", encoding="utf-8")

    assert rqdata_credentials(path) == ("user", "pw")


def test_missing_env_file_leaves_credentials_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)

    assert rqdata_credentials(tmp_path / "absent.env") is None
