"""分片读写与样本外守卫测试。"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from tfcta import config as C
import tfcta.data as data_module
from tfcta.data import (
    PICKLE_EXT, list_shards, load_oos_shard, load_shard, load_validation_shard,
    save_shard,
)


def _frame(start: str = '2021-11-01', days: int = 5) -> pd.DataFrame:
    td = pd.bdate_range(start, periods=days)
    idx = pd.DatetimeIndex([
        pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=9, minute=m + 1)
        for d in td for m in range(3)])
    return pd.DataFrame({
        'closew': range(len(idx)),
        'close': range(len(idx)),
        'volume': 1.0,
        'trading_date': [d for d in td for _ in range(3)],
    }, index=idx)


@pytest.fixture
def shard_dir():
    """独立临时目录。"""
    import tempfile
    return Path(tempfile.mkdtemp(prefix='tfcta_shardio_'))


def test_pickle_roundtrip_preserves_index_and_dtypes(shard_dir):
    df = _frame()
    p = save_shard(df, shard_dir, 'RB', fmt='pickle')
    assert p.suffix == PICKLE_EXT
    back = pd.read_pickle(p)
    pd.testing.assert_frame_equal(back, df)


def test_load_shard_refuses_holdout_directory():
    """指向 holdout_locked/ 必须抛 HoldoutViolation，哪怕文件根本不存在。"""
    with pytest.raises(C.HoldoutViolation):
        load_shard('RB', C.HOLDOUT_DIR)


def test_load_shard_refuses_holdout_subdirectory():
    with pytest.raises(C.HoldoutViolation):
        load_shard('RB', C.HOLDOUT_DIR / 'anything')


def test_load_shard_refuses_index_futures_holdout_directory():
    with pytest.raises(C.HoldoutViolation):
        load_shard('IM', C.INDEX_HOLDOUT_DIR)


def test_validation_reader_only_accepts_2022(shard_dir, monkeypatch):
    validation_dir = shard_dir / 'validation_2022'
    validation_dir.mkdir()
    df = _frame(start='2022-01-03', days=5)
    save_shard(df, validation_dir, 'RB', fmt='pickle')
    monkeypatch.setattr(C, 'VALIDATION_DIR', validation_dir)

    out = load_validation_shard('RB')
    assert out['trading_date'].dt.year.eq(2022).all()

    bad = _frame(start='2022-12-28', days=8)
    save_shard(bad, validation_dir, 'CU', fmt='pickle')
    with pytest.raises(C.HoldoutViolation, match='只含 2022'):
        load_validation_shard('CU')


def test_validation_reader_refuses_locked_source(shard_dir, monkeypatch):
    with pytest.raises(C.HoldoutViolation, match='必须来自'):
        C.assert_validation_only(C.HOLDOUT_DIR / 'RB.pkl')


def test_elapsed_oos_window_does_not_unlock_research(shard_dir, monkeypatch):
    monkeypatch.setattr(C, 'HOLDOUT_DIR', shard_dir / 'holdout_locked')
    monkeypatch.setattr(data_module, 'find_shard', lambda *a: pytest.fail('OOS file searched'))
    with pytest.raises(C.HoldoutViolation, match='封存'):
        load_oos_shard('RB', end='2025-12-31', today='2026-10-05')


def test_load_shard_rejects_holdout_dates_even_in_research_dir(shard_dir):
    """双重保险：即便文件放在 research/ 下，内容越界也必须拒绝。"""
    bad = _frame(start='2021-12-27', days=8)          # 跨过 2022-01-01
    assert bad['trading_date'].max() >= pd.Timestamp(C.HOLDOUT_START)
    save_shard(bad, shard_dir, 'RB', fmt='pickle')
    with pytest.raises(C.HoldoutViolation, match='2022'):
        load_shard('RB', shard_dir)


def test_verify_dates_false_is_the_only_way_to_bypass(shard_dir):
    """显式关掉校验才能读到越界内容——把绕过守卫变成一个必须写出来的动作， 这样它在代码评审和 grep 里都是可见的。"""
    bad = _frame(start='2021-12-27', days=8)
    save_shard(bad, shard_dir, 'RB', fmt='pickle')
    out = load_shard('RB', shard_dir, verify_dates=False)
    assert len(out) == len(bad)


def test_load_shard_sorts_unsorted_shard(shard_dir):
    """分片若因任何原因乱序落盘，读出来必须已排序——持续期计算依赖时间单调。"""
    df = _frame().sample(frac=1.0, random_state=0)
    save_shard(df, shard_dir, 'RB', fmt='pickle')
    out = load_shard('RB', shard_dir)
    assert out.index.is_monotonic_increasing


def test_oos_monolith_sharding_is_guarded_and_keeps_only_2023_to_2025(
    shard_dir, monkeypatch, tmp_path,
):
    import pickle

    from tfcta.prepare import shard as shard_module

    monkeypatch.setattr(C, "SHARD_ROOT", shard_dir / "minute_shards")
    monkeypatch.setattr(C, "HOLDOUT_DIR", C.SHARD_ROOT / "holdout_locked")
    monkeypatch.setattr(C, "assert_oos_research_locked", lambda end: None)
    days = pd.to_datetime(["2022-12-30", "2023-01-03", "2025-12-31"])
    index = pd.to_datetime([
        "2022-12-30 09:00:00", "2023-01-03 09:00:00", "2025-12-31 09:00:00",
    ])
    fields = [*C.FACTOR_FIELDS, *C.PRICE_FIELDS]
    values = {
        field: (days if field == "trading_date" else [1.0] * len(days))
        for field in fields
    }
    panel = pd.DataFrame(values, index=index)
    panel.columns = pd.MultiIndex.from_product([["RB"], fields])
    source = tmp_path / "minute_monolith.pkl"
    source.write_bytes(pickle.dumps(panel))

    results = shard_module.shard_oos_from_monolith(source, ["RB"], fmt="pickle")

    assert results[0]["status"] == "ok"
    shard = pd.read_pickle(C.HOLDOUT_DIR / "RB.pkl")
    assert shard["trading_date"].min() == pd.Timestamp("2023-01-03")
    assert shard["trading_date"].max() == pd.Timestamp("2025-12-31")
    assert list_shards(C.HOLDOUT_DIR) == ["RB"]
