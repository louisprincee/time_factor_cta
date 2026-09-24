"""分片读写与样本外守卫测试。

这些测试守护的是**纪律**而不是功能。load_shard 是唯一的读取入口，守卫只写在那里；
若守卫失效，整条研究管道可以无声地读到 2022+ 的数据，而所有绩效数字都将失去意义。
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import shard_io


def _frame(start: str = '2021-11-01', days: int = 5) -> pd.DataFrame:
    td = pd.bdate_range(start, periods=days)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=m + 1)
                            for d in td for m in range(3)])
    return pd.DataFrame({
        'closew': range(len(idx)),
        'close': range(len(idx)),
        'volume': 1.0,
        'trading_date': [d for d in td for _ in range(3)],
    }, index=idx)


@pytest.fixture
def shard_dir():
    """独立临时目录。与 test_factor_cache / test_universe 保持同一种写法。"""
    import tempfile
    return Path(tempfile.mkdtemp(prefix='tfcta_shardio_'))


def test_pickle_roundtrip_preserves_index_and_dtypes(shard_dir):
    df = _frame()
    p = shard_io.save_shard(df, shard_dir, 'RB', fmt='pickle')
    assert p.suffix == shard_io.PICKLE_EXT
    back = pd.read_pickle(p)
    pd.testing.assert_frame_equal(back, df)


def test_load_shard_refuses_holdout_directory():
    """指向 holdout_locked/ 必须抛 HoldoutViolation，哪怕文件根本不存在。

    守卫在读盘之前，所以"文件不存在"不能掩盖越界意图。
    """
    with pytest.raises(C.HoldoutViolation):
        shard_io.load_shard('RB', C.HOLDOUT_DIR)


def test_load_shard_refuses_holdout_subdirectory():
    with pytest.raises(C.HoldoutViolation):
        shard_io.load_shard('RB', C.HOLDOUT_DIR / 'anything')


def test_load_shard_rejects_holdout_dates_even_in_research_dir(shard_dir):
    """双重保险：即便文件放在 research/ 下，内容越界也必须拒绝。

    这正是分片脚本写错切点时唯一能兜住的一层——目录名是对的，数据是错的。
    """
    bad = _frame(start='2021-12-27', days=8)          # 跨过 2022-01-01
    assert bad['trading_date'].max() >= pd.Timestamp(C.HOLDOUT_START)
    shard_io.save_shard(bad, shard_dir, 'RB', fmt='pickle')
    with pytest.raises(C.HoldoutViolation, match='2022'):
        shard_io.load_shard('RB', shard_dir)


def test_verify_dates_false_is_the_only_way_to_bypass(shard_dir):
    """显式关掉校验才能读到越界内容——把绕过守卫变成一个必须写出来的动作，
    这样它在代码评审和 grep 里都是可见的。"""
    bad = _frame(start='2021-12-27', days=8)
    shard_io.save_shard(bad, shard_dir, 'RB', fmt='pickle')
    out = shard_io.load_shard('RB', shard_dir, verify_dates=False)
    assert len(out) == len(bad)


def test_load_shard_sorts_unsorted_shard(shard_dir):
    """分片若因任何原因乱序落盘，读出来必须已排序——持续期计算依赖时间单调。"""
    df = _frame().sample(frac=1.0, random_state=0)
    shard_io.save_shard(df, shard_dir, 'RB', fmt='pickle')
    out = shard_io.load_shard('RB', shard_dir)
    assert out.index.is_monotonic_increasing
