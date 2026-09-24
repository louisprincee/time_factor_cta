"""分片读写：格式自适应 + 样本外守卫。

为什么不直接用 to_parquet
-------------------------
parquet 需要 pyarrow 或 fastparquet，两者都是可选依赖。在没有它们的环境里
（如离线沙箱）整条管道会在最后一步落盘时才炸掉。所以这里做格式自适应：
有 parquet 引擎就用 parquet（列式 + 压缩，可只读需要的列），否则退回 pickle。

**所有**分片读取都必须走 load_shard，因为样本外守卫在这里。直接 pd.read_parquet
会绕过守卫，那是 bug。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config as C

PARQUET_EXT = '.parquet'
PICKLE_EXT = '.pkl'


def parquet_available() -> bool:
    for mod in ('pyarrow', 'fastparquet'):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def resolve_format(fmt: str = 'auto') -> str:
    if fmt == 'auto':
        return 'parquet' if parquet_available() else 'pickle'
    if fmt == 'parquet' and not parquet_available():
        raise RuntimeError(
            "指定了 parquet 但环境中没有 pyarrow/fastparquet。\n"
            "请 pip install pyarrow，或改用 --format pickle。"
        )
    return fmt


def shard_path(directory: Path, symbol: str, fmt: str = 'auto') -> Path:
    ext = PARQUET_EXT if resolve_format(fmt) == 'parquet' else PICKLE_EXT
    return Path(directory) / f"{symbol}{ext}"


def save_shard(df: pd.DataFrame, directory: Path, symbol: str,
               fmt: str = 'auto') -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    f = resolve_format(fmt)
    p = shard_path(directory, symbol, f)
    if f == 'parquet':
        df.to_parquet(p, compression='snappy')
    else:
        df.to_pickle(p, compression=None)
    return p


def find_shard(directory: Path, symbol: str) -> Path | None:
    """不管落盘时用的是哪种格式都能找到。"""
    for ext in (PARQUET_EXT, PICKLE_EXT):
        p = Path(directory) / f"{symbol}{ext}"
        if p.exists():
            return p
    return None


def list_shards(directory: Path) -> list[str]:
    d = Path(directory)
    if not d.exists():
        return []
    names = {p.stem for p in d.iterdir()
             if p.suffix in (PARQUET_EXT, PICKLE_EXT)}
    return sorted(names)


def load_shard(symbol: str,
               directory: Path | None = None,
               columns: list[str] | None = None,
               verify_dates: bool = True) -> pd.DataFrame:
    """读取单品种分片，并强制执行样本外纪律。

    参数
    ----
    directory    : 默认 config.RESEARCH_DIR。指向 holdout_locked/ 会直接抛异常。
    columns      : 只读指定列（parquet 下可省 IO；pickle 下退化为读后筛选）
    verify_dates : 额外校验 trading_date 未越过 HOLDOUT_START，双重保险
    """
    directory = Path(directory or C.RESEARCH_DIR)
    C.assert_research_only(directory)

    p = find_shard(directory, symbol)
    if p is None:
        raise FileNotFoundError(f"找不到 {symbol} 的分片: {directory}")
    C.assert_research_only(p)

    if p.suffix == PARQUET_EXT:
        df = pd.read_parquet(p, columns=columns)
    else:
        df = pd.read_pickle(p)
        if columns:
            df = df[[c for c in columns if c in df.columns]]

    if 'trading_date' in df.columns:
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        if verify_dates:
            C.assert_no_holdout_dates(df['trading_date'], what=f"{symbol} 分片")
    return df.sort_index(kind='mergesort')
