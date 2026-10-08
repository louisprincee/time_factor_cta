"""分钟级因子的日频缓存（设计文档第 10.2 节的"离线脚本"那一段）。

分钟层计算不能放进日频框架（10.7GB 单体 + stack 必然 OOM），所以先离线把分钟表压成
日频因子落盘，下游只读缓存。

缓存布局
--------
    factor_daily_v3/timestamp/{品种}.ext      时间戳族，不依赖参数，每品种一份
    factor_daily_v3/N{N}_M{M}/{品种}.ext       持续期族，每个 (N, M) 组合一份

每个 (组合, 品种) 是独立文件，已存在即跳过（断点续跑）。只经 ``shard_io.load_shard``
读研究期分钟数据；落盘的是原始因子值，不乘方向，不做任何 fillna。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..data import sessions, shard_io
from . import intraday

TIMESTAMP_DIR_NAME = 'timestamp'
REPORT_DIR_NAME = 'report'
# v3: raw decimal-price durations, positive threshold, full lookback warm-up;
# each individual file is versioned, so a partial rebuild cannot bless old files.
CACHE_VERSION = 3
TIMEPOINT_FACTORS = list(C.TIMESTAMP_FACTORS)
DURATION_FACTORS = [intraday.dfp_name(n) for n in C.FP_TOP_NS]


def combo_grid(lookbacks: list[int] | None = None,
               pcts: list[float] | None = None) -> list[tuple[int, float]]:
    lbs = lookbacks if lookbacks is not None else C.THRESHOLD_LOOKBACKS
    ps = pcts if pcts is not None else C.THRESHOLD_PCTS
    return [(int(n), float(m)) for n in lbs for m in ps]


def combo_name(lookback: int, pct: float) -> str:
    """组合目录名。M 用 %g 去掉无意义的尾零（52.5 -> M52.5，50.0 -> M50）。"""
    return f"N{int(lookback)}_M{pct:g}"


def combo_dir(lookback: int, pct: float, root: Path | None = None) -> Path:
    return (root or C.FACTOR_DAILY_DIR) / combo_name(lookback, pct)


def timestamp_dir(root: Path | None = None) -> Path:
    return (root or C.FACTOR_DAILY_DIR) / TIMESTAMP_DIR_NAME


def report_dir(root: Path | None = None) -> Path:
    return (root or C.FACTOR_DAILY_DIR) / REPORT_DIR_NAME


# --------------------------------------------------------------------------
# 版本
# --------------------------------------------------------------------------
def check_version(path: Path) -> None:
    """Each existing factor file requires its own current-version sidecar."""
    path = Path(path)
    metadata = path.with_suffix(path.suffix + '.json')
    if path.exists() and (not metadata.exists() or
        json.loads(metadata.read_text()).get('version') != CACHE_VERSION):
        raise RuntimeError(f'{path} 因子公式版本不符，请用 build_factors.py --overwrite 重建')


# --------------------------------------------------------------------------
# 计算
# --------------------------------------------------------------------------
def build_symbol(symbol: str,
                 combos: list[tuple[int, float]] | None = None,
                 root: Path | None = None,
                 overwrite: bool = False,
                 fmt: str = 'auto',
                 timestamp: bool = True) -> dict:
    """算一个品种的全部日频因子并落盘。分钟表、坐标、阈值网格各只算一次。

    ``timestamp=False`` 时不碰时间戳族（只重算持续期族时用）。
    """
    combos = combos or combo_grid()
    root = root or C.FACTOR_DAILY_DIR
    lookbacks = sorted({n for n, _ in combos})
    pcts = sorted({m for _, m in combos})

    if not overwrite:
        directories = [combo_dir(n,m,root) for n,m in combos]
        if timestamp:
            directories.append(timestamp_dir(root))
        directories.append(report_dir(root))
        for directory in directories:
            path = shard_io.find_shard(directory,symbol)
            if path is not None:
                check_version(path)
    need_ts = timestamp and (overwrite or shard_io.find_shard(timestamp_dir(root), symbol) is None)
    need_report = overwrite or shard_io.find_shard(report_dir(root), symbol) is None
    todo = [(n, m) for n, m in combos
            if overwrite or shard_io.find_shard(combo_dir(n, m, root), symbol) is None]
    if not need_ts and not todo and not need_report:
        return {'symbol': symbol, 'skipped': True, 'combos_written': 0,
                'timestamp_written': False, 'report_written': False}

    df = sessions.add_intraday_coords(shard_io.load_shard(symbol, columns=C.FACTOR_FIELDS))
    codes, days = intraday.day_codes_of(df)
    info: dict = {'symbol': symbol, 'skipped': False, 'n_days': int(len(days)),
                  'n_bars': int(len(df)),
                  'first_day': str(days[0].date()) if len(days) else '',
                  'last_day': str(days[-1].date()) if len(days) else ''}

    if need_ts:
        ts = intraday.timestamp_factors(df)
        path = shard_io.save_shard(ts, timestamp_dir(root), symbol, fmt)
        path.with_suffix(path.suffix + '.json').write_text(json.dumps({'version': CACHE_VERSION}))
        info['n_timestamp_cols'] = int(ts.shape[1])
    info['timestamp_written'] = bool(need_ts)

    if todo:
        price = intraday.raw_price_path(df)
        grid = intraday.rolling_threshold_grid(
            intraday.intraday_abs_diff(price, codes), codes, lookbacks, pcts)
        for n, m in todo:
            dur = intraday.duration_factors(df, lookback=n, pct=m, thr_p=grid[(n, m)])
            path = shard_io.save_shard(dur, combo_dir(n, m, root), symbol, fmt)
            path.with_suffix(path.suffix + '.json').write_text(json.dumps({'version': CACHE_VERSION}))
            info['n_duration_cols'] = int(dur.shape[1])
    info['combos_written'] = len(todo)
    info['report_written'] = False
    if need_report:
        # 与 DFP 使用同一组 (N, M)。多个组合时取调用方列出的第一组，避免同一目录被后一组覆盖。
        n0, m0 = combos[0]
        report = intraday.report_factors(df, lookback=n0, pct=m0)
        path = shard_io.save_shard(report, report_dir(root), symbol, fmt)
        path.with_suffix(path.suffix + '.json').write_text(json.dumps({'version': CACHE_VERSION}))
        info['report_written'] = True
        info['n_report_cols'] = int(report.shape[1])
    return info


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    """只取当前定义的因子列。旧版缓存里残留的列（已下线的因子）不进入下游。"""
    C.assert_research_only(path)
    check_version(path)
    df = shard_io.read_frame(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{path} 缺少因子列 {missing}，请用 build_factors.py --overwrite 重建")
    df = df[columns]
    df.index = pd.to_datetime(df.index)
    C.assert_no_holdout_dates(df.index, what=str(path))
    return df


def load_symbol(symbol: str, lookback: int, pct: float,
                root: Path | None = None,
                with_timestamp: bool = True) -> pd.DataFrame:
    """单品种在某个 (N, M) 下的日频因子表（持续期族 + 时间戳族 outer 对齐）。"""
    root = root or C.FACTOR_DAILY_DIR
    p = shard_io.find_shard(combo_dir(lookback, pct, root), symbol)
    if p is None:
        raise FileNotFoundError(
            f"缺少 {combo_name(lookback, pct)}/{symbol}，请先运行 build_factors.py")
    out = _read(p, DURATION_FACTORS)
    if with_timestamp:
        q = shard_io.find_shard(timestamp_dir(root), symbol)
        if q is None:
            raise FileNotFoundError(f"缺少 timestamp/{symbol}")
        # outer：两族交易日理论上一致，一旦不一致能看见 NaN 而不是被静默截断
        out = out.join(_read(q, TIMEPOINT_FACTORS), how='outer')
    return out.sort_index()


def load_report(symbol: str, root: Path | None = None) -> pd.DataFrame:
    """报告其余时序因子。与 (N, M) 无关的列和依赖参考阈值的列放在同一张表。"""
    root = root or C.FACTOR_DAILY_DIR
    path = shard_io.find_shard(report_dir(root), symbol)
    if path is None:
        raise FileNotFoundError(f"缺少 report/{symbol}，请先运行 build_factors.py")
    return _read(path, list(intraday.REPORT_COLUMNS)).sort_index()
