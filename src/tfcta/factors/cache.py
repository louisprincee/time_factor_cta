"""分钟级因子的日频缓存（设计文档第 10.2 节的"离线脚本"那一段）。

分钟层计算不能放进日频框架（10.7GB 单体 + stack 必然 OOM），所以先离线把分钟表压成
日频因子落盘，下游只读缓存。

缓存布局
--------
    factor_daily/timestamp/{品种}.ext          时间戳族，不依赖参数，每品种一份
    factor_daily/N{N}_M{M}/{品种}.ext          持续期族，每个 (N, M) 组合一份

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

    need_ts = timestamp and (overwrite or shard_io.find_shard(timestamp_dir(root), symbol) is None)
    todo = [(n, m) for n, m in combos
            if overwrite or shard_io.find_shard(combo_dir(n, m, root), symbol) is None]
    if not need_ts and not todo:
        return {'symbol': symbol, 'skipped': True, 'combos_written': 0,
                'timestamp_written': False}

    df = sessions.add_intraday_coords(shard_io.load_shard(symbol, columns=C.FACTOR_FIELDS))
    codes, days = intraday.day_codes_of(df)
    info: dict = {'symbol': symbol, 'skipped': False, 'n_days': int(len(days)),
                  'n_bars': int(len(df)),
                  'first_day': str(days[0].date()) if len(days) else '',
                  'last_day': str(days[-1].date()) if len(days) else ''}

    if need_ts:
        ts = intraday.timestamp_factors(df)
        shard_io.save_shard(ts, timestamp_dir(root), symbol, fmt)
        info['n_timestamp_cols'] = int(ts.shape[1])
    info['timestamp_written'] = bool(need_ts)

    if todo:
        price = df['closew'].to_numpy(dtype='float64')
        grid = intraday.rolling_threshold_grid(
            intraday.intraday_abs_diff(price, codes), codes, lookbacks, pcts)
        for n, m in todo:
            dur = intraday.duration_factors(df, lookback=n, pct=m, thr_p=grid[(n, m)])
            shard_io.save_shard(dur, combo_dir(n, m, root), symbol, fmt)
            info['n_duration_cols'] = int(dur.shape[1])
    info['combos_written'] = len(todo)
    return info


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    """只取当前定义的因子列。旧版缓存里残留的列（已下线的因子）不进入下游。"""
    df = shard_io.read_frame(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{path} 缺少因子列 {missing}，请用 step3_build_factors.py --overwrite 重建")
    df = df[columns]
    df.index = pd.to_datetime(df.index)
    return df


def load_symbol(symbol: str, lookback: int, pct: float,
                root: Path | None = None,
                with_timestamp: bool = True) -> pd.DataFrame:
    """单品种在某个 (N, M) 下的日频因子表（持续期族 + 时间戳族 outer 对齐）。"""
    root = root or C.FACTOR_DAILY_DIR
    p = shard_io.find_shard(combo_dir(lookback, pct, root), symbol)
    if p is None:
        raise FileNotFoundError(
            f"缺少 {combo_name(lookback, pct)}/{symbol}，请先运行 step3_build_factors.py")
    out = _read(p, DURATION_FACTORS)
    if with_timestamp:
        q = shard_io.find_shard(timestamp_dir(root), symbol)
        if q is None:
            raise FileNotFoundError(f"缺少 timestamp/{symbol}")
        # outer：两族交易日理论上一致，一旦不一致能看见 NaN 而不是被静默截断
        out = out.join(_read(q, TIMEPOINT_FACTORS), how='outer')
    return out.sort_index()


def load_panel(lookback: int, pct: float,
               symbols: list[str] | None = None,
               root: Path | None = None) -> pd.DataFrame:
    """长表 ``(trading_date, symbol) -> 各因子``。"""
    root = root or C.FACTOR_DAILY_DIR
    syms = symbols or shard_io.list_shards(combo_dir(lookback, pct, root))
    parts = [load_symbol(s, lookback, pct, root).assign(symbol=s).set_index('symbol', append=True)
             for s in syms]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out.index.names = ['trading_date', 'symbol']
    return out


def load_wide(factor: str, lookback: int, pct: float,
              symbols: list[str] | None = None,
              root: Path | None = None) -> pd.DataFrame:
    """单个因子的宽表 ``index=trading_date, columns=symbol``。"""
    root = root or C.FACTOR_DAILY_DIR
    syms = symbols or shard_io.list_shards(combo_dir(lookback, pct, root))
    cols = {}
    for s in syms:
        df = load_symbol(s, lookback, pct, root)
        if factor not in df.columns:
            raise KeyError(f"{s} 没有因子列 {factor}；可用: {list(df.columns)}")
        cols[s] = df[factor]
    return pd.DataFrame(cols).sort_index()


# --------------------------------------------------------------------------
# 验收
# --------------------------------------------------------------------------
def panel_health(panel: pd.DataFrame) -> pd.DataFrame:
    """逐因子的非空率与取值范围，第 3 步验收依据。"""
    rows = []
    for c in panel.columns:
        s = panel[c]
        v = s.dropna()
        rows.append({
            'factor': c,
            'scope': '全部',
            'n': int(len(s)),
            'non_null_ratio': float(v.size / len(s)) if len(s) else np.nan,
            'all_zero': bool(v.size and (v == 0).all()),
            'nunique': int(v.nunique()),
            'min': float(v.min()) if v.size else np.nan,
            'p50': float(v.median()) if v.size else np.nan,
            'max': float(v.max()) if v.size else np.nan,
        })
    return pd.DataFrame(rows).set_index('factor')


def check_acceptance(health: pd.DataFrame,
                     min_non_null: float = 0.90) -> pd.DataFrame:
    """三条验收：非空率 > 门槛；不全为 0；时点类因子落在 [0, 1]。"""
    out = health.copy()
    out['pass_non_null'] = out['non_null_ratio'] > min_non_null
    out['pass_not_zero'] = ~out['all_zero']
    in01 = out.index.isin(TIMEPOINT_FACTORS)
    out['pass_range'] = True
    out.loc[in01, 'pass_range'] = (
        (out.loc[in01, 'min'] >= -1e-9) & (out.loc[in01, 'max'] <= 1 + 1e-9))
    out['passed'] = out['pass_non_null'] & out['pass_not_zero'] & out['pass_range']
    reasons = []
    for _, r in out.iterrows():
        bad = []
        if not r['pass_non_null']:
            bad.append(f"非空 {r['non_null_ratio']:.1%}")
        if not r['pass_not_zero']:
            bad.append('全为 0')
        if not r['pass_range']:
            bad.append(f"越界 [{r['min']:.3g}, {r['max']:.3g}]")
        reasons.append('+'.join(bad))
    out['reason'] = reasons
    return out


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')
