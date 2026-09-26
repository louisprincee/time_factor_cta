"""各步脚本共用的上下文：运行留痕目录、JSON 落盘、研究期品种池与日收益。

runs/{时间}_stepN/ 是每次运行的留痕；data/research/ 下是下游接着读的最新一份。
分片或品种池还没落盘时，这里给出原因，由脚本以退出码 2 停下。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config as C
from ...data import bars as B
from ...data import shard_io
from ...data import universe as U


def run_dir(step: str) -> Path:
    d = C.RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{step}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_json(o):
    """numpy 标量转 Python，NaN/inf 转 None。"""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2), encoding='utf-8')


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def not_ready_reason() -> str | None:
    if not shard_io.list_shards(C.RESEARCH_DIR):
        return (f"研究期分片为空: {C.RESEARCH_DIR}\n"
                "数据还在拉取时这是预期状态。分片完成并跑完 step1 之后直接重跑本脚本。")
    uni = C.UNIVERSE_DIR / 'universe_by_year.json'
    if not uni.exists():
        return f"找不到品种池: {uni}\n请先运行 step2_universe.py。"
    return None


def load_context(explicit: list[str] | None = None) -> tuple[dict, list[str], pd.DataFrame]:
    """品种池、实际有分片的品种、研究期日收益宽表。"""
    universe = U.load_universe()
    have = set(shard_io.list_shards(C.RESEARCH_DIR))
    if explicit:
        symbols = [s for s in explicit if s in have]
    else:
        symbols = sorted({s for vs in universe.values() for s in vs if s in have})
    day_ret = B.load_day_returns(symbols)
    C.assert_no_holdout_dates(day_ret.index, what='日收益')
    return universe, symbols, day_ret
