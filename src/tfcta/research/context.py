"""独立运行目录及 JSON 序列化；不在数据输入目录保存实验成绩。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C


def run_dir(step: str) -> Path:
    d = C.RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{step}"
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
