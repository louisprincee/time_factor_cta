"""第 5-11 步的落盘位置，以及各脚本共用的读取。

runs/{时间}_stepN/ 是每次运行的留痕。data/research/ 下是下游接着读的最新一份。
分片或品种池还没落盘时，这里给出原因，由脚本以退出码 2 停下。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..data import shard_io
from ..data import universe as U
from ..factors import factor_cache as FC
from . import costs
from . import panel as returns


def scan_dir() -> Path:
    return C.RESEARCH_OUT_DIR / 'scan'


def selection_path() -> Path:
    return C.RESEARCH_OUT_DIR / 'selection.json'


def ic_path() -> Path:
    return C.RESEARCH_OUT_DIR / 'ic_by_fold.csv'


def combo_path() -> Path:
    return C.RESEARCH_OUT_DIR / 'combo_metrics.csv'


def fee_path() -> Path:
    return C.RESEARCH_OUT_DIR / 'fee_sensitivity.csv'


def backward_path() -> Path:
    return C.RESEARCH_OUT_DIR / 'backward_ic.csv'


def frozen_path() -> Path:
    return C.CONFIG_DIR / 'frozen_config.yaml'


def run_dir(step: str) -> Path:
    d = C.RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{step}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=2),
                    encoding='utf-8')


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
    """品种池、实际有分片的品种、日收益宽表。"""
    universe = U.load_universe()
    have = set(shard_io.list_shards(C.RESEARCH_DIR))
    if explicit:
        symbols = [s for s in explicit if s in have]
    else:
        symbols = sorted({s for vs in universe.values() for s in vs if s in have})
    day_ret = returns.load_day_returns(symbols)
    C.assert_no_holdout_dates(day_ret.index, what='日收益')
    return universe, symbols, day_ret


def load_factor(name: str, lookback: int, pct: float,
                symbols: list[str]) -> pd.DataFrame:
    return FC.load_wide(name, int(lookback), float(pct), symbols)


def load_slippage(symbols: list[str],
                  index: pd.Index,
                  n_ticks: float,
                  rebuild: bool = False):
    """滑点宽表 + 一行人话说明。第 5 步和第 6 步共用这一条路径。

    ``n_ticks == 0`` 时不去建 tick 表（那要扫一遍分钟数据），直接返回 None 让
    ``run_book`` 走"只有手续费"的老路——第 6 步的 0 档对照就靠这个。
    """
    if float(n_ticks) == 0.0:
        return None, '滑点 0 个 tick（对照档，仅供比较，不得用于选参或结论）'
    tab = costs.load_tick_table(symbols, rebuild=rebuild)
    wide = costs.slippage_wide(tab, index, list(symbols), n_ticks)
    bp = costs.cost_summary(tab, n_ticks)['bp_per_turnover']
    note = (f"滑点 {float(n_ticks):g} 个 tick，按换手计费；"
            f"品种间 {bp.min():.1f}~{bp.max():.1f}bp，中位 {bp.median():.1f}bp")
    return wide, note


def parse_bands(items: list[str] | None) -> list[tuple[float, float]]:
    if not items:
        return [(float(a), float(b)) for a, b in C.SIGNAL_BANDS]
    out = []
    for s in items:
        if ':' not in s:
            raise ValueError(f"分位轨应为 低:高，收到 {s}")
        a, b = s.split(':', 1)
        out.append((float(a), float(b)))
    return out


def parse_combos(items: list[str] | None,
                 default_names: list[str]) -> dict[str, list[str]]:
    """``NAME=a,b,c``；省略时用 default_names 里的 COMBO_GROUPS。"""
    if not items:
        return {k: list(C.COMBO_GROUPS[k]) for k in default_names}
    out = {}
    for s in items:
        if '=' not in s:
            raise ValueError(f"组合应为 NAME=f1,f2，收到 {s}")
        name, members = s.split('=', 1)
        out[name] = [m for m in members.split(',') if m]
    return out
