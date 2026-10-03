"""2022 验证分片的导入检查：源文件必须本身就只含 2022，不从混合文件里过滤。

规矩一要求 2022 验证数据来自独立的 2022-only 数据源。这里只做"拒收"而不做"过滤"：
源文件里只要有一个 2022 年以外的交易日或 2023 年以后的时间戳，整份文件都不收。

加法复权价（``closew`` 等）在单独导出时可能换了基准。研究期和验证期首尾相接时，
``closew.diff()`` 跨过边界时如果基准不同就会凭空多出一笔收益，所以还要核对边界处的复权偏移。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .. import config as C

# 2021-12-31（周五）夜盘归属 2022-01-04，是验证期第一根 bar 可能出现的最早墙钟日期
EARLIEST_WALL_CLOCK = dt.date(2021, 12, 31)
OFFSET_RTOL = 1e-6
MAX_MISMATCH_SHARE = 0.2      # 边界上同时换月的品种不会超过两成


def source_window_problems(panel: pd.DataFrame) -> list[str]:
    """宽面板（列为 品种 × 字段）是否只含 2022。返回问题列表，空表示可以收。"""
    problems = []
    idx = pd.DatetimeIndex(panel.index)
    if len(idx) == 0:
        return ["源文件没有任何行"]
    lo, hi = pd.Timestamp(EARLIEST_WALL_CLOCK), pd.Timestamp(C.STRICT_OOS_START)
    if idx.min() < lo or idx.max() >= hi:
        problems.append(f"时间戳范围 {idx.min()}..{idx.max()} 超出 [{lo.date()}, {hi.date()})")
    if 'trading_date' not in panel.columns.get_level_values(1):
        problems.append("源文件没有 trading_date 字段")
        return problems
    td = panel.xs('trading_date', axis=1, level=1)
    dates = pd.to_datetime(pd.Series(td.to_numpy().ravel())).dropna()
    if dates.empty:
        problems.append("trading_date 全部缺失")
        return problems
    first, last = dates.min(), dates.max()
    if first < pd.Timestamp(C.HOLDOUT_START) or last >= hi:
        problems.append(f"trading_date 范围 {first.date()}..{last.date()} 不全在 {C.VALIDATION_YEAR} 年")
    return problems


def adjustment_offset(df: pd.DataFrame, last: bool) -> float | None:
    """``closew − close``：加法复权相对原始价的偏移，取最后（或最先）一根两者都有的 bar。"""
    both = df[['closew', 'close']].dropna()
    if both.empty:
        return None
    row = both.iloc[-1] if last else both.iloc[0]
    return float(row['closew'] - row['close'])


def boundary_check(research: pd.DataFrame | None,
                   validation: pd.DataFrame) -> tuple[str, str]:
    """研究期末与验证期初的复权偏移是否衔接。返回 (状态, 说明)，状态为 ok / mismatch / fail / skip。

    同一主力合约下，加法复权偏移在相邻两根 bar 之间不变。单个品种不等可能只是边界上恰好换月
    （换月表含 2022 年以后的日期，这里不去读它），所以单品种记 mismatch，由 ``boundary_verdict`` 汇总。
    """
    if research is None or research.empty:
        return 'skip', "研究期没有分片（2022 年才有数据），无需衔接"
    tail = pd.DatetimeIndex(research.index).max()
    head = pd.DatetimeIndex(validation.index).min()
    if head <= tail:
        return 'fail', f"验证期首个时间戳 {head} 不晚于研究期末 {tail}"
    a, b = adjustment_offset(research, last=True), adjustment_offset(validation, last=False)
    if a is None or b is None:
        return 'skip', "边界处缺 close/closew，核对不了复权偏移"
    if np.isclose(a, b, rtol=OFFSET_RTOL, atol=1e-6):
        return 'ok', f"复权偏移 {a:.4f} 衔接"
    return 'mismatch', f"复权偏移 {a:.4f} → {b:.4f}：边界换月，或复权基准不同"


def boundary_verdict(statuses: dict[str, str]) -> tuple[bool, str]:
    """跨品种汇总：个别品种不衔接可以是边界换月；超过两成不衔接只能是整份数据换了复权基准。"""
    if any(s == 'fail' for s in statuses.values()):
        bad = sorted(k for k, s in statuses.items() if s == 'fail')
        return False, f"时间戳与研究期重叠: {bad}"
    checked = {k: s for k, s in statuses.items() if s in ('ok', 'mismatch')}
    off = sorted(k for k, s in checked.items() if s == 'mismatch')
    if checked and len(off) > MAX_MISMATCH_SHARE * len(checked):
        return False, (f"{len(off)}/{len(checked)} 个品种复权偏移不衔接，"
                       f"源文件的复权基准与研究期不同: {off}")
    note = f"复权偏移衔接 {len(checked) - len(off)}/{len(checked)}"
    return True, note + (f"；不衔接（按边界换月处理，请核对）: {off}" if off else "")
