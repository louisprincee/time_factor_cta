"""2022 验证分片导入：源文件必须本身只含 2022，复权基准必须和研究期衔接，已有分片不覆盖。"""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import shard_io
from tfcta.data import validation_import as VI


def _bars(days: pd.DatetimeIndex, offset: float, start: float = 100.0) -> pd.DataFrame:
    """每个交易日三根日盘 bar；closew = close + offset（加法复权偏移不变）。"""
    idx = pd.DatetimeIndex([pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=9, minute=m + 1)
                            for d in days for m in range(3)])
    close = start + np.arange(len(idx), dtype=float)
    return pd.DataFrame({
        'closew': close + offset, 'close': close, 'highw': close + offset + 1,
        'loww': close + offset - 1, 'volume': 1.0, 'total_turnover': 1.0,
        'open': close, 'openw': close + offset,
        'trading_date': [d for d in days for _ in range(3)],
    }, index=idx)


def _panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, axis=1)


def test_source_with_any_2023_row_is_refused_whole():
    days = pd.bdate_range('2022-12-26', '2023-01-03')
    problems = VI.source_window_problems(_panel({'RB': _bars(days, 5.0)}))
    assert problems and any('2023' in p for p in problems)


def test_source_with_2021_trading_date_is_refused():
    days = pd.bdate_range('2021-12-30', '2022-01-05')
    assert VI.source_window_problems(_panel({'RB': _bars(days, 5.0)}))


def test_2022_only_source_is_accepted():
    days = pd.bdate_range('2022-01-04', '2022-12-30')
    assert VI.source_window_problems(_panel({'RB': _bars(days, 5.0)})) == []


def test_boundary_offset_must_carry_over():
    research = _bars(pd.bdate_range('2021-12-27', '2021-12-31'), 5.0)
    same = _bars(pd.bdate_range('2022-01-04', '2022-01-07'), 5.0)
    shifted = _bars(pd.bdate_range('2022-01-04', '2022-01-07'), 50.0)
    assert VI.boundary_check(research, same)[0] == 'ok'
    assert VI.boundary_check(research, shifted)[0] == 'mismatch'
    assert VI.boundary_check(None, same)[0] == 'skip'
    assert VI.boundary_check(research, research)[0] == 'fail'


def test_boundary_verdict_tolerates_a_few_rolls_but_not_a_new_base():
    few = {f"S{i}": 'ok' for i in range(9)} | {'X': 'mismatch'}
    many = {f"S{i}": 'ok' for i in range(5)} | {f"X{i}": 'mismatch' for i in range(5)}
    assert VI.boundary_verdict(few)[0]
    assert not VI.boundary_verdict(many)[0]
    assert not VI.boundary_verdict({'A': 'ok', 'B': 'fail'})[0]


@pytest.fixture
def step1(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "step1", Path(__file__).resolve().parents[2] / "scripts" / "step1_shard_minutes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "shards"
    monkeypatch.setattr(C, "RESEARCH_DIR", root / "research")
    monkeypatch.setattr(C, "VALIDATION_DIR", root / "validation_2022")
    monkeypatch.setattr(C, "HOLDOUT_DIR", root / "holdout_locked")
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path / "runs")
    for sym in ('RB', 'CU'):
        shard_io.save_shard(_bars(pd.bdate_range('2021-12-01', '2021-12-31'), 5.0),
                            C.RESEARCH_DIR, sym, fmt='pickle')
    return module


def _run(step1, tmp_path, panel, *extra):
    src = tmp_path / "src_2022.pkl"
    with open(src, 'wb') as f:
        pickle.dump(panel, f)
    import sys
    argv = ["step1", "--validation-source", str(src), "--symbols", "RB", "CU",
            "--format", "pickle", *extra]
    old, sys.argv = sys.argv, argv
    try:
        return step1.main()
    finally:
        sys.argv = old


def test_step1_writes_validation_shards_once(step1, tmp_path):
    days = pd.bdate_range('2022-01-04', '2022-03-31')
    panel = _panel({s: _bars(days, 5.0, start=131.0) for s in ('RB', 'CU')})
    assert _run(step1, tmp_path, panel) == 0
    assert shard_io.list_shards(C.VALIDATION_DIR) == ['CU', 'RB']
    loaded = shard_io.load_validation_shard('RB')
    assert loaded['trading_date'].min() == pd.Timestamp('2022-01-04')
    # 第二次不覆盖
    assert _run(step1, tmp_path, panel) == 2


def test_step1_refuses_mixed_source_and_writes_nothing(step1, tmp_path):
    days = pd.bdate_range('2022-12-01', '2023-01-06')
    panel = _panel({s: _bars(days, 5.0) for s in ('RB', 'CU')})
    assert _run(step1, tmp_path, panel) == 1
    assert shard_io.list_shards(C.VALIDATION_DIR) == []


def test_step1_refuses_rebased_adjustment(step1, tmp_path):
    days = pd.bdate_range('2022-01-04', '2022-03-31')
    panel = _panel({s: _bars(days, 80.0, start=131.0) for s in ('RB', 'CU')})
    assert _run(step1, tmp_path, panel) == 1
    assert shard_io.list_shards(C.VALIDATION_DIR) == []


def test_step1_slices_2022_from_full_monolith(step1, tmp_path, monkeypatch):
    """全样本单体文件按 trading_date 取 2022：研究期和 2023 的行都不落盘。"""
    days = pd.bdate_range('2021-12-01', '2023-02-28')
    panel = _panel({s: _bars(days, 5.0) for s in ('RB', 'CU')})
    src = tmp_path / "monolith.pkl"
    with open(src, 'wb') as f:
        pickle.dump(panel, f)
    import sys
    argv = ["step1", "--validation-from-monolith", "--monolith", str(src),
            "--symbols", "RB", "CU", "--format", "pickle"]
    monkeypatch.setattr(sys, "argv", argv)
    assert step1.main() == 0
    loaded = shard_io.load_validation_shard('RB')
    assert loaded['trading_date'].min() == pd.Timestamp('2022-01-03')
    assert loaded['trading_date'].max() == pd.Timestamp('2022-12-30')
    assert loaded.index.max() < pd.Timestamp('2023-01-01')
