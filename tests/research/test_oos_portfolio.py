"""样本外最终评估：入口门槛、台账、截止日、年份解析与熔断重置。不读任何真实样本外数据。"""
import datetime as dt
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from tfcta import config as C
from tfcta import data as data_module
from tfcta.factors import external
from tfcta.research import costs

oos = importlib.import_module('oos_portfolio')
portfolio = importlib.import_module('research_portfolio')
oor = importlib.import_module('research_morning_oor')

FROZEN = {'chosen': '核心1/2', 'frozen_at': '2026-10-08'}
LATER = dt.date(2026, 10, 8)


def opened(tmp_path, end='2023-12-31', spec=FROZEN, today=LATER):
    cfg = tmp_path / 'cfg.json'
    cfg.write_text('{}', encoding='utf-8')
    ledger = tmp_path / 'ledger.jsonl'
    return C.final_evaluation(spec, [cfg], end, '测试', today=today, ledger=ledger), ledger


@pytest.mark.parametrize('spec', [{'chosen': None, 'frozen_at': '2026-10-08'}, {'chosen': '核心1/2'}])
def test_refuses_unfrozen_config(tmp_path, spec):
    ctx, ledger = opened(tmp_path, spec=spec)
    with pytest.raises(C.HoldoutViolation, match='冻结'):
        ctx.__enter__()
    assert not ledger.exists()
    with pytest.raises(C.HoldoutViolation):
        C.assert_oos_research_locked()


@pytest.mark.parametrize('end', ['2022-12-31', '2026-06-30'])
def test_refuses_end_outside_sharded_window(tmp_path, end):
    ctx, _ = opened(tmp_path, end=end)
    with pytest.raises(C.HoldoutViolation, match='截止日'):
        ctx.__enter__()


def test_refuses_window_not_closed(tmp_path):
    ctx, _ = opened(tmp_path, end='2025-12-31', today=dt.date(2025, 12, 31))
    with pytest.raises(C.HoldoutViolation, match='尚未结束'):
        ctx.__enter__()


def test_writes_ledger_before_unlocking_and_relocks(tmp_path):
    ctx, ledger = opened(tmp_path)
    with ctx as ticket:
        rows = C.read_ledger(ledger)
        assert rows[-1]['status'] == 'opened' and rows[-1]['oos_end'] == '2023-12-31'
        assert ticket['sha256']['cfg.json'] == C._sha256(tmp_path / 'cfg.json')
        C.assert_oos_research_locked('2023-06-30')
        assert C.final_evaluation_end() == dt.date(2023, 12, 31)
        with pytest.raises(C.HoldoutViolation, match='越过'):
            C.assert_oos_research_locked('2024-01-02')
        with pytest.raises(C.HoldoutViolation, match='已有'):
            opened(tmp_path)[0].__enter__()
    with pytest.raises(C.HoldoutViolation):
        C.assert_oos_research_locked()


def test_relocks_after_exception(tmp_path):
    ctx, _ = opened(tmp_path)
    with pytest.raises(ValueError):
        with ctx:
            raise ValueError('中途失败')
    with pytest.raises(C.HoldoutViolation):
        C.final_evaluation_end()


def test_oos_readers_refuse_outside_evaluation():
    with pytest.raises(C.HoldoutViolation):
        data_module.load_oos_shard('RB', end='2023-12-31', today=LATER)
    with pytest.raises(C.HoldoutViolation):
        costs.load_fees('oos')
    with pytest.raises(C.HoldoutViolation):
        external.guard_dates(pd.DatetimeIndex(['2023-01-03']), 'oos')
    with pytest.raises(C.HoldoutViolation):
        external.load_panel(['RB'], 'oos')
    with pytest.raises(C.HoldoutViolation):
        oor.prepare('oos', {}, years=(2023, 2023), pools={})


def test_oos_shard_refuses_read_past_registered_end(tmp_path):
    ctx, _ = opened(tmp_path, end='2023-12-31')
    with ctx:
        with pytest.raises(C.HoldoutViolation, match='越过'):
            data_module.load_oos_shard('RB', end='2024-12-31', today=LATER)


@pytest.mark.parametrize('text, years', [('2024', [2024]), ('2023-2025', [2023, 2024, 2025]),
                                         ('2025,2023', [2023, 2025]), ('2023，2024', [2023, 2024])])
def test_parse_years(text, years):
    assert oos.parse_years(text, [2023, 2024, 2025]) == years


@pytest.mark.parametrize('text', ['2022', '2023-2026', '2025-2023', ''])
def test_parse_years_rejects(text):
    with pytest.raises(ValueError):
        oos.parse_years(text, [2023, 2024, 2025])


def test_block_starts_and_labels():
    assert oos.block_starts([2023, 2024, 2025]) == [pd.Timestamp('2023-01-01')]
    assert oos.block_starts([2023, 2025]) == [pd.Timestamp('2023-01-01'), pd.Timestamp('2025-01-01')]
    assert oos.label_of([2024]) == '2024'
    assert oos.label_of([2023, 2024, 2025]) == '2023-2025'
    assert oos.label_of([2023, 2025]) == '2023_2025'


def test_guard_without_resets_is_plain_breaker():
    days = pd.bdate_range('2021-01-01', periods=600)
    ret = pd.Series(np.random.default_rng(3).normal(-.001, .01, len(days)), index=days)
    pd.testing.assert_series_equal(portfolio.guard(ret, .08), oor.breaker(ret, .08))


def test_guard_reset_restarts_paper_high_from_capital():
    days = pd.bdate_range('2022-01-03', periods=30)  # 停手期间账面照常累计，窗口短到账面回不到阈值内
    ret = pd.Series(.0, index=days)
    ret.iloc[:5] = -.05                      # 先跌穿 8%，续接口径之后一直停手
    ret.iloc[20:] = .01
    cut = days[20]
    plain = portfolio.guard(ret, .08)
    reset = portfolio.guard(ret, .08, [cut])
    assert (plain.iloc[20:] == 0).all()
    pd.testing.assert_series_equal(reset.iloc[:20], plain.iloc[:20])
    assert (reset.iloc[20:] == .01).all()
    assert reset.index.equals(ret.index)


def test_ledger_requires_rerun_for_same_fingerprint():
    hashes = {'portfolio.json': 'a', 'oos_portfolio.py': 'b'}
    finger = oos.fingerprint(hashes, [2023])
    ledger = [{'kind': 'final_evaluation', 'strategy': 'S', 'status': 'result', 'fingerprint': finger,
               'run_at': 'x', 'out': 'y', 'sha256': hashes}]
    with pytest.raises(SystemExit, match='--rerun'):
        oos.check_ledger(ledger, 'S', finger, hashes, rerun=False)
    assert oos.check_ledger(ledger, 'S', finger, hashes, rerun=True) == []
    assert oos.check_ledger(ledger, '别的策略', finger, hashes, rerun=False) == []
    other = {'portfolio.json': 'c', 'oos_portfolio.py': 'b'}
    assert oos.check_ledger(ledger, 'S', oos.fingerprint(other, [2024]), other, False) == ['portfolio.json']
    assert oos.fingerprint(hashes, [2023]) != oos.fingerprint(hashes, [2023, 2024])
