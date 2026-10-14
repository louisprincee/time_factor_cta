"""样本外脚本：年份解析、台账去重、预热年不计入、元策略在接续表上的切换。不读真实样本外数据。"""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts' / 'commodity'))
final = importlib.import_module('oos_final')


def test_parse_years_allows_each_declared_oos_year():
    allowed = json.loads(final.PROTOCOL.read_text(encoding='utf-8'))['years_allowed']
    assert allowed == [2023, 2024, 2025]
    assert final.parse_years('2023', allowed) == [2023]
    assert final.parse_years('2023-2025', allowed) == [2023, 2024, 2025]
    assert final.parse_years('2025，2024', allowed) == [2024, 2025]
    with pytest.raises(ValueError):
        final.parse_years('2026', allowed)
    with pytest.raises(ValueError):
        final.parse_years('2025-2024', allowed)


def test_label_and_fingerprint():
    assert final.label_of([2024]) == '2024' and final.label_of([2024, 2025]) == '2024-2025'
    assert final.fingerprint({'a': '1'}, [2024]) != final.fingerprint({'a': '1'}, [2024, 2025])


def test_check_ledger_refuses_any_retest_without_rerun_and_lists_changed_files():
    ledger = [{'kind': 'final_evaluation', 'strategy': 'S', 'status': 'result', 'fingerprint': 'f0',
               'run_at': 't0', 'sha256': {'a.json': 'old', 'gone.py': 'x'}},
              {'kind': 'final_evaluation', 'strategy': 'S', 'status': 'opened', 'fingerprint': 'f1'}]
    now = {'a.json': 'new', 'src/lib.py': 'y'}
    with pytest.raises(SystemExit):  # 年份或代码变了也是复测
        final.check_ledger(ledger, 'S', 'another', now, rerun=False)
    with pytest.raises(SystemExit):  # 旧名字下跑过也算
        final.check_ledger(ledger, 'S2', 'another', now, rerun=False, previous=['S'])
    assert final.check_ledger(ledger, 'S', 'another', now, rerun=True) == ['a.json', 'gone.py', 'src/lib.py']
    assert final.check_ledger(ledger, 'other', 'f0', now, rerun=False) == []


def test_hashes_cover_library_code_and_cost_data():
    for path in ('src/tfcta/research.py', 'src/tfcta/data.py', 'src/tfcta/config.py',
                 'data/commodity/research/tick_size.csv', 'data/commodity/oos/fee_history.csv'):
        assert path in final.HASHED


def test_ledger_keys_keep_same_named_files_apart():
    keys = {final.C._ledger_key(final.ROOT / p) for p in ('data/commodity/research/fee_history.csv', 'data/commodity/oos/fee_history.csv')}
    assert keys == {'data/commodity/research/fee_history.csv', 'data/commodity/oos/fee_history.csv'}


def test_yearly_only_reports_the_years_asked_for():
    days = pd.bdate_range('2023-01-02', '2024-12-31')
    ret = pd.Series(.001, index=days)
    table = final.yearly({'A': ret}, [2024], [2024])
    assert table.年份.tolist() == [2024] and table.计入绩效.all()


def test_morning_books_switch_on_trailing_loss_across_partitions():
    """接续表里，前一段（预热）的亏损单会让下一段的单反手；当天的单不影响当天。"""
    days = pd.to_datetime(['2022-12-29', '2022-12-30', '2023-01-03'])
    table = pd.DataFrame({'trading_date': days, 'symbol': 'A', 'dev15': .003, 'gross_1130': [.01, .01, .02],
                          'cost_1130': 0.})
    base_side = pd.Series([-1., -1., -1.])  # 主策略：上冲做空，前两笔都亏
    original = final.morning.base_books
    final.morning.base_books = lambda t, base: {'主策略': base_side}
    try:
        books = final.morning_books(table, {}, {'book': '主策略', 'mode': '亏损反手', 'window_trades': 2})
    finally:
        final.morning.base_books = original
    assert books['主策略·原样'].tolist() == [-1., -1., -1.]
    assert books['早盘元策略'].tolist() == [-1., -1., 1.]  # 第三天才看到前两笔亏损


def test_rebuild_runs_end_to_end_on_a_tiny_table(monkeypatch):
    """rebuild 里不能有局部变量遮住 morning 模块（2026-10-12 修过一次）。"""
    days = pd.to_datetime(['2021-12-30', '2021-12-31', '2022-01-04'])
    table = pd.DataFrame({'trading_date': days, 'symbol': 'A', 'dev15': .003, 'gross_1130': [.01, .01, .02],
                          'cost_1130': 0., 'tick_frac': .0001, 'sigma': .01, 'sector': '黑色'})
    monkeypatch.setattr(final.morning, 'prepare_through_2022', lambda base, first: table.copy())
    monkeypatch.setattr(final.morning, 'base_books', lambda t, base: {'主策略': pd.Series(-1., index=t.index)})
    base = {'sizing': {'risk_per_trade': .01, 'max_weight': 2., 'max_sector_gross': 3., 'max_gross': 4.}}
    _, books, series = final.rebuild(base, {'book': '主策略', 'mode': '亏损反手', 'window_trades': 2})
    assert set(series) == set(final.TICKS) and set(series['主口径']) == {'早盘元策略', '主策略·原样'}
    assert books['早盘元策略'].tolist() == [-1., -1., 1.]
