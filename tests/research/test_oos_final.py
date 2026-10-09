"""样本外脚本：年份解析、台账去重、预热年不计入、元策略在接续表上的切换。不读真实样本外数据。"""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
final = importlib.import_module('oos_final')


def test_parse_years_only_allows_declared_years():
    assert final.parse_years('2024-2025', [2024, 2025]) == [2024, 2025]
    assert final.parse_years('2025，2024', [2024, 2025]) == [2024, 2025]
    with pytest.raises(ValueError):
        final.parse_years('2023', [2024, 2025])  # 2023 已被用过，只能当预热
    with pytest.raises(ValueError):
        final.parse_years('2025-2024', [2024, 2025])


def test_label_and_fingerprint():
    assert final.label_of([2024]) == '2024' and final.label_of([2024, 2025]) == '2024-2025'
    assert final.fingerprint({'a': '1'}, [2024]) != final.fingerprint({'a': '1'}, [2024, 2025])


def test_check_ledger_refuses_silent_rerun_and_reports_changed_configs():
    finger = final.fingerprint({'x.json': 'new'}, [2024])
    ledger = [{'kind': 'final_evaluation', 'strategy': 'S', 'status': 'result', 'fingerprint': finger,
               'run_at': 't', 'sha256': {'x.json': 'new'}},
              {'kind': 'final_evaluation', 'strategy': 'S', 'status': 'result', 'fingerprint': 'old',
               'run_at': 't0', 'sha256': {'x.json': 'old'}}]
    with pytest.raises(SystemExit):
        final.check_ledger(ledger, 'S', finger, {'x.json': 'new'}, rerun=False)
    assert final.check_ledger(ledger, 'S', finger, {'x.json': 'new'}, rerun=True) == ['x.json']
    assert final.check_ledger(ledger, 'other', finger, {'x.json': 'new'}, rerun=False) == []


def test_yearly_marks_warmup_year_as_not_scored():
    days = pd.bdate_range('2023-01-02', '2024-12-31')
    ret = pd.Series(.001, index=days)
    table = final.yearly({'A': ret}, [2023, 2024], [2024])
    assert table.set_index('年份').计入绩效.to_dict() == {2023: False, 2024: True}


def test_morning_books_switch_on_trailing_loss_across_partitions():
    """接续表里，前一段（预热）的亏损单会让下一段的单反手；当天的单不影响当天。"""
    days = pd.to_datetime(['2022-12-29', '2022-12-30', '2023-01-03'])
    table = pd.DataFrame({'trading_date': days, 'symbol': 'A', 'dev15': .003, 'gross_1130': [.01, .01, .02],
                          'cost_1130': 0.})
    base_side = pd.Series([-1., -1., -1.])  # 主策略：上冲做空，前两笔都亏
    original = final.oor.base_books
    final.oor.base_books = lambda t, base: {'主策略': base_side}
    try:
        books = final.morning_books(table, {}, {'book': '主策略', 'mode': '亏损反手', 'window_trades': 2})
    finally:
        final.oor.base_books = original
    assert books['主策略·原样'].tolist() == [-1., -1., -1.]
    assert books['早盘元策略'].tolist() == [-1., -1., 1.]  # 第三天才看到前两笔亏损
