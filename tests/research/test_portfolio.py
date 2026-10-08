"""日频核心 + 早盘卫星组合：缩放乘数因果性、风险份额、选择规则与 2022 冻结门槛。"""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
portfolio = importlib.import_module('research_portfolio')


def test_multiplier_uses_only_returns_known_before_decision():
    days = pd.bdate_range('2020-01-01', periods=200)
    ret = pd.Series(np.random.default_rng(1).normal(0, .01, len(days)), index=days)
    base = portfolio.multiplier(ret, .1, 120, 60, 2, 4.)
    shocked = ret.copy()
    shocked.iloc[150:] *= 10  # 第 150 日起的收益改变，不能影响第 151 日及以前的乘数
    again = portfolio.multiplier(shocked, .1, 120, 60, 2, 4.)
    pd.testing.assert_series_equal(base.iloc[:152], again.iloc[:152])
    assert not np.isclose(base.iloc[152], again.iloc[152])


def test_multiplier_is_flat_before_enough_history_and_capped():
    days = pd.bdate_range('2020-01-01', periods=100)
    ret = pd.Series(np.random.default_rng(2).normal(0, 1e-5, len(days)), index=days)
    m = portfolio.multiplier(ret, .1, 120, 60, 1, 4.)
    assert (m.iloc[:60] == 0).all()
    assert (m.iloc[60:] == 4.).all()


@pytest.mark.parametrize('share', [0., .5, .6667, .75, 1.])
def test_leg_targets_split_total_risk(share):
    core, sat = portfolio.leg_targets(share, .1)
    assert np.isclose(core ** 2 + sat ** 2, .1 ** 2)
    if 0 < share < 1:
        assert np.isclose(core / sat, share / (1 - share))


def test_selection_ignores_reference_rows():
    table = pd.DataFrame({'方案': ['核心1/2', '核心2/3', '只卫星'], '夏普95%下限': [.5, .7, 2.]})
    assert portfolio.select(table, {'核心1/2': .5, '核心2/3': .6667}) == '核心2/3'


def test_validation_refuses_without_frozen_choice(tmp_path, monkeypatch):
    config = tmp_path / 'portfolio.json'
    text = portfolio.CONFIG.read_text(encoding='utf-8')
    spec = json.loads(text)
    spec['chosen'] = None
    config.write_text(json.dumps(spec, ensure_ascii=False), encoding='utf-8')
    monkeypatch.setattr(portfolio, 'CONFIG', config)
    monkeypatch.setattr(sys, 'argv', ['research_portfolio.py', '--validation-2022'])
    with pytest.raises(SystemExit, match='chosen'):
        portfolio.main()
