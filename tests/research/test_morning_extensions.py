"""元策略账面：只用已平仓的历史单，不读行情。"""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
meta = importlib.import_module('research_morning_meta')


def trades(days, nets, symbols=None):
    n = len(days)
    return pd.DataFrame({'trading_date': pd.to_datetime(days), 'symbol': symbols or ['A'] * n,
                         'gross_1130': nets, 'cost_1130': 0.})


def test_trailing_mean_uses_only_previous_days():
    t = trades(['2021-01-04', '2021-01-04', '2021-01-05', '2021-01-06'], [.01, .03, -.10, .0], ['A', 'B', 'A', 'A'])
    side = pd.Series(1., index=t.index)
    score = meta.trailing_mean(t, side, 2)
    assert np.isnan(score.iloc[0]) and np.isnan(score.iloc[1])        # 第一天没有历史
    assert score.iloc[2] == pytest.approx(.02)                       # 只用前一天的两笔
    assert score.iloc[3] == pytest.approx((.03 - .10) / 2)           # 当天的 −0.10 次日才进入
    changed = t.copy()
    changed.loc[3, 'gross_1130'] = 9.
    assert meta.trailing_mean(changed, side, 2).iloc[3] == score.iloc[3]


def test_switch_modes():
    side = pd.Series([1., -1., 1.])
    score = pd.Series([.01, -.01, np.nan])
    assert meta.switched(side, score, '亏损停手').tolist() == [1., 0., 1.]
    assert meta.switched(side, score, '亏损反手').tolist() == [1., 1., 1.]
    assert meta.switched(side, score, '原样').tolist() == [1., -1., 1.]


def test_trailing_days_mean_uses_only_previous_days():
    grid = importlib.import_module('research_meta_grid')
    t = trades(['2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07'], [.01, .03, -.10, .0])
    side = pd.Series(1., index=t.index)
    score = grid.trailing_days_mean(t, side, 2)
    assert np.isnan(score.iloc[0]) and np.isnan(score.iloc[1])
    assert score.iloc[2] == pytest.approx(.02)
    assert score.iloc[3] == pytest.approx(-.035)
