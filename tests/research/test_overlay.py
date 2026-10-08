"""时间因子叠加层：反向时否决或减半核心信号，同向或控制信号缺失时不干预。"""
import types

import numpy as np
import pandas as pd
import pytest

from tfcta.factors import library as L
from tfcta.research import study

DAYS = pd.bdate_range('2021-01-04', periods=4)


def data(core, control):
    signed = {'tsmom': pd.DataFrame({'A': core}, index=DAYS),
              'neg_clv': pd.DataFrame({'A': control}, index=DAYS)}
    return types.SimpleNamespace(factors=L.SignalSet(bars={}, signed=signed))


def spec(action, threshold=None):
    overlay = {'factors': {'neg_clv': 1}, 'action': action}
    if threshold is not None:
        overlay['threshold'] = threshold
    return {'id': 'x', 'factors': {'tsmom': 1}, 'overlay': overlay}


@pytest.mark.parametrize('action, keep', [('veto', 0.), ('halve', .5)])
def test_overlay_cuts_only_disagreeing_days(action, keep):
    d = data([.8, .8, -.6, -.6], [.3, -.3, .2, np.nan])
    out = study.signal_for(d, spec(action))['A'].tolist()
    assert out == pytest.approx([.8, .8 * keep, -.6 * keep, -.6])


def test_overlay_threshold_ignores_weak_disagreement():
    d = data([.8, .8, .8, .8], [-.2, -.6, .4, -1.])
    out = study.signal_for(d, spec('veto', .5))['A'].tolist()
    assert out == pytest.approx([.8, 0., .8, 0.])


def test_no_overlay_matches_plain_combine():
    d = data([.8, -.3, .1, 0.], [-1., -1., -1., -1.])
    plain = {'id': 'x', 'factors': {'tsmom': 1}}
    pd.testing.assert_frame_equal(study.signal_for(d, plain), d.factors.signed['tsmom'].clip(-1, 1))


@pytest.mark.parametrize('overlay', [{'factors': {'neg_clv': 1}, 'action': 'flip'},
                                     {'factors': {'neg_clv': 1}, 'action': 'veto', 'threshold': 1.},
                                     {'factors': {'no_such': 1}, 'action': 'veto'},
                                     {'factors': {}, 'action': 'veto'},
                                     {'factors': {'neg_clv': 1}, 'action': 'veto', 'smooth': 5}])
def test_validate_rejects_bad_overlay(overlay):
    with pytest.raises(ValueError):
        study.validate_specs([{'id': 'x', 'factors': {'tsmom': 1}, 'overlay': overlay}])


def test_validate_accepts_declared_overlay():
    study.validate_specs([spec('veto', .5), {**spec('halve'), 'id': 'y'}])
