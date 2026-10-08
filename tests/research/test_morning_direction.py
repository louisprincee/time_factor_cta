"""开盘偏离方向判别：同板块共振比例、顺逆向规则、岭回归只用过去年份。"""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
direction = importlib.import_module('research_morning_direction')

RULE = {'dev_threshold': .002, 'clock': .6667, 'relvol_max': 1.5, 'max_est_cost': .0006}
BASE = {'rule': RULE, 'main': 'M', 'candidates': {'M': {'clock': 'none', 'night_aligned': False, 'volume': True}}}
SPEC = {'rules': {'量能分流': {'kind': 'relvol', 'follow_relvol': 1.5},
                  '信息票数': {'kind': 'votes', 'follow_relvol': 1.5, 'follow_share': .6667},
                  '主策略+信息跟随': {'kind': 'main_plus_votes', 'follow_relvol': 1.5, 'follow_share': .6667},
                  '岭回归': {'kind': 'ridge'}}}


def test_share_same_sign_excludes_self_and_needs_two_peers():
    sign = pd.Series([1., 1., -1., 1., 1.])
    groups = pd.Series(['a', 'a', 'a', 'a', 'b'])
    out = direction.share_same_sign(sign, groups)
    assert out.tolist() == pytest.approx([2 / 3, 2 / 3, 0., 2 / 3, .5])


def table(dev, relvol, peer=.5, eff=.1, eff_median=.5, pred=np.nan):
    n = len(dev)
    return pd.DataFrame({'dev15': dev, 'relvol': relvol, 'peer_share': peer, 'eff_pre': eff,
                         'eff_median': eff_median, 'est_cost': .0003, 'gross_1130': .001, 'cost_1130': .0003,
                         'member': True, 'sigma': .01, 'hclock_td': .5, 'lclock_td': .5,
                         'ret_night': .0, 'ridge_pred': pred}, index=range(n))


def test_relvol_rule_follows_loud_and_fades_quiet():
    t = table([.003, -.003, .003, .001], [2., 2., 1., 2.])
    side = direction.side_of(t, SPEC, BASE, '量能分流')
    assert side.tolist() == [1., -1., -1., 0.]


def test_votes_rule_needs_two_votes_to_follow_and_zero_to_fade():
    t = table([.003, .003, -.003], [2., 2., 1.], peer=[.9, .1, .1])
    side = direction.side_of(t, SPEC, BASE, '信息票数')
    assert side.tolist() == [1., 0., 1.]  # 2 票顺向，1 票不做，0 票逆向（下杀做多）


def test_main_plus_votes_keeps_main_fade_cell():
    t = table([.003, .003], [1., 2.], peer=[.9, .9])
    side = direction.side_of(t, SPEC, BASE, '主策略+信息跟随')
    assert side.tolist() == [-1., 1.]


def test_ridge_rule_trades_only_when_prediction_beats_cost():
    t = table([.003, .003, -.003], [1., 1., 1.])
    side = direction.side_of(t, SPEC, BASE, '岭回归', pd.Series([.001, .0001, -.001]))
    assert side.tolist() == [1., 0., 1.]  # 下杀且预测回吐：逆向做多


def test_ridge_predictions_ignore_same_and_later_year_labels():
    rng = np.random.default_rng(1)
    days = pd.bdate_range('2014-01-01', '2021-12-31')
    t = pd.DataFrame({'trading_date': days, 'dev15': .003, 'est_cost': .0003, 'cost_1130': .0003,
                      'f': rng.normal(size=len(days))})
    t['gross_1130'] = .001 * t.f + rng.normal(0, .001, len(days))
    t['cont'] = t.gross_1130
    spec = {'alpha': 1., 'first_train_year': 2014, 'clip_quantile': .01}
    pred, coefs = direction.ridge_predictions(t, ['f'], RULE, spec)
    changed = t.copy()
    later = changed.trading_date.dt.year >= 2019
    changed.loc[later, 'cont'] = -changed.loc[later, 'cont']
    again, _ = direction.ridge_predictions(changed, ['f'], RULE, spec)
    early = t.trading_date.dt.year.between(2016, 2019)
    pd.testing.assert_series_equal(pred[early], again[early])
    assert pred[t.trading_date.dt.year < 2016].isna().all()
    assert coefs.loc[2016, 'f'] > 0
