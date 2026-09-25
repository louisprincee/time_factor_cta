"""因子族测试。

重点不是"跑得通"，而是三件容易静默出错的事：
1. 无夜盘品种的夜盘因子必须是 NaN 而不是 0（上一轮小时频踩过的坑，把 t=3.05 的
   真因子压成了 t=1.1）
2. DFP 必须已按收盘价归一化，否则跨品种量纲差几个数量级
3. 所有时点类因子必须落在 [0,1]，否则等权合成被长夜盘品种主导
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import sessions, synth
from tfcta.factors import factors


DAYS = pd.bdate_range('2015-01-01', '2015-12-31')
LOOKBACK, PCT = 60, 55.0


def _prep(night_class='night_2300', seed=5, days=DAYS):
    df = synth.make_symbol(days, night_class=night_class, seed=seed)
    return sessions.add_intraday_coords(df)


def test_dfp_uses_longest_duration_bar_and_is_normalized():
    dur = np.array([1.0, 50.0, 2.0, 3.0])
    price = np.array([100.0, 120.0, 110.0, 105.0])
    codes = np.zeros(4, int)
    days = pd.DatetimeIndex(['2015-01-01'])
    out = factors.dfp_factors(dur, price, codes, days, top_ns=[1])
    # FP = 持续期最大(50)那根的价格 120；Close = 最后一根 105
    assert out['dfp_max'].iloc[0] == pytest.approx((120.0 - 105.0) / 105.0)


def test_dfp_is_scale_invariant():
    """把价格整体放大 10 倍，归一化后的 DFP 必须不变——这正是跨品种可比的要求。"""
    dur = np.array([1.0, 50.0, 2.0, 3.0])
    price = np.array([100.0, 120.0, 110.0, 105.0])
    codes, days = np.zeros(4, int), pd.DatetimeIndex(['2015-01-01'])
    a = factors.dfp_factors(dur, price, codes, days, top_ns=[1])['dfp_max'].iloc[0]
    b = factors.dfp_factors(dur, price * 10, codes, days, top_ns=[1])['dfp_max'].iloc[0]
    assert a == pytest.approx(b)


def test_ts_high_finds_the_max_bar():
    df = _prep(night_class='no_night', seed=11, days=DAYS[:5])
    out = factors.timestamp_factors(df)
    for day, g in df.groupby('trading_date'):
        j = g['highw'].to_numpy().argmax()
        assert out.loc[day, 'ts_high'] == pytest.approx(g['gamma_norm'].iloc[j])


def test_symbol_daily_factors_covers_all_expected_columns():
    df = _prep(days=DAYS[:60])
    out = factors.symbol_daily_factors(df, LOOKBACK, PCT, with_coords=True)
    for c in C.FACTOR_SIGNS:
        assert c in out.columns, f"缺少论文先验因子 {c}"
    assert out.index.name == 'trading_date'
    assert out.index.is_monotonic_increasing


def test_apply_signs_flips_negative_factors():
    df = pd.DataFrame({'ts_high': [0.2, 0.8], 'dfp_max': [0.1, -0.1]})
    out = factors.apply_signs(df)
    assert C.FACTOR_SIGNS['ts_high'] == -1
    np.testing.assert_allclose(out['ts_high'], [-0.2, -0.8])
    np.testing.assert_allclose(out['dfp_max'], [0.1, -0.1])


def test_apply_signs_strict_rejects_unknown():
    with pytest.raises(factors.UnsignedFactor):
        factors.apply_signs(pd.DataFrame({'made_up_factor': [1.0]}))


def test_factor_index_is_trading_date_not_wall_clock():
    """因子表的每一行必须对应一个 trading_date；夜盘不得被算成单独一天。"""
    df = _prep(night_class='night_0230', seed=17, days=DAYS[:30])
    out = factors.symbol_daily_factors(df, LOOKBACK, PCT, with_coords=True)
    assert len(out) == df['trading_date'].nunique()
    assert out.index.equals(pd.DatetimeIndex(sorted(df['trading_date'].unique())))
