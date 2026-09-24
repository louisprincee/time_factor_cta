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


def test_basic_aggregations_match_hand_computation():
    dur = np.array([1.0, 2.0, 3.0, 10.0, 4.0, 4.0])
    codes = np.array([0, 0, 0, 0, 0, 0])
    days = pd.DatetimeIndex(['2015-01-01'])
    out = factors.basic_aggregations(dur, codes, days, 'dur')
    assert out['dur_mean'].iloc[0] == pytest.approx(dur.mean())
    assert out['dur_std'].iloc[0] == pytest.approx(dur.std(ddof=0))
    assert out['dur_max'].iloc[0] == 10.0
    assert out['dur_gap'].iloc[0] == 9.0
    thresh = dur.mean() + 2 * dur.std(ddof=0)
    assert out['dur_extreme'].iloc[0] == pytest.approx((dur >= thresh).mean())


def test_all_nan_day_stays_nan():
    """预热期不足的交易日必须整日 NaN，不得退化成 0。"""
    dur = np.array([np.nan, np.nan, np.nan])
    out = factors.basic_aggregations(dur, np.zeros(3, int),
                                pd.DatetimeIndex(['2015-01-01']), 'dur')
    assert out.isna().all().all()


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


def test_vr_night_is_nan_without_night_session():
    """无夜盘时 VR_night 必须是 NaN。填 0 会造成结构性零值，压垮真因子。"""
    vdur = np.array([2.0, 4.0, 10.0, 10.0])
    sess = np.array([C.SESSION_AM, C.SESSION_AM, C.SESSION_PM, C.SESSION_PM])
    out = factors.volume_ratio_factors(vdur, sess, np.zeros(4, int),
                                  pd.DatetimeIndex(['2015-01-01']))
    assert np.isnan(out['vr_night'].iloc[0])


def test_no_night_symbol_has_all_night_factors_nan():
    """端到端：无夜盘品种的三个夜盘因子必须全列 NaN，且绝不出现 0。"""
    df = _prep(night_class='no_night', seed=8)
    out = factors.symbol_daily_factors(df, LOOKBACK, PCT, with_coords=True)
    for c in C.NIGHT_DEPENDENT_FACTORS:
        assert out[c].isna().all(), f"{c} 在无夜盘品种上不应有值"
    assert (out[list(C.NIGHT_DEPENDENT_FACTORS)] == 0).sum().sum() == 0


def test_night_dependent_set_is_exhaustive():
    """把"无夜盘品种上全 NaN 的列"反推出来，必须与 NIGHT_DEPENDENT_FACTORS 完全相等。

    这条是从第 3 步演练里补回来的。原来那张表只列了显式带 night 字样的三个因子，
    漏了 ts_high_night / ts_low_night，后果有两层：验收时它们按全池统计缺失率，
    在有无夜盘品种混合的池子里必然被判不合格；反向检查也不会看它们，等于
    "无夜盘品种上是 NaN 还是 0" 这个最要紧的问题在这两个因子上根本没被检查。

    所以这里不再手写名单，而是由实际输出反推——以后新增任何夜盘相关因子，
    忘了登记就会在这里失败。
    """
    no_night = factors.symbol_daily_factors(
        _prep(night_class='no_night', seed=11), LOOKBACK, PCT, with_coords=True)
    with_night = factors.symbol_daily_factors(
        _prep(night_class='night_0100', seed=11), LOOKBACK, PCT, with_coords=True)
    # 只在有夜盘品种上有值、在无夜盘品种上整列缺失 -> 结构性依赖夜盘
    structural = {c for c in no_night.columns
                  if no_night[c].isna().all() and with_night[c].notna().any()}
    assert structural == C.NIGHT_DEPENDENT_FACTORS, (
        f"漏登记 {sorted(structural - C.NIGHT_DEPENDENT_FACTORS)}；"
        f"多登记 {sorted(C.NIGHT_DEPENDENT_FACTORS - structural)}")


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
    df = pd.DataFrame({'pmt': [0.2, 0.8], 'dfp_max': [0.1, -0.1]})
    out = factors.apply_signs(df)
    assert C.FACTOR_SIGNS['pmt'] == -1
    np.testing.assert_allclose(out['pmt'], [-0.2, -0.8])
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
