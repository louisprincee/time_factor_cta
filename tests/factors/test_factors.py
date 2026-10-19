"""因子族测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import add_intraday_coords, make_symbol
from tfcta.factors import (
    day_codes_of, dfp_factors, duration_factors, duration_series, intraday_abs_diff,
    report_factors, rolling_threshold, timestamp_factors,
)


DAYS = pd.bdate_range('2015-01-01', '2015-12-31')
LOOKBACK, PCT = 60, 55.0


def _all_factors(df):
    return pd.concat([duration_factors(df, LOOKBACK, PCT), timestamp_factors(df),
                      report_factors(df, LOOKBACK, PCT)], axis=1)


def _prep(night_class='night_2300', seed=5, days=DAYS):
    df = make_symbol(days, night_class=night_class, seed=seed)
    return add_intraday_coords(df)


def test_dfp_uses_longest_duration_bar_and_is_normalized():
    dur = np.array([1.0, 50.0, 2.0, 3.0])
    price = np.array([100.0, 120.0, 110.0, 105.0])
    codes = np.zeros(4, int)
    days = pd.DatetimeIndex(['2015-01-01'])
    out = dfp_factors(dur, price, codes, days, top_ns=[1])
    # FP = 持续期最大(50)那根的价格 120；Close = 最后一根 105
    assert out['dfp_max'].iloc[0] == pytest.approx((120.0 - 105.0) / 105.0)


def test_dfp_is_scale_invariant():
    """把价格整体放大 10 倍，归一化后的 DFP 必须不变——这正是跨品种可比的要求。"""
    dur = np.array([1.0, 50.0, 2.0, 3.0])
    price = np.array([100.0, 120.0, 110.0, 105.0])
    codes, days = np.zeros(4, int), pd.DatetimeIndex(['2015-01-01'])
    a = dfp_factors(dur, price, codes, days, top_ns=[1])['dfp_max'].iloc[0]
    b = dfp_factors(dur, price * 10, codes, days, top_ns=[1])['dfp_max'].iloc[0]
    assert a == pytest.approx(b)


def test_ts_high_finds_the_max_bar():
    df = _prep(night_class='no_night', seed=11, days=DAYS[:5])
    out = timestamp_factors(df)
    for day, g in df.groupby('trading_date'):
        j = g['highw'].to_numpy().argmax()
        assert out.loc[day, 'ts_high'] == pytest.approx(g['gamma_norm'].iloc[j])


def test_session_clock_renormalizes_without_night_bars():
    index = pd.to_datetime([
        '2022-01-04 21:00', '2022-01-05 09:00', '2022-01-05 10:00',
        '2022-01-05 13:00', '2022-01-05 14:59',
    ])
    frame = pd.DataFrame({
        'trading_date': pd.Timestamp('2022-01-05'),
        'highw': [999.0, 101.0, 110.0, 104.0, 103.0],
        'loww': [1.0, 99.0, 100.0, 98.0, 90.0],
    }, index=index)
    enriched = add_intraday_coords(frame)

    full_day = timestamp_factors(enriched)
    day_session = timestamp_factors(enriched, sessions=('AM', 'PM'))

    assert full_day.iloc[0].to_list() == [0.0, 0.0]
    assert day_session.loc[pd.Timestamp('2022-01-05'), 'ts_high'] == pytest.approx(1 / 3)
    assert day_session.loc[pd.Timestamp('2022-01-05'), 'ts_low'] == pytest.approx(1.0)


def test_factor_families_cover_all_expected_columns():
    df = _prep(days=DAYS[:60])
    out = _all_factors(df)
    for c in C.FACTOR_SIGNS:
        assert c in out.columns, f"缺少论文先验因子 {c}"
    assert out.index.is_monotonic_increasing


def test_dfp_ignores_additive_adjustment_offset():
    """持续期用 closew，FP 和分母用原始 close：closew 整体平移不能改变 DFP。"""
    df = _prep(days=DAYS[:40])
    base = duration_factors(df, LOOKBACK, PCT)
    shifted = df.copy()
    shifted['closew'] = shifted['closew'] - 5000.0
    pd.testing.assert_frame_equal(base, duration_factors(shifted, LOOKBACK, PCT))

    codes, days = day_codes_of(df)
    price = df['closew'].to_numpy(dtype='float64')
    thr = rolling_threshold(intraday_abs_diff(price, codes), codes, LOOKBACK, PCT)
    dur = duration_series(price, codes, thr)
    expect = dfp_factors(dur, df['close'].to_numpy(dtype='float64'), codes, days)
    np.testing.assert_allclose(base['dfp_max'].to_numpy(), expect['dfp_max'].to_numpy(),
                               equal_nan=True)


def test_dfp_skips_non_positive_close():
    dur = np.array([1.0, 50.0, 2.0])
    price = np.array([100.0, 120.0, 0.0])
    out = dfp_factors(dur, price, np.zeros(3, int),
                              pd.DatetimeIndex(['2015-01-01']), top_ns=[1])
    assert np.isnan(out['dfp_max'].iloc[0])


def test_factor_index_is_trading_date_not_wall_clock():
    """因子表的每一行必须对应一个 trading_date；夜盘不得被算成单独一天。"""
    df = _prep(night_class='night_0230', seed=17, days=DAYS[:30])
    out = _all_factors(df)
    assert len(out) == df['trading_date'].nunique()
    assert out.index.equals(pd.DatetimeIndex(sorted(df['trading_date'].unique())))
