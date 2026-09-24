"""时段划分与日内坐标测试。

这些测试守护的是整个项目最底层的假设。若其中任何一条失败，所有因子结果都无意义。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import sessions, synth


DAYS = pd.bdate_range('2016-03-01', '2016-03-31')


@pytest.fixture(params=list(synth.NIGHT_CLASS))


def sym(request):
    cls = request.param
    df = synth.make_symbol(DAYS, night_class=cls, seed=1)
    return cls, sessions.add_intraday_coords(df)


def _full_days(df: pd.DataFrame) -> pd.DatetimeIndex:
    """去掉样本首个交易日。

    首日的夜盘发生在样本区间开始之前，数据里必然缺失——真实分片的第一天同样如此。
    结构同质性的断言只对其余交易日成立。
    """
    days = pd.DatetimeIndex(sorted(df['trading_date'].unique()))
    return days[1:]


def test_bar_grid_matches_expected(sym):
    """每日 bar 数必须命中 225 / 345 / 465 / 555 四种网格之一。"""
    cls, df = sym
    counts = df.groupby('trading_date').size().loc[_full_days(df)]
    assert counts.nunique() == 1
    assert counts.iloc[0] == C.EXPECTED_BARS_PER_DAY[cls]


def test_night_belongs_to_next_trading_date(sym):
    """夜盘 bar 的墙钟时刻必须早于其 trading_date 的日盘——本项目最关键的一条语义。

    若这条在真实数据上不成立，全部因子的"日"划分都是错的。

    注意不要断言"00:00 之后的夜盘 bar 墙钟日期 == trading_date"。夜盘挂在**前一交易日**
    的晚上，所以周一 trading_date 的凌晨段落在周六，与 trading_date 相差 2 天。
    普遍成立的不变量只有两条：墙钟不晚于 trading_date，且严格早于当日日盘开盘。
    """
    cls, df = sym
    night = df[df['session'] == C.SESSION_NIGHT]
    if cls == 'no_night':
        assert night.empty
        return
    assert not night.empty
    ts = pd.DatetimeIndex(night.index)
    wall = ts.normalize()
    td = night['trading_date']
    assert (wall <= td).all()
    assert (ts < td + pd.Timedelta(hours=C.AM_START.hour, minutes=C.AM_START.minute)).all()
    # 21:00 之后的段一定跨日
    late = ts.hour >= C.NIGHT_START_HOUR
    assert late.any()
    assert (wall[late] < td[late]).all()


def test_gamma_norm_spans_unit_interval(sym):
    """归一化坐标必须恰好覆盖 [0, 1]，否则跨品种不可比。"""
    _, df = sym
    g = df.groupby('trading_date')['gamma_norm']
    assert np.allclose(g.min(), 0.0)
    assert np.allclose(g.max(), 1.0)
    assert df['gamma_norm'].between(0, 1).all()


def test_night_leads_the_day(sym):
    """同一 trading_date 内，夜盘 gamma 必须全部小于日盘——夜盘结构性领先。"""
    cls, df = sym
    if cls == 'no_night':
        return
    checked = 0
    for _, g in df.groupby('trading_date'):
        night = g.loc[g['session'] == C.SESSION_NIGHT, 'gamma']
        if night.empty:          # 样本首日无夜盘，见 _full_days
            continue
        assert night.max() < g.loc[g['session'] != C.SESSION_NIGHT, 'gamma'].min()
        checked += 1
    assert checked >= df['trading_date'].nunique() - 1


def test_missing_trading_date_raises():
    """禁止用 index.date 冒充交易日——必须显式报错而不是静默算错。"""
    df = synth.make_symbol(DAYS[:3], seed=2).drop(columns=['trading_date'])
    with pytest.raises(KeyError, match='trading_date'):
        sessions.add_intraday_coords(df)


def test_unsorted_input_is_sorted_before_numbering():
    """乱序输入必须先排序再编号，否则 gamma 错乱且无任何报错。"""
    df = synth.make_symbol(DAYS[:5], night_class='night_2300', seed=3)
    shuffled = df.sample(frac=1.0, random_state=0)
    out = sessions.add_intraday_coords(shuffled)
    assert out.index.is_monotonic_increasing
    ref = sessions.add_intraday_coords(df)
    pd.testing.assert_series_equal(out['gamma'], ref['gamma'])
