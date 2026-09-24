"""品种池测试（设计文档第 4 节）。

最重要的一组是"时点有效性"：第 y 年的池子绝不能用第 y 年及以后的数据判定。
这类前视偏差不会报错、不会影响任何单测之外的断言，只会让最终绩效虚高，
所以必须用构造数据把它钉死。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import universe as U


def _minute_df(days, bars_per_day=10, night_bars=0, turnover=1.0,
               closew=100.0, nan_days=()) -> pd.DataFrame:
    """最小分钟表：每天 night_bars 根夜盘（前一日 21:00 起）+ 日盘若干根。"""
    rows = []
    for k, d in enumerate(pd.DatetimeIndex(days)):
        stamps = []
        if night_bars:
            prev = pd.DatetimeIndex(days)[k - 1] if k else d - pd.Timedelta(days=1)
            stamps += [prev + pd.Timedelta(hours=21, minutes=m + 1)
                       for m in range(night_bars)]
        stamps += [d + pd.Timedelta(hours=9, minutes=m + 1) for m in range(bars_per_day)]
        is_nan = d.normalize() in {pd.Timestamp(x).normalize() for x in nan_days}
        for ts in stamps:
            rows.append({'ts': ts, 'trading_date': d.normalize(),
                         'closew': np.nan if is_nan else closew,
                         'total_turnover': np.nan if is_nan else turnover})
    out = pd.DataFrame(rows).set_index('ts').sort_index()
    return out


def _stats(rows) -> pd.DataFrame:
    """由 (symbol, year, turnover亿, valid_ratio, valid_days) 构造统计长表。"""
    recs = []
    for sym, year, yi, ratio, vdays in rows:
        recs.append({'symbol': sym, 'year': year, 'turnover_median': yi * 1e8,
                     'valid_ratio': ratio, 'valid_days': vdays,
                     'n_days': vdays, 'turnover_days': vdays,
                     'night_bars_median': 120.0, 'bars_median': 345.0,
                     'calendar_days': 243, 'night_class': 'night_2300',
                     'has_night': True})
    return pd.DataFrame(recs)


PASS = [('RB', 2016, 100.0, 1.0, 240)]


def test_daily_stats_all_nan_day_is_nan_not_zero():
    """全 NaN 的一天必须得 NaN。若给 0，"没有数据"会伪装成"零成交"，
    进而把一个尚未上市的品种误判为僵尸品种。"""
    days = pd.bdate_range('2016-03-01', periods=3)
    df = _minute_df(days, turnover=2.0, nan_days=[days[1]])
    d = U.daily_stats(df)
    assert np.isnan(d.loc[days[1], 'turnover'])
    assert d.loc[days[1], 'n_valid_close'] == 0
    assert d.loc[days[0], 'turnover'] == 20.0


def test_market_calendar_is_union_not_per_symbol_max():
    """分母必须是全市场交易日的并集。用品种自己的天数当分母，
    一个只交易了 100 天的品种会显示为 100% 完整。"""
    a = U.daily_stats(_minute_df(pd.bdate_range('2016-01-04', periods=10)))
    b = U.daily_stats(_minute_df(pd.bdate_range('2016-01-11', periods=10)))
    cal = U.market_calendar({'A': a, 'B': b})
    assert cal.loc[2016] == 15          # 两段各 10 天、重叠 5 天


def test_screen_never_uses_target_year_data():
    """目标年数据再漂亮也不能用：2017 年爆量、2016 年不达标的品种
    在 2017 年的池子里必须缺席。这就是前视偏差的样子。"""
    stats = _stats([('X', 2016, 1.0, 1.0, 240),      # 前一年不达标
                    ('X', 2017, 999.0, 1.0, 240)])   # 当年爆量
    d = U.screen_year(stats, 2017)
    assert not bool(d.loc['X', 'passed'])
    assert d.loc['X', 'reason'] == '成交额'
    assert d.loc['X', 'screen_window'] == '2016-2016'


def test_screen_uses_previous_year_even_if_symbol_later_collapses():
    """反向的另一半：前一年活跃、当年塌缩的品种，当年**应当**在池子里。

    事后我们知道它塌了，但在年初做决策时不知道。把它剔除同样是用未来信息。
    """
    stats = _stats([('ZC', 2017, 500.0, 1.0, 240),
                    ('ZC', 2018, 0.1, 1.0, 240)])
    assert bool(U.screen_year(stats, 2018).loc['ZC', 'passed'])
    # 而塌缩之后的那一年就必须出池
    assert not bool(U.screen_year(stats, 2019).loc['ZC', 'passed'])


def test_screen_rejects_zero_lookback():
    with pytest.raises(ValueError, match='前视'):
        U.screen_year(_stats(PASS), 2017, lookback_years=0)


def test_build_universe_refuses_years_beyond_research_end():
    stats = _stats([('RB', y, 100.0, 1.0, 240) for y in (2020, 2021)])
    with pytest.raises(C.HoldoutViolation):
        U.build_universe(stats, years=[C.RESEARCH_END.year + 2])


def test_completeness_needs_both_ratio_and_absolute_days():
    """占比与绝对天数是两道独立的门槛：半日市密集的年份可能占比够但天数不够。"""
    low_ratio = _stats([('A', 2016, 100.0, C.MIN_VALID_DAY_RATIO - 0.01, 240)])
    low_days = _stats([('B', 2016, 100.0, 1.0, C.MIN_VALID_DAYS - 1)])
    assert not bool(U.screen_year(low_ratio, 2017).loc['A', 'pass_complete'])
    assert not bool(U.screen_year(low_days, 2017).loc['B', 'pass_complete'])
    assert U.screen_year(low_ratio, 2017).loc['A', 'reason'] == '完整度'


def test_load_universe_roundtrip_restores_int_years():
    d = Path(tempfile.mkdtemp(prefix='tfcta_uni_'))
    p = d / 'universe_by_year.json'
    p.write_text(json.dumps({'2016': ['RB'], '2017': ['RB', 'CU']}), encoding='utf-8')
    uni = U.load_universe(p)
    assert set(uni) == {2016, 2017}
    assert uni[2017] == ['RB', 'CU']


def test_financial_symbols_are_excluded_by_construction():
    """金融期货的剔除发生在 COMMODITY_SYMBOLS 层，不依赖流动性门槛——
    T/TF 的成交额很高，靠门槛是筛不掉的。"""
    for s in C.FINANCIAL_SYMBOLS:
        assert s not in C.COMMODITY_SYMBOLS
    assert len(C.COMMODITY_SYMBOLS) == len(C.ALL_SYMBOLS) - len(C.FINANCIAL_SYMBOLS)
