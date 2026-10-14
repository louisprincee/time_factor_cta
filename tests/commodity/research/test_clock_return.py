"""高低点位置与收益：聚类斜率和三条关系的方向。不读行情。"""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts' / 'commodity'))
clock = importlib.import_module('morning_clock_return')


def test_cluster_slope_recovers_the_line_and_withholds_a_single_day():
    day = np.arange(40)
    x = np.linspace(-1, 1, 40)
    out = clock.cluster_slope(day, x, 0.02 * x)
    assert out['斜率'] == pytest.approx(0.02, rel=1e-6)
    assert out['双尾p'] < 1e-6
    one_day = clock.cluster_slope(np.zeros(10), x[:10], x[:10])
    assert np.isnan(one_day['双尾p'])


def test_late_clock_minus_early_clock_is_the_difference_in_means():
    early = np.column_stack([np.arange(30), np.full(30, 0.1), np.zeros(30)])
    late = np.column_stack([np.arange(30, 60), np.full(30, 0.9), np.full(30, 0.01)])
    sample = np.vstack([early, late])
    out = clock.late_minus_early(sample[:, 0], sample[:, 1], sample[:, 2])
    assert out['斜率'] == pytest.approx(0.01)


def test_study_items_use_all_directions_and_unmodified_returns():
    frame = pd.DataFrame({
        'dev15': [0.01, -0.01, 0.0],
        'gross_1130': [0.002, -0.003, 0.004],
        'hclock_td': [0.9, 0.1, 0.5],
        'lclock_td': [0.2, 0.8, 0.4],
        'trading_date': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06']),
    })
    high, low, difference = clock.study_items(frame)
    np.testing.assert_allclose(high['x'], [0.9, 0.1, 0.5])
    np.testing.assert_allclose(high['y'], [0.002, -0.003, 0.004])
    np.testing.assert_allclose(low['x'], [0.2, 0.8, 0.4])
    np.testing.assert_allclose(low['y'], [0.002, -0.003, 0.004])
    np.testing.assert_allclose(difference['x'], [-0.7, 0.7, -0.1])
    np.testing.assert_allclose(difference['y'], [0.002, -0.003, 0.004])


def test_yearly_summary_matches_pooled_position_only_spearman():
    frame = pd.DataFrame({
        'trading_date': pd.to_datetime(['2020-01-02'] * 4 + ['2020-01-03'] * 4),
        'hclock_td': [0.9, 0.6, 0.4, 0.1, 0.1, 0.4, 0.6, 0.9],
        'lclock_td': [0.1, 0.3, 0.6, 0.9, 0.9, 0.6, 0.3, 0.1],
        'gross_1130': [0.01, 0.02, 0.03, 0.04] * 2,
        'dev15': [0.01, -0.01, 0.0, 0.01] * 2,
    })

    yearly = clock.yearly_pooled_summary(frame, '有夜盘', '全部').set_index('关系')

    for item in clock.study_items(frame):
        expected = clock.association(item['day'], item['x'], item['y'])
        actual = yearly.loc[item['关系']]
        assert actual['Spearman'] == pytest.approx(expected['Spearman'])
        if np.isnan(expected['Spearman_p']):
            assert np.isnan(actual['Spearman双尾p'])
        else:
            assert actual['Spearman双尾p'] == pytest.approx(expected['Spearman_p'])
        assert actual['笔数'] == expected['笔数']
        assert actual['天数'] == expected['天数']


def test_segment_table_includes_clock_difference_bins(tmp_path):
    frame = pd.DataFrame({
        'trading_date': pd.date_range('2020-01-01', periods=10),
        'hclock_td': [0.9, 0.9, 0.7, 0.7, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1],
        'lclock_td': [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9],
        'gross_1130': np.arange(10) / 1000,
    })

    segments, summary = clock.segment_table(frame)
    relation = '低点时钟-高点时钟 → 多头收益'
    difference_bins = segments[segments.关系 == relation]

    assert difference_bins.时钟区间.tolist() == clock.DIFF_BIN_LABELS
    assert difference_bins.笔数.tolist() == [2, 2, 2, 2, 2]
    assert relation in summary.关系.tolist()
    output = tmp_path / 'segments.png'
    clock.plot_segments(segments, summary, output)
    assert output.exists()


def test_late_minus_early_uses_thirds_of_each_variables_own_range():
    day = pd.date_range('2020-01-01', periods=8).to_numpy()
    x = np.array([-0.9, -0.5, -0.2, 0.0, 0.1, 0.2, 0.5, 0.9])
    y = np.array([-2.0, -1.0, 9.0, 9.0, 9.0, 9.0, 1.0, 2.0])
    gap = clock.late_minus_early(day, x, y, -1.0, 1.0)
    assert gap['笔数'] == 4 and gap['斜率'] == pytest.approx(3.0)  # 只比 ≥1/3 和 ≤−1/3，中间的 9 不进来


def test_daily_ic_is_a_per_day_cross_section_and_ignores_market_moves():
    days = np.repeat(pd.date_range('2020-01-01', periods=3), 5)
    clock_value = np.tile([0.1, 0.3, 0.5, 0.7, 0.9], 3)
    market = np.repeat([-.05, .0, .05], 5)  # 每天整体涨跌不同，不应影响截面 IC
    frame = pd.DataFrame({'trading_date': days, 'hclock_td': clock_value, 'lclock_td': 1 - clock_value,
                          'gross_1130': market - clock_value / 100})
    ic = clock.daily_ic(frame)
    high = ic[ic.关系 == '高点时钟 → 多头收益']
    assert len(high) == 3 and np.allclose(high.IC, -1.0)
    low = ic[ic.关系 == '低点时钟 → 多头收益']
    assert np.allclose(low.IC, 1.0)
    assert clock.daily_ic(frame.iloc[:4]).empty  # 一天不到 5 个品种不算


def test_ic_summary_reports_whole_period_and_years():
    ic = pd.DataFrame({'关系': 'r', 'trading_date': pd.bdate_range('2019-01-01', periods=40).append(
        pd.bdate_range('2020-01-01', periods=40)), 'IC': np.r_[np.full(40, .1), np.full(40, -.1)] +
        np.random.default_rng(0).normal(0, .01, 80), '品种数': 10})
    out = clock.ic_summary(ic, '全部').set_index('年份')
    assert out.loc[2019, 'IC均值'] > 0 > out.loc[2020, 'IC均值']
    assert abs(out.loc['全期', 'IC均值']) < .01


def test_balance_up_down_equalises_each_year_sector_cell_and_drops_flat_days():
    rng = np.random.default_rng(1)
    n = 400
    frame = pd.DataFrame({'trading_date': pd.to_datetime(rng.choice(['2018-03-01', '2019-03-01'], n)),
                          'sector': rng.choice(['化工', '黑色'], n),
                          'gross_1130': np.r_[rng.normal(-.001, .01, n - 10), np.zeros(10)]})
    out = clock.balance_up_down(frame)
    assert (out.gross_1130 != 0).all()
    cells = out.groupby([out.trading_date.dt.year, out.sector]).gross_1130.apply(lambda s: (s > 0).sum() - (s < 0).sum())
    assert (cells == 0).all()
    pd.testing.assert_frame_equal(out, clock.balance_up_down(frame))  # 同一种子结果不变
    assert not out.equals(clock.balance_up_down(frame, seed=1))
