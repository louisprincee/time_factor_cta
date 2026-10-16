"""早盘时钟：位置、个例、路径、状态和量能。不读行情。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import morning_clock as clock

ROOT = Path(__file__).resolve().parents[2]
PATH_CONFIG = json.loads((ROOT / "config" / "research" / "clock_path.json").read_text(encoding="utf-8"))
FLOW_CONFIG = json.loads((ROOT / "config" / "research" / "clock_flow.json").read_text(encoding="utf-8"))
STATE_CONFIG = json.loads((ROOT / "config" / "research" / "clock_state.json").read_text(encoding="utf-8"))


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


def frame_of(diff, gross, years=None):
    n = len(diff)
    years = years if years is not None else [2018] * n
    return pd.DataFrame({
        'diff': diff, 'gross_1130': gross, 'year': years,
        'trading_date': pd.to_datetime([f'{y}-01-01' for y in years]) + pd.to_timedelta(np.arange(n), 'D'),
    })


def test_label_cases_signs_returns_by_expected_direction_and_drops_flat_days():
    table = frame_of([0.9, -0.9, 0.9, -0.9, 0.1, 0.9], [0.01, 0.01, -0.02, -0.02, 0.05, 0.0])
    out = clock.label_cases(table, 0.6)
    assert list(out.side) == [1, -1, 1, -1]          # 中间段和收益为 0 的日子不进样本
    assert list(out.signed > 0) == [True, False, False, True]


def test_pick_cases_takes_two_days_per_category_and_year_and_is_repeatable():
    rng = np.random.default_rng(0)
    n = 400
    table = frame_of(rng.choice([0.9, -0.9], n), rng.normal(0, 0.01, n), years=[2018] * 200 + [2019] * 200)
    first = clock.pick_cases(table, seed=1)
    again = clock.pick_cases(table, seed=1)
    assert first.groupby(['类别', 'year']).size().eq(2).all()
    assert len(first) == len(clock.CATEGORIES) * 2 * 2
    pd.testing.assert_frame_equal(first, again)
    for _, row in first.iterrows():
        spec = clock.CATEGORIES[row.类别]
        assert np.sign(row['diff']) == spec['side']
        assert (spec['side'] * row.gross_1130 > 0) == spec['win']


def test_rank_regression_recovers_a_cross_sectional_rank_link_and_ignores_day_level_moves():
    rng = np.random.default_rng(3)
    days = np.repeat(pd.date_range('2020-01-01', periods=200), 20)
    x = rng.normal(size=len(days))
    market = np.repeat(rng.normal(0, 5, 200), 20)       # 当天整体涨跌，截面排名后应被去掉
    y = market + x + rng.normal(0, 0.5, len(days))
    frame = pd.DataFrame({'trading_date': days, 'diff': x, 'gross_1130': y})
    coef, t_stat = clock.rank_regression(frame, ['diff'])['diff']
    assert coef > 0.8 and t_stat > 20


def test_heterogeneity_q_is_small_for_equal_years_and_large_for_shifted_years():
    rng = np.random.default_rng(5)
    index = pd.date_range('2016-01-01', '2021-12-31', freq='D')
    same = pd.Series(rng.normal(0.02, 0.2, len(index)), index=index)
    _, p_same = clock.heterogeneity(same)
    shifted = same + np.where(index.year % 2 == 0, 0.1, -0.1)
    _, p_shift = clock.heterogeneity(shifted)
    assert p_same > 0.01 and p_shift < 1e-6


def test_pre_path_points_start_at_first_bar_and_end_at_decision():
    points = clock.pre_path_points(10, 130)
    assert points[0] == 10 and points[-1] == 130 and len(points) == clock.PRE_POINTS
    assert np.all(np.diff(points) >= 0)


def test_day_grid_runs_from_entry_to_close_and_ticks_exist():
    assert clock.DAY_GRID[0] == (9, 17) and clock.DAY_GRID[-1] == (15, 0)
    assert (10, 20) not in clock.DAY_GRID and (12, 0) not in clock.DAY_GRID
    assert all(hm in clock.DAY_GRID for hm in clock.TICKS)
    assert len(clock.DAY_GRID) == pytest.approx(59 + 60 + 90, abs=0)


def test_config_locks_the_two_variables_and_the_decision_bar():
    assert PATH_CONFIG["declared_at"] == "2026-10-14"
    assert "不读 2022" in PATH_CONFIG["sample"]
    assert [item["id"] for item in PATH_CONFIG["variables"]] == ["impulse", "efficiency"]
    assert [item["expected_interaction"] for item in PATH_CONFIG["variables"]] == ["正", "负"]
    decision = PATH_CONFIG["decision"]
    assert decision["interaction_abs_t"] == 2.5
    assert decision["min_years_same_sign"] == 5
    assert decision["min_sectors_same_sign"] == 4


def test_tail_impulse_is_the_share_of_the_range_and_is_blank_when_the_range_is_flat():
    assert clock.tail_impulse(-0.02, 0.05) == pytest.approx(0.4)
    assert clock.tail_impulse(0.02, 0.05) == pytest.approx(0.4)
    assert np.isnan(clock.tail_impulse(0.01, 0.0))
    assert np.isnan(clock.tail_impulse(np.nan, 0.05))


def test_path_efficiency_is_one_on_a_straight_line_and_lower_when_the_path_retraces():
    assert clock.path_efficiency_from_closes([100, 101, 102, 103]) == pytest.approx(1)
    assert clock.path_efficiency_from_closes([100, 110, 90, 104]) == pytest.approx(4 / 44)
    assert np.isnan(clock.path_efficiency_from_closes([100, 100, 100]))
    assert np.isnan(clock.path_efficiency_from_closes([100]))


def test_five_minute_bins_match_the_case_chart_resample_and_keep_the_0916_bar():
    night = pd.date_range("2020-01-02 21:01", periods=10, freq="min")
    morning = pd.date_range("2020-01-03 09:00", "2020-01-03 09:16", freq="min")
    index = night.append(morning)
    values = np.arange(len(index), dtype=float)
    expected = pd.Series(values, index=index).resample(
        "5min", closed="right", label="right", origin="epoch", offset="1min").last().dropna()
    got = clock.five_minute_last(values, index.asi8)
    assert np.allclose(got, expected.to_numpy())
    assert expected.index[-1] == pd.Timestamp("2020-01-03 09:16")


def test_still_extending_follows_the_later_extreme_and_is_blank_without_a_direction():
    out = clock.still_extending([0.8, 0.8, -0.8, -0.8, 0.0], [-0.01, 0.01, 0.01, -0.01, -0.02])
    assert list(out[:4]) == [1.0, 0.0, 1.0, 0.0]
    assert np.isnan(out[4])
    push = clock.aligned_dev([0.8, -0.8], [-0.01, 0.01])
    assert push[0] == pytest.approx(0.01) and push[1] == pytest.approx(0.01)


def test_tercile_is_balanced_inside_one_symbol_year_and_blank_when_too_short():
    labels = clock.assign_tercile(pd.Series(np.arange(9)))
    assert list(labels.value_counts().sort_index()) == [3, 3, 3]
    assert clock.assign_tercile(pd.Series([1.0, 2.0])).isna().all()


def test_interaction_uses_the_product_of_ranks_and_recovers_a_reversal_that_strengthens_with_state():
    rng = np.random.default_rng(7)
    days = np.repeat(pd.date_range("2020-01-01", periods=80), 12)
    diff = rng.normal(size=len(days))
    state = rng.normal(size=len(days))
    extend = rng.normal(size=len(days))
    y = diff * state + rng.normal(scale=0.15, size=len(days))
    frame = pd.DataFrame({
        "trading_date": days, "diff": diff, "impulse": state, "extend": extend, "gross_1130": y,
    })
    outcome, columns, day, n = clock.interaction_terms(frame, "impulse", "extend")
    diff_r = clock.centered_rank(frame, "diff")
    state_r = clock.centered_rank(frame, "impulse")
    assert np.allclose(columns["低减高×状态"], (diff_r * state_r).to_numpy())
    assert n == len(frame)
    fitted = clock.clustered_coefficients(outcome, columns, day)
    coef, t_stat = fitted["低减高×状态"]
    assert coef > 0 and t_stat > 8


def test_daily_spearman_keeps_a_datetime_index_and_is_empty_when_a_day_is_too_thin():
    days = np.repeat(pd.date_range("2020-01-01", periods=3), 6)
    frame = pd.DataFrame({
        "trading_date": days,
        "diff": np.tile(np.arange(6), 3),
        "gross_1130": np.tile(np.arange(6), 3) + np.repeat([0, 1, -1], 6),
    })
    ic = clock.daily_spearman(frame, min_names=5)
    assert isinstance(ic.index, pd.DatetimeIndex) and len(ic) == 3
    thin = clock.daily_spearman(frame.iloc[:4], min_names=5)
    assert isinstance(thin.index, pd.DatetimeIndex) and len(thin) == 0


def test_judge_passes_only_when_both_regressions_and_the_sign_counts_clear():
    decision = PATH_CONFIG["decision"]
    passed = clock.judge(3.1, 2.8, 5, 4, 1, decision)
    assert passed["通过"]
    assert not clock.judge(3.1, 2.4, 5, 4, 1, decision)["通过"]
    assert not clock.judge(3.1, -2.8, 5, 4, 1, decision)["通过"]
    assert not clock.judge(-3.1, -2.8, 5, 4, 1, decision)["通过"]
    assert not clock.judge(3.1, 2.8, 4, 4, 1, decision)["通过"]
    assert not clock.judge(3.1, 2.8, 5, 3, 1, decision)["通过"]
    assert clock.judge(-3.1, -2.8, 5, 4, -1, decision)["通过"]


def test_config_locks_three_states_and_the_reversal_payoff_bar():
    assert STATE_CONFIG["declared_at"] == "2026-10-14"
    assert "不读 2022" in STATE_CONFIG["sample"]
    assert "back" in STATE_CONFIG["sign_convention"]
    assert [item["id"] for item in STATE_CONFIG["variables"]] == ["carry_align", "stock_align", "oi_state"]
    assert [item["expected_interaction"] for item in STATE_CONFIG["variables"]] == ["负", "负", "负"]
    decision = STATE_CONFIG["decision"]
    assert decision["interaction_abs_t"] == 2.5
    assert decision["min_years_same_sign"] == 5
    assert decision["min_sectors_same_sign"] == 4
    assert decision["min_reversal_bp"] == 4.0
    assert decision["reversal_bin"] == "低"


def test_zscore_at_earlier_dates_ignores_a_later_observation():
    series = pd.Series(np.arange(80, dtype=float))
    changed = series.copy()
    changed.iloc[-1] = 10_000
    original = clock.zscore_known(series, window=20, min_periods=10)
    bumped = clock.zscore_known(changed, window=20, min_periods=10)
    pd.testing.assert_series_equal(original.iloc[:-1], bumped.iloc[:-1])


def test_curve_alignment_is_positive_when_backwardation_agrees_with_the_rally():
    carry = pd.Series([2.0, -2.0, -2.0, 2.0, 2.0])
    out = clock.curve_alignment(carry, clock.move_sign(pd.Series([0.01, -0.02, 0.03, -0.04, 0.0])))
    assert out.iloc[0] == pytest.approx(2)
    assert out.iloc[1] == pytest.approx(2)
    assert out.iloc[2] == pytest.approx(-2)
    assert out.iloc[3] == pytest.approx(-2)
    assert np.isnan(out.iloc[4])
    assert clock.move_sign(pd.Series([0.0, np.nan])).isna().all()


def test_stock_alignment_treats_destocking_plus_a_rally_as_continuation():
    move = clock.move_sign(pd.Series([0.01, -0.02, -0.03]))
    stock = pd.Series([-1.5, 1.5, -1.5])
    out = clock.stock_alignment(stock, move)
    assert out.iloc[0] == pytest.approx(1.5)
    assert out.iloc[1] == pytest.approx(1.5)
    assert out.iloc[2] == pytest.approx(-1.5)


def test_warrant_change_is_blank_when_either_stock_level_is_not_positive():
    level = pd.Series([100.0, 100.0, 110.0, 0.0, 120.0])
    change = clock.warrant_change(level, periods=2)
    assert np.isnan(change.iloc[0]) and np.isnan(change.iloc[1])
    assert change.iloc[2] == pytest.approx(0.1)
    assert np.isnan(change.iloc[3])
    assert np.isnan(clock.warrant_change(pd.Series([0.0, 10.0, 20.0]), periods=2).iloc[2])


def test_oi_impulse_scales_the_change_and_drops_a_roll_or_a_nonpositive_baseline():
    out = clock.oi_impulse([10, 10, 10], [100, 100, 0], [True, False, True])
    assert out[0] == pytest.approx(0.1)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_judge_requires_the_reversal_bin_to_clear_four_basis_points():
    decision = STATE_CONFIG["decision"]
    passed = clock.judge_state(-3.0, 5, 4, 4.0, -1, decision)
    assert passed["通过"]
    assert not clock.judge_state(-3.0, 5, 4, 3.9, -1, decision)["通过"]
    assert not clock.judge_state(3.0, 5, 4, 6.0, -1, decision)["通过"]
    assert not clock.judge_state(-3.0, 4, 4, 6.0, -1, decision)["通过"]
    assert not clock.judge_state(-3.0, 5, 3, 6.0, -1, decision)["通过"]


def test_config_freezes_three_predecision_candidates_and_signs():
    assert FLOW_CONFIG["declared_at"] == "2026-10-14"
    assert "不读 2022" in FLOW_CONFIG["sample"]
    assert [spec["id"] for spec in FLOW_CONFIG["variables"]] == [
        "open_volume_surprise", "impact_per_volume", "volume_peak_clock"
    ]
    assert [spec["expected_interaction"] for spec in FLOW_CONFIG["variables"]] == ["负", "正", "负"]
    assert "shift(1)" in FLOW_CONFIG["variables"][0]["formula"]


def test_volume_surprise_uses_only_positive_available_baselines():
    assert clock.volume_surprise(200.0, 100.0) == pytest.approx(2.0)
    assert np.isnan(clock.volume_surprise(200.0, 0.0))
    assert np.isnan(clock.volume_surprise(np.nan, 100.0))


def test_price_impact_is_move_scaled_by_square_root_of_relative_volume():
    assert clock.impact_per_volume(-0.02, 4.0) == pytest.approx(0.01)
    assert np.isnan(clock.impact_per_volume(0.02, 0.0))


def test_volume_peak_clock_is_scaled_to_open_window_and_handles_missing_values():
    assert clock.volume_peak_clock([1, 2, 8, 3, 2]) == pytest.approx(0.5)
    assert np.isnan(clock.volume_peak_clock([np.nan, 1, np.nan]))
    assert np.isnan(clock.volume_peak_clock([4]))