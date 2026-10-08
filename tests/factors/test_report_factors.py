"""报告其余时序因子：手算一天，钉住时点、时段和夜盘排除。"""
import numpy as np
import pandas as pd
import pytest

from tfcta.factors import intraday as F


def test_window_clock_is_first_extreme_inside_the_window():
    values = np.array([1.0, 5.0, 5.0, 0.0])
    mask = np.array([False, True, True, True])
    assert F._window_clock(values, mask, 'max') == pytest.approx(0.0)
    assert F._window_clock(values, mask, 'min') == pytest.approx(1.0)


def test_report_day_matches_hand_calculation():
    # 第一天只提供阈值。第二天：持续期最长在第 3 根；午后高点和低点都在窗口末根。
    rows = []
    day0 = pd.Timestamp('2016-01-04')
    day1 = pd.Timestamp('2016-01-05')
    for day, close, volume, high, low, turnover, session in (
        (day0, [100, 101, 100, 101], [10, 12, 10, 14], [1, 1, 1, 1], [1, 1, 1, 1],
         [1, 1, 1, 1], ['AM', 'AM', 'PM', 'PM']),
        (day1, [100, 100, 100, 102], [5, 5, 5, 20], [10, 12, 11, 12], [9, 8, 7, 6],
         [1, 2, 9, 3], ['AM', 'AM', 'PM', 'PM']),
    ):
        start = pd.Timestamp(day) + pd.Timedelta(9, unit='h')
        idx = pd.date_range(start, periods=4, freq='1min')
        rows.append(pd.DataFrame({
            'close': close, 'volume': volume, 'total_turnover': turnover,
            'highw': high, 'loww': low, 'session': session,
            'gamma_norm': np.arange(4) / 3, 'trading_date': day,
        }, index=idx))
    out = F.report_factors(pd.concat(rows), lookback=1, pct=55)
    got = out.loc[day1]
    assert got['pmt'] == pytest.approx(2 / 3)
    assert got['vmt'] == pytest.approx(2 / 3)
    assert got['vd_ratio'] == pytest.approx((0.5) / 1.5)
    assert got['ts_volume'] == pytest.approx(1.0)
    assert got['ts_turnover'] == pytest.approx(2 / 3)
    assert got['ts_high_pm'] == pytest.approx(1.0)
    assert got['ts_low_pm'] == pytest.approx(1.0)
    assert got['spike_am'] == pytest.approx(1.0)
    assert got['vol_pm'] == pytest.approx(1.0)


def test_night_volume_does_not_enter_the_session_ratio_or_afternoon_flag():
    day0 = pd.Timestamp('2016-01-04')
    day1 = pd.Timestamp('2016-01-05')
    frames = []
    for day, volume, session in (
        (day0, [10, 20, 10, 20, 10], ['NIGHT', 'AM', 'AM', 'PM', 'PM']),
        (day1, [100, 1, 1, 1, 1], ['NIGHT', 'AM', 'AM', 'PM', 'PM']),
    ):
        start = pd.Timestamp(day) + pd.Timedelta(21, unit='h')
        idx = pd.date_range(start, periods=5, freq='1h')
        frames.append(pd.DataFrame({
            'close': [100, 101, 100, 101, 100],
            'volume': volume,
            'total_turnover': volume,
            'highw': [1, 2, 1, 1, 1],
            'loww': [1, 1, 1, 1, 1],
            'session': session,
            'gamma_norm': np.arange(5) / 4,
            'trading_date': day,
        }, index=idx))
    out = F.report_factors(pd.concat(frames), lookback=1, pct=55).loc[day1]
    assert out['vol_pm'] != out['vol_pm']  # 最大量在夜盘，不是上午/下午
    assert np.isfinite(out['vd_ratio'])
    assert out['ts_volume'] == pytest.approx(0.0)
