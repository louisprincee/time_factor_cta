"""火富牛 2022 复盘里写明的动量、波动和均价突破。"""
import numpy as np
import pandas as pd

from tfcta.factors import daily


def _idx(n, start="2016-01-04"):
    return pd.bdate_range(start, periods=n)


def test_three_day_momentum_uses_only_its_own_past():
    idx = _idx(8)
    close = pd.DataFrame({"A": [100, 101, 102, 103, 104, 90, 91, 92]}, index=idx)
    sign = daily.tsmom_sign(close, close.copy(), window=3)["A"]
    assert np.isnan(sign.iloc[2])
    assert sign.iloc[3] == 1.0
    assert sign.iloc[5] == -1.0
    later = close.copy()
    later.iloc[-1, 0] = 1.0
    unchanged = daily.tsmom_sign(later, later, window=3)["A"]
    pd.testing.assert_series_equal(sign.iloc[:-1], unchanged.iloc[:-1])


def test_moving_average_breakout_is_above_or_below_the_trailing_average():
    idx = _idx(21)
    closew = pd.DataFrame({"A": [100.0] * 20 + [110.0]}, index=idx)
    side = daily.ma_breakout(closew, window=20)["A"]
    assert side.iloc[:19].isna().all()
    assert side.iloc[19] == 0.0
    assert side.iloc[20] == 1.0


def test_volatility_change_turns_positive_only_after_the_larger_moves():
    idx = _idx(80)
    small = np.resize([0.001, -0.001], 40)
    large = np.resize([0.02, -0.02], 40)
    returns = np.r_[small, large]
    close = pd.DataFrame({"A": 100 * np.cumprod(1 + returns)}, index=idx)
    change = daily.vol_change_sign(close, close.copy(), window=20)["A"]
    assert change.iloc[:39].isna().all()
    assert change.iloc[59] == 1.0


def test_tails_long_the_top_five_and_ignore_names_outside_the_year():
    idx = _idx(2, start="2020-01-02")
    names = list("ABCDEFGHIJ")
    factor = pd.DataFrame({name: [i, i] for i, name in enumerate(names, start=1)}, index=idx)
    factor["OUT"] = 100.0
    side = daily.cross_sectional_tails(factor, {2020: names}, n=5)
    assert list(side.loc[idx[-1], list("ABCDE")]) == [-1, -1, -1, -1, -1]
    assert list(side.loc[idx[-1], list("FGHIJ")]) == [1, 1, 1, 1, 1]
    assert side["OUT"].isna().all()
    too_few = daily.cross_sectional_tails(factor[names[:9]], {2020: names[:9]}, n=5)
    assert too_few.isna().all().all()


def test_identical_cross_section_has_no_direction():
    names = list('ABCDEFGHIJ')
    frame = pd.DataFrame([np.ones(10)], index=_idx(1, '2021-01-04'), columns=names)
    assert daily.cross_sectional_tails(frame, {2021: names}, n=5).eq(0).all().all()


def test_boundary_ties_are_symmetric_under_sign_reversal():
    names = list('ABCDEFGHIJ')
    frame = pd.DataFrame([[1, 1, 2, 3, 4, 5, 6, 6, 6, 7]],
                         index=_idx(1, '2021-01-04'), columns=names)
    positive = daily.cross_sectional_tails(frame, {2021: names}, n=3)
    negative = daily.cross_sectional_tails(-frame, {2021: names}, n=3)
    pd.testing.assert_frame_equal(positive, -negative)
