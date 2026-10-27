"""Factor duration, timestamps, cache, and precision contracts."""

# --- test_correctness.py ---

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import save_shard
from tfcta.factors import _read, combo_dir, duration_factors, duration_one_day


def test_float32_duration_does_not_depend_on_adjustment_offset():
    # Raw quotes on 0.1 grid; exact duration [0,1,1,1]. Adjusted floats
    # used to give different durations at threshold equality.
    raw = np.array([400, 400.1, 400.2, 400.1], dtype='float32')
    frames = []
    for offset in (0., -211.23):
        frames.append(pd.DataFrame({
            'close': raw, 'closew': (raw + offset).astype('float32'),
            'trading_date': pd.Timestamp('2021-06-15'),
            'gamma_norm': np.arange(4)/3, 'session': 'AM',
        }))
    threshold = pd.Series({0: .1})
    a, b = [duration_factors(f, 1, 55, threshold) for f in frames]
    pd.testing.assert_frame_equal(a, b)
    assert a['dfp_max'].iloc[0] == pytest.approx(0.)
    assert a['dfp_top3'].iloc[0] == pytest.approx((400.1+400.2+400.1)/3/400.1-1)
    promoted = frames[0].astype({'close':'float64','closew':'float64'})
    pd.testing.assert_frame_equal(a,duration_factors(promoted,1,55,threshold))


def test_zero_threshold_is_uninformative_not_a_one_bar_duration():
    assert np.isnan(duration_one_day(np.ones(4), 0)).all()


def test_each_cache_file_requires_its_own_formula_version(tmp_path):
    directory = combo_dir(1, 55, tmp_path)
    p = save_shard(pd.DataFrame({'dfp_max':[0.], 'dfp_top3':[0.]},
        index=pd.to_datetime(['2021-01-04'])), directory, 'OLD', 'pickle')
    (directory / "_version.json").write_text('{"version": 3}')  # cannot validate OLD
    with pytest.raises(RuntimeError):
        _read(p, ['dfp_max', 'dfp_top3'])


def test_future_date_in_research_factor_cache_is_rejected(tmp_path):
    p = save_shard(pd.DataFrame({'dfp_max':[0.], 'dfp_top3':[0.]},
        index=pd.to_datetime(['2023-01-04'])), tmp_path, 'BAD', 'pickle')
    p.with_suffix(p.suffix+'.json').write_text('{"version": 3}')
    with pytest.raises(C.HoldoutViolation):
        _read(p, ['dfp_max', 'dfp_top3'])


# --- test_duration.py ---

import numpy as np
import pandas as pd
import pytest

from tfcta.factors import duration_one_day, duration_series, intraday_abs_diff, rolling_threshold


def brute_duration(values, threshold):
    """论文定义的逐字翻译：对每个 i 向前扫描，取最近的满足者；无则从开盘累计。"""
    v = np.asarray(values, dtype='float64')
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if not np.isfinite(v[i]):
            continue
        hit = None
        for j in range(i - 1, -1, -1):
            if np.isfinite(v[j]) and abs(v[i] - v[j]) >= threshold:
                hit = j
                break
        out[i] = (i - hit) if hit is not None else i
    return out


def test_matches_brute_force_on_random_walks():
    """300 组随机样本上逐元素比对向量化实现与暴力实现。"""
    rng = np.random.default_rng(42)
    for _ in range(300):
        n = int(rng.integers(2, 60))
        v = 100 + np.cumsum(rng.normal(0, 1, n))
        thr = float(rng.uniform(0.2, 3.0))
        np.testing.assert_array_equal(duration_one_day(v, thr), brute_duration(v, thr))


def test_no_qualifying_history_accumulates_from_open():
    """阈值极大时无任何满足者，持续期应为 0,1,2,...（从开盘累计）而非全 0。"""
    v = np.array([100.0, 100.1, 100.2, 100.15, 100.05])
    got = duration_one_day(v, threshold=1e9)
    np.testing.assert_array_equal(got, np.arange(5, dtype='float64'))


def test_picks_most_recent_not_earliest():
    """必须取最近的满足者。"""
    v = np.array([100.0, 110.0, 120.0, 130.0])
    got = duration_one_day(v, threshold=5.0)
    assert got[3] == 1.0


def test_nan_values_propagate_not_zero():
    v = np.array([100.0, np.nan, 102.0, 105.0])
    got = duration_one_day(v, threshold=1.0)
    assert np.isnan(got[1])
    assert np.isfinite(got[2]) and np.isfinite(got[3])


def test_does_not_cross_days():
    """每日独立计算：第二日首根必须重新从 0 起，不得沿用前一日的历史。"""
    v = np.array([100.0, 101.0, 102.0, 200.0, 201.0, 202.0])
    days = np.array([0, 0, 0, 1, 1, 1])
    thr = pd.Series({0: 0.5, 1: 0.5})
    got = duration_series(v, days, thr)
    np.testing.assert_array_equal(got, np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0]))


def test_abs_diff_no_cross_day():
    v = np.array([100.0, 101.0, 500.0, 501.0])
    days = np.array([0, 0, 1, 1])
    d = intraday_abs_diff(v, days)
    assert np.isnan(d[0]) and np.isnan(d[2])   # 每日首根
    np.testing.assert_allclose(d[[1, 3]], [1.0, 1.0])


def test_rolling_threshold_excludes_current_day():
    """当日样本不得进入自身阈值，否则前视。"""
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(6), 20)
    v = 100 + np.cumsum(rng.normal(0, 1, len(days)))
    d = intraday_abs_diff(v, days)
    thr = rolling_threshold(d, days, lookback=3, pct=55.0)
    assert np.isnan(thr.iloc[0])
    assert thr.iloc[:3].isna().all()
    assert thr.iloc[3:].notna().all()


def test_rolling_threshold_pools_not_averages():
    """必须把 N 日样本汇总成一个池子再取分位数，而非"每日分位数再平均"。"""
    days = np.r_[np.zeros(60), np.ones(60), np.full(10, 2), np.full(5, 3)]
    d = np.concatenate([
        np.full(60, np.nan),                       # 第 0 日：无历史，不参与
        np.r_[np.nan, np.full(59, 1.0)],           # 第 1 日：59 个 1
        np.r_[np.nan, np.full(9, 100.0)],          # 第 2 日：9 个 100
        np.full(5, np.nan),                        # 第 3 日：取值无关，只看历史
    ])
    thr = rolling_threshold(d, days, lookback=2, pct=50.0)

    assert thr.loc[2] == pytest.approx(1.0)        # 第 2 日只看第 1 日
    # 第 3 日池子 = 59 个 1 + 9 个 100 -> 中位数被样本多的一侧主导
    assert thr.loc[3] == pytest.approx(1.0)
    # 若错误实现为"每日分位数再平均"，会得到 (1 + 100) / 2
    assert thr.loc[3] != pytest.approx(50.5)


# --- test_factor_cache.py ---

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import find_shard, save_shard
from tests.synthetic import make_symbol
from tfcta.factors import (
    build_symbol, combo_dir, intraday_abs_diff, load_symbol, rolling_threshold,
    rolling_threshold_grid, timestamp_dir,
)


def _diff_and_codes(n_days=40, bars=30, seed=5):
    rng = np.random.default_rng(seed)
    vals, codes = [], []
    price = 100.0
    for d in range(n_days):
        px = np.round(price + np.cumsum(rng.normal(0, 0.3, bars)), 1)
        price = float(px[-1])
        vals.append(px)
        codes.append(np.full(bars, d))
    v, c = np.concatenate(vals), np.concatenate(codes)
    return intraday_abs_diff(v, c), c


def test_threshold_grid_matches_individual_calls():
    diff, codes = _diff_and_codes()
    lbs, pcts = [5, 10], [50.0, 52.5, 55.0]
    grid = rolling_threshold_grid(diff, codes, lbs, pcts)
    for lb in lbs:
        for p in pcts:
            one = rolling_threshold(diff, codes, lb, p)
            got = grid[(lb, p)]
            assert list(got.index) == list(one.index)
            np.testing.assert_allclose(got.to_numpy(), one.to_numpy(),
                                       equal_nan=True, rtol=0, atol=0)


def _tiny_cache(night_class='night_2300', days=60, seed=4):
    """造一个品种的分钟分片，跑 build_symbol，返回 (root, symbol, combos)。"""
    root = Path(tempfile.mkdtemp(prefix='tfcta_fc_'))
    idx = pd.bdate_range('2016-01-04', periods=days)
    df = make_symbol(idx, night_class=night_class, seed=seed)
    save_shard(df[C.FACTOR_FIELDS], root / 'shards', 'XX')
    combos = [(10, 50.0), (10, 55.0), (20, 55.0)]
    return root, df, combos


def _build(root, combos, **kw):
    """build_symbol 默认从 C.RESEARCH_DIR 读，测试里改读临时目录。"""
    orig = C.RESEARCH_DIR
    C.RESEARCH_DIR = root / 'shards'
    try:
        return build_symbol('XX', combos, root=root / 'factors', **kw)
    finally:
        C.RESEARCH_DIR = orig


def test_build_symbol_writes_one_file_per_combo_plus_one_timestamp():
    root, _, combos = _tiny_cache()
    info = _build(root, combos)
    assert info['combos_written'] == len(combos)
    assert info['timestamp_written'] is True
    for n, m in combos:
        assert find_shard(combo_dir(n, m, root / 'factors'), 'XX') is not None
    assert find_shard(timestamp_dir(root / 'factors'), 'XX') is not None


def test_build_symbol_is_resumable():
    """已存在即跳过——42 品种 × 15 组合的重算代价很高，续跑不是可选项。"""
    root, _, combos = _tiny_cache()
    _build(root, combos)
    again = _build(root, combos)
    assert again['skipped'] is True
    assert again['combos_written'] == 0


def test_timestamp_factors_identical_across_combos():
    """时间戳族不依赖 (N, M)：两个组合读出来的时间戳列必须逐元素相同。"""
    root, _, combos = _tiny_cache()
    _build(root, combos)
    a = load_symbol('XX', *combos[0], root=root / 'factors')
    b = load_symbol('XX', *combos[2], root=root / 'factors')
    dur_cols = set(load_symbol('XX', *combos[0], root=root / 'factors',
                                  with_timestamp=False).columns)
    ts_cols = [c for c in a.columns if c not in dur_cols]
    assert ts_cols == ['ts_high', 'ts_low']
    pd.testing.assert_frame_equal(a[ts_cols], b[ts_cols], check_exact=True)


def test_duration_factors_differ_across_combos():
    """反过来，持续期族必须随参数变化——否则说明阈值没接上。"""
    root, _, combos = _tiny_cache()
    _build(root, combos)
    a = load_symbol('XX', *combos[0], root=root / 'factors')
    b = load_symbol('XX', *combos[2], root=root / 'factors')
    assert not a['dfp_max'].equals(b['dfp_max'])


def test_injected_thresholds_match_recomputed():
    """build_symbol 走的是注入阈值的快路径，结果必须与 duration_factors 自算一致。"""
    from tfcta.data import add_intraday_coords
    from tfcta.factors import duration_factors
    root, df, combos = _tiny_cache()
    _build(root, combos)
    n, m = combos[0]
    cached = load_symbol('XX', n, m, root=root / 'factors', with_timestamp=False)
    direct = duration_factors(add_intraday_coords(df[C.FACTOR_FIELDS]),
                              lookback=n, pct=m)
    pd.testing.assert_frame_equal(cached[direct.columns], direct, check_exact=True)


def test_load_symbol_raises_clearly_when_missing():
    root, _, combos = _tiny_cache()
    with pytest.raises(FileNotFoundError, match='prepare/factors.py'):
        load_symbol('XX', 999, 55.0, root=root / 'factors')


# --- test_factors.py ---

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import add_intraday_coords
from tests.synthetic import make_symbol
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


# --- test_report_factors.py ---

import numpy as np
import pandas as pd
import pytest

from tfcta.factors import _window_clock, report_factors


def test_window_clock_is_first_extreme_inside_the_window():
    values = np.array([1.0, 5.0, 5.0, 0.0])
    mask = np.array([False, True, True, True])
    assert _window_clock(values, mask, 'max') == pytest.approx(0.0)
    assert _window_clock(values, mask, 'min') == pytest.approx(1.0)


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
    out = report_factors(pd.concat(rows), lookback=1, pct=55)
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
    out = report_factors(pd.concat(frames), lookback=1, pct=55).loc[day1]
    assert out['vol_pm'] != out['vol_pm']  # 最大量在夜盘，不是上午/下午
    assert np.isfinite(out['vd_ratio'])
    assert out['ts_volume'] == pytest.approx(0.0)
