import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.factors import intraday, cache
from tfcta.research import stats
from tfcta.data import shard_io


def test_drawdown_includes_starting_capital():
    assert stats.performance(pd.Series([-.10, .02]))['max_drawdown'] == pytest.approx(-.10)


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
    a, b = [intraday.duration_factors(f, 1, 55, threshold) for f in frames]
    pd.testing.assert_frame_equal(a, b)
    assert a['dfp_max'].iloc[0] == pytest.approx(0.)
    assert a['dfp_top3'].iloc[0] == pytest.approx((400.1+400.2+400.1)/3/400.1-1)
    promoted = frames[0].astype({'close':'float64','closew':'float64'})
    pd.testing.assert_frame_equal(a,intraday.duration_factors(promoted,1,55,threshold))


def test_zero_threshold_is_uninformative_not_a_one_bar_duration():
    assert np.isnan(intraday.duration_one_day(np.ones(4), 0)).all()


def test_each_cache_file_requires_its_own_formula_version(tmp_path):
    directory = cache.combo_dir(1, 55, tmp_path)
    p = shard_io.save_shard(pd.DataFrame({'dfp_max':[0.], 'dfp_top3':[0.]},
        index=pd.to_datetime(['2021-01-04'])), directory, 'OLD', 'pickle')
    (directory / "_version.json").write_text('{"version": 3}')  # cannot validate OLD
    with pytest.raises(RuntimeError):
        cache._read(p, ['dfp_max', 'dfp_top3'])


def test_future_date_in_research_factor_cache_is_rejected(tmp_path):
    p = shard_io.save_shard(pd.DataFrame({'dfp_max':[0.], 'dfp_top3':[0.]},
        index=pd.to_datetime(['2023-01-04'])), tmp_path, 'BAD', 'pickle')
    p.with_suffix(p.suffix+'.json').write_text('{"version": 3}')
    with pytest.raises(C.HoldoutViolation):
        cache._read(p, ['dfp_max', 'dfp_top3'])
