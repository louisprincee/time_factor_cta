"""日频因子缓存测试（设计文档第 10.2 节 + 第 11 节第 4 行验收）。

这一层最值得钉住的三件事
------------------------
1. ``rolling_threshold_grid`` 与逐个调用 ``rolling_threshold`` **逐元素相同**。
   它是纯优化（15 遍降到 3 遍），一旦结果有偏差，后面所有因子都会悄悄变，
   而且不会有任何报错。
2. 时间戳族**不随 (N, M) 变化**。如果哪天有人把它写进了按组合重算的分支，
   数值不会变但会白算 15 遍；更糟的是若写错成依赖阈值，方向先验就失效了。
3. 夜盘类因子在无夜盘品种上是 NaN 而**不是 0**，且验收口径要按有无夜盘分组，
   否则一个正确结果会被判成不合格，诱导出 fillna(0) 这个致命修法。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import shard_io, synth
from tfcta.factors import intraday as D
from tfcta.factors import cache as FC


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
    return D.intraday_abs_diff(v, c), c


def test_threshold_grid_matches_individual_calls():
    diff, codes = _diff_and_codes()
    lbs, pcts = [5, 10], [50.0, 52.5, 55.0]
    grid = D.rolling_threshold_grid(diff, codes, lbs, pcts)
    for lb in lbs:
        for p in pcts:
            one = D.rolling_threshold(diff, codes, lb, p)
            got = grid[(lb, p)]
            assert list(got.index) == list(one.index)
            np.testing.assert_allclose(got.to_numpy(), one.to_numpy(),
                                       equal_nan=True, rtol=0, atol=0)


def _tiny_cache(night_class='night_2300', days=60, seed=4):
    """造一个品种的分钟分片，跑 build_symbol，返回 (root, symbol, combos)。"""
    root = Path(tempfile.mkdtemp(prefix='tfcta_fc_'))
    idx = pd.bdate_range('2016-01-04', periods=days)
    df = synth.make_symbol(idx, night_class=night_class, seed=seed)
    shard_io.save_shard(df[C.FACTOR_FIELDS], root / 'shards', 'XX')
    combos = [(10, 50.0), (10, 55.0), (20, 55.0)]
    return root, df, combos


def _build(root, combos, **kw):
    """build_symbol 默认从 C.RESEARCH_DIR 读，测试里改读临时目录。"""
    orig = C.RESEARCH_DIR
    C.RESEARCH_DIR = root / 'shards'
    try:
        return FC.build_symbol('XX', combos, root=root / 'factors', **kw)
    finally:
        C.RESEARCH_DIR = orig


def test_build_symbol_writes_one_file_per_combo_plus_one_timestamp():
    root, _, combos = _tiny_cache()
    info = _build(root, combos)
    assert info['combos_written'] == len(combos)
    assert info['timestamp_written'] is True
    for n, m in combos:
        assert shard_io.find_shard(FC.combo_dir(n, m, root / 'factors'), 'XX') is not None
    assert shard_io.find_shard(FC.timestamp_dir(root / 'factors'), 'XX') is not None


def test_build_symbol_is_resumable():
    """已存在即跳过——42 品种 × 15 组合的重算代价很高，续跑不是可选项。"""
    root, _, combos = _tiny_cache()
    _build(root, combos)
    again = _build(root, combos)
    assert again['skipped'] is True
    assert again['combos_written'] == 0


def test_timestamp_factors_identical_across_combos():
    """时间戳族不依赖 (N, M)：两个组合读出来的时间戳列必须逐元素相同。

    时间戳列不用前缀猜（cnt_high_am / is_high_am 没有 ts_ 前缀），而是用
    "含时间戳 - 不含时间戳" 的差集算出来，这样新增因子也不会漏检。
    """
    root, _, combos = _tiny_cache()
    _build(root, combos)
    a = FC.load_symbol('XX', *combos[0], root=root / 'factors')
    b = FC.load_symbol('XX', *combos[2], root=root / 'factors')
    dur_cols = set(FC.load_symbol('XX', *combos[0], root=root / 'factors',
                                  with_timestamp=False).columns)
    ts_cols = [c for c in a.columns if c not in dur_cols]
    assert ts_cols == ['ts_high', 'ts_low']
    pd.testing.assert_frame_equal(a[ts_cols], b[ts_cols], check_exact=True)


def test_duration_factors_differ_across_combos():
    """反过来，持续期族必须随参数变化——否则说明阈值没接上。"""
    root, _, combos = _tiny_cache()
    _build(root, combos)
    a = FC.load_symbol('XX', *combos[0], root=root / 'factors')
    b = FC.load_symbol('XX', *combos[2], root=root / 'factors')
    assert not a['dfp_max'].equals(b['dfp_max'])


def test_injected_thresholds_match_recomputed():
    """build_symbol 走的是注入阈值的快路径，结果必须与 duration_factors 自算一致。"""
    from tfcta.data import sessions
    from tfcta.factors.intraday import duration_factors
    root, df, combos = _tiny_cache()
    _build(root, combos)
    n, m = combos[0]
    cached = FC.load_symbol('XX', n, m, root=root / 'factors', with_timestamp=False)
    direct = duration_factors(sessions.add_intraday_coords(df[C.FACTOR_FIELDS]),
                              lookback=n, pct=m)
    pd.testing.assert_frame_equal(cached[direct.columns], direct, check_exact=True)


def test_load_symbol_raises_clearly_when_missing():
    root, _, combos = _tiny_cache()
    with pytest.raises(FileNotFoundError, match='step3'):
        FC.load_symbol('XX', 999, 55.0, root=root / 'factors')


def _panel(rows: dict) -> pd.DataFrame:
    """由 {品种: {因子: [值...]}} 构造 (trading_date, symbol) 长表。"""
    parts = []
    for sym, cols in rows.items():
        n = len(next(iter(cols.values())))
        idx = pd.MultiIndex.from_product(
            [pd.bdate_range('2016-01-04', periods=n), [sym]],
            names=['trading_date', 'symbol'])
        parts.append(pd.DataFrame(cols, index=idx))
    return pd.concat(parts).sort_index()


def test_acceptance_flags_structural_zeros():
    """全 0 必须判不通过。结构性零值会把因子压成噪声。"""
    panel = _panel({'RB': {'ts_high': [0.0] * 10}})
    t = FC.check_acceptance(FC.panel_health(panel))
    assert not bool(t.loc['ts_high', 'passed'])
    assert '全为 0' in t.loc['ts_high', 'reason']


def test_timepoint_outside_unit_interval_fails():
    panel = _panel({'RB': {'ts_high': [1.5] * 10, 'dfp_max': [0.01] * 10}})
    t = FC.check_acceptance(FC.panel_health(panel))
    assert not bool(t.loc['ts_high', 'passed'])
    assert bool(t.loc['dfp_max', 'passed'])
