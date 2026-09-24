"""第 3 步持续期抽查的逻辑测试（scripts/step3_build_factors.py）。

抽查脚本本身有一处容易做错、且做错了也不会报错的地方：**统计范围**。
阈值需要 N 个交易日的预热，所以每次都要多读两年数据；如果汇总时忘记把预热年
排除，2016 年的诊断里就会混进 2014-2015 年的行为，而且因为预热年的持续期大量为
NaN，NaN 占比会被稀释——看起来更"干净"，实则是错的。下面把这条钉死。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tfcta.factors import duration as D


def _load_step3():
    p = Path(__file__).resolve().parents[1] / 'scripts' / 'step3_build_factors.py'
    spec = importlib.util.spec_from_file_location('step3_build_factors', p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['step3_build_factors'] = mod
    spec.loader.exec_module(mod)
    return mod


S3 = _load_step3()


def _walk(n_days=8, bars=40, seed=3):
    """构造多日随机游走 + 日编码，价格离散到 tick（否则一阶差分处处非零）。"""
    rng = np.random.default_rng(seed)
    vals, codes = [], []
    price = 100.0
    for d in range(n_days):
        step = rng.normal(0, 0.3, bars)
        px = np.round(price + np.cumsum(step), 1)
        price = float(px[-1])
        vals.append(px)
        codes.append(np.full(bars, d))
    return np.concatenate(vals), np.concatenate(codes)


def test_probe_counts_only_kept_rows():
    v, codes = _walk()
    keep = codes >= 4
    r = S3.probe(v, codes, keep, lookback=3, pct=55.0)
    assert r['n_rows'] == int(keep.sum())
    assert r['n'] <= r['n_rows']


def test_probe_stats_ignore_warmup_rows():
    """预热年的行不得进入分布。

    构造"品种在预热年尚未上市（全 NaN）、目标年正常"的情形——这是真实数据里最常
    见的样子。若统计混入了预热年，NaN 占比会被那些结构性缺失的行拉高，看上去像
    因子有大面积缺失；反过来，若预热年数据完整而目标年缺失，NaN 占比会被稀释、
    问题被掩盖。两个方向都会误导，所以统计范围必须严格限定在目标年。
    """
    v, codes = _walk()
    v = v.copy()
    v[codes < 4] = np.nan                     # 前 4 日未上市
    keep = codes >= 4
    only_late = S3.probe(v, codes, keep, lookback=3, pct=55.0)
    everything = S3.probe(v, codes, np.ones(len(v), bool), lookback=3, pct=55.0)

    assert only_late['n_rows'] < everything['n_rows']
    assert only_late['n'] == everything['n']          # 有效观测本来就只在目标年
    assert only_late['nan_ratio'] < everything['nan_ratio']


def test_probe_does_not_change_duration_values():
    """keep 只影响汇总范围，不影响任何一行的持续期取值（持续期逐日独立）。"""
    v, codes = _walk()
    diff = D.intraday_abs_diff(v, codes)
    thr = D.rolling_threshold(diff, codes, 3, 55.0)
    dur = D.duration_series(v, codes, thr)
    keep = codes >= 5
    r = S3.probe(v, codes, keep, lookback=3, pct=55.0)
    d = dur[keep]
    d = d[np.isfinite(d)]
    assert r['n'] == d.size
    assert r['p50'] == float(np.percentile(d, 50))
    assert r['max'] == float(d.max())


def test_probe_all_nan_is_reported_not_crashed():
    """预热不足时整段为 NaN，必须如实报告 all_nan 而不是抛异常或返回 0。"""
    v, codes = _walk(n_days=3, bars=10)
    keep = codes == 0                          # 第 0 日阈值必为 NaN
    r = S3.probe(v, codes, keep, lookback=250, pct=55.0)
    assert r['all_nan'] is True
    assert r['n'] == 0
    assert r['nan_ratio'] == 1.0
    assert np.isnan(r['p50'])


def test_probe_reports_skew_ratio_consistently():
    v, codes = _walk(n_days=20, bars=60, seed=9)
    keep = codes >= 5
    r = S3.probe(v, codes, keep, lookback=4, pct=55.0)
    assert r['p50'] > 0
    assert r['skew_ratio'] == r['p95'] / r['p50']


def test_probe_diagnostics_restricted_to_kept_rows():
    """零变动占比也必须只算目标年——否则一个"整年停牌"的预热年会把它拉高。"""
    v, codes = _walk(n_days=10, bars=30)
    v = v.copy()
    v[codes < 5] = 100.0                       # 前 5 日零变动
    late = S3.probe(v, codes, codes >= 5, lookback=3, pct=55.0)
    allrows = S3.probe(v, codes, np.ones(len(v), bool), lookback=3, pct=55.0)
    assert late['zero_ratio'] < allrows['zero_ratio']


def test_skew_gate_constant_is_three():
    """验收门槛写死为 3，与设计文档第 11 节第 3 行一致；改动需同步改文档。"""
    assert S3.SKEW_MIN == 3.0


def test_probe_cols_cover_price_and_volume():
    """价格持续期与成交量持续期都要抽查——论文两个家族都用到。"""
    assert 'closew' in S3.PROBE_COLS and 'volume' in S3.PROBE_COLS


def test_probe_accepts_series_input():
    """脚本里传的是 to_numpy 的结果，但接口不应对 Series 报错。"""
    v, codes = _walk()
    r = S3.probe(pd.Series(v).to_numpy(dtype='float64'), codes,
                 codes >= 4, lookback=3, pct=55.0)
    assert not r['all_nan']
