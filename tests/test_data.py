"""Data partition, session, universe, and input contracts."""

# --- test_rqdata.py ---

import json
import os
from datetime import date

import pandas as pd

from download.commodity.getRiceQuantExternalData import (
    CoverageCache,
    merge_frames,
    missing_ranges,
    remove_unavailable_cache,
    split_period,
    trading_in,
)
from tfcta.rq_auth import load_rqdata_env, rqdata_credentials


def test_split_period_partitions_crossing_range_without_overlap():
    partitions = split_period(date(2021, 12, 30), date(2023, 1, 2))
    assert partitions == [
        ("research", date(2021, 12, 30), date(2021, 12, 31)),
        ("validation_2022", date(2022, 1, 1), date(2022, 12, 31)),
        ("holdout_locked", date(2023, 1, 1), date(2023, 1, 2)),
    ]


def test_split_period_research_only():
    assert split_period(date(2020, 1, 1), date(2020, 1, 2)) == [
        ("research", date(2020, 1, 1), date(2020, 1, 2))
    ]


def test_missing_ranges_only_returns_uncovered_dates():
    gaps = missing_ranges(
        date(2020, 1, 1), date(2020, 1, 10),
        [["20200103", "20200105"], ["20200108", "20200108"]],
    )
    assert gaps == [
        (date(2020, 1, 1), date(2020, 1, 2)),
        (date(2020, 1, 6), date(2020, 1, 7)),
        (date(2020, 1, 9), date(2020, 1, 10)),
    ]


def test_missing_ranges_merges_adjacent_coverage():
    gaps = missing_ranges(
        date(2020, 1, 1), date(2020, 1, 10),
        [["20200103", "20200105"], ["20200106", "20200108"]],
    )
    assert gaps == [(date(2020, 1, 1), date(2020, 1, 2)),
                    (date(2020, 1, 9), date(2020, 1, 10))]


def test_merge_frames_keeps_new_values_for_duplicate_index():
    old = pd.DataFrame({"value": [1, 2]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    new = pd.DataFrame({"value": [20, 3]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    merged = merge_frames(old, new)
    assert merged["value"].tolist() == [1, 20, 3]


def test_cache_requests_only_uncovered_ranges(tmp_path):
    cache = CoverageCache(tmp_path)
    calls = []

    def request(start, end):
        calls.append((start, end))
        return pd.DataFrame({"value": [len(calls)]}, index=[pd.Timestamp(start)])

    cache.fetch("roll_yield", "CU_main_sub", date(2020, 1, 1), date(2020, 1, 3), request)
    cache.fetch("roll_yield", "CU_main_sub", date(2020, 1, 2), date(2020, 1, 4), request)
    cache.flush()

    assert calls == [("20200101", "20200103"), ("20200104", "20200104")]
    assert (tmp_path / "roll_yield" / "CU_main_sub.pkl").exists()
    assert (tmp_path / "coverage.json").exists()


def test_failed_request_is_not_marked_as_covered(tmp_path):
    cache = CoverageCache(tmp_path)

    def request(_start, _end):
        raise RuntimeError("service unavailable")

    cache.fetch("warehouse", "CU", date(2020, 1, 1), date(2020, 1, 2), request)

    assert cache.failures
    assert cache.manifest["items"]["warehouse/CU"]["coverage"] == []


def test_none_response_is_not_marked_as_covered(tmp_path):
    cache = CoverageCache(tmp_path)
    cache.fetch("member_rank", "CU_long", date(2020, 1, 1), date(2020, 1, 2),
                lambda _start, _end: None, require_nonempty=True)

    assert cache.failures
    assert cache.manifest["items"]["member_rank/CU_long"]["coverage"] == []


def test_empty_member_rank_cache_is_invalidated(tmp_path):
    cache = CoverageCache(tmp_path)
    entry = cache._entry("member_rank", "CU_long")
    entry["coverage"] = [["20200101", "20200102"]]
    path = tmp_path / entry["file"]
    path.parent.mkdir(parents=True)
    pd.DataFrame().to_pickle(path)
    cache.flush()
    calls = []

    cache.fetch("member_rank", "CU_long", date(2020, 1, 1), date(2020, 1, 2),
                lambda start, end: calls.append((start, end)) or None,
                require_nonempty=True)

    assert calls == [("20200101", "20200102")]
    assert cache.manifest["items"]["member_rank/CU_long"]["coverage"] == []


def test_empty_dataset_cache_can_be_invalidated(tmp_path):
    cache = CoverageCache(tmp_path)
    entry = cache._entry("spot", "daily_AU9999.SGEX")
    entry["coverage"] = [["20200101", "20200102"]]
    path = tmp_path / entry["file"]
    path.parent.mkdir(parents=True)
    pd.DataFrame().to_pickle(path)
    cache.flush()

    cache.invalidate_empty("spot", key_prefix="daily_")

    assert cache.manifest["items"]["spot/daily_AU9999.SGEX"]["coverage"] == []
    assert not path.exists()


def test_dry_run_does_not_invalidate_or_delete_cache(tmp_path):
    cache = CoverageCache(tmp_path)
    entry = cache._entry("spot", "daily_AU9999.SGEX")
    entry["coverage"] = [["20200101", "20200102"]]
    path = tmp_path / entry["file"]
    path.parent.mkdir(parents=True)
    pd.DataFrame().to_pickle(path)
    cache.flush()
    dry_cache = CoverageCache(tmp_path, dry_run=True)

    dry_cache.invalidate_empty("spot", key_prefix="daily_")

    assert path.exists()
    assert dry_cache.manifest["items"]["spot/daily_AU9999.SGEX"]["coverage"]


def test_remove_unavailable_cache_entries(tmp_path):
    root = tmp_path
    manifest = {
        "version": 1,
        "items": {
            "member_rank/CU_long": {"file": "member_rank/CU_long.pkl", "coverage": []},
            "dominant/CU_front_month": {"file": "dominant/CU_front_month.pkl", "coverage": []},
            "roll_yield/CU_near_main": {"file": "roll_yield/CU_near_main.pkl", "coverage": []},
            "roll_yield/CU_main_sub": {"file": "roll_yield/CU_main_sub.pkl", "coverage": []},
        },
    }
    (root / "coverage.json").write_text(json.dumps(manifest))

    removed = remove_unavailable_cache(root)
    saved = json.loads((root / "coverage.json").read_text())

    assert removed == 3
    assert set(saved["items"]) == {"roll_yield/CU_main_sub"}

def test_coverage_without_file_is_downloaded_again(tmp_path):
    cache = CoverageCache(tmp_path)
    cache._entry("warehouse", "CU")["coverage"] = [["20200101", "20200110"]]
    calls = []

    def request(left, right):
        calls.append((left, right))
        return pd.DataFrame({"on_warrant": [1.0]}, index=pd.to_datetime(["2020-01-02"]))

    cache.fetch("warehouse", "CU", date(2020, 1, 1), date(2020, 1, 10), request)
    assert calls == [("20200101", "20200110")]
    assert (tmp_path / "warehouse" / "CU.pkl").exists()


def test_trading_in_keeps_contracts_overlapping_window():
    instruments = pd.DataFrame({
        "order_book_id": ["CU1501", "CU2203", "CU2401", "CU9999"],
        "listed_date": ["2014-01-16", "2021-03-16", "2023-01-16", "0000-00-00"],
        "de_listed_date": ["2015-01-15", "2022-03-15", "2024-01-15", "0000-00-00"],
    })
    kept = trading_in(instruments, date(2022, 1, 1), date(2022, 12, 31))
    assert kept["order_book_id"].tolist() == ["CU2203", "CU9999"]


def test_load_rqdata_env_fills_blank_names_without_override(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)
    monkeypatch.setenv("RQDATAC_USERNAME", "already-set")
    path = tmp_path / "rqdata.env"
    path.write_text(
        "\n".join([
            "# comment",
            "RQDATAC_LICENSE=",
            "export RQDATAC_USERNAME='from-file'",
            'RQDATAC_PASSWORD="secret"',
            "",
        ]),
        encoding="utf-8",
    )

    load_rqdata_env(path)

    assert os.environ["RQDATAC_USERNAME"] == "already-set"
    assert os.environ["RQDATAC_PASSWORD"] == "secret"
    assert "RQDATAC_LICENSE" not in os.environ


def test_rqdata_credentials_prefers_license(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)
    path = tmp_path / "rqdata.env"
    path.write_text(
        "RQDATAC_LICENSE=token\nRQDATAC_USERNAME=user\nRQDATAC_PASSWORD=pw\n",
        encoding="utf-8",
    )

    assert rqdata_credentials(path) == ("token",)


def test_rqdata_credentials_uses_password_when_license_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)
    path = tmp_path / "rqdata.env"
    path.write_text("RQDATAC_USERNAME=user\nRQDATAC_PASSWORD=pw\n", encoding="utf-8")

    assert rqdata_credentials(path) == ("user", "pw")


def test_missing_env_file_leaves_credentials_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)

    assert rqdata_credentials(tmp_path / "absent.env") is None


# --- test_sessions.py ---

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import add_intraday_coords
from tests.synthetic import NIGHT_CLASS, make_symbol


DAYS = pd.bdate_range('2016-03-01', '2016-03-31')


@pytest.fixture(params=list(NIGHT_CLASS))


def sym(request):
    cls = request.param
    df = make_symbol(DAYS, night_class=cls, seed=1)
    return cls, add_intraday_coords(df)


def _full_days(df: pd.DataFrame) -> pd.DatetimeIndex:
    """去掉样本首个交易日。"""
    days = pd.DatetimeIndex(sorted(df['trading_date'].unique()))
    return days[1:]


def test_bar_grid_matches_expected(sym):
    """每日 bar 数必须命中 225 / 345 / 465 / 555 四种网格之一。"""
    cls, df = sym
    counts = df.groupby('trading_date').size().loc[_full_days(df)]
    assert counts.nunique() == 1
    assert counts.iloc[0] == C.EXPECTED_BARS_PER_DAY[cls]


def test_night_belongs_to_next_trading_date(sym):
    """夜盘 bar 的墙钟时刻必须早于其 trading_date 的日盘——本项目最关键的一条语义。"""
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
    open_clock = pd.to_datetime(td).map(
        lambda x: pd.Timestamp(year=x.year, month=x.month, day=x.day,
                               hour=C.AM_START.hour, minute=C.AM_START.minute))
    assert (ts < open_clock).all()
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
    df = make_symbol(DAYS[:3], seed=2).drop(columns=['trading_date'])
    with pytest.raises(KeyError, match='trading_date'):
        add_intraday_coords(df)


def test_unsorted_input_is_sorted_before_numbering():
    """乱序输入必须先排序再编号，否则 gamma 错乱且无任何报错。"""
    df = make_symbol(DAYS[:5], night_class='night_2300', seed=3)
    shuffled = df.sample(frac=1.0, random_state=0)
    out = add_intraday_coords(shuffled)
    assert out.index.is_monotonic_increasing
    ref = add_intraday_coords(df)
    pd.testing.assert_series_equal(out['gamma'], ref['gamma'])


# --- test_shard_io.py ---

from pathlib import Path
import pandas as pd
import pytest

from tfcta import config as C
import tfcta.data as data_module
from tfcta.data import (
    PICKLE_EXT, list_shards, load_oos_shard, load_shard, load_validation_shard,
    save_shard,
)


def _frame(start: str = '2021-11-01', days: int = 5) -> pd.DataFrame:
    td = pd.bdate_range(start, periods=days)
    idx = pd.DatetimeIndex([
        pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=9, minute=m + 1)
        for d in td for m in range(3)])
    return pd.DataFrame({
        'closew': range(len(idx)),
        'close': range(len(idx)),
        'volume': 1.0,
        'trading_date': [d for d in td for _ in range(3)],
    }, index=idx)


@pytest.fixture
def shard_dir():
    """独立临时目录。"""
    import tempfile
    return Path(tempfile.mkdtemp(prefix='tfcta_shardio_'))


def test_pickle_roundtrip_preserves_index_and_dtypes(shard_dir):
    df = _frame()
    p = save_shard(df, shard_dir, 'RB', fmt='pickle')
    assert p.suffix == PICKLE_EXT
    back = pd.read_pickle(p)
    pd.testing.assert_frame_equal(back, df)


def test_load_shard_refuses_holdout_directory():
    """指向 holdout_locked/ 必须抛 HoldoutViolation，哪怕文件根本不存在。"""
    with pytest.raises(C.HoldoutViolation):
        load_shard('RB', C.HOLDOUT_DIR)


def test_load_shard_refuses_holdout_subdirectory():
    with pytest.raises(C.HoldoutViolation):
        load_shard('RB', C.HOLDOUT_DIR / 'anything')


def test_load_shard_refuses_index_futures_holdout_directory():
    with pytest.raises(C.HoldoutViolation):
        load_shard('IM', C.INDEX_HOLDOUT_DIR)


def test_validation_reader_only_accepts_2022(shard_dir, monkeypatch):
    validation_dir = shard_dir / 'validation_2022'
    validation_dir.mkdir()
    df = _frame(start='2022-01-03', days=5)
    save_shard(df, validation_dir, 'RB', fmt='pickle')
    monkeypatch.setattr(C, 'VALIDATION_DIR', validation_dir)

    out = load_validation_shard('RB')
    assert out['trading_date'].dt.year.eq(2022).all()

    bad = _frame(start='2022-12-28', days=8)
    save_shard(bad, validation_dir, 'CU', fmt='pickle')
    with pytest.raises(C.HoldoutViolation, match='只含 2022'):
        load_validation_shard('CU')


def test_validation_reader_refuses_locked_source(shard_dir, monkeypatch):
    with pytest.raises(C.HoldoutViolation, match='必须来自'):
        C.assert_validation_only(C.HOLDOUT_DIR / 'RB.pkl')


def test_elapsed_oos_window_does_not_unlock_research(shard_dir, monkeypatch):
    monkeypatch.setattr(C, 'HOLDOUT_DIR', shard_dir / 'holdout_locked')
    monkeypatch.setattr(data_module, 'find_shard', lambda *a: pytest.fail('OOS file searched'))
    with pytest.raises(C.HoldoutViolation, match='封存'):
        load_oos_shard('RB', end='2025-12-31', today='2026-10-05')


def test_load_shard_rejects_holdout_dates_even_in_research_dir(shard_dir):
    """双重保险：即便文件放在 research/ 下，内容越界也必须拒绝。"""
    bad = _frame(start='2021-12-27', days=8)          # 跨过 2022-01-01
    assert bad['trading_date'].max() >= pd.Timestamp(C.HOLDOUT_START)
    save_shard(bad, shard_dir, 'RB', fmt='pickle')
    with pytest.raises(C.HoldoutViolation, match='2022'):
        load_shard('RB', shard_dir)




def test_load_shard_sorts_unsorted_shard(shard_dir):
    """分片若因任何原因乱序落盘，读出来必须已排序——持续期计算依赖时间单调。"""
    df = _frame().sample(frac=1.0, random_state=0)
    save_shard(df, shard_dir, 'RB', fmt='pickle')
    out = load_shard('RB', shard_dir)
    assert out.index.is_monotonic_increasing


# --- test_universe.py ---

import json
import tempfile
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import build_universe, daily_stats, load_universe, market_calendar, screen_year


def _minute_df(days, bars_per_day=10, night_bars=0, turnover=1.0,
               closew=100.0, nan_days=()) -> pd.DataFrame:
    """最小分钟表：每天 night_bars 根夜盘（前一日 21:00 起）+ 日盘若干根。"""
    rows = []
    for k, d in enumerate(pd.DatetimeIndex(days)):
        stamps = []
        if night_bars:
            if k:
                prev = pd.DatetimeIndex(days)[k - 1]
            else:
                prev_d = d.date() - timedelta(days=1)
                prev = pd.Timestamp(year=prev_d.year, month=prev_d.month, day=prev_d.day)
            stamps += [pd.Timestamp(year=prev.year, month=prev.month, day=prev.day,
                                    hour=21, minute=m + 1)
                       for m in range(night_bars)]
        stamps += [pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                hour=9, minute=m + 1)
                   for m in range(bars_per_day)]
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
    """全 NaN 的一天必须得 NaN。"""
    days = pd.bdate_range('2016-03-01', periods=3)
    df = _minute_df(days, turnover=2.0, nan_days=[days[1]])
    d = daily_stats(df)
    assert np.isnan(d.loc[days[1], 'turnover'])
    assert d.loc[days[1], 'n_valid_close'] == 0
    assert d.loc[days[0], 'turnover'] == 20.0


def test_market_calendar_is_union_not_per_symbol_max():
    """分母必须是全市场交易日的并集。"""
    a = daily_stats(_minute_df(pd.bdate_range('2016-01-04', periods=10)))
    b = daily_stats(_minute_df(pd.bdate_range('2016-01-11', periods=10)))
    cal = market_calendar({'A': a, 'B': b})
    assert cal.loc[2016] == 15          # 两段各 10 天、重叠 5 天


def test_screen_never_uses_target_year_data():
    """目标年数据再漂亮也不能用：2017 年爆量、2016 年不达标的品种 在 2017 年的池子里必须缺席。"""
    stats = _stats([('X', 2016, 1.0, 1.0, 240),      # 前一年不达标
                    ('X', 2017, 999.0, 1.0, 240)])   # 当年爆量
    d = screen_year(stats, 2017)
    assert not bool(d.loc['X', 'passed'])
    assert d.loc['X', 'reason'] == '成交额'
    assert d.loc['X', 'screen_window'] == '2016-2016'


def test_screen_uses_previous_year_even_if_symbol_later_collapses():
    """反向的另一半：前一年活跃、当年塌缩的品种，当年应当在池子里。"""
    stats = _stats([('ZC', 2017, 500.0, 1.0, 240),
                    ('ZC', 2018, 0.1, 1.0, 240)])
    assert bool(screen_year(stats, 2018).loc['ZC', 'passed'])
    # 而塌缩之后的那一年就必须出池
    assert not bool(screen_year(stats, 2019).loc['ZC', 'passed'])


def test_screen_rejects_zero_lookback():
    with pytest.raises(ValueError, match='前视'):
        screen_year(_stats(PASS), 2017, lookback_years=0)


def test_build_universe_refuses_years_beyond_research_end():
    stats = _stats([('RB', y, 100.0, 1.0, 240) for y in (2020, 2021)])
    with pytest.raises(C.HoldoutViolation):
        build_universe(stats, years=[C.RESEARCH_END.year + 2])


def test_completeness_needs_both_ratio_and_absolute_days():
    """占比与绝对天数是两道独立的门槛：半日市密集的年份可能占比够但天数不够。"""
    low_ratio = _stats([('A', 2016, 100.0, C.MIN_VALID_DAY_RATIO - 0.01, 240)])
    low_days = _stats([('B', 2016, 100.0, 1.0, C.MIN_VALID_DAYS - 1)])
    assert not bool(screen_year(low_ratio, 2017).loc['A', 'pass_complete'])
    assert not bool(screen_year(low_days, 2017).loc['B', 'pass_complete'])
    assert screen_year(low_ratio, 2017).loc['A', 'reason'] == '完整度'


def test_load_universe_roundtrip_restores_int_years():
    d = Path(tempfile.mkdtemp(prefix='tfcta_uni_'))
    p = d / 'universe_by_year.json'
    p.write_text(json.dumps({'2016': ['RB'], '2017': ['RB', 'CU']}), encoding='utf-8')
    uni = load_universe(p)
    assert set(uni) == {2016, 2017}
    assert uni[2017] == ['RB', 'CU']


def test_financial_symbols_are_excluded_by_construction():
    """金融期货的剔除发生在 COMMODITY_SYMBOLS 层，不依赖流动性门槛—— T/TF 的成交额很高，靠门槛是筛不掉的。"""
    for s in C.FINANCIAL_SYMBOLS:
        assert s not in C.COMMODITY_SYMBOLS
    assert len(C.COMMODITY_SYMBOLS) == len(C.ALL_SYMBOLS) - len(C.FINANCIAL_SYMBOLS)


def test_roll_calendar_past_2022_needs_the_final_evaluation(tmp_path):
    from tfcta import config as C
    from tfcta.data import load_roll_calendar
    (tmp_path / "RB.csv").write_text("trading_date,new_contract,prev_contract\n2021-05-10,RB2110,RB2105\n"
                                     "2023-05-10,RB2310,RB2305\n", encoding="utf-8")
    assert len(load_roll_calendar(["RB"], "2022-12-31", directory=tmp_path)["RB"]) == 1
    with pytest.raises(C.HoldoutViolation):
        load_roll_calendar(["RB"], "2023-12-31", directory=tmp_path)


# --- test_validation_import.py ---

import pickle

import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import (
    boundary_check, boundary_verdict, list_shards, load_validation_shard, save_shard,
    source_window_problems,
)


def _bars(days: pd.DatetimeIndex, offset: float, start: float = 100.0) -> pd.DataFrame:
    """每个交易日三根日盘 bar；closew = close + offset（加法复权偏移不变）。"""
    idx = pd.DatetimeIndex([pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=9, minute=m + 1)
                            for d in days for m in range(3)])
    close = start + np.arange(len(idx), dtype=float)
    return pd.DataFrame({
        'closew': close + offset, 'close': close, 'highw': close + offset + 1,
        'loww': close + offset - 1, 'volume': 1.0, 'total_turnover': 1.0,
        'open': close, 'openw': close + offset,
        'trading_date': [d for d in days for _ in range(3)],
    }, index=idx)


def _panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, axis=1)


def test_source_with_any_2023_row_is_refused_whole():
    days = pd.bdate_range('2022-12-26', '2023-01-03')
    problems = source_window_problems(_panel({'RB': _bars(days, 5.0)}))
    assert problems and any('2023' in p for p in problems)


def test_source_with_2021_trading_date_is_refused():
    days = pd.bdate_range('2021-12-30', '2022-01-05')
    assert source_window_problems(_panel({'RB': _bars(days, 5.0)}))


def test_2022_only_source_is_accepted():
    days = pd.bdate_range('2022-01-04', '2022-12-30')
    assert source_window_problems(_panel({'RB': _bars(days, 5.0)})) == []


def test_boundary_offset_must_carry_over():
    research = _bars(pd.bdate_range('2021-12-27', '2021-12-31'), 5.0)
    same = _bars(pd.bdate_range('2022-01-04', '2022-01-07'), 5.0)
    shifted = _bars(pd.bdate_range('2022-01-04', '2022-01-07'), 50.0)
    assert boundary_check(research, same)[0] == 'ok'
    assert boundary_check(research, shifted)[0] == 'mismatch'
    assert boundary_check(None, same)[0] == 'skip'
    assert boundary_check(research, research)[0] == 'fail'


def test_boundary_verdict_tolerates_a_few_rolls_but_not_a_new_base():
    few = {f"S{i}": 'ok' for i in range(9)} | {'X': 'mismatch'}
    many = {f"S{i}": 'ok' for i in range(5)} | {f"X{i}": 'mismatch' for i in range(5)}
    assert boundary_verdict(few)[0]
    assert not boundary_verdict(many)[0]
    assert not boundary_verdict({'A': 'ok', 'B': 'fail'})[0]


@pytest.fixture
def step1(tmp_path, monkeypatch):
    from tfcta.prepare import shard as module
    root = tmp_path / "shards"
    monkeypatch.setattr(C, "RESEARCH_DIR", root / "research")
    monkeypatch.setattr(C, "VALIDATION_DIR", root / "validation_2022")
    monkeypatch.setattr(C, "HOLDOUT_DIR", root / "holdout_locked")
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path / "runs")
    for sym in ('RB', 'CU'):
        save_shard(_bars(pd.bdate_range('2021-12-01', '2021-12-31'), 5.0),
                            C.RESEARCH_DIR, sym, fmt='pickle')
    return module


def _run(step1, tmp_path, panel, *extra):
    src = tmp_path / "src_2022.pkl"
    with open(src, 'wb') as f:
        pickle.dump(panel, f)
    import sys
    argv = ["step1", "--validation-source", str(src), "--symbols", "RB", "CU",
            "--format", "pickle", *extra]
    old, sys.argv = sys.argv, argv
    try:
        return step1.main()
    finally:
        sys.argv = old


def test_step1_writes_validation_shards_once(step1, tmp_path):
    days = pd.bdate_range('2022-01-04', '2022-03-31')
    panel = _panel({s: _bars(days, 5.0, start=131.0) for s in ('RB', 'CU')})
    assert _run(step1, tmp_path, panel) == 0
    assert list_shards(C.VALIDATION_DIR) == ['CU', 'RB']
    loaded = load_validation_shard('RB')
    assert loaded['trading_date'].min() == pd.Timestamp('2022-01-04')
    # 第二次不覆盖
    assert _run(step1, tmp_path, panel) == 2


def test_step1_refuses_mixed_source_and_writes_nothing(step1, tmp_path):
    days = pd.bdate_range('2022-12-01', '2023-01-06')
    panel = _panel({s: _bars(days, 5.0) for s in ('RB', 'CU')})
    assert _run(step1, tmp_path, panel) == 1
    assert list_shards(C.VALIDATION_DIR) == []


def test_step1_refuses_rebased_adjustment(step1, tmp_path):
    days = pd.bdate_range('2022-01-04', '2022-03-31')
    panel = _panel({s: _bars(days, 80.0, start=131.0) for s in ('RB', 'CU')})
    assert _run(step1, tmp_path, panel) == 1
    assert list_shards(C.VALIDATION_DIR) == []


def test_step1_slices_2022_from_full_monolith(step1, tmp_path, monkeypatch):
    """全样本单体文件按 trading_date 取 2022：研究期和 2023 的行都不落盘。"""
    days = pd.bdate_range('2021-12-01', '2023-02-28')
    panel = _panel({s: _bars(days, 5.0) for s in ('RB', 'CU')})
    src = tmp_path / "monolith.pkl"
    with open(src, 'wb') as f:
        pickle.dump(panel, f)
    import sys
    argv = ["step1", "--validation-from-monolith", "--monolith", str(src),
            "--symbols", "RB", "CU", "--format", "pickle"]
    monkeypatch.setattr(sys, "argv", argv)
    assert step1.main() == 0
    loaded = load_validation_shard('RB')
    assert loaded['trading_date'].min() == pd.Timestamp('2022-01-03')
    assert loaded['trading_date'].max() == pd.Timestamp('2022-12-30')
    assert loaded.index.max() < pd.Timestamp('2023-01-01')
