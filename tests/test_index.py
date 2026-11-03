"""股指数据、执行、机制和策略的合成测试。"""

import pandas as pd
import pytest
import download.index.getIndexL1 as tick_download
import numpy as np
from datetime import date
from tfcta import config as C
from download.index.getIndexFutures import (
    assemble_symbol_frames,
    split_oos_frames,
    split_research_frames,
)
from tfcta.config import HoldoutViolation
from download.index.getIndexL1 import prepare_raw_ticks, validate_window
from strategies.index_close_location import high_open_z
from research.index.index_execution_10m import (
    _opening_row,
    call_auction_volume,
    auction_allowance,
    opening_quotes,
)

# --- test_index_futures.py ---


def api_frame(index, suffix):
    date = pd.Series(pd.to_datetime(index).normalize(), index=index)
    frame = pd.DataFrame({"trading_date": date}, index=index)
    if suffix in ("88", "889"):
        frame = frame.assign(open=100.0, high=101.0, low=99.0, close=100.5)
    if suffix == "88":
        frame["total_turnover"] = 1000.0
        frame["dominant_id"] = "IF2101"
    if suffix == "99":
        frame["volume"] = 10.0
        frame["open_interest"] = 100.0
    return frame


def test_final_index_rule_is_frozen_long_only_lagged_and_uses_fixed_budget():
    from strategies.index_close_location import BOOK, FIXED_MULTIPLIERS
    from strategies.index_catalog import INDEX_FROZEN_BOOKS
    from strategies.index_clv_individual_equal import BOOK as INDIVIDUAL_EQUAL_BOOK
    from strategies.index_clv_pooled_equal import BOOK as POOLED_EQUAL_BOOK
    from strategies.index_clv_return_rank_20d import BOOK as RETURN_RANK_BOOK
    from strategies.index_clv_inverse_vol_20d import BOOK as INVERSE_VOL_BOOK

    assert INDEX_FROZEN_BOOKS == (
        BOOK,
        INDIVIDUAL_EQUAL_BOOK,
        POOLED_EQUAL_BOOK,
        RETURN_RANK_BOOK,
        INVERSE_VOL_BOOK,
    )
    protocol = BOOK.frozen()
    assert protocol["chosen"] == protocol["parameters"] and protocol["frozen_at"]
    days = pd.bdate_range("2018-01-02", periods=100)
    price = pd.DataFrame(100.0, index=days, columns=["IF", "IH", "IC"])
    bars = {"open": price, "openw": price.copy(), "close": price, "closew": price,
            "highw": price+1, "loww": price-1}
    clv = pd.Series(.2, index=days)
    clv.iloc[52] = .8
    z = pd.Series(-1., index=days); z.iloc[50] = 1.
    target = BOOK.targets(clv, bars, rally_z=z)
    assert target.ge(0).all().all() and target.iloc[0].eq(0).all()
    np.testing.assert_allclose(target.iloc[51], np.array(FIXED_MULTIPLIERS)*1.5)
    assert target.iloc[53].eq(0).all()
    assert target.iloc[51].sum() == pytest.approx(4.5)
    altered = z.copy(); altered.iloc[51:] = 10.
    pd.testing.assert_frame_equal(target.iloc[:52], BOOK.targets(clv, bars, rally_z=altered).iloc[:52])


def test_individual_clv_book_splits_budget_across_active_symbols_and_lags():
    from strategies.index_clv_individual_equal import BOOK, PARAMETERS
    from strategies.index_catalog import book_named

    assert book_named(BOOK.ID) is BOOK
    assert BOOK.frozen()["chosen"] == PARAMETERS
    days = pd.bdate_range("2018-01-02", periods=5)
    price = pd.DataFrame(100.0, index=days, columns=BOOK.SYMBOLS)
    bars = {
        "open": price,
        "openw": price.copy(),
        "closew": pd.DataFrame(
            [[20, 80, 80], [20, 80, 20], [80, 80, 80], [20, 20, 20], [80, 80, 80]],
            index=days,
            columns=BOOK.SYMBOLS,
        ),
        "highw": pd.DataFrame(100.0, index=days, columns=BOOK.SYMBOLS),
        "loww": pd.DataFrame(0.0, index=days, columns=BOOK.SYMBOLS),
    }
    rally_z = pd.Series(-1.0, index=days)
    targets = BOOK.targets(BOOK.signal_inputs(bars, days), bars, rally_z=rally_z)

    assert targets.iloc[0].eq(0).all()
    np.testing.assert_allclose(targets.iloc[1], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(targets.iloc[2], [1.0, 0.0, 1.0])
    assert targets.iloc[3].eq(0).all()
    np.testing.assert_allclose(targets.iloc[4], [2.0 / 3, 2.0 / 3, 2.0 / 3])

    rally_targets = BOOK.targets(
        BOOK.signal_inputs(bars, days), bars, rally_z=pd.Series(1.0, index=days)
    )
    np.testing.assert_allclose(rally_targets.iloc[1], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(rally_targets.iloc[2], [1.5, 0.0, 1.5])


def test_index_cli_rejects_unknown_books_and_years_outside_2015_2025(monkeypatch):
    import backtest.engine as engine

    monkeypatch.setattr(engine, "run_index", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应开跑")))
    for args in (
        ["--market", "index", "--year", "2014"],
        ["--market", "index", "--year", "2026"],
        ["index_close_location_bear_short", "--research"],
        ["--market", "index", "--start", "2022", "--end", "2026"],
    ):
        with pytest.raises(SystemExit):
            engine.main(args)


def test_dynamic_allocation_strategies_are_registered_and_weight_rank_and_volatility():
    from strategies.index_clv_return_rank_20d import BOOK as RETURN_RANK_BOOK
    from strategies.index_clv_inverse_vol_20d import BOOK as INVERSE_VOL_BOOK
    from strategies.index_catalog import book_named

    assert book_named(RETURN_RANK_BOOK.ID) is RETURN_RANK_BOOK
    assert book_named(INVERSE_VOL_BOOK.ID) is INVERSE_VOL_BOOK
    assert RETURN_RANK_BOOK.frozen()["chosen"] == RETURN_RANK_BOOK.frozen()["parameters"]
    assert INVERSE_VOL_BOOK.frozen()["chosen"] == INVERSE_VOL_BOOK.frozen()["parameters"]

    days = pd.bdate_range("2018-01-02", periods=80)
    noise = np.sin(np.arange(len(days)) * 0.7)
    returns = pd.DataFrame(
        {
            "IF": 0.0005 + noise * 0.0001,
            "IH": 0.0007 + noise * 0.0002,
            "IC": 0.0010 + noise * 0.0003,
        },
        index=days,
    )
    close = pd.DataFrame(
        {
            symbol: start * (1 + returns[symbol]).cumprod()
            for symbol, start in zip(("IF", "IH", "IC"), (3000.0, 2000.0, 5000.0))
        },
        index=days,
    )
    bars = {
        "open": close.copy(),
        "openw": close.copy(),
        "close": close.copy(),
        "closew": close.copy(),
        "highw": close + 2.0,
        "loww": close - 2.0,
    }
    clv = pd.Series(0.2, index=days)
    rally = pd.Series(-1.0, index=days)
    rank_targets = RETURN_RANK_BOOK.targets(clv, bars, rally_z=rally)
    vol_targets = INVERSE_VOL_BOOK.targets(clv, bars, rally_z=rally)

    from strategies.index_close_location import BOOK as FIXED_BOOK

    fixed_targets = FIXED_BOOK.targets(clv, bars, rally_z=rally)
    assert rank_targets.iloc[0].eq(0).all()
    np.testing.assert_allclose(rank_targets.iloc[21], [0.5, 1.0, 1.5])
    assert vol_targets.iloc[21]["IF"] > vol_targets.iloc[21]["IH"]
    assert vol_targets.iloc[21]["IH"] > vol_targets.iloc[21]["IC"]
    assert rank_targets.iloc[21].sum() == pytest.approx(fixed_targets.iloc[21].sum())
    assert vol_targets.iloc[21].sum() == pytest.approx(3.0)
    rally_targets = RETURN_RANK_BOOK.targets(
        clv, bars, rally_z=pd.Series(1.0, index=days)
    )
    np.testing.assert_allclose(rally_targets.iloc[21], rank_targets.iloc[21] * 1.5)

    changed = {key: value.copy() for key, value in bars.items()}
    changed["close"].iloc[30:] *= 1.5
    changed["closew"].iloc[30:] *= 1.5
    changed["open"].iloc[30:] *= 1.5
    changed["openw"].iloc[30:] *= 1.5
    pd.testing.assert_frame_equal(
        rank_targets.iloc[:31],
        RETURN_RANK_BOOK.targets(clv, changed, rally_z=rally).iloc[:31],
    )


def test_dynamic_allocation_oos_warmup_uses_only_oos_observations(tmp_path):
    from strategies.index_clv_return_rank_20d import BOOK as RETURN_RANK_BOOK
    from strategies.index_clv_inverse_vol_20d import BOOK as INVERSE_VOL_BOOK
    from strategies.index_close_location import SYMBOLS

    research_index = pd.bdate_range(end="2021-12-31", periods=80)
    oos_index = pd.bdate_range(start="2022-01-03", periods=25)

    def make_bars(index, start_step):
        step = np.arange(start_step, start_step + len(index), dtype=float)
        close = pd.DataFrame(
            {
                "IF": 3000.0 + step * 1.0,
                "IH": 2000.0 + step * 0.6,
                "IC": 5000.0 + step * 1.8,
            },
            index=index,
        )
        return {
            "open": close.copy(),
            "openw": close.copy(),
            "close": close.copy(),
            "closew": close,
            "highw": close + 5.0,
            "loww": close - 1.0,
        }

    bars = {
        field: pd.concat(
            [make_bars(research_index, 0)[field], make_bars(oos_index, 100)[field]]
        )
        for field in ("open", "openw", "close", "closew", "highw", "loww")
    }
    index = research_index.append(oos_index)
    market_clv = pd.Series(0.2, index=index)
    rally = pd.Series(-1.0, index=index)
    for book in (RETURN_RANK_BOOK, INVERSE_VOL_BOOK):
        targets = book.targets(market_clv, bars, rally_z=rally)

        for execution_day in oos_index[:21]:
            np.testing.assert_allclose(
                targets.loc[execution_day, list(SYMBOLS)].to_numpy(),
                np.repeat(targets.loc[execution_day].mean(), len(SYMBOLS)),
            )
        assert not np.allclose(
            targets.loc[oos_index[21], list(SYMBOLS)].to_numpy(),
            np.repeat(targets.loc[oos_index[21]].mean(), len(SYMBOLS)),
        )


def test_dynamic_allocation_keeps_the_known_vol_scale_on_the_first_oos_open():
    from strategies.index_clv_return_rank_20d import BOOK
    from strategies.index_close_location import close_vol_scale

    research_index = pd.bdate_range(end="2021-12-31", periods=40)
    oos_index = pd.bdate_range(start="2022-01-03", periods=5)
    shock = np.where(np.arange(len(research_index)) % 2 == 0, 1.08, 0.93)
    research_close = pd.DataFrame(
        {symbol: 3000.0 * np.cumprod(shock) for symbol in BOOK.SYMBOLS},
        index=research_index,
    )
    oos_close = pd.DataFrame(3000.0, index=oos_index, columns=list(BOOK.SYMBOLS))

    def pack(frame):
        return {
            "open": frame.copy(),
            "openw": frame.copy(),
            "close": frame.copy(),
            "closew": frame.copy(),
            "highw": frame + 1.0,
            "loww": frame - 1.0,
        }

    research_bars = pack(research_close)
    bars = {
        field: pd.concat([research_bars[field], pack(oos_close)[field]])
        for field in research_bars
    }
    index = research_index.append(oos_index)
    targets = BOOK.targets(
        pd.Series(0.2, index=index), bars, rally_z=pd.Series(-1.0, index=index)
    )
    research_returns = (
        (research_bars["openw"].shift(-1) - research_bars["openw"]) / research_bars["open"]
    ).mean(axis=1, skipna=False)
    scale = float(close_vol_scale(research_returns).iloc[-1])
    assert scale < 1.0
    np.testing.assert_allclose(targets.loc[oos_index[0]].to_numpy(), np.repeat(scale, 3))


def test_index_cli_runs_any_registered_book_on_research_or_oos(monkeypatch):
    import backtest.engine as engine
    from strategies.index_catalog import DEFAULT_BOOK
    from strategies.index_clv_return_rank_20d import BOOK as RETURN_RANK_BOOK

    calls = []
    monkeypatch.setattr(
        engine,
        "run_index",
        lambda start, end, by_year=False, capital=None, execution=None, book=None: calls.append(
            (start, end, book.ID, by_year)
        ),
    )
    from strategies.index_catalog import INDEX_BOOKS

    engine.main([RETURN_RANK_BOOK.ID, "--start", "2022", "--end", "2023"])
    engine.main([DEFAULT_BOOK.ID, "--oos"])
    engine.main(["--market", "index", "--research"])
    assert calls == [
        (2022, 2023, RETURN_RANK_BOOK.ID, False),
        (2022, 2025, DEFAULT_BOOK.ID, False),
        *[(2015, 2021, book.ID, False) for book in INDEX_BOOKS],
    ]


def test_index_oos_window_loads_prior_oos_years_for_tick_estimation(monkeypatch):
    import backtest.engine as engine
    import research.index.index_execution_10m as execution
    from strategies.index_catalog import DEFAULT_BOOK

    class StopAfterWindow(Exception):
        pass

    monkeypatch.setattr(engine, "_load_index_research_inputs", lambda: (None, None, None, None, None))
    monkeypatch.setattr(execution, "load_opening_quotes", lambda: pd.DataFrame())
    requested = []

    def capture_window(start, end):
        requested.append((start, end))
        raise StopAfterWindow

    monkeypatch.setattr(engine, "_load_index_oos_inputs", capture_window)
    with pytest.raises(StopAfterWindow):
        engine.run_index(2024, 2024, book=DEFAULT_BOOK)
    assert requested == [(pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31"))]


def test_assemble_symbol_frames_maps_raw_adjusted_volume_and_oi_fields():
    index = pd.date_range("2021-01-04 09:31", periods=2, freq="min")
    main = api_frame(index, "88")
    next_main = api_frame(index, "889").assign(close=101.0)
    aggregate = api_frame(index, "99")
    result = assemble_symbol_frames(main, next_main, aggregate)
    assert result.index.equals(index)
    assert result.close.tolist() == [100.5, 100.5]
    assert result.closew.tolist() == [101.0, 101.0]
    assert result.volume.tolist() == [10.0, 10.0]
    assert result.open_interest.tolist() == [100.0, 100.0]
    assert result.trading_date.tolist() == [pd.Timestamp("2021-01-04")] * 2

def test_assemble_symbol_frames_refuses_misaligned_series():
    index = pd.date_range("2021-01-04 09:31", periods=2, freq="min")
    with pytest.raises(ValueError, match="indexes differ"):
        assemble_symbol_frames(api_frame(index, "88"), api_frame(index[:1], "889"),
                               api_frame(index, "99"))

def test_research_split_rejects_holdout_dates():
    index = pd.date_range("2022-01-04 09:31", periods=2, freq="min")
    frame = assemble_symbol_frames(api_frame(index, "88"), api_frame(index, "889"),
                                   api_frame(index, "99"))
    with pytest.raises(C.HoldoutViolation, match="样本外日期"):
        split_research_frames(frame)

def test_research_split_preserves_factor_and_open_interest_fields():
    index = pd.date_range("2021-12-30 09:31", periods=2, freq="min")
    frame = assemble_symbol_frames(api_frame(index, "88"), api_frame(index, "889"),
                                   api_frame(index, "99"))
    prices, oi = split_research_frames(frame)
    assert {"closew", "close", "highw", "loww", "volume", "total_turnover", "open", "openw"} <= set(prices)
    assert set(oi) == {"open_interest", "dominant_id", "trading_date"}
    assert len(prices) == len(oi) == 2
    from download.index.getIndexFutures import combine_price_and_oi

    combined = combine_price_and_oi(prices, oi)
    assert {"open_interest", "dominant_id"} <= set(combined)
    assert combined.index.equals(prices.index)

def test_oos_split_accepts_only_the_declared_2022_to_2025_window():
    index = pd.date_range("2022-07-22 09:31", periods=2, freq="min")
    frame = assemble_symbol_frames(api_frame(index, "88"), api_frame(index, "889"),
                                   api_frame(index, "99"))
    prices, oi = split_oos_frames(frame, pd.Timestamp("2022-07-22").date(),
                                  pd.Timestamp("2025-12-31").date())
    assert len(prices) == len(oi) == 2


def test_oos_downloader_writes_one_combined_symbol_shard(tmp_path, monkeypatch):
    from download.index import getIndexFutures as downloader

    index = pd.date_range("2022-01-03 09:31", periods=2, freq="min")
    frame = assemble_symbol_frames(
        api_frame(index, "88"), api_frame(index, "889"), api_frame(index, "99")
    )
    monkeypatch.setattr(downloader, "download_oos_symbol", lambda *args: frame)
    downloader.save_oos(
        None, ["IF"], pd.Timestamp("2022-01-03").date(),
        pd.Timestamp("2022-01-03").date(), tmp_path,
    )

    root = tmp_path / "minute_shards" / "oos"
    saved = pd.read_parquet(root / "IF.parquet")
    assert {"open_interest", "dominant_id"} <= set(saved.columns)
    assert not (root / "open_interest").exists()


def test_qvix_oos_split_excludes_dates_after_registered_window():
    from download.index.getQvix import split_qvix

    frame = pd.DataFrame({
        "date": pd.to_datetime(["2021-12-31", "2022-01-04", "2025-12-31", "2026-01-02"]),
        "open": [1.0, 1.0, 1.0, 1.0],
        "high": [1.0, 1.0, 1.0, 1.0],
        "low": [1.0, 1.0, 1.0, 1.0],
        "close": [1.0, 1.0, 1.0, 1.0],
    })

    research, oos = split_qvix(frame)

    assert research["date"].tolist() == [pd.Timestamp("2021-12-31")]
    assert oos["date"].tolist() == [pd.Timestamp("2022-01-04"), pd.Timestamp("2025-12-31")]

# --- test_tick_l1.py ---


def test_raw_index_ticks_keep_all_fields_and_add_contract_metadata():
    ticks = pd.DataFrame(
        {
            "trading_date": [pd.Timestamp("2016-01-04")],
            "open": [100.0],
            "last": [100.1],
            "high": [100.2],
            "low": [99.9],
            "volume": [5],
            "a1": [100.2],
            "b1": [100.0],
            "a1_v": [10],
            "b1_v": [12],
            "a2": [0.0],
            "a2_v": [0],
        },
        index=pd.DatetimeIndex(["2016-01-04 09:30:00"], name="datetime"),
    )

    result = prepare_raw_ticks(ticks, "IF", "IF1601")

    assert result.loc[0, "symbol"] == "IF"
    assert result.loc[0, "contract"] == "IF1601"
    assert result.loc[0, "datetime"] == pd.Timestamp("2016-01-04 09:30:00")
    assert result.loc[0, "trading_date"] == pd.Timestamp("2016-01-04")
    assert result.loc[0, "a2"] == 0.0
    assert result.loc[0, "a2_v"] == 0


def test_raw_index_ticks_fail_if_l1_quotes_are_missing():
    ticks = pd.DataFrame(
        {
            "trading_date": [pd.Timestamp("2016-01-04")],
            "open": [100.0],
            "last": [100.1],
            "high": [100.2],
            "low": [99.9],
            "volume": [5],
        },
        index=pd.DatetimeIndex(["2016-01-04 09:30:00"], name="datetime"),
    )

    with pytest.raises(ValueError, match="a1"):
        prepare_raw_ticks(ticks, "IF", "IF1601")


def test_tick_downloader_limits_oos_window():
    validate_window(pd.Timestamp("2022-01-01").date(), pd.Timestamp("2025-12-31").date(), "oos")

    with pytest.raises(HoldoutViolation, match="OOS 股指 L1"):
        validate_window(pd.Timestamp("2021-12-31").date(), pd.Timestamp("2025-12-31").date(), "oos")
    with pytest.raises(HoldoutViolation, match="OOS 股指 L1"):
        validate_window(pd.Timestamp("2022-01-01").date(), pd.Timestamp("2026-01-01").date(), "oos")
    with pytest.raises(HoldoutViolation, match="研究期股指 L1"):
        validate_window(pd.Timestamp("2022-01-01").date(), pd.Timestamp("2022-12-31").date(), "research")


def test_oos_download_writes_raw_ticks_without_research_summaries(tmp_path, monkeypatch):
    day = pd.Timestamp("2022-01-03")

    class FakeFutures:
        @staticmethod
        def get_dominant(symbol, start, end, rule):
            return pd.Series(["IF2201"], index=pd.DatetimeIndex([day]))

    class FakeRQ:
        futures = FakeFutures()

        @staticmethod
        def init(*credentials):
            return None

        @staticmethod
        def get_price(contract, start_date, end_date, frequency):
            return pd.DataFrame(
                {
                    "trading_date": [day, day],
                    "open": [100.0, 100.0],
                    "last": [100.1, 100.2],
                    "high": [100.2, 100.2],
                    "low": [99.9, 99.9],
                    "volume": [1, 2],
                    "a1": [100.2, 100.3],
                    "b1": [100.0, 100.1],
                    "a1_v": [10, 11],
                    "b1_v": [12, 13],
                },
                index=pd.DatetimeIndex(
                    ["2022-01-03 09:30:00", "2022-01-03 09:30:01"],
                    name="datetime",
                ),
            )

    monkeypatch.setattr(tick_download, "OOS_RAW_OUT", tmp_path / "oos" / "l1_ticks")
    monkeypatch.setattr(tick_download, "rqdata_credentials", lambda: ("user", "password"))
    monkeypatch.setitem(__import__("sys").modules, "rqdatac", FakeRQ)

    result = tick_download.download_oos(["IF"], date(2022, 1, 1), date(2022, 1, 4))

    symbol_root = tmp_path / "oos" / "l1_ticks" / "IF"
    parts = list(symbol_root.rglob("*.parquet"))
    assert result == 0
    assert len(parts) == 1
    assert not (tmp_path / "oos" / "l1_close" / "IF.csv").exists()
    metadata = (symbol_root / "download_manifest.json").read_text(encoding="utf-8")
    assert '"content_not_loaded_for_analysis": true' in metadata


def test_buy_auction_allowance_uses_only_prior_auction_volume_and_charges_the_impact_floor():
    from research.index.index_execution_10m import buy_auction_tables

    days = pd.bdate_range("2019-01-02", periods=14)
    volume = [100.0] * 14
    volume[10] = 1_000_000.0  # 当天的竞价量不能放大当天额度
    snap = pd.DataFrame({
        "trading_date": days, "symbol": "IF", "contract": "IF1901", "open": 3000.0,
        "auction_volume": volume, "a1": 3000.0, "a1_v": 50.0, "b1": 2999.8, "b1_v": 5.0,
    })
    snap.loc[12, "a1"] = 3002.0          # 卖一高于开盘：成本取冲击下界
    snap.loc[13, "contract"] = "IF1902"  # 快照合约不对：没有额度
    frame = lambda value, dtype=None: pd.DataFrame({"IF": value}, index=days, dtype=dtype)
    lots, rate, invalid, _impact = buy_auction_tables(
        snap, frame(3000.0), frame("IF1901", object), frame(0.2 / 3000.0), days)

    assert lots["IF"].iloc[:10].eq(0.0).all()
    assert lots["IF"].iloc[10:13].tolist() == [10.0, 10.0, 10.0]
    assert lots["IF"].iloc[13] == 0.0 and invalid == 1
    assert rate["IF"].iloc[10] == pytest.approx(0.2 / 3000.0)
    assert rate["IF"].iloc[12] == pytest.approx(2.0 / 3000.0)


def test_frozen_high_open_z_does_not_look_ahead():

    dates = pd.bdate_range("2010-01-01", periods=100)
    values = pd.Series(range(100), index=dates, dtype=float)
    bars = {
        "open": pd.DataFrame({"IF": 1000.0 + values}, index=dates),
        "openw": pd.DataFrame({"IF": 1000.0 + values}, index=dates),
        "highw": pd.DataFrame({"IF": 1002.0 + values + values.mod(7)}, index=dates),
    }
    original = high_open_z(bars, ("IF",))
    changed = {key: value.copy() for key, value in bars.items()}
    changed["highw"].loc[dates[80]:, "IF"] += 500.0
    revised = high_open_z(changed, ("IF",))

    pd.testing.assert_series_equal(original.loc[:dates[79]], revised.loc[:dates[79]])


def test_opening_quotes_skip_the_auction_and_stop_after_thirty_minutes():
    day = pd.Timestamp("2016-03-01")
    times = day + pd.to_timedelta(["09:00:00", "09:29:00", "09:30:00.5", "09:45:00", "09:59:59", "10:00:00"])
    ticks = pd.DataFrame({
        "datetime": times, "trading_date": day, "open": [0.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [0.0, 10.0, 12, 20, 30, 99],
        "a1": [0.0, 101.0, 100.4, 100.8, 0.0, 200.0],
        "b1": [0.0, 99.0, 100.0, 100.2, 100.0, 1.0], "contract": "IF1603",
    })
    row = opening_quotes(ticks).iloc[0]
    assert row["auction_volume"] == 10.0 and row["volume"] == 20.0
    assert row["tick_open"] == 100.0
    assert np.isclose(row["ask"], 100.6) and np.isclose(row["bid"], 100.1) and row["quotes"] == 2


def test_auction_allowance_is_ten_percent_of_realized_daily_volume():
    days = pd.bdate_range("2020-01-02", periods=2)
    quotes = pd.DataFrame([
        {"trading_date": day, "symbol": symbol, "auction_volume": volume}
        for day, volumes in zip(days, ((47, 9, 20), (99, 19, 0)))
        for symbol, volume in zip(("IF", "IH", "IC"), volumes)
    ])
    allowance = auction_allowance(quotes, days)
    assert allowance.loc[days[0]].tolist() == [4.0, 0.0, 2.0]
    assert allowance.loc[days[1]].tolist() == [9.0, 1.0, 0.0]
    one = auction_allowance(quotes[quotes.symbol.eq("IF")], days, symbols=("IF",))
    assert one.loc[days[0], "IF"] == 4.0 and one.loc[days[1], "IF"] == 9.0


def test_call_auction_uses_the_open_print_only_when_nothing_precedes_it():
    # 起点之前没有 tick：竞价成交记在 09:15:00 这一笔
    stamped = pd.Timestamp("2012-06-01")
    at_open = pd.DataFrame({
        "datetime": [
            stamped + pd.Timedelta(9 * 3600 + 15 * 60, unit="s"),
            stamped + pd.Timedelta(9 * 3600 + 15 * 60 + 1, unit="s"),
        ],
        "volume": [418.0, 500.0],
    })
    assert call_auction_volume(at_open, stamped) == 418.0
    # 起点之前已经有一笔：竞价量用那一笔，不用正好落在 09:15 的连续成交
    earlier = pd.Timestamp("2013-06-03")
    both = pd.DataFrame({
        "datetime": [
            earlier + pd.Timedelta(9 * 3600 + 14 * 60, unit="s"),
            earlier + pd.Timedelta(9 * 3600 + 15 * 60, unit="s"),
        ],
        "volume": [400.0, 722.0],
    })
    assert call_auction_volume(both, earlier) == 400.0


def test_ticks_that_start_after_the_open_are_a_gap():
    day = pd.Timestamp("2011-06-17")
    ticks = pd.DataFrame({
        "datetime": [day + pd.Timedelta(9 * 3600 + 33 * 60, unit="s"), day + pd.Timedelta(9 * 3600 + 34 * 60, unit="s")],
        "trading_date": day,
        "open": 3000.0,
        "volume": [6299.0, 6400.0],
        "a1": [3001.0, 3001.2],
        "b1": [3000.6, 3000.8],
    })
    row = _opening_row(ticks, day, "IF1107")
    assert row["open_gap"] is True and row["auction_volume"] == 0.0 and row["quotes"] == 0


def test_index_market_without_a_name_runs_every_registered_book(monkeypatch):
    seen = []
    import backtest.engine as engine
    from strategies.index_catalog import INDEX_BOOKS

    monkeypatch.setattr(engine, "run_index", lambda start, end, **kwargs: seen.append((start, end, kwargs)))
    engine.main(["--market", "index", "--research"])
    assert seen == [
        (
            2015,
            2021,
            {
                "by_year": False,
                "capital": 10_000_000.0,
                "execution": "l1_30m",
                "book": book,
            },
        )
        for book in INDEX_BOOKS
    ]


def test_named_individual_clv_book_uses_index_research_router(monkeypatch):
    seen = []
    import backtest.engine as engine
    from strategies.index_clv_individual_equal import BOOK

    monkeypatch.setattr(
        engine,
        "run_index",
        lambda start, end, **kwargs: seen.append((start, end, kwargs)),
    )
    engine.main([BOOK.ID, "--research"])

    assert seen == [(
        2015,
        2021,
        {
            "by_year": False,
            "capital": 10_000_000.0,
            "execution": "l1_30m",
            "book": BOOK,
        },
    )]


def test_pooled_clv_equal_book_uses_pooled_signal_and_equal_weights():
    from strategies.index_clv_pooled_equal import BOOK, PARAMETERS
    from strategies.index_catalog import book_named

    assert book_named(BOOK.ID) is BOOK
    assert BOOK.frozen()["chosen"] == PARAMETERS
    days = pd.bdate_range("2018-01-02", periods=5)
    symbols = BOOK.SYMBOLS
    clv = pd.DataFrame(
        [
            [0.2, 0.3, 0.5],
            [0.2, 0.5, 0.51],
            [0.1, 0.2, 0.3],
            [0.8, 0.8, 0.8],
            [0.2, 0.2, 0.2],
        ],
        index=days,
        columns=symbols,
    )
    price = pd.DataFrame(100.0, index=days, columns=symbols)
    bars = {"open": price, "openw": price.copy()}
    rally_z = pd.Series(-1.0, index=days)

    target = BOOK.targets(clv.mean(axis=1), bars, rally_z=rally_z)

    assert target.iloc[0].eq(0).all()
    np.testing.assert_allclose(target.iloc[1], [1.0, 1.0, 1.0])
    assert target.iloc[2].eq(0).all()
    np.testing.assert_allclose(target.iloc[3], [1.0, 1.0, 1.0])
    assert target.iloc[4].eq(0).all()
    rallied = BOOK.targets(clv.mean(axis=1), bars, rally_z=pd.Series(1.0, index=days))
    np.testing.assert_allclose(rallied.iloc[1], [1.5, 1.5, 1.5])


def test_allocation_methods_use_only_available_closes_and_preserve_budget():
    from research.index.index_allocation_regimes import _allocation_multipliers

    days = pd.bdate_range("2015-04-16", periods=35)
    trend = np.arange(len(days), dtype=float)
    closew = pd.DataFrame(
        {
            "IF": 3000.0 + trend * 2.0,
            "IH": 2000.0 + trend * np.where(trend % 2, 1.0, 3.0),
            "IC": 5000.0 + trend * np.where(trend % 3, 5.0, 2.0),
        },
        index=days,
    )
    bars = {"closew": closew, "close": closew.copy()}
    weights, _ = _allocation_multipliers(bars)
    changed = {key: value.copy() for key, value in bars.items()}
    changed["closew"].loc[days[26]:, "IC"] += 500.0
    revised, _ = _allocation_multipliers(changed)

    for method, frame in weights.items():
        np.testing.assert_allclose(frame.sum(axis=1).to_numpy(), 3.0)
        assert frame.max(axis=1).le(2.0).all()
        pd.testing.assert_frame_equal(frame.loc[:days[25]], revised[method].loc[:days[25]])
    np.testing.assert_allclose(weights["relative_return_rank_20d"].iloc[0], [1, 1, 1])
    np.testing.assert_allclose(weights["inverse_volatility_20d"].iloc[0], [1, 1, 1])


def test_named_pooled_equal_clv_book_uses_index_research_router(monkeypatch):
    seen = []
    import backtest.engine as engine
    from strategies.index_clv_pooled_equal import BOOK

    monkeypatch.setattr(
        engine,
        "run_index",
        lambda start, end, **kwargs: seen.append((start, end, kwargs)),
    )
    engine.main([BOOK.ID, "--research"])

    assert seen == [(
        2015,
        2021,
        {
            "by_year": False,
            "capital": 10_000_000.0,
            "execution": "l1_30m",
            "book": BOOK,
        },
    )]


def test_final_index_clv_uses_the_day_range():
    from strategies.index_close_location import SYMBOLS, market_close_location
    index = pd.bdate_range("2016-01-13", periods=2)
    bars = {"loww": pd.DataFrame(0.0, index=index, columns=SYMBOLS),
            "highw": pd.DataFrame(10.0, index=index, columns=SYMBOLS),
            "closew": pd.DataFrame([[2.0] * 3, [9.0] * 3], index=index, columns=SYMBOLS)}
    assert market_close_location(bars, index).tolist() == pytest.approx([0.2, 0.9])


def test_vol_scale_shrinks_a_wild_market_and_ignores_the_unfinished_open():
    from strategies.index_close_location import fixed_long_targets

    index = pd.bdate_range("2016-01-04", periods=160)
    level = pd.Series(np.linspace(200.0, 40.0, len(index)), index=index)
    close = pd.DataFrame({symbol: level for symbol in ("IF", "IH", "IC")})
    high = close + 8.0
    low = close - 2.0
    shock = np.array([0.08 if i % 2 == 0 else -0.07 for i in range(len(index))])
    path = 100.0 * np.cumprod(1.0 + shock)
    opened = pd.DataFrame({symbol: path for symbol in ("IF", "IH", "IC")}, index=index)
    bars = {"close": close, "closew": close.copy(), "highw": high, "loww": low, "open": opened, "openw": opened.copy()}
    market = ((close - low) / (high - low)).mean(axis=1)
    targets = fixed_long_targets(market, bars)
    assert targets.iloc[-1, 0] > 0
    assert targets.iloc[-1, 0] < 0.5
    bumped = opened.copy()
    bumped.iloc[-1] *= 2
    pd.testing.assert_frame_equal(
        targets,
        fixed_long_targets(market, {**bars, "open": bumped, "openw": bumped}),
    )


def test_index_oos_fee_loader_reads_the_single_table_inside_the_requested_window(monkeypatch, tmp_path):
    from tfcta.costs import load_index_oos_fees

    root = tmp_path / "index" / "oos"
    root.mkdir(parents=True)
    monkeypatch.setattr(C, "INDEX_OOS_DATA_DIR", root)
    rows = []
    for year, day in ((2022, "2022-01-04"), (2023, "2023-01-03"), (2024, "2024-01-02")):
        rows.extend(
            {
                "symbol": symbol,
                "trading_date": day,
                "contract": f"{symbol}{str(year)[-2:]}01",
                "commission_type": "by_money",
                "open_commission": 0.000023,
                "close_commission": 0.000023,
                "close_commission_today": 0.000023,
            }
            for symbol in ("IF", "IH", "IC")
        )
    pd.DataFrame(rows).to_csv(root / "fee_history.csv", index=False)
    ignored = root / "2022"
    ignored.mkdir()
    (ignored / "fee_history.csv").write_text("not,read\n", encoding="utf-8")

    table = load_index_oos_fees("2022-01-01", "2023-12-31")

    assert set(table["trading_date"].dt.year) == {2022, 2023}
    assert set(table["symbol"]) == {"IF", "IH", "IC"}


def test_index_oos_market_loader_applies_registered_date_filters(monkeypatch, tmp_path):
    import backtest.engine as engine

    root = tmp_path / "index" / "minute_shards" / "oos"
    monkeypatch.setattr(C, "INDEX_OOS_DIR", root)
    monkeypatch.setattr(engine, "find_shard", lambda directory, symbol: directory / f"{symbol}.parquet")
    seen_filters = []
    dates = pd.to_datetime(["2022-01-04", "2022-01-05"])
    minute_index = pd.DatetimeIndex(["2022-01-04 09:31", "2022-01-05 09:31"])

    def fake_read_frame(path, columns=None, filters=None):
        seen_filters.append(filters)
        return pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "openw": [100.0, 101.0],
                "close": [100.0, 101.0],
                "closew": [100.0, 101.0],
                "highw": [102.0, 103.0],
                "loww": [98.0, 99.0],
                "volume": [1.0, 1.0],
                "total_turnover": [100.0, 100.0],
                "dominant_id": ["IF2201", "IF2201"],
                "trading_date": dates,
            },
            index=minute_index,
        )

    monkeypatch.setattr(engine, "read_frame", fake_read_frame)
    monkeypatch.setattr(
        engine,
        "estimate_ticks",
        lambda close, trading_date: pd.DataFrame({"year": [2022], "tick": [0.2]}),
    )
    daily, contracts, ticks = engine._read_index_oos_symbol(
        "IF", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")
    )

    assert len(daily) == len(contracts) == 2
    assert ticks["tick"].tolist() == [0.2]
    assert all(
        filters == [
            ("trading_date", ">=", pd.Timestamp("2022-01-01")),
            ("trading_date", "<=", pd.Timestamp("2022-12-31")),
        ]
        for filters in seen_filters
    )
