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
from strategies.index_common import high_open_z
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






def test_index_cli_rejects_unknown_books_and_years_outside_2015_2025(monkeypatch):
    import backtest.engine as engine

    monkeypatch.setattr(engine, "run_index", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应开跑")))
    for args in (
        ["--market", "index", "--year", "2014"],
        ["--market", "index", "--year", "2026"],
        ["index_close_location_bear_short", "--research"],
        ["index_clv_rally_equal_long", "--research"],
        ["index_clv_rally_equal_up_long", "--research"],
        ["index_clv_adaptive_long", "--research"],
        ["--market", "index", "--start", "2022", "--end", "2026"],
    ):
        with pytest.raises(SystemExit):
            engine.main(args)


def test_index_registry_contains_only_the_base_long_book():
    from strategies.index_catalog import DEFAULT_BOOK, INDEX_BOOKS, INDEX_FROZEN_BOOKS, book_named
    from strategies.index_clv_rally_base_long import BOOK

    assert DEFAULT_BOOK is BOOK
    assert INDEX_BOOKS == (BOOK,)
    assert INDEX_FROZEN_BOOKS == ()
    assert BOOK.STATUS == "exploratory_predeclared"
    assert BOOK.declared()["chosen"] == BOOK.declared()["parameters"]
    assert BOOK.declared()["parameters"]["base_gross"] == 0.5
    assert BOOK.declared()["parameters"]["momentum_windows"] == [21, 63, 126]
    assert BOOK.declared()["parameters"]["short_rule"] == "none"
    assert book_named(BOOK.ID) is BOOK
    with pytest.raises(ValueError, match="未知股指策略"):
        book_named("index_clv_rally_equal_long")
    with pytest.raises(ValueError, match="未知股指策略"):
        book_named("index_clv_rally_equal_up_long")
    with pytest.raises(ValueError, match="未知股指策略"):
        book_named("index_clv_adaptive_long")


def _priced_bars(days, level):
    from strategies.index_clv_rally_base_long import BOOK

    symbols = list(BOOK.SYMBOLS)
    level = np.asarray(level, dtype=float).reshape(-1)
    price = pd.DataFrame(
        np.repeat(level[:, None], len(symbols), axis=1), index=days, columns=symbols
    )
    return {
        "open": price.copy(),
        "openw": price.copy(),
        "close": price.copy(),
        "closew": price.copy(),
        "highw": price + 2.0,
        "loww": price - 1.0,
    }


def test_base_momentum_reduces_longs_and_transfers_the_budget_to_clv():
    from strategies.index_clv_rally_base_long import BOOK

    days = pd.bdate_range("2018-01-02", periods=320)
    rise = _priced_bars(days, 100.0 + np.arange(len(days)))
    clv = pd.Series(0.6, index=days)
    clv.iloc[280] = 0.1
    clv.iloc[282] = 0.1
    clv.iloc[284] = 0.95
    rally = pd.Series(-1.0, index=days)
    rally.iloc[282] = 1.0
    rally.iloc[284] = 2.0
    targets = BOOK.targets(clv, rise, rally_z=rally)

    assert targets.iloc[0].eq(0).all()
    assert targets.iloc[20].eq(0.5).all()
    assert targets.iloc[280].eq(0.5).all()
    assert targets.iloc[281].eq(1.0).all()
    assert targets.iloc[283].eq(1.25).all()
    assert targets.iloc[285].eq(0.5).all()

    fall = _priced_bars(days, 400.0 - np.arange(len(days)))
    down = BOOK.targets(clv, fall, rally_z=rally)
    assert down.iloc[20].eq(0.5).all()
    assert down.iloc[21].eq(0.5).all()
    assert down.iloc[22].to_numpy() == pytest.approx([1 / 3] * 3)
    assert down.iloc[64].to_numpy() == pytest.approx([1 / 6] * 3)
    assert down.iloc[127].eq(0.0).all()
    assert down.iloc[280].eq(0.0).all()
    assert down.iloc[281].eq(1.0).all()
    assert down.iloc[283].eq(1.5).all()
    assert down.iloc[285].eq(0.0).all()
    assert down.ge(0).all().all() and down.le(1.5).all().all()


def test_base_momentum_does_not_cross_the_research_oos_price_basis():
    from strategies.index_clv_rally_base_long import _base_budget, _by_partition, _close_to_close

    research = pd.bdate_range("2020-09-01", periods=300)
    oos = pd.bdate_range("2022-01-03", periods=90)
    days = research.append(oos)
    level = np.concatenate([
        400.0 - np.arange(len(research)),
        1000.0 + np.arange(len(oos)),
    ])
    frames = _priced_bars(days, level)
    joint = _by_partition(days, frames, lambda subset: _base_budget(_close_to_close(subset)))
    oos_frames = {field: frame.reindex(oos) for field, frame in frames.items()}
    alone = _base_budget(_close_to_close(oos_frames))

    pd.testing.assert_series_equal(joint.loc[oos], alone, check_names=False, check_freq=False)
    assert joint.loc[research[-1]] == 0.0
    assert joint.loc[oos[-1]] == 0.5


def test_base_momentum_counts_votes_instead_of_switching_the_whole_base():
    from strategies.index_clv_rally_base_long import _base_budget

    days = pd.bdate_range("2018-01-02", periods=351)
    returns = pd.Series([np.nan] + [0.005] * 300 + [0.0001] * 50, index=days)
    assert _base_budget(returns).iloc[-1] == 0.5

    # 最后21日跌，但63、126日上涨，底仓保留两份。
    wealth = np.interp(np.arange(301), [0, 174, 237, 279, 300], [100, 80, 90, 110, 100])
    momentum_returns = pd.Series(wealth, index=days[:301]).pct_change()
    assert _base_budget(momentum_returns).iloc[-1] == pytest.approx(1 / 3)
    # 最后21日涨，但63、126日跌，只保留一份。
    wealth = np.interp(np.arange(301), [0, 174, 237, 279, 300], [100, 120, 110, 90, 100])
    assert _base_budget(pd.Series(wealth, index=days[:301]).pct_change()).iloc[-1] == pytest.approx(1 / 6)


def test_base_momentum_zero_return_vote_warmup_and_future_invariance():
    from strategies.index_clv_rally_base_long import _base_budget

    days = pd.bdate_range("2018-01-02", periods=350)
    zero = pd.Series(0.0, index=days)
    zero.iloc[0] = np.nan  # 第一日没有上一收盘，不算一笔已完成收益。
    budgets = _base_budget(zero)
    assert budgets.iloc[20] == 0.5
    assert budgets.iloc[21] == pytest.approx(1 / 3)
    assert budgets.iloc[63] == pytest.approx(1 / 6)
    assert budgets.iloc[126] == 0.0

    returns = pd.Series(np.linspace(-0.004, 0.003, len(days)), index=days)
    original = _base_budget(returns)
    returns.iloc[290:] = -0.1
    pd.testing.assert_series_equal(_base_budget(returns).iloc[:290], original.iloc[:290])


def test_base_components_keep_unused_clv_cash_and_do_not_look_ahead():
    from strategies.index_clv_rally_base_long import BOOK

    days = pd.bdate_range("2018-01-02", periods=170)
    bars = _priced_bars(days, 400.0 - np.arange(len(days)))
    clv = pd.Series(0.4, index=days)
    clv.iloc[140] = 0.1
    rally = pd.Series(1.0, index=days)
    parts = BOOK.components(clv, bars, rally_z=rally)
    assert parts.iloc[140]["底仓预算"] == 0.0
    assert parts.iloc[140]["CLV预算"] == 1.0
    assert parts.iloc[140]["合计"] == 0.0
    assert parts.iloc[141]["合计"] == 1.5
    assert parts.iloc[141]["底仓"] == 0.0
    assert parts.iloc[141]["择时"] == 1.5
    np.testing.assert_allclose((parts["底仓预算"] + parts["CLV预算"]).iloc[1:], 1.0)
    for frame in bars.values():
        frame.iloc[145:] *= 2
    changed = BOOK.components(clv, bars, rally_z=rally)
    pd.testing.assert_frame_equal(parts.iloc[:145], changed.iloc[:145])


def test_base_study_components_preserve_netting_and_lag_each_half():
    from research.index.index_adaptive_attribution import base_target_components

    days = pd.bdate_range("2018-01-02", periods=40)
    frames = _priced_bars(days, np.full(len(days), 100.0))
    clv = pd.Series(0.8, index=days)
    clv.iloc[30:33] = [0.1, 0.1, 0.4]
    rally = pd.Series(1.0, index=days)
    rally.iloc[30] = -1.0
    signs = pd.Series(-1.0, index=days)
    parts = base_target_components(clv, frames, rally, signs)

    assert parts.iloc[0].eq(0.0).all()
    assert parts.iloc[30].tolist() == [-0.5, 0.0, -0.5]
    assert parts.iloc[31].tolist() == [-0.5, 0.5, 0.0]
    assert parts.iloc[32].tolist() == [-0.5, 0.75, 0.25]
    assert parts.iloc[33].tolist() == [-0.5, 0.0, -0.5]


def test_exposure_audit_separates_rounding_from_unchanged_lot_drift():
    from research.index.index_adaptive_attribution import exposure_audit
    from tfcta.costs import INDEX_MULTIPLIER

    days = pd.bdate_range("2018-01-02", periods=3)
    symbols = ["IF", "IH", "IC"]
    opened = pd.DataFrame([[4000, 4000, 6000], [4000, 4000, 6000], [4400, 4400, 6600]], index=days, columns=symbols)
    targets = pd.DataFrame(np.repeat([[0.5], [0.75], [0.75]], 3, axis=1), index=days, columns=symbols)
    raw = pd.DataFrame({"底仓": 0.5, "择时": [0.0, 0.25, 0.25]}, index=days)
    daily = pd.DataFrame({"期初权益": 1e7, "权益": 1e7, "名义敞口": [3.6e6, 7.2e6, 7.92e6],
                          "敞口倍数": [0.36, 0.72, 0.792], "停止新开仓": False,
                          "受阻委托数": 0, "开仓受限数": 0, "换月笔数": 0}, index=days)
    contracts = pd.DataFrame({s: f"{s}1801" for s in symbols}, index=days)
    rows = []
    for i, day in enumerate(days):
        for symbol in symbols:
            quantity = 1 if i == 0 else 2
            rows.append({"交易日": day, "品种": symbol, "手数": quantity,
                         "名义敞口": quantity * opened.loc[day, symbol] * INDEX_MULTIPLIER[symbol]})
    positions = pd.DataFrame(rows)
    bridge, checks = exposure_audit(daily, positions, targets, raw, {"open": opened}, contracts)
    assert bridge["假设每日重算整数手"].tolist() == pytest.approx([0.36, 0.72, 0.396])
    assert bridge["实际开盘敞口"].tolist() == pytest.approx([0.36, 0.72, 0.792])
    assert checks["clean_execution_days"] == 1
    # 保持名义金额与手数一致，但把清洁调仓日少开一手，独立的取整核对必须报错。
    positions.loc[3, "手数"] = 1
    positions.loc[3, "名义敞口"] = 1.2e6
    daily.loc[days[1], ["名义敞口", "敞口倍数"]] = [6e6, 0.6]
    with pytest.raises(ArithmeticError, match="手数"):
        exposure_audit(daily, positions, targets, raw, {"open": opened}, contracts)


def test_exposure_audit_does_not_require_rebalancing_after_a_blocked_target():
    from backtest.engine import simulate_target_lots
    from research.index.index_adaptive_attribution import exposure_audit
    from tfcta.costs import INDEX_MULTIPLIER

    days = pd.bdate_range("2018-01-02", periods=3)
    symbols = ["IF", "IH", "IC"]
    frame = lambda value: pd.DataFrame(value, index=days, columns=symbols)
    opened = pd.DataFrame({s: np.array([1.0, 1.0, 0.98]) * 560000 / INDEX_MULTIPLIER[s] for s in symbols}, index=days)
    target = pd.DataFrame(np.repeat([[0.5], [0.75], [0.5]], 3, axis=1), index=days, columns=symbols)
    high, low = opened + 1.0, opened - 1.0
    high.iloc[1], low.iloc[1] = opened.iloc[1], opened.iloc[1]
    contracts = pd.DataFrame({s: f"{s}1801" for s in symbols}, index=days)
    daily, positions, _ = simulate_target_lots(
        target, opened, opened, opened, (opened.shift(-1) - opened).fillna(0.0), high, low,
        frame(False), contracts, {2018: symbols}, {s: INDEX_MULTIPLIER[s] for s in symbols},
        frame(0.0), frame(0.0), frame(0.0), frame(0.0), initial_capital=1e7,
        margin_rate=0.2, exposure_limit=1.0, drawdown_limit=0.1, halt_days=10,
    )
    assert daily["受阻委托数"].tolist() == [0, 3, 0]
    assert positions.loc[positions["交易日"].eq(days[2]), "手数"].tolist() == [2, 2, 2]
    raw = pd.DataFrame({"底仓": 0.5, "择时": [0.0, 0.25, 0.0]}, index=days)
    _, checks = exposure_audit(daily, positions, target, raw, {"open": opened}, contracts)
    assert checks["clean_execution_days"] == 0


@pytest.mark.parametrize("field,value", [
    ("close", 0.0), ("close", -1.0), ("close", np.inf),
    ("closew", np.inf), ("closew", -np.inf), ("open", np.inf),
])
def test_base_rejects_invalid_prices_instead_of_treating_them_as_warmup(field, value):
    from strategies.index_clv_rally_base_long import BOOK

    days = pd.bdate_range("2018-01-02", periods=320)
    bars = _priced_bars(days, 400.0 - np.arange(len(days)))
    bars[field].loc[days[295], "IF"] = value
    with pytest.raises(ValueError, match="行情|价格"):
        BOOK.targets(pd.Series(0.6, index=days), bars, rally_z=pd.Series(-1.0, index=days))


def test_rally_up_candidate_only_adds_the_declared_signal_and_lags_execution():
    from strategies.index_clv_rally_equal_up_long import BOOK

    days = pd.bdate_range("2018-01-02", periods=80)
    symbols = list(BOOK.SYMBOLS)
    price = pd.DataFrame(100.0, index=days, columns=symbols)
    bars = {
        "open": price.copy(),
        "openw": price.copy(),
        "closew": price.copy(),
        "highw": price + 2.0,
        "loww": price.copy(),
    }

    # Body > 0.7 and CLV >= 0.8: signal is executed on the next session.
    bars["openw"].iloc[20] = 100.2
    bars["closew"].iloc[20] = 101.8
    bars["highw"].iloc[20] = 102.0
    # Strong body without the CLV gate must not signal.
    bars["openw"].iloc[22] = 100.2
    bars["closew"].iloc[22] = 101.78
    bars["highw"].iloc[22] = 102.0
    # Body exactly 0.7 does not pass the strict body threshold.
    bars["openw"].iloc[24] = 100.2
    bars["closew"].iloc[24] = 101.6
    bars["highw"].iloc[24] = 102.0

    clv = pd.Series(0.5, index=days)
    clv.iloc[20] = 0.8
    clv.iloc[22] = 0.79
    clv.iloc[24] = 0.8
    clv.iloc[40] = 0.1
    rally = pd.Series(-1.0, index=days)
    rally.iloc[20] = 1.0
    targets = BOOK.targets(clv, bars, rally_z=rally)

    assert targets.iloc[20].eq(0).all()
    assert targets.iloc[21].eq(1.5).all()
    assert targets.iloc[23].eq(0).all()
    assert targets.iloc[25].eq(0).all()
    assert targets.iloc[41].eq(1.0).all()
    assert BOOK.GROSS == 1.0


def test_rally_up_frozen_book_uses_oos_warmup_only_within_oos_partition():
    from strategies.index_clv_rally_equal_up_long import BOOK

    research_days = pd.bdate_range("2021-10-01", periods=50)
    oos_days = pd.bdate_range("2022-01-03", periods=24)
    days = research_days.append(oos_days)
    symbols = list(BOOK.SYMBOLS)
    research_price = pd.DataFrame(100.0, index=research_days, columns=symbols)
    oos_price = pd.DataFrame(1000.0, index=oos_days, columns=symbols)
    price = pd.concat([research_price, oos_price])
    bars = {
        "open": price.copy(),
        "openw": price.copy(),
        "closew": price.copy(),
        "highw": price + 2.0,
        "loww": price - 2.0,
    }
    clv = pd.Series(0.1, index=days)
    rally = pd.Series(-1.0, index=days)

    targets = BOOK.targets(clv, bars, rally_z=rally)

    # The adjusted-price jump at the partition boundary must not contaminate OOS volatility.
    assert targets.loc[oos_days[0]].eq(1.0).all()
    assert targets.loc[oos_days[1]].eq(1.0).all()


def test_index_oos_routes_to_data_loader_for_frozen_book(monkeypatch):
    import backtest.engine as engine
    from strategies.index_clv_rally_base_long import BOOK as ACTIVE_BOOK
    from strategies.index_clv_rally_equal_up_long import BOOK as FROZEN_BOOK

    monkeypatch.setattr(
        engine,
        "_load_index_research_inputs",
        lambda: (_ for _ in ()).throw(AssertionError("到达股指数据加载入口")),
    )
    with pytest.raises(AssertionError, match="到达股指数据加载入口"):
        engine.run_index(2022, 2022, book=FROZEN_BOOK)
    with pytest.raises(C.HoldoutViolation, match="未冻结"):
        engine.run_index(2022, 2022, book=ACTIVE_BOOK)
    with pytest.raises(SystemExit):
        engine.main(["--market", "index", "--oos"])


def test_annual_index_plots_are_written_for_each_year(tmp_path):
    from backtest.engine import plot_annual_performance

    days = pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-05"])
    strategy = pd.Series([0.01, 0.02, -0.01], index=days)
    benchmark = pd.Series([0.03, -0.01, 0.02], index=days)

    paths = plot_annual_performance(strategy, benchmark, tmp_path, "adaptive")

    assert [path.name for path in paths] == ["2020.png", "2021.png"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_adaptive_signal_grid_matches_its_preregistered_protocol():
    from research.index.index_adaptive_signal_grid import load_protocol

    protocol = load_protocol()
    assert protocol["parameters"]["variants"] == 9












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


def test_high_open_z_does_not_look_ahead():

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






def test_vol_scale_ignores_the_unfinished_open_to_open_return():
    from strategies.index_common import close_vol_scale

    index = pd.bdate_range("2016-01-04", periods=80)
    returns = pd.Series(
        np.resize([0.08, -0.07], len(index)),
        index=index,
    )
    scale = close_vol_scale(returns)
    changed = returns.copy()
    changed.iloc[60:] = 0.0
    revised = close_vol_scale(changed)
    pd.testing.assert_series_equal(scale.iloc[:61], revised.iloc[:61])


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
