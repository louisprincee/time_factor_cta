"""绩效、紧缩夏普、过拟合概率、命令行分派和 2022 以后的台账门槛。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest import engine
from backtest.engine import (
    METRIC_KEYS, deflated_sharpe, max_drawdown, newey_west_t, pbo_cscv, performance,
    plot_performance, sharpe_ratio,
)
from tfcta import config as C


def test_metrics_hand_values():
    # 两天 +10%、-5%：净值 1.1 * 0.95 = 1.045
    r = pd.Series([0.10, -0.05])
    m = performance(r, periods=252)
    ann = 1.045 ** (252 / 2) - 1
    vol = float(r.std(ddof=1) * np.sqrt(252))
    assert m['n_days'] == 2
    assert m['ann_return'] == pytest.approx(ann)
    assert m['ann_vol'] == pytest.approx(vol)
    assert m['ret_risk'] == pytest.approx(ann / vol)
    assert m['win_rate'] == pytest.approx(0.5)
    assert m['max_drawdown'] == pytest.approx(1.045 / 1.1 - 1)
    assert m['calmar'] == pytest.approx(m['ann_return'] / abs(m['max_drawdown']))


def test_write_results_includes_calmar_in_summary_and_annual_csv(monkeypatch, tmp_path):
    dates = pd.to_datetime(["2019-01-02", "2019-01-03"])
    returns = pd.Series([0.10, -0.05], index=dates)
    equity = 100_000_000.0 * (1.0 + returns).cumprod()
    daily = pd.DataFrame({
        "净收益": returns,
        "权益": equity,
        "期初权益": equity.shift(1).fillna(100_000_000.0),
        "敞口倍数": [0.5, 0.5],
        "保证金权益占用": [0.1, 0.1],
        "开仓笔数": [1, 0],
        "平仓笔数": [0, 1],
        "换月笔数": [0, 0],
        "受阻委托数": [0, 0],
    })
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(engine, "plot_performance", lambda *args, **kwargs: None)
    strategy = SimpleNamespace(NAME="测试策略", DETAIL="合成数据", GROSS=1.0, ID="test")

    summary, annual = engine.write_results(
        2019, strategy, daily, pd.DataFrame(), pd.DataFrame(), []
    )

    expected_calmar = performance(returns)["calmar"]
    assert summary["Calmar比率"] == pytest.approx(expected_calmar)
    assert annual[0]["Calmar比率"] == pytest.approx(expected_calmar)
    saved_summary = pd.read_csv(tmp_path / "strategies" / "test" / "commodity" / "2019" / "summary.csv")
    saved_annual = pd.read_csv(tmp_path / "strategies" / "test" / "commodity" / "2019" / "annual.csv")
    assert saved_summary.loc[0, "Calmar比率"] == pytest.approx(expected_calmar)
    assert saved_annual.loc[0, "Calmar比率"] == pytest.approx(expected_calmar)
    engine.write_results(
        2019,
        strategy,
        daily,
        pd.DataFrame(),
        pd.DataFrame(),
        [],
        market="index",
        result_subdirectory=Path("individual") / "IF",
    )
    assert (
        tmp_path
        / "strategies"
        / "test"
        / "index"
        / "individual"
        / "IF"
        / "2019"
        / "summary.csv"
    ).is_file()


def test_annualized_metrics_and_sharpe_use_242_trading_days():
    assert performance(pd.Series([0.01] * 242))["ann_return"] == pytest.approx(1.01 ** 242 - 1)
    varying = pd.Series([0.01, -0.005] * 121)
    assert sharpe_ratio(varying) == pytest.approx(varying.mean() / varying.std(ddof=1) * np.sqrt(242))


def test_performance_and_sharpe_tolerate_tiny_samples():
    for r in (pd.Series(dtype='float64'), pd.Series([0.01])):
        assert set(METRIC_KEYS) <= set(performance(r))
        assert np.isnan(sharpe_ratio(r))


def test_drawdown_includes_starting_capital():
    assert performance(pd.Series([-.10, .02]))['max_drawdown'] == pytest.approx(-.10)
    assert max_drawdown([90.0, 95.0, 80.0, 120.0], 100.0) == pytest.approx(-0.2)


def test_newey_west_t_matches_plain_t_without_autocorrelation_and_shrinks_with_it():
    rng = np.random.default_rng(3)
    white = rng.normal(0.001, 0.01, 5000)
    plain = white.mean() / white.std(ddof=0) * np.sqrt(len(white))
    assert newey_west_t(white, 10) == pytest.approx(plain, rel=0.1)
    persistent = np.repeat(rng.normal(0.001, 0.01, 500), 10)
    plain = persistent.mean() / persistent.std(ddof=0) * np.sqrt(len(persistent))
    assert abs(newey_west_t(persistent, 10)) < abs(plain) * 0.6
    assert np.isnan(newey_west_t([0.1, 0.2], 10))


def test_pbo_separates_noise_from_a_real_edge():
    idx = pd.bdate_range('2016-01-01', periods=1200)
    noise_pbo = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noise = pd.DataFrame(rng.normal(0, .01, (1200, 20)), index=idx)
        edge = noise.copy()
        edge[0] += .002
        assert pbo_cscv(edge, 8)['pbo'] < .05
        noise_pbo.append(pbo_cscv(noise, 8)['pbo'])
    assert .3 < np.mean(noise_pbo) < .7
    with pytest.raises(ValueError):
        pbo_cscv(noise, 7)


def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(.0006, .01, 1500))
    trials = rng.normal(0, .5, 50)
    assert 0 <= deflated_sharpe(ret, trials, 500)['dsr'] < deflated_sharpe(ret, trials, 5)['dsr'] <= 1


def test_performance_plot_writes_nav_and_drawdown_chart(tmp_path):
    returns = pd.Series([0.02, -0.03, 0.01], index=pd.bdate_range("2024-01-02", periods=3))
    path = plot_performance(returns, tmp_path)
    assert path == tmp_path / "performance.png" and path.stat().st_size > 0


def test_cost_coverage_rejects_missing_symbols_instead_of_dropping_them():
    ticks = pd.DataFrame({"symbol": ["RB"], "year": [2021], "tick": [1.0]})
    fees = pd.DataFrame({"symbol": ["RB"]})
    with pytest.raises(ValueError, match="拒绝静默剔除品种.*tick 缓存缺"):
        engine._require_cost_coverage(["RB", "CU"], ticks, fees, "2022")


@pytest.mark.parametrize(("argv", "window"), [
    (["--research"], (2011, 2021, False)),
    (["--research", "--by-year"], (2011, 2021, True)),
    (["--validation"], (2022, 2022, False)),
    (["--oos"], (2023, 2025, False)),
    (["--oos", "--by-year"], (2023, 2025, True)),
    (["--year", "2024"], (2024, 2024, False)),
    (["--start", "2018", "--end", "2020"], (2018, 2020, False)),
    (["--year", "2019"], (2019, 2019, False)),
])
def test_cli_maps_every_switch_to_one_span(monkeypatch, argv, window):
    import research.market_clock_slow_short_20_60_120 as driver

    seen = []
    monkeypatch.setattr(driver, "run_span", lambda start, end, by_year=False, book=None: seen.append(
        (start, end, by_year)))
    engine.main(argv)
    assert seen == [window]


def test_cli_passes_a_named_frozen_replacement_book(monkeypatch):
    import research.market_clock_slow_short_20_60_120 as driver
    from strategies import commodity_close_location_reversal as R

    seen = []
    monkeypatch.setattr(driver, "run_span", lambda *a, by_year=False, book=None: seen.append(book))
    engine.main([R.STRICT_SWAP_BOOK.ID, "--research"])

    assert seen == [R.STRICT_SWAP_BOOK]


def test_cli_routes_index_research_to_the_shared_engine(monkeypatch):
    seen = []
    monkeypatch.setattr(
        engine,
        "run_index_research",
        lambda start, end, by_year=False, include_individuals=False: seen.append(
            (start, end, by_year, include_individuals)
        ),
    )

    engine.main(["index_close_location_reversal_strict_swap", "--research"])

    assert seen == [(2015, 2021, False, False)]


def test_index_cli_can_request_single_symbol_ledgers(monkeypatch):
    seen = []
    monkeypatch.setattr(
        engine,
        "run_index_research",
        lambda start, end, by_year=False, include_individuals=False: seen.append(
            (start, end, include_individuals)
        ),
    )

    engine.main(
        [
            "index_close_location_reversal_strict_swap",
            "--research",
            "--index-individuals",
        ]
    )

    assert seen == [(2015, 2021, True)]


def test_index_cli_rejects_later_windows_before_loading_data(monkeypatch):
    called = []
    monkeypatch.setattr(
        engine,
        "_load_index_research_inputs",
        lambda: called.append(True),
    )

    with pytest.raises(SystemExit):
        engine.main(["--market", "index", "--year", "2024"])

    assert called == []


@pytest.mark.parametrize("argv", [
    [],     ["--start", "2018"], ["--year", "2019", "--by-year"], ["--research", "--strategy", "没有这条"],
    ["没有这条", "--research"],
])
def test_cli_rejects_incomplete_or_unknown_arguments(argv):
    with pytest.raises(SystemExit):
        engine.main(argv)


def _freeze_books(monkeypatch, tmp_path, books):
    """把点名的几条复制成冻结版本，只在测试里用。"""
    for book in books:
        config = json.loads(book.CONFIG_PATH.read_text(encoding="utf-8"))
        config.update(status="frozen", frozen_at="2026-10-21", chosen=config["parameters"],
                      sample="冻结测试。不读 2026。")
        path = tmp_path / book.CONFIG_PATH.name
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(book, "CONFIG_PATH", path)


def test_later_windows_are_refused_before_any_ledger_write_until_frozen(monkeypatch, tmp_path):
    import research.market_clock_slow_short_20_60_120 as driver
    from strategies.catalog import BOOKS

    book = BOOKS[0]
    config = json.loads(book.CONFIG_PATH.read_text(encoding="utf-8"))
    config = {k: v for k, v in config.items() if k not in ("frozen_at", "chosen")}
    config.update(status="exploratory_predeclared", sample="不读取2022及2023年以后，不读 2026。")
    path = tmp_path / "explore.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(book, "CONFIG_PATH", path)
    monkeypatch.setattr(driver, "_load", lambda end: pytest.fail("没冻结不该读行情"))
    for start, end in ((2022, 2022), (2024, 2024), (2023, 2025)):
        ledger = tmp_path / f"ledger_{start}_{end}.jsonl"
        with pytest.raises(C.HoldoutViolation, match="还没冻结"):
            driver.run_span(start, end, ledger=ledger)
        assert not ledger.exists()
    with pytest.raises(C.HoldoutViolation, match="2026"):
        driver.run_span(2026, 2026, ledger=tmp_path / "x.jsonl")


def test_frozen_books_ledger_before_loading_with_the_real_frozen_spec(monkeypatch, tmp_path):
    import research.market_clock_slow_short_20_60_120 as driver
    from strategies.catalog import FROZEN_BOOKS as BOOKS

    _freeze_books(monkeypatch, tmp_path, BOOKS)
    ledger = tmp_path / "ledger.jsonl"
    seen = []

    def fake_load(end):
        opened = [e for e in C.read_ledger(ledger) if e.get("status") == "opened"]
        seen.append((opened[-1]["strategy"], opened[-1]["frozen_at"], opened[-1].get("oos_start")))
        return object()

    monkeypatch.setattr(driver, "_load", fake_load)
    monkeypatch.setattr(driver, "_book", lambda *args, **kwargs: None)
    driver.run_span(2024, 2024, ledger=ledger)
    assert [name for name, _, _ in seen] == [f"{BOOKS[0].ID}_2024"]
    assert all(frozen == "2026-10-21" and start == "2024-01-01" for _, frozen, start in seen)
    driver.run_span(2022, 2022, ledger=ledger)
    assert seen[-1][0] == f"{BOOKS[0].ID}_2022"
    hashed = [e for e in C.read_ledger(ledger) if e.get("strategy") == f"{BOOKS[0].ID}_2022"][0]["sha256"]
    assert len(hashed) == 2 and {
        "strategies/market_clock_slow_short_20_60_120/__init__.py",
    } <= set(hashed)


def test_an_opened_window_without_results_needs_a_void_before_it_can_reopen(monkeypatch, tmp_path):
    import research.market_clock_slow_short_20_60_120 as driver
    from strategies.catalog import book_named

    book = book_named("按池宽1倍加三窗口趋势空头")
    _freeze_books(monkeypatch, tmp_path, [book])
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path / "runs")
    ledger = tmp_path / "ledger.jsonl"
    C.append_ledger({"kind": "year_test", "status": "opened", "strategy": f"{book.ID}_2022"}, ledger)
    monkeypatch.setattr(driver, "_load", lambda end: pytest.fail("没作废不该重读"))
    with pytest.raises(C.HoldoutViolation, match="作废"):
        driver.run_span(2022, 2022, ledger=ledger)
    C.append_ledger({"kind": "void", "status": "void", "strategy": f"{book.ID}_2022"}, ledger)
    monkeypatch.setattr(driver, "_load", lambda end: object())
    monkeypatch.setattr(driver, "_book", lambda *args, **kwargs: None)
    driver.run_span(2022, 2022, ledger=ledger)
    last = C.read_ledger(ledger)[-1]
    assert last["status"] == "finished" and "作废" in last["note"]


def test_research_span_can_start_in_2011_but_not_earlier_or_across_2022(monkeypatch, tmp_path):
    import research.market_clock_slow_short_20_60_120 as driver

    monkeypatch.setattr(C, "RUNS_DIR", tmp_path / "runs")
    seen = []
    monkeypatch.setattr(driver, "_load", lambda end, start=None: seen.append((start, end)) or object())
    monkeypatch.setattr(driver, "_book", lambda data, years, period, item: seen.append((years[0], years[-1], period)))
    driver.run_span(2011, 2021)
    assert seen == [(2011, 2021), (2011, 2021, "research")]
    with pytest.raises(C.HoldoutViolation, match="2011"):
        driver.run_span(2010, 2021)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="2016 年以前"):
        driver._load(2022, 2011)
