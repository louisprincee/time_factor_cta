"""回测绩效、紧缩夏普和过拟合概率。当前没有策略在跑。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import engine
from backtest.engine import (
    METRIC_KEYS, deflated_sharpe, performance, pbo_cscv, plot_performance,
    sharpe_ratio,
)


def test_metrics_hand_values():
    # 两天 +10%、-5%：净值 1.1 * 0.95 = 1.045
    r = pd.Series([0.10, -0.05])
    m = performance(r, periods=252)
    nav = 1.045
    ann = nav ** (252 / 2) - 1
    vol = float(r.std(ddof=1) * np.sqrt(252))
    assert m['n_days'] == 2
    assert m['ann_return'] == pytest.approx(ann)
    assert m['ann_vol'] == pytest.approx(vol)
    assert m['ret_risk'] == pytest.approx(ann / vol)
    assert m['win_rate'] == pytest.approx(0.5)
    # 高点 1.1，回落到 1.045，回撤 1.045/1.1 - 1
    assert m['max_drawdown'] == pytest.approx(1.045 / 1.1 - 1)


def test_annualized_metrics_use_242_trading_days():
    returns = pd.Series([0.01] * 242)
    metrics = performance(returns)
    assert metrics["ann_return"] == pytest.approx(1.01**242 - 1)
    assert metrics["ann_vol"] == pytest.approx(0.0, abs=1e-12)

    varying = pd.Series([0.01, -0.005] * 121)
    expected_sharpe = varying.mean() / varying.std(ddof=1) * np.sqrt(242)
    assert sharpe_ratio(varying) == pytest.approx(expected_sharpe)

def test_performance_and_sharpe_tolerate_tiny_samples():
    for r in (pd.Series(dtype='float64'), pd.Series([0.01])):
        m = performance(r)
        assert set(METRIC_KEYS) <= set(m)
        assert np.isnan(sharpe_ratio(r))

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
    assert .3 < np.mean(noise_pbo) < .7  # 纯噪声时样本内最好的方案在样本外大约一半时间排在中位数以下
    with pytest.raises(ValueError):
        pbo_cscv(noise, 7)


def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(.0006, .01, 1500))
    trials = rng.normal(0, .5, 50)
    few = deflated_sharpe(ret, trials, 5)['dsr']
    many = deflated_sharpe(ret, trials, 500)['dsr']
    assert 0 <= many < few <= 1

def test_drawdown_includes_starting_capital():
    assert performance(pd.Series([-.10, .02]))['max_drawdown'] == pytest.approx(-.10)


def test_daily_equity_frame_records_nav_and_drawdown():
    index = pd.bdate_range("2024-01-02", periods=3)
    result = pd.DataFrame({
        "gross": [0.10, -0.20, 0.05],
        "net": [0.10, -0.20, 0.05],
        "turnover": [1.0, 2.0, 1.0],
        "cost": [0.0, 0.0, 0.0],
    }, index=index)

    daily = engine._daily_equity_frame(result, {"策略": "test"})

    assert daily["净值"].tolist() == pytest.approx([1.10, 0.88, 0.924])
    assert daily["回撤"].tolist() == pytest.approx([0.0, -0.20, -0.16])
    assert daily["交易日"].tolist() == list(index)


def test_performance_plot_writes_nav_and_drawdown_chart(tmp_path):
    returns = pd.Series([0.02, -0.03, 0.01], index=pd.bdate_range("2024-01-02", periods=3))
    path = plot_performance(returns, tmp_path)
    assert path == tmp_path / "performance.png"
    assert path.is_file() and path.stat().st_size > 0


@pytest.mark.parametrize(
    ("argument", "runner"),
    [
        ("--research", "run_integer_lot_research"),
        ("--validation", "run_integer_lot_validation"),
        ("--oos", "run_integer_lot_oos"),
    ],
)
def test_standard_period_cli_uses_integer_lot_runner(monkeypatch, argument, runner):
    called = []
    monkeypatch.setattr(engine, runner, lambda: called.append(runner))

    engine.main([argument])

    assert called == [runner]


def test_validation_cost_coverage_rejects_missing_symbols_instead_of_dropping_them():
    ticks = pd.DataFrame({"symbol": ["RB"], "year": [2021], "tick": [1.0]})
    fees = pd.DataFrame({"symbol": ["RB"]})

    with pytest.raises(ValueError, match="拒绝静默剔除品种.*tick 缓存缺"):
        engine._require_cost_coverage(["RB", "CU"], ticks, fees, "2022 验证")








def test_integer_lot_research_cli_dispatches_without_oos(monkeypatch):
    called = []
    monkeypatch.setattr(
        engine, "run_integer_lot_research", lambda: called.append("integer-lots")
    )

    engine.main(["--integer-lot-research"])

    assert called == ["integer-lots"]


def test_integer_lot_validation_cli_dispatches(monkeypatch):
    called = []
    monkeypatch.setattr(
        engine, "run_integer_lot_validation", lambda: called.append("integer-lot-validation")
    )

    engine.main(["--integer-lot-validation"])

    assert called == ["integer-lot-validation"]


def test_integer_lot_oos_cli_dispatches_to_guarded_runner(monkeypatch):
    called = []
    monkeypatch.setattr(
        engine, "run_integer_lot_oos", lambda: called.append("integer-lot-oos")
    )

    engine.main(["--integer-lot-oos"])

    assert called == ["integer-lot-oos"]




def test_benjamini_hochberg_adjusts_finite_p_values_and_preserves_nan():
    adjusted = engine._benjamini_hochberg([0.01, 0.04, 0.03, np.nan])

    assert adjusted[:3] == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])




def test_oos_run_rejects_unfrozen_strategy_before_data_access(monkeypatch):
    from strategies import ts_low

    monkeypatch.setattr(
        engine, "_oos_audit_files",
        lambda: pytest.fail("unfrozen OOS must fail before inspecting test inputs"),
    )
    monkeypatch.setattr(ts_low, "load_config", lambda allow_frozen=False: {"chosen": None})
    with pytest.raises(ValueError, match="chosen"):
        engine.run("oos")


def test_period_backtest_saves_csv_and_nav_drawdown_plot(tmp_path, monkeypatch):
    index = pd.bdate_range("2021-01-04", periods=8)
    returns = pd.DataFrame({"RB": 0.01}, index=index)
    zeros = returns * 0.0
    data = engine.StudyData(
        factors=engine.SignalSet(bars={}),
        returns=returns,
        universe={2021: ["RB"]},
        rolls=zeros.astype(bool),
        open_fee=zeros,
        close_fee=zeros,
        roll_close_fee=zeros,
        slippage=zeros,
    )
    spec = {
        "回看": "一季",
        "回看交易日": 2,
        "阈值": "前10%",
        "分位": 0.9,
        "持仓": "隔夜",
        "持仓交易日": 1,
    }
    factor = pd.DataFrame({"RB": [0, 0, 0, 1, 1, 1, 1, 1]}, index=index)
    monkeypatch.setattr(engine.C, "RUNS_DIR", tmp_path)

    engine._run_period(
        data, factor, [spec], index[-1], range(2021, 2022), "研究期", "research.csv",
    )

    assert (tmp_path / "strategies/ts_low/research.csv").is_file()
    daily_path = tmp_path / "strategies/ts_low/research_daily.csv"
    assert daily_path.is_file()
    daily = pd.read_csv(daily_path, encoding="utf-8-sig")
    assert len(daily) == len(index)
    assert {"交易日", "净收益", "净值", "回撤", "成本", "换手"} <= set(daily.columns)
    assert daily.iloc[-1]["净值"] == pytest.approx(
        (1 + daily["净收益"]).prod()
    )
    plot = (
        tmp_path / "strategies/ts_low/plots/research"
        / "lookback_2_quantile_0.9_horizon_1/performance.png"
    )
    assert plot.is_file() and plot.stat().st_size > 0
