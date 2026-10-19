"""回测绩效、紧缩夏普、过拟合概率和命令行分派。"""
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


def test_validation_time_factor_uses_timestamp_difference_not_daily_price_range():
    day = pd.Timestamp("2022-01-03")
    history = pd.DataFrame({"RB": [0.25]}, index=[pd.Timestamp("2021-12-31")])
    panel = pd.DataFrame({
        "trading_date": [day],
        "symbol": ["RB"],
        "ts_low": [0.9],
        "ts_high": [0.2],
        "highw": [120.0],
        "loww": [80.0],
    })

    factor = engine._validation_time_factor(history, panel)

    assert factor.loc[day, "RB"] == pytest.approx(0.7)
    assert factor.loc[pd.Timestamp("2021-12-31"), "RB"] == pytest.approx(0.25)

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
    monkeypatch.setattr(engine, runner, lambda **kwargs: called.append(runner))

    engine.main([argument])

    assert called == [runner]


def test_validation_cost_coverage_rejects_missing_symbols_instead_of_dropping_them():
    ticks = pd.DataFrame({"symbol": ["RB"], "year": [2021], "tick": [1.0]})
    fees = pd.DataFrame({"symbol": ["RB"]})

    with pytest.raises(ValueError, match="拒绝静默剔除品种.*tick 缓存缺"):
        engine._require_cost_coverage(["RB", "CU"], ticks, fees, "2022 验证")








def test_market_argument_is_forwarded(monkeypatch):
    seen = {}

    def grab(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(engine, "run_integer_lot_research", grab)
    engine.main(["--research", "--market", "index"])
    assert seen == {"market": "index"}

    engine.main(["--research"])
    assert seen == {"market": "commodity"}


def test_oos_refuses_before_any_holdout_read(monkeypatch):
    monkeypatch.setattr(
        engine, "load_oos",
        lambda *args, **kwargs: pytest.fail("未冻结策略不能读取 2023–2025"),
    )
    monkeypatch.setattr(
        engine, "_oos_audit_files",
        lambda: pytest.fail("未冻结策略不能清点样本外输入"),
    )

    def unfrozen():
        raise ValueError("当前保留的策略没有冻结，不能读取 2023–2025")

    monkeypatch.setattr(engine, "_frozen_strategy_specs", unfrozen)
    with pytest.raises(ValueError, match="没有冻结"):
        engine.run_integer_lot_oos()
