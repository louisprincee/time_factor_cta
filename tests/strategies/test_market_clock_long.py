"""只做多全市场时钟：事前标准化、池内聚合、保护与分散、目标权重和登记一致性。合成数据。"""
import json

import numpy as np
import pandas as pd
import pytest

from strategies import market_clock_long as M
from tfcta import config as C


def _signed(index, symbols, seed=0):
    rng = np.random.default_rng(seed)
    return {
        name: pd.DataFrame(rng.normal(size=(len(index), len(symbols))), index=index, columns=symbols)
        for name in M.FACTORS
    }


def _bars(index, symbols, seed=1, vol=None):
    rng = np.random.default_rng(seed)
    vol = np.full(len(symbols), 0.01) if vol is None else np.asarray(vol)
    steps = rng.normal(size=(len(index), len(symbols))) * vol
    close = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=symbols)
    location = pd.DataFrame(rng.uniform(-0.9, 0.9, size=close.shape), index=index, columns=symbols)
    return {"close": close, "closew": close * (1 + 0.01 * location),
            "highw": close * 1.01, "loww": close * 0.99}


def _universe(index, symbols):
    return {int(y): list(symbols) for y in sorted(set(index.year))}


def test_frozen_config_matches_the_implementation():
    config = M.load_config()
    assert config["status"] == "frozen" and config["frozen_at"]
    assert config["chosen"] == config["parameters"]
    assert config["chosen"]["direction"] == "long_only"
    assert "不读 2026" in config["sample"]
    spec = M.chosen_spec()
    assert spec == {**M.BASE, "vol_scale": True, "trend_cap": True, "weights": "inverse_vol"}
    assert M.EXECUTION == "target"


def test_frozen_spec_matches_the_registered_candidate():
    path = C.CONFIG_DIR / "research" / "market_clock_long_robustness.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    candidate = next(v for v in protocol["variants"] if v["id"] == protocol["candidate"])
    spec = M.chosen_spec()
    assert all(spec[key] == candidate[key] for key in ("score", "vol_scale", "trend_cap", "weights"))


def test_robustness_protocol_lists_the_candidate_and_all_variants():
    path = C.CONFIG_DIR / "research" / "market_clock_long_robustness.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    ids = [v["id"] for v in protocol["variants"]]
    assert protocol["candidate"] in ids and ids[0] == "base"
    assert "不读 2026" in protocol["sample"]


def test_own_history_z_ignores_the_current_and_future_days():
    index = pd.bdate_range("2015-01-01", periods=300)
    frame = pd.DataFrame({"A": np.arange(300, dtype=float)}, index=index)
    z = M.own_history_z(frame)
    later = frame.copy()
    later.iloc[-1, 0] = 1e6
    pd.testing.assert_series_equal(z["A"].iloc[:-1], M.own_history_z(later)["A"].iloc[:-1])
    assert z["A"].iloc[:M.Z_MIN_PERIODS].isna().all()
    prior = frame["A"].iloc[-1 - M.Z_WINDOW:-1]
    assert z["A"].iloc[-1] == pytest.approx((299 - prior.mean()) / prior.std())
    assert M.own_history_z(later)["A"].iloc[-1] == M.Z_CLIP


def test_too_few_symbols_gives_no_score():
    index = pd.bdate_range("2015-01-01", periods=400)
    symbols = [f"S{i:02d}" for i in range(M.MIN_SYMBOLS - 1)]
    score = M.market_score(_signed(index, symbols), {2016: symbols}, index[index.year == 2016])
    assert score.isna().all()


def test_factor_frames_must_cover_the_backtest_window():
    index = pd.bdate_range("2015-01-01", periods=400)
    symbols = [f"S{i:02d}" for i in range(12)]
    signed = _signed(index, symbols)
    signed["dfp_top3"] = signed["dfp_top3"].iloc[:300]
    with pytest.raises(ValueError, match="dfp_top3"):
        M.market_score(signed, {2016: symbols}, index[index.year == 2016])


def test_base_targets_are_the_lagged_share_of_bullish_closes():
    index = pd.bdate_range("2015-01-01", periods=420)
    symbols = [f"S{i:02d}" for i in range(12)]
    signed, bars = _signed(index, symbols), _bars(index, symbols)
    scored = index[index.year >= 2016]
    universe = _universe(scored, symbols)
    target = M.targets(signed, bars, universe, scored, symbols, {})
    bullish = M.market_score(signed, universe, scored).gt(0).astype(float)
    expected = bullish.rolling(5, min_periods=1).mean().shift(1).fillna(0.0)
    assert np.allclose(target["S00"], expected)
    assert (target.nunique(axis=1) <= 1).all()
    later = {k: v.copy() for k, v in signed.items()}
    for frame in later.values():
        frame.iloc[-1] = 9.0
    pd.testing.assert_frame_equal(
        target.iloc[:-1], M.targets(later, bars, universe, scored, symbols, {}).iloc[:-1],
    )


def test_protections_only_reduce_and_inverse_vol_tilts_to_quiet_names():
    index = pd.bdate_range("2014-06-02", periods=700)
    symbols = [f"S{i:02d}" for i in range(12)]
    signed = _signed(index, symbols, seed=4)
    bars = _bars(index, symbols, vol=[0.005] * 6 + [0.03] * 6)
    scored = index[index.year >= 2016]
    universe = _universe(scored, symbols)
    base = M.targets(signed, bars, universe, scored, symbols, {})
    protected = M.targets(signed, bars, universe, scored, symbols, {"vol_scale": True, "trend_cap": True})
    assert (protected <= base + 1e-12).all().all()
    tilted = M.targets(signed, bars, universe, scored, symbols, {"weights": "inverse_vol"})
    live = tilted.sum(axis=1) > 0
    assert live.any()
    assert (tilted.loc[live, "S00"] > tilted.loc[live, "S11"]).all()
    assert tilted.to_numpy().max() <= M.WEIGHT_CAP + 1e-12


def test_targets_are_held_until_the_bullish_share_changes():
    index = pd.bdate_range("2014-06-02", periods=700)
    symbols = [f"S{i:02d}" for i in range(12)]
    signed, bars = _signed(index, symbols, seed=6), _bars(index, symbols)
    scored = index[index.year >= 2016]
    universe = _universe(scored, symbols)
    tilted = M.targets(signed, bars, universe, scored, symbols)
    base = M.targets(signed, bars, universe, scored, symbols, {})
    same_share = base["S00"].eq(base["S00"].shift(1)) & base["S00"].gt(0)
    assert same_share.any()
    assert tilted["S00"][same_share].eq(tilted["S00"].shift(1)[same_share]).all()


def test_clv_control_buys_after_weak_closes():
    index = pd.bdate_range("2015-01-01", periods=420)
    symbols = [f"S{i:02d}" for i in range(12)]
    bars = _bars(index, symbols)
    weak = index[-2]
    bars["closew"].loc[weak] = bars["loww"].loc[weak]
    scored = index[index.year >= 2016]
    target = M.targets({}, bars, _universe(scored, symbols), scored, symbols, {"score": "clv"})
    assert target.loc[index[-1], "S00"] > 0


def test_year_tests_open_in_order_and_only_once(tmp_path):
    from backtest.engine import check_year_order, year_test_name

    ledger = tmp_path / "ledger.jsonl"
    check_year_order(2022, ledger)
    with pytest.raises(C.HoldoutViolation, match="2022"):
        check_year_order(2023, ledger)
    C.append_ledger({"kind": "year_test", "status": "opened", "strategy": year_test_name(2022)}, ledger)
    with pytest.raises(C.HoldoutViolation, match="打开过"):
        check_year_order(2022, ledger)
    with pytest.raises(C.HoldoutViolation):
        check_year_order(2023, ledger)
    C.append_ledger({"kind": "year_test", "status": "finished", "strategy": year_test_name(2022)}, ledger)
    check_year_order(2023, ledger)
    with pytest.raises(C.HoldoutViolation, match="2023"):
        check_year_order(2024, ledger)
    with pytest.raises(C.HoldoutViolation):
        check_year_order(2026, ledger)
