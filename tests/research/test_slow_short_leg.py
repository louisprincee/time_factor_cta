"""慢趋势空头腿检验：篮子同冻结策略空头腿、只用决策时已知的数据、品种池宽度锁定、只开一次。合成数据。"""
import numpy as np
import pandas as pd
import pytest

from research import slow_short_leg as L
from strategies import market_clock_slow_short as T


def _bars(n=700, symbols=("A", "B", "C", "D"), seed=3):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=n)
    close = pd.DataFrame(100 * np.exp(np.cumsum(-0.0008 + 0.01 * rng.normal(size=(n, len(symbols))), axis=0)),
                         index=index, columns=list(symbols))
    opened = close.shift(1).fillna(close.iloc[0]) * (1 + 0.001 * rng.normal(size=close.shape))
    return {"close": close, "closew": close - 40.0, "open": opened, "openw": opened - 40.0}


def _carry(bars, seed=4):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=bars["close"].shape), index=bars["close"].index,
                        columns=bars["close"].columns)


def _universe(symbols=("A", "B", "C", "D")):
    return {y: list(symbols) for y in (2011, 2012)}


def test_protocol_matches_the_code():
    protocol = L.load_protocol()
    assert protocol["pools"]["expected_widths"] == {"2011": 11, "2012": 10, "2013": 12, "2014": 11, "2015": 17}
    assert "不读 2022" in protocol["sample"] and "不读 2026" in protocol["sample"]


def test_basket_equals_the_frozen_short_leg_on_bear_days():
    bars, universe = _bars(), _universe()
    carry = _carry(bars)
    p = L.panels(bars, universe, carry, universe)
    index, columns = bars["close"].index, list(bars["close"].columns)
    lagged = carry.shift(1)
    trend = T.symbol_trend(bars, columns)
    bear = p["bear"]["严格"]
    weights = T.decision_weights(pd.Series(-1.0, index=index), lagged, trend, bear)   # 分数为负，只剩空头腿
    pool = pd.DataFrame(index.year.isin([2011, 2012])[:, None].repeat(4, axis=1), index=index, columns=columns)
    short = weights.lt(0) & pool
    assert bear.any()
    assert short.sum(axis=1)[bear].tolist() == p["basket_n"][bear].tolist()


def test_decision_at_t_does_not_use_later_data():
    bars, universe = _bars(), _universe()
    carry = _carry(bars)
    day = bars["close"].index[600]
    base = L.panels(bars, universe, carry, universe)
    later = {k: v.copy() for k, v in bars.items()}
    for frame in later.values():
        frame.loc[frame.index > day] *= 1.7
    changed_carry = carry.copy()
    changed_carry.loc[changed_carry.index >= day] = 5.0          # 收盘 t 只看到 t−1 的展期
    moved = L.panels(later, universe, changed_carry, universe)
    for key in ("basket_n", "width"):
        assert base[key].loc[:day].equals(moved[key].loc[:day])
    assert base["bear"]["严格"].loc[:day].equals(moved["bear"]["严格"].loc[:day])


def test_forward_is_the_open_after_next_open_and_leg_is_minus_basket_over_width():
    bars, universe = _bars(), _universe()
    p = L.panels(bars, universe, _carry(bars), universe)
    openw, open_ = bars["openw"], bars["open"]
    day, nxt, after = bars["close"].index[[500, 501, 502]]
    expected = ((openw.loc[after] - openw.loc[nxt]) / open_.loc[nxt]).mean()
    assert p["pool_fwd"].loc[day] == pytest.approx(expected)
    bear = pd.Series(True, index=bars["close"].index)
    table = L.daily_quantities(p, bear)
    assert table.loc[day, "空头腿贡献"] == pytest.approx(-p["basket_sum"].loc[day] / 4)
    assert L.daily_quantities(p, ~bear)["空头腿贡献"].eq(0).all()


def test_pool_widths_must_match_the_declaration():
    protocol = L.load_protocol()
    stats = pd.DataFrame({
        "symbol": ["A", "A"], "year": [2010, 2011], "turnover_median": [4e9] * 2,
        "valid_ratio": [0.99] * 2, "valid_days": [242] * 2, "night_class": ["no_night"] * 2,
    })
    with pytest.raises(ValueError, match="宽度"):
        L.early_universe(stats, protocol)


def test_second_run_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "OUT", tmp_path)
    (tmp_path / "summary.csv").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="只开一次"):
        L.run(L.load_protocol())
