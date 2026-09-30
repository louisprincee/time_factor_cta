"""日内 ORB 测试：撮合口径、向量化撮合对照逐 bar 参考循环、手续费口径、
开盘区间与元标签不使用未来信息。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta.research.backtest import costs
from tfcta.research.intraday import engine as E
from tfcta.research.intraday import walk_forward as P


def _bar_time(day, minute: int) -> pd.Timestamp:
    """当天 09:00 之后的第 minute 分钟。

    ``Timedelta(hours=, minutes=)`` 会转成没有单位的 NumPy timedelta，新版本会报警。
    """
    return pd.Timestamp(day) + pd.Timedelta(9 * 60 + int(minute), unit="min")


def test_train_years_stop_before_the_predicted_year():
    assert P.train_years(2019) == [2016, 2017, 2018]
    assert P.train_years(2017) == [2016]
    assert P.train_years(2016) == []
    assert 2022 not in P.train_years(2022)


@pytest.mark.parametrize("model", P.MODELS)
def test_models_skip_empty_columns_and_rank_the_signal(model):
    """训练集里全缺 / 常数的列要剔掉；预测值要和真实信号同向（logit 的 > 0 即概率 > 50%）。"""
    rng = np.random.default_rng(0)
    n = 4000
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n),
                      "empty": np.nan, "const": 1.0})
    X.loc[rng.random(n) < 0.1, "noise"] = np.nan
    y = 0.5 * signal + rng.normal(size=n)
    train = np.arange(n) < 3000
    pred, _, use = P.fit_predict(model, X, y, train, ~train)
    assert set(use) == {"signal", "noise"}
    assert np.isfinite(pred).all() and len(pred) == (~train).sum()
    assert np.corrcoef(pred, signal[~train])[0, 1] > 0.5


# --------------------------------------------------------------------------
# ORB 撮合
# --------------------------------------------------------------------------
def _orb_days(bars=6, days=2):
    """每天 ``bars`` 根平盘 bar（高 101 / 低 99），开盘区间取前 2 根。"""
    rows = []
    for day in pd.bdate_range("2021-01-04", periods=days):
        for b in range(bars):
            rows.append({"timestamp": _bar_time(day, b + 1), "open": 100.0,
                         "openw": 100.0, "highw": 101.0, "loww": 99.0, "closew": 100.0,
                         "volume": 1.0, "trading_date": day})
    return pd.DataFrame(rows).set_index("timestamp")


ORB2 = E.OrbConfig(2)


def _set(df, i, **values):
    df.iloc[i, df.columns.get_indexer([k + "w" if k != "volume" else k for k in values])] = \
        list(values.values())


def test_orb_waits_for_the_range_and_exits_at_the_close():
    df = _orb_days()
    _set(df, 1, high=103.0)                  # 区间内的突破不算，只抬高上轨
    _set(df, 3, high=103.5)                  # 第 4 根穿过上轨 103 → 按 103 入场
    _set(df, 5, close=104.0)                 # 尾盘按收盘价平
    tr = E.orb_trades(E.prepare(df), ORB2, 1)
    assert tr["day"].tolist() == [0] and tr["entry_bar"].tolist() == [3]
    assert tr["exit_bar"].tolist() == [5] and tr["entry"][0] == 103.0 and tr["exit"][0] == 104.0
    assert tr["gross"][0] == pytest.approx(1.0 / 100.0)
    # 只碰到不算穿过
    df2 = _orb_days()
    _set(df2, 3, high=101.0)
    assert not len(E.orb_trades(E.prepare(df2), ORB2, 1)["day"])


def test_orb_open_beyond_level_fills_at_open_and_short_mirrors():
    df = _orb_days()
    _set(df, 2, open=98.0, high=99.5, low=97.0)
    tr = E.orb_trades(E.prepare(df), ORB2, -1)
    assert tr["entry"].tolist() == [98.0] and tr["gross"][0] == pytest.approx(-2.0 / 100.0)
    assert not len(E.orb_trades(E.prepare(df), ORB2, 1)["day"])


def test_orb_zero_volume_bar_cannot_enter():
    df = _orb_days()
    _set(df, 2, high=102.0, volume=0.0)
    _set(df, 4, high=102.0)
    assert E.orb_trades(E.prepare(df), ORB2, 1)["entry_bar"].tolist() == [4]


def test_orb_day_anchor_skips_the_night_and_days_without_day_session():
    idx = pd.to_datetime(["2021-01-04 21:01", "2021-01-04 21:02", "2021-01-05 09:01",
                          "2021-01-05 09:02", "2021-01-05 09:03", "2021-01-05 21:01",
                          "2021-01-05 21:02", "2021-01-05 21:03"])
    df = pd.DataFrame({"openw": 100.0, "highw": [110, 90, 101, 101, 102, 101, 101, 105.0],
                       "loww": 99.0, "closew": 100.0, "volume": 1.0,
                       "trading_date": pd.to_datetime(["2021-01-05"] * 5 + ["2021-01-06"] * 3)},
                      index=idx)
    p = E.prepare(df)
    day = E.orb_trades(p, E.OrbConfig(2, open_anchor="day"), 1)
    assert day["entry_bar"].tolist() == [4]          # 区间 = 日盘前两根，不含夜盘的 110
    first = E.orb_trades(p, E.OrbConfig(2), 1)
    assert first["entry_bar"].tolist() == [7]        # 含夜盘区间上轨 110，第一天不触发


def _random_minutes(seed: int, days: int = 40, bars: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, price = [], 100.0
    for day in pd.bdate_range("2020-01-02", periods=days):
        for b in range(bars):
            o = price + rng.normal(0, 0.1)
            c = o + rng.normal(0, 0.3)
            hi, lo = max(o, c) + abs(rng.normal(0, 0.2)), min(o, c) - abs(rng.normal(0, 0.2))
            price = c
            rows.append({"timestamp": _bar_time(day, b + 1), "open": o + 50, "openw": o,
                         "highw": hi, "loww": lo, "closew": c,
                         "volume": float(rng.random() > 0.05), "trading_date": day})
    return pd.DataFrame(rows).set_index("timestamp")


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("side", [1, -1])
def test_orb_matches_reference_loop(seed, side):
    df = _random_minutes(seed)
    got = E.orb_trades(E.prepare(df), E.OrbConfig(5), side)
    ref = []
    for d, (_, g) in enumerate(df.groupby("trading_date", sort=True)):
        start = df.index.get_loc(g.index[0])
        level = g["highw"].iloc[:5].max() if side > 0 else g["loww"].iloc[:5].min()
        for j in range(5, len(g)):
            bar = g.iloc[j]
            if bar["volume"] > 0 and side * (bar["highw" if side > 0 else "loww"] - level) > 0:
                entry = max(bar["openw"], level) if side > 0 else min(bar["openw"], level)
                ref.append((d, start + j, entry,
                            side * (g["closew"].iloc[-1] - entry) / bar["open"]))
                break
    assert len(ref) > 5
    assert list(zip(got["day"], got["entry_bar"])) == [r[:2] for r in ref]
    assert np.allclose(got["entry"], [r[2] for r in ref])
    assert np.allclose(got["gross"], [r[3] for r in ref])


def test_tick_per_day_carries_the_last_known_year():
    dates = pd.DatetimeIndex(["2019-06-01", "2020-06-01", "2023-06-01"])
    assert E.tick_per_day({2020: 0.5, 2021: 1.0}, dates).tolist() == [0.5, 0.5, 1.0]
    assert E.tick_per_day(2.0, dates).tolist() == [2.0] * 3
    with pytest.raises(ValueError):
        E.tick_per_day({2020: np.nan}, dates)


# --------------------------------------------------------------------------
# 池级研究：上下文、过滤、面板
# --------------------------------------------------------------------------
def _ctx_minutes(n_days=30, bars=("09:01", "09:30", "10:00", "14:30", "15:00")):
    rows = []
    for d, day in enumerate(pd.bdate_range("2020-01-01", periods=n_days)):
        for b, hm in enumerate(bars):
            px = 100.0 + d + b * 0.1
            rows.append({"time": pd.Timestamp(f"{day.date()} {hm}"), "open": px, "openw": px,
                         "close": px + 0.05, "closew": px + 0.05, "highw": px + 0.5,
                         "loww": px - 0.5, "volume": 10.0 * (d + 1), "trading_date": day})
    return pd.DataFrame(rows).set_index("time")


def test_day_context_atr_ignores_today_and_short_days_have_no_range():
    ctx = P.day_context(_ctx_minutes())
    # ATR 只用前一日及以前：前 20 天都未知
    assert ctx["atr"].iloc[:20].isna().all() and np.isfinite(ctx["atr"].iloc[20])
    # 测试日只有 5 根 bar，不够 15 根开盘区间，不能用后面不存在的 bar 补出来
    assert ctx["or30_atr"].isna().all() and ctx["or30day_atr"].isna().all()


def test_nr4_uses_previous_day_range():
    m = _ctx_minutes(10)
    last = m["trading_date"] == m["trading_date"].iloc[-1]
    m.loc[last, "highw"] += 50           # 当日区间放大不应影响当日 nr4
    ctx = P.day_context(m)
    assert ctx["nr4"].iloc[-1] == ctx["nr4"].iloc[-2] == 1.0


def test_panel_book_and_pool():
    dates = pd.bdate_range("2020-01-01", periods=3)
    panel = P.Panel(dates, ["A", "B"], avail=np.array([[1, 1], [1, 0], [1, 1]], bool),
                    weight=np.array([[1.0, 2.0], [1.0, 0.0], [1.0, 2.0]]),
                    di=np.array([0, 0, 2, 1]), si=np.array([0, 0, 1, 1]))
    value = np.array([0.01, 0.02, 0.03, 0.05])
    book = panel.book(value, np.array([True, True, True, True]))
    assert np.allclose(book[0], [0.03, 0.0])     # 同日同品种两笔相加；B 可交易但没交易记 0
    assert np.isnan(book[1, 1])                  # 不可交易
    assert np.allclose(book[2], [0.0, 0.06])
    member = np.array([[1, 1], [1, 1], [0, 1]], bool)
    ret = panel.pool(book, member)
    assert np.allclose(ret.to_numpy(), [0.015, 0.0, 0.06])


# --------------------------------------------------------------------------
# 手续费：2026 表、历史交易所费率（注入小表）、情景重组、成本上下文
# --------------------------------------------------------------------------
def _fee_table():
    """RB 按比例（2021-01-05 起平今翻倍）；AL 按手、平今交易所收 0。"""
    rows = [("RB", "2020-01-01", "BY_MONEY", 1e-4, 1.5e-4, 3e-4),
            ("RB", "2021-01-05", "BY_MONEY", 1e-4, 1.5e-4, 6e-4),
            ("AL", "2020-01-01", "BY_VOLUME", 3.0, 3.0, 0.0)]
    return pd.DataFrame([{"symbol": s, "trading_date": pd.Timestamp(d), "contract": s + "2101",
                          "commission_type": t, "open_commission": o, "close_commission": c,
                          "close_commission_today": ct} for s, d, t, o, c, ct in rows])


@pytest.fixture
def fee_history():
    costs.set_fee_history(_fee_table())
    yield
    costs.set_fee_history(None)


def test_table_fees_rates_fixed_and_close_today():
    T = costs.FeeModel("2026")
    assert costs.round_trip("RB", 4000.0, T) == pytest.approx(2 * 1.01e-4)
    assert costs.round_trip("AP", 8000.0, T) == pytest.approx((5.05 + 10.0) / (8000 * 10))
    assert costs.round_trip("AL", 20000.0, T) == pytest.approx((3.03 + 0.01) / (20000 * 5))
    assert costs.round_trip("AL", 20000.0, costs.FeeModel("2026", close_today_as_open=True)) == \
        pytest.approx(2 * 3.03 / (20000 * 5))
    # 2026 表没有平昨：隔夜平仓按开仓
    assert costs.round_trip("AL", 20000.0, T, close_today=False) == pytest.approx(2 * 3.03 / (20000 * 5))
    assert costs.round_trip("CU", 60000.0, costs.FeeModel("2026", scale=2.0)) == pytest.approx(2 * 1.51e-4)
    assert all(costs.has_schedule(s) for s in costs.FEES_2026)


def test_hist_fees_broker_markup_legs_and_dates(fee_history):
    H = costs.FeeModel("hist")
    d0, d1 = pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")
    # 按比例 ×1.01；平今 / 平昨分开；按交易日取此前最近一条
    assert costs.round_trip("RB", 4000.0, H, dates=d0) == pytest.approx((1e-4 + 3e-4) * 1.01)
    assert costs.round_trip("RB", 4000.0, H, dates=d1) == pytest.approx((1e-4 + 6e-4) * 1.01)
    assert costs.round_trip("RB", 4000.0, H, close_today=False, dates=d1) == \
        pytest.approx((1e-4 + 1.5e-4) * 1.01)
    # 早于第一条的日期用第一条
    assert costs.round_trip("RB", 4000.0, H, dates=pd.Timestamp("2019-01-02")) == \
        pytest.approx((1e-4 + 3e-4) * 1.01)
    # 按手 +0.01 元；交易所收 0 的腿按 0.01 元
    notional = 20000.0 * 5
    assert costs.round_trip("AL", 20000.0, H, dates=d0) == pytest.approx((3.01 + 0.01) / notional)
    assert costs.round_trip("AL", 20000.0, H, close_today=False, dates=d0) == \
        pytest.approx(2 * 3.01 / notional)
    assert costs.round_trip("AL", 20000.0, costs.FeeModel("hist", close_today_as_open=True),
                            dates=d0) == pytest.approx(2 * 3.01 / notional)
    # 向量输入逐日取费率
    got = costs.round_trip("RB", np.array([4000.0, 4000.0]), H, dates=pd.DatetimeIndex([d0, d1]))
    assert np.allclose(got, [(1e-4 + 3e-4) * 1.01, (1e-4 + 6e-4) * 1.01])
    assert costs.has_schedule("RB", "hist") and not costs.has_schedule("CU", "hist")
    with pytest.raises(ValueError):
        costs.round_trip("RB", 4000.0, H)
    # 候选交易的三列手续费：主口径（平今）、平昨、2026 表
    fee, fee_oo, fee26 = P.trade_fees("RB", np.array([4000.0]), pd.DatetimeIndex([d1]))
    assert np.allclose([fee[0], fee_oo[0], fee26[0]],
                       [(1e-4 + 6e-4) * 1.01, (1e-4 + 1.5e-4) * 1.01, 2 * 1.01e-4])


def test_scenario_net_recombines_cost_parts():
    t = pd.DataFrame({"gross": [0.01], "slip": [0.001], "fee": [0.0002], "fee_oo": [0.0004],
                      "fee26": [0.0001]})
    assert np.isclose(P.scenario_net(t, "gross")[0], 0.01)
    assert np.isclose(P.scenario_net(t, "main")[0], 0.01 - 0.001 - 0.0002)
    assert np.isclose(P.scenario_net(t, "slip2")[0], 0.01 - 0.002 - 0.0002)
    assert np.isclose(P.scenario_net(t, "fee_x2")[0], 0.01 - 0.001 - 0.0004)
    assert np.isclose(P.scenario_net(t, "fee_2026")[0], 0.01 - 0.001 - 0.0001)
    assert np.isclose(P.scenario_net(t, "close_yday")[0], 0.01 - 0.001 - 0.0004)
    assert np.isclose(P.scenario_net(t, "fee_only")[0], 0.01 - 0.0002)


def test_cost_context_uses_hist_close_today_fee(fee_history):
    ctx = P.add_cost_context(P.day_context(_ctx_minutes(30)), "RB", {2020: 1.0})
    day = ctx.index[25]
    raw = ctx.loc[day, "raw_open"]
    assert np.isclose(ctx.loc[day, "cost_rt"], (1e-4 + 3e-4) * 1.01 + 2 * 1.0 / raw)
    assert np.isclose(ctx.loc[day, "cost_atr"], ctx.loc[day, "cost_rt"] / ctx.loc[day, "atr_pct"])


def test_day_session_range_ignores_the_night():
    idx = pd.to_datetime(["2021-01-04 21:01", "2021-01-04 21:02",
                          "2021-01-05 09:01", "2021-01-05 09:02", "2021-01-05 09:03"])
    codes = np.zeros(5, np.int64)
    starts, ends = np.array([0]), np.array([4])
    origin = E.day_open_offset(idx, codes, starts)
    assert origin.tolist() == [2]
    high = np.array([10.0, 1.0, 3.0, 4.0, 2.0])
    low = np.array([9.0, 0.0, 2.0, 3.0, 1.0])
    hi, lo = E.opening_range(high, low, codes, starts, ends, 2, origin)
    assert hi.tolist() == [4.0] and lo.tolist() == [2.0]


def test_opening_range_uses_only_the_first_bars():
    high = np.array([1.0, 5.0, 9.0, 2.0, 4.0, 8.0])
    low = np.array([0.0, 1.0, 3.0, 0.0, 1.0, 6.0])
    codes = np.array([0, 0, 0, 1, 1, 1])
    starts, ends = np.array([0, 3]), np.array([2, 5])
    hi, lo = E.opening_range(high, low, codes, starts, ends, 2)
    assert hi.tolist() == [5.0, 4.0] and lo.tolist() == [0.0, 0.0]
    hi_short, _ = E.opening_range(high, low, codes, starts, ends, 4)
    assert np.isnan(hi_short).all()


def test_signed_features_flip_with_the_trade_and_factors_lag_one_day():
    dates = pd.bdate_range("2020-01-02", periods=3)
    trades = pd.DataFrame({
        "symbol": ["RB", "RB"], "base": ["orb30|eod", "orb30|eod"],
        "side": [1, -1], "date": [dates[1], dates[2]], "entry_pos": [30, 31],
        "open_move_atr": [0.4, 0.4],
    })
    ctx = pd.DataFrame({
        "atr_pct": 0.01, "atr_rel": 1.0, "prev_range_atr": 1.0, "nr4": 0.0,
        "gap_atr": [0.1, 0.2, 0.5], "night": 0.0, "cost_atr": 0.1,
        "or30_atr": [1.0, 1.2, 1.4], "vol30_ratio": [0.8, 0.9, 1.1],
    }, index=dates)
    factor = pd.DataFrame({"RB": [1.0, 2.0, 3.0]}, index=dates)
    names = P.FACTORS
    out = P.attach_features(trades, {"RB": ctx}, {name: factor for name in names})
    # 第二天的交易看到的是前一日收盘的因子，不是当天的值
    assert out["f_tsmom"].tolist() == pytest.approx([1.0, 2.0])
    X = P.ml_matrix(out, "core+trend")
    assert X.loc[0, "gap_atr"] == pytest.approx(0.2)
    assert X.loc[1, "gap_atr"] == pytest.approx(-0.5)
    assert X.loc[1, "f_tsmom"] == pytest.approx(-2.0)


def test_economic_gate_masks_use_signed_trend_and_finite_thresholds():
    trades = pd.DataFrame({
        "side": [1, -1, 1, -1],
        "f_tsmom": [1.0, 1.0, -1.0, -1.0],
        "or_atr": [0.4, 0.6, 0.7, np.nan],
        "open_move_atr": [0.3, 0.5, 0.6, np.nan],
    })
    assert P.economic_gate_mask(trades, "trend_align").tolist() == [True, False, False, True]
    assert P.economic_gate_mask(trades, "range_compress", 0.6).tolist() == [True, True, False, False]
    assert P.economic_gate_mask(trades, "avoid_chase").tolist() == [True, True, False, False]
    with pytest.raises(ValueError):
        P.economic_gate_mask(trades, "range_compress")
    with pytest.raises(ValueError):
        P.economic_gate_mask(trades, "unknown")


def test_scope_predictions_filters_only_short_predictions():
    trades = pd.DataFrame({"side": [-1, 1, -1, 1]})
    predictions = np.array([0.4, -0.2, -0.1, 0.8])
    assert P.scope_predictions(predictions, trades).tolist() == predictions.tolist()
    assert np.allclose(P.scope_predictions(predictions, trades, "long_only")[[1, 3]], [-0.2, 0.8])
    assert np.isnan(P.scope_predictions(predictions, trades, "long_only")[[0, 2]]).all()
    with pytest.raises(ValueError):
        P.scope_predictions(predictions[:-1], trades, "long_only")
    with pytest.raises(ValueError):
        P.scope_predictions(predictions, trades, "short_only")


# --------------------------------------------------------------------------
# 多策略清单：通用撮合、多日记账、小时线、池级序列与选格
# --------------------------------------------------------------------------
def test_touch_trades_uses_per_day_levels_and_skips_negative_start():
    df = _orb_days(bars=6, days=3)
    _set(df, 3, high=103.0)                       # 第 1 天第 4 根摸到 102
    _set(df, 6 + 2, low=97.0)                     # 第 2 天第 3 根摸到 98（做多不看）
    _set(df, 12 + 4, high=103.0)                  # 第 3 天第 5 根也摸到，但 start = -1 不做
    p = E.prepare(df)
    got = E.touch_trades(p, np.array([102.0, 102.0, 102.0]), np.array([1, 1, -1]), 1)
    assert got["day"].tolist() == [0] and got["entry_bar"].tolist() == [3]
    assert got["entry"][0] == 102.0 and got["exit_bar"][0] == 5
    assert np.isclose(got["gross"][0], (100.0 - 102.0) / 100.0)


def test_next_open_trades_enters_on_the_next_bar_and_ignores_the_last_bar():
    df = _orb_days(bars=4, days=2)
    _set(df, 2, open=105.0)
    p = E.prepare(df)
    sig = np.zeros(len(df), bool)
    sig[[1, 2, 7]] = True                         # 第 1 天第 2、3 根；第 2 天最后一根（无下一根）
    got = E.next_open_trades(p, sig, -1)
    assert got["day"].tolist() == [0] and got["entry_bar"].tolist() == [2]
    assert np.isclose(got["gross"][0], -(100.0 - 105.0) / 100.0)


def test_position_returns_charges_turnover_and_books_gaps_to_the_old_position():
    o = np.array([100.0, 101.0, 103.0, 102.0])
    c = np.array([101.0, 102.0, 102.0, 104.0])
    codes = np.array([0, 0, 1, 1])
    sig = np.array([1.0, np.nan, -1.0, np.nan])   # 第 1 根收盘做多，第 3 根收盘翻空
    r = E.position_returns(o, c, o, codes, 2, sig, np.array([1.0, 2.0]),
                           {"slip": np.array([0.01, 0.02])})
    # bar1 多 1：(102-101)/101；bar2 多 2：跳空 1×(103-102) + 2×(102-103)；bar3 空 2：跳空 2×0 + (-2)×2
    assert np.allclose(r["gross"], [1 / 101, (1 - 2) / 103 - 4 / 102])
    # 换手：bar1 0→1，bar2 1→2（权重变），bar3 2→-2
    assert np.allclose(r["slip"], [0.01, 1 * 0.02 + 4 * 0.02])
    assert r["n"].tolist() == [1.0, 1.0]


def test_hourly_bars_split_on_clock_hours_and_trading_days():
    rows = []
    for day, stamps in (("2021-01-04", ["21:01", "21:59", "22:00", "22:01"]),
                        ("2021-01-05", ["09:01", "10:00"])):
        for k, hm in enumerate(stamps):
            ts = pd.Timestamp(f"{day} {hm}") - pd.Timedelta(int(hm >= "21"), unit="D")
            px = 100.0 + k
            rows.append({"timestamp": ts, "open": px, "openw": px, "highw": px + 1,
                         "loww": px - 1, "closew": px + 0.5, "volume": 1.0,
                         "trading_date": pd.Timestamp(day)})
    h = P.hourly_bars(E.prepare(pd.DataFrame(rows).set_index("timestamp")))
    # 21:01..22:00 是 21 点那根；22:01 是 22 点；次日 09:01..10:00 一根
    assert h["codes"].tolist() == [0, 0, 1]
    assert h["o"].tolist() == [100.0, 103.0, 100.0] and h["c"].tolist() == [102.5, 103.5, 101.5]


def test_slate_series_pools_weighted_legs_and_select_variant_uses_train_years():
    dates = pd.bdate_range("2019-12-30", periods=4)
    panel = P.Panel(dates, ["A", "B"], avail=np.ones((4, 2), bool),
                    weight=np.full((4, 2), 9.0), di=np.array([], int), si=np.array([], int))
    legs = pd.DataFrame({
        "symbol": ["A", "B", "A", "A"], "strategy": "s", "variant": ["x", "x", "x", "y"],
        "date": dates[[0, 0, 2, 2]], "gross": [0.02, 0.04, 0.01, -0.01],
        "slip": [0.01, 0.0, 0.0, 0.0], "fee": 0.0, "fee_oo": 0.0, "fee26": 0.0, "n": 1.0})
    out = P.slate_series(legs, panel, np.ones((4, 2), bool))
    # 已乘权重，不再乘面板权重 9
    assert np.allclose(out[("s", "x", "main")].to_numpy(), [0.025, 0.0, 0.005, 0.0])
    assert np.allclose(out[("s", "y", "main")].to_numpy(), [0.0, 0.0, -0.005, 0.0])
    s = pd.Series([0.01, -0.01, 0.02, 0.03], index=pd.bdate_range("2019-12-30", periods=4))
    good = pd.Series([0.0, 0.0, 0.01, 0.012], index=s.index)
    assert P.select_variant({"a": s, "b": good}, [2020]) == "b"
    assert P.select_variant({"a": s * 0, "b": s * 0}, [2020]) == "a"


def test_slate_legs_cover_every_book_without_lookahead_columns(fee_history):
    df = _random_minutes(3, days=60, bars=40)
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    si = P.slate_input("RB", df, 1.0, carry=pd.Series(0.1, index=df["trading_date"].unique()))
    legs = P.slate_legs(si, start="2020-01-01")
    assert list(legs.columns) == P.LEG_COLUMNS
    assert set(legs["strategy"]) <= set(P.SLATE)
    assert np.isfinite(legs[P.SLATE_NUMERIC].to_numpy(np.float64)).all()
    assert (legs["slip"] >= 0).all() and (legs["fee"] >= 0).all()
    # 前 20 天 ATR 未知，权重为 NaN：任何书都不能在那里记账
    first = pd.DatetimeIndex(si.p.dates[np.isfinite(si.weight)]).min()
    assert pd.DatetimeIndex(legs["date"]).min() >= first
