"""日内模块测试：撮合与信息时点、向量化撮合对照逐 bar 参考循环、手续费口径、
开盘区间与元标签不使用未来信息。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfcta.research.intraday import BacktestConfig, StrategyConfig, run_intraday
from tfcta.research.backtest import costs
from tfcta.research.intraday import engine as E
from tfcta.research.intraday import walk_forward as P


def _bar_time(day, minute: int) -> pd.Timestamp:
    """当天 09:00 之后的第 minute 分钟。

    ``Timedelta(hours=, minutes=)`` 会转成没有单位的 NumPy timedelta，新版本会报警。
    """
    return pd.Timestamp(day) + pd.Timedelta(9 * 60 + int(minute), unit="min")


def _minutes(days: int = 3) -> pd.DataFrame:
    rows = []
    for day in pd.bdate_range("2021-01-04", periods=days):
        for minute in range(5):
            value = 100.0
            rows.append({
                "timestamp": _bar_time(day, minute),
                "openw": value, "highw": value, "loww": value,
                "closew": value, "trading_date": day,
            })
    return pd.DataFrame(rows).set_index("timestamp")


def test_requires_completed_previous_day_for_fali():
    df = _minutes(2)
    df.loc[df.index[5], "highw"] = 105.0
    result = run_intraday(df, StrategyConfig(name="fali"),
                          BacktestConfig(margin_rate=0.3), "RB")
    assert result.trades.iloc[0]["entry_time"].date() == pd.Timestamp("2021-01-05").date()


def test_same_bar_stop_loss_wins_over_target():
    df = _minutes(2)
    df.loc[df.index[5], "highw"] = 105.0
    df.loc[df.index[6], ["openw", "highw", "loww", "closew"]] = [105.0, 110.0, 90.0, 100.0]
    result = run_intraday(
        df, StrategyConfig(name="fali", atr_window=1,
                           stop_atr_multiple=1, target_atr_multiple=1),
        BacktestConfig(margin_rate=0.3), "RB")
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "stop_loss"


def test_every_position_is_closed_at_end_of_day():
    df = _minutes(2)
    df.loc[df.index[5], "highw"] = 105.0
    result = run_intraday(df, StrategyConfig(name="fali"),
                          BacktestConfig(margin_rate=0.3), "RB")
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "end_of_day"
    assert result.trades.iloc[0]["exit_time"].date() == pd.Timestamp("2021-01-05").date()


def test_margin_rate_scales_daily_return():
    df = _minutes(2)
    df.loc[df.index[5], "highw"] = 105.0
    df.loc[df.index[6], ["openw", "highw", "loww", "closew"]] = [105.0, 105.0, 105.0, 110.0]
    one = run_intraday(df, StrategyConfig(name="fali"),
                       BacktestConfig(fee_rate=0, margin_rate=1), "RB")
    half = run_intraday(df, StrategyConfig(name="fali"),
                        BacktestConfig(fee_rate=0, margin_rate=0.5), "RB")
    assert half.daily.iloc[-1]["net_return"] == pytest.approx(
        2 * one.daily.iloc[-1]["net_return"])


def test_train_years_stop_before_the_predicted_year():
    assert P.train_years(2019) == [2016, 2017, 2018]
    assert P.train_years(2017) == [2016]
    assert P.train_years(2016) == []
    assert 2022 not in P.train_years(2022)


# --------------------------------------------------------------------------
# 向量化撮合与保守成交口径
# --------------------------------------------------------------------------
def _reference(df, long_level, short_level, stop_dist, target_dist, max_entries, start=0):
    """逐 bar 参考实现，口径与 engine.simulate 的文档一致。"""
    out = []
    for d, (_, day) in enumerate(df.groupby("trading_date", sort=True)):
        o, h, l, c = (day[k].to_numpy() for k in ("openw", "highw", "loww", "closew"))
        vol = day["volume"].to_numpy()
        up, dn = long_level[d], short_level[d]
        i, entries = start, 0
        while i < len(day) and entries < max_entries:
            side = 0
            for j in range(i, len(day)):
                if vol[j] <= 0:
                    continue
                hl, hs = h[j] > up, l[j] < dn
                if hl and hs:
                    ol, os_ = o[j] >= up, o[j] <= dn
                    if ol == os_:
                        side = None
                        break
                    side = 1 if ol else -1
                elif hl:
                    side = 1
                elif hs:
                    side = -1
                if side:
                    break
            if not side:
                break
            entries += 1
            px = max(o[j], up) if side > 0 else min(o[j], dn)
            stop, target = px - side * stop_dist[d], px + side * target_dist[d]
            exit_bar, exit_px, reason = len(day) - 1, c[-1], "end_of_day"
            for k in range(j, len(day)):
                if (side > 0 and l[k] <= stop) or (side < 0 and h[k] >= stop):
                    exit_bar, reason = k, "stop_loss"
                    exit_px = stop if k == j else (min(o[k], stop) if side > 0 else max(o[k], stop))
                    break
                if k > j and ((side > 0 and h[k] >= target) or (side < 0 and l[k] <= target)):
                    exit_bar, reason = k, "take_profit"
                    exit_px = max(o[k], target) if side > 0 else min(o[k], target)
                    break
            out.append((day.index[j], day.index[exit_bar], side, px, exit_px, reason))
            if reason == "end_of_day":
                break
            i = exit_bar + 1
    return out


def _random_minutes(seed: int, days: int = 60, bars: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, price = [], 100.0
    for day in pd.bdate_range("2020-01-02", periods=days):
        for b in range(bars):
            o = price + rng.normal(0, 0.1)
            c = o + rng.normal(0, 0.3)
            hi, lo = max(o, c) + abs(rng.normal(0, 0.2)), min(o, c) - abs(rng.normal(0, 0.2))
            price = c
            rows.append({"timestamp": _bar_time(day, b + 1),
                         "open": o, "openw": o, "highw": hi, "loww": lo, "closew": c,
                         "volume": float(rng.random() > 0.05), "trading_date": day})
    return pd.DataFrame(rows).set_index("timestamp")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("max_entries", [1, 3])
def test_vectorized_engine_matches_reference_loop(seed, max_entries):
    df = _random_minutes(seed)
    cfg = StrategyConfig(name="dual_thrust", lookback=2, k1=0.3, k2=0.3, atr_window=3,
                         stop_atr_multiple=0.3, target_atr_multiple=0.5,
                         max_entries_per_day=max_entries)
    result = run_intraday(df, cfg, BacktestConfig(fee_rate=0, margin_rate=1), "T")
    from tfcta.research.intraday.strategies import prepared_daily_levels
    levels = prepared_daily_levels(
        df.assign(open=df["openw"], high=df["highw"], low=df["loww"], close=df["closew"]), cfg)
    ref = _reference(df, levels["long_level"].to_numpy(), levels["short_level"].to_numpy(),
                     levels["atr"].to_numpy() * 0.3, levels["atr"].to_numpy() * 0.5, max_entries)
    got = list(result.trades[["entry_time", "exit_time", "direction", "entry_price",
                              "exit_price", "exit_reason"]].itertuples(index=False, name=None))
    assert len(got) == len(ref) > 0
    for a, b in zip(got, ref):
        assert a[:3] == b[:3] and a[5] == b[5]
        assert a[3] == pytest.approx(b[3]) and a[4] == pytest.approx(b[4])


def _two_days(bars=4):
    rows = []
    for day in pd.bdate_range("2021-01-04", periods=2):
        for b in range(bars):
            rows.append({"timestamp": _bar_time(day, b + 1),
                         "openw": 100.0, "highw": 101.0, "loww": 99.0, "closew": 100.0,
                         "volume": 1.0, "trading_date": day})
    return pd.DataFrame(rows).set_index("timestamp")


FALI = StrategyConfig(name="fali", atr_window=1, stop_atr_multiple=0.5)


def test_bar_crossing_both_levels_skips_the_day():
    df = _two_days()
    df.iloc[5, df.columns.get_indexer(["highw", "loww"])] = [102.0, 98.0]
    assert run_intraday(df, FALI, BacktestConfig(margin_rate=1)).trades.empty


def test_open_beyond_one_level_resolves_the_order():
    df = _two_days()
    df.iloc[5, df.columns.get_indexer(["openw", "highw", "loww"])] = [101.5, 102.0, 98.0]
    trades = run_intraday(df, FALI, BacktestConfig(margin_rate=1)).trades
    assert trades.iloc[0]["direction"] == 1 and trades.iloc[0]["entry_price"] == 101.5


def test_entry_bar_stop_fills_at_stop_not_open():
    df = _two_days()
    # ATR=2，止损距离 1：入场 101 后同一根 bar 跌到 99.5，按 100 止损而不是开盘 100.5
    df.iloc[5, df.columns.get_indexer(["openw", "highw", "loww"])] = [100.5, 102.0, 99.5]
    trade = run_intraday(df, FALI, BacktestConfig(margin_rate=1)).trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["entry_price"] == 101.0 and trade["exit_price"] == 100.0


def test_zero_volume_bar_cannot_open_a_position():
    df = _two_days()
    df.iloc[5, df.columns.get_indexer(["highw", "volume"])] = [102.0, 0.0]
    assert run_intraday(df, FALI, BacktestConfig(margin_rate=1)).trades.empty


def test_reentry_after_stop_on_later_bar():
    df = _two_days(bars=6)
    df.iloc[6:, df.columns.get_indexer(["openw", "highw", "loww", "closew"])] = [100.6, 100.8, 100.4, 100.6]
    df.iloc[7, df.columns.get_indexer(["highw"])] = [102.0]     # 入场 101
    df.iloc[8, df.columns.get_indexer(["loww"])] = [99.8]       # 止损 100
    df.iloc[10, df.columns.get_indexer(["highw"])] = [102.0]    # 再次入场
    cfg = StrategyConfig(name="fali", atr_window=1, stop_atr_multiple=0.5, max_entries_per_day=2)
    trades = run_intraday(df, cfg, BacktestConfig(margin_rate=1)).trades
    assert list(trades["exit_reason"]) == ["stop_loss", "end_of_day"]
    one = run_intraday(df, FALI, BacktestConfig(margin_rate=1)).trades
    assert len(one) == 1


def test_slippage_ticks_by_year():
    df = _two_days()
    df.iloc[5, df.columns.get_indexer(["highw"])] = [102.0]
    cfg = StrategyConfig(name="fali")
    zero = run_intraday(df, cfg, BacktestConfig(fee_rate=0, margin_rate=1)).trades.iloc[0]
    slip = run_intraday(df, cfg, BacktestConfig(fee_rate=0, margin_rate=1, slippage_ticks=1),
                        tick={2021: 0.5}).trades.iloc[0]
    assert slip["entry_price"] == zero["entry_price"] + 0.5
    assert slip["exit_price"] == zero["exit_price"] - 0.5


# --------------------------------------------------------------------------
# 带反手的撮合（E.simulate_path）
# --------------------------------------------------------------------------
def _find_entry(p, lv, j, e, al, ash):
    o, h, l, vol = p.o, p.h, p.l, p.tradable
    EL, ES = lv["enter_long"], lv["enter_short"]
    for k in range(j, e + 1):
        if not vol[k]:
            continue
        hl, hs = al and h[k] > EL[k], ash and l[k] < ES[k]
        if hl and hs:
            ol, osh = o[k] >= EL[k], o[k] <= ES[k]
            return k, (None if ol == osh else (1 if ol else -1))
        if hl or hs:
            return k, (1 if hl else -1)
    return e + 1, None


def _first_exit(p, lv, d, eb, side, px):
    o, h, l, c, vol = p.o, p.h, p.l, p.c, p.tradable
    RS, RL, e = lv["rev_to_short"], lv["rev_to_long"], p.ends[d]
    stop, target = px - side * lv["stop_dist"][d], px + side * lv["target_dist"][d]
    for k in range(eb, e + 1):
        sh = (l[k] <= stop) if side > 0 else (h[k] >= stop)
        th = k > eb and ((h[k] >= target) if side > 0 else (l[k] <= target))
        rh = k > eb and vol[k] and ((l[k] < RS[k]) if side > 0 else (h[k] > RL[k]))
        spx = stop if k == eb else (min(o[k], stop) if side > 0 else max(o[k], stop))
        rpx = min(o[k], RS[k]) if side > 0 else max(o[k], RL[k])
        if sh or rh:
            if rh and (not sh or side * (rpx - spx) >= 0):
                return k, rpx, "reverse"
            return k, spx, "stop_loss"
        if th:
            return k, (max(o[k], target) if side > 0 else min(o[k], target)), "take_profit"
    return e, c[e], "end_of_day"


def _reference_path(p, lv, allow_long, allow_short, max_legs):
    """逐 bar 参考实现，口径与 E.simulate_path 的文档一致。"""
    out = []
    for d, (s, e) in enumerate(zip(p.starts, p.ends)):
        al, ash = allow_long[d], allow_short[d]
        j, legs, side = s + lv["start_bar"][d], 0, 0
        while j <= e:
            if side == 0:
                if legs >= max_legs:
                    break
                eb, side = _find_entry(p, lv, j, e, al, ash)
                if side is None:
                    break
                legs += 1
                px = max(p.o[eb], lv["enter_long"][eb]) if side > 0 else \
                    min(p.o[eb], lv["enter_short"][eb])
            xi, xpx, reason = _first_exit(p, lv, d, eb, side, px)
            out.append((d, eb, xi, side, px, xpx, reason))
            if reason == "end_of_day":
                break
            if reason == "reverse" and xi < e and legs < max_legs and (ash if side > 0 else al):
                side, eb, px, legs = -side, xi, xpx, legs + 1
                continue
            side, j = 0, xi + 1
    return out


def _compare_path(p, lv, allow_long, allow_short, max_legs):
    sim = E.simulate_path(p.o, p.h, p.l, p.c, p.codes, p.starts, p.ends,
                             lv["enter_long"], lv["enter_short"], lv["rev_to_short"],
                             lv["rev_to_long"], lv["stop_dist"], lv["target_dist"],
                             lv["start_bar"], allow_long, allow_short, p.tradable, max_legs)
    ref = _reference_path(p, lv, allow_long, allow_short, max_legs)
    got = list(zip(sim["day"], sim["entry_bar"], sim["exit_bar"], sim["direction"],
                   sim["entry"], sim["exit"], sim["reason"]))
    assert len(got) == len(ref) > 0
    for a, b in zip(got, ref):
        assert tuple(int(x) for x in a[:4]) == b[:4] and a[6] == b[6]
        assert a[4] == pytest.approx(b[4]) and a[5] == pytest.approx(b[5])
    return ref


PATH_STRATEGIES = [
    StrategyConfig("dual_thrust", lookback=2, k1=0.2, k2=0.2, atr_window=3, reverse=True),
    StrategyConfig("dual_thrust", lookback=2, k1=0.2, k2=0.3, atr_window=3, reverse=True,
                   stop_atr_multiple=0.4, target_atr_multiple=0.8),
    StrategyConfig("rbreaker", atr_window=3, rbreaker_setup=0.1, rbreaker_break=0.1),
    StrategyConfig("rbreaker", atr_window=3, rbreaker_setup=0.1, stop_pct=0.004),
    StrategyConfig("fali", atr_window=3, reverse=True, target_pct=0.01),
]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("max_legs", [1, 2, 5])
@pytest.mark.parametrize("strategy", PATH_STRATEGIES)
def test_simulate_path_matches_reference(seed, max_legs, strategy):
    from tfcta.research.intraday.engine import bar_levels, prepare
    p = prepare(_random_minutes(seed, days=40, bars=60))
    lv = bar_levels(p, strategy)
    rng = np.random.default_rng(seed + 100)
    n = len(p.starts)
    everything = np.ones(n, bool)
    _compare_path(p, lv, everything, everything, max_legs)
    _compare_path(p, lv, rng.random(n) > 0.3, rng.random(n) > 0.3, max_legs)


def test_reverse_flips_and_rbreaker_gates_use_prior_bars():
    from tfcta.research.intraday.engine import bar_levels, prepare
    from tfcta.research.intraday.strategies import prepared_daily_levels
    p = prepare(_random_minutes(5, days=40, bars=60))
    cfg = StrategyConfig("rbreaker", atr_window=3, rbreaker_setup=0.1, rbreaker_break=0.1)
    lv = bar_levels(p, cfg)
    everything = np.ones(len(p.starts), bool)
    ref = _compare_path(p, lv, everything, everything, 5)
    assert any(r[6] == "reverse" for r in ref)
    lvl = prepared_daily_levels(p.df, cfg).reindex(p.dates)
    for d, (s, e) in enumerate(zip(p.starts, p.ends)):
        for k in range(s, e + 1):
            prior = p.h[s:k].max() if k > s else -np.inf
            expect = lvl["sell_enter"].iloc[d] if prior > lvl["sell_setup"].iloc[d] else np.nan
            got = lv["rev_to_short"][k]
            assert (np.isnan(expect) and np.isnan(got)) or got == pytest.approx(expect)


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
# 手续费：2026 表、历史交易所费率（注入小表）、情景重组、成本上下文、持有书
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
    df = _two_days()
    df["open"] = df["openw"]
    df.iloc[5, df.columns.get_indexer(["highw"])] = [102.0]
    trade = run_intraday(df, StrategyConfig("fali"),
                         BacktestConfig(margin_rate=1, fee_schedule="table"), "RB").trades.iloc[0]
    assert trade["cost"] == pytest.approx(2 * 1.01e-4)


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
    df = _two_days()
    df["open"] = df["openw"]
    df.iloc[5, df.columns.get_indexer(["highw"])] = [102.0]
    trade = run_intraday(df, StrategyConfig("fali"),
                         BacktestConfig(margin_rate=1, fee_schedule="hist"), "RB").trades.iloc[0]
    assert trade["cost"] == pytest.approx((1e-4 + 6e-4) * 1.01)


def test_fee_wide_uses_prior_year_price_and_fallback(fee_history):
    tick_table = pd.DataFrame({"symbol": ["AL", "AL"], "year": [2020, 2021],
                               "median_close": [10000.0, 20000.0]})
    index = pd.DatetimeIndex(["2020-06-01", "2021-06-01"])
    wide = costs.fee_wide(index, ["AL", "ZZ"], costs.FeeModel("hist"), tick_table, fallback=2e-4)
    # 第一年没有上一年，用当年；2021 用 2020 的价位；单边 = (开仓 + 平昨) / 2
    assert np.allclose(wide["AL"], [3.01 / (10000 * 5), 3.01 / (10000 * 5)])
    assert np.allclose(wide["ZZ"], 2e-4)
    with pytest.raises(KeyError):
        costs.fee_wide(index, ["ZZ"], costs.FeeModel("hist"), tick_table)


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
    names = ("tsmom", "tsmom_20", "cs_mom_ra_250", "er", "vol_ratio")
    out = P.attach_features(trades, {"RB": ctx}, {name: factor for name in names})
    # 第二天的交易看到的是前一日收盘的因子，不是当天的值
    assert out["f_tsmom"].tolist() == pytest.approx([1.0, 2.0])
    X = P.ml_matrix(out, "core+trend")
    assert X.loc[0, "gap_atr"] == pytest.approx(0.2)
    assert X.loc[1, "gap_atr"] == pytest.approx(-0.5)
    assert X.loc[1, "f_tsmom"] == pytest.approx(-2.0)
