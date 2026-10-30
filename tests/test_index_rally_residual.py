import numpy as np
import pandas as pd

from research.if_before_ih_ic import _opening_row, call_auction_volume
from research.index_execution_10m import auction_allowance
from research.index_rally_residual import opening_quotes
from strategies.index_bear_short_rally_residual import RESIDUAL_MINIMUM, day_features, long_weight, rally_residual


def test_residual_uses_only_earlier_weak_closes_and_signal_day_bars():
    days = pd.bdate_range("2018-01-01", periods=260)
    rng = np.random.default_rng(1)
    opens = pd.DataFrame(100.0, index=days, columns=["IF"])
    highs = opens + rng.uniform(0.2, 2.0, size=(260, 1))
    closes = opens + rng.normal(-0.5, 1.0, size=(260, 1))
    bars = {"open": opens, "openw": opens, "highw": highs, "closew": closes}
    clv = pd.Series(rng.uniform(0, 1, 260), index=days)
    base = rally_residual(day_features(bars, ("IF",)), clv)
    changed = {**bars, "highw": highs.copy()}
    changed["highw"].iloc[200:] += 5.0
    assert base.iloc[:200].equals(rally_residual(day_features(changed, ("IF",)), clv).iloc[:200])
    assert base[clv.ge(0.4)].isna().all()
    weak_with_features = clv.lt(0.4) & day_features(bars, ("IF",)).notna().all(axis=1)
    assert base[weak_with_features].isna().sum() == RESIDUAL_MINIMUM
    weight = long_weight(base)
    assert weight[base.isna()].eq(1.0).all() and weight[base.le(0)].eq(0.0).all()


def test_opening_quotes_skip_the_auction_and_stop_after_thirty_minutes():
    day = pd.Timestamp("2016-03-01")
    times = day + pd.to_timedelta(["09:29:00", "09:30:00.5", "09:45:00", "09:59:59", "10:00:00"])
    ticks = pd.DataFrame({
        "datetime": times, "trading_date": day, "open": 100.0, "volume": [10.0, 12, 20, 30, 99],
        "a1": [101.0, 100.4, 100.8, 0.0, 200.0], "b1": [99.0, 100.0, 100.2, 100.0, 1.0], "contract": "IF1603",
    })
    row = opening_quotes(ticks).iloc[0]
    assert row["auction_volume"] == 10.0 and row["volume"] == 20.0
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
            stamped + pd.Timedelta(hours=9, minutes=15),
            stamped + pd.Timedelta(hours=9, minutes=15, seconds=1),
        ],
        "volume": [418.0, 500.0],
    })
    assert call_auction_volume(at_open, stamped) == 418.0
    # 起点之前已经有一笔：竞价量用那一笔，不用正好落在 09:15 的连续成交
    earlier = pd.Timestamp("2013-06-03")
    both = pd.DataFrame({
        "datetime": [
            earlier + pd.Timedelta(hours=9, minutes=14),
            earlier + pd.Timedelta(hours=9, minutes=15),
        ],
        "volume": [400.0, 722.0],
    })
    assert call_auction_volume(both, earlier) == 400.0


def test_ticks_that_start_after_the_open_are_a_gap():
    day = pd.Timestamp("2011-06-17")
    ticks = pd.DataFrame({
        "datetime": [day + pd.Timedelta(hours=9, minutes=33), day + pd.Timedelta(hours=9, minutes=34)],
        "trading_date": day,
        "open": 3000.0,
        "volume": [6299.0, 6400.0],
        "a1": [3001.0, 3001.2],
        "b1": [3000.6, 3000.8],
    })
    row = _opening_row(ticks, day, "IF1107")
    assert row["open_gap"] is True and row["auction_volume"] == 0.0 and row["quotes"] == 0
