import numpy as np
import pandas as pd

from research.index_price_extremes import aggregate_daily, daily_features


def _synthetic_minutes(days=120):
    rows = []
    dates = pd.bdate_range("2010-01-04", periods=days)
    for i, date in enumerate(dates):
        open_price = 4000 + i * 0.6
        offset = (i // 20) * 50.0
        for minute, volume in enumerate((10.0, 20.0)):
            close = open_price + (minute + 1) * 0.3
            high = close + 2.0
            low = open_price - 1.5
            rows.append(
                {
                    "trading_date": date,
                    "datetime": date.replace(hour=9, minute=31 + minute),
                    "open": open_price if minute == 0 else close - 0.2,
                    "close": close,
                    "openw": (open_price if minute == 0 else close - 0.2) + offset,
                    "closew": close + offset,
                    "highw": high + offset,
                    "loww": low + offset,
                    "volume": volume,
                    "total_turnover": close * volume * 300,
                }
            )
    return pd.DataFrame(rows)


def test_twap_uses_minute_typical_price_in_adjusted_coordinates():
    minutes = _synthetic_minutes()
    daily = aggregate_daily(minutes, "IF")
    first = daily.iloc[0]
    first_date = minutes["trading_date"].iloc[0]
    same_day = minutes.loc[minutes["trading_date"] == first_date]
    expected_twap = ((same_day["highw"] + same_day["loww"] + same_day["closew"]) / 3).mean()
    assert np.isclose(first["twapw"], expected_twap)


def test_price_extreme_features_use_only_information_available_at_close():
    minutes = _synthetic_minutes()
    original = daily_features(minutes, "IF")
    signal_date = original.index[90]

    changed = minutes.copy()
    future = changed["trading_date"] > signal_date
    for column in ("open", "close", "openw", "closew", "highw", "loww"):
        changed.loc[future, column] *= 1.5
    changed.loc[future, "total_turnover"] *= 1.5
    revised = daily_features(changed, "IF")

    pd.testing.assert_series_equal(
        original.loc[signal_date, ["clv", "high_open", "low_open", "range_open",
                                   "high_twap", "low_twap", "range_twap"]],
        revised.loc[signal_date, ["clv", "high_open", "low_open", "range_open",
                                  "high_twap", "low_twap", "range_twap"]],
    )
    assert original.at[signal_date, "outcome"] != revised.at[signal_date, "outcome"]


def test_weak_close_shape_features_use_only_the_signal_day():
    from research.index_price_extremes import shape_from_minutes

    rows = []
    dates = pd.bdate_range("2018-01-02", periods=3)
    # 第一天：开 100，涨到 110（第 2 根是首个最高点），再跌到 90 收；第二、三天平稳
    paths = [[(100, 104), (104, 110), (110, 110), (110, 95), (95, 90)],
             [(90, 91), (91, 92), (92, 93), (93, 94), (94, 95)],
             [(95, 96), (96, 97), (97, 98), (98, 99), (99, 100)]]
    for date, path in zip(dates, paths):
        for k, (start, end) in enumerate(path):
            rows.append({"trading_date": date, "datetime": date + pd.Timedelta(minutes=571 + k),
                         "open": start, "close": end, "openw": start + 1000, "closew": end + 1000,
                         "highw": max(start, end) + 1000, "loww": min(start, end) + 1000})
    out = shape_from_minutes(pd.DataFrame(rows), "IF")
    first = out.iloc[0]
    assert first["clv"] == 0.0
    assert first["open_location"] == 0.5                         # 开盘 100 在 90–110 的正中
    assert first["high_time"] == 0.25                            # 5 根里第 2 根，位置 1/4
    assert first["info_discreteness"] == -1 * (2 / 5 - 2 / 5)    # 下跌日：下跌分钟 2，上涨分钟 2
    assert first["outcome"] == (95 - 90) / 90                    # 第二天开盘到第三天开盘
    changed = pd.DataFrame(rows)
    changed.loc[changed["trading_date"] == dates[2], ["closew", "highw"]] += 50
    assert shape_from_minutes(changed, "IF").iloc[0][["clv", "open_location", "high_time"]].equals(
        first[["clv", "open_location", "high_time"]])
