import json
from pathlib import Path
import pandas as pd
import pytest
from tfcta import config as C
from research.time_factor_screen import fair_prices, load_config, prepare, rows_for_symbol, screen_row, window_clocks

def _config():
    return load_config()

def _frame(rows):
    frame = pd.DataFrame(rows).set_index("stamp")
    frame.index.name = None
    return frame

def _bar(stamp, trading_date, price, high=None):
    high = price if high is None else high
    return {
        "stamp": pd.Timestamp(stamp), "trading_date": pd.Timestamp(trading_date),
        "open": price, "openw": price, "close": price, "closew": price,
        "highw": high, "loww": price - 1, "volume": 10.0, "total_turnover": price * 10,
    }

def _session(trading_date, night_open, morning_open, later_high):
    previous = pd.Timestamp(trading_date) - pd.Timedelta(days=1)
    return [
        _bar(f"{previous:%Y-%m-%d} 21:01", trading_date, night_open),
        _bar(f"{trading_date} 09:01", trading_date, morning_open),
        _bar(f"{trading_date} 09:30", trading_date, morning_open, high=morning_open + 1),
        _bar(f"{trading_date} 10:00", trading_date, morning_open),
        _bar(f"{trading_date} 10:01", trading_date, morning_open),
        _bar(f"{trading_date} 11:30", trading_date, morning_open),
        _bar(f"{trading_date} 13:31", trading_date, morning_open, high=later_high),
        _bar(f"{trading_date} 15:00", trading_date, morning_open),
    ]

def test_ten_oclock_clock_ignores_the_afternoon_high():
    rows = []
    for offset, day in enumerate(("2021-01-04", "2021-01-05", "2021-01-06")):
        rows.extend(_session(day, 100 + offset, 110 + offset, 1000 + offset))
    built = pd.DataFrame(rows_for_symbol(_frame(rows), "RB", "commodity", _config()))
    morning = built[(built.decision == "10:00") & (built.factor == "ts_high") & (built.trading_date == "2021-01-04")]
    close = built[(built.decision == "close") & (built.factor == "ts_high") & (built.trading_date == "2021-01-04")]
    # 夜盘、09:01、09:30、10:00 四根里，高点在第三根。
    assert morning.value.iloc[0] == pytest.approx(2 / 3)
    assert close.value.iloc[0] > morning.value.iloc[0]

def test_night_product_next_session_uses_the_next_night_open():
    rows = []
    for day, night, morning in (
        ("2021-01-04", 100, 110), ("2021-01-05", 200, 210), ("2021-01-06", 300, 310),
    ):
        rows.extend(_session(day, night, morning, morning + 5))
    built = pd.DataFrame(rows_for_symbol(_frame(rows), "RB", "commodity", _config()))
    trade = built[(built.decision == "close") & (built.trading_date == "2021-01-04")].iloc[0]
    assert trade.gross == pytest.approx((300 - 200) / 200)

def test_fair_price_uses_only_previous_days_threshold():
    def day(trading_date, prices):
        stamps = pd.date_range(f"{trading_date} 09:01", periods=len(prices), freq="min")
        return [_bar(stamp, trading_date, price) for stamp, price in zip(stamps, prices)]
    frame = _frame(day("2021-01-04", [100, 102, 104, 106]) + day("2021-01-05", [100, 101, 102, 100]))
    fair, _ = fair_prices(prepare(frame), lookback=1, percentile=50, fixed=None)
    # 前一日变动是 2，当天最后一根才回到 100。最长持续期落在 102，而不是收盘。
    assert fair.loc[pd.Timestamp("2021-01-05"), "dfp_max"] == pytest.approx((102 - 100) / 100)

def test_low_minus_high_is_positive_when_the_low_is_later():
    part = pd.DataFrame({
        "highw": [10.0, 1.0, 1.0],
        "loww": [5.0, 4.0, 0.0],
        "volume": [1.0, 1.0, 1.0],
        "total_turnover": [1.0, 1.0, 1.0],
    })
    clocks = window_clocks(part)
    assert clocks["ts_high"] == pytest.approx(0.0)
    assert clocks["ts_low"] == pytest.approx(1.0)
    assert clocks["ts_low_minus_high"] == pytest.approx(1.0)

def _priced_session(market: str, day: str, prices: dict[str, tuple[float, float]]):
    rows = []
    for hm, (open_px, close_px) in prices.items():
        rows.append({
            "stamp": pd.Timestamp(f"{day} {hm}"), "trading_date": pd.Timestamp(day),
            "open": open_px, "openw": open_px, "close": close_px, "closew": close_px,
            "highw": max(open_px, close_px), "loww": min(open_px, close_px),
            "volume": 10.0, "total_turnover": close_px * 10,
        })
    return rows

def test_entry_and_exit_follow_the_declared_session_rules():
    commodity = _priced_session("commodity", "2021-01-04", {
        "09:01": (90, 90), "10:00": (95, 95), "10:01": (100, 100),
        "11:30": (150, 150), "13:31": (160, 160), "15:00": (180, 180),
    })
    index = _priced_session("index", "2021-01-04", {
        "09:31": (90, 90), "10:00": (95, 95), "10:01": (100, 100),
        "11:30": (150, 150), "13:01": (160, 160), "15:00": (180, 180),
    })
    config = _config()
    commodity_rows = pd.DataFrame(rows_for_symbol(_frame(commodity), "RB", "commodity", config))
    index_rows = pd.DataFrame(rows_for_symbol(_frame(index), "IF", "index", config))
    commodity_ten = commodity_rows[commodity_rows.decision.eq("10:00")].iloc[0]
    index_ten = index_rows[index_rows.decision.eq("10:00")].iloc[0]
    commodity_lunch = commodity_rows[commodity_rows.decision.eq("11:30")].iloc[0]
    assert commodity_ten.gross == pytest.approx((150 - 100) / 100)
    assert index_ten.gross == pytest.approx((180 - 100) / 100)
    assert commodity_lunch.gross == pytest.approx((180 - 160) / 160)

def test_ten_oclock_requires_the_exact_next_minute():
    rows = _priced_session("commodity", "2021-01-04", {
        "09:01": (90, 90), "10:00": (95, 95), "11:30": (150, 150), "13:31": (160, 160), "15:00": (180, 180),
    })
    built = pd.DataFrame(rows_for_symbol(_frame(rows), "RB", "commodity", _config()))
    assert built[built.decision.eq("10:00")].empty
    assert not built[built.decision.eq("11:30")].empty

def test_positive_factor_passes_only_with_the_declared_sign():
    config = _config()
    rule = config["pass"]
    dates = pd.bdate_range("2016-03-01", periods=45)
    rows = []
    sectors = ["农产品", "黑色", "有色", "化工"]
    for year in range(2016, 2022):
        for day in dates:
            trading = day.replace(year=year)
            for slot in range(16):
                value = float(slot)
                rows.append({
                    "market": "commodity", "symbol": f"S{slot}", "sector": sectors[slot // 4],
                    "trading_date": trading, "decision": "close", "factor": "ts_low",
                    "expected": 1, "value": value, "gross": value / 100, "cost": 0.0,
                    "ctrl_return": ((slot * 3) % 7) / 10, "ctrl_location": 0.1 + ((slot * 5) % 7) / 10,
                    "ctrl_log_volume": 1.0 + (slot % 4),
                })
    panel = pd.DataFrame(rows)
    passed = screen_row(panel, "commodity", "截面", "close", "ts_low", 1, rule)
    rejected = screen_row(panel, "commodity", "截面", "close", "ts_low", -1, rule)
    assert passed["达到统计门槛"]
    assert rejected["达到统计门槛"] is False

def test_config_rejects_a_preset_volume_sign(tmp_path):
    config = _config()
    config["factors"].append({"id": "ts_volume", "expected": 1, "decisions": ["close"], "window": "upto"})
    path = tmp_path / "screen.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="成交量时戳"):
        load_config(path)

def test_day_session_product_enters_at_the_next_morning_open():
    rows = []
    for day, price in (("2021-01-04", 100), ("2021-01-05", 200), ("2021-01-06", 300)):
        rows.extend([
            _bar(f"{day} 09:01", day, price),
            _bar(f"{day} 15:00", day, price),
        ])
    built = pd.DataFrame(rows_for_symbol(_frame(rows), "AP", "commodity", _config()))
    trade = built[(built.decision == "close") & (built.trading_date == "2021-01-04")].iloc[0]
    assert trade.gross == pytest.approx((300 - 200) / 200)

def test_report_behavior_factor_is_only_emitted_for_close_decision():
    rows = []
    for day, price in (("2021-01-04", 100), ("2021-01-05", 101), ("2021-01-06", 102)):
        rows.extend(_session(day, price, price + 1, price + 10))
    built = pd.DataFrame(rows_for_symbol(_frame(rows), "RB", "commodity", _config()))
    report_factor = built[built.factor.eq("spike_am")]
    assert not report_factor.empty
    assert report_factor.decision.eq("close").all()

def test_collinear_controls_do_not_crash_the_regression():
    from research.time_factor_screen import _cross_section_coef
    slot = list(range(16))
    group = pd.DataFrame({
        "value": slot, "gross": slot, "ctrl_return": slot,
        "ctrl_location": [item * 2 for item in slot], "ctrl_log_volume": [item + 3 for item in slot],
    })
    assert pd.isna(_cross_section_coef(group))

def test_research_guard_rejects_2022():
    frame = _frame([_bar("2022-01-04 09:01", "2022-01-04", 100), _bar("2022-01-04 09:02", "2022-01-04", 101)])
    with pytest.raises(C.HoldoutViolation):
        prepare(frame)
