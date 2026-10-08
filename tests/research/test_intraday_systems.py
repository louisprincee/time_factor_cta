"""日内文档规则的方向、时点和止损。不读行情。"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import intraday_systems as rules

SPEC = json.loads((Path(__file__).resolve().parents[2] / "config" / "intraday_systems.json").read_text(encoding="utf-8"))
SHARED = SPEC["shared"]


def session(close, high=None, low=None, volume=None, start=9 * 60, raw=None):
    close = np.asarray(close, dtype="float64")
    high = close.copy() if high is None else np.asarray(high, dtype="float64")
    low = close.copy() if low is None else np.asarray(low, dtype="float64")
    open_ = np.r_[close[0], close[:-1]]
    price = close if raw is None else np.asarray(raw, dtype="float64")
    open_raw = np.r_[price[0], price[:-1]]
    volume = np.ones(len(close)) if volume is None else np.asarray(volume, dtype="float64")
    return rules.Session(start + np.arange(len(close)), open_, high, low, close, open_raw, price, volume)


def test_stop_is_filled_before_the_target_on_the_same_bar():
    bars = session([100, 100, 100], high=[100, 103, 100], low=[100, 97, 100])
    exit_i, gross = rules.walk_exit(1, 1, bars.open_w[1], bars.open_raw[1], bars, 0.02, 0.02)
    assert exit_i == 1 and gross == -0.02


def test_gap_fade_shorts_a_higher_open():
    bars = session([101, 101, 101])
    trade = rules.gap_fade(bars, {"prior_close_raw": 100.0}, SPEC["rules"]["跳空反向"], SHARED)[0]
    assert trade["side"] == -1 and trade["entry"] == 0


def test_opening_range_waits_until_0930():
    close = np.full(40, 100.0)
    high = np.full(40, 100.2)
    low = np.full(40, 99.8)
    close[31] = 100.8
    high[31] = 100.8
    bars = session(close, high, low)
    hist = {"min_range_3": 0.01}
    trades = rules.opening_range_break(bars, hist, SPEC["rules"]["开盘三十分钟突破"], SHARED)
    assert trades[0]["side"] == 1 and trades[0]["entry"] == 32


def test_compression_break_follows_the_escape_from_a_narrow_box():
    close = np.full(42, 100.0)
    high = np.full(42, 100.1)
    low = np.full(42, 99.95)
    close[40] = 100.6
    high[40] = 100.6
    bars = session(close, high, low)
    trades = rules.compression_break(bars, {}, SPEC["rules"]["窄幅横盘突破"], SHARED)
    assert trades[0]["side"] == 1 and trades[0]["entry"] == 41


def test_climax_fades_a_fast_drop():
    close = np.r_[np.full(20, 100.0), np.linspace(100, 97.5, 10)]
    bars = session(close, high=np.maximum.accumulate(close), low=close)
    # 低点就是收盘，保证这段是向下的。
    bars.high_w[:] = np.maximum(bars.close_w, bars.open_w)
    bars.low_w[:] = np.minimum(bars.close_w, bars.open_w)
    trades = rules.climax_fade(bars, {}, SPEC["rules"]["急动反向"], SHARED)
    assert trades and trades[0]["side"] == 1


def test_rubber_fades_a_150_point_drop_after_0910():
    close = np.full(20, 12000.0)
    close[12:] = 11800.0
    bars = session(close)
    trades = rules.rubber_open(bars, {"symbol": "RU"}, SPEC["rules"]["天胶开盘反向"], SHARED)
    assert trades[0]["side"] == 1 and bars.minute[trades[0]["entry"]] >= 9 * 60 + 10


def test_hour_break_goes_with_the_hour():
    close = np.linspace(100, 102, 70)
    bars = session(close)
    trades = rules.hour_break(bars, {"min_range_3": 0.01}, SPEC["rules"]["一小时突破"], SHARED)
    assert trades[0]["side"] == 1 and trades[0]["entry"] >= 60


def test_flag_continues_after_a_quiet_pause():
    close = np.r_[np.linspace(100, 101.2, 8), np.full(4, 101.0), [101.5, 101.5]]
    volume = np.r_[np.ones(7), np.full(7, 0.1)]
    bars = session(close, high=close + 0.01, low=close - 0.01, volume=volume)
    trades = rules.flag(bars, {}, SPEC["rules"]["旗形再突破"], SHARED)
    assert trades and trades[0]["side"] == 1


def test_resonance_buys_the_first_pullback():
    rise = np.linspace(100, 101.5, 20)
    drop = np.linspace(101.5, 99.0, 25)
    back = np.linspace(99.2, 101.2, 8)
    close = np.r_[rise, drop, back]
    bars = session(close, high=close + 0.02, low=close - 0.02)
    rsi = rules.rsi_wilder(close, 14)
    trades = rules.resonance(bars, rsi, {"avg_range_3": 0.02}, SPEC["rules"]["首轮波动共振"], SHARED)
    assert trades and trades[0]["side"] == 1 and trades[0]["entry"] > 15
