"""把日内文档里的固定数字收成可回测的规则。信号在当根收盘确认，下一根开盘成交。"""
import numpy as np


class Session:
    """一个品种、一个交易日的日盘。复权价做路径，原始价做收益率分母。"""

    def __init__(self, minute, open_w, high_w, low_w, close_w, open_raw, close_raw, volume):
        self.minute = np.asarray(minute, dtype=int)
        self.open_w = np.asarray(open_w, dtype="float64")
        self.high_w = np.asarray(high_w, dtype="float64")
        self.low_w = np.asarray(low_w, dtype="float64")
        self.close_w = np.asarray(close_w, dtype="float64")
        self.open_raw = np.asarray(open_raw, dtype="float64")
        self.close_raw = np.asarray(close_raw, dtype="float64")
        self.volume = np.asarray(volume, dtype="float64")
        self.first_minute = int(self.minute[0]) if len(self.minute) else 0


def rsi_wilder(close, period):
    """标准 Wilder RSI。前 period 根没有值。"""
    close = np.asarray(close, dtype="float64")
    out = np.full(len(close), np.nan)
    if len(close) <= period:
        return out
    change = np.diff(close)
    gain = np.clip(change, 0.0, None)
    loss = np.clip(-change, 0.0, None)
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(change)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        out[i + 1] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def walk_exit(side, entry, entry_w, entry_raw, bars, stop_pct, target_pct):
    """进场价是该根开盘。跳空穿过止损或目标时按开盘价成交；同一根里先算止损。"""
    last = len(bars.close_w) - 1
    for j in range(entry, last + 1):
        open_ret = (bars.open_w[j] - entry_w) / entry_raw
        if j > entry and side * open_ret <= -stop_pct:
            return j, side * open_ret
        if j > entry and target_pct is not None and side * open_ret >= target_pct:
            return j, side * open_ret
        up = (bars.high_w[j] - entry_w) / entry_raw
        down = (bars.low_w[j] - entry_w) / entry_raw
        if side > 0 and down <= -stop_pct:
            return j, -stop_pct
        if side < 0 and up >= stop_pct:
            return j, -stop_pct
        if target_pct is not None and side > 0 and up >= target_pct:
            return j, target_pct
        if target_pct is not None and side < 0 and down <= -target_pct:
            return j, target_pct
    price_ret = (bars.close_w[last] - entry_w) / entry_raw
    return last, side * price_ret


def _trade(side, signal, bars, stop_pct, target_pct, max_stop):
    entry = signal + 1
    if entry >= len(bars.close_w) or not bars.open_raw[entry] > 0:
        return None
    stop_pct = min(float(stop_pct), max_stop)
    if not np.isfinite(stop_pct) or stop_pct <= 1e-8:
        return None
    if target_pct is not None and (not np.isfinite(target_pct) or target_pct <= 0):
        target_pct = None
    entry_w = bars.open_w[entry]
    entry_raw = bars.open_raw[entry]
    exit_i, gross = walk_exit(side, entry, entry_w, entry_raw, bars, stop_pct, target_pct)
    return {"side": side, "entry": entry, "exit": exit_i, "stop_pct": stop_pct,
            "gross": gross, "entry_raw": entry_raw, "exit_raw": bars.close_raw[exit_i]}


def _clock(text):
    hour, minute = (int(part) for part in text.split(":"))
    return hour * 60 + minute


def _cap_stop(distance, max_stop):
    if not np.isfinite(distance) or distance <= 0:
        return None
    return min(float(distance), max_stop)


def resonance(bars, rsi, hist, rule, shared):
    """第一段达到近 3 日振幅的一半后，回调到 0.5，RSI 极端，再走回原方向。"""
    avg3 = hist.get("avg_range_3")
    if not np.isfinite(avg3) or avg3 <= 0 or len(bars.close_w) < 3:
        return []
    origin = bars.open_w[0]
    scale = bars.open_raw[0]
    if scale <= 0:
        return []
    move = (bars.close_w - origin) / scale
    hit = np.flatnonzero(np.abs(move) >= rule["impulse"] * avg3)
    if hit.size == 0:
        return []
    k = int(hit[0])
    side = 1.0 if move[k] > 0 else -1.0
    extreme = bars.high_w[:k + 1].max() if side > 0 else bars.low_w[:k + 1].min()
    level = origin + rule["retracement"] * (extreme - origin)
    touched = False
    was_extreme = False
    for i in range(k + 1, len(bars.close_w) - 1):
        if side > 0 and bars.low_w[i] <= level:
            touched = True
        if side < 0 and bars.high_w[i] >= level:
            touched = True
        if touched and (rsi[i] <= rule["rsi_oversold"] if side > 0 else rsi[i] >= rule["rsi_overbought"]):
            was_extreme = True
        crossed = bars.close_w[i] > level if side > 0 else bars.close_w[i] < level
        if touched and was_extreme and crossed:
            entry_w = bars.open_w[i + 1]
            entry_raw = bars.open_raw[i + 1]
            stop = _cap_stop(abs(entry_w - origin) / entry_raw, shared["max_stop"])
            target = side * (extreme - entry_w) / entry_raw
            trade = None if stop is None else _trade(side, i, bars, stop, target, shared["max_stop"])
            return [] if trade is None else [trade]
    return []


def opening_range_break(bars, hist, rule, shared):
    """日盘前 30 分钟的高低点，9:30 之后第一次突破顺势做。逢低没有给出幅度，按突破成交。"""
    hour, minute = (int(x) for x in rule["clock"].split(":"))
    clock = hour * 60 + minute
    if bars.first_minute <= clock - rule["open_minutes"] + 15:
        split = int(np.searchsorted(bars.minute, clock))
    else:
        split = min(rule["open_minutes"], len(bars.minute) - 1)
    if split < 5 or split >= len(bars.close_w) - 1:
        return []
    hi = bars.high_w[:split].max()
    lo = bars.low_w[:split].min()
    signal = None
    side = 0.0
    for i in range(split, len(bars.close_w) - 1):
        if bars.close_w[i] > hi:
            signal, side = i, 1.0
            break
        if bars.close_w[i] < lo:
            signal, side = i, -1.0
            break
    if signal is None:
        return []
    boundary = hi if side > 0 else lo
    entry_w = bars.open_w[signal + 1]
    entry_raw = bars.open_raw[signal + 1]
    stop = _cap_stop(abs(entry_w - (lo if side > 0 else hi)) / entry_raw, shared["max_stop"])
    target = None
    min3 = hist.get("min_range_3")
    if stop is not None and np.isfinite(min3):
        target = side * (boundary - entry_w) / entry_raw + 0.5 * min3
    trade = None if stop is None else _trade(side, signal, bars, stop, target, shared["max_stop"])
    return [] if trade is None else [trade]


def flag(bars, hist, rule, shared):
    """先走出至少 1% 且回撤不超过一半，缩量不超过 15 根后再顺向突破，目标为前一段的 1 倍。"""
    close, high, low = bars.close_w, bars.high_w, bars.low_w
    raw, volume = bars.close_raw, bars.volume
    n = len(close)
    i = 0
    while i < n - 3:
        peak, trough = high[i], low[i]
        direction = 0.0
        end = None
        for k in range(i + 1, n):
            peak = max(peak, high[k])
            trough = min(trough, low[k])
            up = (peak - low[i]) / raw[i]
            down = (high[i] - trough) / raw[i]
            up_back = (peak - low[k]) / raw[i]
            down_back = (high[k] - trough) / raw[i]
            done = False
            if up >= rule["impulse"]:
                if up_back <= rule["rebound_limit"] * up:
                    direction, end, extreme, origin = 1.0, k, peak, low[i]
                done = True
            elif down >= rule["impulse"]:
                if down_back <= rule["rebound_limit"] * down:
                    direction, end, extreme, origin = -1.0, k, trough, high[i]
                done = True
            if done:
                break
        if end is None:
            i += 1
            continue
        impulse_vol = volume[i:end + 1].mean()
        size = abs(extreme - origin) / raw[i]
        box_hi, box_lo = high[end], low[end]
        quiet = 0
        signal = None
        for c in range(end + 1, min(n - 1, end + 1 + rule["quiet_bars_max"])):
            if quiet >= 1 and ((direction > 0 and close[c] > box_hi) or (direction < 0 and close[c] < box_lo)):
                signal = c
                break
            if not impulse_vol > 0 or volume[c] >= rule["quiet_volume"] * impulse_vol:
                break
            quiet += 1
            box_hi = max(box_hi, high[c])
            box_lo = min(box_lo, low[c])
        if signal is None:
            i = end + 1
            continue
        entry_w = bars.open_w[signal + 1]
        entry_raw = bars.open_raw[signal + 1]
        invalid = box_lo if direction > 0 else box_hi
        stop = _cap_stop(abs(entry_w - invalid) / entry_raw, shared["max_stop"])
        target = size * rule["measured_move"] - direction * (entry_w - (box_hi if direction > 0 else box_lo)) / entry_raw
        trade = None if stop is None else _trade(direction, signal, bars, stop, target, shared["max_stop"])
        if trade is None:
            i = end + 1
            continue
        return [trade]
    return []


def rubber_open(bars, hist, rule, shared):
    """天胶：9:10 至 9:15，开盘方向超过 150 个价格点就反向做，目标取回这段的一半。"""
    if hist.get("symbol") != rule["symbol"]:
        return []
    origin = bars.open_raw[0]
    if origin <= 0:
        return []
    signal = None
    for i, minute in enumerate(bars.minute[:-1]):
        if minute < _clock(rule["not_before"]) or minute > _clock(rule["observe_until"]):
            continue
        delta = bars.close_raw[i] - origin
        if abs(delta) >= rule["points"]:
            signal = i
            break
    if signal is None:
        return []
    delta = bars.close_raw[signal] - origin
    side = -np.sign(delta)
    entry_raw = bars.open_raw[signal + 1]
    entry_w = bars.open_w[signal + 1]
    extreme = bars.close_w[signal]
    stop = _cap_stop((abs(entry_w - extreme) + rule["points"]) / entry_raw, shared["max_stop"])
    target = abs(delta) * rule["target_retracement"] / entry_raw - abs(entry_w - extreme) / entry_raw
    trade = None if stop is None else _trade(side, signal, bars, stop, target, shared["max_stop"])
    return [] if trade is None else [trade]


def hour_break(bars, hist, rule, shared):
    """至少一小时的高低点被收盘突破，并且这一小时的涨跌方向一致。"""
    n = len(bars.close_w)
    width = rule["minutes"]
    if n < width + 2:
        return []
    for i in range(width, n - 1):
        prior_hi = bars.high_w[i - width:i].max()
        prior_lo = bars.low_w[i - width:i].min()
        base = bars.close_raw[i - width]
        if base <= 0:
            continue
        ret = (bars.close_w[i] - bars.close_w[i - width]) / base
        if bars.close_w[i] > prior_hi and ret > 0:
            side, level = 1.0, prior_hi
            break
        if bars.close_w[i] < prior_lo and ret < 0:
            side, level = -1.0, prior_lo
            break
    else:
        return []
    entry_w = bars.open_w[i + 1]
    entry_raw = bars.open_raw[i + 1]
    stop = _cap_stop(abs(entry_w - level) / entry_raw, shared["max_stop"])
    target = None
    min3 = hist.get("min_range_3")
    if stop is not None and np.isfinite(min3):
        target = side * (level - entry_w) / entry_raw + 0.5 * min3
    trade = None if stop is None else _trade(side, i, bars, stop, target, shared["max_stop"])
    return [] if trade is None else [trade]


def climax_fade(bars, hist, rule, shared):
    """5 到 15 分钟内波动达到 1.5% 就反向做，目标为这段的 0.382，极值外再放 0.5% 止损。"""
    close, raw = bars.close_w, bars.close_raw
    n = len(close)
    for i in range(rule["minutes_max"], n - 1):
        best = None
        for width in range(rule["minutes_min"], rule["minutes_max"] + 1):
            start = i - width
            base = raw[start]
            if base <= 0:
                continue
            ret = (close[i] - close[start]) / base
            if abs(ret) >= rule["move"] and (best is None or abs(ret) > abs(best[0])):
                best = (ret, start)
        if best is None:
            continue
        ret, start = best
        side = -np.sign(ret)
        if side > 0:
            extreme = bars.low_w[start:i + 1].min()
            begin = bars.high_w[start:i + 1].max()
        else:
            extreme = bars.high_w[start:i + 1].max()
            begin = bars.low_w[start:i + 1].min()
        entry_w = bars.open_w[i + 1]
        entry_raw = bars.open_raw[i + 1]
        impulse = abs(extreme - begin) / entry_raw
        stop = _cap_stop(abs(entry_w - extreme) / entry_raw + rule["beyond_extreme"], shared["max_stop"])
        target = impulse * rule["target_retracement"] - abs(entry_w - extreme) / entry_raw
        trade = None if stop is None else _trade(side, i, bars, stop, target, shared["max_stop"])
        return [] if trade is None else [trade]
    return []


def compression_break(bars, hist, rule, shared):
    """至少 30 根振幅不超过 0.5%，收盘离开中心 0.25% 顺势做。每多半小时，目标多 0.5 个点。"""
    n = len(bars.close_w)
    if n < rule["minutes"] + 2 or bars.open_raw[0] <= 0:
        return []
    width = rule["width"] * bars.open_raw[0]
    start, window_max, window_min = _tight_windows(bars.high_w, bars.low_w, width)
    for i in range(rule["minutes"] - 1, n - 2):
        if start[i] < 0 or i - start[i] + 1 < rule["minutes"]:
            continue
        center = 0.5 * (window_max[i] + window_min[i])
        band = rule["break"] * bars.open_raw[0]
        move = bars.close_w[i + 1] - center
        if abs(move) <= band:
            continue
        side = 1.0 if move > 0 else -1.0
        half_hours = (i - int(start[i]) + 1) // rule["minutes"]
        entry_w = bars.open_w[i + 2]
        entry_raw = bars.open_raw[i + 2]
        from_center = side * (center - entry_w) / entry_raw
        profit = rule["target_profit"] + rule["extra_half_hour"] * max(half_hours - 1, 0)
        target = from_center + band / entry_raw + profit
        invalid = center - side * band
        stop = _cap_stop(abs(entry_w - invalid) / entry_raw, shared["max_stop"])
        trade = None if stop is None else _trade(side, i + 1, bars, stop, target, shared["max_stop"])
        return [] if trade is None else [trade]
    return []


def _tight_windows(high, low, width_points):
    """每根 K 线为止、振幅仍不超过给定价格宽度的最长窗口。"""
    from collections import deque
    n = len(high)
    left = np.full(n, -1)
    wmax = np.full(n, np.nan)
    wmin = np.full(n, np.nan)
    maxd, mind = deque(), deque()
    j = 0
    for i in range(n):
        while maxd and high[maxd[-1]] <= high[i]:
            maxd.pop()
        maxd.append(i)
        while mind and low[mind[-1]] >= low[i]:
            mind.pop()
        mind.append(i)
        while maxd and high[maxd[0]] - low[mind[0]] > width_points:
            j += 1
            while maxd and maxd[0] < j:
                maxd.popleft()
            while mind and mind[0] < j:
                mind.popleft()
        if maxd and mind and j <= i:
            left[i] = j
            wmax[i] = high[maxd[0]]
            wmin[i] = low[mind[0]]
    return left, wmax, wmin


def gap_fade(bars, hist, rule, shared):
    """相对前一日盘收盘高开做空、低开做多，持有到收盘，价格反向 2% 止损。"""
    prior = hist.get("prior_close_raw")
    if prior is None or not prior > 0 or bars.open_raw[0] <= 0:
        return []
    gap = bars.open_raw[0] / prior - 1.0
    if gap == 0 or not np.isfinite(gap):
        return []
    side = -np.sign(gap)
    entry_w = bars.open_w[0]
    entry_raw = bars.open_raw[0]
    exit_i, gross = walk_exit(side, 0, entry_w, entry_raw, bars, rule["stop"], None)
    return [{"side": side, "entry": 0, "exit": exit_i, "stop_pct": rule["stop"],
             "gross": gross, "entry_raw": entry_raw, "exit_raw": bars.close_raw[exit_i]}]


RULES = {
    "首轮波动共振": resonance,
    "开盘三十分钟突破": opening_range_break,
    "旗形再突破": flag,
    "天胶开盘反向": rubber_open,
    "一小时突破": hour_break,
    "急动反向": climax_fade,
    "窄幅横盘突破": compression_break,
    "跳空反向": gap_fade,
}
