"""时间戳和公允价偏离的单因子预测性检验。只读研究期。

    python -m research.time_factor_screen

规则在 config/research/time_factor_screen.json，而且必须先于结果写定。
不读 2022，不读 2023–2025，不构造因子配对。
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tfcta import config as C
from tfcta.data import load_shard, tag_session
from tfcta.factors import (
    add_intraday_coords, day_codes_of, dfp_factors, duration_series,
    intraday_abs_diff, raw_price_path, report_factors, rolling_threshold,
)
from tfcta.costs import MULTIPLIER, commission_rate, load_fees

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "research" / "time_factor_screen.json"
OUT = ROOT / "runs" / "research" / "time_factor_screen"
EVAL_START = pd.Timestamp("2016-01-01")
EVAL_END = pd.Timestamp("2021-12-31")
MINUTE = timedelta(minutes=1)


def load_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("status") != "declared_screen_not_a_strategy":
        raise ValueError("检验配置必须声明这不是策略")
    if "不读 2022" not in config.get("sample", "") or "2023" not in config.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 和 2023 年以后")
    if any(item.get("expected") is not None and item["id"].startswith("ts_") and "volume" in item["id"]
           or item.get("expected") is not None and item["id"] in {"ts_turnover", "ts_half_volume"}
           for item in config["factors"]):
        raise ValueError("成交量时戳不能在这一轮写入预期符号")
    return config


def stamp(day, hm: str) -> pd.Timestamp:
    hour, minute = (int(part) for part in hm.split(":"))
    day = pd.Timestamp(day)
    return pd.Timestamp(day.year, day.month, day.day, hour, minute)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """排序并拒绝 2022 及以后。夜盘已由 trading_date 归到所属交易日。"""
    if "trading_date" not in frame.columns:
        raise KeyError("缺少 trading_date")
    frame = frame.sort_index(kind="mergesort")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("分钟索引必须是时间")
    if frame.index.has_duplicates:
        raise ValueError("分钟时间戳重复")
    frame = frame.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(frame["trading_date"], "时间因子检验")
    if not bool(frame["trading_date"].is_monotonic_increasing):
        raise ValueError("trading_date 没有按墙钟时间排好，夜盘归属可能错了")
    return frame


def _position(values: np.ndarray, mode: str) -> float:
    values = np.asarray(values, dtype="float64")
    if values.size < 2 or not np.isfinite(values).any():
        return np.nan
    if mode == "max":
        scored = np.where(np.isfinite(values), values, -np.inf)
        loc = int(np.argmax(scored))
    else:
        scored = np.where(np.isfinite(values), values, np.inf)
        loc = int(np.argmin(scored))
    return float(loc / (values.size - 1))


def _half_volume(volume: np.ndarray) -> float:
    volume = np.asarray(volume, dtype="float64")
    if volume.size < 2 or not np.isfinite(volume).all() or float(volume.sum()) <= 0:
        return np.nan
    reached = np.flatnonzero(np.cumsum(volume) >= 0.5 * float(volume.sum()))
    if reached.size == 0:
        return np.nan
    return float(reached[0] / (volume.size - 1))


def window_clocks(part: pd.DataFrame) -> dict[str, float]:
    """窗口内的高低点和成交量时戳。0 是第一根，1 是最后一根，相同极值取最早一根。"""
    high = _position(part.highw.to_numpy(), "max")
    low = _position(part.loww.to_numpy(), "min")
    return {
        "ts_high": high,
        "ts_low": low,
        "ts_low_minus_high": low - high if np.isfinite(high) and np.isfinite(low) else np.nan,
        "ts_volume": _position(part.volume.to_numpy(), "max"),
        "ts_turnover": _position(part.total_turnover.to_numpy(), "max"),
        "ts_half_volume": _half_volume(part.volume.to_numpy()),
    }


def extreme_clocks(part: pd.DataFrame, prefix: str) -> dict[str, float]:
    if len(part) < 2:
        return {f"ts_high_{prefix}": np.nan, f"ts_low_{prefix}": np.nan}
    return {
        f"ts_high_{prefix}": _position(part.highw.to_numpy(), "max"),
        f"ts_low_{prefix}": _position(part.loww.to_numpy(), "min"),
    }


def controls_of(part: pd.DataFrame) -> dict[str, float]:
    """决策时点已经走完的涨跌幅、区间位置和成交量。涨跌幅用加法复权。"""
    empty = {"ctrl_return": np.nan, "ctrl_location": np.nan, "ctrl_log_volume": np.nan}
    if part.empty:
        return empty
    entry = float(part.open.iloc[0])
    if not np.isfinite(entry) or entry <= 0:
        return empty
    change = float(part.closew.iloc[-1] - part.openw.iloc[0]) / entry
    high = float(np.nanmax(part.highw.to_numpy()))
    low = float(np.nanmin(part.loww.to_numpy()))
    last = float(part.closew.iloc[-1])
    location = (last - low) / (high - low) if np.isfinite(high) and np.isfinite(low) and high > low else np.nan
    volume = float(np.nansum(part.volume.to_numpy()))
    return {
        "ctrl_return": change,
        "ctrl_location": location,
        "ctrl_log_volume": float(np.log(volume)) if volume > 0 else np.nan,
    }


def adjusted_return(entry_w, entry_raw, exit_w) -> float:
    if not (np.isfinite(entry_w) and np.isfinite(entry_raw) and np.isfinite(exit_w) and entry_raw > 0):
        return np.nan
    return float(exit_w - entry_w) / float(entry_raw)


def fair_prices(frame: pd.DataFrame, lookback: int, percentile: float, fixed: float | None):
    """公允价偏离。滚动阈值不含当天。固定阈值只在调用方要求时计算。"""
    codes, days = day_codes_of(frame)
    price = raw_price_path(frame)
    rolling = rolling_threshold(intraday_abs_diff(price, codes), codes, lookback, percentile)
    fair = dfp_factors(duration_series(price, codes, rolling), price, codes, days)
    fair.index = pd.DatetimeIndex(fair.index).normalize()
    fixed_fair = None
    if fixed is not None:
        constant = pd.Series(float(fixed), index=rolling.index)
        fixed_fair = dfp_factors(duration_series(price, codes, constant), price, codes, days)
        fixed_fair.index = pd.DatetimeIndex(fixed_fair.index).normalize()
    return fair, fixed_fair


def _bar(part: pd.DataFrame, day, hm: str):
    ts = stamp(day, hm)
    if ts not in part.index:
        return None
    row = part.loc[ts]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"{ts} 有重复分钟")
    return row


def _next_exact_minute(part: pd.DataFrame, day, hm: str):
    ts = stamp(day, hm) + MINUTE
    if ts not in part.index:
        return None
    return part.loc[ts]


def _next_later_bar(part: pd.DataFrame, day, hm: str):
    later = part.loc[part.index > stamp(day, hm)]
    if later.empty:
        return None
    return later.iloc[0]


def trade_outcome(days: dict, day, spec: dict):
    """返回主收益、成本所需价格，以及隔时段的路径分解。路径不参与判定。"""
    part = days[day]
    kind = spec["kind"]
    if kind == "intraday":
        if spec["entry"] == "next_minute":
            entry = _next_exact_minute(part, day, spec["id"])
        else:
            entry = _next_later_bar(part, day, spec["id"])
        exit_bar = _bar(part, day, spec["exit"])
        if entry is None or exit_bar is None:
            return None
        gross = adjusted_return(entry.openw, entry.open, exit_bar.closew)
        return {
            "gross": gross, "entry_raw": float(entry.open), "exit_raw": float(exit_bar.close),
            "intraday": True, "gap_before_entry": np.nan, "hold_night": np.nan,
            "hold_morning": np.nan, "hold_afternoon": np.nan, "hold_after_close": np.nan,
        }
    ordered = sorted(days)
    pos = ordered.index(day)
    if pos + 2 >= len(ordered):
        return None
    hold, after = ordered[pos + 1], ordered[pos + 2]
    entry = days[hold].iloc[0]
    exit_bar = days[after].iloc[0]
    signal_close_w = float(part.closew.iloc[-1])
    signal_close = float(part.close.iloc[-1])
    gross = adjusted_return(entry.openw, entry.open, exit_bar.openw)
    labels = tag_session(days[hold].index).to_numpy()
    morning = days[hold].loc[labels == C.SESSION_AM]
    afternoon_bar = _bar(days[hold], hold, "15:00")
    morning_end = _bar(days[hold], hold, "11:30")
    hold_night = hold_morning = hold_afternoon = hold_after = np.nan
    if morning_end is not None and afternoon_bar is not None and len(morning):
        entry_raw = float(entry.open)
        if tag_session(pd.DatetimeIndex([entry.name])).iloc[0] == C.SESSION_NIGHT:
            hold_night = adjusted_return(entry.openw, entry_raw, float(morning.openw.iloc[0]))
            hold_morning = adjusted_return(float(morning.openw.iloc[0]), entry_raw, float(morning_end.closew))
        else:
            hold_night = 0.0
            hold_morning = adjusted_return(entry.openw, entry_raw, float(morning_end.closew))
        hold_afternoon = adjusted_return(float(morning_end.closew), entry_raw, float(afternoon_bar.closew))
        hold_after = adjusted_return(float(afternoon_bar.closew), entry_raw, float(exit_bar.openw))
    return {
        "gross": gross, "entry_raw": float(entry.open), "exit_raw": float(exit_bar.open),
        "intraday": False,
        "gap_before_entry": adjusted_return(signal_close_w, signal_close, float(entry.openw)),
        "hold_night": hold_night, "hold_morning": hold_morning,
        "hold_afternoon": hold_afternoon, "hold_after_close": hold_after,
    }


def round_trip_cost(symbol: str, day, entry_raw: float, exit_raw: float, intraday: bool,
                    fees, ticks) -> float:
    if symbol not in MULTIPLIER or fees is None:
        return np.nan
    history = fees.get(symbol)
    if history is None or history.empty:
        return np.nan
    eligible = history.loc[:pd.Timestamp(day)]
    if eligible.empty:
        return np.nan
    fee = eligible.iloc[-1]
    tick = (ticks or {}).get(symbol, {}).get(pd.Timestamp(day).year - 1)
    if tick is None or not np.isfinite(entry_raw) or entry_raw <= 0:
        return np.nan
    close_fee = fee.close_commission_today if intraday else fee.close_commission
    if pd.isna(close_fee):
        close_fee = fee.close_commission
    if pd.isna(close_fee) or pd.isna(fee.open_commission):
        return np.nan
    opened = commission_rate(fee.commission_type, fee.open_commission, entry_raw, entry_raw, MULTIPLIER[symbol])
    closed = commission_rate(fee.commission_type, close_fee, exit_raw, entry_raw, MULTIPLIER[symbol])
    return float(opened + closed + 2.0 * float(tick) / entry_raw)


def _slice_until(part: pd.DataFrame, day, hm: str):
    end = stamp(day, hm)
    if end not in part.index:
        return None
    return part.loc[:end]


def rows_for_symbol(frame: pd.DataFrame, symbol: str, market: str, config: dict,
                    universe: dict | None = None, fees=None, ticks=None) -> list[dict]:
    """一个品种在各决策时点上、当时已经看得到的因子和其后收益。"""
    frame = prepare(frame)
    duration = config["duration"]
    fixed = float(duration["fixed_price_threshold"]) if market in duration["fixed_markets"] else None
    fair, fixed_fair = fair_prices(frame, int(duration["lookback"]), float(duration["percentile"]), fixed)
    report = None
    if market == "commodity":
        report = report_factors(
            add_intraday_coords(frame),
            int(duration["lookback"]),
            float(duration["percentile"]),
        )
    days = {pd.Timestamp(day): part for day, part in frame.groupby("trading_date", sort=True)}
    sector_of = {name: sector for sector, names in config["sectors"].items() for name in names}
    applicable = [item for item in config["factors"] if market in item.get("markets", [market])]
    by_decision = {}
    for item in applicable:
        for decision_id in item["decisions"]:
            by_decision.setdefault(decision_id, []).append(item)
    specs = {item["id"]: item for item in config["decisions"][market]}
    rows = []
    for day, part in days.items():
        if day < EVAL_START or day > EVAL_END:
            continue
        if universe is not None and symbol not in universe.get(str(day.year), ()):
            continue
        labels = tag_session(part.index).to_numpy()
        morning = part.loc[labels == C.SESSION_AM]
        afternoon = part.loc[labels == C.SESSION_PM]
        for decision_id, factors in by_decision.items():
            spec = specs[decision_id]
            if spec["kind"] == "next_session" and _bar(part, day, "15:00") is None:
                continue
            window = part if spec["kind"] == "next_session" else _slice_until(part, day, decision_id)
            if window is None or len(window) < 2:
                continue
            outcome = trade_outcome(days, day, spec)
            if outcome is None or not np.isfinite(outcome["gross"]):
                continue
            values = window_clocks(window)
            if decision_id in {"11:30", "close"}:
                values.update(extreme_clocks(morning, "am"))
            if decision_id == "close":
                values.update(extreme_clocks(afternoon, "pm"))
                if day in fair.index:
                    values["dfp_max"] = float(fair.at[day, "dfp_max"])
                    values["dfp_top3"] = float(fair.at[day, "dfp_top3"])
                if fixed_fair is not None and day in fixed_fair.index:
                    values["dfp_max_fixed"] = float(fixed_fair.at[day, "dfp_max"])
                    values["dfp_top3_fixed"] = float(fixed_fair.at[day, "dfp_top3"])
                if report is not None and day in report.index:
                    for factor in ("pmt", "vmt", "vd_ratio", "spike_am", "vol_pm"):
                        values[factor] = float(report.at[day, factor])
            ctrl = controls_of(window)
            cost_day = day if outcome["intraday"] else sorted(days)[sorted(days).index(day) + 1]
            cost = round_trip_cost(symbol, cost_day, outcome["entry_raw"], outcome["exit_raw"],
                                    outcome["intraday"], fees, ticks) if market == "commodity" else np.nan
            for item in factors:
                value = values.get(item["id"], np.nan)
                if not np.isfinite(value):
                    continue
                rows.append({
                    "market": market, "symbol": symbol, "sector": sector_of.get(symbol, "未分板块"),
                    "trading_date": day, "decision": decision_id, "factor": item["id"],
                    "expected": item["expected"], "value": float(value), "gross": outcome["gross"],
                    "cost": cost, **ctrl, "gap_before_entry": outcome["gap_before_entry"],
                    "hold_night": outcome["hold_night"], "hold_morning": outcome["hold_morning"],
                    "hold_afternoon": outcome["hold_afternoon"], "hold_after_close": outcome["hold_after_close"],
                })
    return rows


def _correlation(x, y) -> float:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return np.nan
    result = stats.spearmanr(x[ok], y[ok])
    value = getattr(result, "correlation", result.statistic)
    return float(value)


def nw_mean(values: np.ndarray, max_lags: int) -> tuple[float, float]:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    count = len(values)
    if count < 3:
        return (float(values.mean()) if count else np.nan, np.nan)
    centered = values - values.mean()
    variance = float(centered @ centered) / count
    lags = min(max_lags, count - 1)
    for lag in range(1, lags + 1):
        gamma = float(centered[lag:] @ centered[:-lag]) / count
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    scale = np.sqrt(max(variance, 0.0) / count)
    mean = float(values.mean())
    if scale > 0:
        return mean, float(mean / scale)
    # 每天完全相同且不为零时，标准误会变成 0，这比临界值更强，不能记成缺失。
    if mean == 0:
        return mean, np.nan
    return mean, float(np.sign(mean) * np.inf)


def ols_hac(y: np.ndarray, columns: list[np.ndarray], lags: int) -> tuple[float, float]:
    """第一列斜率的 HAC t。columns 不含截距。"""
    y = np.asarray(y, dtype="float64")
    design = np.column_stack([np.ones(len(y))] + [np.asarray(col, dtype="float64") for col in columns])
    ok = np.isfinite(y) & np.isfinite(design).all(axis=1)
    y, design = y[ok], design[ok]
    if len(y) < design.shape[1] + 5:
        return np.nan, np.nan
    try:
        bread = np.linalg.pinv(design.T @ design)
        beta = bread @ design.T @ y
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    if not np.isfinite(beta[1]):
        return np.nan, np.nan
    score = design * (y - design @ beta)[:, None]
    meat = score.T @ score
    use = min(lags, len(y) - 1)
    for lag in range(1, use + 1):
        gamma = score[lag:].T @ score[:-lag]
        weight = 1.0 - lag / (use + 1.0)
        meat += weight * (gamma + gamma.T)
    variance = bread @ meat @ bread
    scale = float(np.sqrt(max(variance[1, 1], 0.0)))
    coef = float(beta[1])
    if scale > 0:
        return coef, float(coef / scale)
    if coef == 0:
        return coef, np.nan
    return coef, float(np.sign(coef) * np.inf)


def _daily_statistic(part: pd.DataFrame, minimum: int, function) -> pd.Series:
    rows = {}
    for day, group in part.groupby("trading_date", sort=True):
        if len(group) < minimum:
            continue
        rows[pd.Timestamp(day)] = function(group)
    return pd.Series(rows, dtype="float64").sort_index()


def _cross_section_ic(group: pd.DataFrame) -> float:
    return _correlation(group.value, group.gross)


def _cross_section_coef(group: pd.DataFrame) -> float:
    fields = ["value", "ctrl_return", "ctrl_location", "ctrl_log_volume", "gross"]
    if group[fields].nunique().min() < 2:
        return np.nan
    y = group.gross.rank(pct=True).to_numpy() - 0.5
    columns = [(group[name].rank(pct=True).to_numpy() - 0.5) for name in fields[:-1]]
    design = np.column_stack([np.ones(len(group))] + columns)
    try:
        beta = np.linalg.solve(design.T @ design, design.T @ y)
    except np.linalg.LinAlgError:
        return np.nan
    coef = float(beta[1])
    # 秩回归的因变量本身只在 -0.5 到 0.5。过大的系数是共线，不是经济效应。
    if not np.isfinite(coef) or abs(coef) > 10:
        return np.nan
    return coef


def _matched_years(part: pd.DataFrame, daily: pd.Series, year_values: pd.Series,
                   market: str, target: float, minimum_days: int) -> tuple[int, int]:
    judgeable = match = 0
    for year, value in year_values.items():
        if market == "commodity":
            enough = int((daily.index.year == int(year)).sum()) >= minimum_days
        else:
            enough = int(part.trading_date.dt.year.eq(int(year)).sum()) >= minimum_days
        if not enough or not np.isfinite(value):
            continue
        judgeable += 1
        if np.isfinite(target) and target != 0 and float(value) * target > 0:
            match += 1
    return judgeable, match


def _matched_sectors(sector_ic: dict[str, pd.Series], target: float, minimum_days: int) -> tuple[int, int]:
    """板块用当天至少 4 个品种的截面相关。贵金属通常不足 4 个，记为不足。"""
    judgeable = match = 0
    for values in sector_ic.values():
        usable = values[np.isfinite(values)]
        if len(usable) < minimum_days:
            continue
        judgeable += 1
        if np.isfinite(target) and target != 0 and float(usable.mean()) * target > 0:
            match += 1
    return judgeable, match


def _year_ic(part: pd.DataFrame, daily: pd.Series, market: str) -> pd.Series:
    if market == "commodity":
        return daily.groupby(daily.index.year).apply(lambda values: float(np.nanmean(values)))
    rows = {}
    for year, group in part.groupby(part.trading_date.dt.year):
        rows[int(year)] = _correlation(group.value, group.gross)
    return pd.Series(rows, dtype="float64")


def _sector_ic(part: pd.DataFrame, sectors: list[str], minimum_names: int) -> dict[str, pd.Series]:
    out = {}
    for sector in sectors:
        chosen = part[part.sector.eq(sector)]
        if chosen.empty:
            out[sector] = pd.Series(dtype="float64")
            continue
        out[sector] = _daily_statistic(chosen, minimum_names, _cross_section_ic).dropna()
    return out


def trailing_bucket_spread(part: pd.DataFrame, lookback: int, minimum: int) -> tuple[float, float]:
    """单合约用过去窗口的分位，不含当天。返回高低桶平均收益差，以及每边打平所需的往返成本。"""
    ordered = part.sort_values("trading_date")
    values = ordered.value.to_numpy(dtype="float64")
    gross = ordered.gross.to_numpy(dtype="float64")
    top, bottom = [], []
    for pos in range(len(ordered)):
        start = max(0, pos - lookback)
        history = values[start:pos]
        history = history[np.isfinite(history)]
        if history.size < minimum or not np.isfinite(values[pos]) or not np.isfinite(gross[pos]):
            continue
        if values[pos] >= np.quantile(history, 0.8):
            top.append(gross[pos])
        elif values[pos] <= np.quantile(history, 0.2):
            bottom.append(gross[pos])
    if not top or not bottom:
        return np.nan, np.nan
    spread = float(np.mean(top) - np.mean(bottom))
    return spread, spread / 2.0


def commodity_quintile(part: pd.DataFrame, minimum: int) -> tuple[float, float, float]:
    gross_rows, net_rows = [], []
    for _, group in part.groupby("trading_date", sort=True):
        usable = group[np.isfinite(group.value) & np.isfinite(group.gross) & np.isfinite(group.cost)]
        if len(usable) < minimum or usable.value.nunique() < 5:
            continue
        rank = usable.value.rank(pct=True)
        top, bottom = usable[rank >= 0.8], usable[rank <= 0.2]
        if top.empty or bottom.empty:
            continue
        gross_rows.append(float(top.gross.mean() - bottom.gross.mean()))
        net_rows.append(float((top.gross - top.cost).mean() - (bottom.gross - bottom.cost).mean()))
    gross_mean = float(np.mean(gross_rows)) if gross_rows else np.nan
    net_mean = float(np.mean(net_rows)) if net_rows else np.nan
    return gross_mean, net_mean, float(len(gross_rows))


def screen_row(part: pd.DataFrame, market: str, symbol: str, decision: str, factor: str,
               expected, rule: dict) -> dict:
    lags = int(rule["hac_lags"])
    if market == "commodity":
        daily = _daily_statistic(part, int(rule["min_names_ic"]), _cross_section_ic).dropna()
        coef_daily = _daily_statistic(part, int(rule["min_names_regression"]), _cross_section_coef).dropna()
        ic_mean, ic_t = nw_mean(daily.to_numpy(), lags)
        controlled_coef, controlled_t = nw_mean(coef_daily.to_numpy(), lags)
        gross_q, net_q, q_days = commodity_quintile(part, int(rule["min_names_ic"]))
        break_even = np.nan
        sector_ic = _sector_ic(part, rule["sectors"], 4)
    else:
        ic_mean = _correlation(part.value, part.gross)
        _, ic_t = ols_hac(part.gross.to_numpy(), [part.value.to_numpy()], lags)
        controlled_coef, controlled_t = ols_hac(
            part.gross.to_numpy(),
            [part[name].to_numpy() for name in ("value", "ctrl_return", "ctrl_location", "ctrl_log_volume")],
            lags)
        gross_q, break_even = trailing_bucket_spread(
            part, int(rule["index_quintile_lookback"]), int(rule["index_quintile_min_obs"]))
        net_q, q_days = np.nan, np.nan
        sector_judgeable = sector_match = np.nan
        daily = pd.Series(dtype="float64")
    year_values = _year_ic(part, daily, market)
    full_sign = np.sign(ic_mean) if np.isfinite(ic_mean) and ic_mean != 0 else np.nan
    target = expected if expected is not None else full_sign
    year_judgeable, year_match = _matched_years(part, daily, year_values, market, target, int(rule["min_days_per_year"]))
    if market == "commodity":
        sector_judgeable, sector_match = _matched_sectors(
            sector_ic, target, int(rule["min_days_per_sector"]))
        sector_ok = sector_judgeable >= rule["min_sectors_same_sign"] and sector_match >= rule["min_sectors_same_sign"]
    else:
        sector_judgeable = sector_match = np.nan
        sector_ok = True
    controlled_aligns = np.isfinite(controlled_coef) and np.isfinite(target) and controlled_coef * target > 0
    t_ok = bool(abs(ic_t) > rule["abs_t"] and abs(controlled_t) > rule["abs_t"])
    year_ok = year_judgeable >= rule["min_years_same_sign"] and year_match >= rule["min_years_same_sign"]
    meets = bool(t_ok and year_ok and sector_ok and controlled_aligns and np.isfinite(full_sign))
    positive = 0
    symbols = 0
    for _, group in part.groupby("symbol"):
        if len(group) < int(rule["min_symbol_obs"]):
            continue
        symbols += 1
        relation = _correlation(group.value, group.gross)
        if np.isfinite(relation) and relation > 0:
            positive += 1
    return {
        "市场": market, "决策": decision, "因子": factor, "品种": symbol,
        "预期符号": expected if expected is not None else "不预设",
        "全样本方向": full_sign, "IC或系数": ic_mean, "单因子t": ic_t,
        "控制后系数": controlled_coef, "控制后t": controlled_t,
        "年份可判": year_judgeable, "年份同号": year_match,
        "板块可判": sector_judgeable, "板块同号": sector_match,
        "毛价差": gross_q, "净价差": net_q, "每边打平成本": break_even, "价差天数": q_days,
        "品种数": symbols, "品种正相关数": positive, "观测数": int(len(part)),
        "达到统计门槛": meets, "允许配对": False,
    }


def summarize(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    rule = config["pass"]
    rows = []
    for (market, decision, factor), part in panel.groupby(["market", "decision", "factor"], sort=True):
        expected = part.expected.dropna().iloc[0] if part.expected.notna().any() else None
        if market == "commodity":
            rows.append(screen_row(part, market, "截面", decision, factor, expected, rule))
        else:
            for symbol, group in part.groupby("symbol", sort=True):
                rows.append(screen_row(group, market, symbol, decision, factor, expected, rule))
    return pd.DataFrame(rows)


def path_diagnostics(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """隔时段收益拆开后的截面或时间序列相关。只描述路径，不参与门槛。"""
    lags = int(config["pass"]["hac_lags"])
    rows = []
    chosen = panel[panel.decision.eq("close")]
    pieces = ["gap_before_entry", "hold_night", "hold_morning", "hold_afternoon", "hold_after_close"]
    for (market, factor), part in chosen.groupby(["market", "factor"], sort=True):
        groups = [("截面", part)] if market == "commodity" else list(part.groupby("symbol"))
        for symbol, group in groups:
            for piece in pieces:
                if market == "commodity":
                    daily = _daily_statistic(
                        group.dropna(subset=[piece]), int(config["pass"]["min_names_ic"]),
                        lambda frame, column=piece: _correlation(frame.value, frame[column]))
                    mean, t_stat = nw_mean(daily.dropna().to_numpy(), lags)
                else:
                    mean = _correlation(group.value, group[piece])
                    _, t_stat = ols_hac(group[piece].to_numpy(), [group.value.to_numpy()], lags)
                rows.append({"市场": market, "品种": symbol, "因子": factor, "路径": piece,
                             "相关或系数": mean, "t": t_stat})
    return pd.DataFrame(rows)


def _fee_lookup(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    table = table[table.trading_date.le(EVAL_END)].copy()
    C.assert_no_holdout_dates(table.trading_date, "时间因子手续费")
    return {symbol: part.sort_values("trading_date").set_index("trading_date")
            for symbol, part in table.groupby("symbol", sort=False)}


def _tick_lookup() -> dict[str, dict[int, float]]:
    ticks = pd.read_csv(C.RESEARCH_OUT_DIR / "tick_size.csv")
    out = {}
    for symbol, part in ticks.groupby("symbol", sort=False):
        out[symbol] = {int(row.year): float(row.tick) for row in part.itertuples(index=False)}
    return out


def _universe(config: dict) -> dict[str, set[str]]:
    pools = json.loads((C.UNIVERSE_DIR / "universe_by_year.json").read_text(encoding="utf-8"))
    return {year: set(names) for year, names in pools.items() if 2016 <= int(year) <= 2021}


def main():
    config = load_config()
    universe = _universe(config)
    fees = _fee_lookup(load_fees("research"))
    ticks = _tick_lookup()
    rows = []
    symbols = sorted({symbol for names in universe.values() for symbol in names})
    for symbol in symbols:
        frame = load_shard(symbol, directory=C.RESEARCH_DIR, columns=[
            "trading_date", "open", "openw", "close", "closew", "highw", "loww", "volume", "total_turnover"])
        built = rows_for_symbol(frame, symbol, "commodity", config, universe, fees, ticks)
        rows.extend(built)
        print(f"commodity {symbol}: {len(built)}", flush=True)
    for symbol in ("IF", "IH", "IC"):
        frame = load_shard(symbol, directory=C.INDEX_RESEARCH_DIR, columns=[
            "trading_date", "open", "openw", "close", "closew", "highw", "loww", "volume", "total_turnover"])
        built = rows_for_symbol(frame, symbol, "index", config)
        rows.extend(built)
        print(f"index {symbol}: {len(built)}", flush=True)
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError("没有产出任何研究期观测")
    C.assert_no_holdout_dates(panel.trading_date, "时间因子结果")
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT / "panel.parquet", index=False)
    summary = summarize(panel, config)
    paths = path_diagnostics(panel, config)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    paths.to_csv(OUT / "path_diagnostics.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    passed = summary[summary["达到统计门槛"]]
    print(summary.to_string(index=False), flush=True)
    print(f"达到统计门槛: {len(passed)} / {len(summary)}", flush=True)
    print("本轮不构造配对，成交量时戳即使稳定也不写入方向。", flush=True)
    print(f"结果：{OUT}", flush=True)


if __name__ == "__main__":
    main()
