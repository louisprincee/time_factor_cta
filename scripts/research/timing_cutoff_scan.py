"""Compare predeclared early-session decision cutoffs for commodity and index futures.

Commodity cutoffs are 09:10/09:15/09:20/09:30; index cutoffs are the
equivalent 5/10/15/20-minute observations after the stored 09:31 open.
Signals use only bars through the cutoff; execution is at the next minute open.
The 2016-2018 lead period chooses a time only for a 2019-2021 forward check.
No validation or locked partition is opened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tfcta import config as C
from tfcta.data import find_shard, load_shard
from tfcta.research import MULTIPLIER, commission_rate

COMMODITY_SHARDS = ROOT / "data" / "minute_shards" / "research"
INDEX_SHARDS = C.INDEX_RESEARCH_DIR
UNIVERSE_FILE = ROOT / "data" / "universe" / "universe_by_year.json"
FEE_FILE = ROOT / "data" / "research" / "fee_history.csv"
TICK_FILE = ROOT / "data" / "research" / "tick_size.csv"
OUT = ROOT / "runs" / "research" / "timing_cutoff_scan"
STUDY_START = pd.Timestamp("2016-01-01")
STUDY_END = pd.Timestamp("2021-12-31")
LEAD_END_YEAR = 2018
COMMODITY_CUTOFFS = ("09:10", "09:15", "09:20", "09:30")
INDEX_CUTOFFS = ("09:35", "09:40", "09:45", "09:50")
COMMODITY_EXITS = ("11:30", "15:00")
INDEX_EXITS = ("11:30",)
MINUTE_NS = 60_000_000_000
HAC_LAGS = 5


def _at(day: pd.Timestamp, hm: str) -> pd.Timestamp:
    hour, minute = map(int, hm.split(":"))
    return pd.Timestamp(day.year, day.month, day.day, hour, minute)


def path_features(window: pd.DataFrame) -> dict[str, float]:
    """Compute factors solely from a complete contiguous price/volume window."""
    window = window.rename(columns={"highw": "high", "loww": "low"})
    empty = {
        "open_move": np.nan,
        "tail_move_5m": np.nan,
        "clock_diff": np.nan,
        "path_efficiency": np.nan,
        "volume_peak_clock": np.nan,
    }
    if len(window) < 2 or not isinstance(window.index, pd.DatetimeIndex):
        return empty
    if window.index.has_duplicates or not (np.diff(window.index.asi8) == MINUTE_NS).all():
        return empty
    columns = ["open", "close", "high", "low", "volume"]
    if window[columns].isna().any().any():
        return empty
    opened = window.open.to_numpy(dtype="float64")
    close = window.close.to_numpy(dtype="float64")
    high = window.high.to_numpy(dtype="float64")
    low = window.low.to_numpy(dtype="float64")
    volume = window.volume.to_numpy(dtype="float64")
    if opened[0] <= 0 or not np.isfinite(np.r_[opened, close, high, low, volume]).all():
        return empty
    steps = abs(close[0] - opened[0]) + np.abs(np.diff(close)).sum()
    path_efficiency = abs(close[-1] - opened[0]) / steps if steps > 0 else np.nan
    n = len(window)
    return {
        "open_move": float(close[-1] / opened[0] - 1.0),
        "tail_move_5m": float(close[-1] / close[-6] - 1.0) if n >= 6 else np.nan,
        "clock_diff": float((np.argmin(low) - np.argmax(high)) / (n - 1)),
        "path_efficiency": float(path_efficiency),
        "volume_peak_clock": float(np.argmax(volume) / (n - 1)),
    }


def _load_fee_lookup() -> tuple[dict[str, pd.DataFrame], dict[str, dict[int, float]]]:
    fees = pd.read_csv(FEE_FILE, parse_dates=["trading_date"])
    fees["trading_date"] = pd.to_datetime(fees.trading_date).dt.normalize()
    C.assert_no_holdout_dates(fees.trading_date, "timing scan fees")
    fees = fees[fees.trading_date.le(STUDY_END)]
    fee_lookup = {
        symbol: part.sort_values("trading_date").set_index("trading_date")
        for symbol, part in fees.groupby("symbol", sort=False)
    }
    ticks = pd.read_csv(TICK_FILE)
    tick_lookup: dict[str, dict[int, float]] = {}
    for symbol, part in ticks.groupby("symbol", sort=False):
        tick_lookup[symbol] = {
            int(row.year): float(row.tick) for row in part.itertuples(index=False)
        }
    return fee_lookup, tick_lookup


def _fee_at(history: pd.DataFrame, day: pd.Timestamp) -> pd.Series | None:
    if history is None:
        return None
    eligible = history.loc[history.index <= day]
    return eligible.iloc[-1] if len(eligible) else None


def _commodity_cost(
    symbol: str,
    day: pd.Timestamp,
    entry_raw: float,
    exit_raw: float,
    fee_history: pd.DataFrame | None,
    tick_history: dict[int, float] | None,
) -> float:
    fee = _fee_at(fee_history, day)
    tick = (tick_history or {}).get(day.year - 1)
    if fee is None or tick is None or entry_raw <= 0:
        return np.nan
    close_today = fee.close_commission_today
    if pd.isna(close_today):
        # This known CZCE data gap is handled conservatively as in the morning runner.
        close_today = fee.close_commission
    mult = MULTIPLIER[symbol]
    open_rate = commission_rate(
        fee.commission_type, fee.open_commission, entry_raw, entry_raw, mult
    )
    close_rate = commission_rate(
        fee.commission_type, close_today, exit_raw, entry_raw, mult
    )
    return float(open_rate + close_rate + 2.0 * tick / entry_raw)


def _minute_frame(symbol: str, directory: Path, columns: list[str]) -> pd.DataFrame:
    frame = load_shard(symbol, directory=directory, columns=columns)
    C.assert_no_holdout_dates(frame.trading_date, f"{symbol} timing scan source")
    frame = frame.sort_index(kind="mergesort").copy()
    frame["trading_date"] = pd.to_datetime(frame.trading_date).dt.normalize()
    return frame.loc[frame.trading_date.between(STUDY_START, STUDY_END)]


def commodity_rows() -> pd.DataFrame:
    pools = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    fees, ticks = _load_fee_lookup()
    symbols = sorted({symbol for members in pools.values() for symbol in members})
    rows: list[dict] = []
    columns = ["open", "close", "highw", "loww", "volume", "trading_date"]
    for symbol in symbols:
        path = find_shard(COMMODITY_SHARDS, symbol)
        if path is None:
            print(f"commodity missing research shard: {symbol}", flush=True)
            continue
        frame = _minute_frame(symbol, COMMODITY_SHARDS, columns)
        fee_history = fees.get(symbol)
        tick_history = ticks.get(symbol)
        for day, part in frame.groupby("trading_date", sort=True):
            day = pd.Timestamp(day).normalize()
            if symbol not in pools.get(str(day.year), []):
                continue
            for cutoff in COMMODITY_CUTOFFS:
                start, decision = _at(day, "09:01"), _at(day, cutoff)
                entry_time = pd.Timestamp(decision.value + MINUTE_NS)
                window = part.reindex(pd.date_range(start, decision, freq="min"))
                factors = path_features(window)
                row = {
                    "market": "commodity", "symbol": symbol, "trading_date": day,
                    "cutoff": cutoff, "decision_minutes": int(
                        (decision.value - start.value) / MINUTE_NS + 1
                    ),
                    **factors,
                }
                if not np.isfinite(factors["open_move"]) or entry_time not in part.index:
                    rows.append(row)
                    continue
                entry_raw = float(part.at[entry_time, "open"])
                for exit_time in COMMODITY_EXITS:
                    target_time = _at(day, exit_time)
                    if target_time not in part.index or entry_raw <= 0:
                        row[f"gross_{exit_time}"] = np.nan
                        row[f"cost_{exit_time}"] = np.nan
                        row[f"net_{exit_time}"] = np.nan
                        continue
                    exit_raw = float(part.at[target_time, "close"])
                    gross = exit_raw / entry_raw - 1.0 if exit_raw > 0 else np.nan
                    cost = _commodity_cost(
                        symbol, day, entry_raw, exit_raw, fee_history, tick_history
                    )
                    row[f"gross_{exit_time}"] = gross
                    row[f"cost_{exit_time}"] = cost
                    row[f"net_{exit_time}"] = gross - cost if np.isfinite(cost) else np.nan
                rows.append(row)
        print(f"commodity {symbol}: processed", flush=True)
    result = pd.DataFrame(rows)
    if len(result) and result.trading_date.dt.year.max() > 2021:
        raise C.HoldoutViolation("Commodity cutoff scan exceeded 2021")
    return result


def index_rows() -> pd.DataFrame:
    rows: list[dict] = []
    columns = ["openw", "closew", "highw", "loww", "volume", "trading_date"]
    for symbol in ("IF", "IH", "IC"):
        frame = _minute_frame(symbol, INDEX_SHARDS, columns)
        daily_closes = frame.groupby("trading_date", sort=True).tail(1).set_index("trading_date").closew
        for day, part in frame.groupby("trading_date", sort=True):
            day = pd.Timestamp(day).normalize()
            start = _at(day, "09:31")
            if start not in part.index:
                continue
            prior = daily_closes.loc[daily_closes.index < day]
            prior_close = float(prior.iloc[-1]) if len(prior) else np.nan
            gap = float(part.at[start, "openw"] / prior_close - 1.0) if prior_close > 0 else np.nan
            for cutoff in INDEX_CUTOFFS:
                decision = _at(day, cutoff)
                entry_time = pd.Timestamp(decision.value + MINUTE_NS)
                window = part.reindex(pd.date_range(start, decision, freq="min"))
                factors = path_features(window.rename(columns={
                    "openw": "open", "closew": "close"
                }))
                row = {
                    "market": "index", "symbol": symbol, "trading_date": day,
                    "cutoff": cutoff, "decision_minutes": int(
                        (decision - start).total_seconds() / 60 + 1
                    ),
                    "gap_return": gap,
                    **factors,
                }
                exit_time = _at(day, "11:30")
                if entry_time in part.index and exit_time in part.index:
                    entry = float(part.at[entry_time, "openw"])
                    exit_price = float(part.at[exit_time, "closew"])
                    row["gross_11:30"] = exit_price / entry - 1.0 if entry > 0 else np.nan
                else:
                    row["gross_11:30"] = np.nan
                rows.append(row)
        print(f"index {symbol}: processed", flush=True)
    result = pd.DataFrame(rows)
    if len(result) and result.trading_date.dt.year.max() > 2021:
        raise C.HoldoutViolation("Index cutoff scan exceeded 2021")
    return result


def _daily_cross_section(part: pd.DataFrame, factor: str, outcome: str, cost: str | None) -> pd.DataFrame:
    rows = []
    for day, group in part.dropna(subset=[factor, outcome]).groupby("trading_date", sort=True):
        if len(group) < 8 or group[factor].nunique() < 5:
            continue
        rank = group[factor].rank(method="first")
        bucket = pd.qcut(rank, 5, labels=False)
        top, bottom = group.loc[bucket.eq(4)], group.loc[bucket.eq(0)]
        gross_spread = float(top[outcome].mean() - bottom[outcome].mean())
        net_spread = gross_spread
        if cost:
            costs = group[cost]
            if not costs.loc[top.index].notna().all() or not costs.loc[bottom.index].notna().all():
                continue
            net_spread -= float(costs.loc[top.index].mean() + costs.loc[bottom.index].mean())
        rows.append({
            "trading_date": day,
            "rank_ic": float(group[factor].corr(group[outcome], method="spearman")),
            "gross_q5_q1": gross_spread,
            "net_q5_q1": net_spread,
            "n_symbols": len(group),
        })
    return pd.DataFrame(rows)


def commodity_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = ("clock_diff", "open_move", "path_efficiency", "volume_peak_clock")
    rows = []
    daily_rows = []
    for (cutoff, exit_hm), part in frame.groupby(["cutoff", "exit_hm"], sort=True):
        outcome, cost = f"gross_{exit_hm}", f"cost_{exit_hm}"
        for factor in factors:
            daily = _daily_cross_section(part, factor, outcome, cost)
            if daily.empty:
                continue
            daily["cutoff"], daily["exit"], daily["factor"] = cutoff, exit_hm, factor
            daily["year"] = pd.to_datetime(daily.trading_date).dt.year
            daily_rows.append(daily)
            for period, sample in (
                ("lead_2016_2018", daily[daily.year <= LEAD_END_YEAR]),
                ("forward_2019_2021", daily[daily.year > LEAD_END_YEAR]),
                ("all_2016_2021", daily),
                *[(str(y), daily[daily.year.eq(y)]) for y in range(2016, 2022)],
            ):
                if sample.empty:
                    continue
                rows.append({
                    "market": "commodity", "cutoff": cutoff, "exit": exit_hm,
                    "factor": factor, "period": period, "days": len(sample),
                    "mean_daily_rank_ic": float(sample.rank_ic.mean()),
                    "mean_gross_q5_q1_bp": float(sample.gross_q5_q1.mean() * 1e4),
                    "mean_net_q5_q1_bp": float(sample.net_q5_q1.mean() * 1e4),
                    "positive_net_days": int(sample.net_q5_q1.gt(0).sum()),
                    "mean_names": float(sample.n_symbols.mean()),
                })
    return pd.DataFrame(rows), pd.concat(daily_rows, ignore_index=True)


def nw_mean(values: np.ndarray, max_lags: int = HAC_LAGS) -> tuple[float, float]:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return (float(np.mean(values)) if n else np.nan, np.nan)
    centered = values - values.mean()
    long_run_var = float(centered @ centered) / n
    lag_count = min(max_lags, n - 1)
    for lag in range(1, lag_count + 1):
        gamma = float(centered[lag:] @ centered[:-lag]) / n
        long_run_var += 2.0 * (1.0 - lag / (lag_count + 1.0)) * gamma
    se = np.sqrt(max(long_run_var, 0.0) / n)
    return float(values.mean()), float(values.mean() / se) if se > 0 else np.nan


def index_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (symbol, cutoff), part in frame.groupby(["symbol", "cutoff"], sort=True):
        valid = part.dropna(subset=["gap_return", "gross_11:30"]).copy()
        gap_sign = np.sign(valid.gap_return.to_numpy())
        move_sign = np.sign(valid.open_move.to_numpy())
        confirmation = (gap_sign * move_sign > 0)
        strategies = {
            "gap_direction": np.sign(valid.gap_return.to_numpy()) * valid["gross_11:30"].to_numpy(),
            "confirmed_gap_direction": (
                np.sign(valid.gap_return.to_numpy()) * valid["gross_11:30"].to_numpy()
            ) * confirmation,
        }
        for strategy, pnl in strategies.items():
            valid[f"pnl_{strategy}"] = pnl
            for period, sample in (
                ("lead_2016_2018", valid[valid.trading_date.dt.year <= LEAD_END_YEAR]),
                ("forward_2019_2021", valid[valid.trading_date.dt.year > LEAD_END_YEAR]),
                ("all_2016_2021", valid),
                *[(str(y), valid[valid.trading_date.dt.year.eq(y)]) for y in range(2016, 2022)],
            ):
                mean, t_stat = nw_mean(sample[f"pnl_{strategy}"].to_numpy())
                rows.append({
                    "market": "index", "symbol": symbol, "cutoff": cutoff,
                    "strategy": strategy, "period": period, "days": len(sample),
                    "active_fraction": float(
                        (sample[f"pnl_{strategy}"] != 0).mean()
                    ) if len(sample) else np.nan,
                    "mean_directional_return_bp": mean * 1e4,
                    "nw_t_lag5": t_stat,
                })
    return pd.DataFrame(rows)


def main() -> None:
    protocol = {
        "status": "cutoff_comparison_and_chronological_screen_not_frozen_strategy",
        "commodity_source": "data/minute_shards/research only; annual eligible universes from data/universe/universe_by_year.json; fee/tick history from data/research",
        "index_source": "data/index/minute_shards/research only",
        "period": "2016-01-01 through 2021-12-31; lead 2016-2018; forward check 2019-2021; no 2022+ read",
        "commodity_cutoffs": {
            "decision_close": list(COMMODITY_CUTOFFS),
            "entry": "next minute open",
            "exits": list(COMMODITY_EXITS),
            "primary_factor": "low-clock minus high-clock, measured from the 09:01-stamped first bar through cutoff; its open represents the 09:00 session open; positive means the low was later than the high; positive Q5-Q1 future return is consistent with reversal",
            "secondary_factors": ["open_move", "path_efficiency", "volume_peak_clock"],
            "net_spread": "daily cross-sectional top quintile minus bottom quintile return, less the observed round-trip commission and one prior-year tick of slippage per side for both tails",
            "time_selection": "Use only 2016-2018 mean daily net Q5-Q1 for the primary clock factor and 11:30 exit to nominate a cutoff; then report the nominated cutoff on 2019-2021. This screen is not a blind test because the broader period has prior research exposure.",
        },
        "index_cutoffs": {
            "decision_close": list(INDEX_CUTOFFS),
            "entry": "next minute open",
            "exits": list(INDEX_EXITS),
            "primary": "overnight gap direction, with a second variant that holds only when the cutoff-window return confirms the gap direction",
            "fees": "Gross only; this scan does not contain an independently verified time-varying IF/IH/IC fee and slippage model.",
        },
        "shared_features": {
            "clock_diff": "low extreme timestamp minus high extreme timestamp, normalized to the observed window",
            "path_efficiency": "absolute first-open-to-final-close displacement divided by the first open-to-close move plus subsequent absolute minute close changes",
            "limitations": "Minute OHLC does not identify order-book liquidity or informed flow. Factor scan is cross-sectional for commodities and contract-specific time-series for indices. Annual means are descriptive.",
        },
    }
    commodity = commodity_rows()
    index = index_rows()
    for exit_hm in COMMODITY_EXITS:
        if f"gross_{exit_hm}" not in commodity:
            commodity[f"gross_{exit_hm}"] = np.nan
            commodity[f"cost_{exit_hm}"] = np.nan
            commodity[f"net_{exit_hm}"] = np.nan
    commodity_parts = []
    for exit_hm in COMMODITY_EXITS:
        block = commodity.copy()
        block["exit_hm"] = exit_hm
        commodity_parts.append(block)
    commodity_panel = pd.concat(commodity_parts, ignore_index=True)
    commodity_result, commodity_daily = commodity_diagnostics(commodity_panel)
    index_result = index_diagnostics(index)

    OUT.mkdir(parents=True, exist_ok=True)
    commodity.to_csv(OUT / "commodity_cutoff_features.csv", index=False)
    commodity_result.to_csv(OUT / "commodity_cutoff_diagnostics.csv", index=False)
    commodity_daily.to_csv(OUT / "commodity_cutoff_daily.csv", index=False)
    index.to_csv(OUT / "index_cutoff_features.csv", index=False)
    index_result.to_csv(OUT / "index_cutoff_diagnostics.csv", index=False)
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lead = commodity_result[
        commodity_result.period.eq("lead_2016_2018")
        & commodity_result.factor.eq("clock_diff")
        & commodity_result.exit.eq("11:30")
    ].sort_values("mean_net_q5_q1_bp", ascending=False)
    print("COMMODITY CUTOFFS: PRIMARY CLOCK, 11:30, TRAINING PERIOD")
    print(lead.to_string(index=False))
    print("\nCOMMODITY PRIMARY CLOCK: SELECTED CUTOFF FORWARD CHECK")
    if len(lead):
        chosen = lead.iloc[0].cutoff
        forward = commodity_result[
            commodity_result.cutoff.eq(chosen)
            & commodity_result.exit.eq("11:30")
            & commodity_result.factor.eq("clock_diff")
            & commodity_result.period.eq("forward_2019_2021")
        ]
        print(forward.to_string(index=False))
    print("\nINDEX GAP STRATEGY CUTOFFS")
    print(index_result[
        index_result.period.eq("lead_2016_2018")
        & index_result.strategy.eq("gap_direction")
    ].to_string(index=False))
    print("\nINDEX CONFIRMED GAP STRATEGY FORWARD CHECK")
    print(index_result[
        index_result.period.eq("forward_2019_2021")
        & index_result.strategy.eq("confirmed_gap_direction")
    ].to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
