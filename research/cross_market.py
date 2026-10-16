"""First-pass diagnostics for morning timing and order-flow features.

Commodity decisions use 09:16/09:17; index decisions use 09:46/09:47,
the equivalent 16th minute after the 09:30 cash-session open. Only the
2016-2021 research sample is evaluated. This is feature screening, not a
strategy backtest; no locked or validation partitions are read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tfcta import config as C
from tfcta.data import load_shard

ROOT = Path(__file__).resolve().parents[1]

COMMODITY_FEATURES = ROOT / "runs" / "commodity" / "morning" / "clock_flow" / "features.parquet"
OUT = C.RUNS_DIR / "research" / "morning_first_pass"
EVALUATION_START = pd.Timestamp("2016-01-01")
EVALUATION_END = pd.Timestamp("2021-12-31")
INDEX_SYMBOLS = ("IF", "IH", "IC")
INDEX_WINDOW_START = (9, 31)
INDEX_DECISION = (9, 46)
INDEX_ENTRY = (9, 47)
EXIT = (11, 30)
WINDOW_MINUTES = 16

COMMODITY_VARIABLES = {
    "diff": "clock_diff_full_session",
    "dev15": "dev15",
    "open_volume_surprise": "volume_surprise",
    "impact_per_volume": "impact_per_volume",
    "volume_peak_clock": "volume_peak_clock",
    "push": "clock_aligned_open_push",
}
INDEX_VARIABLES = {
    "early_clock_diff": "early_clock_diff",
    "dev15": "dev15",
    "volume_surprise": "volume_surprise",
    "impact_per_volume": "impact_per_volume",
    "volume_peak_clock": "volume_peak_clock",
    "gap_return": "overnight_gap_return",
    "open_move": "open_to_decision_return",
    "price_duration_mean": "price_duration_mean",
    "volume_duration_mean": "volume_duration_mean",
    "volume_duration_slope": "volume_duration_slope",
}


def _minute_stamp(day: pd.Timestamp, hm: tuple[int, int]) -> pd.Timestamp:
    return day.normalize().replace(hour=hm[0], minute=hm[1])


def _duration(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    out = np.full(values.size, np.nan)
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        prior = np.flatnonzero(np.isfinite(values[:i]) & (np.abs(values[:i] - value) > threshold))
        out[i] = i - prior[-1] if prior.size else i
    return out


def _slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 2:
        return np.nan
    x = np.arange(values.size, dtype="float64")[valid]
    y = np.asarray(values, dtype="float64")[valid]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom > 0 else np.nan


def index_morning_rows(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build 16-minute open features and 09:47-to-11:30 target for one index."""
    required = {"trading_date", "openw", "closew", "highw", "loww", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{symbol} minute shard is missing {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{symbol} minute shard index must be a DatetimeIndex")
    frame = frame.sort_index(kind="mergesort").copy()
    if frame.index.has_duplicates:
        raise ValueError(f"{symbol} minute shard has duplicate timestamps")
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(frame["trading_date"], what=f"{symbol} index morning input")

    daily = []
    for day, part in frame.groupby("trading_date", sort=True):
        day = pd.Timestamp(day).normalize()
        start = _minute_stamp(day, INDEX_WINDOW_START)
        decision = _minute_stamp(day, INDEX_DECISION)
        entry = _minute_stamp(day, INDEX_ENTRY)
        exit_time = _minute_stamp(day, EXIT)
        expected = pd.date_range(start, decision, freq="min")
        window = part.reindex(expected)
        valid_window = (
            len(expected) == WINDOW_MINUTES
            and window[["openw", "closew", "highw", "loww", "volume"]].notna().all().all()
        )
        row = {
            "symbol": symbol,
            "trading_date": day,
            "open_move": np.nan,
            "dev15": np.nan,
            "early_clock_diff": np.nan,
            "volume_16m": np.nan,
            "volume_peak_clock": np.nan,
            "price_duration_mean": np.nan,
            "price_duration_vol": np.nan,
            "volume_duration_mean": np.nan,
            "volume_duration_vol": np.nan,
            "volume_duration_slope": np.nan,
            "gross_1130": np.nan,
            "gap_return": np.nan,
        }
        if valid_window:
            close = window.closew.to_numpy(dtype="float64")
            high = window.highw.to_numpy(dtype="float64")
            low = window.loww.to_numpy(dtype="float64")
            volume = window.volume.to_numpy(dtype="float64")
            p_duration = _duration(close, 3.0)
            v_duration = _duration(volume, 50.0)
            price_sd = float(np.std(p_duration, ddof=1))
            volume_sd = float(np.std(v_duration, ddof=1))
            row.update({
                "open_move": float(close[-1] / window.openw.iloc[0] - 1.0),
                "dev15": float(close[-1] / np.mean(close[:-1]) - 1.0),
                "early_clock_diff": float(
                    np.argmin(low) / (WINDOW_MINUTES - 1)
                    - np.argmax(high) / (WINDOW_MINUTES - 1)
                ),
                "volume_16m": float(np.sum(volume)),
                "volume_peak_clock": float(np.argmax(volume) / (WINDOW_MINUTES - 1)),
                "price_duration_mean": float(np.mean(p_duration)),
                "price_duration_vol": price_sd,
                "volume_duration_mean": float(np.mean(v_duration)),
                "volume_duration_vol": volume_sd,
                "volume_duration_slope": _slope(v_duration),
            })
            if entry in frame.index and exit_time in frame.index:
                entry_open = float(frame.at[entry, "openw"])
                exit_close = float(frame.at[exit_time, "closew"])
                if entry_open > 0 and exit_close > 0:
                    row["gross_1130"] = exit_close / entry_open - 1.0

        if start in part.index and day > frame.trading_date.min():
            prior_days = frame.loc[frame.trading_date < day].groupby("trading_date", sort=True)
            previous_close = float(prior_days.tail(1).closew.iloc[-1]) if len(prior_days) else np.nan
            first_open = float(part.at[start, "openw"])
            if np.isfinite(previous_close) and previous_close > 0 and first_open > 0:
                row["gap_return"] = first_open / previous_close - 1.0
        daily.append(row)

    out = pd.DataFrame(daily).sort_values("trading_date").reset_index(drop=True)
    if "volume_16m" in out:
        baseline = out.volume_16m.rolling(20, min_periods=15).median().shift(1)
        out["volume_surprise"] = out.volume_16m / baseline.where(baseline > 0)
        out["impact_per_volume"] = out.open_move.abs() / np.sqrt(out.volume_surprise)
    return out


def load_index_rows() -> pd.DataFrame:
    parts = []
    for symbol in INDEX_SYMBOLS:
        frame = load_shard(
            symbol,
            directory=C.INDEX_RESEARCH_DIR,
            columns=["trading_date", "openw", "closew", "highw", "loww", "volume"],
        )
        C.assert_no_holdout_dates(frame.trading_date, what=f"{symbol} index morning")
        parts.append(index_morning_rows(frame, symbol))
        print(f"{symbol}: {len(parts[-1])} daily rows", flush=True)
    return pd.concat(parts, ignore_index=True)


def load_commodity_rows() -> pd.DataFrame:
    if not COMMODITY_FEATURES.exists():
        raise FileNotFoundError(
            f"Missing existing commodity feature cache: {COMMODITY_FEATURES}; "
            "build it with python -m research.morning_clock flow first."
        )
    frame = pd.read_parquet(COMMODITY_FEATURES)
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(frame.trading_date, what="commodity morning feature cache")
    frame = frame[
        frame.member.fillna(False)
        & frame.trading_date.between(EVALUATION_START, EVALUATION_END)
    ].copy()
    frame["market"] = "commodity"
    return frame


def _spread_summary(part: pd.DataFrame, factor: str, market: str, symbol: str, period: str) -> dict:
    valid = part[[factor, "gross_1130"]].dropna()
    row = {
        "market": market, "symbol": symbol, "period": period, "factor": factor,
        "n": len(valid), "spearman": np.nan, "top_minus_bottom_gross_bp": np.nan,
        "top_minus_bottom_net_bp": np.nan, "break_even_cost_per_leg_bp": np.nan,
    }
    if len(valid) < 20 or valid[factor].nunique() < 5:
        return row
    row["spearman"] = float(stats.spearmanr(valid[factor], valid.gross_1130).statistic)
    rank = valid[factor].rank(method="first")
    bucket = pd.qcut(rank, 5, labels=False)
    top = valid.loc[bucket.eq(4)]
    bottom = valid.loc[bucket.eq(0)]
    gross_spread = float(top.gross_1130.mean() - bottom.gross_1130.mean())
    row["top_minus_bottom_gross_bp"] = gross_spread * 1e4
    row["break_even_cost_per_leg_bp"] = gross_spread * 1e4 / 2.0
    if "cost_1130" in part:
        cost = part.loc[valid.index, "cost_1130"]
        row["top_minus_bottom_net_bp"] = float(
            top.gross_1130.mean() - bottom.gross_1130.mean()
            - cost.loc[top.index].mean() - cost.loc[bottom.index].mean()
        ) * 1e4
    return row


def association_tables(commodity: pd.DataFrame, index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    annual = []
    for market, frame, variables in (
        ("commodity", commodity, COMMODITY_VARIABLES),
        ("index", index, INDEX_VARIABLES),
    ):
        frame = frame.copy()
        frame["year"] = frame.trading_date.dt.year
        periods = [("2016-2021", frame)]
        periods.extend((str(year), part) for year, part in frame.groupby("year", sort=True))
        for source, factor in variables.items():
            if source not in frame:
                continue
            for symbol, symbol_rows in frame.groupby("symbol", sort=True):
                for period, sample in periods:
                    part = sample[sample.symbol.eq(symbol)]
                    result = _spread_summary(part, source, market, symbol, period)
                    result["factor_label"] = factor
                    results.append(result)
                    if period != "2016-2021":
                        annual.append(result)
    return pd.DataFrame(results), pd.DataFrame(annual)


def cross_section_ic(commodity: pd.DataFrame) -> pd.DataFrame:
    """Daily commodity rank ICs; each date contributes one observation."""
    rows = []
    for source, label in COMMODITY_VARIABLES.items():
        for day, part in commodity.groupby("trading_date", sort=True):
            valid = part[[source, "gross_1130"]].dropna()
            if len(valid) < 8 or valid[source].nunique() < 3:
                continue
            rows.append({
                "trading_date": day, "year": pd.Timestamp(day).year,
                "factor": label, "n_symbols": len(valid),
                "rank_ic": float(stats.spearmanr(valid[source], valid.gross_1130).statistic),
            })
    return pd.DataFrame(rows)


def protocol() -> dict:
    return {
        "status": "exploratory_factor_diagnostics_not_a_strategy",
        "sample": "2016-01-01 through 2021-12-31 only; 2015 index rows used only for trailing volume baseline; no 2022+ data read",
        "commodity": {
            "decision": "09:16 close", "entry": "09:17 open", "exit": "11:30 close",
            "source": "Existing clock_flow research cache, active commodity universe and valid night-session observations only",
            "clock_diff": "Existing low-clock minus high-clock from the trading-day session (can include night session), through 09:16",
        },
        "index": {
            "decision": "09:46 close", "entry": "09:47 open", "exit": "11:30 close",
            "features": "09:31-09:46 inclusive (16 minute bars); trailing volume baseline is prior 20 valid sessions' same-window median, shifted one session, minimum 15",
            "price_duration_threshold_points": 3.0,
            "volume_duration_threshold": 50.0,
            "overnight_gap": "First 09:31 adjusted open divided by prior trading date's final adjusted close minus one",
        },
        "common_features": ["short-term price deviation", "opening volume surprise", "impact per sqrt(volume surprise)", "opening volume peak clock"],
        "target": "Adjusted 09:17/09:47 open to same trading date 11:30 close; missing exact timestamps are excluded",
        "interpretation": "Pooled correlations and full-period quintiles are descriptive discovery statistics, not deployable rules. Commodity top-minus-bottom net subtracts each tail's observed round-trip cost; index figures are gross and report the per-leg break-even transaction cost.",
        "known_noncomparability": "Commodity legacy clock_diff includes the night session; index early_clock_diff uses only 09:31-09:46. Compare mechanisms, not raw coefficients.",
    }


def main() -> None:
    commodity = load_commodity_rows()
    index = load_index_rows()
    index = index[
        index.trading_date.between(EVALUATION_START, EVALUATION_END)
    ].copy()
    index["market"] = "index"

    commodity_rows = commodity[["symbol", "trading_date", "gross_1130", "cost_1130", *COMMODITY_VARIABLES]].copy()
    index_rows = index[["symbol", "trading_date", "gross_1130", *INDEX_VARIABLES]].copy()
    associations, annual = association_tables(commodity_rows, index_rows)
    csic = cross_section_ic(commodity_rows)
    if csic.empty:
        cs_summary = pd.DataFrame(columns=["factor", "year", "days", "mean_rank_ic", "daily_sd", "descriptive_t"])
    else:
        cs_summary = (
            csic.groupby(["factor", "year"], as_index=False)
            .agg(days=("rank_ic", "size"), mean_rank_ic=("rank_ic", "mean"), daily_sd=("rank_ic", "std"))
        )
        cs_summary["descriptive_t"] = cs_summary.mean_rank_ic / (
            cs_summary.daily_sd / np.sqrt(cs_summary.days)
        )

    OUT.mkdir(parents=True, exist_ok=True)
    commodity_rows.to_csv(OUT / "commodity_features.csv", index=False)
    index_rows.to_csv(OUT / "index_features.csv", index=False)
    associations.to_csv(OUT / "factor_diagnostics.csv", index=False)
    annual.to_csv(OUT / "annual_diagnostics.csv", index=False)
    csic.to_csv(OUT / "commodity_daily_rank_ic.csv", index=False)
    cs_summary.to_csv(OUT / "commodity_rank_ic_summary.csv", index=False)
    (OUT / "protocol.json").write_text(
        json.dumps(protocol(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"commodity observations: {len(commodity_rows)}", flush=True)
    print(f"index observations: {len(index_rows)}", flush=True)
    print(f"outputs: {OUT}", flush=True)
    full_period = associations[associations.period.eq("2016-2021")]
    summary = (
        full_period.groupby(["market", "factor_label"])
        .agg(
            symbols=("symbol", "nunique"),
            observations=("n", "sum"),
            mean_symbol_spearman=("spearman", "mean"),
            positive_symbols=("spearman", lambda values: int((values > 0).sum())),
            median_gross_q5_q1_bp=("top_minus_bottom_gross_bp", "median"),
            median_net_q5_q1_bp=("top_minus_bottom_net_bp", "median"),
            median_gross_break_even_cost_per_leg_bp=("break_even_cost_per_leg_bp", "median"),
        )
        .round(3)
    )
    print(
        summary.to_string(),
        flush=True,
    )
    print(cs_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
