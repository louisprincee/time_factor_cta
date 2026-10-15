"""Test fixed-cutoff impulse confirmation strategies and incremental timing factors.

Uses the existing 09:30 commodity and 09:40 index cutoff features only.
All inputs are limited to the 2016-2021 research partition; index results are
gross because no independently verified historical index execution-cost model
is available in this repository.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from tfcta import config as C
from timing_cutoff_scan import HAC_LAGS, LEAD_END_YEAR, STUDY_END, nw_mean

INPUT_DIR = ROOT / "runs" / "research" / "timing_cutoff_scan"
OUT = ROOT / "runs" / "research" / "impulse_confirmation"
COMMODITY_CUTOFF = "09:30"
INDEX_CUTOFF = "09:40"
EXIT = "11:30"
MIN_CROSS_SECTION = 8
MIN_REGRESSION_CROSS_SECTION = 12
COMMODITY_FACTORS = ("clock_diff", "path_efficiency", "volume_peak_clock")


def _validate_research_dates(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "trading_date" not in frame:
        raise KeyError(f"{label} is missing trading_date")
    frame = frame.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(frame["trading_date"], label)
    if frame["trading_date"].gt(STUDY_END).any():
        raise C.HoldoutViolation(f"{label} exceeds the frozen 2021 research boundary")
    return frame


def _periods(frame: pd.DataFrame):
    year = frame.trading_date.dt.year
    yield "lead_2016_2018", frame.loc[year.le(LEAD_END_YEAR)]
    yield "forward_2019_2021", frame.loc[year.gt(LEAD_END_YEAR)]
    yield "all_2016_2021", frame
    for value in range(2016, 2022):
        yield str(value), frame.loc[year.eq(value)]


def commodity_strategy_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Equal-notional daily strategy returns; flat names remain in the denominator."""
    required = {
        "trading_date", "symbol", "open_move", "tail_move_5m",
        f"gross_{EXIT}", f"cost_{EXIT}",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"commodity feature table is missing {missing}")

    valid = frame.dropna(subset=[
        "open_move", "tail_move_5m", f"gross_{EXIT}", f"cost_{EXIT}"
    ]).copy()
    valid["confirmed"] = (
        np.sign(valid.open_move) * np.sign(valid.tail_move_5m) > 0
    )
    valid["direction"] = np.sign(valid.open_move)
    valid["conditional_direction"] = np.where(
        valid.confirmed, valid.direction, -valid.direction
    )
    signals = {
        "open_direction_momentum": valid.direction,
        "open_direction_reversal": -valid.direction,
        "confirmed_momentum_unconfirmed_reversal": valid.conditional_direction,
    }
    rows = []
    for strategy, direction in signals.items():
        active = direction.ne(0)
        gross_pnl = direction * valid[f"gross_{EXIT}"]
        net_pnl = gross_pnl - active.astype(float) * valid[f"cost_{EXIT}"]
        part = valid.assign(
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            active=active.astype(float),
        )
        for day, group in part.groupby("trading_date", sort=True):
            if len(group) < MIN_CROSS_SECTION:
                continue
            rows.append({
                "trading_date": day,
                "strategy": strategy,
                "gross_return": float(group.gross_pnl.mean()),
                "net_return": float(group.net_pnl.mean()),
                "active_fraction": float(group.active.mean()),
                "n_names": len(group),
                "confirmed_fraction": float(group.confirmed.mean()),
            })
    return pd.DataFrame(rows)


def commodity_factor_state_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily Q5-Q1 factor spreads, separately for confirmed and unconfirmed impulses."""
    required = {
        "trading_date", "open_move", "tail_move_5m", f"gross_{EXIT}", f"cost_{EXIT}",
        *COMMODITY_FACTORS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"commodity feature table is missing {missing}")
    valid = frame.dropna(subset=[
        "open_move", "tail_move_5m", f"gross_{EXIT}", f"cost_{EXIT}",
        *COMMODITY_FACTORS,
    ]).copy()
    valid["confirmation_state"] = np.where(
        np.sign(valid.open_move) * np.sign(valid.tail_move_5m) > 0,
        "confirmed",
        "unconfirmed",
    )
    rows = []
    for (day, state), group in valid.groupby(
        ["trading_date", "confirmation_state"], sort=True
    ):
        if len(group) < MIN_CROSS_SECTION:
            continue
        for factor in COMMODITY_FACTORS:
            if group[factor].nunique() < 5:
                continue
            rank = group[factor].rank(method="first")
            bucket = pd.qcut(rank, 5, labels=False)
            top = group.loc[bucket.eq(4)]
            bottom = group.loc[bucket.eq(0)]
            gross_spread = float(
                top[f"gross_{EXIT}"].mean() - bottom[f"gross_{EXIT}"].mean()
            )
            net_spread = gross_spread - float(
                top[f"cost_{EXIT}"].mean() + bottom[f"cost_{EXIT}"].mean()
            )
            rows.append({
                "trading_date": day,
                "confirmation_state": state,
                "factor": factor,
                "gross_q5_q1": gross_spread,
                "net_q5_q1": net_spread,
                "n_names": len(group),
            })
    return pd.DataFrame(rows)


def _standardize(values: pd.Series) -> np.ndarray | None:
    array = values.to_numpy(dtype="float64")
    scale = float(np.std(array))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return None
    return (array - float(np.mean(array))) / scale


def _daily_design(frame: pd.DataFrame, *, index: bool) -> tuple[np.ndarray, list[str]] | None:
    confirmation = (
        np.sign(frame.gap_return) * np.sign(frame.open_move) > 0
        if index
        else np.sign(frame.open_move) * np.sign(frame.tail_move_5m) > 0
    )
    state = 2.0 * confirmation.to_numpy(dtype="float64") - 1.0
    predictors: dict[str, np.ndarray] = {}
    if index:
        predictors["gap_abs_z"] = _standardize(frame.gap_return.abs())
        predictors["open_move_aligned_z"] = _standardize(
            np.sign(frame.gap_return) * frame.open_move
        )
    else:
        predictors["open_move_z"] = _standardize(frame.open_move)
        predictors["tail_move_5m_z"] = _standardize(frame.tail_move_5m)
    for factor in COMMODITY_FACTORS:
        predictors[f"{factor}_z"] = _standardize(frame[factor])
    if any(value is None for value in predictors.values()):
        return None

    names = ["intercept", *predictors, "confirmation_centered"]
    columns = [np.ones(len(frame))]
    columns.extend(predictors.values())
    columns.append(state)
    for factor in COMMODITY_FACTORS:
        factor_z = predictors[f"{factor}_z"]
        names.append(f"{factor}_x_confirmation")
        columns.append(factor_z * state)
    return np.column_stack(columns), names


def commodity_incremental_coefficients(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional slopes for timing features after price controls."""
    required = {
        "trading_date", "open_move", "tail_move_5m", f"gross_{EXIT}",
        *COMMODITY_FACTORS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"commodity regression input is missing {missing}")
    valid = frame.dropna(subset=[
        "open_move", "tail_move_5m", f"gross_{EXIT}", *COMMODITY_FACTORS
    ]).copy()
    rows = []
    for day, group in valid.groupby("trading_date", sort=True):
        if len(group) < MIN_REGRESSION_CROSS_SECTION:
            continue
        design = _daily_design(group, index=False)
        if design is None:
            continue
        matrix, names = design
        outcome = group[f"gross_{EXIT}"].to_numpy(dtype="float64")
        beta, _, rank, _ = np.linalg.lstsq(matrix, outcome, rcond=None)
        if rank < matrix.shape[1]:
            continue
        row = {"trading_date": day, "n_names": len(group)}
        row.update(dict(zip(names, beta, strict=True)))
        for factor in COMMODITY_FACTORS:
            base = row[f"{factor}_z"]
            interaction = row[f"{factor}_x_confirmation"]
            row[f"{factor}_unconfirmed_slope"] = base - interaction
            row[f"{factor}_confirmed_slope"] = base + interaction
        rows.append(row)
    return pd.DataFrame(rows)


def _hac_ols(
    matrix: np.ndarray, outcome: np.ndarray, max_lags: int = HAC_LAGS
) -> tuple[np.ndarray, np.ndarray] | None:
    beta, _, rank, _ = np.linalg.lstsq(matrix, outcome, rcond=None)
    if rank < matrix.shape[1] or len(outcome) <= matrix.shape[1]:
        return None
    residual = outcome - matrix @ beta
    scores = matrix * residual[:, None]
    meat = scores.T @ scores
    lag_count = min(max_lags, len(outcome) - 1)
    for lag in range(1, lag_count + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        weight = 1.0 - lag / (lag_count + 1.0)
        meat += weight * (gamma + gamma.T)
    bread = np.linalg.inv(matrix.T @ matrix)
    covariance = bread @ meat @ bread
    return beta, covariance


def commodity_incremental_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    coefficient_names = [
        "open_move_z",
        "tail_move_5m_z",
        "confirmation_centered",
        *[
            f"{factor}_{state}_slope"
            for factor in COMMODITY_FACTORS
            for state in ("unconfirmed", "confirmed")
        ],
    ]
    for period, sample in _periods(daily):
        if sample.empty:
            continue
        for coefficient in coefficient_names:
            mean, t_stat = nw_mean(sample[coefficient].to_numpy(), HAC_LAGS)
            rows.append({
                "market": "commodity",
                "coefficient": coefficient,
                "period": period,
                "days": len(sample),
                "mean_bp_per_1sd": mean * 1e4,
                "nw_t_lag5": t_stat,
            })
    return pd.DataFrame(rows)


def index_strategy_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily gap strategies; execution costs are intentionally not imputed."""
    required = {"trading_date", "symbol", "gap_return", "open_move", f"gross_{EXIT}"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"index feature table is missing {missing}")
    valid = frame.dropna(subset=["gap_return", "open_move", f"gross_{EXIT}"]).copy()
    gap_direction = np.sign(valid.gap_return)
    confirmed = gap_direction * np.sign(valid.open_move) > 0
    valid["confirmed"] = confirmed
    positions = {
        "gap_direction": gap_direction,
        "confirmed_only": gap_direction.where(confirmed, 0.0),
        "unconfirmed_gap_direction": gap_direction.where(~confirmed, 0.0),
        "fade_unconfirmed": (-gap_direction).where(~confirmed, 0.0),
    }
    rows = []
    for symbol, part in valid.groupby("symbol", sort=True):
        for strategy, position in positions.items():
            symbol_position = position.loc[part.index]
            for day, group in part.groupby("trading_date", sort=True):
                daily_position = symbol_position.loc[group.index]
                rows.append({
                    "trading_date": day,
                    "symbol": symbol,
                    "strategy": strategy,
                    "gross_return": float(
                        (daily_position * group[f"gross_{EXIT}"]).mean()
                    ),
                    "active_fraction": float(
                        daily_position.ne(0).mean()
                    ),
                    "confirmed_fraction": float(group.confirmed.mean()),
                    "n_names": len(group),
                })
    return pd.DataFrame(rows)


def index_incremental_regression_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-contract HAC regressions: timing features conditional on gap and open move."""
    required = {
        "trading_date", "symbol", "gap_return", "open_move", f"gross_{EXIT}",
        *COMMODITY_FACTORS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"index regression input is missing {missing}")
    valid = frame.dropna(subset=[
        "gap_return", "open_move", f"gross_{EXIT}", *COMMODITY_FACTORS
    ]).sort_values(["symbol", "trading_date"]).copy()
    rows = []
    for symbol, symbol_frame in valid.groupby("symbol", sort=True):
        for period, sample in _periods(symbol_frame):
            if len(sample) < 30:
                continue
            design = _daily_design(sample, index=True)
            if design is None:
                continue
            matrix, names = design
            outcome = (
                np.sign(sample.gap_return).to_numpy(dtype="float64")
                * sample[f"gross_{EXIT}"].to_numpy(dtype="float64")
            )
            fitted = _hac_ols(matrix, outcome)
            if fitted is None:
                continue
            beta, covariance = fitted
            positions = {name: index for index, name in enumerate(names)}
            estimates = [
                ("gap_abs_z", positions["gap_abs_z"], 1.0),
                ("open_move_aligned_z", positions["open_move_aligned_z"], 1.0),
                ("confirmation_centered", positions["confirmation_centered"], 1.0),
            ]
            for factor in COMMODITY_FACTORS:
                base = positions[f"{factor}_z"]
                interaction = positions[f"{factor}_x_confirmation"]
                for state, sign in (("unconfirmed", -1.0), ("confirmed", 1.0)):
                    weights = np.zeros(len(beta))
                    weights[base] = 1.0
                    weights[interaction] = sign
                    estimate = float(weights @ beta)
                    variance = float(weights @ covariance @ weights)
                    t_stat = estimate / np.sqrt(variance) if variance > 0 else np.nan
                    rows.append({
                        "market": "index",
                        "symbol": symbol,
                        "coefficient": f"{factor}_{state}_slope",
                        "period": period,
                        "days": len(sample),
                        "mean_bp_per_1sd": estimate * 1e4,
                        "nw_t_lag5": t_stat,
                    })
            for coefficient, position, _ in estimates:
                variance = float(covariance[position, position])
                t_stat = (
                    float(beta[position] / np.sqrt(variance))
                    if variance > 0 else np.nan
                )
                rows.append({
                    "market": "index",
                    "symbol": symbol,
                    "coefficient": coefficient,
                    "period": period,
                    "days": len(sample),
                    "mean_bp_per_1sd": float(beta[position] * 1e4),
                    "nw_t_lag5": t_stat,
                })
    return pd.DataFrame(rows)


def summarize_factor_states(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state, factor), part in daily.groupby(
        ["confirmation_state", "factor"], sort=True
    ):
        for period, sample in _periods(part):
            if sample.empty:
                continue
            mean_gross, gross_t = nw_mean(sample.gross_q5_q1.to_numpy(), HAC_LAGS)
            mean_net, net_t = nw_mean(sample.net_q5_q1.to_numpy(), HAC_LAGS)
            rows.append({
                "market": "commodity",
                "confirmation_state": state,
                "factor": factor,
                "period": period,
                "days": len(sample),
                "mean_names": float(sample.n_names.mean()),
                "mean_gross_q5_q1_bp": mean_gross * 1e4,
                "gross_nw_t_lag5": gross_t,
                "mean_net_q5_q1_bp": mean_net * 1e4,
                "net_nw_t_lag5": net_t,
                "positive_net_days": int(sample.net_q5_q1.gt(0).sum()),
            })
    return pd.DataFrame(rows)


def summarize_strategies(
    commodity_daily: pd.DataFrame, index_daily: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for strategy, part in commodity_daily.groupby("strategy", sort=True):
        for period, sample in _periods(part):
            if sample.empty:
                continue
            mean_net, t_stat = nw_mean(sample.net_return.to_numpy(), HAC_LAGS)
            rows.append({
                "market": "commodity",
                "symbol": "cross_section",
                "strategy": strategy,
                "period": period,
                "days": len(sample),
                "active_fraction": float(sample.active_fraction.mean()),
                "mean_gross_bp": float(sample.gross_return.mean() * 1e4),
                "mean_net_bp": mean_net * 1e4,
                "nw_t_lag5": t_stat,
                "mean_names": float(sample.n_names.mean()),
                "break_even_round_trip_cost_bp_per_active_name": np.nan,
            })
    for (symbol, strategy), part in index_daily.groupby(
        ["symbol", "strategy"], sort=True
    ):
        for period, sample in _periods(part):
            if sample.empty:
                continue
            mean_gross, t_stat = nw_mean(sample.gross_return.to_numpy(), HAC_LAGS)
            active_fraction = float(sample.active_fraction.mean())
            break_even = (
                mean_gross / active_fraction * 1e4 if active_fraction > 0 else np.nan
            )
            rows.append({
                "market": "index",
                "symbol": symbol,
                "strategy": strategy,
                "period": period,
                "days": len(sample),
                "active_fraction": active_fraction,
                "mean_gross_bp": mean_gross * 1e4,
                "mean_net_bp": np.nan,
                "nw_t_lag5": t_stat,
                "mean_names": 1.0,
                "break_even_round_trip_cost_bp_per_active_name": break_even,
            })
    return pd.DataFrame(rows)


def load_fixed_cutoff_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    commodity_path = INPUT_DIR / "commodity_cutoff_features.csv"
    index_path = INPUT_DIR / "index_cutoff_features.csv"
    for path in (commodity_path, index_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cutoff features at {path}; run timing_cutoff_scan.py first."
            )
    commodity = pd.read_csv(commodity_path)
    index = pd.read_csv(index_path)
    commodity = _validate_research_dates(commodity, "commodity confirmation inputs")
    index = _validate_research_dates(index, "index confirmation inputs")
    commodity = commodity.loc[commodity.cutoff.eq(COMMODITY_CUTOFF)].copy()
    index = index.loc[index.cutoff.eq(INDEX_CUTOFF)].copy()
    if commodity.empty or index.empty:
        raise ValueError("Fixed-cutoff feature inputs are empty")
    return commodity, index


def main() -> None:
    commodity, index = load_fixed_cutoff_inputs()
    commodity_strategies = commodity_strategy_daily(commodity)
    commodity_factor_daily = commodity_factor_state_daily(commodity)
    commodity_regression_daily = commodity_incremental_coefficients(commodity)
    index_strategies = index_strategy_daily(index)
    index_regression_summary = index_incremental_regression_summary(index)
    strategy_summary = summarize_strategies(commodity_strategies, index_strategies)
    factor_summary = summarize_factor_states(commodity_factor_daily)
    commodity_regression_summary = commodity_incremental_summary(
        commodity_regression_daily
    )

    protocol = {
        "status": "fixed_cutoff_confirmation_test_not_a_frozen_or_deployable_strategy",
        "sample": "2016-01-01 through 2021-12-31 research shards only; no 2022+ input",
        "commodity": {
            "decision": "09:30 close",
            "entry": "09:31 open",
            "exit": "11:30 close",
            "confirmation": "sign(09:01-open-to-09:30-close return) equals sign(09:25-close-to-09:30-close return); zero is unconfirmed",
            "strategy_comparison": [
                "open-direction momentum",
                "open-direction reversal",
                "momentum when confirmed and reversal otherwise",
            ],
            "time_factor_increment": "daily Q5-Q1 of clock_diff, path_efficiency, and volume_peak_clock separately within confirmed/unconfirmed cross sections; each factor sort requires at least 8 eligible instruments and 5 distinct factor values",
            "incremental_model": "daily cross-sectional regression of next-window return on standardized open and tail-five-minute moves, confirmation, standardized timing/path factors, and factor-by-confirmation interactions; report Fama-MacBeth means with Newey-West 5-lag t statistics",
            "portfolio": "equal notional per eligible instrument; flat signals retain zero return in the daily denominator; commodity costs are existing commission plus two one-tick slippage estimates",
        },
        "index": {
            "decision": "09:40 close",
            "entry": "09:41 open",
            "exit": "11:30 close",
            "confirmation": "sign(overnight gap) equals sign(09:31-open-to-09:40-close return); zero is unconfirmed",
            "strategies": [
                "always follow gap",
                "follow gap only when confirmed",
                "follow gap only when unconfirmed",
                "fade gap only when unconfirmed",
            ],
            "costs": "gross only; the repository has no independently verified historical IF/IH/IC fee/slippage model; report break-even round-trip cost per active trade, not imputed net PnL",
            "incremental_model": "within each contract and time period, regress gap-direction-adjusted next-window return on standardized gap magnitude, gap-aligned open move, confirmation, standardized early time/path factors, and factor-by-confirmation interactions; use Newey-West HAC standard errors with 5 lags",
        },
        "statistics": "Daily Newey-West mean t-statistics with 5 lags; forward_2019_2021 is an exploratory chronological check, not a blind holdout",
        "multiple_testing": "All specified states and factors are reported; no winner is selected from these results and no multiplicity correction is applied",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    commodity_strategies.to_csv(OUT / "commodity_strategy_daily.csv", index=False)
    commodity_factor_daily.to_csv(OUT / "commodity_factor_state_daily.csv", index=False)
    commodity_regression_daily.to_csv(
        OUT / "commodity_incremental_regression_daily.csv", index=False
    )
    index_strategies.to_csv(OUT / "index_strategy_daily.csv", index=False)
    strategy_summary.to_csv(OUT / "strategy_summary.csv", index=False)
    factor_summary.to_csv(OUT / "commodity_factor_state_summary.csv", index=False)
    commodity_regression_summary.to_csv(
        OUT / "commodity_incremental_regression_summary.csv", index=False
    )
    index_regression_summary.to_csv(
        OUT / "index_incremental_regression_summary.csv", index=False
    )
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("STRATEGY COMPARISON")
    print(strategy_summary[
        strategy_summary.period.isin(["lead_2016_2018", "forward_2019_2021"])
    ].to_string(index=False))
    print("\nCOMMODITY TIMING/PATH FACTORS BY CONFIRMATION STATE")
    print(factor_summary[
        factor_summary.period.isin(["lead_2016_2018", "forward_2019_2021"])
    ].to_string(index=False))
    print("\nTIME/PATH FACTORS CONDITIONAL ON SIMPLE PRICE CONTROLS")
    print(commodity_regression_summary[
        commodity_regression_summary.period.isin([
            "lead_2016_2018", "forward_2019_2021"
        ])
    ].to_string(index=False))
    print(index_regression_summary[
        index_regression_summary.period.isin([
            "lead_2016_2018", "forward_2019_2021"
        ])
    ].to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
