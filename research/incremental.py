"""Conditional-information diagnostics for morning timing factors.

Commodity: test whether early-session clock and dev15 interact in daily
cross-sections after centering ranks within date.
Index: test whether the overnight gap adds information beyond the first
16-minute return within IF/IH/IC, using Newey-West HAC standard errors.
Only the 2016-2021 research-period feature files are read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs" / "research" / "morning_first_pass"
OUT = ROOT / "runs" / "research" / "morning_incremental"
HAC_LAGS = 5


def _rank_by_date(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("trading_date")[column].rank(pct=True) - 0.5


def clustered_ols(
    y: np.ndarray, predictors: dict[str, np.ndarray], groups: np.ndarray
) -> dict[str, float]:
    """OLS with a one-way cluster-robust covariance estimate."""
    y = np.asarray(y, dtype="float64")
    names = list(predictors)
    design = np.column_stack([np.ones(len(y))] + [predictors[name] for name in names])
    if not np.isfinite(design).all() or not np.isfinite(y).all():
        raise ValueError("Cluster regression inputs must be finite")
    bread = np.linalg.pinv(design.T @ design)
    beta = bread @ design.T @ y
    residual = y - design @ beta
    scores = pd.DataFrame(design * residual[:, None])
    scores["_group"] = groups
    grouped = scores.groupby("_group", sort=False).sum().to_numpy()
    n_groups = len(grouped)
    if n_groups < 3:
        raise ValueError(f"At least 3 clusters are required, got {n_groups}")
    covariance = bread @ (grouped.T @ grouped) @ bread
    covariance *= n_groups / (n_groups - 1)
    fitted = design @ beta
    centered = y - y.mean()
    total = float(centered @ centered)
    r_squared = 1.0 - float(residual @ residual) / total if total > 0 else np.nan
    result = {"n": int(len(y)), "clusters": int(n_groups), "r_squared": r_squared}
    for i, name in enumerate(names, start=1):
        se = np.sqrt(max(0.0, float(covariance[i, i])))
        result[f"coef:{name}"] = float(beta[i])
        result[f"t:{name}"] = float(beta[i] / se) if se > 0 else np.nan
    return result


def hac_ols(
    y: np.ndarray, predictors: dict[str, np.ndarray], max_lags: int = HAC_LAGS
) -> dict[str, float]:
    """OLS with Bartlett-kernel Newey-West standard errors on ordered dates."""
    y = np.asarray(y, dtype="float64")
    names = list(predictors)
    design = np.column_stack([np.ones(len(y))] + [predictors[name] for name in names])
    if not np.isfinite(design).all() or not np.isfinite(y).all():
        raise ValueError("HAC regression inputs must be finite")
    n, p = design.shape
    if n <= p or max_lags < 0:
        raise ValueError("Insufficient observations or invalid HAC lag count")
    bread = np.linalg.pinv(design.T @ design)
    beta = bread @ design.T @ y
    scores = design * (y - design @ beta)[:, None]
    meat = scores.T @ scores
    lag_count = min(max_lags, n - 1)
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    covariance *= n / (n - p)
    residual = y - design @ beta
    centered = y - y.mean()
    total = float(centered @ centered)
    r_squared = 1.0 - float(residual @ residual) / total if total > 0 else np.nan
    result = {"n": int(n), "r_squared": r_squared}
    for i, name in enumerate(names, start=1):
        se = np.sqrt(max(0.0, float(covariance[i, i])))
        result[f"coef:{name}"] = float(beta[i])
        result[f"t:{name}"] = float(beta[i] / se) if se > 0 else np.nan
    return result


def _commodity_fit(part: pd.DataFrame, period: str) -> list[dict]:
    required = ["trading_date", "symbol", "gross_1130", "diff", "dev15"]
    data = part.dropna(subset=required).copy()
    data = data[data["diff"] != 0]
    if data.empty:
        return []
    data["y"] = _rank_by_date(data, "gross_1130")
    data["clock"] = _rank_by_date(data, "diff")
    data["dev"] = _rank_by_date(data, "dev15")
    data["clock_x_dev"] = data.clock * data.dev
    data["clock_x_dev_sign"] = np.sign(data.clock) * np.sign(data.dev)
    models = (
        ("dev15_only", {"dev15": data.dev.to_numpy()}),
        ("add_clock", {"dev15": data.dev.to_numpy(), "clock": data.clock.to_numpy()}),
        ("clock_dev_interaction", {
            "dev15": data.dev.to_numpy(),
            "clock": data.clock.to_numpy(),
            "clock_x_dev": data.clock_x_dev.to_numpy(),
        }),
    )
    rows = []
    for model, predictors in models:
        fit = clustered_ols(data.y.to_numpy(), predictors, data.trading_date.to_numpy())
        rows.append({
            "market": "commodity", "symbol": "pooled_cross_section", "period": period,
            "model": model, **fit,
        })
    # Also test whether the continuation segment changes the reversal association.
    if "push" in data:
        data["push_rank"] = _rank_by_date(data, "push")
        data["clock_x_push"] = data.clock * data.push_rank
        fit = clustered_ols(
            data.y.to_numpy(),
            {
                "dev15": data.dev.to_numpy(),
                "clock": data.clock.to_numpy(),
                "clock_x_dev": data.clock_x_dev.to_numpy(),
                "push": data.push_rank.to_numpy(),
                "clock_x_push": data.clock_x_push.to_numpy(),
            },
            data.trading_date.to_numpy(),
        )
        rows.append({
            "market": "commodity", "symbol": "pooled_cross_section", "period": period,
            "model": "add_aligned_push_control", **fit,
        })
    return rows


def commodity_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    if frame.trading_date.dt.year.max() > 2021:
        raise ValueError("Commodity diagnostics must not include 2022+ dates")
    samples = [("2016-2021", frame)]
    samples.extend((str(year), group) for year, group in frame.groupby(frame.trading_date.dt.year))
    rows = []
    for period, sample in samples:
        rows.extend(_commodity_fit(sample, period))
    return pd.DataFrame(rows)


def _standardize(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype="float64")
    scale = float(np.std(array, ddof=1))
    if scale <= 0 or not np.isfinite(scale):
        return np.full(len(array), np.nan)
    return (array - float(np.mean(array))) / scale


def _index_fit(part: pd.DataFrame, symbol: str, period: str) -> dict:
    data = part[["trading_date", "gross_1130", "gap_return", "open_move"]].dropna().sort_values("trading_date")
    if len(data) < 30:
        return {
            "market": "index", "symbol": symbol, "period": period, "n": len(data),
            "hac_lags": HAC_LAGS, "open_move_coef": np.nan, "open_move_t_hac": np.nan,
            "gap_coef": np.nan, "gap_t_hac": np.nan, "r2_open_move": np.nan,
            "r2_gap_only": np.nan, "r2_joint": np.nan, "delta_r2_gap_given_open": np.nan,
        }
    y = _standardize(data.gross_1130)
    gap = _standardize(data.gap_return)
    open_move = _standardize(data.open_move)
    if not (np.isfinite(y).all() and np.isfinite(gap).all() and np.isfinite(open_move).all()):
        raise ValueError(f"Non-finite standardized inputs for {symbol}, {period}")
    open_fit = hac_ols(y, {"open_move": open_move})
    gap_fit = hac_ols(y, {"gap": gap})
    joint = hac_ols(y, {"open_move": open_move, "gap": gap})
    return {
        "market": "index", "symbol": symbol, "period": period, "n": len(data),
        "hac_lags": HAC_LAGS,
        "open_move_coef": open_fit["coef:open_move"],
        "open_move_t_hac": open_fit["t:open_move"],
        "gap_only_coef": gap_fit["coef:gap"],
        "gap_only_t_hac": gap_fit["t:gap"],
        "gap_coef": joint["coef:gap"],
        "gap_t_hac": joint["t:gap"],
        "open_move_joint_coef": joint["coef:open_move"],
        "open_move_joint_t_hac": joint["t:open_move"],
        "r2_open_move": open_fit["r_squared"],
        "r2_gap_only": gap_fit["r_squared"],
        "r2_joint": joint["r_squared"],
        "delta_r2_gap_given_open": joint["r_squared"] - open_fit["r_squared"],
    }


def _index_confirmation_fit(part: pd.DataFrame, symbol: str, period: str) -> dict:
    """Test whether an overnight gap has more continuation after open confirmation."""
    data = part[["trading_date", "gross_1130", "gap_return", "open_move"]].dropna().sort_values("trading_date")
    if len(data) < 30:
        return {
            "market": "index", "symbol": symbol, "period": period, "n": len(data),
            "hac_lags": HAC_LAGS, "gap_x_confirmation_coef": np.nan,
            "gap_x_confirmation_t_hac": np.nan, "r2_base": np.nan,
            "r2_confirmation": np.nan, "delta_r2_confirmation": np.nan,
        }
    y = _standardize(data.gross_1130)
    gap = _standardize(data.gap_return)
    open_move = _standardize(data.open_move)
    confirmed = (data.gap_return.to_numpy() * data.open_move.to_numpy() > 0).astype(float)
    confirmed_centered = confirmed - confirmed.mean()
    interaction = gap * confirmed_centered
    if not all(np.isfinite(values).all() for values in (y, gap, open_move, interaction)):
        raise ValueError(f"Non-finite confirmation model inputs for {symbol}, {period}")

    base = hac_ols(y, {"open_move": open_move, "gap": gap})
    fit = hac_ols(
        y,
        {
            "open_move": open_move,
            "gap": gap,
            "confirmation": confirmed_centered,
            "gap_x_confirmation": interaction,
        },
    )
    return {
        "market": "index", "symbol": symbol, "period": period, "n": len(data),
        "hac_lags": HAC_LAGS,
        "confirmed_fraction": float(confirmed.mean()),
        "gap_coef": fit["coef:gap"],
        "gap_t_hac": fit["t:gap"],
        "confirmation_coef": fit["coef:confirmation"],
        "confirmation_t_hac": fit["t:confirmation"],
        "gap_x_confirmation_coef": fit["coef:gap_x_confirmation"],
        "gap_x_confirmation_t_hac": fit["t:gap_x_confirmation"],
        "r2_base": base["r_squared"],
        "r2_confirmation": fit["r_squared"],
        "delta_r2_confirmation": fit["r_squared"] - base["r_squared"],
    }


def index_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    if frame.trading_date.dt.year.max() > 2021:
        raise ValueError("Index diagnostics must not include 2022+ dates")
    rows = []
    for symbol, part in frame.groupby("symbol", sort=True):
        samples = [("2016-2021", part)]
        samples.extend((str(year), group) for year, group in part.groupby(part.trading_date.dt.year))
        for period, sample in samples:
            rows.append(_index_fit(sample, symbol, period))
    return pd.DataFrame(rows)


def index_confirmation_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    if frame.trading_date.dt.year.max() > 2021:
        raise ValueError("Index diagnostics must not include 2022+ dates")
    rows = []
    for symbol, part in frame.groupby("symbol", sort=True):
        samples = [("2016-2021", part)]
        samples.extend((str(year), group) for year, group in part.groupby(part.trading_date.dt.year))
        for period, sample in samples:
            rows.append(_index_confirmation_fit(sample, symbol, period))
    return pd.DataFrame(rows)


def main() -> None:
    commodity_path = INPUT / "commodity_features.csv"
    index_path = INPUT / "index_features.csv"
    commodity = pd.read_csv(commodity_path, parse_dates=["trading_date"])
    index = pd.read_csv(index_path, parse_dates=["trading_date"])
    commodity = commodity[commodity.trading_date.dt.year.between(2016, 2021)]
    index = index[index.trading_date.dt.year.between(2016, 2021)]
    commodity_result = commodity_diagnostics(commodity)
    index_result = index_diagnostics(index)
    confirmation_result = index_confirmation_diagnostics(index)

    OUT.mkdir(parents=True, exist_ok=True)
    commodity_result.to_csv(OUT / "commodity_incremental.csv", index=False)
    index_result.to_csv(OUT / "index_gap_incremental.csv", index=False)
    confirmation_result.to_csv(OUT / "index_gap_confirmation.csv", index=False)
    protocol = {
        "status": "conditional_information_diagnostic_not_strategy",
        "sample": "2016-2021 only; feature inputs are the first-pass research CSVs; no 2022+ data read",
        "commodity": "Within-date percentile ranks of 09:17-11:30 outcome and available predictors. OLS models: dev15; dev15+clock; dev15+clock+interaction; then aligned-push and its clock interaction. Standard errors clustered by trading date.",
        "commodity_interaction": "clock_x_dev coefficient tests whether the dev15 relationship varies with the timing of the session extreme; negative is consistent with reversal being stronger when the clock is late and dev15 is still moving toward that extreme.",
        "index": "Separate IF/IH/IC time series. Outcome, overnight gap and 09:31-09:46 return are standardized within each fitted sample. Compare gap-only and open-move-only OLS R2 with joint OLS; report Newey-West HAC t statistics with 5 trading-date lags for joint coefficients.",
        "index_confirmation": "Predeclared economic hypothesis: if the 09:31-09:46 open move is in the same direction as the overnight gap, the gap's continuation association with the subsequent 09:47-11:30 return should be stronger. Test with standardized outcome/gap/open_move, a centered confirmation indicator and gap×confirmation interaction; expected interaction sign is positive. Report Newey-West HAC t statistics with 5 trading-date lags and annual signs.",
        "caution": "Full-period scaling, feature ranks and model screening are retrospective discovery; yearly sign consistency is descriptive, not independent validation. No execution-cost index data are included.",
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("COMMODITY CONDITIONAL MODELS")
    print(commodity_result[commodity_result.period.eq("2016-2021")].to_string(index=False))
    print("\nINDEX GAP INCREMENTAL MODELS")
    print(index_result[index_result.period.eq("2016-2021")].to_string(index=False))
    print("\nINDEX GAP CONFIRMATION MODELS")
    print(confirmation_result[confirmation_result.period.eq("2016-2021")].to_string(index=False))
    print(f"\noutputs: {OUT}")


if __name__ == "__main__":
    main()
