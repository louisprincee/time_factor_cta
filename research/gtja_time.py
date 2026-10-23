"""筛查国泰君安报告中的 28 个股指时间因子，不构造策略或回测组合。

    python -m research.gtja_time
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tfcta import config as C
from tfcta.data import load_shard
from tfcta.index.gtja_time import COLUMNS, daily_factors, open_to_open, standardize

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "research" / "gtja_time_factors.json"
OUT = C.RUNS_DIR / "index" / "gtja_time"
EVAL_START = pd.Timestamp("2016-01-04")
EVAL_END = pd.Timestamp("2021-12-31")


def load_config(path=CONFIG_PATH):
    config = json.loads(path.read_text(encoding="utf-8"))
    if "不读 2022" not in config["sample"]:
        raise ValueError("样本口径必须写明不读 2022")
    return config


def load_symbol(symbol):
    frame = load_shard(symbol, directory=C.INDEX_RESEARCH_DIR, columns=[
        "trading_date", "open", "openw", "close", "closew", "highw", "loww", "volume", "total_turnover"])
    if pd.to_datetime(frame.trading_date).dt.year.gt(2021).any():
        raise C.HoldoutViolation(f"{symbol} 研究分片进入了 2022")
    return frame


def hac_slope(factor: pd.Series, target: pd.Series, lags: int) -> tuple[float, float, int]:
    paired = pd.concat([factor.rename("x"), target.rename("y")], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    count = len(paired)
    if count < max(20, lags + 3) or paired.x.std(ddof=1) <= 0:
        return np.nan, np.nan, count
    x = paired.x.to_numpy(dtype="float64")
    x = (x - x.mean()) / x.std(ddof=1)
    y = paired.y.to_numpy(dtype="float64")
    design = np.column_stack([np.ones(count), x])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        return np.nan, np.nan, count
    residual = y - design @ beta
    score = design * residual[:, None]
    meat = score.T @ score
    used_lags = min(int(lags), count - 1)
    for lag in range(1, used_lags + 1):
        gamma = score[lag:].T @ score[:-lag]
        weight = 1.0 - lag / (used_lags + 1.0)
        meat += weight * (gamma + gamma.T)
    bread = np.linalg.inv(design.T @ design)
    variance = bread @ meat @ bread
    se = float(np.sqrt(max(variance[1, 1], 0.0)))
    return float(beta[1]), float(beta[1] / se) if se > 0 else np.nan, count


def factor_summary(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall, yearly = [], []
    for (symbol, factor), group in panel.groupby(["symbol", "factor"], sort=True):
        valid = group.dropna(subset=["value", "forward_return"])
        spearman = (
            float(stats.spearmanr(valid.value, valid.forward_return).statistic)
            if len(valid) >= 3 else np.nan
        )
        beta, t_value, count = hac_slope(valid.standardized, valid.forward_return, int(config["hac_lags"]))
        overall.append({
            "symbol": symbol, "factor": factor, "observations": count,
            "spearman": spearman, "beta_per_sd": beta, "nw_t": t_value,
        })
        for year, year_group in valid.groupby(valid.trading_date.dt.year, sort=True):
            yearly_spearman = (
                float(stats.spearmanr(year_group.value, year_group.forward_return).statistic)
                if len(year_group) >= 3 else np.nan
            )
            year_beta, year_t, year_count = hac_slope(
                year_group.standardized, year_group.forward_return, int(config["hac_lags"])
            )
            yearly.append({
                "symbol": symbol, "factor": factor, "year": int(year),
                "observations": year_count, "spearman": yearly_spearman,
                "beta_per_sd": year_beta, "nw_t": year_t,
            })
    return pd.DataFrame(overall), pd.DataFrame(yearly)


def build_panel(config: dict) -> pd.DataFrame:
    rows = []
    for symbol in config["instruments"]:
        frame = load_symbol(symbol)
        factors = daily_factors(frame, float(config["price_threshold"]), float(config["volume_threshold"]))
        z_factors = standardize(factors[list(COLUMNS)], int(config["z_window"]), int(config["z_min_periods"]))
        # Close-known factors predict the open-to-open interval after the next open.
        target = open_to_open(frame).shift(-1)
        dates = pd.DatetimeIndex(factors.index.intersection(target.index))
        keep = (dates >= EVAL_START) & (dates <= EVAL_END)
        C.assert_no_holdout_dates(dates[keep], f"{symbol} GTJA 因子样本")
        factors = factors.reindex(dates)
        z_factors = z_factors.reindex(dates)
        target = target.reindex(dates)
        for factor in COLUMNS:
            rows.extend({
                "trading_date": day,
                "symbol": symbol,
                "factor": factor,
                "value": float(value) if np.isfinite(value) else np.nan,
                "standardized": float(zvalue) if np.isfinite(zvalue) else np.nan,
                "forward_return": float(ret) if np.isfinite(ret) else np.nan,
            } for day, value, zvalue, ret in zip(
                dates[keep],
                factors.loc[keep, factor].to_numpy(dtype="float64"),
                z_factors.loc[keep, factor].to_numpy(dtype="float64"),
                target.loc[keep].to_numpy(dtype="float64"),
            ))
        print(f"{symbol}: {int(keep.sum())} research dates, {len(COLUMNS)} factors", flush=True)
    return pd.DataFrame(rows)


def main():
    config = load_config()
    panel = build_panel(config)
    summary, yearly = factor_summary(panel, config)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT / "factor_panel.parquet", index=False)
    summary.to_csv(OUT / "factor_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT / "factor_yearly.csv", index=False, encoding="utf-8-sig")
    (OUT / "factor_protocol.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"因子研究结果：{OUT}", flush=True)


if __name__ == "__main__":
    main()
