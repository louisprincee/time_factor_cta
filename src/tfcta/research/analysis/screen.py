"""研究期因子稳定性与板块、单品种异质性。只读研究期。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ... import config as C
from ...data import sectors
from . import stats

MIN_ABS_AVG_IC = 0.01
MIN_ABS_T = 2.0
MIN_SYMBOL_YEARS = 4
TERM_STRUCTURE_FACTORS = [
    "carry",
    "carry_main_sub_yield",
    "carry_main_sub_annualized",
    "carry_main_sub_annualized_trading",
]


def passes_screen(annual_ics: list[float], t_value: float,
                  min_years: int = 6) -> bool:
    """年度 IC 同号，|平均 IC| 超过门槛，且时序 |t| 超过门槛。"""
    values = np.asarray(annual_ics, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size < min_years or values.size == 0 or not np.isfinite(t_value):
        return False
    if np.any(values == 0) or not np.all(np.sign(values) == np.sign(values[0])):
        return False
    return abs(float(values.mean())) > MIN_ABS_AVG_IC and abs(float(t_value)) > MIN_ABS_T


def _factor_metrics(factor: pd.DataFrame, forward: pd.DataFrame,
                    universe: dict[int, list[str]], name: str,
                    years: list[int], cross_sectional: bool = False) -> dict:
    if cross_sectional:
        table = stats.cross_sectional_ic_table(
            factor, forward, universe, name, years)
    else:
        table = stats.factor_ic_table(factor, forward, universe, name, years)
    annual = table[table["fold"].isin([str(year) for year in years])]
    overall = table.loc[table["fold"] == "mean_of_folds"].iloc[0]
    values = {
        "factor": name,
        "avg_ic": float(annual["ic"].mean()) if annual["ic"].notna().any() else np.nan,
        "ic_ts": float(overall["ic_ts"]),
        "t": float(overall["t"]),
        "n_periods": int(overall["n_periods"]),
        "n_years": int(annual["ic"].notna().sum()),
    }
    for _, row in annual.iterrows():
        values[f"ic_{row['fold']}"] = row["ic"]
        values[f"ic_ts_{row['fold']}"] = row["ic_ts"]
    annual_ics = [values.get(f"ic_{year}", np.nan) for year in years]
    finite = np.asarray(annual_ics, dtype="float64")
    values["same_sign"] = bool(
        len(annual_ics) == len(years)
        and np.isfinite(finite).all()
        and np.all(np.sign(finite) == np.sign(finite[0]))
        and finite[0] != 0
    )
    values["selected"] = passes_screen(annual_ics, values["t"], min_years=len(years))
    return values


def _subgroup_metrics(factor: pd.DataFrame, forward: pd.DataFrame,
                      universe: dict[int, list[str]], name: str,
                      years: list[int], label: str, members: set[str],
                      min_years: int, cross_sectional: bool = False) -> dict:
    subgroup_universe = {
        year: [symbol for symbol in universe.get(year, []) if symbol in members]
        for year in years
    }
    metrics = _factor_metrics(factor, forward, subgroup_universe, name, years,
                              cross_sectional=cross_sectional)
    metrics["group"] = label
    annual_counts = [len(subgroup_universe[year]) for year in years]
    metrics["symbols_by_year"] = ";".join(
        f"{year}:{count}" for year, count in zip(years, annual_counts))
    metrics["mean_symbols"] = float(np.mean(annual_counts))
    annual_ics = [metrics.get(f"ic_{year}", np.nan) for year in years]
    metrics["selected"] = passes_screen(annual_ics, metrics["t"], min_years)
    return metrics


def heterogeneity_tables(factor_set, forward: pd.DataFrame,
                         universe: dict[int, list[str]], symbols: list[str],
                         years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unmapped = sorted(set(symbols) - set(sectors.SECTOR_BY_SYMBOL))
    if unmapped:
        raise ValueError(f"请先为当前研究池品种指定板块: {unmapped}")

    factors = {**factor_set.signed, **factor_set.unsigned}
    overall = pd.DataFrame([
        _factor_metrics(factor, forward, universe, name, years,
                        cross_sectional=factor_set.family[name].startswith("截面"))
        for name, factor in factors.items()
    ]).sort_values(["selected", "t"], ascending=[False, False])
    selected = set(overall.loc[overall["selected"], "factor"])
    subgroup_factors = sorted(selected | (set(TERM_STRUCTURE_FACTORS) & set(factors)))

    sector_rows, symbol_rows = [], []
    present_sectors = sorted({sectors.SECTOR_BY_SYMBOL[symbol] for symbol in symbols})
    for name in subgroup_factors:
        factor = factors[name]
        cross_sectional = factor_set.family[name].startswith("截面")
        for sector in present_sectors:
            members = {symbol for symbol in symbols
                       if sectors.SECTOR_BY_SYMBOL[symbol] == sector}
            sector_rows.append(_subgroup_metrics(
                factor, forward, universe, name, years, sector, members,
                min_years=len(years), cross_sectional=cross_sectional))
        if cross_sectional:
            continue
        for symbol in symbols:
            symbol_rows.append(_subgroup_metrics(
                factor, forward, universe, name, years, symbol, {symbol},
                min_years=MIN_SYMBOL_YEARS))

    sector_table = pd.DataFrame(sector_rows)
    symbol_table = pd.DataFrame(symbol_rows)
    if not sector_table.empty:
        sector_table = sector_table.sort_values(["selected", "t"], ascending=[False, False])
    if not symbol_table.empty:
        symbol_table = symbol_table.sort_values(["selected", "t"], ascending=[False, False])
    return overall, sector_table, symbol_table


def write_heterogeneity(factor_set, forward: pd.DataFrame,
                        universe: dict[int, list[str]], symbols: list[str],
                        years: list[int], run: Path | None = None) -> None:
    overall, sector_table, symbol_table = heterogeneity_tables(
        factor_set, forward, universe, symbols, years)
    tables = {
        "factor_stability.csv": overall,
        "sector_factor_ic.csv": sector_table,
        "symbol_factor_ic.csv": symbol_table,
    }
    C.ensure_dirs()
    for filename, table in tables.items():
        table.to_csv(C.RESEARCH_OUT_DIR / filename, index=False, encoding="utf-8-sig")
        if run is not None:
            table.to_csv(Path(run) / filename, index=False, encoding="utf-8-sig")

    columns = ["factor", "avg_ic", "ic_ts", "t", "n_periods"]
    print("\n== 4. 研究期稳定性：年度 IC 同号，|年均 IC| > "
          f"{MIN_ABS_AVG_IC:g}，月度时序 |t| > {MIN_ABS_T:g}")
    chosen = overall.loc[overall["selected"], columns]
    print(chosen.to_string(index=False) if not chosen.empty else "（没有因子同时通过）")
    print("\n通过同一门槛的板块：")
    sector_cols = ["factor", "group", "avg_ic", "ic_ts", "t", "n_periods", "mean_symbols"]
    picked = sector_table.loc[sector_table["selected"], sector_cols] if not sector_table.empty else sector_table
    print(picked.to_string(index=False) if not picked.empty else "（没有板块组合通过）")
    print(f"\n异质性结果: {C.RESEARCH_OUT_DIR}")
