"""Build and load daily factors from the isolated RQData external cache."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..data import shard_io

PARTITIONS = ("research", "validation_2022", "holdout_locked")
EXTERNAL_DATA_ROOT = C.DATA_ROOT / "external_rqdata"
EXTERNAL_FACTOR_ROOT = C.FACTOR_DAILY_DIR / "external"
SPOT_CODES = {"AU": "AU9999.SGEX", "AG": "AG9999.SGEX"}


def partition_dates(symbol: str, partition: str) -> pd.DatetimeIndex:
    if partition == "research":
        bars = shard_io.load_shard(symbol, columns=["trading_date"])
        dates = pd.DatetimeIndex(pd.to_datetime(bars["trading_date"]).dt.normalize().unique())
        C.assert_no_holdout_dates(dates, what=f"{symbol} 外部因子日历")
    elif partition == "validation_2022":
        bars = shard_io.load_validation_shard(symbol, columns=["trading_date"])
        dates = pd.DatetimeIndex(pd.to_datetime(bars["trading_date"]).dt.normalize().unique())
        C.assert_validation_2022_dates(dates, what=f"{symbol} 外部因子日历")
    elif partition == "holdout_locked":
        dates = shard_io.load_holdout_trading_dates(symbol)
    else:
        raise ValueError(f"未知时间分区: {partition}")
    dates = dates.sort_values()
    dates.name = "trading_date"
    return dates


def _source_root(partition: str, root: Path | None = None) -> Path:
    root = Path(root or EXTERNAL_DATA_ROOT)
    return root if partition == "research" else root / partition


def _load_source(dataset: str, key: str, partition: str,
                 root: Path | None = None):
    source_root = _source_root(partition, root)
    manifest_path = source_root / "coverage.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("items", {}).get(f"{dataset}/{key}")
    if not entry or not entry.get("coverage"):
        return None
    path = source_root / entry["file"]
    return pd.read_pickle(path) if path.exists() else None


def _date_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        for name in ("date", "trading_date"):
            if name in index.names:
                values = index.get_level_values(name)
                return pd.DatetimeIndex(pd.to_datetime(values).normalize())
    return pd.DatetimeIndex(pd.to_datetime(index).normalize())


def _series(data, column: str | None = None,
            numeric: bool = True) -> pd.Series:
    if data is None:
        return pd.Series(dtype="float64")
    if isinstance(data, pd.Series):
        values = data
    elif isinstance(data, pd.DataFrame) and column in data.columns:
        values = data[column]
    else:
        return pd.Series(dtype="float64")
    result = pd.Series(values.to_numpy(), index=_date_index(data.index))
    result = result[~result.index.isna()]
    result = result[~result.index.duplicated(keep="last")].sort_index()
    if numeric:
        result = pd.to_numeric(result, errors="coerce")
    return result


def align_asof(source: pd.Series, calendar: pd.DatetimeIndex,
               lag: int = 1, fill_limit: int = 5) -> pd.Series:
    """Forward-fill only already-observed values, then apply the signal lag."""
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().sort_values()
    source = source[~source.index.duplicated(keep="last")].sort_index()
    if source.empty:
        return pd.Series(np.nan, index=calendar, dtype="float64")
    aligned = source.reindex(calendar, method="ffill", limit=fill_limit)
    return aligned.shift(lag)


def _main_contract_series(symbol: str, field: str, partition: str,
                          dominant: pd.Series, source_root: Path) -> pd.Series:
    result = pd.Series(np.nan, index=dominant.index, dtype="float64")
    for contract in dominant.dropna().astype(str).unique():
        prices = _load_source("contracts", contract, partition, source_root)
        contract_values = _series(prices, field)
        dates = dominant.index[dominant.astype(str) == contract]
        result.loc[dates] = contract_values.reindex(dates).to_numpy()
    return result


def build_symbol_factors(symbol: str, calendar: pd.DatetimeIndex,
                         partition: str,
                         source_root: Path | None = None) -> pd.DataFrame:
    source_root = _source_root(partition, source_root)
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().sort_values()
    if calendar.empty:
        return pd.DataFrame(index=calendar)

    factors = pd.DataFrame(index=calendar)

    roll = _load_source("roll_yield", f"{symbol}_main_sub", partition, source_root)
    for source_column, factor_name in (
        ("yield", "carry_main_sub_yield"),
        ("annualized_yield", "carry_main_sub_annualized"),
        ("annualized_yield_trading", "carry_main_sub_annualized_trading"),
    ):
        values = _series(roll, source_column)
        if not values.empty:
            factors[factor_name] = align_asof(values, calendar)

    warehouse = _load_source("warehouse", symbol, partition, source_root)
    warrant = _series(warehouse, "on_warrant")
    if not warrant.empty:
        aligned = align_asof(warrant, calendar)
        factors["warehouse_on_warrant"] = aligned
        factors["warehouse_log_level"] = np.log1p(aligned.where(aligned >= 0))
        factors["warehouse_low"] = -factors["warehouse_log_level"]
        change = aligned.div(aligned.shift(20).where(aligned.shift(20) != 0)).sub(1)
        factors["warehouse_change_20d"] = change
        factors["warehouse_drawdown_20d"] = -change

    dominant_data = _load_source("dominant", f"{symbol}_rank1", partition, source_root)
    dominant_raw = _series(dominant_data, numeric=False)
    if not dominant_raw.empty:
        dominant = align_asof(dominant_raw, calendar, lag=1, fill_limit=3)
        open_interest = _main_contract_series(
            symbol, "open_interest", partition, dominant_raw, source_root)
        oi = align_asof(open_interest, calendar)
        factors["main_open_interest"] = oi
        for window in (20, 60):
            prior_oi = oi.shift(window)
            same_contract = dominant.eq(dominant.shift(window))
            change = oi.div(prior_oi.where(prior_oi != 0)).sub(1)
            factors[f"main_oi_change_{window}d"] = change.where(same_contract)

    spot_code = SPOT_CODES.get(symbol)
    if spot_code:
        spot = _load_source("spot", f"benchmark_{spot_code}", partition, source_root)
        settlement = _main_contract_series(
            symbol, "settlement", partition, dominant_raw, source_root) \
            if not dominant_raw.empty else pd.Series(dtype="float64")
        for time_name in ("morning", "noon"):
            spot_price = _series(spot, time_name)
            common_dates = settlement.index.intersection(spot_price.index)
            if common_dates.empty:
                continue
            basis = settlement.reindex(common_dates).sub(spot_price.reindex(common_dates))
            basis_pct = basis.div(spot_price.reindex(common_dates).where(
                spot_price.reindex(common_dates) > 0))
            factors[f"spot_basis_{time_name}"] = align_asof(basis, calendar)
            factors[f"spot_basis_{time_name}_pct"] = align_asof(basis_pct, calendar)

    factors = factors.dropna(how="all")
    factors.index.name = "trading_date"
    return factors


def _write_manifest(root: Path, partition: str, items: dict) -> None:
    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": 1, "partitions": {}}
    manifest["generated"] = datetime.now().isoformat(timespec="seconds")
    manifest["partitions"][partition] = items
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def build_partition(partition: str, symbols: list[str] | None = None,
                    output_root: Path | None = None,
                    source_root: Path | None = None) -> dict:
    output_root = Path(output_root or EXTERNAL_FACTOR_ROOT)
    source_root = Path(source_root or EXTERNAL_DATA_ROOT)
    minute_dir = {
        "research": C.RESEARCH_DIR,
        "validation_2022": C.VALIDATION_DIR,
        "holdout_locked": C.HOLDOUT_DIR,
    }[partition]
    present = set(shard_io.list_shards(minute_dir))
    requested = set(symbols) if symbols else set(C.COMMODITY_SYMBOLS)
    selected = sorted(requested.intersection(C.COMMODITY_SYMBOLS, present))
    output_dir = output_root / partition
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for symbol in selected:
        calendar = partition_dates(symbol, partition)
        factors = build_symbol_factors(symbol, calendar, partition, source_root)
        if factors.empty:
            summary[symbol] = {"rows": 0, "columns": [], "first": None, "last": None}
            continue
        path = shard_io.save_shard(factors, output_dir, symbol)
        summary[symbol] = {
            "file": path.name,
            "rows": int(len(factors)),
            "columns": list(factors.columns),
            "first": factors.index.min().date().isoformat(),
            "last": factors.index.max().date().isoformat(),
        }
        print(f"{partition} {symbol}: {len(factors)} 日, {len(factors.columns)} 因子")

    _write_manifest(output_root, partition, summary)
    return summary


def load_external_wide(factor: str, partition: str,
                       symbols: list[str] | None = None,
                       root: Path | None = None) -> pd.DataFrame:
    """Load one external factor as trading_date x symbol from the factor library."""
    root = Path(root or EXTERNAL_FACTOR_ROOT) / partition
    selected = symbols or shard_io.list_shards(root)
    columns = {}
    for symbol in selected:
        path = shard_io.find_shard(root, symbol)
        if path is None:
            continue
        frame = pd.read_parquet(path) if path.suffix == shard_io.PARQUET_EXT else pd.read_pickle(path)
        if factor in frame:
            columns[symbol] = frame[factor]
    return pd.DataFrame(columns).sort_index()


def load_external_panel(symbols: list[str] | None = None,
                        partition: str = "research",
                        root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all external candidates as factor -> (trading_date x symbol)."""
    directory = Path(root or EXTERNAL_FACTOR_ROOT) / partition
    available = shard_io.list_shards(directory)
    selected = [symbol for symbol in (symbols or available) if symbol in available]
    values: dict[str, dict[str, pd.Series]] = {}
    for symbol in selected:
        path = shard_io.find_shard(directory, symbol)
        if path is None:
            continue
        frame = (pd.read_parquet(path) if path.suffix == shard_io.PARQUET_EXT
                 else pd.read_pickle(path))
        frame.index = pd.to_datetime(frame.index)
        for factor in frame.columns:
            values.setdefault(factor, {})[symbol] = frame[factor]
    return {factor: pd.DataFrame(columns).sort_index()
            for factor, columns in values.items()}