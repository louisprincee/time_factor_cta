"""外部数据因子：从隔离的 RQData 缓存构造，按时间分区落盘和读取。

分区与样本外
------------
``research``（≤2021）、``validation_2022``、``holdout_locked``（2023 起，须给出已经过去的截止日）。
原始数据按同样的分区存放：研究期在 ``external_rqdata/`` 根目录，其余在同名子目录。

连续计算
--------
构造某个分区时，把它**之前各分区**的原始数据和交易日历按时间接在前面一起算，最后只写出
本分区的日期。否则 20 日、60 日变化这类窗口在分区开头会被截断，2022 年前一两个月整段为 NaN。
接在前面的只有更早的数据，不会引入未来信息。

时间对齐：原始数据只向前填充已经发布的值，再整体滞后一个交易日（``align_asof``）。
"""
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

FAMILIES = {
    'carry_main_sub_yield': '外部期限结构(候选)',
    'carry_main_sub_annualized': '外部期限结构(候选)',
    'carry_main_sub_annualized_trading': '外部期限结构(候选)',
    'warehouse_on_warrant': '外部库存(候选)',
    'warehouse_log_level': '外部库存(候选)',
    'warehouse_low': '外部库存(候选)',
    'warehouse_change_20d': '外部库存(候选)',
    'warehouse_drawdown_20d': '外部库存(候选)',
    'main_open_interest': '外部持仓(候选)',
    'main_oi_change_20d': '外部持仓(候选)',
    'main_oi_change_60d': '外部持仓(候选)',
    'spot_basis_morning': '外部基差(候选)',
    'spot_basis_morning_pct': '外部基差(候选)',
    'spot_basis_noon': '外部基差(候选)',
    'spot_basis_noon_pct': '外部基差(候选)',
}


def partitions_through(partition: str) -> tuple[str, ...]:
    """``partition`` 及其之前的全部分区，按时间顺序。"""
    if partition not in PARTITIONS:
        raise ValueError(f"未知时间分区: {partition}")
    return PARTITIONS[:PARTITIONS.index(partition) + 1]


# --------------------------------------------------------------------------
# 日历与原始数据
# --------------------------------------------------------------------------
def partition_dates(symbol: str, partition: str, end=None) -> pd.DatetimeIndex:
    """单个分区的交易日历。``holdout_locked`` 必须给出已经过去的截止日，日历截到那一天。"""
    if partition == "holdout_locked":
        if end is None:
            raise C.HoldoutViolation("构造样本外外部因子必须给出截止日期")
        C.assert_test_window_closed(end)
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
        dates = dates[dates <= pd.Timestamp(C.to_date(end))]
    else:
        raise ValueError(f"未知时间分区: {partition}")
    dates = dates.sort_values()
    dates.name = "trading_date"
    return dates


def history_calendar(symbol: str, partition: str, end=None) -> pd.DatetimeIndex:
    """``partition`` 及之前各分区的日历拼接。更早的分区缺分片（上市晚）就跳过。"""
    pieces = []
    for part in partitions_through(partition):
        try:
            pieces.append(partition_dates(symbol, part, end=end if part == partition else None))
        except FileNotFoundError:
            if part == partition:
                raise
    dates = pd.DatetimeIndex(np.concatenate([p.values for p in pieces])).unique().sort_values()
    dates.name = "trading_date"
    return dates


def _source_root(partition: str, root: Path | None = None) -> Path:
    root = Path(root or EXTERNAL_DATA_ROOT)
    return root if partition == "research" else root / partition


def _load_one(dataset: str, key: str, partition: str, root: Path | None):
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


def _load_source(dataset: str, key: str, partition: str, root: Path | None = None):
    """``partition`` 及之前各分区的同一份原始数据，按时间接起来。"""
    parts = [x for x in (_load_one(dataset, key, p, root)
                         for p in partitions_through(partition)) if x is not None]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else pd.concat(parts)


def _date_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        for name in ("date", "trading_date"):
            if name in index.names:
                return pd.DatetimeIndex(pd.to_datetime(index.get_level_values(name)).normalize())
    return pd.DatetimeIndex(pd.to_datetime(index).normalize())


def _series(data, column: str | None = None, numeric: bool = True) -> pd.Series:
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
    """只向前填充已经观测到的值，再整体滞后 ``lag`` 个交易日。"""
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().sort_values()
    source = source[~source.index.duplicated(keep="last")].sort_index()
    if source.empty:
        return pd.Series(np.nan, index=calendar, dtype="float64")
    return source.reindex(calendar, method="ffill", limit=fill_limit).shift(lag)


def _main_contract_series(field: str, partition: str,
                          dominant: pd.Series, source_root: Path | None) -> pd.Series:
    result = pd.Series(np.nan, index=dominant.index, dtype="float64")
    for contract in dominant.dropna().astype(str).unique():
        values = _series(_load_source("contracts", contract, partition, source_root), field)
        dates = dominant.index[dominant.astype(str) == contract]
        result.loc[dates] = values.reindex(dates).to_numpy()
    return result


# --------------------------------------------------------------------------
# 构造
# --------------------------------------------------------------------------
def build_symbol_factors(symbol: str, calendar: pd.DatetimeIndex,
                         partition: str,
                         source_root: Path | None = None) -> pd.DataFrame:
    """在 ``calendar`` 上构造外部因子。

    ``source_root`` 是外部数据根目录；原始数据读 ``partition`` 及之前的各分区。
    ``calendar`` 应当同样覆盖之前的分区（见 :func:`history_calendar`），窗口才连续。
    """
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

    warrant = _series(_load_source("warehouse", symbol, partition, source_root), "on_warrant")
    if not warrant.empty:
        aligned = align_asof(warrant, calendar)
        factors["warehouse_on_warrant"] = aligned
        factors["warehouse_log_level"] = np.log1p(aligned.where(aligned >= 0))
        factors["warehouse_low"] = -factors["warehouse_log_level"]
        prior = aligned.shift(20)
        change = aligned.div(prior.where(prior != 0)).sub(1)
        factors["warehouse_change_20d"] = change
        factors["warehouse_drawdown_20d"] = -change

    dominant_raw = _series(_load_source("dominant", f"{symbol}_rank1", partition, source_root),
                           numeric=False)
    if not dominant_raw.empty:
        dominant = align_asof(dominant_raw, calendar, lag=1, fill_limit=3)
        oi = align_asof(_main_contract_series("open_interest", partition, dominant_raw,
                                              source_root), calendar)
        factors["main_open_interest"] = oi
        for window in (20, 60):
            prior_oi = oi.shift(window)
            same_contract = dominant.eq(dominant.shift(window))
            change = oi.div(prior_oi.where(prior_oi != 0)).sub(1)
            factors[f"main_oi_change_{window}d"] = change.where(same_contract)

    spot_code = SPOT_CODES.get(symbol)
    if spot_code:
        spot = _load_source("spot", f"benchmark_{spot_code}", partition, source_root)
        settlement = (_main_contract_series("settlement", partition, dominant_raw, source_root)
                      if not dominant_raw.empty else pd.Series(dtype="float64"))
        for time_name in ("morning", "noon"):
            spot_price = _series(spot, time_name)
            common = settlement.index.intersection(spot_price.index)
            if common.empty:
                continue
            px = spot_price.reindex(common)
            basis = settlement.reindex(common).sub(px)
            factors[f"spot_basis_{time_name}"] = align_asof(basis, calendar)
            factors[f"spot_basis_{time_name}_pct"] = align_asof(basis.div(px.where(px > 0)),
                                                                calendar)

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
                    source_root: Path | None = None,
                    end=None) -> dict:
    output_root = Path(output_root or EXTERNAL_FACTOR_ROOT)
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
        own = partition_dates(symbol, partition, end=end)
        calendar = history_calendar(symbol, partition, end=end)
        factors = build_symbol_factors(symbol, calendar, partition, source_root)
        factors = factors[factors.index.isin(own)].dropna(how="all")
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


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def load_panel(symbols: list[str] | None = None,
               partition: str = "research",
               root: Path | None = None) -> dict[str, pd.DataFrame]:
    """单个分区的全部外部因子：{因子: 交易日 × 品种}。"""
    directory = Path(root or EXTERNAL_FACTOR_ROOT) / partition
    available = shard_io.list_shards(directory)
    values: dict[str, dict[str, pd.Series]] = {}
    for symbol in [s for s in (symbols or available) if s in available]:
        frame = shard_io.read_frame(shard_io.find_shard(directory, symbol))
        frame.index = pd.to_datetime(frame.index)
        for factor in frame.columns:
            values.setdefault(factor, {})[symbol] = frame[factor]
    return {factor: pd.DataFrame(cols).sort_index() for factor, cols in values.items()}


def load_wide(symbols: list[str], partitions, index: pd.Index) -> dict[str, pd.DataFrame]:
    """按时间顺序拼接多个分区，对齐到 ``index × symbols``。"""
    if isinstance(partitions, str):
        partitions = (partitions,)
    pieces: dict[str, list[pd.DataFrame]] = {}
    for partition in partitions:
        for name, frame in load_panel(symbols, partition=partition).items():
            pieces.setdefault(name, []).append(frame)
    out = {}
    for name, frames in pieces.items():
        wide = pd.concat(frames).sort_index()
        wide = wide[~wide.index.duplicated(keep='last')]
        out[name] = wide.reindex(index=index, columns=symbols)
    return out


def require_partitions(factors, partitions) -> None:
    """选了外部因子却没构造对应分区时停下，而不是让信号在那一段悄悄变成 NaN。"""
    used = [name for name in factors if name in FAMILIES]
    if not used:
        return
    missing = [p for p in partitions if not shard_io.list_shards(EXTERNAL_FACTOR_ROOT / p)]
    if missing:
        hint = " --oos-end YYYY-MM-DD" if "holdout_locked" in missing else ""
        raise FileNotFoundError(
            f"所选外部因子 {used} 缺少分区 {missing}，"
            f"请先运行 step3_build_factors.py --external-partitions {' '.join(missing)}{hint}")
