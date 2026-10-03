"""One external control: main/subdominant annualized carry, lagged one day.

Legacy raw roll-yield caches are retained as source data. Warehouse/OI factors
are removed because publication-time/vintage guarantees were not established.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .. import config as C
from ..data import shard_io

EXTERNAL_DATA_ROOT = C.DATA_ROOT / "external_rqdata"
EXTERNAL_FACTOR_ROOT = C.FACTOR_DAILY_DIR / "external"
FAMILIES = {"carry_main_sub_annualized":"期限结构对照"}


def guard_dates(dates,partition):
    if partition == "research":
        C.assert_no_holdout_dates(dates,"carry缓存")
    elif partition == "validation_2022":
        C.assert_validation_2022_dates(dates,"carry缓存")
    else:
        raise C.HoldoutViolation("OOS external factors are locked")

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




def build_partition(partition="research", symbols=None):
    if partition not in ("research","validation_2022"):
        raise C.HoldoutViolation("OOS external factors are locked")
    roots = [EXTERNAL_DATA_ROOT]
    if partition == "validation_2022":
        roots.append(EXTERNAL_DATA_ROOT / partition)
    summary = {}
    for symbol in symbols or shard_io.list_shards(C.RESEARCH_DIR):
        pieces = []
        for root in roots:
            manifest = root / "coverage.json"
            if not manifest.exists():
                continue
            entry = json.loads(manifest.read_text()).get("items",{}).get(f"roll_yield/{symbol}_main_sub")
            if not entry or not entry.get("coverage"):
                continue
            path = (root / entry["file"]).resolve()
            if root.resolve() not in path.parents:
                raise ValueError("carry源路径越界")
            source = pd.read_pickle(path)
            guard_dates(_date_index(source.index), "research" if root==EXTERNAL_DATA_ROOT else partition)
            pieces.append(_series(source,"annualized_yield"))
        if not pieces:
            continue
        minute = shard_io.load_shard(symbol,columns=["trading_date"])
        if partition == "validation_2022":
            minute = pd.concat([minute,shard_io.load_validation_shard(symbol,columns=["trading_date"])])
        calendar = pd.DatetimeIndex(pd.to_datetime(minute.trading_date).unique()).sort_values()
        aligned = align_asof(pd.concat(pieces),calendar)
        keep = calendar.year<=2021 if partition=="research" else calendar.year==2022
        frame = pd.DataFrame({"carry_main_sub_annualized":aligned.loc[keep]})
        guard_dates(frame.index,partition)
        path = shard_io.save_shard(frame,EXTERNAL_FACTOR_ROOT/partition,symbol)
        summary[symbol] = {"file":str(path),"lag_days":1,"fill_limit":5}
    return summary


def load_panel(symbols=None, partition="research", root=None):
    if partition not in ("research","validation_2022"):
        raise C.HoldoutViolation("OOS external factors are locked")
    directory = Path(root or EXTERNAL_FACTOR_ROOT)/partition
    values = {}
    for symbol in symbols or shard_io.list_shards(directory):
        path = shard_io.find_shard(directory,symbol)
        if path is None:
            continue
        if partition == 'research':
            C.assert_research_only(path)
        frame = shard_io.read_frame(path,["carry_main_sub_annualized"])
        guard_dates(frame.index,partition)
        if frame.index.has_duplicates:
            raise ValueError(f"{path} carry日期重复")
        if "carry_main_sub_annualized" in frame:
            values[symbol] = frame["carry_main_sub_annualized"]
    return {"carry_main_sub_annualized":pd.DataFrame(values).sort_index()} if values else {}


def load_wide(symbols,partitions,index):
    if isinstance(partitions,str):
        partitions = (partitions,)
    frames = [load_panel(symbols,p).get("carry_main_sub_annualized") for p in partitions]
    frames = [f for f in frames if f is not None]
    if not frames:
        return {}
    joined = pd.concat(frames).sort_index()
    if joined.index.has_duplicates:
        raise ValueError("carry分区重叠")
    return {"carry_main_sub_annualized":joined.reindex(index=index,columns=symbols)}
