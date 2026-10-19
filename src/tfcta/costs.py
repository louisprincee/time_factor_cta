"""手续费、合约乘数和最小变动价位。研究期与 2022、样本外分区分开读取。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tfcta import config as C


MULTIPLIER: dict[str, float] = {
    "A": 10, "B": 10, "C": 10, "CS": 10, "M": 10, "Y": 10, "P": 10, "JD": 10,
    "LH": 16, "RR": 10, "L": 5, "V": 5, "PP": 5, "EG": 10, "EB": 5, "PG": 20,
    "I": 100, "J": 100, "JM": 60, "BB": 500, "FB": 10, "LG": 90,
    "CU": 5, "AL": 5, "ZN": 5, "PB": 5, "NI": 1, "SN": 1, "AU": 1000, "AG": 15,
    "BC": 5, "AO": 20, "AD": 10, "RB": 10, "HC": 10, "WR": 10, "SS": 5,
    "RU": 10, "NR": 10, "BR": 5, "BU": 10, "FU": 10, "LU": 10, "SP": 10, "SC": 1000,
    "EC": 50,
    "AP": 10, "CF": 5, "CY": 5, "SR": 10, "TA": 5, "MA": 10, "OI": 10, "RM": 10,
    "RS": 10, "FG": 20, "SA": 20, "UR": 20, "PF": 5, "PK": 5, "CJ": 5, "PX": 5,
    "PR": 15, "SH": 30, "SF": 5, "SM": 5, "ZC": 100, "WH": 20, "PM": 50,
    "RI": 20, "LR": 20, "JR": 20, "SI": 5, "LC": 1, "PS": 3,
    # 中金所股指：每点人民币。IF/IH 为 300，IC/IM 为 200。
    "IF": 300, "IH": 300, "IC": 200, "IM": 200,
}
INDEX_MULTIPLIER = {symbol: MULTIPLIER[symbol] for symbol in ("IF", "IH", "IC", "IM")}

def require_multipliers(symbols):
    """点名的品种必须有乘数；缺了就停，不跳过。"""
    missing = sorted({s for s in symbols if s not in MULTIPLIER})
    if missing:
        raise KeyError(f"缺少合约乘数: {missing}")
    return list(symbols)

def commission_rate(kind, value, price, base, multiplier):
    """单边手续费占 base 对应名义金额的比例。按金额 ×1.01，按手数 +0.01 元，零费率按 0.01 元。"""
    kind_a = np.atleast_1d(np.asarray(kind, dtype=object))
    value_a = np.atleast_1d(np.asarray(value, dtype=float))
    price_a = np.atleast_1d(np.asarray(price, dtype=float))
    base_a = np.atleast_1d(np.asarray(base, dtype=float))
    notional = base_a * float(multiplier)
    money = kind_a == "by_money"
    volume = kind_a == "by_volume"
    valid = (money | volume) & np.isfinite(price_a) & np.isfinite(value_a) & (notional > 0)
    rate = np.full(value_a.shape, np.nan)
    rate = np.where(volume & valid, (value_a + 0.01) / notional, rate)
    rate = np.where(money & valid & (value_a == 0), 0.01 / notional, rate)
    rate = np.where(money & valid & (value_a != 0), value_a * 1.01 * price_a / base_a, rate)
    if rate.size == 1 and np.ndim(kind) == 0 and not isinstance(kind, (pd.Series, np.ndarray)):
        return float(rate[0])
    index = getattr(price, "index", None)
    return pd.Series(np.asarray(rate, dtype=float), index=index)

def fee_path(partition):
    paths = {"research": C.RESEARCH_OUT_DIR / "fee_history.csv",
             "validation_2022": C.COMMODITY_DATA / "validation_2022" / "fee_history.csv",
             "oos": C.COMMODITY_DATA / "oos" / "fee_history.csv"}
    if partition not in paths:
        raise C.HoldoutViolation(f"未知手续费分区 {partition}")
    return paths[partition]

def index_fee_path(partition):
    """股指手续费与商品分开放，避免和商品缓存混读。"""
    root = C.INDEX_DATA_ROOT
    paths = {"research": root / "research" / "fee_history.csv",
             "validation_2022": root / "validation_2022" / "fee_history.csv",
             "oos": root / "oos" / "fee_history.csv"}
    if partition not in paths:
        raise C.HoldoutViolation(f"未知股指手续费分区 {partition}")
    return paths[partition]


def _read_fee_table(path, partition):
    if partition == "oos":
        end = pd.Timestamp(C.final_evaluation_end())  # 未打开最终评估时这里就拒绝
    if not Path(path).is_file():
        raise FileNotFoundError(f"手续费缓存不存在: {path}")
    table = pd.read_csv(path, parse_dates=["trading_date"])
    if partition == "oos":
        table = table[table.trading_date <= end]
        C.assert_strict_oos_dates(table.trading_date, what="手续费缓存")
    else:
        guard = C.assert_no_holdout_dates if partition == "research" else C.assert_validation_2022_dates
        guard(table.trading_date, what="手续费缓存")
    if table.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("手续费缓存有重复品种/日期")
    return fill_close_today(table)


def load_index_fees(partition="research"):
    return _read_fee_table(index_fee_path(partition), partition)


def load_fees(partition="research"):
    return _read_fee_table(fee_path(partition), partition)


CZCE_CLOSE_TODAY_FROM = pd.Timestamp("2017-06-23")
CZCE_SYMBOLS = frozenset({
    "CF", "FG", "JR", "LR", "MA", "OI", "PM", "RI",
    "RM", "RS", "SF", "SM", "SR", "TA", "WH", "ZC",
})


def fill_close_today(table):
    """Conservatively fill only the known pre-2017-06-23 CZCE close-today fee gap."""
    table = table.sort_values(["symbol", "trading_date"]).copy()
    later = table.groupby("symbol").close_commission_today.transform(lambda values: values.bfill())
    source_kind = table.commission_type.where(table.close_commission_today.notna())
    same_kind = table.commission_type.eq(
        source_kind.groupby(table.symbol).transform(lambda values: values.bfill())
    )
    known_gap = (
        table.symbol.isin(CZCE_SYMBOLS)
        & (table.trading_date < CZCE_CLOSE_TODAY_FROM)
    )
    gap = table.close_commission_today.isna() & later.notna() & same_kind & known_gap
    table.loc[gap, "close_commission_today"] = np.maximum(
        table.close_commission[gap], later[gap]
    )
    table["close_today_filled"] = gap
    return table


def check_costs(table, pools):
    """Fail when active pool observations are missing estimated transaction costs."""
    year = table.trading_date.dt.year
    member = pd.Series(
        [symbol in pools.get(year, ()) for symbol, year in zip(table.symbol, year)],
        index=table.index,
    )
    ready = table.dev15.notna() & member
    missing = table[ready & table.est_cost.isna()]
    if len(missing):
        total = table[ready].groupby([table.symbol, year]).size()
        share = missing.groupby([missing.symbol, missing.trading_date.dt.year]).size() / total
        bad = share[share > 0.05]
        if len(bad):
            raise ValueError(
                f"池内品种缺成本（手续费或 tick），按品种、年的缺失比例: "
                f"{bad.round(3).to_dict()}"
            )


def fee_tables(table, prices):
    """开仓费和隔夜平仓费，按当日原始名义金额的比例计算。"""
    opened, closed, roll_closed = [pd.DataFrame(np.nan, index=prices.index,
        columns=prices.columns) for _ in range(3)]
    for symbol in prices.columns:
        sub = table[table.symbol == symbol].set_index("trading_date").sort_index()
        if sub.empty:
            raise KeyError(f"手续费缓存缺 {symbol}")
        if sub.index.has_duplicates:
            raise ValueError(f"手续费缓存 {symbol} 日期重复")
        q = sub.reindex(prices.index, method="ffill")  # never backfill
        mult = MULTIPLIER[symbol]
        opened[symbol] = commission_rate(q.commission_type, q.open_commission, prices[symbol], prices[symbol], mult)
        closed[symbol] = commission_rate(q.commission_type, q.close_commission, prices[symbol], prices[symbol], mult)
        # The previous dominant rate is only substituted when contract changes.
        old = q.shift(1).where(q.contract.ne(q.contract.shift(1)), q)
        roll_closed[symbol] = commission_rate(old.commission_type, old.close_commission, prices[symbol], prices[symbol], mult)
    return opened, closed, roll_closed

TICK_MIN_SHARE = 0.01

TICK_MIN_DIFFS = 100

def tick_path(partition):
    paths = {"research": C.RESEARCH_OUT_DIR / "tick_size.csv",
             "validation_2022": C.COMMODITY_DATA / "validation_2022" / "tick_size.csv"}
    if partition not in paths:
        raise C.HoldoutViolation(f"tick 表只落盘研究期和 2022，样本外在最终评估里现算: {partition}")
    return paths[partition]

def estimate_ticks(close, trading_date, min_share=TICK_MIN_SHARE, min_diffs=TICK_MIN_DIFFS):
    """按年估计最小变动价位：相邻分钟原始收盘价的非零变动里，占比至少 min_share 的最小步长。
    close 应是 factors.raw_price_path 之后的原始价。非零变动少于 min_diffs 次的年份 tick 留空。
    返回 year、tick、tick_mode、n_diff、median_close。"""
    price = pd.Series(np.asarray(close, dtype="float64"))
    year = pd.Series(pd.to_datetime(np.asarray(trading_date)).year)
    rows = []
    for y, p in price.groupby(year.to_numpy()):
        step = p.diff().abs().round(4)
        step = step[step > 0]
        if step.empty:
            continue
        share = step.value_counts(normalize=True).sort_index()
        tick = float(share[share >= min_share].index.min()) if len(step) >= min_diffs else np.nan
        rows.append({"year": int(y), "tick": tick,
                     "tick_mode": float(step.mode().iloc[0]), "n_diff": int(len(step)),
                     "median_close": float(p.median())})
    return pd.DataFrame(rows, columns=["year", "tick", "tick_mode", "n_diff", "median_close"])

def load_ticks(partition="research"):
    """研究期的 tick 表；partition="validation_2022" 时再接上 2022 年那一行（供 2023 交易用）。"""
    table = pd.read_csv(tick_path("research"))
    C.assert_no_holdout_dates(pd.to_datetime(table.year.astype(str)+"-01-01"), "tick 缓存")
    if partition == "research":
        return table
    later = pd.read_csv(tick_path("validation_2022"))
    C.assert_validation_2022_dates(pd.to_datetime(later.year.astype(str)+"-01-01"), "2022 tick 缓存")
    return pd.concat([table, later], ignore_index=True)

def slippage_tables(ticks, prices, n_ticks=1.):
    """用上一年的最小变动价位除以成交日的原始开盘价。"""
    result = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for symbol in prices.columns:
        sub = ticks[ticks.symbol.eq(symbol)].set_index("year").tick.sort_index()
        if sub.empty:
            raise KeyError(f"tick 缓存缺 {symbol}")
        # Remove float32 quote noise in the existing estimates.
        sub = sub.round(2)
        years = sorted(set(prices.index.year) | set(sub.index+1))
        prior = sub.rename(index=lambda y: int(y)+1).reindex(years).ffill()
        result[symbol] = prior.reindex(prices.index.year).to_numpy()/prices[symbol] * n_ticks
    return result
