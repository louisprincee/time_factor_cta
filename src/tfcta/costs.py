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

# 合约乘数改过的品种：(首个合约 yymm, 末个合约 yymm, 乘数)，2010 年以后的合约，2026-10-23 用米筐 instruments 逐合约核对。
# RU1208 起 10 吨，此前 5 吨；FU1202–FU1812 是 50 吨的旧合约；PB1109–PB1308 是 25 吨的旧合约。
CONTRACT_MULTIPLIER_RANGES: dict[str, tuple[tuple[int, int, float], ...]] = {
    "RU": ((1001, 1207, 5.0),),
    "FU": ((1202, 1812, 50.0),),
    "PB": ((1109, 1308, 25.0),),
}


def contract_multiplier(symbol, contract) -> float:
    """这张合约的乘数：落在改动区间里的用区间值，否则用品种现行乘数。"""
    base = float(MULTIPLIER[symbol])
    if not isinstance(contract, str) or not contract.startswith(symbol):
        return base
    digits = contract[len(symbol):]
    if len(digits) != 4 or not digits.isdigit():
        return base
    yymm = int(digits)
    for first, last, value in CONTRACT_MULTIPLIER_RANGES.get(symbol, ()):
        if first <= yymm <= last:
            return float(value)
    return base


def contract_multiplier_overrides(symbol, contracts) -> dict[str, float]:
    """这些合约里乘数不等于品种现行乘数的，供账本按持有合约取乘数。"""
    base = float(MULTIPLIER[symbol])
    out = {}
    for contract in set(contracts):
        value = contract_multiplier(symbol, contract)
        if value != base:
            out[contract] = value
    return out

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
    notional = base_a * np.asarray(multiplier, dtype=float)
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


def load_fees(partition="research"):
    table = _read_fee_table(fee_path(partition), partition)
    return backfill_shfe_fees(table) if partition == "research" else table


# 米筐的上期所历史手续费从 2013-10-08 才有。用户 2026-10-23 决定：2011-01-01 到 2013-10-07 用每个品种
# 在米筐的第一条费率（2013-10-08）往前沿用。这是假设，不是历史费率；大商所 2010 年起有真实费率，不受影响。
SHFE_FEE_FIRST = pd.Timestamp("2013-10-08")
SHFE_FEE_BACKFILL_FROM = pd.Timestamp(C.EARLY_START)
SHFE_FEE_BACKFILL_SYMBOLS = ("AG", "AL", "AU", "CU", "RB", "RU", "ZN")


def backfill_shfe_fees(table):
    """给上期所品种补 SHFE_FEE_BACKFILL_FROM 到第一条费率前一天的每个工作日，费率等于第一条。"""
    parts = [table]
    days = pd.bdate_range(SHFE_FEE_BACKFILL_FROM, SHFE_FEE_FIRST - pd.Timedelta(1, unit="D"))
    for symbol in SHFE_FEE_BACKFILL_SYMBOLS:
        sub = table[table.symbol == symbol].sort_values("trading_date")
        if sub.empty:
            continue                      # 文件里没有这个品种，用到时 fee_tables 会报缺
        if sub.trading_date.iloc[0] != SHFE_FEE_FIRST:
            raise ValueError(f"{symbol} 的第一条手续费不在 {SHFE_FEE_FIRST.date()}，上期所回填假设不成立，先核对")
        first = sub.iloc[[0]]
        parts.append(first.loc[first.index.repeat(len(days))].assign(trading_date=days, contract=np.nan))
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "trading_date"]).reset_index(drop=True)


def load_index_fees(partition="research"):
    """读取单独保存的股指手续费表。"""
    return _read_fee_table(index_fee_path(partition), partition)


# 中金所非套保“日内开仓交易量较大”：买开与卖开之和，超过下列手数即违规，故上限就是该手数。
# 2015-08-26 至 2018-12-02 按单个产品（该产品全部合约合计）；2018-12-03 起改为单个合约。
# 本账本每个品种只开主力，换月只开新合约，两种口径下手数相同。套保豁免不适用。
# 公告：中金所 2015-08-25、2015-08-28、2015-09-02、2017-02-16、2018-12-02、2019-04-19。
CFFEX_OPEN_LIMITS = (
    (pd.Timestamp("2015-08-26"), 600),
    (pd.Timestamp("2015-08-31"), 100),
    (pd.Timestamp("2015-09-07"), 10),
    (pd.Timestamp("2017-02-17"), 20),
    (pd.Timestamp("2018-12-03"), 50),
    (pd.Timestamp("2019-04-22"), 500),
)


def index_open_limits(index, symbols) -> pd.DataFrame:
    """交易日 × 品种的当日最多开仓手数；2015-08-26 以前不限（inf）。"""
    dates = pd.DatetimeIndex(index)
    limit = pd.Series(np.inf, index=dates)
    for start, lots in CFFEX_OPEN_LIMITS:
        limit[dates >= start] = float(lots)
    return pd.DataFrame({symbol: limit for symbol in symbols}, index=dates)


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


FEE_MAX_STALE = pd.Timedelta(31, unit="D")


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
        # 只向后沿用不超过 FEE_MAX_STALE 的旧费率，从不回填；文件中途断档时成交会报错
        q = sub.reindex(prices.index, method="ffill", tolerance=FEE_MAX_STALE)
        # 按手数收费要换成名义比例，乘数用这一行费率对应的合约（RU 等改过乘数的品种）
        mult = np.array([contract_multiplier(symbol, c) for c in q["contract"]], dtype=float)
        opened[symbol] = commission_rate(q.commission_type, q.open_commission, prices[symbol], prices[symbol], mult)
        closed[symbol] = commission_rate(q.commission_type, q.close_commission, prices[symbol], prices[symbol], mult)
        # The previous dominant rate is only substituted when contract changes.
        old = q.shift(1).where(q.contract.ne(q.contract.shift(1)), q)
        old_mult = np.array([contract_multiplier(symbol, c) for c in old["contract"]], dtype=float)
        roll_closed[symbol] = commission_rate(old.commission_type, old.close_commission, prices[symbol], prices[symbol], old_mult)
    return opened, closed, roll_closed

TICK_MIN_SHARE = 0.01

TICK_MIN_DIFFS = 100
TICK_MAX_STALE = 1

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
        # 上一年估不出 tick（成交太稀）时最多再往前沿用 TICK_MAX_STALE 年，表截断时成交会报错
        prior = sub.rename(index=lambda y: int(y)+1).reindex(years).ffill(limit=TICK_MAX_STALE)
        result[symbol] = prior.reindex(prices.index.year).to_numpy()/prices[symbol] * n_ticks
    return result
