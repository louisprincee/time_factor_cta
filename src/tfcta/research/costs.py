"""Historical daily commissions and observed-price tick estimates. No automatic downloads."""
from pathlib import Path
import numpy as np
import pandas as pd
from .. import config as C

MULTIPLIER: dict[str, float] = {
    "A": 10, "B": 10, "C": 10, "CS": 10, "M": 10, "Y": 10, "P": 10, "JD": 10,
    "L": 5, "V": 5, "PP": 5, "EG": 10, "I": 100, "J": 100, "JM": 60,
    "CU": 5, "AL": 5, "ZN": 5, "PB": 5, "NI": 1, "SN": 1, "AU": 1000, "AG": 15,
    "RB": 10, "HC": 10, "RU": 10, "BU": 10, "FU": 10, "SP": 10, "SC": 1000,
    "AP": 10, "CF": 5, "SR": 10, "TA": 5, "RM": 10, "OI": 10, "FG": 20,
    "SF": 5, "SM": 5, "MA": 10, "ZC": 100,
}



def load_fees(partition="research"):
    path = (C.RESEARCH_OUT_DIR / "fee_history.csv" if partition == "research"
            else C.DATA_ROOT / "validation_2022" / "fee_history.csv")
    if partition not in ("research", "validation_2022"):
        raise C.HoldoutViolation("OOS fees are locked")
    table = pd.read_csv(path, parse_dates=["trading_date"])
    guard = C.assert_no_holdout_dates if partition == "research" else C.assert_validation_2022_dates
    guard(table.trading_date, what="手续费缓存")
    if table.duplicated(["symbol","trading_date"]).any():
        raise ValueError("手续费缓存有重复品种/日期")
    return table


def fee_tables(table, prices):
    """Opening and overnight-closing fees as fractions of current raw notional.

    At roll, the old-contract closing fee uses its last observed dominant rate.
    This is explicit historical-rate approximation, not contract-level execution.
    No same-day round trips in the daily engine; broker surcharge +.01 yuan
    for by-volume rates, +1% for by-money rates.
    """
    opened, closed, roll_closed = [pd.DataFrame(np.nan, index=prices.index,
        columns=prices.columns) for _ in range(3)]
    for symbol in prices.columns:
        sub = table[table.symbol == symbol].set_index("trading_date").sort_index()
        if sub.empty:
            raise KeyError(f"手续费缓存缺 {symbol}")
        if sub.index.has_duplicates:
            raise ValueError(f"手续费缓存 {symbol} 日期重复")
        q = sub.reindex(prices.index, method="ffill")  # never backfill
        notional = prices[symbol] * MULTIPLIER[symbol]
        def rate(source, field):
            money = source.commission_type.eq("by_money")
            valid = source.commission_type.isin(["by_money", "by_volume"])
            values = source[field]
            result = np.where(money, values*1.01, (values+.01)/notional)
            # Even zero ad-valorem exchange fee has broker's .01 yuan fee.
            result = np.where(money & values.eq(0), .01/notional, result)
            return pd.Series(result, index=prices.index).where(valid)
        opened[symbol] = rate(q, "open_commission")
        closed[symbol] = rate(q, "close_commission")
        # The previous dominant rate is only substituted when contract changes.
        old = q.shift(1).where(q.contract.ne(q.contract.shift(1)), q)
        roll_closed[symbol] = rate(old, "close_commission")
    return opened, closed, roll_closed


def load_ticks():
    table = pd.read_csv(C.RESEARCH_OUT_DIR / "tick_size.csv")
    C.assert_no_holdout_dates(pd.to_datetime(table.year.astype(str)+"-01-01"), "tick 缓存")
    return table


def slippage_tables(ticks, prices, n_ticks=1.):
    """Observed tick from preceding year / actual execution day's raw open.

    These are estimates, not an official historical tick schedule. First year
    without prior tick is unavailable; cash until covered. No future backfill.
    """
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


def intraday_costs(samples, table, ticks, n_ticks=1.):
    """Decision-time cost estimate and realized same-day roundtrip cost.

    All costs are fractions of entry notional (estimated: reference notional).
    Ad-valorem closing fees use exit price; 平今 never substitutes overnight
    rates. Tick estimates use strictly preceding years. Missing history is NaN.
    """
    if not np.isfinite(n_ticks) or n_ticks < 0:
        raise ValueError('滑点必须非负且有限')
    if 'close_commission_today' not in table:
        raise ValueError('手续费缓存缺少平今 close_commission_today')
    if table.duplicated(['symbol','trading_date']).any() or ticks.duplicated(['symbol','year']).any():
        raise ValueError('费用或 tick 历史重复')
    C.assert_no_holdout_dates(samples.trading_date, '日内成交费用')
    C.assert_no_holdout_dates(table.trading_date, '日内手续费历史')
    C.assert_no_holdout_dates(pd.to_datetime(ticks.year.astype(str)+'-01-01'), '日内 tick 历史')
    result = pd.DataFrame(np.nan, index=samples.index, columns=['estimated_cost','actual_cost'])
    for symbol, sample in samples.groupby('symbol', sort=False):
        history = table[table.symbol.eq(symbol)].copy()
        history['trading_date'] = pd.to_datetime(history.trading_date)
        history = history.set_index('trading_date').sort_index()
        if history.empty:
            raise KeyError(f'手续费缓存缺 {symbol}')
        dates = pd.DatetimeIndex(sample.trading_date)
        q = history.reindex(dates, method='ffill').reset_index(drop=True)
        price = sample[['reference_price','entry_price','exit_price']].to_numpy(dtype=float)
        if not np.isfinite(price).all() or (price <= 0).any():
            raise ValueError('费用分母必须为有限的正价格')
        ref, entry, exit_price = price.T
        tick_history = ticks[ticks.symbol.eq(symbol)].set_index('year').tick.sort_index().round(2)
        prior = tick_history.rename(index=lambda y:int(y)+1)
        prior = prior.reindex(sorted(set(prior.index)|set(dates.year))).ffill()
        tick = prior.reindex(dates.year).to_numpy()
        def commission(field, execution, denominator):
            values = q[field].to_numpy(dtype=float)
            money = q.commission_type.eq('by_money').to_numpy()
            known = q.commission_type.isin(['by_money','by_volume']).to_numpy()
            volume_rate = (values+.01)/(denominator*MULTIPLIER[symbol])
            money_rate = values*1.01*execution/denominator
            money_rate = np.where(values==0, .01/(denominator*MULTIPLIER[symbol]), money_rate)
            return np.where(known, np.where(money,money_rate,volume_rate), np.nan)
        result.loc[sample.index,'estimated_cost'] = (commission('open_commission',ref,ref)
            + commission('close_commission_today',ref,ref) + 2*n_ticks*tick/ref)
        result.loc[sample.index,'actual_cost'] = (commission('open_commission',entry,entry)
            + commission('close_commission_today',exit_price,entry) + 2*n_ticks*tick/entry)
    return result
