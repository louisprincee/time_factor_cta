"""One daily continuous-notional research engine: next open, historical fees, rolls.

An exposure is a fraction of research capital. Annual universe weights are fixed
before signals/returns are known; missing signals leave their allocation in cash.
No integer lots, margin financing, limits or simulated intraday executions.
"""
import numpy as np
import pandas as pd


def rebalance(signal, days=1, mode="single", phase=0):
    """Close targets: a single periodic book or equal staggered phases.

    Schedule uses the full common trading calendar, including warm-up history;
    phase is frozen, never chosen from test returns. Missing scheduled signal
    resets that leg to cash instead of resurrecting an earlier opinion.
    """
    if int(days) != days or days < 1 or mode not in ("single","staggered"):
        raise ValueError("调仓周期须为正整数，模式为 single/staggered")
    if not 0 <= phase < days:
        raise ValueError("调仓相位越界")
    order = np.arange(len(signal)) % days
    if mode == "staggered":
        return sum(rebalance(signal, days, "single", k) for k in range(days))/days
    scheduled = pd.Series(order == phase, index=signal.index)
    return signal.fillna(0.).where(scheduled, axis=0).ffill().fillna(0.)


def position(signal, vol=None, vol_target=.20, cap=1., days=1, mode="single", phase=0):
    if vol_target:
        if vol is None:
            raise ValueError("波动率目标需要事前波动率")
        ann = vol.reindex_like(signal)*np.sqrt(252)
        signal = signal * (vol_target/ann.where(ann>0)).clip(upper=cap)
    return rebalance(signal.clip(-cap,cap), days, mode, phase).shift(1).fillna(0.)


def allocate(position, universe):
    out = position*0.
    for year, members in universe.items():
        members = list(dict.fromkeys(members))
        present = [s for s in members if s in position.columns]
        if members:
            mask = position.index.year == int(year)
            out.loc[mask,present] = position.loc[mask,present].fillna(0.)/len(members)
    return out.fillna(0.)


def trade_legs(exposure, rolls=None):
    previous = exposure.shift(1).fillna(0.)
    same = exposure*previous > 0
    retained = pd.DataFrame(np.minimum(exposure.abs(), previous.abs()),
        index=exposure.index, columns=exposure.columns).where(same,0.)
    if rolls is not None:
        retained = retained.where(~rolls.reindex_like(exposure).fillna(False).astype(bool),0.)
    return exposure.abs()-retained, previous.abs()-retained


def backtest(position, returns, universe, open_fee=0., close_fee=0., slippage=0.,
             rolls=None, roll_close_fee=None):
    position = position.reindex_like(returns)
    exposure = allocate(position, universe)
    opened, closed = trade_legs(exposure, rolls)
    if ((exposure != 0) & ~np.isfinite(returns)).any().any():
        raise ValueError("持仓对应的市场收益缺失，不能跳过品种或填零")
    def cost(size, rate):
        if isinstance(rate,pd.DataFrame):
            rate = rate.reindex_like(size)
        values = (size*rate).where(size>0,0.)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("实际成交对应的成本缺失")
        return values
    closing = close_fee
    if rolls is not None and roll_close_fee is not None:
        closing = close_fee.where(~rolls.reindex_like(exposure).fillna(False), roll_close_fee)
    charge = cost(opened,open_fee)+cost(closed,closing)+cost(opened+closed,slippage)
    gross = (exposure*returns.fillna(0.)).sum(axis=1)
    return pd.DataFrame({"gross":gross, "net":gross-charge.sum(axis=1),
        "turnover":(opened+closed).sum(axis=1), "cost":charge.sum(axis=1)})
