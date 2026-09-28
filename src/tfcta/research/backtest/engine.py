"""回测引擎：调仓节奏、波动率目标、成交滞后、按换手扣费、当年池内等权、测试年拼接。

时间口径（与 ``data.bars`` 一致）：t 日收盘的信号 → ``execute_position`` 之后 t+1 开盘持有 →
赚 ``day_ret[t+1]``，换手在 t+1 开盘成交并在同一天扣费。

成本口径：单品种先按换手扣手续费与滑点，再在当年品种池里等权；退池那天补一笔平仓。
报告的年换手与扣费用的是同一份成交量（:func:`trades`）。

调仓节奏默认用 :func:`staggered`：五批各持有五个交易日、错开一天调仓，目标仓位取五批平均。
只在周五调仓的单批书对相位极敏感（同一组因子换成周一调仓，Sharpe 能从 +1.1 掉到 -0.5），
错开分批等于把五个相位一起持有，结果不再取决于哪天调仓。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ... import config as C


# --------------------------------------------------------------------------
# 信号 → 仓位
# --------------------------------------------------------------------------
def weekly(sig: pd.DataFrame) -> pd.DataFrame:
    """每周最后一个交易日取值，其后交易日沿用，直到下一次调仓。

    输出仍是收盘时的目标，不做成交滞后。调仓日信号为 NaN 时，这一周起目标就是空仓，
    不把更早的仓位填过来：数据缺口不能伪装成仍持有上一周的观点。
    """
    index = pd.DatetimeIndex(sig.index)
    iso = index.isocalendar()
    key = pd.Series((iso.year.astype(str) + '-' + iso.week.astype(str)).to_numpy(),
                    index=index)
    is_reb = key.ne(key.shift(-1)).fillna(True)
    regime = is_reb.cumsum()
    reb_value = sig.where(is_reb, axis=0)

    def _week_value(block: pd.Series) -> pd.Series:
        return pd.Series(block.iloc[0], index=block.index)

    return reb_value.groupby(regime).transform(_week_value)


def staggered(sig: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """n 批错开调仓：第 k 批在交易日序号 ≡ k (mod n) 的收盘取值，持有 n 个交易日。

    输出是 n 批目标的平均，仍是收盘时的目标，不做成交滞后。某批调仓日信号为 NaN 时
    这一批在它的持有期内空仓（按 0 计入平均），与 :func:`weekly` 的缺口口径一致；
    n 批全空时输出 NaN。相位按交易日序号而不是星期几划分，节假日不会让某一批多持几天。
    """
    n = int(n)
    if n <= 1:
        return sig.copy()
    order = np.arange(len(sig)) % n
    total = np.zeros(sig.shape, dtype='float64')
    live = np.zeros(sig.shape, dtype=bool)
    for k in range(n):
        is_reb = pd.Series(order == k, index=sig.index)
        value = sig.where(is_reb, axis=0).ffill()
        # 调仓日有值记 1、缺值记 0，其余日沿用最近一次调仓日的标记
        valid = sig.notna().astype('float64').where(is_reb, axis=0).ffill().eq(1.0)
        held = value.where(valid).to_numpy(dtype='float64')
        ok = np.isfinite(held)
        total += np.where(ok, held, 0.0)
        live |= ok
    out = np.where(live, total / n, np.nan)
    return pd.DataFrame(out, index=sig.index, columns=sig.columns)


def rebalance(sig: pd.DataFrame, tranches: int) -> pd.DataFrame:
    """``tranches=0`` 表示旧的周五单批（:func:`weekly`），否则 :func:`staggered`。"""
    return weekly(sig) if int(tranches) == 0 else staggered(sig, int(tranches))


def vol_target(signal: pd.DataFrame,
               vol: pd.DataFrame | None,
               target: float | None,
               cap: float = C.VOL_TARGET_CAP) -> pd.DataFrame:
    """仓位 = 信号 × 目标年化波动 / 事前年化波动，杠杆截到 ``cap``。

    ``vol`` 是截至 t 日收盘的日收益波动（``library.SignalSet.vol``），与 t 日信号同时可知。
    波动为 0 或缺失（停板、刚上市）时仓位为 NaN，不猜杠杆。``target`` 为 None 或 0 时原样返回。
    """
    if not target:
        return signal
    if vol is None:
        raise ValueError('波动率目标需要事前波动 vol')
    ann = vol.reindex(index=signal.index, columns=signal.columns) * np.sqrt(252.0)
    scale = (float(target) / ann.where(ann > 0)).clip(upper=float(cap))
    return signal * scale


def execute_position(signal: pd.DataFrame) -> pd.DataFrame:
    """收盘信号 → 下一交易日开盘才持有的仓位。"""
    return signal.shift(1)


# --------------------------------------------------------------------------
# 品种池与成交
# --------------------------------------------------------------------------
def universe_mask(index: pd.Index, columns: pd.Index, universe: dict) -> pd.DataFrame:
    """(交易日 × 品种) 的"当年在池"布尔表。"""
    allowed: dict[str, set[int]] = {}
    for y, syms in universe.items():
        for s in syms:
            allowed.setdefault(s, set()).add(int(y))
    year = pd.DatetimeIndex(index).year.to_numpy()
    data = {c: np.isin(year, list(allowed.get(c, ()))) for c in columns}
    return pd.DataFrame(data, index=index, columns=list(columns))


def trades(position: pd.DataFrame, inpool: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (持仓, 成交量)。仓位 NaN 当 0。

    给了 ``inpool`` 就在"最后一个在池日"补一笔平仓。不补的话这笔成交落在第一个池外日，
    而那一天不进当年池内等权的分母，费用凭空消失。在池状态必须来自品种池，不能用
    ``position.notna()`` 反推：信号中途断掉也会产生 NaN，那笔平仓本来就算在池内了。
    """
    cur = position.fillna(0.0)
    size = (cur - position.shift(1).fillna(0.0)).abs()
    if inpool is not None:
        # reindex 引入的 NaN 用 .eq(True) 当作"不在池"
        ip = inpool.reindex(index=position.index, columns=position.columns).eq(True)
        # fill_value=True：数据末尾不算退池，否则最后一天会凭空多一笔平仓费
        exiting = ip & ~ip.shift(-1, fill_value=True)
        size = size + cur.abs().where(exiting, 0.0)
    return cur, size


def symbol_net(position: pd.DataFrame,
               day_ret: pd.DataFrame,
               fee: float,
               inpool: pd.DataFrame | None = None,
               slippage: pd.DataFrame | float | None = None) -> pd.DataFrame:
    """对齐到 day_ret 的单品种扣费后收益。

    ``slippage`` 是逐 (交易日, 品种) 的单边比例滑点（见 ``costs``），与 ``fee`` 按同一份
    换手计费。无换手的格子强制记 0 成本：``NaN × 0`` 是 NaN，会把那一天整条净值变成 NaN，
    进而在等权时被跳过——成本反倒变成 0。
    """
    pos = position.reindex(index=day_ret.index, columns=day_ret.columns)
    cur, size = trades(pos, inpool)
    rate = float(fee)
    if isinstance(slippage, pd.DataFrame):
        rate = slippage.reindex(index=day_ret.index, columns=day_ret.columns) + float(fee)
    elif slippage is not None:
        rate = float(fee) + float(slippage)
    cost = (rate * size).where(size > 0, 0.0)
    pnl = cur * day_ret
    # 当天收益缺失（该品种当天没有开盘价）但确有成交时，成本照扣，不随收益一起变成 NaN
    pnl = pnl.where(day_ret.notna() | ~(size > 0), 0.0)
    return pnl - cost


def _nanmean_rows(block: np.ndarray) -> np.ndarray:
    ok = np.isfinite(block)
    cnt = ok.sum(axis=1)
    acc = np.where(ok, block, 0.0).sum(axis=1)
    out = np.full(block.shape[0], np.nan)
    good = cnt > 0
    out[good] = acc[good] / cnt[good]
    return out


def portfolio_return(net: pd.DataFrame, universe: dict) -> pd.Series:
    """当年池内等权。某品种当天收益是 NaN 时不进分母，也不填 0。"""
    values = np.full(len(net), np.nan)
    year = net.index.year.to_numpy()
    for y, syms in universe.items():
        cols = [s for s in syms if s in net.columns]
        mask = year == int(y)
        if mask.any() and cols:
            values[mask] = _nanmean_rows(net.loc[mask, cols].to_numpy(dtype='float64'))
    return pd.Series(values, index=net.index, name='port_ret')


def run_book(position: pd.DataFrame,
             day_ret: pd.DataFrame,
             universe: dict,
             fee: float,
             slippage: pd.DataFrame | float | None = None) -> pd.Series:
    """仓位（尚未套用品种池）→ 扣掉手续费与滑点后的等权组合日收益。"""
    inpool = universe_mask(position.index, position.columns, universe)
    net = symbol_net(position.where(inpool), day_ret, fee, inpool=inpool, slippage=slippage)
    return portfolio_return(net, universe)


def annual_turnover(position: pd.DataFrame, universe: dict, years: list[int]) -> float:
    """单品种年换手按当年池内品种取均值，再对年份取均值。

    与 :func:`run_book` 扣费的是同一份成交量：池外仓位记 0，进池当天从 0 建仓，
    退池当天补平仓。
    """
    inpool = universe_mask(position.index, position.columns, universe)
    _, size = trades(position.where(inpool), inpool)
    size = size.where(inpool)
    per = []
    for y in years:
        cols = [s for s in universe.get(int(y), []) if s in size.columns]
        block = size.loc[size.index.year == int(y), cols]
        if cols and not block.empty:
            per.append(float(block.sum().mean()))
    return float(np.mean(per)) if per else float('nan')


# --------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------
def walk_forward_folds(test_years: list[int] | None = None) -> list[dict]:
    """研究期逐年折。所有参数都是事前固定的，没有在训练窗口上拟合的东西，
    所以这里只给测试年；逐年结果只是把 2016–2021 切成六段看稳定性，不是样本外。"""
    years = C.WF_TEST_YEARS_LIST if test_years is None else list(test_years)
    return [{
        'test_year': int(y),
        'sparse_night': int(y) in C.WF_FOLDS_WITH_SPARSE_NIGHT,
    } for y in years]


def stitch_test_years(port: pd.Series, years: list[int]) -> pd.Series:
    """把各测试年的收益按时间拼成一条。空年份跳过，不插 0。"""
    parts = [port.loc[port.index.year == int(y)].dropna() for y in years]
    parts = [p for p in parts if len(p)]
    if not parts:
        return port.iloc[0:0]
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep='first')]
    out.name = getattr(port, 'name', None) or 'port_ret'
    return out
