"""等权回测与分族合成。

单品种先按换手扣费，再在当年品种池里等权。组合信号是成员信号的等权平均，
NaN 不投票，不再对组合做第二次参数搜索。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from . import panel
from . import protocol as folds
from . import stats as metrics


def universe_mask(index: pd.Index,
                  columns: pd.Index,
                  universe: dict) -> pd.DataFrame:
    """(交易日 × 品种) 的"当年在池"布尔表。"""
    allowed: dict[str, set[int]] = {}
    for y, syms in universe.items():
        for s in syms:
            allowed.setdefault(s, set()).add(int(y))
    year = pd.DatetimeIndex(index).year.to_numpy()
    data = {c: np.isin(year, list(allowed.get(c, ())))
            for c in columns}
    return pd.DataFrame(data, index=index, columns=list(columns))


def mask_universe(position: pd.DataFrame, universe: dict) -> pd.DataFrame:
    """池外年份的仓位置为 NaN，后面当 0 持仓处理。"""
    return position.where(universe_mask(position.index, position.columns, universe))


def symbol_net(position: pd.DataFrame,
               day_ret: pd.DataFrame,
               fee: float,
               inpool: pd.DataFrame | None = None,
               slippage: pd.DataFrame | float | None = None) -> pd.DataFrame:
    """对齐到 day_ret 的宽表。仓位 NaN 当 0。

    给了 ``inpool`` 就在"最后一个在池日"补一笔平仓手续费。不补的话这笔费用会丢：
    退池后仓位是 NaN→0，换手确实算得出来，但它落在**第一个池外日**，而
    ``portfolio_return`` 只在当年池内品种上取平均，那一行不进分母，费用凭空消失。

    在池状态必须来自 ``universe``，不能用 ``pos.notna()`` 反推：信号中途断掉也会
    产生 NaN 仓位，但那笔平仓费本来就算在池内、已经计入，反推会重复收费。

    ``slippage`` 是**逐 (交易日, 品种) 的单边比例滑点**（见 :mod:`research.costs`），
    与 ``fee`` 按同一份换手计费，包括退池那笔平仓。给标量就等价于把 ``fee`` 调大，
    有意义的用法是给宽表——tick / 价位这个比例在品种间实测差 13.7 倍，等权组合里
    低价位品种的成本贡献远高于高价位品种，用一个数建模会把这个结构抹平。

    无换手的格子强制记 0 成本，不让缺失的费率漏进来：``NaN × 0`` 在 numpy 里是 NaN，
    会把那一天的净值整条变成 NaN，进而在等权时被跳过——成本反倒变成 0。
    """
    pos = position.reindex(index=day_ret.index, columns=day_ret.columns)
    prev = pos.shift(1).fillna(0.0)
    cur = pos.fillna(0.0)
    turnover = (cur - prev).abs()
    if inpool is not None:
        # reindex 会引入 NaN 把 dtype 变成 object，用 .eq(True) 拿回干净的布尔表
        # （NaN.eq(True) 是 False，正好是"没这一格就当不在池"）
        ip = inpool.reindex(index=day_ret.index, columns=day_ret.columns).eq(True)
        # fill_value=True：数据末尾不算退池，否则最后一天会凭空多一笔平仓费
        exiting = ip & ~ip.shift(-1, fill_value=True)
        turnover = turnover + cur.abs().where(exiting, 0.0)
    rate = float(fee)
    if isinstance(slippage, pd.DataFrame):
        slip = slippage.reindex(index=day_ret.index, columns=day_ret.columns)
        rate = slip + float(fee)
    elif slippage is not None:
        rate = float(fee) + float(slippage)
    cost = (rate * turnover).where(turnover > 0, 0.0)
    return cur * day_ret - cost


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
    by_year: dict[int, list[str]] = {}
    for y, syms in universe.items():
        by_year[int(y)] = [s for s in syms if s in net.columns]
    for y, cols in by_year.items():
        mask = year == y
        if not mask.any() or not cols:
            continue
        block = net.loc[mask, cols].to_numpy(dtype='float64')
        values[mask] = _nanmean_rows(block)
    return pd.Series(values, index=net.index, name='port_ret')


def run_book(position: pd.DataFrame,
             day_ret: pd.DataFrame,
             universe: dict,
             fee: float,
             slippage: pd.DataFrame | float | None = None) -> pd.Series:
    """仓位（尚未套用品种池）→ 扣掉手续费与滑点后的等权组合日收益。"""
    inpool = universe_mask(position.index, position.columns, universe)
    net = symbol_net(position.where(inpool), day_ret, fee,
                     inpool=inpool, slippage=slippage)
    return portfolio_return(net, universe)


def evaluate_signal_grid(factor_raw: pd.DataFrame,
                         name: str,
                         day_ret: pd.DataFrame,
                         universe: dict,
                         windows: list[int],
                         bands: list[tuple[float, float]],
                         fee: float,
                         std_window: int,
                         test_years: list[int],
                         lookback: int | None = None,
                         pct: float | None = None,
                         slippage: pd.DataFrame | float | None = None) -> pd.DataFrame:
    """在已经算好的原始因子上扫描 W 与分位轨。

    因子先乘方向符号，再做 MAD 标准化，再出信号。标准化对每个 (N, M) 只做一次。
    绩效是各测试年拼接后的一条曲线；折内数字另存，避免只看一个混合值。

    ``slippage`` 直通 :func:`run_book`。选参必须在**含滑点**的口径下做：
    滑点按换手计费，它会系统性地惩罚高换手的窗口，无滑点时选出来的 W 偏小。
    """
    if name not in C.ALL_SIGNS:
        raise KeyError(f"{name} 没有登记方向，拒绝扫描")
    signed = factor_raw * C.ALL_SIGNS[name]
    std = panel.mad_standardize(signed, window=std_window)
    rows = []
    for w in windows:
        for lo, hi in bands:
            sig = panel.quantile_signal(std, int(w), float(lo), float(hi))
            port = run_book(panel.execute_position(sig), day_ret, universe, fee,
                            slippage=slippage)
            stitched = folds.stitch_test_years(port, test_years)
            rec = metrics.performance(stitched)
            rec.update(lookback=lookback, pct=pct,
                       window=int(w), q_low=float(lo), q_high=float(hi))
            for y in test_years:
                fm = metrics.performance(port.loc[port.index.year == int(y)])
                rec[f'y{y}_ann_return'] = fm['ann_return']
                rec[f'y{y}_ret_risk'] = fm['ret_risk']
            rows.append(rec)
    return pd.DataFrame(rows)


def average_signals(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """等权平均。全 NaN 的位置保持 NaN。"""
    if not frames:
        raise ValueError('组合里没有因子')
    idx = frames[0].index
    cols = frames[0].columns
    for f in frames[1:]:
        idx = idx.union(f.index)
        cols = cols.union(f.columns)
    idx = idx.sort_values()
    acc = np.zeros((len(idx), len(cols)), dtype='float64')
    cnt = np.zeros((len(idx), len(cols)), dtype='float64')
    for f in frames:
        x = f.reindex(index=idx, columns=cols).to_numpy(dtype='float64')
        ok = np.isfinite(x)
        acc += np.where(ok, x, 0.0)
        cnt += ok
    out = np.full_like(acc, np.nan)
    good = cnt > 0
    out[good] = acc[good] / cnt[good]
    return pd.DataFrame(out, index=idx, columns=list(cols))


def member_signal(factor_raw: pd.DataFrame,
                  name: str,
                  window: int,
                  q_low: float,
                  q_high: float,
                  std_window: int) -> pd.DataFrame:
    if name not in C.ALL_SIGNS:
        raise KeyError(f"{name} 没有登记方向")
    signed = factor_raw * C.ALL_SIGNS[name]
    std = panel.mad_standardize(signed, window=std_window)
    return panel.quantile_signal(std, int(window), float(q_low), float(q_high))


def combo_signal_from_specs(loader,
                            members: list[str],
                            selection: dict,
                            std_window: int) -> pd.DataFrame:
    """loader(name, lookback, pct) → 原始因子宽表。缺选参的成员直接报错。"""
    missing = [m for m in members if m not in selection or 'center' not in selection[m]]
    if missing:
        raise KeyError(f"以下因子没有中心点参数，不能进组合: {missing}")
    frames = []
    for name in members:
        spec = selection[name]['center']
        raw = loader(name, spec.get('lookback'), spec.get('pct'))
        frames.append(member_signal(
            raw, name, spec['window'], spec['q_low'], spec['q_high'], std_window))
    return average_signals(frames)
