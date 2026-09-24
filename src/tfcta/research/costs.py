"""滑点：每次换手按 n_ticks × tick / 当年价位计单边比例。

tick 从分钟 close 的非零差分里估（频繁出现的最小档，逐年），不查交易所表。
用原始 close，不要用复权价。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..data import shard_io


# 频次下限：占最高频档的比例，以及一个绝对条数下限。二者取大。
TICK_FREQ_FRAC = 0.02
TICK_FREQ_MIN = 50
# float32 归并容差（相对）。同一个 tick 的浮点噪声远小于这个数，
# 相邻两档（1 个 tick 与 2 个 tick）远大于它，所以 1e-3 两边都够宽。
TICK_MERGE_TOL = 1e-3
# 年内非零差分少于这个数就不逐年估，退回全样本估计
TICK_MIN_DIFFS_PER_YEAR = 2000


def estimate_tick(close: np.ndarray) -> tuple[float, float]:
    """由分钟 ``close`` 估最小变动价位，返回 (估计值, 众数)。

    用**原始** close，不能用 closew：复权价乘过因子之后不在 tick 网格上，差分会退化成
    一堆互不相同的浮点数，估出来的 tick 毫无意义。

    取的是"出现得足够频繁的最小档"：先按相对容差把 float32 噪声归并成档，再剔掉频次
    低于下限的档（噪声与个别异常跳动都在这里被滤掉），剩下的取最小。众数一并返回，
    只作留痕——它对活跃品种会偏大一倍（最常见的分钟变动是两个 tick）。
    """
    d = np.abs(np.diff(np.asarray(close, dtype='float64')))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return np.nan, np.nan
    # 按相对容差分档：同一个 tick 的浮点变体落进同一个 key
    key = np.rint(np.log(d) / np.log1p(TICK_MERGE_TOL)).astype('int64')
    order = np.argsort(key, kind='stable')
    key, d = key[order], d[order]
    _, start, cnt = np.unique(key, return_index=True, return_counts=True)
    rep = np.minimum.reduceat(d, start)
    mode = float(rep[int(cnt.argmax())])
    keep = cnt >= max(TICK_FREQ_FRAC * float(cnt.max()), TICK_FREQ_MIN)
    # 一档都不够频繁（样本极短）时退回众数，而不是给 NaN 让成本静默变 0
    return (float(rep[keep].min()) if keep.any() else mode), mode


def build_tick_table(symbols: list[str]) -> pd.DataFrame:
    """逐 (品种, 年) 的 tick / 价位 / 比例成本。走 shard_io 因此受样本外守卫保护。"""
    rows = []
    for s in symbols:
        df = shard_io.load_shard(s, columns=['close', 'trading_date'])
        px = df['close'].to_numpy(dtype='float64')
        whole, whole_mode = estimate_tick(px)
        year = pd.DatetimeIndex(df['trading_date']).year.to_numpy()
        for y in sorted(set(year.tolist())):
            sel = year == y
            sub = px[sel]
            med = float(np.nanmedian(sub)) if sub.size else np.nan
            n_diff = int(np.isfinite(np.diff(sub)).sum() if sub.size > 1 else 0)
            if n_diff >= TICK_MIN_DIFFS_PER_YEAR:
                tick, mode, src = (*estimate_tick(sub), 'year')
            else:
                tick, mode, src = whole, whole_mode, 'symbol'
            rows.append({
                'symbol': s, 'year': int(y), 'tick': tick,
                'tick_mode': mode, 'tick_source': src, 'n_diff': n_diff,
                'median_close': med,
                'rate_per_tick': (tick / med
                                  if np.isfinite(tick) and med > 0 else np.nan),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        C.assert_no_holdout_dates(
            pd.to_datetime(out['year'].astype(str) + '-01-01'), what='tick 表')
    return out


def tick_table_path():
    return C.RESEARCH_OUT_DIR / 'tick_size.csv'


def load_tick_table(symbols: list[str], rebuild: bool = False) -> pd.DataFrame:
    """读缓存，缺品种就重建。分钟数据扫一遍不便宜，但一次就够。"""
    p = tick_table_path()
    if not rebuild and p.exists():
        got = pd.read_csv(p)
        if not set(symbols) - set(got['symbol']):
            return got
    out = build_tick_table(symbols)
    C.ensure_dirs()
    out.to_csv(p, index=False, encoding='utf-8-sig')
    return out


def slippage_wide(table: pd.DataFrame,
                  index: pd.Index,
                  columns: pd.Index,
                  n_ticks: float) -> pd.DataFrame:
    """(交易日 × 品种) 的单边比例滑点 = ``n_ticks × tick / 当年价位中位数``。

    与 ``fee`` 同样按换手计费，所以 -1 翻到 +1（换手 2）会被收两份——反手确实是
    两笔成交，各自穿一次价差，这个口径和手续费是一致的。

    整个品种都不在表里就直接报错——静默返回 NaN 会让那个品种的净值全变成 NaN，
    在等权组合里表现为"这个品种被跳过了"，成本反而变成 0，方向恰好是**低估**。
    品种在表里但缺某几年（上市晚、退池早）则按该品种最近的有效年份补齐：那些年份
    本来就没有仓位，补齐只是为了避免 ``NaN × 0 = NaN`` 把无换手的格子污染掉。
    """
    cols = list(columns)
    if table.empty or float(n_ticks) == 0.0:
        return pd.DataFrame(0.0, index=index, columns=cols)
    rate = table.set_index(['symbol', 'year'])['rate_per_tick'].sort_index()
    have = set(rate.index.get_level_values(0))
    gone = [c for c in cols if c not in have]
    if gone:
        raise KeyError(f"tick 表缺这些品种，滑点无法定价: {gone}")
    year = pd.DatetimeIndex(index).year
    span = sorted(set(year.tolist()) | set(rate.index.get_level_values(1).tolist()))
    data = {}
    for c in cols:
        per_year = rate.loc[c].reindex(span).ffill().bfill()
        data[c] = per_year.reindex(year).to_numpy(dtype='float64')
    return pd.DataFrame(data, index=index, columns=cols) * float(n_ticks)


def cost_summary(table: pd.DataFrame, n_ticks: float) -> pd.DataFrame:
    """逐品种的比例成本（bp），用来看横截面差异有多大。

    ``tick_changed`` 标出样本内换过最小变动价位的品种（如黄金）：那些品种的 ``tick``
    列只是各年的中位数，真正参与定价的是逐年的 ``rate_per_tick``。
    """
    if table.empty:
        return pd.DataFrame()
    g = table.groupby('symbol')
    # 比较之前先按归并容差量化，否则 float32 噪声（0.199951 对 0.199982）会把
    # 每个小 tick 品种都标成"换过 tick"，那这个标记就没用了
    tol = np.log1p(TICK_MERGE_TOL)
    quant = table.assign(
        _k=np.rint(np.log(table['tick'].where(table['tick'] > 0)) / tol))
    out = pd.DataFrame({
        'tick': g['tick'].median(),
        'tick_changed': quant.groupby('symbol')['_k'].nunique() > 1,
        'median_close': g['median_close'].median(),
        'bp_per_turnover': g['rate_per_tick'].mean() * float(n_ticks) * 1e4,
    })
    return out.sort_values('bp_per_turnover', ascending=False)
