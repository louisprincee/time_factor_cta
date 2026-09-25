"""品种池：时点有效的逐年筛选（设计文档第 4 节）。

为什么不能用全样本选池
----------------------
用全样本流动性选池是这类研究里最常见、也最容易被忽略的前视偏差。文档里记着一个活
例子：ZC（动力煤）的日均成交额从 109.89 亿塌缩到几乎 0。全样本选池会把 ZC 选进
每一年的池子，包括它早已不可交易的年份，于是它在后期只贡献噪声——而这个噪声是
"因为我们知道它曾经很活跃"才被引入的，属于泄漏。

本模块的规则：第 y 年的池子只用 ``[y - UNIVERSE_LOOKBACK_YEARS, y-1]`` 的统计量判定，
并在 ``build_universe`` 里对这条做硬断言。

夜盘分组
--------
``has_night`` / ``night_class`` 是**逐年**的，不是逐品种的：多数商品在 2013-2016 年间
陆续开出夜盘，同一个品种在 2014 年无夜盘、2016 年有夜盘是常态。夜盘类因子在无夜盘的
品种-年份上必须取 NaN，绝不填 0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from . import sessions
from . import shard_io

# 计算品种池只需要这三列，其余列不读（parquet 下直接省 IO）
STAT_COLUMNS = ['closew', 'total_turnover', 'trading_date']

YEAR_STAT_COLS = ['n_days', 'valid_days', 'turnover_median', 'turnover_days',
                  'night_bars_median', 'bars_median']


def daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """单品种分钟表 -> 以 trading_date 为 index 的日度统计。

    刻意不调用 ``add_intraday_coords``：那会复制整张分钟表，而这里只需要夜盘 bar 的
    计数。对 75 个品种 × 12 年的全量扫描，省下的这份拷贝是有意义的。
    """
    td = pd.to_datetime(df['trading_date']).dt.normalize()
    is_night = pd.Series(
        sessions.tag_session(df.index).to_numpy() == C.SESSION_NIGHT,
        index=df.index)

    g = df.groupby(td, sort=True)
    out = pd.DataFrame({
        # min_count=1：全 NaN 的一天应得 NaN 而不是 0，否则"无数据"会伪装成"零成交"
        'turnover': g['total_turnover'].sum(min_count=1),
        'n_valid_close': g['closew'].count(),
        'n_bars': g.size(),
    })
    out['night_bars'] = is_night.groupby(td).sum()
    out.index.name = 'trading_date'
    return out


def market_calendar(daily_by_symbol: dict[str, pd.DataFrame]) -> pd.Series:
    """逐年的市场交易日数 = 全部品种 trading_date 的并集。

    这是数据完整度的**唯一正确分母**。用单个品种自己的天数当分母会让
    "只交易了 100 天的品种"显示为 100% 完整。
    """
    per_year: dict[int, set] = {}
    for d in daily_by_symbol.values():
        for y, idx in pd.Series(d.index, index=d.index).groupby(d.index.year):
            per_year.setdefault(int(y), set()).update(idx.to_numpy())
    return pd.Series({y: len(v) for y, v in sorted(per_year.items())},
                     name='calendar_days', dtype='int64')


def yearly_stats(daily: pd.DataFrame, calendar: pd.Series | None = None) -> pd.DataFrame:
    """日度统计 -> 逐年统计（index 为年份）。"""
    year = daily.index.year
    g = daily.groupby(year)
    out = pd.DataFrame({
        'n_days': g.size(),
        'valid_days': g['n_valid_close'].apply(lambda s: int((s > 0).sum())),
        # 中位数覆盖全部在场交易日（含零成交日）——僵尸品种正是靠这一点被压到门槛以下
        'turnover_median': g['turnover'].median(),
        'turnover_days': g['turnover'].count(),
        'night_bars_median': g['night_bars'].median(),
        'bars_median': g['n_bars'].median(),
    })
    out.index.name = 'year'
    if calendar is not None:
        out['calendar_days'] = calendar.reindex(out.index).to_numpy()
    else:
        out['calendar_days'] = out['n_days']
    out['valid_ratio'] = out['valid_days'] / out['calendar_days'].replace(0, np.nan)
    out['night_class'] = [sessions.classify_night_bars(m) for m in out['night_bars_median']]
    out['has_night'] = out['night_class'] != 'no_night'
    return out


def collect_stats(symbols: list[str] | None = None,
                  directory=None,
                  verbose: bool = False) -> pd.DataFrame:
    """扫描研究期分片，产出长表 ``(symbol, year, ...)``。

    只读 ``RESEARCH_DIR``，且经由 ``shard_io.load_shard``——样本外守卫在那里。
    """
    directory = directory or C.RESEARCH_DIR
    available = shard_io.list_shards(directory)
    if symbols is None:
        symbols = [s for s in C.COMMODITY_SYMBOLS if s in available]

    daily_by_symbol: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        if sym not in available:
            if verbose:
                print(f"[{i}/{len(symbols)}] {sym:<4} 无分片，跳过")
            continue
        df = shard_io.load_shard(sym, directory, columns=STAT_COLUMNS)
        daily_by_symbol[sym] = daily_stats(df)
        if verbose:
            d = daily_by_symbol[sym]
            print(f"[{i}/{len(symbols)}] {sym:<4} {len(d):>5} 交易日  "
                  f"{d.index.min().date()}..{d.index.max().date()}", flush=True)
        del df

    if not daily_by_symbol:
        return pd.DataFrame(columns=['symbol', 'year'] + YEAR_STAT_COLS)

    cal = market_calendar(daily_by_symbol)
    frames = []
    for sym, d in daily_by_symbol.items():
        y = yearly_stats(d, cal).reset_index()
        y.insert(0, 'symbol', sym)
        frames.append(y)
    stats = pd.concat(frames, ignore_index=True).sort_values(['symbol', 'year'])

    C.assert_no_holdout_dates(
        pd.to_datetime(stats['year'].astype(str) + '-01-01'), what='品种池统计')
    return stats.reset_index(drop=True)


# --------------------------------------------------------------------------
# 筛选
# --------------------------------------------------------------------------
def screen_year(stats: pd.DataFrame, year: int,
                lookback_years: int | None = None,
                min_turnover: float | None = None,
                min_valid_ratio: float | None = None,
                min_valid_days: int | None = None) -> pd.DataFrame:
    """判定第 ``year`` 年的池子，返回逐品种的判定明细。

    时点有效性由本函数保证：窗口是 ``[year - lookback, year - 1]``，
    绝不含 ``year`` 本身及以后。
    """
    lookback = C.UNIVERSE_LOOKBACK_YEARS if lookback_years is None else lookback_years
    min_turnover = C.MIN_DAILY_TURNOVER if min_turnover is None else min_turnover
    min_ratio = C.MIN_VALID_DAY_RATIO if min_valid_ratio is None else min_valid_ratio
    min_days = C.MIN_VALID_DAYS if min_valid_days is None else min_valid_days

    if lookback < 1:
        raise ValueError("lookback_years 必须 >= 1，否则用到了当年数据（前视）")
    lo, hi = year - lookback, year - 1
    assert hi < year, "筛选窗口越界：品种池不得使用目标年份及以后的数据"

    win = stats[(stats['year'] >= lo) & (stats['year'] <= hi)]
    if win.empty:
        return pd.DataFrame(columns=['symbol', 'turnover', 'valid_ratio',
                                     'valid_days', 'pass_turnover',
                                     'pass_complete', 'passed', 'reason'])

    g = win.groupby('symbol')
    out = pd.DataFrame({
        'turnover': g['turnover_median'].median(),
        # 完整度取窗口内最差的一年：宁可漏掉，不可放进一个半年没数据的品种
        'valid_ratio': g['valid_ratio'].min(),
        'valid_days': g['valid_days'].min(),
        'years_seen': g['year'].nunique(),
        'night_class': g['night_class'].last(),
    })
    out['pass_turnover'] = out['turnover'] >= min_turnover
    out['pass_complete'] = (out['valid_ratio'] >= min_ratio) & (out['valid_days'] >= min_days)
    out['passed'] = out['pass_turnover'] & out['pass_complete']

    reason = np.where(out['passed'], '',
                      np.where(~out['pass_turnover'] & ~out['pass_complete'], '成交额+完整度',
                               np.where(~out['pass_turnover'], '成交额', '完整度')))
    out['reason'] = reason
    out.insert(0, 'screen_window', f"{lo}-{hi}")
    return out.sort_index()


def build_universe(stats: pd.DataFrame,
                   years: list[int] | None = None,
                   **screen_kw) -> tuple[dict[int, list[str]], pd.DataFrame]:
    """逐年构建品种池。

    返回
    ----
    universe : {年份: [品种, ...]}
    detail   : 长表，含每个年份每个品种的判定依据（供人工核对与留痕）
    """
    years = C.UNIVERSE_YEARS if years is None else years
    universe: dict[int, list[str]] = {}
    details = []
    for y in years:
        if y > C.RESEARCH_END.year + 1:
            raise C.HoldoutViolation(
                f"{y} 年的池子需要 {y - 1} 年的数据，已越过研究期终点 {C.RESEARCH_END}")
        d = screen_year(stats, y, **screen_kw)
        universe[y] = sorted(d.index[d['passed']].tolist()) if len(d) else []
        if len(d):
            dd = d.reset_index().rename(columns={'index': 'symbol'})
            dd.insert(0, 'year', y)
            details.append(dd)
    detail = (pd.concat(details, ignore_index=True) if details
              else pd.DataFrame(columns=['year', 'symbol']))
    return universe, detail


def universe_turnover_report(stats: pd.DataFrame,
                             symbols: list[str] | None = None) -> pd.DataFrame:
    """逐品种的成交额年度轨迹（亿元），用于人工核对僵尸品种。

    验收项要求"人工核对无僵尸品种"，所以这张表必须打出来看，而不是只看池子大小。
    塌缩型品种的特征是轨迹单调下行并跌破门槛，而不是某一年偶然偏低。
    """
    s = stats if symbols is None else stats[stats['symbol'].isin(symbols)]
    tab = (s.pivot_table(index='symbol', columns='year', values='turnover_median')
           / 1e8).round(1)
    last = tab.ffill(axis=1).iloc[:, -1]
    peak = tab.max(axis=1)
    tab['峰值'] = peak.round(1)
    tab['末年'] = last.round(1)
    tab['末年/峰值'] = (last / peak.replace(0, np.nan)).round(3)
    return tab.sort_values('末年/峰值')


def entries_and_exits(universe: dict[int, list[str]]) -> pd.DataFrame:
    """逐年进出池明细。池子的换手率本身就是一个健康指标：
    每年换掉一半说明门槛太靠近品种的实际水平，结果会对门槛极其敏感。"""
    rows = []
    years = sorted(universe)
    for i, y in enumerate(years):
        cur = set(universe[y])
        prev = set(universe[years[i - 1]]) if i else set()
        rows.append({'year': y, 'n': len(cur),
                     'entered': ','.join(sorted(cur - prev)) if i else '(首年)',
                     'exited': ','.join(sorted(prev - cur)) if i else ''})
    return pd.DataFrame(rows)


def load_universe(path=None) -> dict[int, list[str]]:
    """读取已落盘的 universe_by_year.json，键转回 int。"""
    import json
    p = path or (C.UNIVERSE_DIR / 'universe_by_year.json')
    raw = json.loads(open(p, encoding='utf-8').read())
    return {int(k): list(v) for k, v in raw.items()}
