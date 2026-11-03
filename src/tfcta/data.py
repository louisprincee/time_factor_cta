"""分钟分片、交易时段、日线、换月表和逐年品种池。"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from tfcta import config as C


def tag_session(index: pd.DatetimeIndex) -> pd.Series:
    """按墙钟时间给每根 bar 打 NIGHT / AM / PM 标签。"""
    idx = pd.DatetimeIndex(index)
    hour = idx.hour
    t = idx.time

    out = pd.Series(np.nan, index=idx, dtype=object)
    is_night = (hour >= C.NIGHT_START_HOUR) | (hour <= C.NIGHT_END_HOUR)
    is_am = np.array([C.AM_START <= x <= C.AM_END for x in t])
    is_pm = np.array([C.PM_START <= x <= C.PM_END for x in t])

    # 夜盘优先：夜盘时段与 AM/PM 的墙钟区间不重叠，但先赋值可防止边界意外
    out[is_pm] = C.SESSION_PM
    out[is_am] = C.SESSION_AM
    out[is_night] = C.SESSION_NIGHT
    return out


def add_intraday_coords(df: pd.DataFrame) -> pd.DataFrame:
    """为单品种分钟数据添加日内坐标列。"""
    if 'trading_date' not in df.columns:
        raise KeyError("缺少 trading_date 列——这是唯一合法的'日'定义，不能用 index.date 替代")

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    # 必须先按时间排序，否则 gamma 编号错乱
    out = out.sort_index(kind='mergesort')
    out['trading_date'] = pd.to_datetime(out['trading_date']).dt.normalize()

    # transform 取的列必须在建 grp 时就已存在。原来这里取的是 grp['gamma']，
    # 而 gamma 是建完 grp 之后才赋的——能跑通只是因为 groupby 持有的是 out 的引用，
    # 属于实现细节，跨 pandas 版本不保证。改成对 trading_date 自己 transform。
    grp = out.groupby('trading_date', sort=False)
    out['gamma'] = grp.cumcount() + 1
    out['n_bars'] = grp['trading_date'].transform('size')

    denom = (out['n_bars'] - 1).astype('float64')
    out['gamma_norm'] = np.where(denom > 0, (out['gamma'] - 1) / denom, np.nan)

    out['session'] = tag_session(out.index).to_numpy()
    return out


def day_bar_counts(df: pd.DataFrame) -> pd.DataFrame:
    """每个 trading_date 的分时段 bar 数，用于 step1 验收与结构诊断。"""
    if 'session' not in df.columns:
        df = add_intraday_coords(df)
    tab = (df.groupby(['trading_date', 'session'], dropna=False)
             .size().unstack(fill_value=0))
    for s in C.SESSIONS:
        if s not in tab.columns:
            tab[s] = 0
    tab['TOTAL'] = tab[C.SESSIONS].sum(axis=1)
    return tab[C.SESSIONS + ['TOTAL']]


def classify_night_bars(median_night_bars: float) -> str:
    """由夜盘 bar 数的中位数归类网格类别。"""
    med = float(median_night_bars)
    if not np.isfinite(med) or med < C.NIGHT_BARS_MIN:
        return 'no_night'
    if med <= C.NIGHT_BARS_2300:
        return 'night_2300'
    if med <= C.NIGHT_BARS_0100:
        return 'night_0100'
    return 'night_0230'


def classify_night_length(bar_counts: pd.DataFrame) -> str:
    """按夜盘 bar 数中位数归类品种的网格类别，用于验收比对。"""
    return classify_night_bars(bar_counts[C.SESSION_NIGHT].median())


PARQUET_EXT = '.parquet'
PICKLE_EXT = '.pkl'


def parquet_available() -> bool:
    for mod in ('pyarrow', 'fastparquet'):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def resolve_format(fmt: str = 'auto') -> str:
    if fmt == 'auto':
        return 'parquet' if parquet_available() else 'pickle'
    if fmt == 'parquet' and not parquet_available():
        raise RuntimeError(
            "指定了 parquet 但环境中没有 pyarrow/fastparquet。\n"
            "请 pip install pyarrow，或改用 --format pickle。"
        )
    return fmt


def shard_path(directory: Path, symbol: str, fmt: str = 'auto') -> Path:
    ext = PARQUET_EXT if resolve_format(fmt) == 'parquet' else PICKLE_EXT
    return Path(directory) / f"{symbol}{ext}"


def save_shard(df: pd.DataFrame, directory: Path, symbol: str,
               fmt: str = 'auto') -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    f = resolve_format(fmt)
    p = shard_path(directory, symbol, f)
    if f == 'parquet':
        df.to_parquet(p, compression='snappy')
    else:
        df.to_pickle(p, compression=None)
    return p


def find_shard(directory: Path, symbol: str) -> Path | None:
    """不管落盘时用的是哪种格式都能找到。"""
    for ext in (PARQUET_EXT, PICKLE_EXT):
        p = Path(directory) / f"{symbol}{ext}"
        if p.exists():
            return p
    return None


def list_shards(directory: Path) -> list[str]:
    d = Path(directory)
    if not d.exists():
        return []
    names = {p.stem for p in d.iterdir()
             if p.suffix in (PARQUET_EXT, PICKLE_EXT)}
    return sorted(names)


def read_frame(path: Path, columns: list[str] | None = None, filters=None) -> pd.DataFrame:
    """按扩展名读一个分片或因子文件。"""
    path = Path(path)
    if path.suffix == PARQUET_EXT:
        return pd.read_parquet(path, columns=columns, filters=filters)
    if filters:
        raise ValueError(f"只有 Parquet 分片支持按日期筛读: {path}")
    df = pd.read_pickle(path)
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    return df


def load_shard(symbol: str,
               directory: Path | None = None,
               columns: list[str] | None = None,
               verify_dates: bool = True) -> pd.DataFrame:
    """读取单品种分片，并强制执行样本外纪律。"""
    directory = Path(directory or C.RESEARCH_DIR)
    C.assert_research_only(directory)

    p = find_shard(directory, symbol)
    if p is None:
        raise FileNotFoundError(f"找不到 {symbol} 的分片: {directory}")
    C.assert_research_only(p)

    df = read_frame(p, columns)
    if 'trading_date' in df.columns:
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        if verify_dates:
            C.assert_no_holdout_dates(df['trading_date'], what=f"{symbol} 分片")
    return df.sort_index(kind='mergesort')


def load_oos_shard(symbol: str,
                   end,
                   columns: list[str] | None = None) -> pd.DataFrame:
    """读取商品样本外分片，只保留 OOS_START 到 end 的行。end 不得晚于 2025-12-31。"""
    end_day = C.to_date(end)
    if end_day < C.OOS_START or end_day > C.DEFAULT_OOS_END:
        raise C.HoldoutViolation(
            f"样本外窗口须在 {C.OOS_START}..{C.DEFAULT_OOS_END}，收到 {end_day}"
        )

    path = find_shard(C.OOS_DIR, symbol)
    if path is None:
        raise FileNotFoundError(f"找不到 {symbol} 的样本外分片: {C.OOS_DIR}")
    C.assert_oos_only(path)
    if path.suffix != PARQUET_EXT:
        raise ValueError(f"OOS 分片必须使用 Parquet 以限制读取日期: {path}")
    df = read_frame(
        path,
        columns,
        filters=[
            ("trading_date", ">=", pd.Timestamp(C.OOS_START)),
            ("trading_date", "<=", pd.Timestamp(end_day)),
        ],
    )
    if 'trading_date' not in df.columns:
        raise KeyError(f"{symbol} 的样本外分片缺少 trading_date")
    df['trading_date'] = pd.to_datetime(df['trading_date'])
    if df.index.has_duplicates:
        raise ValueError(f"{symbol} 的样本外分片有重复时间戳")
    stop = pd.Timestamp(end_day)
    C.assert_oos_dates(df['trading_date'], what=f"{symbol} 样本外分片")
    if len(df) and pd.Timestamp(df['trading_date'].max()) > stop:
        raise C.HoldoutViolation(f"{symbol} 样本外分片超出截止日期 {end_day}")
    return df.sort_index(kind='mergesort')


MINUTE_COLUMNS = list(dict.fromkeys([*C.FACTOR_FIELDS, *C.PRICE_FIELDS]))
DAILY_AGG = {
    'open': 'first', 'openw': 'first', 'close': 'last', 'closew': 'last',
    'highw': 'max', 'loww': 'min', 'volume': 'sum',
}
DAILY_FIELDS = tuple(DAILY_AGG)


def daily_bars(minute: pd.DataFrame) -> pd.DataFrame:
    """单品种分钟表 → 日线。"""
    td = pd.to_datetime(minute['trading_date']).dt.normalize()
    agg = {k: v for k, v in DAILY_AGG.items() if k in minute.columns}
    out = minute.groupby(td, sort=True).agg(agg)
    out.index.name = 'trading_date'
    return out


def wide_by_field(bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """{品种: 日线} → {字段: 交易日 × 品种}。"""
    fields = next(iter(bars_by_symbol.values())).columns
    return {
        field: pd.DataFrame({s: frame[field] for s, frame in bars_by_symbol.items()}).sort_index()
        for field in fields
    }


def load_daily_bars(symbols: list[str], directory=None) -> dict[str, pd.DataFrame]:
    """研究期日线，每个字段一张 ``trading_date × 品种`` 宽表。"""
    per_symbol = {}
    for s in symbols:
        df = load_shard(s, directory=directory,
                                 columns=list(DAILY_FIELDS) + ['trading_date'])
        per_symbol[s] = daily_bars(df)
    if not per_symbol:
        return {k: pd.DataFrame() for k in DAILY_FIELDS}
    return wide_by_field(per_symbol)


def load_roll_calendar(symbols: list[str], end, directory=None) -> dict[str, pd.DataFrame]:
    """各品种换月日与新旧合约代码，只保留 ``end`` 及以前的行。"""
    root = Path(directory) if directory is not None else Path(C.ROLL_DIR)
    end = pd.Timestamp(end)
    if end > pd.Timestamp(C.DEFAULT_OOS_END):
        raise C.HoldoutViolation("换月表不读 2026")
    out = {}
    for s in symbols:
        path = root / f"{s}.csv"
        if not path.exists():
            continue
        rows = []
        with path.open() as source:
            previous = None
            for row in csv.DictReader(source):
                date = pd.Timestamp(row['trading_date'])
                if previous is not None and date < previous:
                    raise ValueError(f'{path} 换月记录未按日期排序')
                if date > end:
                    break  # do not load later records into a dataframe
                previous = date
                rows.append(row)
        out[s] = pd.DataFrame(rows, columns=['trading_date','new_contract','prev_contract'])
        out[s]['trading_date'] = pd.to_datetime(out[s]['trading_date'])
    return out


def load_minutes(symbol: str, *,
                 oos_end: dt.date | None = None,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """研究期 + （可选）已结束的样本外窗口，按时间拼成一张分钟表。"""
    columns = MINUTE_COLUMNS if columns is None else columns
    parts = []
    if find_shard(C.RESEARCH_DIR, symbol) is not None:
        parts.append(load_shard(symbol, directory=C.RESEARCH_DIR, columns=columns))
    if oos_end is not None:
        parts.append(load_oos_shard(symbol, end=oos_end, columns=columns))
    if not parts:
        raise FileNotFoundError(f"找不到 {symbol} 的任何分片")
    minute = pd.concat(parts).sort_index(kind='mergesort')
    if minute.index.has_duplicates:
        raise ValueError(f"{symbol} 各时间分区有重复时间戳")
    return minute


# 计算品种池只需要这三列，其余列不读（parquet 下直接省 IO）
STAT_COLUMNS = ['closew', 'total_turnover', 'trading_date']

YEAR_STAT_COLS = ['n_days', 'valid_days', 'turnover_median', 'turnover_days',
                  'night_bars_median', 'bars_median']


def daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """单品种分钟表 -> 以 trading_date 为 index 的日度统计。"""
    td = pd.to_datetime(df['trading_date']).dt.normalize()
    is_night = pd.Series(
        tag_session(df.index).to_numpy() == C.SESSION_NIGHT,
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
    """逐年的市场交易日数 = 全部品种 trading_date 的并集。"""
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
    out['night_class'] = [classify_night_bars(m) for m in out['night_bars_median']]
    out['has_night'] = out['night_class'] != 'no_night'
    return out


def collect_stats(symbols: list[str] | None = None,
                  directory=None,
                  verbose: bool = False) -> pd.DataFrame:
    """扫描研究期分片，产出长表 ``(symbol, year, ...)``。"""
    directory = directory or C.RESEARCH_DIR
    available = list_shards(directory)
    if symbols is None:
        symbols = [s for s in C.COMMODITY_SYMBOLS if s in available]

    daily_by_symbol: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        if sym not in available:
            if verbose:
                print(f"[{i}/{len(symbols)}] {sym:<4} 无分片，跳过")
            continue
        df = load_shard(sym, directory, columns=STAT_COLUMNS)
        daily_by_symbol[sym] = daily_stats(df)
        if verbose:
            d = daily_by_symbol[sym]
            print(f"[{i}/{len(symbols)}] {sym:<4} {len(d):>5} 交易日  "
                  f"{d.index.min().date()}..{d.index.max().date()}", flush=True)
        del df

    stats = stats_from_daily(daily_by_symbol)
    if len(stats):
        C.assert_no_holdout_dates(
            pd.to_datetime(stats['year'].astype(str) + '-01-01'), what='品种池统计')
    return stats


def stats_from_daily(daily_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{品种: 日度统计} → 长表 ``(symbol, year, ...)``，完整度分母是全部品种的交易日并集。"""
    if not daily_by_symbol:
        return pd.DataFrame(columns=['symbol', 'year'] + YEAR_STAT_COLS)
    cal = market_calendar(daily_by_symbol)
    frames = []
    for sym, d in daily_by_symbol.items():
        y = yearly_stats(d, cal).reset_index()
        y.insert(0, 'symbol', sym)
        frames.append(y)
    stats = pd.concat(frames, ignore_index=True).sort_values(['symbol', 'year'])
    return stats.reset_index(drop=True)


# --------------------------------------------------------------------------
# 筛选
# --------------------------------------------------------------------------
def screen_year(stats: pd.DataFrame, year: int,
                lookback_years: int | None = None,
                min_turnover: float | None = None,
                min_valid_ratio: float | None = None,
                min_valid_days: int | None = None) -> pd.DataFrame:
    """判定第 ``year`` 年的池子，返回逐品种的判定明细。"""
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
    """逐年构建品种池。"""
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
    """逐品种的成交额年度轨迹（亿元），用于人工核对僵尸品种。"""
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
    """逐年进出池明细。"""
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
    with open(p, encoding='utf-8') as handle:
        raw = json.loads(handle.read())
    return {int(k): list(v) for k, v in raw.items()}


# --------------------------------------------------------------------------
# OOS：同一口径，第 y 年只用第 y-1 年
# --------------------------------------------------------------------------
def screen_params() -> dict:
    """第 2 步实际用的门槛；没有落盘时用 config 默认值。2022 和样本外的池子必须和研究期同一门槛。"""
    path = C.UNIVERSE_DIR / 'screen_params.json'
    if not path.exists():
        return {}
    import json
    return json.loads(path.read_text(encoding='utf-8'))


def oos_2022_universe() -> tuple[list[str], pd.DataFrame]:
    """2022 的池子，用研究期统计（即 2021 年）判定。"""
    path = C.UNIVERSE_DIR / 'yearly_stats.csv'
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先运行 python src/tfcta/prepare/universe.py")
    stats = pd.read_csv(path)
    C.assert_no_holdout_dates(
        pd.to_datetime(stats['year'].astype(str) + '-01-01'), what='2022 品种池统计')
    detail = screen_year(stats, C.OOS_START.year, **screen_params())
    return sorted(detail.index[detail['passed']]), detail


def oos_years(end) -> list[int]:
    return list(range(C.OOS_START.year + 1, C.to_date(end).year + 1))


def oos_universe(symbols: list[str], end) -> tuple[dict[int, list[str]], pd.DataFrame]:
    """样本外各年的池子。"""
    end = C.to_date(end)
    if end > C.DEFAULT_OOS_END:
        raise C.HoldoutViolation("品种池不读 2026")
    universe = {C.OOS_START.year: oos_2022_universe()[0]}
    daily = {}
    for sym in symbols:
        parts = []
        if find_shard(C.OOS_DIR, sym) is not None:
            parts.append(load_oos_shard(sym, end=end, columns=STAT_COLUMNS))
        if parts:
            daily[sym] = daily_stats(pd.concat(parts).sort_index(kind='mergesort'))
    stats = stats_from_daily(daily)
    if stats.empty:
        return {}, pd.DataFrame()
    details = []
    for year in oos_years(end):
        d = screen_year(stats, year, **screen_params())
        universe[year] = sorted(d.index[d['passed']]) if len(d) else []
        if len(d):
            details.append(d.reset_index().rename(columns={'index': 'symbol'}).assign(year=year))
    return universe, (pd.concat(details, ignore_index=True) if details else pd.DataFrame())
