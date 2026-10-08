"""分钟分片、交易时段、日收益、品种池和合成数据。"""
from __future__ import annotations

import csv
import datetime as dt
import types
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


def read_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """按扩展名读一个分片或因子文件。"""
    path = Path(path)
    if path.suffix == PARQUET_EXT:
        return pd.read_parquet(path, columns=columns)
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


def load_validation_shard(symbol: str,
                          columns: list[str] | None = None) -> pd.DataFrame:
    """读取专用 2022 验证分片；验证年份外的任何日期都会被拒绝。"""
    directory = C.VALIDATION_DIR
    C.assert_validation_only(directory)
    p = find_shard(directory, symbol)
    if p is None:
        raise FileNotFoundError(f"找不到 {symbol} 的 2022 验证分片: {directory}")
    C.assert_validation_only(p)

    df = read_frame(p, columns)
    if 'trading_date' not in df.columns:
        raise KeyError(f"{symbol} 的验证分片缺少 trading_date")
    df['trading_date'] = pd.to_datetime(df['trading_date'])
    C.assert_validation_2022_dates(df['trading_date'], what=f"{symbol} 验证分片")
    return df.sort_index(kind='mergesort')


def load_oos_shard(symbol: str,
                   end,
                   columns: list[str] | None = None,
                   today=None) -> pd.DataFrame:
    """最终测试接口；日期已结束也不解锁，只有 C.final_evaluation() 块内、且不越过登记截止日才放行。"""
    end_day = C.to_date(end)
    if end_day < C.STRICT_OOS_START:
        raise C.HoldoutViolation(
            f"样本外窗口必须从 {C.STRICT_OOS_START} 起，收到的截止日期是 {end_day}"
        )
    C.assert_test_window_closed(end_day, today=today)
    C.assert_oos_research_locked(end_day)

    directory = C.HOLDOUT_DIR
    p = find_shard(directory, symbol)
    if p is None:
        raise FileNotFoundError(f"找不到 {symbol} 的样本外分片: {directory}")
    C.assert_holdout_only(p)
    df = read_frame(p, columns)
    if 'trading_date' not in df.columns:
        raise KeyError(f"{symbol} 的样本外分片缺少 trading_date")
    df['trading_date'] = pd.to_datetime(df['trading_date'])
    start = pd.Timestamp(C.STRICT_OOS_START)
    stop = pd.Timestamp(end_day)
    df = df[(df['trading_date'] >= start) & (df['trading_date'] <= stop)]
    C.assert_strict_oos_dates(df['trading_date'], what=f"{symbol} 样本外分片")
    if len(df) and pd.Timestamp(df['trading_date'].max()) > stop:
        raise C.HoldoutViolation(f"{symbol} 样本外分片超出截止日期 {end_day}")
    return df.sort_index(kind='mergesort')


# 品种类别 -> 夜盘收盘墙钟 (小时, 分钟)；None 表示无夜盘
# 夜盘 bar 数 = 收盘时刻 - 21:00，配合日盘 225 根得到 225/345/465/555 四种网格
NIGHT_CLASS = {
    'no_night': None,
    'night_2300': (23, 0),     # 120 分钟 -> 全日 345
    'night_0100': (25, 0),     # 240 分钟 -> 全日 465（25 = 次日 01:00）
    'night_0230': (26, 30),    # 330 分钟 -> 全日 555（26:30 = 次日 02:30）
}


def _session_minutes(night_class: str):
    """返回 (时段, 起小时, 起分钟, 止小时, 止分钟) 的构造参数。"""
    blocks = []
    end = NIGHT_CLASS[night_class]
    if end is not None:
        blocks.append(('night', 21, 0, end[0], end[1]))
    blocks.append(('am', 9, 0, 10, 15))
    blocks.append(('am', 10, 30, 11, 30))
    blocks.append(('pm', 13, 30, 15, 0))
    return blocks


def _day_timestamps(trading_day: pd.Timestamp, night_class: str,
                    prev_trading_day: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """构造某个 trading_date 的全部分钟时间戳。"""
    stamps = []
    prev = None if prev_trading_day is None else pd.Timestamp(prev_trading_day).normalize()
    for kind, h0, m0, h1, m1 in _session_minutes(night_class):
        if kind == 'night':
            if prev is None:
                continue          # 样本首日没有"前一交易日"，其夜盘不在数据范围内
            start = prev + pd.to_timedelta(h0, unit='h') + pd.to_timedelta(m0, unit='m')
            end = prev + pd.to_timedelta(h1, unit='h') + pd.to_timedelta(m1, unit='m')
        else:
            start = trading_day + pd.to_timedelta(h0, unit='h') + pd.to_timedelta(m0, unit='m')
            end = trading_day + pd.to_timedelta(h1, unit='h') + pd.to_timedelta(m1, unit='m')
        stamps.append(pd.date_range(start + pd.to_timedelta(1, unit='min'), end, freq='1min'))
    return pd.DatetimeIndex(np.concatenate([s.values for s in stamps])).sort_values()


def _vol_shape(n: int, has_night: bool) -> np.ndarray:
    """日内波动率的 U 形（开盘与收盘高、盘中低），夜盘段整体略高。"""
    u = np.linspace(0, 1, n)
    shape = 0.6 + 1.4 * (2 * u - 1) ** 2          # 端点 2.0，中点 0.6
    if has_night:
        shape[:int(n * 0.35)] *= 1.25             # 夜盘承接隔夜信息，波动更高
    return shape


def make_symbol(trading_days: pd.DatetimeIndex,
                night_class: str = 'night_2300',
                start_price: float = 3000.0,
                seed: int = 0,
                roll_every: int = 60,
                tick_ratio: float = 3e-4,
                vol_scale=1.0,
                listed_from: pd.Timestamp | None = None) -> pd.DataFrame:
    """生成单品种分钟表（含 15 个字段中因子需要的全部列 + dominant_id）。"""
    rng = np.random.default_rng(seed)
    frames = []
    price = start_price
    contract_no = 0
    tick = max(start_price * tick_ratio, 1e-8)
    has_night = NIGHT_CLASS[night_class] is not None

    for k, day in enumerate(trading_days):
        prev_day = pd.Timestamp(trading_days[k - 1]) if k > 0 else None
        idx = _day_timestamps(pd.Timestamp(day), night_class, prev_day)
        n = len(idx)
        if n == 0:
            continue
        if k % roll_every == 0:
            contract_no += 1

        # 随机游走 + 日内 U 形波动率 + 日间波动聚集，再离散化到最小变动价位
        day_vol = start_price * 2e-4 * rng.lognormal(0, 0.35)
        step = rng.normal(0, 1, n) * day_vol * _vol_shape(n, has_night)
        close = np.round((price + np.cumsum(step)) / tick) * tick
        price = float(close[-1])

        # 日内极值：在收盘价上下各加不到一个 tick 的随机幅度，再对齐 tick
        high = close + np.ceil(np.abs(rng.normal(0, 0.8, n))) * tick
        low = close - np.ceil(np.abs(rng.normal(0, 0.8, n))) * tick
        open_ = np.r_[close[0], close[:-1]]
        # 成交量同样呈 U 形，且取整（真实成交量是整数手，大量重复值影响量能持续期）
        scale = vol_scale(pd.Timestamp(day)) if callable(vol_scale) else float(vol_scale)
        vol = np.round(rng.lognormal(6.0, 0.8, n) * _vol_shape(n, has_night) * scale)
        turnover = vol * close

        # 未复权 vs 复权：给复权价加一个随合约递增的乘数，模拟 889 的拼接
        adj = 1.0 + 0.001 * contract_no
        df = pd.DataFrame({
            'open': open_, 'high': high, 'low': low, 'close': close,
            'volume': vol, 'total_turnover': turnover,
            'open_interest': rng.lognormal(9, 0.3, n),
            'trading_date': pd.Timestamp(day),
            'dominant_id': f'SYN{contract_no:04d}',
            'openw': open_ * adj, 'closew': close * adj,
            'highw': high * adj, 'loww': low * adj,
            'open_interest99': rng.lognormal(9, 0.3, n),
            'volume99': vol * 1.3,
        }, index=idx)
        frames.append(df)

    out = pd.concat(frames, axis=0).sort_index()
    if listed_from is not None:
        num = out.columns.difference(['trading_date', 'dominant_id'])
        out.loc[out.index < listed_from, num] = np.nan
    return out


# 2021-12-31（周五）夜盘归属 2022-01-04，是验证期第一根 bar 可能出现的最早墙钟日期
EARLIEST_WALL_CLOCK = dt.date(2021, 12, 31)
OFFSET_RTOL = 1e-6
MAX_MISMATCH_SHARE = 0.2      # 边界上同时换月的品种不会超过两成


def source_window_problems(panel: pd.DataFrame) -> list[str]:
    """宽面板（列为 品种 × 字段）是否只含 2022。"""
    problems = []
    idx = pd.DatetimeIndex(panel.index)
    if len(idx) == 0:
        return ["源文件没有任何行"]
    lo, hi = pd.Timestamp(EARLIEST_WALL_CLOCK), pd.Timestamp(C.STRICT_OOS_START)
    if idx.min() < lo or idx.max() >= hi:
        problems.append(f"时间戳范围 {idx.min()}..{idx.max()} 超出 [{lo.date()}, {hi.date()})")
    if 'trading_date' not in panel.columns.get_level_values(1):
        problems.append("源文件没有 trading_date 字段")
        return problems
    td = panel.xs('trading_date', axis=1, level=1)
    dates = pd.to_datetime(pd.Series(td.to_numpy().ravel())).dropna()
    if dates.empty:
        problems.append("trading_date 全部缺失")
        return problems
    first, last = dates.min(), dates.max()
    if first < pd.Timestamp(C.HOLDOUT_START) or last >= hi:
        problems.append(f"trading_date 范围 {first.date()}..{last.date()} 不全在 {C.VALIDATION_YEAR} 年")
    return problems


def adjustment_offset(df: pd.DataFrame, last: bool) -> float | None:
    """``closew − close``：加法复权相对原始价的偏移，取最后（或最先）一根两者都有的 bar。"""
    both = df[['closew', 'close']].dropna()
    if both.empty:
        return None
    row = both.iloc[-1] if last else both.iloc[0]
    return float(row['closew'] - row['close'])


def boundary_check(research: pd.DataFrame | None,
                   validation: pd.DataFrame) -> tuple[str, str]:
    """研究期末与验证期初的复权偏移是否衔接。"""
    if research is None or research.empty:
        return 'skip', "研究期没有分片（2022 年才有数据），无需衔接"
    tail = pd.DatetimeIndex(research.index).max()
    head = pd.DatetimeIndex(validation.index).min()
    if head <= tail:
        return 'fail', f"验证期首个时间戳 {head} 不晚于研究期末 {tail}"
    a, b = adjustment_offset(research, last=True), adjustment_offset(validation, last=False)
    if a is None or b is None:
        return 'skip', "边界处缺 close/closew，核对不了复权偏移"
    if np.isclose(a, b, rtol=OFFSET_RTOL, atol=1e-6):
        return 'ok', f"复权偏移 {a:.4f} 衔接"
    return 'mismatch', f"复权偏移 {a:.4f} → {b:.4f}：边界换月，或复权基准不同"


def boundary_verdict(statuses: dict[str, str]) -> tuple[bool, str]:
    """跨品种汇总：个别品种不衔接可以是边界换月；超过两成不衔接只能是整份数据换了复权基准。"""
    if any(s == 'fail' for s in statuses.values()):
        bad = sorted(k for k, s in statuses.items() if s == 'fail')
        return False, f"时间戳与研究期重叠: {bad}"
    checked = {k: s for k, s in statuses.items() if s in ('ok', 'mismatch')}
    off = sorted(k for k, s in checked.items() if s == 'mismatch')
    if checked and len(off) > MAX_MISMATCH_SHARE * len(checked):
        return False, (f"{len(off)}/{len(checked)} 个品种复权偏移不衔接，"
                       f"源文件的复权基准与研究期不同: {off}")
    note = f"复权偏移衔接 {len(checked) - len(off)}/{len(checked)}"
    return True, note + (f"；不衔接（按边界换月处理，请核对）: {off}" if off else "")


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


def forward_return(day_ret: pd.DataFrame) -> pd.DataFrame:
    """factor[t] 所预测的那段收益：day_ret[t+1]。"""
    return day_ret.shift(-1)


def holding_forward_return(day_ret: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """t 日信号在持仓期内的累计收益 ``day_ret[t+1] + … + day_ret[t+H]``。"""
    h = int(horizon)
    fwd = forward_return(day_ret)
    out = fwd if h == 1 else fwd.rolling(h, min_periods=h).sum().shift(-(h - 1))
    out = out.copy()
    out.attrs['horizon'] = h
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
        df = shard_io.load_shard(s, directory=directory,
                                 columns=list(DAILY_FIELDS) + ['trading_date'])
        day = daily_bars(df)
        C.assert_no_holdout_dates(day.index, what=f"{s} 日频行情")
        per_symbol[s] = day
    if not per_symbol:
        return {k: pd.DataFrame() for k in DAILY_FIELDS}
    return wide_by_field(per_symbol)


def load_roll_calendar(symbols: list[str], end, directory=None) -> dict[str, pd.DataFrame]:
    """各品种换月日与新旧合约代码，只保留 ``end`` 及以前的行。"""
    root = Path(directory) if directory is not None else Path(C.ROLL_DIR)
    end = pd.Timestamp(end)
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
                 include_validation: bool,
                 oos_end: dt.date | None = None,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """研究期 + （可选）2022 验证期 + （可选）已结束的样本外窗口，按时间拼成一张分钟表。"""
    columns = MINUTE_COLUMNS if columns is None else columns
    parts = []
    if shard_io.find_shard(C.RESEARCH_DIR, symbol) is not None:
        research = shard_io.load_shard(symbol, directory=C.RESEARCH_DIR, columns=columns)
        C.assert_no_holdout_dates(research['trading_date'], what=f"{symbol} 研究期")
        parts.append(research)
    if include_validation or oos_end is not None:
        if shard_io.find_shard(C.VALIDATION_DIR, symbol) is not None:
            parts.append(shard_io.load_validation_shard(symbol, columns=columns))
        elif oos_end is None:
            raise FileNotFoundError(f"找不到 {symbol} 的 2022 验证分片")
    if oos_end is not None:
        parts.append(shard_io.load_oos_shard(symbol, end=oos_end, columns=columns))
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
    out['night_class'] = [sessions.classify_night_bars(m) for m in out['night_bars_median']]
    out['has_night'] = out['night_class'] != 'no_night'
    return out


def collect_stats(symbols: list[str] | None = None,
                  directory=None,
                  verbose: bool = False) -> pd.DataFrame:
    """扫描研究期分片，产出长表 ``(symbol, year, ...)``。"""
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
    raw = json.loads(open(p, encoding='utf-8').read())
    return {int(k): list(v) for k, v in raw.items()}


# --------------------------------------------------------------------------
# 2022 验证与样本外：同一口径，第 y 年只用第 y-1 年
# --------------------------------------------------------------------------
def validation_universe() -> tuple[list[str], pd.DataFrame]:
    """2022 的池子，用第 2 步落盘的研究期逐年统计（即 2021 年）判定。"""
    path = C.UNIVERSE_DIR / 'yearly_stats.csv'
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先运行 step2_universe.py")
    stats = pd.read_csv(path)
    C.assert_no_holdout_dates(
        pd.to_datetime(stats['year'].astype(str) + '-01-01'), what='2022 品种池统计')
    detail = screen_year(stats, C.VALIDATION_YEAR)
    return sorted(detail.index[detail['passed']]), detail


def oos_years(end) -> list[int]:
    return list(range(C.STRICT_OOS_START.year, C.to_date(end).year + 1))


def oos_universe(symbols: list[str], end) -> tuple[dict[int, list[str]], pd.DataFrame]:
    """样本外各年的池子。"""
    end = C.to_date(end)
    C.assert_test_window_closed(end)
    daily = {}
    for sym in symbols:
        parts = []
        if shard_io.find_shard(C.VALIDATION_DIR, sym) is not None:
            parts.append(shard_io.load_validation_shard(sym, columns=STAT_COLUMNS))
        if shard_io.find_shard(C.HOLDOUT_DIR, sym) is not None:
            parts.append(shard_io.load_oos_shard(sym, end=end, columns=STAT_COLUMNS))
        if parts:
            daily[sym] = daily_stats(pd.concat(parts).sort_index(kind='mergesort'))
    stats = stats_from_daily(daily)
    if stats.empty:
        return {}, pd.DataFrame()
    universe, details = {}, []
    for year in oos_years(end):
        d = screen_year(stats, year)
        universe[year] = sorted(d.index[d['passed']]) if len(d) else []
        if len(d):
            details.append(d.reset_index().rename(columns={'index': 'symbol'}).assign(year=year))
    return universe, (pd.concat(details, ignore_index=True) if details else pd.DataFrame())


sessions = types.SimpleNamespace(tag_session=tag_session, add_intraday_coords=add_intraday_coords, day_bar_counts=day_bar_counts, classify_night_bars=classify_night_bars, classify_night_length=classify_night_length)
shard_io = types.SimpleNamespace(PARQUET_EXT=PARQUET_EXT, PICKLE_EXT=PICKLE_EXT, parquet_available=parquet_available, resolve_format=resolve_format, shard_path=shard_path, save_shard=save_shard, find_shard=find_shard, list_shards=list_shards, read_frame=read_frame, load_shard=load_shard, load_validation_shard=load_validation_shard, load_oos_shard=load_oos_shard)
synth = types.SimpleNamespace(NIGHT_CLASS=NIGHT_CLASS, _session_minutes=_session_minutes, _day_timestamps=_day_timestamps, _vol_shape=_vol_shape, make_symbol=make_symbol)
validation_import = types.SimpleNamespace(EARLIEST_WALL_CLOCK=EARLIEST_WALL_CLOCK, OFFSET_RTOL=OFFSET_RTOL, MAX_MISMATCH_SHARE=MAX_MISMATCH_SHARE, source_window_problems=source_window_problems, adjustment_offset=adjustment_offset, boundary_check=boundary_check, boundary_verdict=boundary_verdict)
bars = types.SimpleNamespace(MINUTE_COLUMNS=MINUTE_COLUMNS, DAILY_AGG=DAILY_AGG, DAILY_FIELDS=DAILY_FIELDS, daily_bars=daily_bars, forward_return=forward_return, holding_forward_return=holding_forward_return, wide_by_field=wide_by_field, load_daily_bars=load_daily_bars, load_roll_calendar=load_roll_calendar, load_minutes=load_minutes)
universe = types.SimpleNamespace(STAT_COLUMNS=STAT_COLUMNS, YEAR_STAT_COLS=YEAR_STAT_COLS, daily_stats=daily_stats, market_calendar=market_calendar, yearly_stats=yearly_stats, collect_stats=collect_stats, stats_from_daily=stats_from_daily, screen_year=screen_year, build_universe=build_universe, universe_turnover_report=universe_turnover_report, entries_and_exits=entries_and_exits, load_universe=load_universe, validation_universe=validation_universe, oos_years=oos_years, oos_universe=oos_universe)
