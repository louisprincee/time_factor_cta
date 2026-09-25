"""日频因子缓存（设计文档第 10.2 节的"离线脚本"那一段）。

为什么要有这一层
----------------
框架里的因子函数**不能**做分钟层计算：`mmt_last` 那套 `pickle.load(10.7GB)` +
`stack(level=0)`（约 1.74 亿行）必然 OOM。所以流程拆成两段——先离线把分钟表压成
日频因子落盘，框架内的因子函数只 read + 拼 open/openw。这一层就是前一段。

缓存布局
--------
    factor_daily/timestamp/{品种}.ext          时间戳族，**不依赖参数**，每品种一份
    factor_daily/N{N}_M{M}/{品种}.ext          持续期族，每个 (N, M) 组合一份

时间戳族单独放是有意的：它不含阈值参数，如果跟着每个 (N, M) 各存一遍，不但浪费，
更糟的是留下"它好像也依赖参数"的错觉，下游很容易写出按组合重算时间戳因子的代码。
分开存之后，`load_panel` 负责把两边按 trading_date 对齐拼起来。

断点续跑
--------
每个 (组合, 品种) 是独立文件，已存在即跳过。单品种全样本的
持续期计算是分钟级的，中断重来的代价很高，所以续跑不是锦上添花。

纪律
----
* 只从 ``config.RESEARCH_DIR`` 经 ``shard_io.load_shard`` 读分钟数据——样本外守卫
  只写在那一处。
* 落盘的是**原始因子值**，不乘方向符号。方向由 ``factors.apply_signs`` 在入模前施加，
  这样任何符号翻转都必须改 config 并留痕。
* 夜盘类因子在无夜盘品种上是 NaN，不是 0。这里不做任何 fillna。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..data import sessions, shard_io
from . import duration as D
from .factors import day_codes_of, duration_factors, timestamp_factors

TIMESTAMP_DIR_NAME = 'timestamp'


def combo_grid(lookbacks: list[int] | None = None,
               pcts: list[float] | None = None) -> list[tuple[int, float]]:
    """全部 (N, M) 阈值参数组合，默认 3 × 5 = 15 组（设计文档第 7 节）。"""
    lbs = lookbacks if lookbacks is not None else C.THRESHOLD_LOOKBACKS
    ps = pcts if pcts is not None else C.THRESHOLD_PCTS
    return [(int(n), float(m)) for n in lbs for m in ps]


def combo_name(lookback: int, pct: float) -> str:
    """组合目录名。M 用 %g 去掉无意义的尾零（52.5 -> M52.5，50.0 -> M50）。"""
    return f"N{int(lookback)}_M{pct:g}"


def combo_dir(lookback: int, pct: float, root: Path | None = None) -> Path:
    return (root or C.FACTOR_DAILY_DIR) / combo_name(lookback, pct)


def timestamp_dir(root: Path | None = None) -> Path:
    return (root or C.FACTOR_DAILY_DIR) / TIMESTAMP_DIR_NAME


# --------------------------------------------------------------------------
# 计算
# --------------------------------------------------------------------------
def build_symbol(symbol: str,
                 combos: list[tuple[int, float]] | None = None,
                 root: Path | None = None,
                 overwrite: bool = False,
                 fmt: str = 'auto') -> dict:
    """算一个品种的全部日频因子并落盘，返回该品种的汇总信息。

    分钟表只读一次、坐标只算一次、阈值网格只算一次——这三件事各自都比因子聚合贵，
    而它们在多个 (N, M) 之间是可以共享的（阈值网格的池化切分与 M 无关）。
    """
    combos = combos or combo_grid()
    root = root or C.FACTOR_DAILY_DIR
    lookbacks = sorted({n for n, _ in combos})
    pcts = sorted({m for _, m in combos})

    need_ts = overwrite or shard_io.find_shard(timestamp_dir(root), symbol) is None
    todo = [(n, m) for n, m in combos
            if overwrite or shard_io.find_shard(combo_dir(n, m, root), symbol) is None]
    if not need_ts and not todo:
        return {'symbol': symbol, 'skipped': True, 'combos_written': 0,
                'timestamp_written': False}

    df = sessions.add_intraday_coords(
        shard_io.load_shard(symbol, columns=C.FACTOR_FIELDS))
    codes, days = day_codes_of(df)
    info: dict = {'symbol': symbol, 'skipped': False, 'n_days': int(len(days)),
                  'n_bars': int(len(df)),
                  'first_day': str(days[0].date()) if len(days) else '',
                  'last_day': str(days[-1].date()) if len(days) else ''}

    if need_ts:
        ts = timestamp_factors(df)
        timestamp_dir(root).mkdir(parents=True, exist_ok=True)
        shard_io.save_shard(ts, timestamp_dir(root), symbol, fmt)
        info['timestamp_written'] = True
        info['n_timestamp_cols'] = int(ts.shape[1])
    else:
        info['timestamp_written'] = False

    n_dur_cols = 0
    if todo:
        price = df['closew'].to_numpy(dtype='float64')
        grid_p = D.rolling_threshold_grid(
            D.intraday_abs_diff(price, codes), codes, lookbacks, pcts)
        for n, m in todo:
            dur = duration_factors(df, lookback=n, pct=m, thr_p=grid_p[(n, m)])
            d = combo_dir(n, m, root)
            d.mkdir(parents=True, exist_ok=True)
            shard_io.save_shard(dur, d, symbol, fmt)
            # 各组合的列集合相同，循环里显式记下来，不靠循环变量漏出去
            n_dur_cols = int(dur.shape[1])
        info['n_duration_cols'] = n_dur_cols
    info['combos_written'] = len(todo)
    return info


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def _read(path: Path) -> pd.DataFrame:
    df = (pd.read_parquet(path) if path.suffix == shard_io.PARQUET_EXT
          else pd.read_pickle(path))
    df.index = pd.to_datetime(df.index)
    return df


def load_symbol(symbol: str, lookback: int, pct: float,
                root: Path | None = None,
                with_timestamp: bool = True) -> pd.DataFrame:
    """取单品种在某个 (N, M) 下的日频因子表（持续期族 + 时间戳族对齐拼接）。"""
    root = root or C.FACTOR_DAILY_DIR
    p = shard_io.find_shard(combo_dir(lookback, pct, root), symbol)
    if p is None:
        raise FileNotFoundError(
            f"缺少 {combo_name(lookback, pct)}/{symbol}，请先运行 step3_build_factors.py")
    out = _read(p)
    if with_timestamp:
        q = shard_io.find_shard(timestamp_dir(root), symbol)
        if q is None:
            raise FileNotFoundError(f"缺少 timestamp/{symbol}")
        ts = _read(q)
        # outer 对齐：两族的交易日理论上完全一致，用 outer 是为了一旦不一致能看见
        # NaN 而不是被静默截断
        out = out.join(ts, how='outer')
    return out.sort_index()


def load_panel(lookback: int, pct: float,
               symbols: list[str] | None = None,
               root: Path | None = None) -> pd.DataFrame:
    """长表 ``(trading_date, symbol) -> 各因子``。"""
    root = root or C.FACTOR_DAILY_DIR
    syms = symbols or shard_io.list_shards(combo_dir(lookback, pct, root))
    parts = []
    for s in syms:
        df = load_symbol(s, lookback, pct, root)
        df = df.assign(symbol=s)
        parts.append(df.set_index('symbol', append=True))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out.index.names = ['trading_date', 'symbol']
    return out


def load_wide(factor: str, lookback: int, pct: float,
              symbols: list[str] | None = None,
              root: Path | None = None) -> pd.DataFrame:
    """单个因子的宽表 ``index=trading_date, columns=symbol``——第 5/6 步的输入形状。"""
    root = root or C.FACTOR_DAILY_DIR
    syms = symbols or shard_io.list_shards(combo_dir(lookback, pct, root))
    cols = {}
    for s in syms:
        df = load_symbol(s, lookback, pct, root)
        if factor not in df.columns:
            raise KeyError(f"{s} 没有因子列 {factor}；可用: {list(df.columns)}")
        cols[s] = df[factor]
    return pd.DataFrame(cols).sort_index()


# --------------------------------------------------------------------------
# 验收
# --------------------------------------------------------------------------
def night_rows(panel: pd.DataFrame, night_class: pd.Series) -> np.ndarray:
    """把"有没有夜盘"对齐到 panel 的每一行，返回布尔数组。

    ``night_class`` 接受两种口径：

    * 以 ``(trading_date, symbol)`` 为 MultiIndex —— **正确口径**。多数商品是在
      某一年才开始挂夜盘的（玉米、淀粉、燃油、塑料、PP、PVC 都是 2019 年），
      "这个品种有没有夜盘"本身就不是一个逐品种的问题，而是逐 (品种, 交易日) 的。
    * 以品种为 index 的 ``{品种: bool}`` —— 退化口径，只在测试与小范围诊断里用。
      对上面那六个品种它会给出错误答案（判成无夜盘，于是 2019 年后真实存在的
      夜盘因子值看起来像"违规"）。

    对不上的格子当成"没有夜盘"（``NaN.eq(True)`` 即 False），宁可把一行排除在
    夜盘因子的统计之外，也不要把它当成有夜盘而拉低非空率。
    """
    if isinstance(night_class.index, pd.MultiIndex):
        return night_class.reindex(panel.index).eq(True).to_numpy()
    return night_class.reindex(
        panel.index.get_level_values('symbol')).eq(True).to_numpy()


def panel_health(panel: pd.DataFrame,
                 night_class: pd.Series | None = None) -> pd.DataFrame:
    """逐因子的缺失率与取值范围，第 3 步验收（设计文档第 11 节第 4 行）依据。

    参数
    ----
    night_class : 可选的夜盘标记，口径见 :func:`night_rows`。给了之后夜盘类因子的
        缺失率会**只在有夜盘的那些格子上**统计——否则一个 42 品种里有 8 个无夜盘的
        池子，vr_night 的整体缺失率必然 >19%，会把一个完全正确的结果判成不合格，
        进而诱导出 ``fillna(0)`` 这个致命修法。
    """
    rows = []
    for c in panel.columns:
        s = panel[c]
        v = s.dropna()
        rows.append({
            'factor': c,
            'scope': '全部',
            'n': int(len(s)),
            'non_null_ratio': float(v.size / len(s)) if len(s) else np.nan,
            'all_zero': bool(v.size and (v == 0).all()),
            'nunique': int(v.nunique()),
            'min': float(v.min()) if v.size else np.nan,
            'p50': float(v.median()) if v.size else np.nan,
            'max': float(v.max()) if v.size else np.nan,
        })
    return pd.DataFrame(rows).set_index('factor')


TIMEPOINT_FACTORS = ['ts_high', 'ts_low']


def check_acceptance(health: pd.DataFrame,
                     min_non_null: float = 0.90) -> pd.DataFrame:
    """把第 11 节第 4 行的三条验收标准落成一张判定表。

        1. 每个因子非空比例 > 90%（夜盘类因子只在有夜盘品种上算，见 panel_health）
        2. 夜盘类因子不得全为 0（结构性零值是上一轮把真因子压成噪声的原因）
        3. 时点类因子取值必须落在 [0, 1]
    """
    out = health.copy()
    out['pass_non_null'] = out['non_null_ratio'] > min_non_null
    out['pass_not_zero'] = ~out['all_zero']
    in01 = out.index.isin(TIMEPOINT_FACTORS)
    out['pass_range'] = True
    out.loc[in01, 'pass_range'] = (
        (out.loc[in01, 'min'] >= -1e-9) & (out.loc[in01, 'max'] <= 1 + 1e-9))
    out['passed'] = out['pass_non_null'] & out['pass_not_zero'] & out['pass_range']
    reasons = []
    for _, r in out.iterrows():
        bad = []
        if not r['pass_non_null']:
            bad.append(f"非空 {r['non_null_ratio']:.1%}")
        if not r['pass_not_zero']:
            bad.append('全为 0')
        if not r['pass_range']:
            bad.append(f"越界 [{r['min']:.3g}, {r['max']:.3g}]")
        reasons.append('+'.join(bad))
    out['reason'] = reasons
    return out


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')
