"""交易成本：滑点（tick 表）与手续费（历史交易所费率 / 2026 期货公司费率表）。

滑点：每次换手按 n_ticks × tick / 当年价位计单边比例。tick 从分钟 close 的非零差分里估
（频繁出现的最小档，逐年），不查交易所表。用原始 close，不要用复权价。
手续费见文件后半部分。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ... import config as C
from ...data import shard_io


# 频次下限：占最高频档的比例，以及一个绝对条数下限。二者取大。
TICK_FREQ_FRAC = 0.02
TICK_FREQ_MIN = 50
# float32 归并容差（相对）。同一个 tick 的浮点噪声远小于这个数，
# 相邻两档（1 个 tick 与 2 个 tick）远大于它，所以 1e-3 两边都够宽。
TICK_MERGE_TOL = 1e-3
# 年内非零差分少于这个数就不单独估这一年，只用截至该年的价格，不用以后年份
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


def tick_rows(symbol: str, close, trading_date) -> list[dict]:
    """单品种逐年的 tick / 价位 / 比例成本。调用方负责只传入已经允许读取的分钟数据。

    某年差分太少时，只用截至该年的价格来估，不用以后年份。全样本估计会把后来的
    最小变动价位填进早年的成本。
    """
    px = np.asarray(close, dtype='float64')
    year = pd.DatetimeIndex(trading_date).year.to_numpy()
    rows = []
    prior_tick, prior_mode = np.nan, np.nan
    for y in sorted(set(int(v) for v in year.tolist())):
        sub = px[year == y]
        med = float(np.nanmedian(sub)) if sub.size else np.nan
        n_diff = int(np.isfinite(np.diff(sub)).sum() if sub.size > 1 else 0)
        if n_diff >= TICK_MIN_DIFFS_PER_YEAR:
            tick, mode, src = (*estimate_tick(sub), 'year')
        else:
            tick, mode = estimate_tick(px[year <= y])
            if not np.isfinite(tick):
                tick, mode = prior_tick, prior_mode
            src = 'through_year'
        if np.isfinite(tick):
            prior_tick, prior_mode = tick, mode
        rows.append({
            'symbol': symbol, 'year': int(y), 'tick': tick,
            'tick_mode': mode, 'tick_source': src, 'n_diff': n_diff,
            'median_close': med,
            'rate_per_tick': (tick / med
                              if np.isfinite(tick) and med > 0 else np.nan),
        })
    return rows


def build_tick_table(symbols: list[str]) -> pd.DataFrame:
    """逐 (品种, 年) 的 tick / 价位 / 比例成本。走 shard_io 因此受样本外守卫保护。"""
    rows = []
    for s in symbols:
        df = shard_io.load_shard(s, columns=['close', 'trading_date'])
        rows.extend(tick_rows(s, df['close'], df['trading_date']))
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
    """(交易日 × 品种) 的单边比例滑点 = ``n_ticks × tick / 上一年价位中位数``。

    第 y 年用第 y-1 年的 ``rate_per_tick``：当年价位中位数要到年底才知道，直接用
    会让成本随当年价格事后调整。品种的第一年没有上一年，只能用当年自己的估计。

    与 ``fee`` 同样按换手计费，所以 -1 翻到 +1（换手 2）会被收两份——反手确实是
    两笔成交，各自穿一次价差，这个口径和手续费是一致的。

    整个品种都不在表里就直接报错——静默返回 NaN 会让那个品种的净值全变成 NaN，
    在等权组合里表现为"这个品种被跳过了"，成本反而变成 0，方向恰好是**低估**。
    品种在表里但缺某几年：缺口用此前最近一年的费率向前填；第一条记录之前（以及第一年）
    的日期才用最早一年回填。那些日期本来没有仓位，补齐是为了避免 ``NaN × 0 = NaN``。
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
        per_year = rate.loc[c].reindex(span).ffill().shift(1).bfill()
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


def research_slippage(symbols: list[str],
                      index: pd.Index,
                      n_ticks: float,
                      rebuild: bool = False) -> tuple[pd.DataFrame | None, str]:
    """研究期滑点宽表 + 一行说明。``n_ticks == 0`` 时不建 tick 表，只扣手续费。"""
    if float(n_ticks) == 0.0:
        return None, '滑点 0 个 tick（对照档，仅供比较，不得用于选参或结论）'
    tab = load_tick_table(symbols, rebuild=rebuild)
    wide = slippage_wide(tab, index, list(symbols), n_ticks)
    bp = cost_summary(tab[tab['symbol'].isin(symbols)], n_ticks)['bp_per_turnover']
    note = (f"滑点 {float(n_ticks):g} 个 tick，按换手计费；"
            f"品种间 {bp.min():.1f}~{bp.max():.1f}bp，中位 {bp.median():.1f}bp")
    return wide, note


# --------------------------------------------------------------------------
# 手续费
# --------------------------------------------------------------------------
# 两套逐品种口径，都分按比例（占成交额）与按手（元/手），开仓 / 平昨 / 平今分开：
# - "hist"（研究期主口径）：RQData ``futures.get_trading_parameters`` 的逐日交易所标准，
#   取当日主力合约（``futures.get_dominant``），再加期货公司加收——按手 +0.01 元，
#   按比例 ×1.01（与 2026 表 "开户总" = 交易所 + 0.01 同口径）；交易所收 0 时按 +0.01 元/手。
#   缓存在 ``RESEARCH_OUT_DIR/fee_history.csv``，只取 ≤2021。
# - "2026"：用户提供的 2026 年期货公司手续费表（已含加收）。比例以小数给出（万 1.01 → 1.01e-4）；
#   "平今" 空白视为与开仓相同；ZC 表中 151.5 元/手是 2026 年的限制性收费，按 4.01 元/手；
#   表里没有平昨，平昨按开仓。
# 合约乘数用 RQData 合约日线 ``total_turnover / (volume × close)`` 核对过（2016–2021）。

# 品种: (开仓比例, 开仓元/手, 平今比例, 平今元/手)
FEES_2026: dict[str, tuple[float, float, float, float]] = {
    # 大商所 农产品 / 化工 / 黑色
    "A": (0.0, 2.02, 0.0, 2.02), "B": (0.0, 1.01, 0.0, 1.01),
    "C": (0.0, 1.21, 0.0, 1.21), "CS": (0.0, 1.52, 0.0, 1.52),
    "M": (0.0, 1.52, 0.0, 1.52), "Y": (0.0, 2.53, 0.0, 2.53),
    "P": (0.0, 2.53, 0.0, 2.53), "JD": (1.52e-4, 0.0, 1.52e-4, 0.0),
    "L": (0.0, 1.01, 0.0, 1.01), "V": (0.0, 1.01, 0.0, 1.01),
    "PP": (0.0, 1.01, 0.0, 1.01), "EG": (0.0, 3.03, 0.0, 3.03),
    "I": (1.01e-4, 0.0, 1.01e-4, 0.0), "J": (1.01e-4, 0.0, 1.4e-4, 0.0),
    "JM": (1.01e-4, 0.0, 1.01e-4, 0.0),
    # 上期所 / 能源中心
    "CU": (0.51e-4, 0.0, 1.0e-4, 0.0), "AL": (0.0, 3.03, 0.0, 0.01),
    "ZN": (0.0, 3.03, 0.0, 0.01), "PB": (0.404e-4, 0.0, 0.0, 0.01),
    "NI": (0.0, 3.03, 0.0, 3.03), "SN": (0.0, 3.03, 0.0, 3.03),
    "AU": (0.0, 10.1, 0.0, 0.01), "AG": (0.101e-4, 0.0, 0.101e-4, 0.0),
    "RB": (1.01e-4, 0.0, 1.01e-4, 0.0), "HC": (1.01e-4, 0.0, 1.01e-4, 0.0),
    "RU": (0.0, 3.03, 0.0, 0.01), "BU": (0.51e-4, 0.0, 0.0, 0.01),
    "FU": (0.101e-4, 0.0, 0.0, 0.01), "SP": (0.51e-4, 0.0, 0.0, 0.01),
    "SC": (0.0, 20.2, 0.0, 0.01),
    # 郑商所
    "AP": (0.0, 5.05, 0.0, 10.0), "CF": (0.0, 4.34, 0.0, 0.01),
    "SR": (0.0, 2.02, 0.0, 0.01), "TA": (0.0, 3.03, 0.0, 0.01),
    "RM": (0.0, 1.52, 0.0, 0.01), "OI": (0.0, 2.02, 0.0, 2.02),
    "FG": (0.0, 2.02, 0.0, 2.02), "SF": (0.0, 2.02, 0.0, 0.01),
    "SM": (0.0, 2.02, 0.0, 0.01), "MA": (1.01e-4, 0.0, 1.01e-4, 0.0),
    "ZC": (0.0, 4.01, 0.0, 4.01),
}

MULTIPLIER: dict[str, float] = {
    "A": 10, "B": 10, "C": 10, "CS": 10, "M": 10, "Y": 10, "P": 10, "JD": 10,
    "L": 5, "V": 5, "PP": 5, "EG": 10, "I": 100, "J": 100, "JM": 60,
    "CU": 5, "AL": 5, "ZN": 5, "PB": 5, "NI": 1, "SN": 1, "AU": 1000, "AG": 15,
    "RB": 10, "HC": 10, "RU": 10, "BU": 10, "FU": 10, "SP": 10, "SC": 1000,
    "AP": 10, "CF": 5, "SR": 10, "TA": 5, "RM": 10, "OI": 10, "FG": 20,
    "SF": 5, "SM": 5, "MA": 10, "ZC": 100,
}

FEE_BROKER_FIX = 0.01      # 期货公司按手加收（元/手）
FEE_BROKER_REL = 0.01      # 期货公司按比例加收（相对交易所费率）
FEE_HISTORY_START = "2015-01-01"
FEE_HISTORY_END = "2021-12-31"
FEE_FIELDS = ["commission_type", "open_commission", "close_commission", "close_commission_today"]
_LEG_COLUMNS = ("open_commission", "close_commission", "close_commission_today")
_FEE_STATE: dict = {}


@dataclass(frozen=True)
class FeeModel:
    """``schedule``："hist" 历史交易所费率（研究期主口径）或 "2026" 期货公司费率表。

    ``scale`` 乘在全部费用上；``close_today_as_open`` 取消平今优惠（平今按开仓收）。
    """
    schedule: str = "2026"
    scale: float = 1.0
    close_today_as_open: bool = False


def fee_history_path():
    return C.RESEARCH_OUT_DIR / "fee_history.csv"


def fetch_fee_history(symbols: list[str], start: str = FEE_HISTORY_START,
                      end: str = FEE_HISTORY_END) -> pd.DataFrame:
    """逐 (品种, 交易日) 的主力合约与交易所手续费标准（不含期货公司加收）。只取 ≤2021。"""
    import rqdatac
    from ...rq_auth import rqdata_credentials
    if pd.Timestamp(end) > pd.Timestamp(FEE_HISTORY_END):
        raise ValueError("手续费表只取研究期（≤2021）")
    args = rqdata_credentials()
    rqdatac.init(*args) if args else rqdatac.init()
    frames = []
    for s in symbols:
        dom = rqdatac.futures.get_dominant(s, start, end)
        if dom is None or len(dom) == 0:
            raise KeyError(f"RQData 没有 {s} 的主力合约")
        key = pd.DataFrame({"trading_date": pd.to_datetime(dom.index), "contract": dom.to_numpy()})
        ids = sorted(key["contract"].dropna().unique())
        tp = rqdatac.futures.get_trading_parameters(ids, start, end, fields=FEE_FIELDS)
        tp = tp.reset_index().rename(columns={"order_book_id": "contract"})
        tp["trading_date"] = pd.to_datetime(tp["trading_date"])
        frames.append(key.merge(tp, on=["contract", "trading_date"], how="left").assign(symbol=s))
    out = pd.concat(frames, ignore_index=True)[["symbol", "trading_date", "contract", *FEE_FIELDS]]
    C.assert_no_holdout_dates(out["trading_date"], what="手续费表")
    return out


def load_fee_history(symbols: list[str], rebuild: bool = False) -> pd.DataFrame:
    """读缓存，缺品种就补抓；写回缓存后清空进程内的解析结果。"""
    p = fee_history_path()
    if not rebuild and p.exists():
        got = pd.read_csv(p, parse_dates=["trading_date"])
        missing = sorted(set(symbols) - set(got["symbol"]))
        if not missing:
            return got
        got = pd.concat([got, fetch_fee_history(missing)], ignore_index=True)
    else:
        got = fetch_fee_history(symbols)
    C.ensure_dirs()
    got.to_csv(p, index=False, encoding="utf-8-sig")
    _FEE_STATE.clear()
    return got


def set_fee_history(table: pd.DataFrame | None) -> None:
    """指定进程内使用的历史手续费表（测试用）；None 表示下次从缓存文件读。"""
    _FEE_STATE.clear()
    if table is not None:
        _FEE_STATE["__table__"] = table


def _hist_table() -> pd.DataFrame:
    if "__table__" not in _FEE_STATE:
        p = fee_history_path()
        if not p.exists():
            raise FileNotFoundError(f"没有历史手续费缓存 {p}，先调 load_fee_history(symbols)")
        _FEE_STATE["__table__"] = pd.read_csv(p, parse_dates=["trading_date"])
    return _FEE_STATE["__table__"]


def _hist_parsed(symbol: str) -> dict:
    """单品种的逐日费率：每一腿拆成 (比例, 元/手) 两列，已含期货公司加收。

    主力合约某天查不到交易参数时先向前、再向后填——手续费标准变动很少，缺一天用相邻日即可。
    """
    key = "sym:" + symbol
    if key in _FEE_STATE:
        return _FEE_STATE[key]
    tab = _hist_table()
    sub = tab[tab["symbol"] == symbol].sort_values("trading_date")
    if sub.empty:
        raise KeyError(f"历史手续费表缺 {symbol}")
    sub = sub.ffill().bfill()
    money = sub["commission_type"].astype(str).str.lower().str.contains("money").to_numpy()
    out = {"dates": sub["trading_date"].to_numpy(dtype="datetime64[ns]")}
    for col in _LEG_COLUMNS:
        v = sub[col].to_numpy(dtype="float64")
        rate = np.where(money, v * (1.0 + FEE_BROKER_REL), 0.0)
        fix = np.where(money, 0.0, v) + np.where(~money | (v == 0), FEE_BROKER_FIX, 0.0)
        out[col] = (rate, fix)
    _FEE_STATE[key] = out
    return out


def has_schedule(symbol: str, schedule: str = "2026") -> bool:
    if symbol not in MULTIPLIER:
        return False
    if schedule == "2026":
        return symbol in FEES_2026
    try:
        _hist_parsed(symbol)
    except (KeyError, FileNotFoundError):
        return False
    return True


def fee_legs(symbol: str, raw_price, model: FeeModel = FeeModel(),
             close_today: bool = True, dates=None):
    """(开仓, 平仓) 单边费用，占成交额比例。按手收费折成比例要用**原始**价格。

    ``close_today`` False 表示隔夜平仓（平昨）。"hist" 口径必须给 ``dates``（交易日），
    取当日（或此前最近一日）的交易所标准；"2026" 表没有平昨，平昨按开仓。
    """
    price = np.asarray(raw_price, dtype="float64")
    notional = price * float(MULTIPLIER[symbol])
    if model.schedule == "2026":
        o_rate, o_fix, ct_rate, ct_fix = FEES_2026[symbol]
        open_leg = o_rate + o_fix / notional
        close_leg = (ct_rate + ct_fix / notional
                     if close_today and not model.close_today_as_open else open_leg)
    elif model.schedule == "hist":
        if dates is None:
            raise ValueError("历史手续费口径需要交易日 dates")
        p = _hist_parsed(symbol)
        d = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(dates))).to_numpy(dtype="datetime64[ns]")
        i = np.clip(np.searchsorted(p["dates"], d, side="right") - 1, 0, len(p["dates"]) - 1)
        if np.ndim(dates) == 0:
            i = i[0]

        def leg(col):
            rate, fix = p[col]
            return rate[i] + fix[i] / notional

        open_leg = leg("open_commission")
        col = ("open_commission" if model.close_today_as_open
               else "close_commission_today" if close_today else "close_commission")
        close_leg = leg(col)
    else:
        raise ValueError(f"未知手续费口径: {model.schedule}")
    s = float(model.scale)
    if np.ndim(open_leg) == 0 and np.ndim(close_leg) == 0:
        return float(open_leg) * s, float(close_leg) * s
    return np.asarray(open_leg) * s, np.asarray(close_leg) * s


def round_trip(symbol: str, raw_price, model: FeeModel = FeeModel(),
               close_today: bool = True, dates=None):
    """一开一平的手续费合计（比例）。"""
    o, c = fee_legs(symbol, raw_price, model, close_today=close_today, dates=dates)
    return o + c


def fee_wide(index: pd.Index, columns, model: FeeModel, tick_table: pd.DataFrame,
             fallback: float | None = None) -> pd.DataFrame:
    """日频书用的 (交易日 × 品种) 单边手续费比例 = (开仓 + 平昨) / 2。

    与 ``slippage_wide`` 同口径：第 y 年按第 y-1 年的价位中位数把按手收费折成比例，
    按换手计费，可以直接加在滑点宽表上。没有费率的品种用 ``fallback``（单边比例），
    不给就报错。
    """
    cols = list(columns)
    px = tick_table.set_index(["symbol", "year"])["median_close"].sort_index()
    year = pd.DatetimeIndex(index).year
    span = sorted(set(year.tolist()) | set(px.index.get_level_values(1).tolist()))
    data = {}
    for c in cols:
        if not has_schedule(c, model.schedule):
            if fallback is None:
                raise KeyError(f"{c} 没有 {model.schedule} 手续费")
            data[c] = np.full(len(index), float(fallback) * float(model.scale))
            continue
        price = px.loc[c].reindex(span).ffill().shift(1).bfill().reindex(year).to_numpy("float64")
        o, cl = fee_legs(c, price, model, close_today=False, dates=index)
        data[c] = (np.asarray(o) + np.asarray(cl)) / 2.0
    return pd.DataFrame(data, index=index, columns=cols)
