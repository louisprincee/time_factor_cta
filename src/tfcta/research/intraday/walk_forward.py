"""日内开盘区间突破（ORB）元标签策略：数据、候选交易、特征、池级组合与嵌套走步件。

口径：
- 池子：``U.load_universe()`` 逐年条件池（第 y 年由 y-1 年统计选出），全部板块，不逐品种调参。
- 候选交易：30 根 bar 的开盘区间突破，尾盘平。开盘可以是交易日第一根（含夜盘），
  或日盘第一根。只做多、只做空两条单边路径分别撮合。
- 成本逐笔拆成 不含滑点收益 / 每 tick 往返滑点 / 手续费（历史平今、历史平昨、2026 表）三块，
  情景层（``SCENARIOS``）重组，不重跑撮合。主口径 = 历史手续费 + 每边 1 tick。
- 特征只用入场时已知的信息：开盘区间本身（前 m 根 bar）、盘前 ATR 与昨日区间、入场价、
  滞后一日的日频因子。带方向的特征乘以交易方向。
- 池级组合：当年池内品种按 ``0.01 / ATR%`` 风险缩放后等权，没交易的可交易日记 0。
- 研究只读研究期分片（≤2021）；验证期 / 样本外的分钟数据由调用方传进 ``orb_candidates``。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ... import config as C
from ...data import shard_io
from ..backtest import costs
from .engine import (BacktestConfig, day_open_offset, opening_range, performance, prepare,
                     run_intraday, tick_per_day)
from .strategies import StrategyConfig

# --------------------------------------------------------------------------
# 数据与品种池
# --------------------------------------------------------------------------
MINUTE_COLUMNS = ["open", "openw", "close", "closew", "highw", "loww", "volume", "trading_date"]
RESEARCH_LAST = pd.Timestamp("2021-12-31")


def load_minutes(symbol: str, start=None, end=None) -> pd.DataFrame:
    """研究期分钟分片（≤2021）。"""
    minute = shard_io.load_shard(symbol, C.RESEARCH_DIR, columns=MINUTE_COLUMNS)
    td = pd.to_datetime(minute["trading_date"])
    keep = td <= min(pd.Timestamp(end), RESEARCH_LAST) if end else td <= RESEARCH_LAST
    if start:
        keep &= td >= pd.Timestamp(start)
    return minute.loc[keep]


def tick_by_year(symbols: list[str]) -> dict[str, dict[int, float]]:
    """{品种: {年份: 最小变动价位}}，来自研究期 tick 表（≤2021）。"""
    return tick_dict(costs.load_tick_table(symbols))


def tick_dict(table: pd.DataFrame) -> dict[str, dict[int, float]]:
    return {s: dict(zip(g["year"].astype(int), g["tick"].astype(float)))
            for s, g in table.groupby("symbol")}


def membership(members: dict[int, list[str]], index: pd.DatetimeIndex,
               columns: list[str]) -> pd.DataFrame:
    year = pd.DatetimeIndex(index).year
    return pd.DataFrame({s: [s in members.get(int(y), ()) for y in year] for s in columns},
                        index=index)


def risk_weights(atr_pct: pd.DataFrame, risk_per_atr: float = 0.01,
                 cap: float = 3.0) -> pd.DataFrame:
    """每个品种名义敞口 = ``risk_per_atr / ATR%``：1 个 ATR 的波动约等于账户 1%。

    ATR 在开盘前已知；上限 ``cap`` 倍名义，防止极低波动期放大杠杆。
    """
    return (risk_per_atr / atr_pct.where(atr_pct > 0)).clip(upper=cap)


def yearly_sharpe(ret: pd.Series) -> dict[int, float]:
    return {int(y): performance(g)["sharpe"] for y, g in ret.groupby(ret.index.year)}


def summarize(ret_net: pd.Series) -> dict:
    net = performance(ret_net)
    row = {"net_sharpe": net["sharpe"], "net_ann": net["ann_return"],
           "net_vol": net["ann_vol"], "net_mdd": net["max_drawdown"], "n_days": net["n_days"]}
    ys = yearly_sharpe(ret_net)
    row.update({f"sr_{y}": v for y, v in ys.items()})
    row["pos_years"] = int(sum(np.isfinite(v) and v > 0 for v in ys.values()))
    return row


# --------------------------------------------------------------------------
# 成本情景
# --------------------------------------------------------------------------
HIST_FEES = costs.FeeModel("hist")
FEES_2026 = costs.FeeModel("2026")
# (每边滑点 tick 数, 手续费倍数, 手续费列)。手续费列：
#   fee    历史交易所费率（当日主力合约）+ 期货公司加收，开仓 + 平今 —— 主口径
#   fee_oo 同上但平仓按平昨收（没有平今加价 / 优惠）
#   fee26  2026 期货公司费率表，开仓 + 平今
# 滑点只做 1、2 tick：主力合约用市价单，1 tick 是下限。
SCENARIOS = {
    "gross": (0.0, 0.0, "fee"),
    "main": (1.0, 1.0, "fee"),
    "slip2": (2.0, 1.0, "fee"),
    "fee_x2": (1.0, 2.0, "fee"),
    "fee_2026": (1.0, 1.0, "fee26"),
    "close_yday": (1.0, 1.0, "fee_oo"),
    "fee_only": (0.0, 1.0, "fee"),
}


def scenario_net(trades: pd.DataFrame, scenario: str) -> np.ndarray:
    ticks, scale, fee_col = SCENARIOS[scenario]
    return (trades["gross"].to_numpy(np.float64) - ticks * trades["slip"].to_numpy(np.float64)
            - scale * trades[fee_col].to_numpy(np.float64))


# --------------------------------------------------------------------------
# 盘前 / 开盘区间上下文
# --------------------------------------------------------------------------
ATR_WINDOW = 20
ATR_REL_WINDOW = 250
VOL_WINDOW = 20
ORB_MINUTES = (30,)


def day_context(minute: pd.DataFrame) -> pd.DataFrame:
    """逐交易日：盘前量（ATR、相对 ATR、昨日区间、NR4、跳空、是否夜盘开盘）与开盘区间量。

    ``or{m}_atr``：前 m 根 bar 的区间宽度 / ATR（区间越窄，突破前的压缩越强）；
    ``vol{m}_ratio``：前 m 根 bar 成交量 / 此前 20 日同一时段成交量均值（分母不含当日）。
    二者到第 m 根 bar 收盘才已知，ORB(m) 最早在第 m+1 根 bar 入场，所以入场时都已知。
    """
    df = minute.sort_index(kind="mergesort")
    td = pd.to_datetime(df["trading_date"]).dt.normalize().to_numpy()
    g = df.groupby(td)
    out = pd.DataFrame({"open": g["openw"].first(), "high": g["highw"].max(),
                        "low": g["loww"].min(), "close": g["closew"].last(),
                        "raw_open": g["open"].first(), "n_bars": g.size()})
    out.index = pd.DatetimeIndex(out.index, name="trading_date")
    prev_close = out["close"].shift(1)
    tr = pd.concat([out["high"] - out["low"], (out["high"] - prev_close).abs(),
                    (out["low"] - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.shift(1).rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean()
    out["atr_pct"] = out["atr"] / out["raw_open"].where(out["raw_open"] > 0)
    # 相对 ATR：当前 ATR% 与过去一年 ATR% 中位数之比（两者都只用到昨天）
    out["atr_rel"] = out["atr_pct"] / out["atr_pct"].rolling(
        ATR_REL_WINDOW, min_periods=ATR_REL_WINDOW // 2).median()
    prev_range = (out["high"] - out["low"]).shift(1)
    out["prev_range_atr"] = prev_range / out["atr"]
    out["nr4"] = (prev_range <= prev_range.rolling(4, min_periods=4).min()).astype(float)
    out["gap_atr"] = (out["open"] - prev_close) / out["atr"]
    first_hour = pd.DatetimeIndex(df.index).hour.to_numpy()
    out["night"] = (pd.Series(first_hour, index=td).groupby(level=0).first()
                    .reindex(out.index).to_numpy() >= 18).astype(float)

    from .engine import day_layout
    codes, starts, ends = day_layout(td)
    hi, lo = df["highw"].to_numpy("float64"), df["loww"].to_numpy("float64")
    vol = pd.to_numeric(df["volume"], errors="coerce").to_numpy("float64")
    pos = np.arange(len(df)) - starts[codes]
    origin_day = day_open_offset(df.index, codes, starts)
    atr = out["atr"].to_numpy()
    for m in ORB_MINUTES:
        for tag, origin in (("", np.zeros(len(starts), np.int64)), ("day", origin_day)):
            h, l = opening_range(hi, lo, codes, starts, ends, m, origin)
            with np.errstate(invalid="ignore", divide="ignore"):
                width = np.where(atr > 0, (h - l) / atr, np.nan)
            out[f"or{m}{tag}_atr"] = width
            rel = pos - origin[codes]
            take = (origin[codes] >= 0) & (rel >= 0) & (rel < m)
            v = pd.Series(np.add.reduceat(np.where(take, np.nan_to_num(vol), 0.0), starts),
                          index=out.index).where(np.isfinite(h))
            out[f"vol{m}{tag}_ratio"] = v / v.shift(1).rolling(
                VOL_WINDOW, min_periods=VOL_WINDOW // 2).mean()
    return out


def add_cost_context(ctx: pd.DataFrame, symbol: str, tick) -> pd.DataFrame:
    """开盘时已知的成本量：主情景往返成本占名义的比例、占 ATR 的比例。"""
    out = ctx.copy()
    raw = out["raw_open"].where(out["raw_open"] > 0).to_numpy()
    t = tick_per_day(tick, out.index)
    fee = costs.round_trip(symbol, raw, HIST_FEES, close_today=True, dates=out.index)
    out["tick"] = t
    out["cost_rt"] = fee + 2.0 * t / raw
    out["cost_atr"] = out["cost_rt"] / out["atr_pct"]
    return out


# --------------------------------------------------------------------------
# 候选交易
# --------------------------------------------------------------------------
EXITS = {"eod": {}}
BASES = {f"orb{m}|{e}": replace(StrategyConfig("opening_range_assumption",
                                               opening_range_minutes=m, atr_window=ATR_WINDOW),
                                **kw)
         for m in ORB_MINUTES for e, kw in EXITS.items()}
# 日盘开盘后的前 30 根，尾盘平。和 orb30|eod（从交易日第一根、含夜盘算起）是同一条策略的两种开盘定义。
BASES["orb30day|eod"] = StrategyConfig(
    "opening_range_assumption", opening_range_minutes=30, atr_window=ATR_WINDOW,
    open_anchor="day")
SIDES = ((1, True, False), (-1, False, True))
# 撮合只算不含滑点的收益和历史手续费；滑点、费率倍数与其他费率口径在情景层重算
CANDIDATE_BT = BacktestConfig(margin_rate=1.0, fee_schedule="hist")


def range_cols(base: str) -> tuple[str, str]:
    """这笔交易实际用的区间宽度、区间成交量列名。``orb30day|eod`` 对应日盘那 30 根。"""
    body = str(base).split("|")[0][3:]
    return f"or{body}_atr", f"vol{body}_ratio"


def _alt_fees(symbol, ref, dates):
    """平昨口径的历史费率与 2026 表费率（往返），供情景层替换。"""
    return (costs.round_trip(symbol, ref, HIST_FEES, close_today=False, dates=dates),
            costs.round_trip(symbol, ref, FEES_2026, close_today=True))


def orb_candidates(symbol: str, minute: pd.DataFrame, tick, start="2016-01-01",
                   bases=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """单品种的单边 ORB 候选交易与逐日上下文。``minute`` 需含 ATR 预热期（约一年）。

    ``tick`` 必须给（{年份: tick}），缺了滑点会静默为 0。
    """
    if tick is None:
        raise ValueError(f"{symbol} 没有 tick，滑点无法定价")
    minute = minute.sort_index(kind="mergesort")
    ctx = add_cost_context(day_context(minute), symbol, tick)
    p = prepare(minute)
    n_days = len(p.starts)
    tick_day = tick_per_day(tick, p.dates)
    atr_day = ctx["atr"].reindex(p.dates).to_numpy("float64")
    first_open = p.o[p.starts]
    origin_day = day_open_offset(p.df.index, p.codes, p.starts)
    day_session_open = np.full(n_days, np.nan)
    ok_origin = origin_day >= 0
    day_session_open[ok_origin] = p.o[p.starts[ok_origin] + origin_day[ok_origin]]
    frames = []
    for name in (bases or BASES):
        anchor_open = day_session_open if BASES[name].open_anchor == "day" else first_open
        for side, al, ash in SIDES:
            tr = run_intraday(p, BASES[name], CANDIDATE_BT, symbol, None,
                              allow_long=np.full(n_days, al),
                              allow_short=np.full(n_days, ash)).trades
            if tr.empty:
                continue
            bar = p.df.index.get_indexer(pd.DatetimeIndex(tr["entry_time"]))
            day = p.codes[bar]
            ref = p.raw_open[bar]
            ref = np.where(np.isfinite(ref) & (ref > 0), ref, tr["entry_price"].to_numpy())
            entry = tr["entry_price"].to_numpy("float64")
            fee_oo, fee26 = _alt_fees(symbol, ref, p.dates[day])
            f32 = lambda v: np.asarray(v, np.float32)
            frames.append(pd.DataFrame({
                "base": name, "side": np.int8(side), "date": p.dates[day],
                "entry_pos": np.asarray(bar - p.starts[day], np.int16),
                # 入场价相对当日开盘已走了多少 ATR（按交易方向）：追高 / 追低的程度
                "open_move_atr": f32(np.divide(
                    side * (entry - anchor_open[day]), atr_day[day],
                    out=np.full(len(day), np.nan), where=atr_day[day] > 0)),
                "gross": f32(tr["gross_return"]), "slip": f32(2.0 * tick_day[day] / ref),
                "fee": f32(tr["cost"]), "fee_oo": f32(fee_oo), "fee26": f32(fee26)}))
    start = pd.Timestamp(start)
    cols = ["base", "side", "date", "entry_pos", "open_move_atr", "gross", "slip", "fee",
            "fee_oo", "fee26"]
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    ok = (trades["date"] >= start) & np.isfinite(trades["gross"].astype(float)) \
        & np.isfinite(trades["fee"].astype(float)) & np.isfinite(trades["slip"].astype(float))
    trades = trades[ok].reset_index(drop=True)
    trades.insert(0, "symbol", symbol)
    return trades, ctx[ctx.index >= start]


# --------------------------------------------------------------------------
# 池级面板
# --------------------------------------------------------------------------
@dataclass
class Panel:
    dates: pd.DatetimeIndex
    symbols: list[str]
    avail: np.ndarray        # 当日有行情且 ATR 已知
    weight: np.ndarray       # 风险缩放权重
    di: np.ndarray           # 每笔候选交易的日期/品种下标
    si: np.ndarray

    @classmethod
    def build(cls, trades: pd.DataFrame, ctxs: dict[str, pd.DataFrame]) -> "Panel":
        atr = pd.concat({s: c["atr_pct"] for s, c in ctxs.items()}, axis=1).sort_index()
        dates, symbols = atr.index, list(atr.columns)
        di = dates.get_indexer(pd.DatetimeIndex(trades["date"]))
        si = pd.Index(symbols).get_indexer(trades["symbol"].astype(str))
        if (di < 0).any() or (si < 0).any():
            raise ValueError("候选交易的日期或品种不在上下文面板内")
        w = risk_weights(atr).to_numpy()
        return cls(dates, symbols, np.isfinite(w), np.nan_to_num(w), di.astype(np.int32),
                   si.astype(np.int32))

    def members(self, members: dict[int, list[str]]) -> np.ndarray:
        return membership(members, self.dates, self.symbols).to_numpy() & self.avail

    def book(self, value: np.ndarray, keep: np.ndarray, di=None, si=None) -> np.ndarray:
        """选中交易按 (日, 品种) 求和 × 风险权重；可交易但没交易记 0，不可交易为 NaN。

        ``di``/``si`` 缺省为全部候选交易的下标；传入时 ``value``/``keep`` 与之同长。
        """
        di = self.di if di is None else di
        si = self.si if si is None else si
        n_sym = len(self.symbols)
        flat = di[keep].astype(np.int64) * n_sym + si[keep]
        out = np.bincount(flat, weights=value[keep], minlength=len(self.dates) * n_sym)
        out = out.reshape(len(self.dates), n_sym)
        return np.where(self.avail, out * self.weight, np.nan)

    def pool(self, book: np.ndarray, member: np.ndarray) -> pd.Series:
        live = member & np.isfinite(book)
        n = live.sum(axis=1)
        total = np.where(live, book, 0.0).sum(axis=1)
        ret = pd.Series(np.where(n > 0, total / np.maximum(n, 1), np.nan), index=self.dates)
        return ret[n > 0]


# --------------------------------------------------------------------------
# 元标签特征
# --------------------------------------------------------------------------
# 日频因子（t 日收盘值）整体滞后一天，决定 t+1 交易日（从夜盘开盘起）的特征
FACTORS = ("tsmom", "tsmom_20", "cs_mom_ra_250", "er", "vol_ratio",
           "carry_ms", "cs_carry_ms", "time_combo", "neg_clv", "neg_ret_day")
CTX_FEATURES = ("atr_pct", "atr_rel", "prev_range_atr", "nr4", "gap_atr", "night", "cost_atr")
# 每组：(带方向、要乘交易方向的列, 不带方向的列)。
# core 是开盘区间本身；trend / carry / time 是开盘前已经知道的日频状态，只决定做不做这笔日内交易。
FEATURE_GROUPS = {
    "core": (("gap_atr",),
             ("side", "or_atr", "or_vol_ratio", "entry_pos", "open_move_atr", "night",
              "cost_atr", "atr_pct", "atr_rel", "prev_range_atr", "nr4")),
    "trend": (("f_tsmom", "f_tsmom_20", "f_cs_mom_ra_250"), ("f_er", "f_vol_ratio")),
    "carry": (("f_carry_ms", "f_cs_carry_ms"), ()),
    "time": (("f_time_combo", "f_neg_clv", "f_neg_ret_day"), ()),
}
GROUP_SETS = {
    "core": ("core",),
    "core+trend": ("core", "trend"),
    "core+carry": ("core", "carry"),
    "core+time": ("core", "time"),
    "core+trend+carry": ("core", "trend", "carry"),
    "all": ("core", "trend", "carry", "time"),
}


def attach_features(trades: pd.DataFrame, ctxs: dict[str, pd.DataFrame],
                    factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """给每笔候选交易拼上盘前 / 开盘区间上下文与滞后一日的日频因子（原始值，方向另乘）。"""
    ctx = pd.concat({s: c for s, c in ctxs.items()}, names=["symbol", "date"])
    key = pd.MultiIndex.from_arrays([trades["symbol"].astype(str), pd.DatetimeIndex(trades["date"])])
    feat = ctx[list(CTX_FEATURES)].reindex(key)
    feat.index = trades.index
    out = pd.concat([trades, feat], axis=1)
    out["or_atr"] = np.nan
    out["or_vol_ratio"] = np.nan
    bases = trades["base"].astype(str).to_numpy()
    for base in np.unique(bases):
        sel = bases == base
        atr_col, vol_col = range_cols(base)
        vals = ctx[[atr_col, vol_col]].reindex(key[sel]).to_numpy("float64")
        out.loc[sel, "or_atr"] = vals[:, 0]
        out.loc[sel, "or_vol_ratio"] = vals[:, 1]
    for name, frame in factors.items():
        lagged = frame.sort_index().shift(1).stack()
        lagged.index.names = ["date", "symbol"]
        out[f"f_{name}"] = lagged.swaplevel().reindex(key).to_numpy()
    return out


def ml_matrix(trades: pd.DataFrame, groups: str = "all") -> pd.DataFrame:
    """方向调整后的特征矩阵：带方向的量乘以交易方向；inf 当作缺失，不填 0。"""
    side = trades["side"].to_numpy(float)
    X = pd.DataFrame(index=trades.index)
    for g in GROUP_SETS[groups]:
        signed, plain = FEATURE_GROUPS[g]
        for col in signed:
            X[col] = trades[col].to_numpy(float) * side
        for col in plain:
            X[col] = side if col == "side" else trades[col].to_numpy(float)
    return X.replace([np.inf, -np.inf], np.nan)


def target(net: np.ndarray, atr_pct: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """风险单位净收益：主情景净收益 / ATR%，截在 ±5，与池级风险缩放同口径。"""
    atr = np.asarray(atr_pct, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(np.asarray(net, np.float64) / np.where(atr > 0, atr, np.nan), -clip, clip)


# --------------------------------------------------------------------------
# 模型与嵌套走步
# --------------------------------------------------------------------------
MODELS = ("ridge1", "ridge10", "ridge100", "ridge1000", "hgb")
TRAIN_WINDOW = 3
FIRST_TRAIN_YEAR = 2016


def make_model(name: str):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_depth=3, min_samples_leaf=300,
            l2_regularization=1.0, random_state=0)
    if name.startswith("ridge"):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             Ridge(alpha=float(name[5:])))
    raise ValueError(name)


def train_years(year: int, window: int = TRAIN_WINDOW, first: int = FIRST_TRAIN_YEAR) -> list[int]:
    """预测第 ``year`` 年用的训练年份：此前 ``window`` 年，不早于 ``first``，严格早于 ``year``。"""
    return list(range(max(first, year - window), year))


def fit_predict(model: str, X: pd.DataFrame, y: np.ndarray, train: np.ndarray,
                test: np.ndarray) -> tuple[np.ndarray, object, list[str]]:
    """训练集上拟合、测试集上预测。训练集里全缺或常数的列不用（否则插补器会丢列错位）。"""
    Xtr = X.loc[train]
    use = [c for c in X.columns if Xtr[c].notna().any() and Xtr[c].nunique(dropna=True) > 1]
    fitted = make_model(model).fit(Xtr[use], y[train])
    return fitted.predict(X.loc[test, use]), fitted, use
