"""日内开盘区间突破（ORB）元标签策略与多策略清单：数据、候选交易、特征、池级组合与走步件。

口径：
- 池子：``U.load_universe()`` 逐年条件池（第 y 年由 y-1 年统计选出），全部板块，不逐品种调参。
- 候选交易：30 根 bar 的开盘区间突破，尾盘平。开盘可以是交易日第一根（含夜盘），
  或日盘第一根。只做多、只做空两条单边路径分别撮合。
- 成本逐笔拆成 不含滑点收益 / 每 tick 往返滑点 / 手续费（历史平今、历史平昨、2026 表）三块，
  情景层（``SCENARIOS``）重组，不重跑撮合。主口径 = 历史手续费 + 每边 1 tick。
- 特征只用入场时已知的信息：开盘区间本身（前 m 根 bar）、盘前 ATR 与昨日区间、入场价、
  滞后一日的日频因子。带方向的特征乘以交易方向。
- 池级组合：当年池内品种按 ``0.01 / ATR%`` 风险缩放后等权，没交易的可交易日记 0。
- 研究只读研究期分片（≤2021）；验证期 / 样本外的分钟数据由调用方传进 ``orb_candidates``
  或 ``slate_input``。
- 多策略清单（``SLATE``）：日内、隔夜、多日持仓的规则书，逐 (品种, 日) 记账后共用同一池级组合。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ... import config as C
from ...data import shard_io
from ..backtest import costs
from .engine import (OrbConfig, day_layout, day_open_offset, next_open_trades, opening_range,
                     orb_trades, performance, position_returns, prepare, tick_per_day,
                     touch_trades)

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
# 交易日第一根起（含夜盘）/ 日盘第一根起的前 m 根，尾盘平。是同一条策略的两种开盘定义。
BASES = {f"orb{m}|eod": OrbConfig(m) for m in ORB_MINUTES}
BASES["orb30day|eod"] = OrbConfig(30, open_anchor="day")
SIDES = (1, -1)


def range_cols(base: str) -> tuple[str, str]:
    """这笔交易实际用的区间宽度、区间成交量列名。``orb30day|eod`` 对应日盘那 30 根。"""
    body = str(base).split("|")[0][3:]
    return f"or{body}_atr", f"vol{body}_ratio"


def trade_fees(symbol, ref, dates):
    """往返手续费占名义的比例：主口径（历史、平今）、平昨口径的历史费率、2026 表。"""
    return (costs.round_trip(symbol, ref, HIST_FEES, close_today=True, dates=dates),
            costs.round_trip(symbol, ref, HIST_FEES, close_today=False, dates=dates),
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
        for side in SIDES:
            tr = orb_trades(p, BASES[name], side)
            if not len(tr["day"]):
                continue
            day, bar, ref, entry = tr["day"], tr["entry_bar"], tr["ref"], tr["entry"]
            fee, fee_oo, fee26 = trade_fees(symbol, ref, p.dates[day])
            f32 = lambda v: np.asarray(v, np.float32)
            frames.append(pd.DataFrame({
                "base": name, "side": np.int8(side), "date": p.dates[day],
                "entry_pos": np.asarray(bar - p.starts[day], np.int16),
                # 入场价相对当日开盘已走了多少 ATR（按交易方向）：追高 / 追低的程度
                "open_move_atr": f32(np.divide(
                    side * (entry - anchor_open[day]), atr_day[day],
                    out=np.full(len(day), np.nan), where=atr_day[day] > 0)),
                "gross": f32(tr["gross"]), "slip": f32(2.0 * tick_day[day] / ref),
                "fee": f32(fee), "fee_oo": f32(fee_oo), "fee26": f32(fee26)}))
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

    def book(self, value: np.ndarray, keep: np.ndarray, di=None, si=None,
             scaled: bool = True) -> np.ndarray:
        """选中交易按 (日, 品种) 求和 × 风险权重；可交易但没交易记 0，不可交易为 NaN。

        ``di``/``si`` 缺省为全部候选交易的下标；传入时 ``value``/``keep`` 与之同长。
        ``scaled=False``：``value`` 已经乘过权重（多策略清单的逐日记账），不再乘。
        """
        di = self.di if di is None else di
        si = self.si if si is None else si
        n_sym = len(self.symbols)
        flat = di[keep].astype(np.int64) * n_sym + si[keep]
        out = np.bincount(flat, weights=value[keep], minlength=len(self.dates) * n_sym)
        out = out.reshape(len(self.dates), n_sym)
        return np.where(self.avail, out * self.weight if scaled else out, np.nan)

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
GROUP_SETS = {  # 不含 core 的组只用盘前日频状态决定做不做，开盘区间本身的形态不进模型
    "carry": ("carry",),
    "time": ("time",),
    "carry+time": ("carry", "time"),
    "core": ("core",),
    "core+trend": ("core", "trend"),
    "core+carry": ("core", "carry"),
    "core+time": ("core", "time"),
    "core+trend+carry": ("core", "trend", "carry"),
    "core+carry+time": ("core", "carry", "time"),
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


def economic_gate_mask(trades: pd.DataFrame, rule: str,
                       compression_cutoff: float | None = None) -> np.ndarray:
    """研究用的可解释 ORB 门控；区间压缩阈值必须由调用方用历史训练样本估计。"""
    if rule == "trend_align":
        return (trades["side"].to_numpy(float) * trades["f_tsmom"].to_numpy(float)) > 0
    if rule == "range_compress":
        if compression_cutoff is None or not np.isfinite(compression_cutoff):
            raise ValueError("range_compress 需要有限的历史 compression_cutoff")
        width = trades["or_atr"].to_numpy(float)
        return np.isfinite(width) & (width <= compression_cutoff)
    if rule == "avoid_chase":
        extension = trades["open_move_atr"].to_numpy(float)
        return np.isfinite(extension) & (extension <= 0.5)
    raise ValueError(f"未知经济门控规则: {rule}")


def scope_predictions(predictions: np.ndarray, trades: pd.DataFrame,
                     scope: str = "both_sides") -> np.ndarray:
    """Restrict scored trades without changing the pooled model's training sample."""
    scoped = np.asarray(predictions, dtype="float64").copy()
    if len(scoped) != len(trades):
        raise ValueError("预测数量必须与候选交易数一致")
    if scope == "both_sides":
        return scoped
    if scope == "long_only":
        scoped[trades["side"].to_numpy() < 0] = np.nan
        return scoped
    raise ValueError(f"未知交易方向范围: {scope}")


def target(net: np.ndarray, atr_pct: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """风险单位净收益：主情景净收益 / ATR%，截在 ±5，与池级风险缩放同口径。"""
    atr = np.asarray(atr_pct, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(np.asarray(net, np.float64) / np.where(atr > 0, atr, np.nan), -clip, clip)


# --------------------------------------------------------------------------
# 模型与嵌套走步
# --------------------------------------------------------------------------
MODELS = ("ridge1", "ridge10", "ridge100", "ridge1000", "huber", "logit",
          "hgb", "hgb_d2", "rf", "et")
TRAIN_WINDOW = 3
FIRST_TRAIN_YEAR = 2016


class SignClassifier:
    """把目标的正负当标签做逻辑回归，``predict`` 返回对数几率：> 0 即"赚钱概率 > 50%"。

    和回归一样按 ``pred > 0`` 取交易；它只看方向不看幅度，对肥尾标签不敏感。
    """

    def __init__(self, C: float = 0.1):
        self.C = C

    def fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.model_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    LogisticRegression(C=self.C, max_iter=1000))
        self.model_.fit(X, np.asarray(y) > 0)
        return self

    def predict(self, X):
        return self.model_.decision_function(X)


def make_model(name: str):
    """模型都预测风险单位净收益（``logit`` 预测它为正的对数几率），``pred > 0`` 才做。

    树模型的叶子至少几百笔：单笔标签噪声很大，叶子太小就是在记样本。
    """
    from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_depth=3, min_samples_leaf=300,
            l2_regularization=1.0, random_state=0)
    if name == "hgb_d2":
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_depth=2, min_samples_leaf=500,
            l2_regularization=1.0, random_state=0)
    if name in ("rf", "et"):
        cls = RandomForestRegressor if name == "rf" else ExtraTreesRegressor
        return make_pipeline(SimpleImputer(strategy="median"), cls(
            n_estimators=300, max_depth=8, min_samples_leaf=200, max_features=0.33,
            n_jobs=-1, random_state=0))
    if name == "huber":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500))
    if name == "logit":
        return SignClassifier()
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


# --------------------------------------------------------------------------
# 多策略清单：日内与多日持仓的规则书
# --------------------------------------------------------------------------
# 每条策略的参数只有一小格候选（``SLATE`` 里的格），走步时第 y 年用 train_years(y)
# 的主口径池级 Sharpe 选一格；2022 与样本外沿用 2019–2021 选出、冻结在配置里的那格。
# 全部是池级、多空对称的规则，不逐品种调参，方向先验事先写定。
#
# 记账统一成"逐 (品种, 日) 的腿"：gross / slip / fee / fee_oo / fee26 都已乘风险权重，
# 由 ``Panel.book(..., scaled=False)`` 加总、``Panel.pool`` 等权。情景层与 ORB 相同。
# - 日内书：每笔一行，权重取入场日；slip = 2 × tick / 入场价（每边 1 tick），fee 为开 + 平今；
# - 隔夜书：同上，但 fee 与 fee_oo 都按平昨，fee26 用 2026 表开 + 平昨，记在平仓日；
# - 多日书：逐日记账，名义仓位 = 方向 × 当日风险权重，换手（含权重变化）按单边计
#   1 tick（slip 列）与单边平昨手续费。
LEG_COLUMNS = ["symbol", "strategy", "variant", "date", "gross", "slip", "fee", "fee_oo",
               "fee26", "n"]
SLATE_START = "2016-01-01"
DAY_BARS = 30            # "开盘 30 分钟"、"最后 30 分钟"都按 30 根 1 分钟 bar


@dataclass
class SlateInput:
    """一个品种算清单所需的全部量，逐交易日数组都和 ``p.dates`` 对齐。"""
    symbol: str
    p: object
    ctx: pd.DataFrame
    tick: np.ndarray
    weight: np.ndarray        # 风险权重，ATR 未知为 NaN
    day: dict                 # 日线 o/h/l/c（复权价）与 raw（原始开盘价）
    carry: np.ndarray | None  # 主力-次主力年化展期收益（t 日收盘已知），没有为 None


def slate_input(symbol: str, minute: pd.DataFrame, tick,
                carry: pd.Series | None = None) -> SlateInput:
    if tick is None:
        raise ValueError(f"{symbol} 没有 tick，滑点无法定价")
    minute = minute.sort_index(kind="mergesort")
    p = prepare(minute)
    ctx = add_cost_context(day_context(minute), symbol, tick).reindex(p.dates)
    s = p.starts
    day = {"o": p.o[s], "h": np.maximum.reduceat(p.h, s), "l": np.minimum.reduceat(p.l, s),
           "c": p.c[p.ends], "raw": p.raw_open[s]}
    w = risk_weights(ctx[["atr_pct"]]).iloc[:, 0].to_numpy("float64")
    cv = None if carry is None else carry.reindex(p.dates).to_numpy("float64")
    return SlateInput(symbol, p, ctx, tick_per_day(tick, p.dates), w, day, cv)


def hourly_bars(p) -> dict[str, np.ndarray]:
    """1 分钟 bar 按钟点并成小时 bar（时间戳是 bar 结束时刻，先减 1 分钟再取整点），不跨交易日。"""
    stamp = (pd.DatetimeIndex(p.df.index) - pd.Timedelta(1, unit="min")).floor("h").to_numpy()
    change = np.r_[True, (stamp[1:] != stamp[:-1]) | (p.codes[1:] != p.codes[:-1])]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(stamp)] - 1
    return {"o": p.o[starts], "h": np.maximum.reduceat(p.h, starts),
            "l": np.minimum.reduceat(p.l, starts), "c": p.c[ends], "raw": p.raw_open[starts],
            "codes": p.codes[starts]}


def _sign(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return np.where(np.isfinite(x), np.sign(x), np.nan)


def _lag_trend(c: np.ndarray, lookback: int) -> np.ndarray:
    """t 日开盘可用的趋势方向：sign(c[t-1] − c[t-1-L])（复权价差，加法复权下方向正确）。"""
    s = pd.Series(c)
    return _sign((s.shift(1) - s.shift(1 + int(lookback))).to_numpy())


def _orb_book(si: SlateInput, allow_long=None, allow_short=None):
    """交易日第一根起 30 根的开盘区间突破，``allow_*`` 为逐日是否允许该方向。"""
    p = si.p
    upper, lower = opening_range(p.h, p.l, p.codes, p.starts, p.ends, DAY_BARS)
    for side, level, allow in ((1, upper, allow_long), (-1, lower, allow_short)):
        start = np.full(len(p.starts), DAY_BARS)
        if allow is not None:
            start = np.where(allow, start, -1)
        yield side, touch_trades(p, level, start, side)


def s_orb30(si):
    yield from _orb_book(si)


def s_orb30_trend(si, lookback):
    """只做与日线趋势同向的突破。"""
    trend = _lag_trend(si.day["c"], lookback)
    yield from _orb_book(si, trend > 0, trend < 0)


def s_orb30_compress(si, rule):
    """只在前一日区间收窄时做突破：``nr4`` 或 昨日区间 ≤ rule × ATR。"""
    if rule == "nr4":
        ok = si.ctx["nr4"].to_numpy() == 1
    else:
        with np.errstate(invalid="ignore"):
            ok = si.ctx["prev_range_atr"].to_numpy("float64") <= float(rule)
    yield from _orb_book(si, ok, ok)


def s_dual_thrust(si, k, n=4):
    """Dual Thrust：区间 = max(近 n 日最高 − 最低收盘, 最高收盘 − 最低)，上下轨 = 当日开盘 ± k × 区间。"""
    d = {key: pd.Series(v) for key, v in si.day.items()}
    hh = d["h"].rolling(n, min_periods=n).max().shift(1)
    ll = d["l"].rolling(n, min_periods=n).min().shift(1)
    hc = d["c"].rolling(n, min_periods=n).max().shift(1)
    lc = d["c"].rolling(n, min_periods=n).min().shift(1)
    rng = np.maximum(hh - lc, hc - ll).to_numpy()
    start = np.ones(len(rng), np.int64)          # 第一根开盘价不追，从第二根起才触价
    for side in (1, -1):
        yield side, touch_trades(si.p, si.day["o"] + side * k * rng, start, side)


def _at_bar(p, day_mask, bar_pos) -> np.ndarray:
    """逐日 → 逐 bar：当日 ``day_mask`` 为真时，只在全局位置 ``bar_pos`` 那根上为真。"""
    out = np.zeros(len(p.o), bool)
    ok = np.asarray(day_mask, bool) & (bar_pos >= p.starts) & (bar_pos <= p.ends)
    out[bar_pos[ok]] = True
    return out


def s_intraday_momentum(si):
    """昨收到日盘第 30 根收盘的收益方向，做当日最后 30 根（日盘不足 60 根不做）。"""
    p = si.p
    origin = day_open_offset(p.df.index, p.codes, p.starts)
    probe = p.starts + np.maximum(origin, 0) + DAY_BARS - 1
    signal_bar = p.ends - DAY_BARS                 # 这根收盘下单，下一根开盘入场
    enough = (origin >= 0) & (signal_bar > probe)
    prev_close = np.r_[np.nan, si.day["c"][:-1]]
    ret = np.where(enough, p.c[np.minimum(probe, len(p.c) - 1)] - prev_close, np.nan)
    sig = _sign(ret)
    for side in (1, -1):
        yield side, next_open_trades(p, _at_bar(p, sig == side, signal_bar), side)


def s_vwap_fade(si, k):
    """偏离当日 VWAP 超过 k × ATR 时反向，下一根开盘入场、尾盘平。开盘 30 根内和最后 30 根不开仓。"""
    p = si.p
    vol = np.nan_to_num(pd.to_numeric(p.df["volume"], errors="coerce").to_numpy("float64"))
    pv = pd.Series(p.c * vol).groupby(p.codes).cumsum().to_numpy()
    cv = pd.Series(vol).groupby(p.codes).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cv > 0, pv / cv, np.nan)
        dev = (p.c - vwap) / si.ctx["atr"].to_numpy("float64")[p.codes]
        pos = np.arange(len(p.o)) - p.starts[p.codes]
        left = p.ends[p.codes] - np.arange(len(p.o))
        window = (pos >= DAY_BARS - 1) & (left > DAY_BARS)
        long_sig, short_sig = window & (dev < -k), window & (dev > k)
    yield 1, next_open_trades(p, long_sig, 1)
    yield -1, next_open_trades(p, short_sig, -1)


def s_gap_fade(si, g):
    """跳空（当日第一根开盘相对昨收）超过 g × ATR 时反向，第二根开盘入场、尾盘平。"""
    gap = si.ctx["gap_atr"].to_numpy("float64")
    with np.errstate(invalid="ignore"):
        masks = {1: gap < -g, -1: gap > g}
    for side in (1, -1):
        yield side, next_open_trades(si.p, _at_bar(si.p, masks[side], si.p.starts), side)


def s_overnight_momentum(si, window):
    """尾盘顺势隔夜：倒数第二根收盘时看当日收益方向，最后一根开盘入场，次一交易日第一根收盘平。

    ``window="day"`` 用日盘开盘起的收益，``"last60"`` 用最后 60 根的收益。记在平仓日。
    """
    p = si.p
    probe = p.ends - 1
    if window == "day":
        origin = day_open_offset(p.df.index, p.codes, p.starts)
        ok = origin >= 0
        base = np.where(ok, p.o[p.starts + np.maximum(origin, 0)], np.nan)
    else:
        back = p.ends - 61
        ok = back >= p.starts
        base = np.where(ok, p.c[np.maximum(back, 0)], np.nan)
    sig = np.where(ok & (probe > p.starts), _sign(p.c[probe] - base), np.nan)
    ei, xi = p.ends[:-1], p.starts[1:]
    has = np.isfinite(sig[:-1]) & p.tradable[ei]
    for side in (1, -1):
        days = np.flatnonzero(has & (sig[:-1] == side))
        e, x = ei[days], xi[days]
        ref = p.raw_open[e]
        ref = np.where(np.isfinite(ref) & (ref > 0), ref, p.o[e])
        yield side, {"day": days, "book_day": days + 1, "ref": ref,
                     "gross": side * (p.c[x] - p.o[e]) / ref}


def p_donchian_hourly(si, n):
    """小时线 n 根通道突破，反向突破才翻仓（一旦入场始终持仓），下一根小时 bar 开盘成交。"""
    h = hourly_bars(si.p)
    c = pd.Series(h["c"])
    up = c > pd.Series(h["h"]).rolling(n, min_periods=n).max().shift(1)
    dn = c < pd.Series(h["l"]).rolling(n, min_periods=n).min().shift(1)
    return h, np.where(up, 1.0, np.where(dn, -1.0, np.nan))


def p_ema_hourly(si, fast, slow):
    """小时线快慢 EMA 方向。"""
    h = hourly_bars(si.p)
    c = pd.Series(h["c"])
    diff = (c.ewm(span=fast, adjust=False, min_periods=fast).mean()
            - c.ewm(span=slow, adjust=False, min_periods=slow).mean())
    return h, _sign(diff.to_numpy())


def _daily_bars(si):
    d = dict(si.day)
    d["codes"] = np.arange(len(d["c"]))
    return d


def p_tsmom_daily(si, lookback):
    """日线时间序列动量：sign(c[t] − c[t−L])，t 日收盘定方向，下一交易日第一根开盘成交。"""
    d = _daily_bars(si)
    c = pd.Series(d["c"])
    return d, _sign((c - c.shift(int(lookback))).to_numpy())


def p_bollinger_reversion(si, entry, exit=0.5, window=20):
    """日线布林反转：z > entry 做空、z < −entry 做多，|z| < exit 平仓，其间保持。"""
    d = _daily_bars(si)
    c = pd.Series(d["c"])
    z = ((c - c.rolling(window).mean()) / c.rolling(window).std()).to_numpy()
    with np.errstate(invalid="ignore"):
        sig = np.where(z > entry, -1.0, np.where(z < -entry, 1.0,
                       np.where(np.abs(z) < exit, 0.0, np.nan)))
    return d, sig


def _carry_sign(si, band):
    if si.carry is None:
        raise ValueError(f"{si.symbol} 没有展期收益数据")
    x = si.carry
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(x), np.where(np.abs(x) > band, np.sign(x), 0.0), 0.0)


def p_carry_daily(si, band):
    """展期收益方向（贴水做多、升水做空），|年化| ≤ band 或缺数据空仓。"""
    return _daily_bars(si), _carry_sign(si, band)


def p_tsmom_carry_daily(si, lookback, band=0.0):
    """日线动量方向与展期方向等权：同向满仓，相反空仓。"""
    d, trend = p_tsmom_daily(si, lookback)
    return d, (np.nan_to_num(trend) + _carry_sign(si, band)) / 2.0


# 名字 → (类型, 函数, {格名: 参数})。"intraday"/"overnight" 逐笔，"position" 逐日记账。
SLATE = {
    "orb30": ("intraday", s_orb30, {"m30": {}}),
    "orb30_trend": ("intraday", s_orb30_trend,
                    {f"L{n}": {"lookback": n} for n in (20, 60, 120)}),
    "orb30_compress": ("intraday", s_orb30_compress,
                       {"nr4": {"rule": "nr4"}, "pr0.8": {"rule": 0.8}}),
    "dual_thrust": ("intraday", s_dual_thrust, {f"k{k}": {"k": k} for k in (0.3, 0.5, 0.7)}),
    "intraday_momentum": ("intraday", s_intraday_momentum, {"last30": {}}),
    "vwap_fade": ("intraday", s_vwap_fade, {f"k{k}": {"k": k} for k in (0.5, 1.0)}),
    "gap_fade": ("intraday", s_gap_fade, {f"g{g}": {"g": g} for g in (0.3, 0.6)}),
    "overnight_momentum": ("overnight", s_overnight_momentum,
                           {w: {"window": w} for w in ("day", "last60")}),
    "donchian_hourly": ("position", p_donchian_hourly,
                        {f"n{n}": {"n": n} for n in (60, 120, 240)}),
    "ema_hourly": ("position", p_ema_hourly,
                   {f"{a}/{b}": {"fast": a, "slow": b} for a, b in ((10, 60), (20, 120), (40, 240))}),
    "tsmom_daily": ("position", p_tsmom_daily, {f"L{n}": {"lookback": n} for n in (60, 120, 250)}),
    "bollinger_reversion": ("position", p_bollinger_reversion,
                            {f"z{z}": {"entry": z} for z in (1.5, 2.0)}),
    "carry_daily": ("position", p_carry_daily, {f"band{b:g}": {"band": b} for b in (0.0, 0.05)}),
    "tsmom_carry_daily": ("position", p_tsmom_carry_daily,
                          {f"L{n}": {"lookback": n} for n in (60, 120, 250)}),
}
CARRY_STRATEGIES = frozenset({"carry_daily", "tsmom_carry_daily"})
SLATE_NUMERIC = ["gross", "slip", "fee", "fee_oo", "fee26"]


def _trade_legs(si, tr, overnight: bool) -> pd.DataFrame:
    day = tr["day"]
    book = tr.get("book_day", day)
    dates = si.p.dates
    w = si.weight[day]
    ref = tr["ref"]
    if overnight:
        fee = costs.round_trip(si.symbol, ref, HIST_FEES, close_today=False, dates=dates[day])
        fee_oo = fee
        fee26 = costs.round_trip(si.symbol, ref, FEES_2026, close_today=False)
    else:
        fee, fee_oo, fee26 = trade_fees(si.symbol, ref, dates[day])
    return pd.DataFrame({
        "date": dates[book], "gross": tr["gross"] * w, "slip": 2.0 * si.tick[day] / ref * w,
        "fee": np.asarray(fee) * w, "fee_oo": np.asarray(fee_oo) * w,
        "fee26": np.asarray(fee26) * w, "n": np.ones(len(day))})


def _position_legs(si, bars, signal) -> pd.DataFrame:
    raw = si.day["raw"]
    dates = si.p.dates
    o_leg, c_leg = costs.fee_legs(si.symbol, raw, HIST_FEES, close_today=False, dates=dates)
    o26, c26 = costs.fee_legs(si.symbol, raw, FEES_2026, close_today=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        side_fee = (np.asarray(o_leg) + np.asarray(c_leg)) / 2.0
        rates = {"slip": si.tick / raw, "fee": side_fee, "fee_oo": side_fee,
                 "fee26": (np.asarray(o26) + np.asarray(c26)) / 2.0}
    w = np.where(np.isfinite(si.weight), si.weight, 0.0)
    r = position_returns(bars["o"], bars["c"], bars["raw"], bars["codes"], len(dates),
                         signal, w, rates)
    out = pd.DataFrame({"date": dates, **{k: r[k] for k in (*SLATE_NUMERIC, "n")}})
    return out[np.isfinite(si.weight)]


def slate_legs(si: SlateInput, books: dict[str, list[str]] | None = None,
               start=SLATE_START) -> pd.DataFrame:
    """单品种全部（或指定的 {策略: [格]}）规则书的逐日腿。"""
    frames = []
    for name, variants in (books or {n: list(v[2]) for n, v in SLATE.items()}).items():
        kind, fn, grid = SLATE[name]
        if name in CARRY_STRATEGIES and si.carry is None:
            continue
        for variant in variants:
            if kind == "position":
                legs = _position_legs(si, *fn(si, **grid[variant]))
            else:
                legs = pd.concat([_trade_legs(si, tr, kind == "overnight")
                                  for _, tr in fn(si, **grid[variant])], ignore_index=True)
            legs.insert(0, "variant", variant)
            legs.insert(0, "strategy", name)
            frames.append(legs)
    if not frames:
        return pd.DataFrame(columns=LEG_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    ok = (pd.DatetimeIndex(out["date"]) >= pd.Timestamp(start)) & np.isfinite(
        out[SLATE_NUMERIC].to_numpy(np.float64)).all(axis=1)
    out = out[ok].reset_index(drop=True)
    for col in SLATE_NUMERIC:
        out[col] = out[col].astype(np.float32)
    out.insert(0, "symbol", si.symbol)
    return out[LEG_COLUMNS]


def slate_panel(ctxs: dict[str, pd.DataFrame]) -> Panel:
    empty = pd.DataFrame({"date": pd.DatetimeIndex([]), "symbol": pd.Series([], dtype=str)})
    return Panel.build(empty, ctxs)


def slate_series(legs: pd.DataFrame, panel: Panel, member: np.ndarray,
                 scenarios=("main",)) -> dict[tuple[str, str, str], pd.Series]:
    """{(策略, 格, 情景): 池级逐日净收益}。"""
    di = panel.dates.get_indexer(pd.DatetimeIndex(legs["date"]))
    si = pd.Index(panel.symbols).get_indexer(legs["symbol"].astype(str))
    if (di < 0).any() or (si < 0).any():
        raise ValueError("腿的日期或品种不在面板内")
    out = {}
    for (name, variant), idx in legs.groupby(["strategy", "variant"], sort=False).indices.items():
        part = legs.iloc[idx]
        keep = np.ones(len(idx), bool)
        for sc in scenarios:
            book = panel.book(scenario_net(part, sc), keep, di[idx], si[idx], scaled=False)
            out[(name, variant, sc)] = panel.pool(book, member)
    return out


def select_variant(series: dict[str, pd.Series], years) -> str:
    """训练年份里主口径池级 Sharpe 最高的格（并列或都无效时取先列出的）。"""
    best, best_sr = None, -np.inf
    for variant, s in series.items():
        sr = performance(s[s.index.year.isin(list(years))])["sharpe"]
        if best is None:
            best = variant
        if np.isfinite(sr) and sr > best_sr:
            best, best_sr = variant, sr
    return best
