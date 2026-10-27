"""收盘位置反转的经济机制：分段、期限、波动和成交量条件、结算价偏离、五分组和 L1 盘口。只读研究期。

    python -m research.close_location_mechanism            # 日线和分钟部分，商品 2011–2021
    python -m research.close_location_mechanism --l1       # L1 盘口部分，5 个品种 2016–2021（先跑 download.commodity.getTickL1Close）
    python -m research.close_location_mechanism --all-l1   # 所有已下载商品 L1，检验流动性枯竭/提供假设
    python -m research.close_location_mechanism --followup # 盘口变差、持仓变化、换月和交割前平仓三项后续检验

规则在 config/research/close_location_mechanism.json，先登记后运行。只诊断，不改冻结策略。
结果在 runs/research/close_location_mechanism/。不读 2022 及以后。
"""
from __future__ import annotations

import argparse
import json
import math
import warnings

import numpy as np
import pandas as pd

from backtest.engine import load_research, newey_west_t
from research.market_clock_mechanism import LAGS, day_shape, slope, trailing_median_above
from research.time_factor_screen import ols_hac
from strategies import market_clock_slow_short_20_60_120 as S
from tfcta import config as C
from tfcta.data import build_universe, daily_bars, load_shard, load_universe, wide_by_field

CONFIG_PATH = C.CONFIG_DIR / "research" / "close_location_mechanism.json"
OUT = C.RUNS_DIR / "research" / "close_location_mechanism"
L1_DIR = C.COMMODITY_DATA / "external_rqdata" / "l1_close"
L1_SYMBOLS = ("RB", "CU", "M", "TA", "I")
FIRST_YEAR = C.EARLY_START.year
YEARS = tuple(range(FIRST_YEAR, C.RESEARCH_END.year + 1))
L1_YEARS = tuple(range(2016, C.RESEARCH_END.year + 1))
VOL_WINDOW, VOLUME_WINDOW, HORIZONS = 20, 20, (1, 2, 3, 4, 5)
CONTRACT_DIR = C.COMMODITY_DATA / "external_rqdata" / "contracts"
DOMINANT_DIR = C.COMMODITY_DATA / "external_rqdata" / "dominant"
MATURITY_PATH = C.COMMODITY_DATA / "external_rqdata" / "contract_maturity.csv"
L1_LIQUIDITY_CONFIG_PATH = C.CONFIG_DIR / "research" / "commodity_l1_liquidity.json"
L1_LIQUIDITY_OUT = C.RUNS_DIR / "research" / "commodity_l1_liquidity"
TAIL_START, DAY_START, DAY_END = "14:30", "09:00", "15:00"
ROLL_LOOKBACK = 5
L1_STATE_LOOKBACK, L1_STATE_MIN_PERIODS = 250, 125
L1_MIN_PRODUCT_OBSERVATIONS, L1_MIN_SECTOR_OBSERVATIONS = 500, 500
L1_MIN_CALENDAR_YEARS, L1_MIN_SECTOR_PRODUCTS = 3, 2


def load_protocol(path=CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "declared_test_not_a_strategy":
        raise ValueError("检验登记必须声明这不是策略")
    if "不读 2022" not in protocol.get("sample", "") or "不读 2026" not in protocol.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    return protocol


def load_l1_liquidity_protocol(path=L1_LIQUIDITY_CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    parameters = protocol.get("parameters", {})
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022" not in sample
        or "不读 2026" not in sample
        or parameters.get("commodity_window") != [2011, 2021]
        or parameters.get("state_lookback") != L1_STATE_LOOKBACK
        or parameters.get("state_min_periods") != L1_STATE_MIN_PERIODS
        or parameters.get("hac_lags") != LAGS
        or parameters.get("minimum_product_observations") != L1_MIN_PRODUCT_OBSERVATIONS
        or parameters.get("minimum_product_calendar_years") != L1_MIN_CALENDAR_YEARS
        or parameters.get("minimum_sector_products_per_day") != L1_MIN_SECTOR_PRODUCTS
        or parameters.get("minimum_sector_observations") != L1_MIN_SECTOR_OBSERVATIONS
        or parameters.get("minimum_sector_calendar_years") != L1_MIN_CALENDAR_YEARS
        or parameters.get("sector_aggregation")
        != "Equal-weight product means within the active research pool on each date; require at least two products with complete inputs for the model."
    ):
        raise ValueError("商品 L1 流动性检验登记与实现或样本边界不一致")
    return protocol


# --------------------------------------------------------------------------
# 收盘 t 之后的各段收益，和收盘 t 已知的条件变量
# --------------------------------------------------------------------------
def segments(bars: dict, columns) -> dict:
    """g1 收盘 t→下一开盘，i1 下一开盘→下一收盘，g2 下一收盘→再下一开盘，fwd 下一开盘→再下一开盘（账本口径）。"""
    o, ow, c, cw = (bars[k].reindex(columns=columns) for k in ("open", "openw", "close", "closew"))
    gap = (ow.shift(-1) - cw) / c.where(c > 0)
    intraday = (cw - ow) / o.where(o > 0)
    oo = (ow.shift(-1) - ow) / o.where(o > 0)
    out = {"g1": gap, "i1": intraday.shift(-1), "g2": gap.shift(-1), "fwd": oo.shift(-1)}
    out.update({f"h{k}": oo.shift(-k) for k in HORIZONS})
    return out


def abnormal_volume(volume: pd.DataFrame, window: int = VOLUME_WINDOW) -> pd.DataFrame:
    """log(当天成交量 / 过去 window 日平均，不含当天)。"""
    base = volume.shift(1).rolling(window, min_periods=window).mean()
    return np.log(volume.where(volume > 0) / base.where(base > 0))


def minute_vwap_dev(minute: pd.DataFrame) -> pd.Series:
    """单品种分钟表 → 每个交易日 (收盘 − 成交量加权的分钟原始收盘) / 收盘；近似结算价偏离。"""
    m = minute.sort_index()
    day = pd.to_datetime(m["trading_date"]).dt.normalize()
    vol = m["volume"].where(m["volume"] > 0)
    vwap = (m["close"] * vol).groupby(day).sum() / vol.groupby(day).sum()
    last = m["close"].groupby(day).last()
    out = (last - vwap) / last.where(last > 0)
    out.index.name = "trading_date"
    return out


def quintile_table(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    frame = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    frame["q"] = pd.qcut(frame["x"], 5, labels=False) + 1
    rows = []
    for q, part in frame.groupby("q"):
        rows.append({"组": int(q), "收盘位置均值": float(part["x"].mean()), "天数": len(part),
                     "下一段bp": float(part["y"].mean() * 1e4), "NW_t": newey_west_t(part["y"], LAGS)})
    return pd.DataFrame(rows)


def yearly_signs(y: pd.Series, x: pd.Series, years) -> tuple[int, dict]:
    by_year = {yr: slope(y[y.index.year == yr], [x[x.index.year == yr]])[0] for yr in years}
    return int(sum(v < 0 for v in by_year.values() if pd.notna(v))), by_year


def prediction(tag, label, value, t, expected: str) -> dict:
    right = (value < 0) if expected == "<0" else (value > 0) if expected == ">0" else np.nan
    return {"预测": tag, "内容": label, "值": value, "NW_t": t, "预期": expected,
            "方向对": bool(right) if isinstance(right, (bool, np.bool_)) else right,
            "t≥2": bool(pd.notna(t) and abs(t) >= 2)}


# --------------------------------------------------------------------------
# 日线和分钟部分
# --------------------------------------------------------------------------
def run_daily(protocol) -> None:
    data = load_research(1.0, first_year=FIRST_YEAR)
    C.assert_no_holdout_dates(data.returns.index, "收盘位置机制收益")
    columns = sorted({s for members in data.universe.values() for s in members})
    bars = {k: v.reindex(columns=columns) for k, v in data.factors.bars.items()}
    days = bars["close"].index
    pool = S.pool_mask(data.universe, days, columns)
    keep = data.returns.index
    shape = {k: v.where(pool) for k, v in day_shape(bars, columns).items()}
    clv = shape["clv"].mean(axis=1)
    same = shape["same_day"].mean(axis=1)
    seg = {k: v.where(pool).mean(axis=1) for k, v in segments(bars, columns).items()}
    score = S.market_score(data.factors.signed, data.universe, keep).reindex(days)
    bull = score.gt(0).astype(float).where(score.notna())
    pool_ret = S.close_return(bars, columns).where(pool).mean(axis=1)
    rv = pool_ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    high_vol = trailing_median_above(rv).astype(float).where(trailing_median_above(rv).notna())
    av = abnormal_volume(bars["volume"]).where(pool).mean(axis=1)
    high_av = trailing_median_above(av).astype(float).where(trailing_median_above(av).notna())

    dev = {}
    for symbol in columns:
        minute = load_shard(symbol, columns=["close", "volume", "trading_date"])
        C.assert_no_holdout_dates(pd.to_datetime(minute["trading_date"]), f"{symbol} 分钟行情")
        dev[symbol] = minute_vwap_dev(minute)
    vwap_dev = pd.DataFrame(dev).reindex(index=days, columns=columns).where(pool).mean(axis=1)

    k = lambda s: s.reindex(keep)                     # noqa: E731
    y = k(seg["fwd"])
    rows = []
    for name in ("g1", "i1", "g2", "fwd", *(f"h{h}" for h in HORIZONS)):
        b, t = slope(k(seg[name]), [k(clv)])
        bs, ts = slope(k(seg[name]), [k(bull)])
        neg, _ = yearly_signs(k(seg[name]), k(clv), YEARS)
        rows.append({"段": name, "收盘位置斜率bp": b * 1e4, "NW_t": t, "斜率为负年数": neg, "年数": len(YEARS),
                     "分数>0择时差bp": bs * 1e4, "分数NW_t": ts})
    seg_table = pd.DataFrame(rows)

    def interaction(x, state):
        b, t = slope(y, [k(x) * k(state), k(x), k(state)])
        lo = slope(y[k(state) == 0], [k(x)[k(state) == 0]])
        hi = slope(y[k(state) == 1], [k(x)[k(state) == 1]])
        return b, t, lo, hi

    preds = []
    b, t, lo, hi = interaction(clv, high_vol)
    preds.append(prediction("P1", "clv × 高波动 交乘（bp）", b * 1e4, t, "<0"))
    cond = [{"条件": "低波动", "clv斜率bp": lo[0] * 1e4, "NW_t": lo[1]}, {"条件": "高波动", "clv斜率bp": hi[0] * 1e4, "NW_t": hi[1]}]
    b, t, lo, hi = interaction(clv, high_av)
    preds.append(prediction("P2", "clv × 高成交量 交乘（bp）", b * 1e4, t, "<0"))
    cond += [{"条件": "成交量正常", "clv斜率bp": lo[0] * 1e4, "NW_t": lo[1]}, {"条件": "成交量放大", "clv斜率bp": hi[0] * 1e4, "NW_t": hi[1]}]
    b2, t2, lo2, hi2 = interaction(same, high_av)
    preds.append(prediction("P2b", "当天收益 × 高成交量 交乘", b2, t2, "<0"))
    cond += [{"条件": "当天收益·成交量正常", "clv斜率bp": lo2[0] * 1e4, "NW_t": lo2[1]},
             {"条件": "当天收益·成交量放大", "clv斜率bp": hi2[0] * 1e4, "NW_t": hi2[1]}]
    h3 = seg_table.set_index("段").loc[["h3", "h4", "h5"]]
    preds.append(prediction("P3", "k≥3 的最大 |t|", float(h3["收盘位置斜率bp"].iloc[h3["NW_t"].abs().argmax()]),
                            float(h3["NW_t"].abs().max()), "|t|<2"))
    bd, td = slope(y, [k(vwap_dev)])
    bc, tc = slope(y, [k(clv), k(vwap_dev)])
    bv, tv = slope(y, [k(vwap_dev), k(clv)])
    preds.append(prediction("P6", "结算价偏离单独（每 1% bp）", bd * 1e2, td, "<0"))
    preds.append(prediction("P6b", "同时放入时 clv 斜率（bp）", bc * 1e4, tc, "|t|<2"))
    preds.append(prediction("P6c", "同时放入时结算价偏离（每 1% bp）", bv * 1e2, tv, "<0"))
    seg_i = seg_table.set_index("段")
    preds.append(prediction("P7", "跳空 g1 的 clv 斜率（bp）", float(seg_i.loc["g1", "收盘位置斜率bp"]), float(seg_i.loc["g1", "NW_t"]), "<0"))
    preds.append(prediction("P7b", "账本 fwd 的 clv 斜率（bp）", float(seg_i.loc["fwd", "收盘位置斜率bp"]), float(seg_i.loc["fwd", "NW_t"]), "<0"))
    quint = quintile_table(k(clv), y)
    preds.append(prediction("P10", "最低组下一段（bp）", float(quint["下一段bp"].iloc[0]), float(quint["NW_t"].iloc[0]), ">0"))
    preds.append(prediction("P10b", "最高组下一段（bp）", float(quint["下一段bp"].iloc[-1]), float(quint["NW_t"].iloc[-1]), "|t|<2"))
    for p in preds:
        if p["预期"] == "|t|<2":
            p["方向对"] = bool(pd.notna(p["NW_t"]) and abs(p["NW_t"]) < 2)
    corr = {"结算价偏离与clv": float(k(vwap_dev).corr(k(clv))), "结算价偏离与分数": float(k(vwap_dev).corr(k(score))),
            "高波动日占比": float(k(high_vol).mean()), "高成交量日占比": float(k(high_av).mean()),
            "条件变量可用天数": int(k(high_vol).notna().sum())}

    OUT.mkdir(parents=True, exist_ok=True)
    seg_table.to_csv(OUT / "segments_horizons.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cond).to_csv(OUT / "conditional_slopes.csv", index=False, encoding="utf-8-sig")
    quint.to_csv(OUT / "quintiles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(preds).to_csv(OUT / "predictions_daily.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps({
        "protocol": protocol, "window": f"{keep.min().date()}..{keep.max().date()}", "descriptive": corr,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 240)
    print(seg_table.round(3).to_string(index=False))
    print(pd.DataFrame(cond).round(3).to_string(index=False))
    print(quint.round(3).to_string(index=False))
    print(pd.DataFrame(preds).round(3).to_string(index=False))
    print({kk: round(v, 3) for kk, v in corr.items()})
    print(f"结果：{OUT}")


# --------------------------------------------------------------------------
# L1 部分
# --------------------------------------------------------------------------
def load_l1() -> dict:
    frames = []
    for symbol in L1_SYMBOLS:
        path = L1_DIR / f"{symbol}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"缺 {path}，先运行 python -m download.commodity.getTickL1Close")
        frames.append(pd.read_csv(path, parse_dates=["trading_date"]))
    table = pd.concat(frames, ignore_index=True)
    C.assert_no_holdout_dates(table["trading_date"], "L1 汇总")
    table["spread_ratio"] = table["spread_last30"] / table["spread_day"]
    table["depth_ratio"] = table["depth_last30"] / table["depth_day"]
    fields = ("imb_last30", "imb_day", "spread_ratio", "depth_ratio", "close_vs_mid", "half_spread_final", "clv_last", "clv_mid")
    return {f: table.pivot_table(index="trading_date", columns="symbol", values=f) for f in fields}


def trailing_worse_state(
    values: pd.Series,
    *,
    higher_is_worse: bool,
    window: int = L1_STATE_LOOKBACK,
    min_periods: int = L1_STATE_MIN_PERIODS,
) -> pd.Series:
    """用不含当天的历史中位数标记价差变宽或深度变薄。"""
    median = values.shift(1).rolling(window, min_periods=min_periods).median()
    state = values.gt(median) if higher_is_worse else values.lt(median)
    return state.astype("float64").where(median.notna())


def benjamini_hochberg_qvalues(t_statistics: pd.Series) -> pd.Series:
    """用正态近似双侧 p 值计算 BH q 值。"""
    q_values = pd.Series(np.nan, index=t_statistics.index, dtype="float64")
    valid = t_statistics.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return q_values
    p_values = valid.abs().map(lambda value: math.erfc(float(value) / math.sqrt(2.0)))
    order = np.argsort(p_values.to_numpy())
    sorted_p = p_values.to_numpy()[order]
    ranks = np.arange(1, len(sorted_p) + 1, dtype="float64")
    adjusted = np.minimum.accumulate((sorted_p * len(sorted_p) / ranks)[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    q_values.loc[p_values.index.to_numpy()[order]] = adjusted
    return q_values


L1_MODELS = (
    {
        "id": "reversal",
        "label": "次日盘中收益 ~ 收盘位置",
        "outcome": "next_intraday",
        "predictors": ("clv_mid",),
        "tested": 0,
        "expected": "<0",
        "return_units": True,
    },
    {
        "id": "spread_depletion",
        "label": "尾盘相对价差 ~ 收盘位置",
        "outcome": "spread_ratio",
        "predictors": ("clv_mid",),
        "tested": 0,
        "expected": "<0",
        "return_units": False,
    },
    {
        "id": "depth_depletion",
        "label": "尾盘相对深度 ~ 收盘位置",
        "outcome": "depth_ratio",
        "predictors": ("clv_mid",),
        "tested": 0,
        "expected": ">0",
        "return_units": False,
    },
    {
        "id": "spread_provision",
        "label": "收盘位置 × 尾盘价差偏宽状态",
        "outcome": "next_intraday",
        "predictors": ("clv_spread_interaction", "clv_mid", "spread_worse"),
        "tested": 0,
        "expected": "<0",
        "return_units": True,
        "interaction": True,
    },
    {
        "id": "depth_provision",
        "label": "收盘位置 × 尾盘深度偏薄状态",
        "outcome": "next_intraday",
        "predictors": ("clv_depth_interaction", "clv_mid", "depth_worse"),
        "tested": 0,
        "expected": "<0",
        "return_units": True,
        "interaction": True,
    },
    {
        "id": "flow_pressure",
        "label": "尾盘买卖不平衡 ~ 收盘位置",
        "outcome": "imb_last30",
        "predictors": ("clv_mid",),
        "tested": 0,
        "expected": ">0",
        "return_units": False,
    },
    {
        "id": "flow_prediction",
        "label": "次日盘中收益 ~ 尾盘不平衡度（控制收盘位置）",
        "outcome": "next_intraday",
        "predictors": ("imb_last30", "clv_mid"),
        "tested": 0,
        "expected": "descriptive",
        "return_units": True,
    },
    {
        "id": "reversal_controlled",
        "label": "次日盘中收益 ~ 收盘位置（控制尾盘不平衡度）",
        "outcome": "next_intraday",
        "predictors": ("clv_mid", "imb_last30"),
        "tested": 0,
        "expected": "<0",
        "return_units": True,
    },
)


def _l1_fit_row(
    group: pd.DataFrame,
    *,
    scope: str,
    entity: str,
    model: dict,
    minimum_observations: int,
    minimum_years: int,
) -> dict:
    predictors = list(model["predictors"])
    fields = [model["outcome"], *predictors]
    sample = group[fields].replace([np.inf, -np.inf], np.nan).dropna()
    years = sorted(set(sample.index.year))
    enough = len(sample) >= minimum_observations and len(years) >= minimum_years
    scale = 1e4 if model["return_units"] else 1.0
    row = {
        "范围": scope,
        "品种或板块": entity,
        "检验": model["id"],
        "内容": model["label"],
        "观测数": len(sample),
        "年份数": len(years),
        "起始日": str(sample.index.min().date()) if len(sample) else "",
        "结束日": str(sample.index.max().date()) if len(sample) else "",
        "最低样本要求满足": enough,
        "预期方向": model["expected"],
        "系数": np.nan,
        "系数单位": "bp" if model["return_units"] else "ratio / imbalance unit",
        "NW_t": np.nan,
        "p_value": np.nan,
        "BH_q": np.nan,
        "差流动性状态下CLV斜率": np.nan,
    }
    if not enough:
        return row
    y = sample[model["outcome"]].to_numpy(dtype="float64")

    def fit_term(term_index: int) -> tuple[float, float]:
        ordered = [predictors[term_index]] + [
            predictor for i, predictor in enumerate(predictors) if i != term_index
        ]
        return ols_hac(y, [sample[name].to_numpy(dtype="float64") for name in ordered], LAGS)

    coefficient, t_stat = fit_term(model["tested"])
    row.update(
        {
            "系数": coefficient * scale,
            "NW_t": t_stat,
            "p_value": math.erfc(abs(t_stat) / math.sqrt(2.0)) if np.isfinite(t_stat) else np.nan,
        }
    )
    if model.get("interaction"):
        base_clv, _ = fit_term(1)
        row["差流动性状态下CLV斜率"] = (base_clv + coefficient) * scale
    return row


def _l1_model_rows(
    panel: pd.DataFrame,
    *,
    scope: str,
    entity: str,
    minimum_observations: int,
    minimum_years: int,
    minimum_products_per_day: int = 1,
) -> list[dict]:
    rows = []
    for model in L1_MODELS:
        fields = [model["outcome"], *model["predictors"]]
        complete = panel[fields].replace([np.inf, -np.inf], np.nan).dropna()
        if minimum_products_per_day > 1:
            grouped = complete.assign(交易日=complete.index).groupby("交易日")
            complete = grouped.filter(lambda day: len(day) >= minimum_products_per_day)
            series = complete.groupby("交易日")[fields].mean()
            series.index = pd.DatetimeIndex(series.index)
        else:
            series = complete
        rows.append(
            _l1_fit_row(
                series,
                scope=scope,
                entity=entity,
                model=model,
                minimum_observations=minimum_observations,
                minimum_years=minimum_years,
            )
        )
    return rows


def run_all_l1(protocol: dict) -> None:
    """按年池检验全部可用商品 L1 对流动性枯竭/提供假设的支持程度。"""
    C.assert_research_only(C.RESEARCH_DIR)
    C.assert_no_holdout_dates(pd.DatetimeIndex([pd.Timestamp("2021-12-31")]), "商品 L1 研究截止日")
    stored_universe = load_universe()
    universe = {year: members for year, members in stored_universe.items() if 2016 <= year <= 2021}
    if FIRST_YEAR < 2016:
        stats = pd.read_csv(C.UNIVERSE_DIR / "yearly_stats.csv")
        C.assert_no_holdout_dates(
            pd.to_datetime(stats["year"].astype(str) + "-01-01"), "商品年度品种池统计"
        )
        early_universe, _ = build_universe(stats, years=list(range(FIRST_YEAR, 2016)))
        if early_universe[2015] != sorted(stored_universe[2015]):
            raise ValueError("重建的 2015 商品研究池与已保存的品种池不一致")
        universe = {
            **{year: early_universe[year] for year in range(FIRST_YEAR, 2016)},
            **universe,
        }
    pool_symbols = sorted({symbol for members in universe.values() for symbol in members})
    daily_by_symbol = {}
    for symbol in pool_symbols:
        minute = load_shard(
            symbol,
            columns=["open", "openw", "closew", "trading_date"],
        )
        daily_by_symbol[symbol] = daily_bars(minute)
    bars = wide_by_field(daily_by_symbol)
    days = bars["open"].index
    pool = S.pool_mask(universe, days, pool_symbols)
    sector_table = json.loads(
        (C.CONFIG_DIR / "research" / "time_factor_screen.json").read_text(encoding="utf-8")
    )["sectors"]
    sector_of = {
        symbol: sector for sector, members in sector_table.items() for symbol in members
    }
    l1_paths = sorted(L1_DIR.glob("*.csv"))
    if not l1_paths:
        raise FileNotFoundError(f"没有商品 L1 汇总文件：{L1_DIR}")

    l1_fields = {
        "clv_mid",
        "clv_last",
        "spread_day",
        "spread_last30",
        "depth_day",
        "depth_last30",
        "imb_last30",
    }
    l1_by_symbol = {}
    coverage_rows = []
    for path in l1_paths:
        symbol = path.stem.upper()
        table = pd.read_csv(path, parse_dates=["trading_date"])
        missing_fields = l1_fields - set(table.columns)
        if missing_fields:
            raise KeyError(f"{path} 缺少 L1 字段：{sorted(missing_fields)}")
        table["trading_date"] = pd.to_datetime(table["trading_date"]).dt.normalize()
        C.assert_no_holdout_dates(table["trading_date"], f"{symbol} 商品 L1 汇总")
        if table["trading_date"].duplicated().any():
            raise ValueError(f"{symbol} 商品 L1 汇总存在重复交易日")
        table = table.set_index("trading_date").sort_index()
        table["spread_ratio"] = table["spread_last30"] / table["spread_day"].where(
            table["spread_day"] > 0
        )
        table["depth_ratio"] = table["depth_last30"] / table["depth_day"].where(
            table["depth_day"] > 0
        )
        table["spread_worse"] = trailing_worse_state(
            table["spread_ratio"], higher_is_worse=True
        )
        table["depth_worse"] = trailing_worse_state(
            table["depth_ratio"], higher_is_worse=False
        )
        l1_by_symbol[symbol] = table

    selected_symbols = sorted(set(l1_by_symbol) & set(pool_symbols))
    unmapped = sorted(set(selected_symbols) - set(sector_of))
    missing_bars = sorted(set(selected_symbols) - set(bars["open"].columns))
    if unmapped or missing_bars:
        raise ValueError(
            f"L1 研究品种缺板块映射 {unmapped} 或研究分钟分片 {missing_bars}"
        )

    panel_rows = []
    for path in l1_paths:
        symbol = path.stem.upper()
        l1 = l1_by_symbol[symbol]
        in_pool = symbol in selected_symbols
        if in_pool:
            next_intraday = (
                bars["closew"][symbol].shift(-1) - bars["openw"][symbol].shift(-1)
            ) / bars["open"][symbol].shift(-1).where(
                bars["open"][symbol].shift(-1) > 0
            )
            sample = l1.reindex(days)
            sample = sample.assign(
                next_intraday=next_intraday.reindex(days),
                active_pool=pool[symbol].reindex(days).fillna(False),
                symbol=symbol,
                sector=sector_of[symbol],
            )
            sample = sample.loc[sample["active_pool"].astype(bool)].copy()
            sample["clv_spread_interaction"] = sample["clv_mid"] * sample["spread_worse"]
            sample["clv_depth_interaction"] = sample["clv_mid"] * sample["depth_worse"]
            panel_rows.append(sample)
        else:
            sample = l1
        base_valid = (
            sample["clv_mid"].notna()
            & (sample["next_intraday"].notna() if in_pool else False)
        )
        coverage_rows.append(
            {
                "品种": symbol,
                "板块": sector_of.get(symbol, "未分类"),
                "L1行数": len(l1),
                "L1起始日": str(l1.index.min().date()) if len(l1) else "",
                "L1结束日": str(l1.index.max().date()) if len(l1) else "",
                "在研究池": in_pool,
                "池内CLV及次日收益有效天数": int(base_valid.sum()),
                "池内有效年份": int(
                    len(set(sample.index[base_valid].year)) if in_pool else 0
                ),
                "缺失中间价CLV": int(l1["clv_mid"].isna().sum()),
                "中间价CLV超出[0,1]": int(
                    (l1["clv_mid"].lt(0) | l1["clv_mid"].gt(1)).sum()
                ),
                "缺失价差比": int(
                    (l1["spread_ratio"].isna() | ~np.isfinite(l1["spread_ratio"])).sum()
                ),
                "缺失深度比": int(
                    (l1["depth_ratio"].isna() | ~np.isfinite(l1["depth_ratio"])).sum()
                ),
            }
        )
    if not panel_rows:
        raise ValueError("商品研究池与 L1 文件没有可交集品种")
    panel = pd.concat(panel_rows).sort_index()
    panel.index.name = "trading_date"

    def adjust(rows: list[dict]) -> None:
        result = pd.DataFrame(rows)
        for _, indices in result.groupby(["范围", "检验"]).groups.items():
            q_values = benjamini_hochberg_qvalues(result.loc[indices, "NW_t"])
            for idx, value in q_values.items():
                rows[idx]["BH_q"] = float(value) if pd.notna(value) else np.nan

    def fit_all(sample_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        symbol_rows = []
        for symbol, group in sample_panel.groupby("symbol", sort=True):
            symbol_rows.extend(
                _l1_model_rows(
                    group,
                    scope="品种",
                    entity=symbol,
                    minimum_observations=L1_MIN_PRODUCT_OBSERVATIONS,
                    minimum_years=L1_MIN_CALENDAR_YEARS,
                )
            )
        sector_rows = []
        for sector, group in sample_panel.groupby("sector", sort=True):
            sector_rows.extend(
                _l1_model_rows(
                    group,
                    scope="板块",
                    entity=sector,
                    minimum_observations=L1_MIN_SECTOR_OBSERVATIONS,
                    minimum_years=L1_MIN_CALENDAR_YEARS,
                    minimum_products_per_day=L1_MIN_SECTOR_PRODUCTS,
                )
            )
        adjust(symbol_rows)
        adjust(sector_rows)
        return pd.DataFrame(symbol_rows), pd.DataFrame(sector_rows)

    symbol_results, sector_results = fit_all(panel)
    clipped_panel = panel.copy()
    clipped_panel["clv_mid"] = clipped_panel["clv_mid"].clip(0.0, 1.0)
    clipped_panel["clv_spread_interaction"] = (
        clipped_panel["clv_mid"] * clipped_panel["spread_worse"]
    )
    clipped_panel["clv_depth_interaction"] = (
        clipped_panel["clv_mid"] * clipped_panel["depth_worse"]
    )
    clipped_products, clipped_sectors = fit_all(clipped_panel)
    L1_LIQUIDITY_OUT.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(coverage_rows)
    symbol_results.to_csv(
        L1_LIQUIDITY_OUT / "product_results.csv", index=False, encoding="utf-8-sig"
    )
    sector_results.to_csv(
        L1_LIQUIDITY_OUT / "sector_results.csv", index=False, encoding="utf-8-sig"
    )
    clipped_products.to_csv(
        L1_LIQUIDITY_OUT / "product_results_clipped_clv_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    clipped_sectors.to_csv(
        L1_LIQUIDITY_OUT / "sector_results_clipped_clv_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage.to_csv(
        L1_LIQUIDITY_OUT / "coverage.csv", index=False, encoding="utf-8-sig"
    )
    effective = panel.loc[panel["clv_mid"].notna() & panel["next_intraday"].notna()]
    summary = {
        "sample_start": str(effective.index.min().date()),
        "sample_end": str(effective.index.max().date()),
        "l1_files": len(l1_paths),
        "tested_products": int(panel["symbol"].nunique()),
        "tested_sectors": int(panel["sector"].nunique()),
        "pool_observations": len(effective),
        "midpoint_clv_out_of_bounds_rows": int(
            coverage["中间价CLV超出[0,1]"].sum()
        ),
        "read_partitions": [
            "commodity research minute shards (daily bar fields only)",
            "commodity annual universe and prior-year turnover statistics",
            "commodity L1 close summaries",
        ],
        "oos_or_forward_data_read": False,
    }
    (L1_LIQUIDITY_OUT / "protocol.json").write_text(
        json.dumps(
            {"protocol": protocol, "sample": summary},
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print("商品 L1 流动性假设：品种级结果")
    print(
        symbol_results[
            symbol_results["检验"].isin(
                ["reversal", "spread_depletion", "depth_depletion", "spread_provision", "depth_provision"]
            )
        ][["品种或板块", "检验", "观测数", "系数", "NW_t", "BH_q", "最低样本要求满足"]]
        .round(4)
        .to_string(index=False)
    )
    print("商品 L1 流动性假设：板块级结果")
    print(
        sector_results[
            sector_results["检验"].isin(
                ["reversal", "spread_depletion", "depth_depletion", "spread_provision", "depth_provision"]
            )
        ][["品种或板块", "检验", "观测数", "系数", "NW_t", "BH_q", "最低样本要求满足"]]
        .round(4)
        .to_string(index=False)
    )
    print(f"样本与覆盖：{json.dumps(summary, ensure_ascii=False)}")
    print(f"研究结果：{L1_LIQUIDITY_OUT}")


def run_l1(protocol) -> None:
    l1 = load_l1()
    data = load_research(1.0)
    bars = {k: v.reindex(columns=list(L1_SYMBOLS)) for k, v in data.factors.bars.items()}
    keep = data.returns.index[data.returns.index.year.isin(L1_YEARS)]
    seg = segments(bars, list(L1_SYMBOLS))
    fwd = seg["fwd"].mean(axis=1).reindex(keep)
    clv_bar = day_shape(bars, list(L1_SYMBOLS))["clv"]
    m = {f: v.reindex(index=keep, columns=list(L1_SYMBOLS)).mean(axis=1) for f, v in l1.items()}
    clv5 = clv_bar.reindex(keep).mean(axis=1)
    coverage = {s: float(l1["clv_last"][s].reindex(keep).notna().mean()) for s in L1_SYMBOLS}
    agree = float(pd.concat([clv_bar.reindex(keep).stack(), l1["clv_last"].reindex(keep).stack()], axis=1).corr().iloc[0, 1])
    low = (clv5 <= clv5.quantile(0.2)).astype(float).where(clv5.notna())

    preds = []
    for field, label, expected in (("imb_last30", "尾盘不平衡度，低位收盘日减其余日", "<0"),
                                   ("spread_ratio", "尾盘价差/全天，低位收盘日减其余日", ">0"),
                                   ("depth_ratio", "尾盘深度/全天，低位收盘日减其余日", "<0")):
        b, t = slope(m[field], [low])
        preds.append(prediction("P4", label, b, t, expected))
    b, t = slope(m["imb_last30"], [clv5])
    preds.append(prediction("P4b", "尾盘不平衡度对 clv 的斜率", b, t, ">0"))
    b, t = slope(fwd, [m["imb_last30"]])
    preds.append(prediction("P5", "尾盘不平衡度对下一段（bp/单位）", b * 1e4, t, "<0"))
    bc, tc = slope(fwd, [clv5, m["imb_last30"]])
    bi, ti = slope(fwd, [m["imb_last30"], clv5])
    b0, t0 = slope(fwd, [clv5])
    preds.append(prediction("P5b", "5 品种 clv 单独（bp）", b0 * 1e4, t0, "<0"))
    preds.append(prediction("P5c", "加尾盘不平衡度后 clv（bp）", bc * 1e4, tc, "<0"))
    preds.append(prediction("P5d", "同时放入时尾盘不平衡度（bp/单位）", bi * 1e4, ti, "<0"))
    bl, tl = slope(fwd, [m["clv_last"]])
    bm, tm = slope(fwd, [m["clv_mid"]])
    preds.append(prediction("P8", "tick 成交价 clv（bp）", bl * 1e4, tl, "<0"))
    preds.append(prediction("P8b", "tick 中间价 clv（bp）", bm * 1e4, tm, "<0"))
    q = quintile_table(clv5, fwd)
    effect = float(q["下一段bp"].iloc[0] - q["下一段bp"].iloc[-1])
    bounce = float(m["close_vs_mid"].abs().mean() * 1e4)
    preds.append(prediction("P8c", "收盘价偏离中间价 |bp| 对效应（最低组减最高组 bp）之比", bounce / effect if effect else np.nan, np.nan, "小于 1"))
    desc = {"L1 覆盖率": coverage, "tick clv 与分钟 clv 相关": agree, "5 品种 clv 与池子 clv 的说明": "见 protocol",
            "最低组减最高组bp": effect, "收盘价偏离中间价平均|bp|": bounce,
            "尾盘价差/全天均值": float(m["spread_ratio"].mean()), "尾盘深度/全天均值": float(m["depth_ratio"].mean())}

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(OUT / "predictions_l1.csv", index=False, encoding="utf-8-sig")
    q.to_csv(OUT / "quintiles_l1.csv", index=False, encoding="utf-8-sig")
    (OUT / "l1_protocol.json").write_text(json.dumps({
        "protocol": protocol, "window": f"{keep.min().date()}..{keep.max().date()}", "descriptive": desc,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 240)
    print(pd.DataFrame(preds).round(4).to_string(index=False))
    print(q.round(3).to_string(index=False))
    print(json.dumps(desc, ensure_ascii=False, default=float))


# --------------------------------------------------------------------------
# 后续检验：盘口变差、持仓变化、换月和交割前平仓
# --------------------------------------------------------------------------
def minute_tail_impact(minute: pd.DataFrame) -> pd.Series:
    """单品种分钟表 → 每个交易日 14:30–15:00 与 09:00–15:00 的 Amihud 冲击之比（Σ|分钟对数收益| / Σ成交额）。"""
    m = minute.sort_index()
    day = pd.to_datetime(m["trading_date"]).dt.normalize()
    r = np.log(m["close"].where(m["close"] > 0)).groupby(day).diff().abs()
    clock = pd.Series(m.index.strftime("%H:%M"), index=m.index)
    in_day = (clock > DAY_START) & (clock <= DAY_END)
    in_tail = (clock > TAIL_START) & (clock <= DAY_END)
    turn = m["total_turnover"].where(m["total_turnover"] > 0)

    def impact(mask):
        num = r.where(mask).groupby(day).sum(min_count=1)
        den = turn.where(mask & r.notna()).groupby(day).sum(min_count=1)
        return num / den

    out = impact(in_tail) / impact(in_day)
    out.index.name = "trading_date"
    return out.replace([np.inf, -np.inf], np.nan)


def worse_share(frame: pd.DataFrame, higher_is_worse: bool = True) -> pd.Series:
    """各列相对自己过去 250 日（不含当天）中位数是否变差，再取有判断的列里变差的比例。"""
    source = frame if higher_is_worse else -frame
    flags = source.apply(trailing_median_above)
    return flags.mean(axis=1, skipna=True).where(flags.notna().sum(axis=1) > 0)


def total_open_interest(symbol: str, days: pd.DatetimeIndex, maturity: pd.Series) -> tuple[pd.Series, pd.Series]:
    """全部合约持仓量之和，和交割月前一个月合约的持仓减少量（未除总持仓）。逐合约日线收盘 t 已知。"""
    oi = {}
    for path in CONTRACT_DIR.glob(f"{symbol}*.pkl"):
        code = path.stem
        if not (code[: len(symbol)] == symbol and code[len(symbol):].isdigit()):
            continue
        frame = pd.read_pickle(path)
        if frame.empty:
            continue
        dates = pd.DatetimeIndex(frame.index.get_level_values(-1))
        C.assert_no_holdout_dates(dates, f"{code} 逐合约日线")
        oi[code] = pd.Series(frame["open_interest"].to_numpy(dtype=float), index=dates)
    table = pd.DataFrame(oi).reindex(days)
    total = table.sum(axis=1, min_count=1)
    month = days.to_period("M")
    pre = pd.DataFrame(False, index=days, columns=table.columns)
    for code in table.columns:
        if code in maturity.index:
            pre[code] = month == (maturity[code].to_period("M") - 1)
    previous = table.shift(1)
    drop = (previous - table).clip(lower=0).where(pre & previous.notna(), 0.0)
    return total, drop.sum(axis=1)


def roll_recent(symbol: str, days: pd.DatetimeIndex, lookback: int = ROLL_LOOKBACK) -> pd.Series:
    """过去 lookback 个交易日（含当天）主力合约换过为真。主力表当天开盘前已定。"""
    dom = pd.read_pickle(DOMINANT_DIR / f"{symbol}_rank1.pkl")
    dom.index = pd.to_datetime(dom.index)
    C.assert_no_holdout_dates(dom.index, f"{symbol} 主力表")
    dom = dom.reindex(days).ffill()
    switched = dom.ne(dom.shift(1)) & dom.shift(1).notna()
    return switched.astype(float).rolling(lookback, min_periods=1).max().astype(bool)


def interaction_row(tag, label, y, x, state, expected="<0") -> dict:
    """y 对 x × state、x、state 回归，交乘项的系数和 NW t；state 为 0/1 时另报两组各自的斜率。"""
    b, t = slope(y, [x * state, x, state])
    row = prediction(tag, label, b * 1e4, t, expected)
    values = set(pd.unique(state.dropna()))
    if values and values <= {0.0, 1.0}:
        lo = slope(y[state == 0], [x[state == 0]])
        hi = slope(y[state == 1], [x[state == 1]])
        row.update({"状态0斜率bp": lo[0] * 1e4, "状态0t": lo[1], "状态1斜率bp": hi[0] * 1e4, "状态1t": hi[1]})
    return row


def run_followup(protocol) -> None:
    data = load_research(1.0, first_year=FIRST_YEAR)
    C.assert_no_holdout_dates(data.returns.index, "后续检验收益")
    columns = sorted({s for members in data.universe.values() for s in members})
    bars = {k: v.reindex(columns=columns) for k, v in data.factors.bars.items()}
    days = bars["close"].index
    keep = data.returns.index
    pool = S.pool_mask(data.universe, days, columns)
    shape = {k: v.where(pool) for k, v in day_shape(bars, columns).items()}
    fwd_sym = segments(bars, columns)["fwd"].where(pool)

    def k(series):
        return series.reindex(keep)

    y, clv = k(fwd_sym.mean(axis=1)), k(shape["clv"].mean(axis=1))
    same = shape["same_day"]

    impact = {}
    for symbol in columns:
        minute = load_shard(symbol, columns=["close", "total_turnover", "trading_date"])
        C.assert_no_holdout_dates(pd.to_datetime(minute["trading_date"]), f"{symbol} 分钟行情")
        impact[symbol] = minute_tail_impact(minute)
    impact = pd.DataFrame(impact).reindex(index=days, columns=columns).where(pool)
    maturity = pd.read_csv(MATURITY_PATH, parse_dates=["maturity_date"]).set_index("order_book_id")["maturity_date"]
    oi, pre_drop, rolled = {}, {}, {}
    for symbol in columns:
        oi[symbol], pre_drop[symbol] = total_open_interest(symbol, days, maturity)
        rolled[symbol] = roll_recent(symbol, days)
    oi = pd.DataFrame(oi).where(pool)
    d_oi = np.log(oi / oi.shift(1)).where(pool)
    exit_share = (pd.DataFrame(pre_drop) / oi.shift(1)).where(pool)
    rolled = pd.DataFrame(rolled).astype(float).where(pool)

    rows = []
    # F1：L1 五个品种
    l1 = load_l1()
    l1_keep = keep[keep.year.isin(L1_YEARS)]
    sub = list(L1_SYMBOLS)
    sub_bars = {kk: v.reindex(columns=sub) for kk, v in data.factors.bars.items()}
    y5 = segments(sub_bars, sub)["fwd"].mean(axis=1).reindex(l1_keep)
    clv5 = day_shape(sub_bars, sub)["clv"].mean(axis=1).reindex(l1_keep)
    spread_worse = worse_share(l1["spread_ratio"].reindex(columns=sub)).reindex(l1_keep)
    depth_worse = worse_share(l1["depth_ratio"].reindex(columns=sub), higher_is_worse=False).reindex(l1_keep)
    rows.append(interaction_row("F1a", "clv × 尾盘价差变宽比例（L1，5 品种）", y5, clv5, spread_worse))
    rows.append(interaction_row("F1a", "clv × 尾盘深度变薄比例（L1，5 品种）", y5, clv5, depth_worse))
    proxy5 = impact.reindex(columns=sub)
    proxy5_worse = worse_share(proxy5).reindex(l1_keep)
    def rank_corr(a, b):
        a = a.reindex(index=l1_keep, columns=sub).to_numpy().ravel()
        b = b.reindex(index=l1_keep, columns=sub).to_numpy().ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        return float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman"))

    agree = {
        "分钟冲击之比与L1价差之比秩相关": rank_corr(proxy5, l1["spread_ratio"]),
        "分钟冲击之比与L1深度之比秩相关": rank_corr(proxy5, l1["depth_ratio"]),
        "分钟与L1两种变差比例的日相关（5 品种）": float(proxy5_worse.corr(spread_worse)),
    }
    impact_worse = k(worse_share(impact))
    rows.append(interaction_row("F1b", "clv × 尾盘冲击变大比例（分钟代理，全池 2011–2021）", y, clv, impact_worse))
    rows.append(interaction_row("F1b", "clv × 尾盘冲击变大比例（分钟代理，5 品种 2016–2021）", y5, clv5, proxy5_worse))
    median = impact_worse.median()
    rows.append(interaction_row("F1c", "clv × 1{尾盘冲击变大比例高于中位}（全池）", y, clv,
                                (impact_worse > median).astype(float).where(impact_worse.notna())))
    # F2：持仓变化
    pool_doi = k(d_oi.mean(axis=1))
    rows.append(interaction_row("F2a", "clv × 1{池内平均持仓减少}", y, clv,
                                (pool_doi < 0).astype(float).where(pool_doi.notna())))
    rows.append(interaction_row("F2a", "clv × 池内平均持仓减少幅度（−ΔOI，连续）", y, clv, -pool_doi))
    down = same < 0
    liq = (down & (d_oi < 0)).astype(float).where(pool & d_oi.notna() & same.notna())
    news = (down & (d_oi > 0)).astype(float).where(pool & d_oi.notna() & same.notna())
    down_n = down.astype(float).where(pool & same.notna()).sum(axis=1)
    liq_share = k(liq.sum(axis=1) / down_n.replace(0, np.nan))
    pool_down = k(same.mean(axis=1)) < 0
    b, t = slope(y[pool_down], [liq_share[pool_down]])
    rows.append(prediction("F2b", "池子下跌日：减仓下跌占比对下一段（bp/全部）", b * 1e4, t, ">0"))
    diff = fwd_sym.where(liq == 1).mean(axis=1) - fwd_sym.where(news == 1).mean(axis=1)
    enough = (liq.sum(axis=1) >= 2) & (news.sum(axis=1) >= 2)
    diff = k(diff.where(enough)).dropna()
    rows.append(prediction("F2c", "截面：减仓下跌减增仓下跌的下一段（bp）", float(diff.mean() * 1e4),
                           newey_west_t(diff, LAGS), ">0"))
    # F3：换月和交割前被动平仓
    roll_share = rolled.mean(axis=1)
    roll_hi = trailing_median_above(roll_share)
    rows.append(interaction_row("F3a", "clv × 1{换月占比高于自身过去中位}", y, clv,
                                k(roll_hi.astype(float).where(roll_hi.notna()))))
    exit_pool = exit_share.mean(axis=1)
    exit_hi = trailing_median_above(exit_pool)
    rows.append(interaction_row("F3b", "clv × 1{交割前被动平仓高于自身过去中位}", y, clv,
                                k(exit_hi.astype(float).where(exit_hi.notna()))))
    low_clv = clv <= clv.quantile(0.2)
    desc = {**agree, "全池冲击变差比例均值": float(impact_worse.mean()),
            "池内平均ΔOI<0日占比": float((pool_doi < 0).mean()), "池子下跌日数": int(pool_down.sum()),
            "截面差可用天数": int(len(diff)), "换月占比均值": float(k(roll_share).mean()),
            "交割前平仓占总持仓均值": float(k(exit_pool).mean()),
            "低位收盘日的减仓下跌占比": float(liq_share[low_clv].mean()), "其余日": float(liq_share[~low_clv].mean())}

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "predictions_followup.csv", index=False, encoding="utf-8-sig")
    (OUT / "followup_protocol.json").write_text(json.dumps({
        "protocol": protocol.get("followups"), "window": f"{keep.min().date()}..{keep.max().date()}",
        "descriptive": desc,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 20)
    print(table.round(3).to_string(index=False))
    print(json.dumps(desc, ensure_ascii=False, default=float))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="收盘位置反转的经济机制检验")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--l1", action="store_true", help="只跑 L1 盘口部分")
    mode.add_argument("--all-l1", action="store_true", help="用所有可用商品 L1 检验流动性枯竭/提供假设")
    mode.add_argument("--followup", action="store_true", help="盘口变差、持仓变化、换月和交割前平仓")
    args = parser.parse_args(argv)
    protocol = load_l1_liquidity_protocol() if args.all_l1 else load_protocol()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        (
            run_all_l1
            if args.all_l1
            else run_l1
            if args.l1
            else run_followup
            if args.followup
            else run_daily
        )(protocol)


if __name__ == "__main__":
    main()
