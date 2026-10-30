"""检验股指日内高低点相对开盘与VWAP的位置和区间。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from research.index_close_discriminators import _bh_adjust
from research.time_factor_screen import ols_hac
from tfcta import config as C

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "config" / "research" / "index_price_extremes.json"
SYMBOLS = ("IF", "IH", "IC")
FEATURES = (
    "high_open",
    "low_open",
    "range_open",
    "high_twap",
    "low_twap",
    "range_twap",
)
REQUIRED_FIELDS = (
    "trading_date", "open", "close", "openw", "closew",
    "highw", "loww",
)
IF_START = pd.Timestamp("2011-01-01")
COMMON_START = pd.Timestamp("2015-04-16")
SAMPLE_END = pd.Timestamp("2021-12-31")
ROLLING_WINDOW = 60
ROLLING_MINIMUM = 40
HAC_LAGS = 5
WEAK_CLOSE = 0.4


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parameters = protocol.get("parameters", {})
    if (
        protocol.get("status") != "declared_exploratory_research"
        or "不读2022" not in protocol.get("sample_boundary", "")
        or parameters.get("instruments") != list(SYMBOLS)
        or parameters.get("feature_names") != list(FEATURES)
        or parameters.get("rolling_window") != ROLLING_WINDOW
        or parameters.get("rolling_minimum") != ROLLING_MINIMUM
        or parameters.get("hac_lags") != HAC_LAGS
        or parameters.get("weak_close_threshold") != WEAK_CLOSE
    ):
        raise ValueError("高低点偏离检验协议与实现或样本边界不匹配")
    return protocol


def aggregate_daily(minute: pd.DataFrame, symbol: str) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_FIELDS) - set(minute.columns))
    if missing:
        raise KeyError(f"{symbol} 分钟行情缺少字段：{missing}")
    frame = minute.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(frame["trading_date"], what=f"{symbol}高低点偏离分钟数据")
    if "datetime" in frame.columns:
        frame = frame.sort_values(["trading_date", "datetime"], kind="mergesort")
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.sort_index(kind="mergesort")
    else:
        raise KeyError(f"{symbol}分钟数据既没有datetime列，也没有DatetimeIndex")
    if frame[["open", "close", "openw", "closew", "highw", "loww"]].isna().any().any():
        raise ValueError(f"{symbol}分钟价格存在缺失")
    frame["typical_price"] = (frame["highw"] + frame["loww"] + frame["closew"]) / 3

    grouped = frame.groupby("trading_date", sort=True)
    daily = grouped.agg(
        open=("open", "first"),
        openw=("openw", "first"),
        close=("close", "last"),
        closew=("closew", "last"),
        highw=("highw", "max"),
        loww=("loww", "min"),
        twapw=("typical_price", "mean"),
    )
    if daily["open"].le(0).any() or daily["twapw"].le(0).any():
        raise ValueError(f"{symbol}开盘价或分钟典型价TWAP代理非正")
    if daily["twapw"].lt(daily["loww"]).any() or daily["twapw"].gt(daily["highw"]).any():
        raise ValueError(f"{symbol}分钟典型价TWAP代理落在当日高低区间之外")
    return daily


def features_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    symbol_fields = ("open", "openw", "close", "closew", "highw", "loww", "twapw")
    missing = sorted(set(symbol_fields) - set(daily.columns))
    if missing:
        raise KeyError(f"日线缺少特征计算字段：{missing}")
    high_low = daily["highw"] - daily["loww"]
    if high_low.le(0).any() or daily["open"].le(0).any() or daily["twapw"].le(0).any():
        raise ValueError("日线高低区间、开盘价或TWAP代理无效")
    raw_features = pd.DataFrame(index=daily.index)
    raw_features["high_open"] = (daily["highw"] - daily["openw"]) / daily["open"] * 1e4
    raw_features["low_open"] = (daily["loww"] - daily["openw"]) / daily["open"] * 1e4
    raw_features["range_open"] = high_low / daily["open"] * 1e4
    raw_features["high_twap"] = (daily["highw"] - daily["twapw"]) / daily["twapw"] * 1e4
    raw_features["low_twap"] = (daily["loww"] - daily["twapw"]) / daily["twapw"] * 1e4
    raw_features["range_twap"] = high_low / daily["twapw"] * 1e4

    standardized = {}
    for feature in FEATURES:
        prior = raw_features[feature].shift(1)
        mean = prior.rolling(ROLLING_WINDOW, min_periods=ROLLING_MINIMUM).mean()
        scale = prior.rolling(ROLLING_WINDOW, min_periods=ROLLING_MINIMUM).std()
        standardized[feature] = (raw_features[feature] - mean) / scale.where(scale > 0)

    clv = (daily["closew"] - daily["loww"]) / high_low
    outcome = (
        (daily["openw"].shift(-2) - daily["openw"].shift(-1))
        / daily["open"].shift(-1).where(daily["open"].shift(-1) > 0)
    )
    result = pd.DataFrame({"clv": clv, "outcome": outcome}, index=daily.index)
    for feature in FEATURES:
        result[feature] = standardized[feature]
    return result.replace([np.inf, -np.inf], np.nan)


def daily_features(minute: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return features_from_daily(aggregate_daily(minute, symbol))


def _read_minutes(path: Path, symbol: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=list(REQUIRED_FIELDS),
        filters=[("trading_date", "<=", SAMPLE_END.to_pydatetime())],
    )
    if frame.empty:
        raise ValueError(f"{symbol}分钟分片为空：{path}")
    dates = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(dates, what=f"{symbol}价格极值输入")
    if dates.max() > SAMPLE_END:
        raise ValueError(f"{symbol}价格极值输入超过登记结束日 {SAMPLE_END.date()}")
    return frame


def load_if_history() -> pd.DataFrame:
    early_path = C.INDEX_DATA_ROOT / "if_before_ih_ic" / "minute_shards" / "IF.parquet"
    research_path = C.INDEX_RESEARCH_DIR / "IF.parquet"
    early = _read_minutes(early_path, "IF")
    research = _read_minutes(research_path, "IF")
    early_daily = aggregate_daily(early, "IF")
    research_daily = aggregate_daily(research, "IF")

    overlap = early_daily.index.intersection(research_daily.index)
    if overlap.empty:
        raise ValueError("早期IF与研究期IF分片没有重叠日期，无法核验拼接")
    compare_fields = (
        "open", "openw", "close", "closew", "highw", "loww", "twapw",
    )
    left = early_daily.loc[overlap, list(compare_fields)]
    right = research_daily.loc[overlap, list(compare_fields)]
    pd.testing.assert_frame_equal(
        left,
        right,
        check_exact=False,
        rtol=1e-8,
        atol=1e-10,
        check_names=False,
        obj="早期与研究期IF重叠日行情",
    )
    early_only = early_daily.loc[early_daily.index < overlap.max()]
    stitched_daily = pd.concat(
        [early_only, research_daily.loc[research_daily.index >= overlap.max()]]
    )
    if stitched_daily.index.has_duplicates or not stitched_daily.index.is_monotonic_increasing:
        raise ValueError("IF专用历史与研究分片拼接后日期重复或乱序")
    return features_from_daily(stitched_daily)


def load_common_panel() -> pd.DataFrame:
    per_symbol = {}
    for symbol in SYMBOLS:
        path = C.INDEX_RESEARCH_DIR / f"{symbol}.parquet"
        per_symbol[symbol] = daily_features(_read_minutes(path, symbol), symbol)
    common = per_symbol[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        common = common.intersection(per_symbol[symbol].index)
    if common.empty:
        raise ValueError("IF/IH/IC没有共同交易日期")
    C.assert_no_holdout_dates(common, what="IF/IH/IC共同日期")
    panel = pd.DataFrame(index=common)
    for column in ("clv", "outcome", *FEATURES):
        aligned = pd.concat(
            {symbol: per_symbol[symbol][column].reindex(common) for symbol in SYMBOLS},
            axis=1,
        )
        panel[column] = aligned.mean(axis=1, skipna=False)
    return panel


def _p_value(t_stat: float, n: int, columns: int) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    return float(2 * student_t.sf(abs(t_stat), df=max(n - columns - 1, 1)))


def _fit_effects(panel: pd.DataFrame, scope: str, period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = panel.dropna(subset=["outcome", "clv", *FEATURES]).copy()
    if period == "弱收盘":
        sample = sample.loc[sample["clv"] < WEAK_CLOSE]
    primary_rows = []
    interaction_rows = []
    weak = (sample["clv"] < WEAK_CLOSE).astype(float)
    y = sample["outcome"].to_numpy(dtype=float)
    clv = sample["clv"].to_numpy(dtype=float)

    for feature in FEATURES:
        x = sample[feature].to_numpy(dtype=float)
        coefficient, t_stat = ols_hac(y, [x, clv], HAC_LAGS)
        primary_rows.append(
            {
                "样本": scope,
                "区间": period,
                "特征": feature,
                "观测数": len(sample),
                "每1标准差收益变化_bp": coefficient * 1e4,
                "HAC5_t": t_stat,
                "p值": _p_value(t_stat, len(sample), 2),
            }
        )
        if period == "全样本":
            interaction = x * weak.to_numpy(dtype=float)
            int_coef, int_t = ols_hac(
                y,
                [interaction, x, clv, weak.to_numpy(dtype=float)],
                HAC_LAGS,
            )
            interaction_rows.append(
                {
                    "样本": scope,
                    "特征": feature,
                    "观测数": len(sample),
                    "弱收盘增量_bp每标准差": int_coef * 1e4,
                    "HAC5_t": int_t,
                    "p值": _p_value(int_t, len(sample), 4),
                }
            )
    primary = pd.DataFrame(primary_rows)
    primary["BH_q"] = _bh_adjust(primary["p值"].to_numpy())
    interactions = pd.DataFrame(interaction_rows)
    if not interactions.empty:
        interactions["BH_q"] = _bh_adjust(interactions["p值"].to_numpy())
    return primary, interactions


def _annual_effects(panel: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    eligible = panel.dropna(subset=["outcome", "clv", *FEATURES])
    for year, sample in eligible.groupby(eligible.index.year, sort=True):
        for feature in FEATURES:
            coef, t_stat = ols_hac(
                sample["outcome"].to_numpy(dtype=float),
                [
                    sample[feature].to_numpy(dtype=float),
                    sample["clv"].to_numpy(dtype=float),
                ],
                HAC_LAGS,
            )
            rows.append(
                {
                    "样本": scope,
                    "年份": int(year),
                    "特征": feature,
                    "观测数": len(sample),
                    "每1标准差收益变化_bp": coef * 1e4,
                    "HAC5_t": t_stat,
                }
            )
    return pd.DataFrame(rows)


def run() -> None:
    protocol = load_protocol()
    C.assert_research_only(C.INDEX_RESEARCH_DIR)
    if_panel = load_if_history()
    common_panel = load_common_panel()
    if_panel = if_panel.loc[(if_panel.index >= IF_START) & (if_panel.index <= SAMPLE_END)]
    common_panel = common_panel.loc[
        (common_panel.index >= COMMON_START) & (common_panel.index <= SAMPLE_END)
    ]
    if if_panel.empty or common_panel.empty:
        raise ValueError("登记样本内没有可用观测")
    C.assert_no_holdout_dates(if_panel.index, what="IF 2011–2021计分日")
    C.assert_no_holdout_dates(common_panel.index, what="共同样本计分日")

    outputs = []
    interactions = []
    yearly = []
    for name, panel in (("IF 2011-2021", if_panel), ("IF/IH/IC 2015-2021", common_panel)):
        valid = panel.dropna(subset=["outcome", "clv", *FEATURES])
        full, weak_interaction = _fit_effects(valid, name, "全样本")
        weak, _ = _fit_effects(valid, name, "弱收盘")
        outputs.extend([full, weak])
        interactions.append(weak_interaction)
        yearly.append(_annual_effects(valid, name))

    result_dir = C.RUNS_DIR / "index" / "index_price_extremes"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    primary = pd.concat(outputs, ignore_index=True)
    int_result = pd.concat(interactions, ignore_index=True)
    annual = pd.concat(yearly, ignore_index=True)
    primary.to_csv(result_dir / "primary_effects.csv", index=False, encoding="utf-8-sig")
    int_result.to_csv(result_dir / "weak_close_interactions.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(result_dir / "annual_effects.csv", index=False, encoding="utf-8-sig")
    print("主效应（每1标准差特征变化对应的收益bp，已控制CLV）:")
    print(primary.to_string(index=False))
    print("\n弱收盘交互（weak close < 0.4）:")
    print(int_result.to_string(index=False))
    print(f"\nIF有效样本: {len(if_panel.dropna(subset=['outcome', 'clv', *FEATURES]))} 日")
    print(f"IF/IH/IC有效共同样本: {len(common_panel.dropna(subset=['outcome', 'clv', *FEATURES]))} 日")
    print(f"结果目录: {result_dir}")


# --------------------------------------------------------------------------
# 弱收盘日的日内形态（config/research/index_weak_close_shape.json）
# --------------------------------------------------------------------------
SHAPE_PATH = ROOT / "config" / "research" / "index_weak_close_shape.json"
SHAPE_OUT = C.RUNS_DIR / "index" / "index_weak_close_shape"
SHAPE_FEATURES = ("open_location", "high_time", "dfp_top3", "low_dwell", "info_discreteness")
SHAPE_EXTRA = ("high_open", "range_open", "intraday_return")
SHAPE_FIELDS = ("trading_date", "datetime", "open", "close", "openw", "closew", "highw", "loww")
IF_REPLICATION = (pd.Timestamp("2011-01-01"), pd.Timestamp("2015-04-14"))
LOW_DWELL_BAND = 0.2
DFP_LOOKBACK, DFP_PCT = 250, 55.0


def load_shape_protocol() -> dict:
    protocol = json.loads(SHAPE_PATH.read_text(encoding="utf-8"))
    parameters = protocol.get("parameters", {})
    if (
        protocol.get("status") != "declared_exploratory_research"
        or "不读2022" not in protocol.get("sample_boundary", "")
        or parameters.get("features") != list(SHAPE_FEATURES)
        or parameters.get("weak_close_threshold") != WEAK_CLOSE
        or parameters.get("rolling_window") != ROLLING_WINDOW
        or parameters.get("low_dwell_band") != LOW_DWELL_BAND
        or parameters.get("dfp_lookback") != DFP_LOOKBACK
        or parameters.get("dfp_percentile") != DFP_PCT
    ):
        raise ValueError("弱收盘形态检验协议与实现不一致")
    return protocol


def shape_from_minutes(minute: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """单品种分钟表 → 每日原始形态特征（收盘时已知）和下一开盘段收益。"""
    frame = minute.copy()
    if "datetime" not in frame.columns:
        frame = frame.reset_index()                      # 研究分片把 datetime 存成索引
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(frame["trading_date"], what=f"{symbol} 弱收盘形态分钟")
    frame = frame.set_index(pd.to_datetime(frame["datetime"])).sort_index(kind="mergesort")
    if frame[["open", "close", "openw", "closew", "highw", "loww"]].isna().any().any():
        raise ValueError(f"{symbol} 分钟价格缺失")
    day = frame["trading_date"]
    g = frame.groupby(day, sort=True)
    daily = g.agg(open=("open", "first"), openw=("openw", "first"), closew=("closew", "last"),
                  highw=("highw", "max"), loww=("loww", "min"))
    width = (daily["highw"] - daily["loww"]).where(lambda w: w > 0)
    pos = g.cumcount()
    n = g["closew"].transform("size")
    at_max = frame["highw"].eq(g["highw"].transform("max"))
    first_max = at_max & at_max.astype(int).groupby(day).cumsum().eq(1)      # 最高价首次出现的那根
    high_pos = (pos / (n - 1).where(n > 1))[first_max].groupby(day[first_max]).first()
    low_cut = daily["loww"] + LOW_DWELL_BAND * width
    dwell = (frame["closew"] <= low_cut.reindex(day).to_numpy()).groupby(day).mean()
    previous = g["closew"].shift(1)
    step = frame["closew"] - previous.fillna(frame["openw"])
    up, down = (step > 0).groupby(day).mean(), (step < 0).groupby(day).mean()
    intraday = (daily["closew"] - daily["openw"]) / daily["open"]
    out = pd.DataFrame({
        "clv": (daily["closew"] - daily["loww"]) / width,
        "open_location": (daily["openw"] - daily["loww"]) / width,
        "high_time": high_pos,
        "low_dwell": dwell,
        "info_discreteness": np.sign(intraday) * (down - up),
        "high_open": (daily["highw"] - daily["openw"]) / daily["open"],
        "range_open": width / daily["open"],
        "intraday_return": intraday,
        "outcome": (daily["openw"].shift(-2) - daily["openw"].shift(-1)) / daily["open"].shift(-1),
    }, index=daily.index)
    out["dfp_top3"] = _dfp_top3(frame).reindex(out.index)
    return out.replace([np.inf, -np.inf], np.nan)


def _dfp_top3(frame: pd.DataFrame) -> pd.Series:
    """冻结策略口径的 dfp_top3：250 日、55% 分位持续期阈值（不含当天）。"""
    from tfcta.data import add_intraday_coords
    from tfcta.factors import duration_factors

    coords = add_intraday_coords(frame.drop(columns=["datetime"], errors="ignore"))
    return duration_factors(coords, DFP_LOOKBACK, DFP_PCT)["dfp_top3"]


def _rolling_z(series: pd.Series) -> pd.Series:
    prior = series.shift(1)
    mean = prior.rolling(ROLLING_WINDOW, min_periods=ROLLING_MINIMUM).mean()
    scale = prior.rolling(ROLLING_WINDOW, min_periods=ROLLING_MINIMUM).std()
    return (series - mean) / scale.where(scale > 0)


def shape_panel(per_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """等权市场面板：原始特征取均值，标准化特征逐品种 60 日 z 后取均值。"""
    common = None
    for frame in per_symbol.values():
        common = frame.index if common is None else common.intersection(frame.index)
    panel = pd.DataFrame(index=common)
    for column in ("clv", "outcome", *SHAPE_FEATURES, *SHAPE_EXTRA):
        stacked = pd.concat({s: f[column].reindex(common) for s, f in per_symbol.items()}, axis=1)
        panel[column] = stacked.mean(axis=1, skipna=False)
        if column not in ("clv", "outcome"):
            z = pd.concat({s: _rolling_z(f[column]).reindex(common) for s, f in per_symbol.items()}, axis=1)
            panel[f"{column}_z"] = z.mean(axis=1, skipna=False)
    return panel


def _fit(y, xs) -> tuple[float, float, int]:
    frame = pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 40:
        return np.nan, np.nan, len(frame)
    coef, t_stat = ols_hac(frame.iloc[:, 0].to_numpy(dtype=float),
                           [frame.iloc[:, k].to_numpy(dtype=float) for k in range(1, frame.shape[1])], HAC_LAGS)
    return coef, t_stat, len(frame)


def shape_tests(panel: pd.DataFrame, scope: str, signs: dict) -> pd.DataFrame:
    rows = []
    weak = panel["clv"] < WEAK_CLOSE
    w = panel[weak]
    z = lambda name: w[f"{name}_z"]                                      # noqa: E731
    for label, xs in (
        ("high_open_z + CLV（复述）", [z("high_open"), w["clv"]]),
        ("OL + CLV", [w["open_location"], w["clv"]]),
        ("high_open_z | + OL + range_z + CLV", [z("high_open"), w["open_location"], z("range_open"), w["clv"]]),
        ("OL | + high_open_z + range_z + CLV", [w["open_location"], z("high_open"), z("range_open"), w["clv"]]),
        ("OL | + intraday_z + range_z + CLV", [w["open_location"], z("intraday_return"), z("range_open"), w["clv"]]),
    ):
        coef, t_stat, nobs = _fit(w["outcome"], xs)
        rows.append({"样本": scope, "检验": "T0", "模型": label, "第一个变量系数_bp": coef * 1e4, "HAC5_t": t_stat,
                     "天数": nobs})
    p_vals = []
    for feature in SHAPE_FEATURES:
        coef, t_stat, nobs = _fit(w["outcome"], [z(feature), w["clv"]])
        centered = panel[f"{feature}_z"]
        weak_f = weak.astype(float)
        icoef, it, inobs = _fit(panel["outcome"], [centered * weak_f, centered, panel["clv"], weak_f])
        right = np.sign(coef) == signs[feature]
        rows.append({"样本": scope, "检验": "T1", "模型": feature, "预测方向": "+" if signs[feature] > 0 else "−",
                     "弱收盘斜率_bp每标准差": coef * 1e4, "HAC5_t": t_stat, "天数": nobs, "方向一致": bool(right),
                     "全日交互_bp": icoef * 1e4, "交互_t": it})
        p_vals.append(_p_value(t_stat, nobs, 2))
    table = pd.DataFrame(rows)
    t1 = table["检验"] == "T1"
    table.loc[t1, "BH_q"] = _bh_adjust(np.array(p_vals))
    return table


def shape_terciles(panel: pd.DataFrame, years: range, scope: str) -> pd.DataFrame:
    """扩展窗口三分位：切点只用此前年份的弱收盘日。"""
    weak = panel[panel["clv"] < WEAK_CLOSE].dropna(subset=["outcome"])
    rows = []
    for feature in SHAPE_FEATURES:
        for year in years:
            train = weak.loc[weak.index.year < year, feature].dropna()
            test = weak.loc[weak.index.year == year].dropna(subset=[feature])
            if len(train) < 60 or test.empty:
                continue
            low, high = train.quantile([1 / 3, 2 / 3])
            if not low < high:
                raise ValueError(f"{scope} {feature} {year} 三分位切点相等")
            group = np.where(test[feature] <= low, "低", np.where(test[feature] <= high, "中", "高"))
            for name in ("低", "中", "高"):
                part = test.loc[group == name, "outcome"]
                rows.append({"样本": scope, "特征": feature, "年份": int(year), "组": name, "天数": len(part),
                             "反弹率_%": float((part > 0).mean() * 100) if len(part) else np.nan,
                             "均值_bp": float(part.mean() * 1e4) if len(part) else np.nan,
                             "中位_bp": float(part.median() * 1e4) if len(part) else np.nan})
    return pd.DataFrame(rows)


def _tercile_summary(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, feature), part in groups.groupby(["样本", "特征"], sort=False):
        pooled = {}
        for name in ("低", "中", "高"):
            g = part[part["组"] == name]
            days = g["天数"].sum()
            pooled[name] = (days, (g["均值_bp"] * g["天数"]).sum() / days, (g["反弹率_%"] * g["天数"]).sum() / days)
        spread = part.pivot(index="年份", columns="组", values="均值_bp")
        diff = spread["高"] - spread["低"]
        rows.append({"样本": scope, "特征": feature,
                     **{f"{k}组_天数": v[0] for k, v in pooled.items()},
                     **{f"{k}组_均值_bp": v[1] for k, v in pooled.items()},
                     **{f"{k}组_反弹率_%": v[2] for k, v in pooled.items()},
                     "高减低为正年份": f"{int((diff > 0).sum())}/{int(diff.notna().sum())}"})
    return pd.DataFrame(rows)


def _liquidity_link(panel: pd.DataFrame) -> pd.DataFrame:
    from download.index.getQvix import load_research_qvix

    l1 = pd.read_csv(C.RUNS_DIR / "index" / "index_close_location_mechanism" / "daily.csv",
                     parse_dates=["trading_date"]).set_index("trading_date")
    C.assert_no_holdout_dates(l1.index, what="L1 日汇总")
    qvix = load_research_qvix().set_index("date")["close"]
    q = qvix.groupby(qvix.index.year).transform(lambda s: (s - s.mean()) / s.std())
    flows = pd.read_csv(C.INDEX_DATA_ROOT / "external_rqdata" / "market_flows.csv", parse_dates=["date"]).set_index("date")
    C.assert_no_holdout_dates(flows.index, what="市场资金流")
    margin_net = (flows["buy_on_margin_value"] - flows["margin_repayment"]) / flows["margin_balance"].shift(1)
    proxies = pd.DataFrame({"尾盘价差比": l1["spread_ratio"], "尾盘深度比": l1["depth_ratio"],
                            "QVIX年内z": q, "融资净买入": margin_net}).reindex(panel.index)
    weak = panel[(panel["clv"] < WEAK_CLOSE)].join(proxies)
    rows = []
    for feature in SHAPE_FEATURES:
        row = {"特征": feature}
        for name in proxies.columns:
            pair = weak[[feature, name]].dropna()
            row[f"与{name}相关"] = float(pair[feature].rank().corr(pair[name].rank())) if len(pair) > 30 else np.nan
        sub = weak.dropna(subset=["尾盘价差比", "QVIX年内z", f"{feature}_z", "outcome"])
        coef, t_stat, nobs = _fit(sub["outcome"], [sub[f"{feature}_z"], sub["clv"], sub["尾盘价差比"], sub["QVIX年内z"]])
        base, base_t, _ = _fit(sub["outcome"], [sub[f"{feature}_z"], sub["clv"]])
        row.update({"同样本斜率_bp": base * 1e4, "同样本_t": base_t,
                    "控制价差与QVIX后斜率_bp": coef * 1e4, "控制后_t": t_stat, "天数": nobs})
        rows.append(row)
    return pd.DataFrame(rows)


def _early_if_minutes() -> pd.DataFrame:
    early = pd.read_parquet(C.INDEX_DATA_ROOT / "if_before_ih_ic" / "minute_shards" / "IF.parquet",
                            columns=list(SHAPE_FIELDS),
                            filters=[("trading_date", "<=", pd.Timestamp("2015-04-16").to_pydatetime())])
    research = pd.read_parquet(C.INDEX_RESEARCH_DIR / "IF.parquet", columns=list(SHAPE_FIELDS),
                               filters=[("trading_date", "<=", pd.Timestamp("2015-04-16").to_pydatetime())])
    early = early.reset_index() if "datetime" not in early.columns else early
    research = research.reset_index() if "datetime" not in research.columns else research
    for frame, label in ((early, "早期 IF"), (research, "研究期 IF")):
        C.assert_no_holdout_dates(pd.to_datetime(frame["trading_date"]), what=label)
    overlap = sorted(set(pd.to_datetime(early["trading_date"])) & set(pd.to_datetime(research["trading_date"])))
    if not overlap:
        raise ValueError("早期 IF 与研究期 IF 没有重叠日期")
    key = ["datetime", "open", "close", "openw", "closew", "highw", "loww"]
    a = early[pd.to_datetime(early["trading_date"]).isin(overlap)].sort_values("datetime")[key].reset_index(drop=True)
    b = research[pd.to_datetime(research["trading_date"]).isin(overlap)].sort_values("datetime")[key].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-8, atol=1e-8, check_dtype=False,
                                  obj="早期与研究期 IF 重叠日分钟")
    return early


def run_shape() -> None:
    protocol = load_shape_protocol()
    signs = protocol["parameters"]["predicted_sign"]
    C.assert_research_only(C.INDEX_RESEARCH_DIR)
    per_symbol = {}
    for symbol in SYMBOLS:
        minute = pd.read_parquet(C.INDEX_RESEARCH_DIR / f"{symbol}.parquet", columns=list(SHAPE_FIELDS),
                                 filters=[("trading_date", "<=", SAMPLE_END.to_pydatetime())])
        per_symbol[symbol] = shape_from_minutes(minute, symbol)
    common = shape_panel(per_symbol)
    common = common[(common.index >= COMMON_START) & (common.index <= SAMPLE_END)]
    early = shape_panel({"IF": shape_from_minutes(_early_if_minutes(), "IF")})
    early = early[(early.index >= IF_REPLICATION[0]) & (early.index <= IF_REPLICATION[1])]
    C.assert_no_holdout_dates(common.index, what="共同市场形态计分日")
    C.assert_no_holdout_dates(early.index, what="早期 IF 形态计分日")
    tests = pd.concat([shape_tests(common, "IF/IH/IC 2015-2021", signs),
                       shape_tests(early, "IF 2011-2015.04", signs)], ignore_index=True)
    groups = pd.concat([shape_terciles(common, range(2016, 2022), "IF/IH/IC 2015-2021"),
                        shape_terciles(early, range(2012, 2016), "IF 2011-2015.04")], ignore_index=True)
    summary = _tercile_summary(groups)
    weak_common = common[common["clv"] < WEAK_CLOSE]
    corr = weak_common[[*SHAPE_FEATURES, "high_open_z", "intraday_return_z", "range_open_z"]].corr(method="spearman")
    liquidity = _liquidity_link(common)
    SHAPE_OUT.mkdir(parents=True, exist_ok=True)
    tests.to_csv(SHAPE_OUT / "tests.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(SHAPE_OUT / "terciles_by_year.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(SHAPE_OUT / "terciles.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(SHAPE_OUT / "weak_close_correlations.csv", encoding="utf-8-sig")
    liquidity.to_csv(SHAPE_OUT / "liquidity_link.csv", index=False, encoding="utf-8-sig")
    pd.concat({"common": common, "if_early": early}).to_csv(SHAPE_OUT / "daily_panel.csv", encoding="utf-8-sig")
    coverage = {"共同市场": [str(common.index.min().date()), str(common.index.max().date()), int((common["clv"] < WEAK_CLOSE).sum())],
                "早期IF": [str(early.index.min().date()), str(early.index.max().date()), int((early["clv"] < WEAK_CLOSE).sum())],
                "dfp首个有效日_共同": str(common["dfp_top3"].dropna().index.min().date())}
    (SHAPE_OUT / "protocol.json").write_text(json.dumps({"protocol": protocol, "coverage": coverage},
                                                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(tests.round(3).to_string(index=False))
    print(summary.round(1).to_string(index=False))
    print(corr.round(2).to_string())
    print(liquidity.round(3).to_string(index=False))
    print(coverage)
    print(f"结果：{SHAPE_OUT}")


if __name__ == "__main__":
    import sys

    run_shape() if "--shape" in sys.argv[1:] else run()
