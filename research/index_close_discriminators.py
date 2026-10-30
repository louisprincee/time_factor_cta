"""检验弱收盘日的常用量价因子是否能区分反转与延续。

    python -m research.index_close_discriminators
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata, t as student_t

from research.time_factor_screen import ols_hac
from tfcta import config as C
from tfcta.data import load_shard

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "research" / "index_close_discriminators.json"
OUT = C.RUNS_DIR / "index" / "index_close_discriminators"
SYMBOLS = ("IF", "IH", "IC")
FEATURES = (
    "ret_5d",
    "ret_20d",
    "overnight_gap",
    "intraday_return",
    "range_surprise",
    "volume_surprise",
    "signed_volume_surprise",
    "amihud_surprise",
)
WEAK_CLOSE = 0.4
SCORE_START = pd.Timestamp("2015-04-16")
SCORE_END = pd.Timestamp("2021-12-31")
EVAL_YEARS = tuple(range(2016, 2022))
HAC_LAGS = 5
ROLLING_WINDOW = 60
ROLLING_MINIMUM = 40
LOGISTIC_C = 1.0
RANGE_GROUPS = ("低", "中", "高")
REQUIRED_FIELDS = (
    "trading_date", "open", "openw", "close", "closew", "highw", "loww",
    "volume", "total_turnover",
)


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    parameters = protocol.get("parameters", {})
    grouped_follow_up = protocol.get("follow_up_grouped_range_surprise", {})
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022" not in sample
        or "不读 2026" not in sample
        or protocol.get("candidate_features") is None
        or tuple(protocol["candidate_features"]) != FEATURES
        or parameters.get("instruments") != list(SYMBOLS)
        or parameters.get("weak_close_threshold") != WEAK_CLOSE
        or parameters.get("rolling_window") != ROLLING_WINDOW
        or parameters.get("rolling_minimum") != ROLLING_MINIMUM
        or parameters.get("hac_lags") != HAC_LAGS
        or parameters.get("logistic_C") != LOGISTIC_C
        or parameters.get("evaluation_years") != list(EVAL_YEARS)
        or grouped_follow_up.get("status") != "post_hoc_exploratory_follow_up"
        or grouped_follow_up.get("feature") != "使用已登记的 range_surprise：日内振幅 / 此前20日振幅中位数，再按此前60日均值和标准差标准化；标准化不含当天。"
    ):
        raise ValueError("弱收盘区分力检验登记与实现或样本边界不一致")
    return protocol


def _daily_from_minutes(minute: pd.DataFrame, symbol: str) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_FIELDS) - set(minute.columns))
    if missing:
        raise KeyError(f"{symbol} 分片缺少区分力检验所需字段：{missing}")
    frame = minute.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(frame["trading_date"], what=f"{symbol} 区分力分钟数据")
    grouped = frame.groupby("trading_date", sort=True)
    daily = pd.DataFrame({
        "open": grouped["open"].first(),
        "openw": grouped["openw"].first(),
        "close": grouped["close"].last(),
        "closew": grouped["closew"].last(),
        "highw": grouped["highw"].max(),
        "loww": grouped["loww"].min(),
        "volume": grouped["volume"].sum(min_count=1),
        "turnover": grouped["total_turnover"].sum(min_count=1),
    })
    daily.index = pd.DatetimeIndex(daily.index)
    return daily


def symbol_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """单品种日线转为收盘已知特征及下一开盘段结果。"""
    required = {
        "open", "openw", "close", "closew", "highw", "loww", "volume", "turnover",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise KeyError(f"日线缺少字段：{missing}")
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("日线交易日必须唯一且升序")

    previous_close = daily["close"].shift(1).where(daily["close"].shift(1) > 0)
    close_to_close = (daily["closew"] - daily["closew"].shift(1)) / previous_close
    overnight_gap = (daily["openw"] - daily["closew"].shift(1)) / previous_close
    intraday_return = (daily["closew"] - daily["openw"]) / daily["open"].where(daily["open"] > 0)
    daily_range = (daily["highw"] - daily["loww"]) / previous_close

    prior_range = daily_range.shift(1).rolling(20, min_periods=20).median()
    prior_volume = daily["volume"].shift(1).rolling(20, min_periods=20).median()
    amihud = np.log(
        close_to_close.abs().where(close_to_close.abs() > 0) /
        daily["turnover"].where(daily["turnover"] > 0)
    )
    prior_amihud = amihud.shift(1).rolling(20, min_periods=20).median()

    raw_features = pd.DataFrame(index=daily.index)
    raw_features["ret_5d"] = (
        daily["closew"] - daily["closew"].shift(5)
    ) / daily["close"].shift(5).where(daily["close"].shift(5) > 0)
    raw_features["ret_20d"] = (
        daily["closew"] - daily["closew"].shift(20)
    ) / daily["close"].shift(20).where(daily["close"].shift(20) > 0)
    raw_features["overnight_gap"] = overnight_gap
    raw_features["intraday_return"] = intraday_return
    raw_features["range_surprise"] = daily_range / prior_range.where(prior_range > 0)
    raw_features["volume_surprise"] = np.log(
        daily["volume"].where(daily["volume"] > 0) / prior_volume.where(prior_volume > 0)
    )
    raw_features["signed_volume_surprise"] = np.sign(close_to_close) * raw_features["volume_surprise"]
    raw_features["amihud_surprise"] = amihud - prior_amihud

    standardized = {}
    for name in FEATURES:
        past = raw_features[name].shift(1)
        mean = past.rolling(ROLLING_WINDOW, min_periods=ROLLING_MINIMUM).mean()
        scale = past.rolling(ROLLING_WINDOW, min_periods=ROLLING_MINIMUM).std()
        standardized[name] = (
            (raw_features[name] - mean) / scale.where(scale > 0)
        )

    high_low = daily["highw"] - daily["loww"]
    clv = (daily["closew"] - daily["loww"]) / high_low.where(high_low > 0)
    future_return = (
        daily["openw"].shift(-2) - daily["openw"].shift(-1)
    ) / daily["open"].shift(-1).where(daily["open"].shift(-1) > 0)
    out = pd.DataFrame({"clv": clv, "outcome_return": future_return}, index=daily.index)
    for name, values in standardized.items():
        out[name] = values
    return out.replace([np.inf, -np.inf], np.nan)


def build_market_panel(daily_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if tuple(daily_by_symbol) != SYMBOLS:
        raise ValueError(f"必须按 {SYMBOLS} 顺序提供全部品种")
    per_symbol = {symbol: symbol_panel(daily_by_symbol[symbol]) for symbol in SYMBOLS}
    dates = per_symbol[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        dates = dates.intersection(per_symbol[symbol].index)
    dates = dates[(dates >= SCORE_START) & (dates <= SCORE_END)]
    if dates.empty:
        raise ValueError("共同上市研究期没有共同交易日")
    C.assert_no_holdout_dates(dates, what="股指弱收盘区分力共同日期")
    panel = pd.DataFrame(index=dates)
    for name in ("clv", "outcome_return", *FEATURES):
        aligned = pd.concat(
            {symbol: per_symbol[symbol][name].reindex(dates) for symbol in SYMBOLS},
            axis=1,
        )
        panel[name] = aligned.mean(axis=1, skipna=False)
    return panel


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _fit_logistic(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    if len(train_y) < 30 or np.unique(train_y).size < 2:
        raise ValueError("扩展窗口训练集不足或只有一个结果类别")
    train_z, test_z = _standardize(train_x, test_x)
    design = np.column_stack([np.ones(len(train_z)), train_z])
    test_design = np.column_stack([np.ones(len(test_z)), test_z])

    def objective(beta):
        linear = design @ beta
        loss = np.logaddexp(0.0, linear).sum() - np.dot(train_y, linear)
        penalty = 0.5 * np.dot(beta[1:], beta[1:]) / LOGISTIC_C
        gradient = design.T @ (expit(linear) - train_y)
        gradient[1:] += beta[1:] / LOGISTIC_C
        return loss + penalty, gradient

    fitted = minimize(
        objective,
        np.zeros(design.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"logistic 回归未收敛：{fitted.message}")
    return expit(test_design @ fitted.x)


def _classification_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=int)
    probability = np.asarray(probability, dtype=float)
    positives = int(actual.sum())
    negatives = len(actual) - positives
    if not positives or not negatives:
        auc = np.nan
        balanced_accuracy = np.nan
    else:
        ranks = rankdata(probability, method="average")
        auc = float((ranks[actual == 1].sum() - positives * (positives + 1) / 2) /
                    (positives * negatives))
        predicted = probability >= 0.5
        sensitivity = float(predicted[actual == 1].mean())
        specificity = float((~predicted[actual == 0]).mean())
        balanced_accuracy = (sensitivity + specificity) / 2
    return {
        "AUC": auc,
        "Brier": float(np.mean((probability - actual) ** 2)),
        "平衡准确率": balanced_accuracy,
        "反转占比": float(actual.mean()),
        "样本数": int(len(actual)),
    }


def _walk_forward(panel: pd.DataFrame, weak_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []
    for feature in FEATURES:
        columns = ["clv", "outcome_return", feature]
        usable = panel.dropna(subset=columns).copy()
        usable["label"] = (usable["outcome_return"] > 0).astype(int)
        factor_name = feature
        for year in EVAL_YEARS:
            train = usable[usable.index.year < year]
            test = usable[usable.index.year == year]
            if weak_only:
                train = train[train["clv"] < WEAK_CLOSE]
                test = test[test["clv"] < WEAK_CLOSE]
            if test.empty:
                raise ValueError(f"{year} 年 {factor_name} 预测集为空")

            for model_label, additional_feature in (("CLV基线", None), (feature, feature)):
                predictor_columns = ["clv"] + ([additional_feature] if additional_feature else [])
                probability = _fit_logistic(
                    train[predictor_columns].to_numpy(dtype=float),
                    train["label"].to_numpy(dtype=int),
                    test[predictor_columns].to_numpy(dtype=float),
                )
                actual = test["label"].to_numpy(dtype=int)
                metrics = _classification_metrics(actual, probability)
                rows.append({"样本": "弱收盘" if weak_only else "全样本", "年份": year,
                             "因子": factor_name, "模型": model_label, **metrics})
                predictions.extend(
                    {"样本": "弱收盘" if weak_only else "全样本", "年份": year,
                     "因子": factor_name, "模型": model_label,
                     "日期": date, "标签": int(y), "概率": float(p)}
                    for date, y, p in zip(test.index, actual, probability)
                )
    fold_table = pd.DataFrame(rows)
    prediction_table = pd.DataFrame(predictions)
    pooled = []
    for (factor_name, model), group in prediction_table.groupby(["因子", "模型"], sort=False):
        metrics = _classification_metrics(group["标签"].to_numpy(), group["概率"].to_numpy())
        pooled.append({"样本": "弱收盘" if weak_only else "全样本", "年份": "合并",
                       "因子": factor_name, "模型": model, **metrics})
    return pd.concat([fold_table, pd.DataFrame(pooled)], ignore_index=True), prediction_table


def _range_surprise_group_predictions(
    panel: pd.DataFrame,
    evaluation_years: tuple[int, ...] = EVAL_YEARS,
    score_start: pd.Timestamp | None = None,
    score_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """用扩展训练样本的相对振幅三分位切点分组弱收盘日。"""
    feature = "range_surprise"
    usable = panel.loc[
        panel["clv"] < WEAK_CLOSE, [feature, "outcome_return"]
    ].dropna()
    rows = []
    for year in evaluation_years:
        train = usable[usable.index.year < year]
        test = usable[usable.index.year == year]
        if score_start is not None:
            test = test.loc[test.index >= score_start]
        if score_end is not None:
            test = test.loc[test.index <= score_end]
        if len(train) < 30 or test.empty:
            raise ValueError(f"{year} 年相对振幅分组训练集或预测集为空")
        lower, upper = train[feature].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"{year} 年相对振幅训练三分位切点无效")
        codes = np.searchsorted([lower, upper], test[feature].to_numpy(), side="right")
        rows.extend(
            {
                "年份": year,
                "日期": date,
                "相对振幅_z": float(value),
                "分组": RANGE_GROUPS[int(code)],
                "训练低切点": float(lower),
                "训练高切点": float(upper),
                "下一开盘段收益_bp": float(outcome * 1e4),
                "反转代理": bool(outcome > 0),
            }
            for date, value, code, outcome in zip(
                test.index, test[feature], codes, test["outcome_return"]
            )
        )
    return pd.DataFrame(rows)


def _early_if_range_panel(path: Path) -> pd.DataFrame:
    required = ("trading_date", "open", "openw", "close", "closew", "highw", "loww")
    minute = pd.read_parquet(path, columns=list(required))
    missing = sorted(set(required) - set(minute.columns))
    if missing:
        raise KeyError(f"早期 IF 分片缺少字段：{missing}")
    minute["trading_date"] = pd.to_datetime(minute["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(minute["trading_date"], what="IF 早期相对振幅分钟数据")
    if (
        minute["trading_date"].min() != pd.Timestamp("2010-04-16")
        or minute["trading_date"].max() != pd.Timestamp("2015-04-16")
    ):
        raise ValueError("早期 IF 相对振幅分片日期范围与登记不符")

    grouped = minute.groupby("trading_date", sort=True)
    daily = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "openw": grouped["openw"].first(),
            "close": grouped["close"].last(),
            "closew": grouped["closew"].last(),
            "highw": grouped["highw"].max(),
            "loww": grouped["loww"].min(),
        }
    )
    previous_close = daily["close"].shift(1).where(daily["close"].shift(1) > 0)
    daily_range = (daily["highw"] - daily["loww"]) / previous_close
    prior_range = daily_range.shift(1).rolling(20, min_periods=20).median()
    range_surprise = daily_range / prior_range.where(prior_range > 0)
    past = range_surprise.shift(1)
    mean = past.rolling(ROLLING_WINDOW, min_periods=ROLLING_MINIMUM).mean()
    scale = past.rolling(ROLLING_WINDOW, min_periods=ROLLING_MINIMUM).std()
    clv = (daily["closew"] - daily["loww"]) / (daily["highw"] - daily["loww"])
    outcome_return = (
        (daily["openw"].shift(-2) - daily["openw"].shift(-1))
        / daily["open"].shift(-1).where(daily["open"].shift(-1) > 0)
    )
    panel = pd.DataFrame(
        {
            "clv": clv,
            "range_surprise": (range_surprise - mean) / scale.where(scale > 0),
            "outcome_return": outcome_return,
        },
        index=daily.index,
    ).replace([np.inf, -np.inf], np.nan)
    if pd.isna(panel.loc["2015-04-14", "outcome_return"]):
        raise ValueError("缺少 2015-04-14 的完整下一开盘段收益")
    if pd.notna(panel.loc["2015-04-15", "outcome_return"]):
        raise ValueError("2015-04-15 不应有 IF 单独样本的下一开盘段收益")
    return panel


def run_early_if_range_groups() -> None:
    spec = json.loads(
        (C.PROJECT_ROOT / "config" / "research" / "if_before_ih_ic.json").read_text(
            encoding="utf-8"
        )
    )
    follow_up = spec.get("follow_up_grouped_range_surprise", {})
    if (
        follow_up.get("status") != "post_hoc_exploratory_follow_up"
        or "不读 2022" not in spec.get("sample", "")
        or "不读取任何 holdout_locked 分区" not in spec.get("sample", "")
        or follow_up.get("validation", "").find("2011-01-01 至 2015-04-14") < 0
    ):
        raise ValueError("早期 IF 相对振幅分组未登记或样本边界不匹配")

    source = C.INDEX_DATA_ROOT / "if_before_ih_ic" / "minute_shards" / "IF.parquet"
    panel = _early_if_range_panel(source)
    predictions = _range_surprise_group_predictions(
        panel,
        evaluation_years=(2011, 2012, 2013, 2014, 2015),
        score_start=pd.Timestamp("2011-01-01"),
        score_end=pd.Timestamp("2015-04-14"),
    )
    summary = _summarize_range_groups(predictions)
    output = C.RUNS_DIR / "index" / "if_before_ih_ic"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output / "range_surprise_group_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output / "range_surprise_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output / "range_surprise_protocol.json").write_text(
        json.dumps(follow_up, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"IF 单独信号日: {predictions['日期'].min().date()}..{predictions['日期'].max().date()}")
    print(f"有效弱收盘观测: {len(predictions)}")
    print(summary.to_string(index=False))
    print(f"结果目录: {output}")


def _summarize_range_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, year_data in [
        *[(int(year), group) for year, group in predictions.groupby("年份", sort=True)],
        ("合并", predictions),
    ]:
        for label in RANGE_GROUPS:
            group = year_data[year_data["分组"] == label]
            returns = group["下一开盘段收益_bp"]
            rows.append(
                {
                    "年份": year,
                    "相对振幅组": label,
                    "观测数": len(group),
                    "反转率_%": group["反转代理"].mean() * 100,
                    "平均收益_bp": returns.mean(),
                    "中位收益_bp": returns.median(),
                }
            )
    return pd.DataFrame(rows)


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for position in range(len(values) - 1, -1, -1):
        rank = position + 1
        original_index = order[position]
        running = min(running, values[original_index] * len(values) / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted


def _interaction_regression(panel: pd.DataFrame, start, end, label: str) -> pd.DataFrame:
    sample = panel.loc[(panel.index >= start) & (panel.index <= end)].copy()
    weak = (sample["clv"] < WEAK_CLOSE).astype(float)
    rows = []
    for name in FEATURES:
        predictors = [
            ("interaction", sample[name] * weak),
            ("clv", sample["clv"]),
            ("weak_close", weak),
            (name, sample[name]),
        ]
        coefficient, t_stat = ols_hac(
            sample["outcome_return"].to_numpy(),
            [series.to_numpy() for _, series in predictors],
            HAC_LAGS,
        )
        n = int(sample[["outcome_return", "clv", name]].dropna().shape[0])
        p_value = (
            float(2 * student_t.sf(abs(t_stat), df=max(n - len(predictors) - 1, 1)))
            if np.isfinite(t_stat) else np.nan
        )
        rows.append({
            "样本": label, "因子": name, "交互系数_bp": coefficient * 1e4,
            "NW_t": t_stat, "p_双侧": p_value, "样本数": n,
        })
    result = pd.DataFrame(rows)
    result["BH_q"] = _bh_adjust(result["p_双侧"].to_numpy())
    return result


def _yearly_interactions(panel: pd.DataFrame, start, end, label: str) -> pd.DataFrame:
    parts = []
    sample = panel.loc[(panel.index >= start) & (panel.index <= end)]
    for year, group in sample.groupby(sample.index.year):
        part = _interaction_regression(group, group.index.min(), group.index.max(), label)
        parts.append(part.assign(年份=int(year)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run() -> None:
    protocol = load_protocol()
    C.assert_research_only(C.INDEX_RESEARCH_DIR)
    daily_by_symbol = {}
    for symbol in SYMBOLS:
        minute = load_shard(
            symbol,
            directory=C.INDEX_RESEARCH_DIR,
            columns=list(REQUIRED_FIELDS),
        )
        daily_by_symbol[symbol] = _daily_from_minutes(minute, symbol)
    panel = build_market_panel(daily_by_symbol)
    main = _interaction_regression(panel, SCORE_START, SCORE_END, "IF/IH/IC共同")
    main_years = _yearly_interactions(panel, SCORE_START, SCORE_END, "IF/IH/IC共同")

    weak_scores, weak_predictions = _walk_forward(panel, weak_only=True)
    all_scores, all_predictions = _walk_forward(panel, weak_only=False)
    range_group_predictions = _range_surprise_group_predictions(panel)
    range_group_summary = _summarize_range_groups(range_group_predictions)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    main.to_csv(OUT / "common_interactions.csv", index=False, encoding="utf-8-sig")
    main_years.to_csv(OUT / "common_interactions_by_year.csv", index=False, encoding="utf-8-sig")
    weak_scores.to_csv(OUT / "weak_close_walk_forward.csv", index=False, encoding="utf-8-sig")
    all_scores.to_csv(OUT / "all_days_walk_forward.csv", index=False, encoding="utf-8-sig")
    weak_predictions.to_csv(OUT / "weak_close_predictions.csv", index=False, encoding="utf-8-sig")
    all_predictions.to_csv(OUT / "all_days_predictions.csv", index=False, encoding="utf-8-sig")
    range_group_predictions.to_csv(
        OUT / "range_surprise_groups.csv", index=False, encoding="utf-8-sig"
    )
    range_group_summary.to_csv(
        OUT / "range_surprise_group_summary.csv", index=False, encoding="utf-8-sig"
    )

    print(f"共同面板: {panel.index.min().date()}..{panel.index.max().date()}, {len(panel)} 天")
    print(f"弱收盘有效样本: {(panel['clv'] < WEAK_CLOSE).sum()} 天")
    print("\n共同市场弱收盘交互回归:")
    print(main[["因子", "交互系数_bp", "NW_t", "BH_q", "样本数"]].to_string(index=False))
    print("\n弱收盘逐年扩展窗口预测差（相对同因子样本的 CLV 基线）:")
    weak_pooled = weak_scores[weak_scores["年份"] == "合并"].set_index(["因子", "模型"])
    for name in FEATURES:
        base = weak_pooled.loc[(name, "CLV基线")]
        row = weak_pooled.loc[(name, name)]
        factor_folds = weak_scores[
            (weak_scores["因子"] == name) & (weak_scores["模型"] == name)
            & (weak_scores["年份"] != "合并")
        ].set_index("年份")
        baseline_folds = weak_scores[
            (weak_scores["因子"] == name) & (weak_scores["模型"] == "CLV基线")
            & (weak_scores["年份"] != "合并")
        ].set_index("年份")
        better_years = int((factor_folds["Brier"] < baseline_folds["Brier"]).sum())
        print(
            f"{name}: n={int(row['样本数'])}, ΔAUC={row['AUC'] - base['AUC']:+.3f}, "
            f"ΔBrier={row['Brier'] - base['Brier']:+.4f}, "
            f"Brier改善年份={better_years}/6"
        )
    print("\n早期 IF 2011–2014 不在当前股指研究分片中；该阶段只引用既有研究复核记录。")
    print("\n弱收盘日相对振幅分组（逐年仅用此前样本的三分位切点）:")
    print(range_group_summary.to_string(index=False))
    print(f"\n结果目录: {OUT}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--early-if-range", action="store_true")
    args = parser.parse_args()
    if args.early_if_range:
        run_early_if_range_groups()
    else:
        run()
