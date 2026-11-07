"""联合检验弱收盘广度、日内路径和 L1 压力的逐年预测增量。"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from backtest.engine import _load_index_research_inputs
from research.time_factor_screen import ols_hac
from strategies.index_common import SYMBOLS, high_open_z, market_close_location
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_weak_close_mechanism_classifier.json"
OUT = C.study_dir("index", "index_weak_close_mechanism_classifier")
SCORE_START = pd.Timestamp("2016-01-13")
SCORE_END = pd.Timestamp("2021-12-29")
WEAK_THRESHOLD = 0.4
L1_FEATURES = (
    "spread_day",
    "spread_last30",
    "depth_day",
    "depth_last30",
    "imb_last30",
    "n_ticks",
)
FEATURES = ("weak_breadth", "high_open_z", "l1_pressure")
MODELS = {
    "CLV基线": (),
    "CLV+弱收盘广度": ("weak_breadth",),
    "CLV+日内先涨后跌": ("high_open_z",),
    "CLV+L1尾盘压力": ("l1_pressure",),
    "CLV+全部信号块": FEATURES,
}
MODEL_IDS = (
    "clv_baseline",
    "clv_plus_breadth",
    "clv_plus_high_open",
    "clv_plus_l1_pressure",
    "clv_plus_all",
)
EVALUATION_YEARS = (2017, 2018, 2019, 2020, 2021)
LOGISTIC_C = 1.0
HAC_LAGS = 5


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parameters = protocol.get("parameters", {})
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or parameters.get("instruments") != list(SYMBOLS)
        or parameters.get("weak_close_threshold") != WEAK_THRESHOLD
        or parameters.get("score_start") != str(SCORE_START.date())
        or parameters.get("score_end") != str(SCORE_END.date())
        or parameters.get("evaluation_years") != list(EVALUATION_YEARS)
        or parameters.get("logistic_C") != LOGISTIC_C
        or parameters.get("hac_lags") != HAC_LAGS
        or parameters.get("models") != list(MODEL_IDS)
        or parameters.get("training_years") != [2016]
        or not protocol.get("coverage_exclusions")
        or protocol["coverage_exclusions"][0].get("date") != "2020-02-03"
        or any(phrase not in sample for phrase in ("不读 2022", "不读取任何 OOS"))
    ):
        raise ValueError("弱收盘机制分类协议与实现或样本边界不一致")
    return protocol


def _individual_clv(bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    close = bars["closew"].reindex(index=index, columns=SYMBOLS)
    high = bars["highw"].reindex(index=index, columns=SYMBOLS)
    low = bars["loww"].reindex(index=index, columns=SYMBOLS)
    width = high - low
    if close.isna().any().any() or width.isna().any().any() or width.le(0).any().any():
        raise ValueError("机制分类的单品种 CLV 行情缺失或区间无效")
    clv = (close - low) / width
    if not np.isfinite(clv.to_numpy()).all() or not clv.ge(0).all().all() or not clv.le(1).all().all():
        raise ValueError("机制分类的单品种 CLV 超出 [0, 1]")
    return clv


def _load_l1_daily(
    index: pd.DatetimeIndex,
    excluded_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    parts = []
    required = {"trading_date", *L1_FEATURES}
    missing_source_dates = pd.DatetimeIndex([])
    for symbol in SYMBOLS:
        path = C.INDEX_L1_CLOSE_DIR / f"{symbol}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {symbol} 研究期 L1 收盘汇总：{path}")
        C.assert_research_only(path)
        frame = pd.read_csv(path, parse_dates=["trading_date"])
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{symbol} L1 收盘汇总缺少字段：{missing}")
        frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
        C.assert_no_holdout_dates(frame["trading_date"], what=f"{symbol} 弱收盘分类 L1")
        if frame.duplicated("trading_date").any():
            raise ValueError(f"{symbol} L1 收盘汇总存在重复日期")
        frame = frame.set_index("trading_date").reindex(index)
        missing_source_dates = missing_source_dates.union(
            frame.index[frame[list(L1_FEATURES)].isna().any(axis=1)]
        )
        parts.append(frame[list(L1_FEATURES)])

    if not excluded_dates.isin(index).all():
        raise ValueError("登记的 L1 覆盖排除日期不在计分区间内")
    if not excluded_dates.isin(missing_source_dates).all():
        raise ValueError("登记的 L1 覆盖排除原因已不复现；请重新登记研究样本")
    daily = pd.concat(
        {symbol: part for symbol, part in zip(SYMBOLS, parts)},
        axis=1,
    )
    missing = daily.isna().any(axis=1)
    unexpected_missing = daily.index[missing].difference(excluded_dates)
    if len(unexpected_missing):
        raise ValueError(f"弱收盘分类 L1 日汇总覆盖不完整：{unexpected_missing[:5].tolist()}")
    if not daily.loc[excluded_dates].isna().any(axis=1).all():
        raise ValueError("登记排除日的 L1 覆盖缺口未复现")
    daily = daily.loc[~daily.index.isin(excluded_dates)]
    values = daily.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (daily.xs("n_ticks", axis=1, level=1) <= 0).any().any():
        raise ValueError("弱收盘分类 L1 数据含无效值或无 tick 日期")

    spread_day = daily.xs("spread_day", axis=1, level=1)
    spread_tail = daily.xs("spread_last30", axis=1, level=1)
    depth_day = daily.xs("depth_day", axis=1, level=1)
    depth_tail = daily.xs("depth_last30", axis=1, level=1)
    if (
        spread_day.le(0).any().any()
        or spread_tail.le(0).any().any()
        or depth_day.le(0).any().any()
        or depth_tail.le(0).any().any()
    ):
        raise ValueError("L1 相对价差和深度原始均值必须为正")
    spread = spread_tail / spread_day
    depth = depth_tail / depth_day
    if spread.le(0).any().any() or depth.le(0).any().any():
        raise ValueError("L1 相对价差和深度必须为正，才能计算压力指标")
    pressure = np.log(spread / depth).mean(axis=1, skipna=False)
    return pd.DataFrame(
        {
            "l1_pressure": pressure,
            "l1_imbalance": daily.xs("imb_last30", axis=1, level=1).mean(axis=1, skipna=False),
            "l1_spread_ratio": spread.mean(axis=1, skipna=False),
            "l1_depth_ratio": depth.mean(axis=1, skipna=False),
        },
        index=daily.index,
    )


def build_panel(
    bars: dict[str, pd.DataFrame],
    l1: pd.DataFrame,
    excluded_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    dates = bars["open"].index
    dates = dates[(dates >= SCORE_START) & (dates <= SCORE_END)]
    expected_dates = dates.difference(excluded_dates)
    if not l1.index.equals(expected_dates):
        raise ValueError("弱收盘机制分类 L1 日期与登记计分日期/排除项不一致")
    dates = expected_dates
    C.assert_no_holdout_dates(dates, what="弱收盘机制分类计分日期")
    if dates.empty:
        raise ValueError("弱收盘机制分类没有研究期计分日期")

    per_symbol = _individual_clv(bars, bars["open"].index)
    pooled_clv = market_close_location(bars, bars["open"].index)
    weak_breadth = per_symbol.lt(WEAK_THRESHOLD).mean(axis=1)
    next_open = bars["open"].shift(-1)
    next_openw = bars["openw"].shift(-1)
    after_openw = bars["openw"].shift(-2)
    outcome_by_symbol = (after_openw - next_openw) / next_open.where(next_open > 0)
    outcome = outcome_by_symbol.mean(axis=1, skipna=False)
    rally_z = high_open_z(bars)
    panel = pd.DataFrame(
        {
            "market_clv": pooled_clv,
            "outcome_return": outcome,
            "weak_breadth": weak_breadth,
            "high_open_z": rally_z,
        },
        index=bars["open"].index,
    ).join(l1, how="left")
    panel = panel.loc[dates]
    required = [
        "market_clv", "outcome_return", *FEATURES, "l1_imbalance",
        "l1_spread_ratio", "l1_depth_ratio",
    ]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    if panel[required].isna().any().any():
        bad = panel[required].isna().any(axis=1)
        raise ValueError(f"弱收盘机制分类特征或结果缺失：{panel.index[bad][:5].tolist()}")
    return panel


def _fit_logistic(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    if len(train_y) < 30 or np.unique(train_y).size < 2:
        raise ValueError("弱收盘机制分类训练样本不足或只有一个结果类别")
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_z = (train_x - mean) / scale
    test_z = (test_x - mean) / scale
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
        raise RuntimeError(f"弱收盘机制分类 logistic 未收敛：{fitted.message}")
    return expit(test_design @ fitted.x)


def _metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=int)
    probability = np.asarray(probability, dtype=float)
    positives = int(actual.sum())
    negatives = len(actual) - positives
    if not positives or not negatives:
        auc = balanced_accuracy = np.nan
    else:
        ranks = rankdata(probability, method="average")
        auc = float((ranks[actual == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
        predicted = probability >= 0.5
        sensitivity = float(predicted[actual == 1].mean())
        specificity = float((~predicted[actual == 0]).mean())
        balanced_accuracy = (sensitivity + specificity) / 2
    return {
        "AUC": auc,
        "Brier": float(np.mean((probability - actual) ** 2)),
        "平衡准确率": balanced_accuracy,
        "反转率": float(actual.mean()),
        "样本数": int(len(actual)),
    }


def walk_forward(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = panel.loc[panel["market_clv"] < WEAK_THRESHOLD].copy()
    usable["label"] = (usable["outcome_return"] > 0).astype(int)
    rows, prediction_rows = [], []
    for year in EVALUATION_YEARS:
        train = usable.loc[usable.index.year < year]
        test = usable.loc[usable.index.year == year]
        if train.empty or test.empty:
            raise ValueError(f"{year} 年弱收盘机制分类训练集或预测集为空")
        for model, extra in MODELS.items():
            columns = ["market_clv", *extra]
            probability = _fit_logistic(
                train[columns].to_numpy(dtype=float),
                train["label"].to_numpy(dtype=int),
                test[columns].to_numpy(dtype=float),
            )
            actual = test["label"].to_numpy(dtype=int)
            rows.append({"年份": year, "模型": model, **_metrics(actual, probability)})
            prediction_rows.extend(
                {
                    "日期": date,
                    "年份": year,
                    "模型": model,
                    "实际标签": int(label),
                    "预测概率": float(prob),
                    "下一开盘段收益_bp": float(ret * 1e4),
                }
                for date, label, prob, ret in zip(
                    test.index, actual, probability, test["outcome_return"].to_numpy(dtype=float)
                )
            )

    predictions = pd.DataFrame(prediction_rows)
    summary_rows = []
    for model, group in predictions.groupby("模型", sort=False):
        summary_rows.append(
            {"年份": "合并", "模型": model, **_metrics(group["实际标签"], group["预测概率"])}
        )
    fold = pd.DataFrame(rows)
    summary = pd.concat([fold, pd.DataFrame(summary_rows)], ignore_index=True)

    comparison = []
    for year, group in summary.groupby("年份", sort=False):
        baseline = group.loc[group["模型"] == "CLV基线"].iloc[0]
        joint = group.loc[group["模型"] == "CLV+全部信号块"].iloc[0]
        comparison.append(
            {
                "年份": year,
                "基线AUC": baseline["AUC"],
                "联合AUC": joint["AUC"],
                "AUC差": joint["AUC"] - baseline["AUC"],
                "基线Brier": baseline["Brier"],
                "联合Brier": joint["Brier"],
                "Brier改善": baseline["Brier"] - joint["Brier"],
                "基线平衡准确率": baseline["平衡准确率"],
                "联合平衡准确率": joint["平衡准确率"],
            }
        )
    return summary, predictions, pd.DataFrame(comparison)


def _weak_sample_regressions(panel: pd.DataFrame) -> pd.DataFrame:
    weak = panel.loc[panel["market_clv"] < WEAK_THRESHOLD]
    rows = []
    for feature in FEATURES:
        frame = weak[["outcome_return", "market_clv", feature]].dropna()
        coefficient, t_stat = ols_hac(
            frame["outcome_return"].to_numpy(),
            [frame[feature].to_numpy(), frame["market_clv"].to_numpy()],
            HAC_LAGS,
        )
        rows.append(
            {
                "信号块": feature,
                "条件收益系数_bp": coefficient * 1e4,
                "NW_t": t_stat,
                "样本数": len(frame),
                "解释": "控制连续 pooled CLV 后的条件相关；不作因果解释",
            }
        )
    return pd.DataFrame(rows)


def run(*, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = load_protocol()
    if (OUT / "classification_metrics.csv").exists() and not overwrite:
        raise FileExistsError(f"弱收盘机制分类结果已存在；使用 --overwrite 刷新：{OUT}")
    C.assert_research_only(C.INDEX_RESEARCH_DIR)
    _, bars, _, _, _ = _load_index_research_inputs()
    l1_dates = bars["open"].index[
        (bars["open"].index >= SCORE_START) & (bars["open"].index <= SCORE_END)
    ]
    excluded_dates = pd.DatetimeIndex(
        pd.to_datetime([item["date"] for item in protocol["coverage_exclusions"]])
    )
    l1 = _load_l1_daily(l1_dates, excluded_dates)
    panel = build_panel(bars, l1, excluded_dates)
    metrics, predictions, comparison = walk_forward(panel)
    pooled = comparison.loc[comparison["年份"] == "合并"].iloc[0]
    yearly = comparison.loc[comparison["年份"] != "合并"]
    improved_years = int(yearly["Brier改善"].gt(0).sum())
    passed = bool(
        pooled["AUC差"] > 0
        and pooled["Brier改善"] > 0
        and improved_years >= 3
    )
    decision = pd.DataFrame(
        [
            {
                "联合模型合并AUC提高": bool(pooled["AUC差"] > 0),
                "联合模型合并Brier降低": bool(pooled["Brier改善"] > 0),
                "Brier改善年度数": improved_years,
                "要求改善年度数": 3,
                "是否符合探索性门槛": passed,
                "结论": "存在待验证的区分信号" if passed else "未达到预登记门槛",
            }
        ]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / "classification_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "joint_vs_baseline.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(OUT / "decision.csv", index=False, encoding="utf-8-sig")
    _weak_sample_regressions(panel).to_csv(
        OUT / "conditional_signal_associations.csv", index=False, encoding="utf-8-sig"
    )
    panel.to_csv(OUT / "daily_panel.csv", encoding="utf-8-sig", index_label="trading_date")
    pd.DataFrame(protocol["coverage_exclusions"]).to_csv(
        OUT / "coverage_exclusions.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.round(4).to_string(index=False), flush=True)
    print(f"弱收盘机制分类结果：{OUT}", flush=True)
    return metrics, predictions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="弱收盘流动性压力与信息冲击代理分类")
    parser.add_argument("--overwrite", action="store_true", help="覆盖同一预登记协议的输出")
    args = parser.parse_args(argv)
    run(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
