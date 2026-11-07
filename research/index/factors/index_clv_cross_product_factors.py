"""三品种 CLV 与时间因子异质性诊断；只读研究期。

    python -m research.index factor cross-product

规格在 config/research/index_clv_cross_product_factors.json。只做信号层面分析。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from research.index.factors.index_time_factor_increment import (
    COLUMNS,
    FACTORS,
    SYMBOLS,
    regression,
    symbol_panel,
)
from research.time_factor_screen import ols_hac
from strategies.index_common import THRESHOLD
from tfcta import config as C
from tfcta.data import daily_bars, load_shard

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_clv_cross_product_factors.json"
OUT = C.study_dir("index", "index_clv_cross_product_factors")
START, END = pd.Timestamp("2015-04-16"), pd.Timestamp("2021-12-29")
FACTOR_START = pd.Timestamp("2016-01-04")
HAC_LAGS, MIN_YEAR_DAYS = 5, 60


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "declared_research_not_a_strategy"
        or "不读 2022–2025" not in protocol.get("sample", "")
        or "任何 oos/" not in protocol.get("sample", "")
        or "Holm" not in protocol.get("studies", {})
        .get("clv_cross_product", {})
        .get("primary", "")
        or len(FACTORS) != 3
    ):
        raise ValueError("CLV 与因子异质性研究登记与实现不一致")
    return protocol


def p_value(t: float) -> float:
    return float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else np.nan


def fit(frame: pd.DataFrame, factor: str) -> dict:
    data = frame[["y", factor]].dropna()
    coef, t = ols_hac(data["y"].to_numpy(), [data[factor].to_numpy()], HAC_LAGS)
    return {
        "系数": coef,
        "t": t,
        "p": p_value(t),
        "bp_per_sd": coef * float(data[factor].std()) * 1e4 if len(data) else np.nan,
        "天数": len(data),
    }


def holm(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * values[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def load_panels() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    clv_by_symbol: dict[str, pd.Series] = {}
    for symbol in SYMBOLS:
        minute = load_shard(
            symbol, directory=C.INDEX_RESEARCH_DIR, columns=COLUMNS
        )
        dates = pd.to_datetime(minute["trading_date"]).dt.normalize()
        C.assert_no_holdout_dates(dates, f"{symbol} CLV 因子研究")
        daily = daily_bars(minute)
        width = daily["highw"] - daily["loww"]
        clv = (daily["closew"] - daily["loww"]) / width.where(width > 0)
        if clv.isna().any() or not clv.between(0, 1).all():
            raise ValueError(f"{symbol} 日线 CLV 缺失或越界")
        clv_by_symbol[symbol] = clv.rename(symbol)
        panels[symbol] = symbol_panel(minute)
        print(f"{symbol} 日线 CLV 与时间因子完成", flush=True)

    common = clv_by_symbol[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        common = common.intersection(clv_by_symbol[symbol].index)
    common = common[(common >= START) & (common <= END)]
    if common.empty:
        raise ValueError("IF/IH/IC 没有共同研究日期")
    clv = pd.DataFrame(
        {symbol: clv_by_symbol[symbol].reindex(common) for symbol in SYMBOLS},
        index=common,
    )
    if clv.isna().any().any():
        raise ValueError("三品种 CLV 行情没有完全对齐")
    C.assert_no_holdout_dates(common, "CLV 共同信号日")
    return panels, clv


def clv_study(
    panels: dict[str, pd.DataFrame], clv: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = clv.index
    returns = pd.DataFrame(
        {
            symbol: panels[symbol]["y"].reindex(common)
            for symbol in SYMBOLS
        },
        index=common,
    )
    if returns.isna().any().any():
        raise ValueError("三品种下一开盘收益没有完全对齐")
    market_clv = clv.mean(axis=1)
    market_return = returns.mean(axis=1)
    signals = clv.lt(THRESHOLD)
    rows = []
    for name, signal, y in [
        *((symbol, signals[symbol], returns[symbol]) for symbol in SYMBOLS),
        ("三品种等权", market_clv.lt(THRESHOLD), market_return),
    ]:
        data = pd.DataFrame({"y": y, "弱收盘": signal.astype(float)}).dropna()
        result = fit(data, "弱收盘")
        weak_mean = float(data.loc[data["弱收盘"].eq(1), "y"].mean())
        other_mean = float(data.loc[data["弱收盘"].eq(0), "y"].mean())
        rows.append({
            "信号": name,
            "检验": "弱收盘相对非弱收盘",
            "弱收盘天数": int(data["弱收盘"].sum()),
            "弱收盘占比": float(data["弱收盘"].mean()),
            "弱收盘均值_bp": weak_mean * 1e4,
            "其他日均值_bp": other_mean * 1e4,
            **result,
        })

    for name, x, y in [
        *((symbol, clv[symbol], returns[symbol]) for symbol in SYMBOLS),
        ("三品种等权", market_clv, market_return),
    ]:
        frame = pd.DataFrame({"y": y, "CLV": x})
        rows.append({
            "信号": name,
            "检验": "连续 CLV 斜率",
            "弱收盘天数": np.nan,
            "弱收盘占比": np.nan,
            "弱收盘均值_bp": np.nan,
            "其他日均值_bp": np.nan,
            **fit(frame, "CLV"),
        })

    table = pd.DataFrame(rows)
    table["Holm_p_8项"] = holm(table["p"].tolist())
    table["判定"] = np.where(
        table["Holm_p_8项"].lt(0.05), "校正后显著", "校正后不显著"
    )

    correlation_rows = []
    for metric, frame in (
        ("CLV", clv),
        ("下一开盘收益", returns),
        ("弱收盘指示", signals.astype(float)),
    ):
        for i, left in enumerate(SYMBOLS):
            for right in SYMBOLS[i + 1:]:
                pair = frame[[left, right]].dropna()
                correlation_rows.append({
                    "指标": metric,
                    "品种A": left,
                    "品种B": right,
                    "Pearson": pair[left].corr(pair[right], method="pearson"),
                    "Spearman": pair[left].corr(pair[right], method="spearman"),
                    "共同天数": len(pair),
                })
    correlations = pd.DataFrame(correlation_rows)
    n = len(signals)
    market_signal = market_clv.lt(THRESHOLD)
    two_or_more = signals.sum(axis=1).ge(2)
    market_weak_days = int(market_signal.sum())
    agreement = pd.DataFrame([{
        "共同天数": n,
        "三品种全弱收盘_pct": float(signals.all(axis=1).mean() * 100),
        "三品种全非弱收盘_pct": float((~signals).all(axis=1).mean() * 100),
        "信号分歧_pct": float((signals.nunique(axis=1) > 1).mean() * 100),
        "至少两品种弱收盘_pct": float(two_or_more.mean() * 100),
        "市场弱收盘_pct": float(market_signal.mean() * 100),
        "市场弱收盘日中至少两品种也弱_pct": (
            float((market_signal & two_or_more).sum() / market_weak_days * 100)
            if market_weak_days else np.nan
        ),
    }])
    return table, correlations, agreement


def yearly_consistency(frame: pd.DataFrame, factor: str, sign: float) -> str:
    years = []
    for _, part in frame.groupby(frame.index.year):
        data = part[["y", factor, "ctrl_return", "ctrl_location", "ctrl_log_volume"]].dropna()
        if len(data) >= MIN_YEAR_DAYS:
            coef = regression(part, factor)["系数"]
            if np.isfinite(coef):
                years.append(np.sign(coef) == sign)
    return f"{sum(years)}/{len(years)}"


def factor_study(
    panels: dict[str, pd.DataFrame], clv: pd.DataFrame
) -> pd.DataFrame:
    market_weak = clv.mean(axis=1).lt(THRESHOLD)
    mask = (market_weak.index >= FACTOR_START) & market_weak.to_numpy()
    index = market_weak.index[mask]
    rows = []
    for symbol in SYMBOLS:
        panel = panels[symbol].reindex(index)
        for factor in FACTORS:
            result = regression(panel, factor)
            sign = np.sign(result["系数"]) if np.isfinite(result["系数"]) else np.nan
            rows.append({
                "品种": symbol,
                "因子": factor,
                **result,
                "逐年同号": yearly_consistency(panel, factor, sign),
            })
    table = pd.DataFrame(rows)
    table["Holm_p_9项"] = holm(table["p"].tolist())
    expected = {"ts_low_pm": 1, "dfp_top3": 1, "ts_high": 0}
    table["先验方向符合"] = [
        "不预设" if expected[row["因子"]] == 0
        else (
            "符合"
            if np.sign(row["系数"]) == expected[row["因子"]]
            else "不符合"
        )
        for _, row in table.iterrows()
    ]
    table["探索性线索"] = table.apply(
        lambda row: bool(
            np.isfinite(row["Holm_p_9项"])
            and row["Holm_p_9项"] < 0.05
            and row["先验方向符合"] != "不符合"
            and int(row["逐年同号"].split("/")[0]) >= 5
            and row["逐年同号"].split("/")[1] == "6"
        ),
        axis=1,
    )
    return table


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = load_protocol()
    C.assert_research_only(C.INDEX_RESEARCH_DIR)
    panels, clv = load_panels()
    clv_results, correlations, agreement = clv_study(panels, clv)
    factor_results = factor_study(panels, clv)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    clv_results.to_csv(OUT / "clv_tests.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(OUT / "correlations.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(OUT / "signal_agreement.csv", index=False, encoding="utf-8-sig")
    factor_results.to_csv(
        OUT / "factor_by_product.csv", index=False, encoding="utf-8-sig"
    )
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\nCLV 检验", flush=True)
        print(clv_results.round(4).to_string(index=False), flush=True)
        print("\n品种间相关", flush=True)
        print(correlations.round(4).to_string(index=False), flush=True)
        print("\n弱收盘一致性", flush=True)
        print(agreement.round(4).to_string(index=False), flush=True)
        print("\n弱收盘日因子-品种检验", flush=True)
        print(factor_results.round(4).to_string(index=False), flush=True)
    print(f"结果：{OUT}", flush=True)
    return clv_results, factor_results


if __name__ == "__main__":
    run()
