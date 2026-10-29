"""用 QVIX 重做「波动越高、收盘位置反转越强」。

    python -m research.index_qvix_reversal

规则在 config/research/index_qvix_reversal.json。只读研究期 QVIX
（download.index.getQvix 写下的 qvix_50etf.csv）和股指研究期分钟分片。
不读 holdout_locked，不读 2022 及以后。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.time_factor_screen import ols_hac
from strategies.index_close_location import market_close_location
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_qvix_reversal.json"
OUT = C.RUNS_DIR / "index" / "index_qvix_reversal"
SYMBOLS = ("IF", "IH", "IC")
LAGS = 5


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    parameters = protocol.get("parameters") or {}
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022" not in sample
        or "不读 2026" not in sample
        or list(protocol.get("predictions", {})) != ["Q1", "Q2", "Q3"]
        or parameters.get("primary_start") != "2016-01-04"
        or parameters.get("full_start") != "2015-04-16"
        or parameters.get("outcome_end") != "2021-12-31"
        or parameters.get("hac_lags") != LAGS
        or parameters.get("windows") != [1, 5, 20, 60]
        or parameters.get("overlap_min_corr") != 0.99
    ):
        raise ValueError("QVIX 反转检验登记与实现或样本边界不一致")
    return protocol


def _fit(y: pd.Series, xs: list[pd.Series]) -> tuple[float, float, int]:
    frame = pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 60:
        return np.nan, np.nan, len(frame)
    coef, tstat = ols_hac(
        frame.iloc[:, 0].to_numpy(),
        [frame.iloc[:, i].to_numpy() for i in range(1, frame.shape[1])],
        LAGS,
    )
    return coef, tstat, len(frame)


def _verdict(coef: float, tstat: float, expected: str) -> str:
    if pd.isna(tstat):
        return "不确定"
    right = coef < 0 if expected == "<0" else coef > 0
    if abs(tstat) >= 2:
        return "支持" if right else "否定"
    return "不确定"


def _yearly(y: pd.Series, xs: list[pd.Series]) -> str:
    signs = []
    for year in sorted(set(y.dropna().index.year)):
        mask = y.index.year == year
        coef = _fit(y[mask], [x[mask] for x in xs])[0]
        signs.append(f"{year}:{'+' if coef > 0 else '−' if coef < 0 else '?'}")
    return " ".join(signs)


def futures_panel() -> pd.DataFrame:
    """信号日收盘位置，以及下一交易日开盘到收盘的等权收益。"""
    from tfcta.data import load_daily_bars

    bars = load_daily_bars(list(SYMBOLS), directory=C.INDEX_RESEARCH_DIR)
    width = bars["highw"] - bars["loww"]
    ok = (width.gt(0) & bars["closew"].notna() & bars["open"].gt(0)).all(axis=1)
    index = ok.index[ok]
    index = index[(index >= pd.Timestamp("2015-04-16")) & (index <= pd.Timestamp("2021-12-31"))]
    clv = market_close_location(bars, index)
    intra = ((bars["closew"] - bars["openw"]) / bars["open"]).mean(axis=1, skipna=False)
    panel = pd.DataFrame({"clv": clv, "next_intraday": intra.reindex(index).shift(-1)})
    C.assert_no_holdout_dates(panel.index, "QVIX 反转检验日期")
    return panel.dropna(subset=["clv"])


def attach_qvix(panel: pd.DataFrame, qvix: pd.DataFrame) -> pd.DataFrame:
    """信号日用当天 QVIX 收盘。下一交易日的 QVIX 不进入这一行。"""
    level = qvix.set_index("date")["close"].sort_index()
    C.assert_no_holdout_dates(level.index, "QVIX 研究期")
    out = panel.copy()
    out["qvix"] = level.reindex(out.index)
    return out


def vol_on_signal_day(level: pd.Series, window: int) -> pd.Series:
    """水平，或截至信号日的均值。滚动窗口含当天、不含未来。"""
    if window == 1:
        return level
    return level.rolling(window, min_periods=window).mean()


def assert_matches_official(qvix: pd.DataFrame, minimum_corr: float, low: float, high: float) -> dict:
    """研究期内和官方中国波指重叠的日子必须对得上。"""
    from tfcta.rq_auth import rqdata_credentials
    import rqdatac as rq

    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证，无法核对官方中国波指")
    rq.init(*credentials)
    official = rq.get_price(
        "000188.XSHG", start_date="2016-11-28", end_date="2018-02-14", frequency="1d", fields=["close"],
    )
    if official is None or official.empty:
        raise RuntimeError("官方中国波指没有返回研究期价格")
    official = official.reset_index()
    official["date"] = pd.to_datetime(official["date"]).dt.normalize()
    C.assert_no_holdout_dates(official["date"], "官方中国波指")
    merged = qvix.merge(official[["date", "close"]], on="date", suffixes=("_qvix", "_official"))
    if len(merged) < 60:
        raise ValueError(f"QVIX 与官方中国波指只重叠 {len(merged)} 天")
    corr = float(merged["close_qvix"].corr(merged["close_official"]))
    ratio = float((merged["close_qvix"] / merged["close_official"]).median())
    if corr < minimum_corr or not low <= ratio <= high:
        raise ValueError(f"QVIX 与官方中国波指不一致：相关 {corr:.4f}，中位比 {ratio:.4f}")
    return {"重叠天数": len(merged), "相关": corr, "中位比": ratio}


def tests(panel: pd.DataFrame, windows: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = panel["next_intraday"]
    centered = panel["clv"] - panel["clv"].mean()
    rows = []
    for window in windows:
        vol = vol_on_signal_day(panel["qvix"], window)
        z = (vol - vol.mean()) / vol.std()
        label = "QVIX水平" if window == 1 else f"QVIX{window}日均值"
        specs = (
            ("clv", [panel["clv"], z, centered * z], ""),
            (label, [z, panel["clv"], centered * z], ""),
            (f"clv×{label}", [centered * z, panel["clv"], z], "<0"),
        )
        for name, xs, expected in specs:
            coef, tstat, nobs = _fit(y, xs)
            rows.append({
                "检验": "Q1", "变量": name, "系数_bp": coef * 1e4, "NW_t": tstat, "天数": nobs,
                "预期": expected or "—", "判断": _verdict(coef, tstat, expected) if expected else "—",
                "逐年符号": _yearly(y, xs) if expected else "",
            })
    level = panel["qvix"]
    past = level.shift(1)
    edges = [past.rolling(250, min_periods=120).quantile(q) for q in (0.2, 0.4, 0.6, 0.8)]
    valid = edges[0].notna() & level.notna()
    bucket = pd.Series(np.nan, index=panel.index)
    bucket.loc[valid] = 1 + sum((level.loc[valid] > edge.loc[valid]).astype(int) for edge in edges)
    bins = []
    slopes = []
    for k in range(1, 6):
        mask = bucket == k
        coef, tstat, nobs = _fit(y[mask], [panel.loc[mask, "clv"]])
        bins.append({
            "档": k, "天数": nobs, "QVIX均值": float(level[mask].mean()) if mask.any() else np.nan,
            "clv斜率_bp": coef * 1e4, "NW_t": tstat,
        })
        slopes.append(coef)
    rho = float(spearmanr(range(1, 6), slopes).correlation) if np.isfinite(slopes).all() else np.nan
    rows.append({
        "检验": "Q2", "变量": "档位与斜率秩相关", "系数_bp": rho, "NW_t": np.nan,
        "天数": int(bucket.notna().sum()), "预期": "<0", "判断": "见数值", "逐年符号": "",
    })
    gross = (0.5 - panel["clv"]) * y
    z_level = (level - level.mean()) / level.std()
    coef, tstat, nobs = _fit(gross, [z_level])
    rows.append({
        "检验": "Q3", "变量": "QVIX水平", "系数_bp": coef * 1e4, "NW_t": tstat, "天数": nobs,
        "预期": ">0", "判断": _verdict(coef, tstat, ">0"), "逐年符号": _yearly(gross, [z_level]),
    })
    return pd.DataFrame(rows), pd.DataFrame(bins)


def run() -> None:
    from download.index.getQvix import load_research_qvix

    protocol = load_protocol()
    parameters = protocol["parameters"]
    qvix = load_research_qvix()
    gate = assert_matches_official(
        qvix, parameters["overlap_min_corr"], parameters["overlap_ratio_low"], parameters["overlap_ratio_high"],
    )
    panel = attach_qvix(futures_panel(), qvix)
    panel = panel[panel.index <= pd.Timestamp(parameters["outcome_end"])]
    tables, bins = [], []
    for label, start in (
        ("2016-01-04 起", parameters["primary_start"]),
        ("2015-04-16 起", parameters["full_start"]),
    ):
        part = panel[panel.index >= pd.Timestamp(start)]
        table, bucket = tests(part, parameters["windows"])
        tables.append(table.assign(样本=label))
        bins.append(bucket.assign(样本=label))
    table = pd.concat(tables, ignore_index=True)
    buckets = pd.concat(bins, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "tests.csv", index=False, encoding="utf-8-sig")
    buckets.to_csv(OUT / "buckets.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(OUT / "daily_panel.csv", encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(
        json.dumps({"protocol": protocol, "官方核对": gate}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 80)
    print(table.round(3).to_string(index=False), flush=True)
    print(buckets.round(3).to_string(index=False), flush=True)
    print(json.dumps(gate, ensure_ascii=False), flush=True)
    print(f"结果：{OUT}", flush=True)


if __name__ == "__main__":
    run()
