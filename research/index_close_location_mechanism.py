"""股指收盘位置与 L1 盘口机制诊断；--shape 检验相对位置与绝对幅度、以及几个可检验的经济渠道。

    python -m research.index_close_location_mechanism            # L1 盘口
    python -m research.index_close_location_mechanism --shape    # 登记在 index_close_location_shape.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import newey_west_t
from research.time_factor_screen import ols_hac
from tfcta import config as C
from tfcta.data import load_daily_bars

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_close_location_mechanism.json"
OUT = C.RUNS_DIR / "index" / "index_close_location_mechanism"
L1_DIR = C.INDEX_DATA_ROOT / "external_rqdata" / "l1_close"
SYMBOLS = ("IF", "IH", "IC")
SCORE_START = pd.Timestamp("2016-01-13")
SCORE_END = pd.Timestamp("2021-12-31")
HAC_LAGS = 5


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    parameters = protocol.get("parameters", {})
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "Do not read 2022" not in sample
        or "or 2026" not in sample
        or parameters.get("instruments") != list(SYMBOLS)
        or parameters.get("score_start") != str(SCORE_START.date())
        or parameters.get("score_end") != str(SCORE_END.date())
        or parameters.get("hac_lags") != HAC_LAGS
    ):
        raise ValueError("股指机制检验登记与样本边界或实现不一致")
    return protocol


def build_daily_panel(bars: dict[str, pd.DataFrame], l1: pd.DataFrame) -> pd.DataFrame:
    """对齐三个品种的 L1 汇总和次日收益，按固定等权口径形成日序列。"""
    dates = bars["open"].index
    dates = dates[(dates >= SCORE_START) & (dates <= SCORE_END)]
    if dates.empty:
        raise ValueError("股指机制检验没有有效研究日期")
    C.assert_no_holdout_dates(dates, what="股指机制检验日期")

    l1 = l1.copy()
    l1["trading_date"] = pd.to_datetime(l1["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(l1["trading_date"], what="股指 L1 机制检验汇总")
    if l1.duplicated(["trading_date", "symbol"]).any():
        raise ValueError("股指 L1 汇总存在重复的品种-日期")
    indexed = l1.set_index(["trading_date", "symbol"]).sort_index()
    required = (
        "clv_last", "clv_mid", "spread_day", "spread_last30",
        "depth_day", "depth_last30", "imb_last30", "n_ticks",
    )
    missing = sorted(set(required) - set(indexed.columns))
    if missing:
        raise KeyError(f"股指 L1 汇总缺少字段：{missing}")
    expected_rows = pd.MultiIndex.from_product(
        [dates, SYMBOLS], names=["trading_date", "symbol"]
    )
    missing_rows = expected_rows.difference(indexed.index)
    if len(missing_rows):
        raise ValueError(f"股指 L1 日汇总记录不完整：缺少 {len(missing_rows)} 个品种-日期")

    metrics = {}
    for field in required:
        wide = indexed[field].unstack("symbol").reindex(index=dates, columns=SYMBOLS)
        metrics[field] = wide.mean(axis=1, skipna=False)
    if metrics["n_ticks"].le(0).any():
        raise ValueError("股指 L1 汇总出现没有日盘 tick 的日期")

    open_, openw = bars["open"].reindex(dates), bars["openw"].reindex(dates)
    close, closew = bars["close"].reindex(dates), bars["closew"].reindex(dates)
    previous_close = bars["close"].shift(1).reindex(dates)
    previous_closew = bars["closew"].shift(1).reindex(dates)
    next_open = bars["open"].shift(-1).reindex(dates)
    next_openw = bars["openw"].shift(-1).reindex(dates)
    next_closew = bars["closew"].shift(-1).reindex(dates)
    next_next_openw = bars["openw"].shift(-2).reindex(dates)
    out = pd.DataFrame(metrics, index=dates)
    out["spread_ratio"] = out["spread_last30"] / out["spread_day"].where(out["spread_day"] > 0)
    out["depth_ratio"] = out["depth_last30"] / out["depth_day"].where(out["depth_day"] > 0)
    out["same_day_return"] = (
        (closew - previous_closew) / previous_close.where(previous_close > 0)
    ).mean(axis=1)
    out["day_range"] = (
        (bars["highw"].reindex(dates) - bars["loww"].reindex(dates))
        / previous_close.where(previous_close > 0)
    ).mean(axis=1)
    out["next_gap"] = ((next_openw - closew) / close.where(close > 0)).mean(axis=1)
    out["next_intraday"] = (
        (next_closew - next_openw) / next_open.where(next_open > 0)
    ).mean(axis=1)
    out["next_open_to_open"] = (
        (next_next_openw - next_openw) / next_open.where(next_open > 0)
    ).mean(axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def _fit_row(label: str, outcome: pd.Series, predictors: list[tuple[str, pd.Series]]) -> dict:
    frame = pd.concat([outcome.rename("y"), *[x.rename(name) for name, x in predictors]], axis=1).dropna()
    if frame.empty:
        return {"检验": label, "样本数": 0, "系数": np.nan, "NW_t": np.nan}
    coefficient, t_stat = ols_hac(
        frame["y"].to_numpy(),
        [frame[name].to_numpy() for name, _ in predictors],
        HAC_LAGS,
    )
    return {"检验": label, "样本数": len(frame), "系数": coefficient, "NW_t": t_stat}


def analyze(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行登记的收盘位置、买卖价反弹与尾盘流动性诊断。"""
    records = []
    outcomes = {
        "次日盘中": "next_intraday",
        "次日开盘跳空": "next_gap",
        "次日开盘至再下一开盘": "next_open_to_open",
    }
    for label, outcome in outcomes.items():
        for clv_name, signal in (("成交价收盘位置", "clv_last"), ("中间价收盘位置", "clv_mid")):
            records.append(_fit_row(f"{label} ~ {clv_name}", panel[outcome], [(clv_name, panel[signal])]))
        records.append(
            _fit_row(
                f"{label} ~ 成交价收盘位置 + 当日收益",
                panel[outcome],
                [("clv_last", panel["clv_last"]), ("same_day_return", panel["same_day_return"])],
            )
        )
    for label, outcome in (
        ("当日收盘到收盘", "same_day_return"),
        ("次日盘中", "next_intraday"),
        ("次日开盘至再下一开盘", "next_open_to_open"),
    ):
        records.append(
            _fit_row(
                f"{label} ~ 当日振幅",
                panel[outcome],
                [("day_range", panel["day_range"])],
            )
        )
    for label, outcome in (("次日盘中", "next_intraday"), ("次日开盘至再下一开盘", "next_open_to_open")):
        records.append(_fit_row(f"{label} ~ 尾盘 tick-rule 成交不平衡", panel[outcome], [("imb_last30", panel["imb_last30"])]))
        imbalance_interaction = panel["clv_last"] * panel["imb_last30"]
        records.append(
            _fit_row(
                f"{label} ~ 收盘位置 × 尾盘成交不平衡（描述）",
                panel[outcome],
                [("interaction", imbalance_interaction), ("clv_last", panel["clv_last"]),
                 ("imb_last30", panel["imb_last30"])],
            )
        )

    interaction_terms = (
        ("尾盘价差", panel["spread_ratio"], -1),
        ("尾盘深度变薄", -panel["depth_ratio"], -1),
    )
    for label, outcome in (("次日盘中", "next_intraday"), ("次日开盘至再下一开盘", "next_open_to_open")):
        for name, liquidity, expected in interaction_terms:
            z = (liquidity - liquidity.mean()) / liquidity.std()
            interaction = panel["clv_last"] * z
            records.append(
                _fit_row(
                    f"{label} ~ 收盘位置 × {name}",
                    panel[outcome],
                    [("interaction", interaction), ("clv_last", panel["clv_last"]), ("liquidity", z)],
                )
                | {"预期交乘方向": "<0" if expected < 0 else ">0"}
            )

    median_close = panel["clv_last"].median()
    low_close = panel["clv_last"].le(median_close).astype(float)
    state_rows = []
    for name, series, expected in (
        ("尾盘/全天相对价差", panel["spread_ratio"], ">0"),
        ("尾盘/全天相对深度", panel["depth_ratio"], "<0"),
    ):
        coefficient, t_stat = ols_hac(series.to_numpy(), [low_close.to_numpy()], HAC_LAGS)
        state_rows.append(
            {"状态比较": f"低收盘位置 − 高收盘位置：{name}", "差值": coefficient, "NW_t": t_stat, "预期": expected}
        )

    quintile = panel[["clv_last", "next_intraday"]].dropna().copy()
    quintile["组"] = pd.qcut(quintile["clv_last"], 5, labels=False, duplicates="drop") + 1
    quintile_rows = []
    for group, part in quintile.groupby("组", observed=True):
        avg = float(part["next_intraday"].mean())
        t_stat = newey_west_t(part["next_intraday"], HAC_LAGS)
        quintile_rows.append(
            {"收盘位置五分组": int(group), "平均收盘位置": part["clv_last"].mean(),
             "天数": len(part), "次日盘中收益": avg, "NW_t": t_stat}
        )

    return (
        pd.DataFrame(records),
        pd.DataFrame(state_rows),
        pd.DataFrame(quintile_rows),
    )


def run(*, overwrite: bool = False) -> None:
    protocol = load_protocol()
    if (OUT / "summary.csv").exists() and not overwrite:
        raise FileExistsError(f"股指机制结果已存在；用 --overwrite 刷新：{OUT}")
    bars = load_daily_bars(list(SYMBOLS), directory=C.INDEX_RESEARCH_DIR)
    l1_frames = []
    for symbol in SYMBOLS:
        path = L1_DIR / f"{symbol}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {symbol} L1 日汇总：{path}；先运行下载器")
        C.assert_research_only(path)
        frame = pd.read_csv(path, parse_dates=["trading_date"])
        C.assert_no_holdout_dates(frame["trading_date"], f"{symbol} 股指 L1 汇总")
        frame["symbol"] = symbol
        l1_frames.append(frame)
    l1 = pd.concat(l1_frames, ignore_index=True)
    panel = build_daily_panel(bars, l1)
    tests, states, quintiles = analyze(panel)

    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "daily.csv", encoding="utf-8-sig", index_label="trading_date")
    tests.to_csv(OUT / "summary.csv", encoding="utf-8-sig", index=False)
    states.to_csv(OUT / "liquidity_states.csv", encoding="utf-8-sig", index=False)
    quintiles.to_csv(OUT / "close_location_quintiles.csv", encoding="utf-8-sig", index=False)
    (OUT / "protocol.json").write_text(
        json.dumps(
            {"protocol": protocol, "scored_start": str(panel.index.min().date()),
             "scored_end": str(panel.index.max().date()), "observations": len(panel)},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"股指 L1 机制诊断：{panel.index.min().date()}..{panel.index.max().date()}，{len(panel)} 天")
    print(tests.to_string(index=False))
    print(states.to_string(index=False))


# --------------------------------------------------------------------------
# 收盘位置的形状：相对位置还是绝对幅度（index_close_location_shape.json）
# --------------------------------------------------------------------------
SHAPE_PATH = C.CONFIG_DIR / "research" / "index_close_location_shape.json"
SHAPE_OUT = C.RUNS_DIR / "index" / "index_close_location_shape"
SHAPE_START = pd.Timestamp("2015-04-16")
SHAPE_SUB_START = pd.Timestamp("2016-01-04")
CRASH = (pd.Timestamp("2015-06-15"), pd.Timestamp("2016-02-29"))


def load_shape_protocol(path: Path = SHAPE_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022" not in sample or "不读 2026" not in sample
        or sorted(protocol.get("tests", {})) != ["C1", "C2", "C3", "C4", "M1", "M2", "M3"]
    ):
        raise ValueError("收盘位置形状检验登记与样本边界或实现不一致")
    return protocol


def trailing_above_median(series: pd.Series, window: int = 250, minimum: int = 120) -> pd.Series:
    """高于自身过去 window 日（不含当天）中位数为 1，否则 0；历史不足为缺失。"""
    median = series.shift(1).rolling(window, min_periods=minimum).median()
    return (series > median).astype(float).where(median.notna() & series.notna())


def trailing_tercile(series: pd.Series, window: int = 250, minimum: int = 120) -> pd.Series:
    """当天值在过去 window 日（不含当天）里的三分位：0 低、1 中、2 高；历史不足为缺失。"""
    past = series.shift(1)
    low = past.rolling(window, min_periods=minimum).quantile(1 / 3)
    high = past.rolling(window, min_periods=minimum).quantile(2 / 3)
    out = pd.Series(np.where(series <= low, 0, np.where(series <= high, 1, 2)), index=series.index, dtype=float)
    return out.where(low.notna() & series.notna())


def shape_panel(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """逐品种的日形状变量和结果，收盘 t 已知；结果是次日盘中和账本口径下一段。"""
    dates = bars["open"].index
    dates = dates[(dates >= SHAPE_START) & (dates <= SCORE_END)]
    C.assert_no_holdout_dates(dates, what="股指形状检验日期")
    open_, openw, close, closew, high, low = (
        bars[k].reindex(index=dates, columns=list(SYMBOLS))
        for k in ("open", "openw", "close", "closew", "highw", "loww")
    )
    raw = close.where(close > 0)
    width = high - low
    out = {
        "clv": (closew - low) / width.where(width > 0),
        "d_low": (closew - low) / raw,
        "d_high": (high - closew) / raw,
        "range": width / raw,
        "same_day": (closew - openw) / open_.where(open_ > 0),
        "next_intraday": ((closew - openw) / open_.where(open_ > 0)).shift(-1),
        "next_open_to_open": ((openw.shift(-1) - openw) / open_.where(open_ > 0)).shift(-1),
    }
    scale = out["range"].shift(1).rolling(20, min_periods=20).mean()
    out["d_low_scaled"] = out["d_low"] / scale
    out["d_high_scaled"] = out["d_high"] / scale
    out["vol20"] = scale
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


def shape_market(per: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """三品种等权的市场层面形状变量，加状态哑变量（都只用收盘 t 及以前的信息）。"""
    frame = pd.DataFrame({k: v.mean(axis=1, skipna=False) for k, v in per.items()})
    frame["high_vol"] = trailing_above_median(frame["vol20"])
    following = pd.Series(list(frame.index[1:]) + [pd.NaT], index=frame.index)
    gap_days = (following - frame.index.to_series()).dt.days
    frame["weekend_or_holiday"] = ((frame.index.weekday == 4) | (gap_days > 3)).astype(float)
    return frame


def _fit_all(label: str, y: pd.Series, predictors: list[tuple[str, pd.Series]]) -> list[dict]:
    """每个解释变量轮流放第一列，得到全部系数的 HAC t；系数单位 bp。"""
    rows = []
    for k, (name, _) in enumerate(predictors):
        ordered = [predictors[k]] + predictors[:k] + predictors[k + 1:]
        row = _fit_row(label, y, ordered)
        rows.append({"检验": label, "变量": name, "系数_bp": row["系数"] * 1e4, "NW_t": row["NW_t"],
                     "样本数": row["样本数"]})
    return rows


def shape_tests(m: pd.DataFrame, per: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, groups = [], []
    x = lambda *names: [(n, m[n]) for n in names]                       # noqa: E731
    for outcome, oname in (("next_intraday", "次日盘中"), ("next_open_to_open", "账本下一段")):
        y = m[outcome]
        cases = (
            ("M1", "clv", x("clv")),
            ("M1", "d_low", x("d_low")),
            ("M1", "d_high", x("d_high")),
            ("M1", "d_low + d_high", x("d_low", "d_high")),
            ("M1", "clv + range", x("clv", "range")),
            ("M1", "clv + range + clv×range", x("clv", "range") + [("clv×range", m["clv"] * m["range"])]),
            ("M1", "clv + same_day", x("clv", "same_day")),
            ("M2", "d_low_scaled", x("d_low_scaled")),
            ("M2", "d_high_scaled", x("d_high_scaled")),
            ("M2", "d_low_scaled + d_high_scaled", x("d_low_scaled", "d_high_scaled")),
            ("M2", "clv + d_high_scaled", x("clv", "d_high_scaled")),
            ("C2", "clv + 周五或节前 + 交乘",
             x("clv", "weekend_or_holiday") + [("clv×周五或节前", m["clv"] * m["weekend_or_holiday"])]),
            ("C3", "clv + 高波动 + 交乘", x("clv", "high_vol") + [("clv×高波动", m["clv"] * m["high_vol"])]),
        )
        for test, label, preds in cases:
            rows += [{"预测": test, **row} for row in _fit_all(f"{oname} ~ {label}", y, preds)]
        tercile = trailing_tercile(m["range"])
        for k, name in enumerate(("低振幅", "中振幅", "高振幅")):
            mask = tercile == k
            rows += [{"预测": "M3", **row}
                     for row in _fit_all(f"{oname} ~ clv | {name}", y[mask], [("clv", m["clv"][mask])])]
        quintile = pd.qcut(m["clv"], 5, labels=False)
        for q in range(5):
            part = y[quintile == q].dropna()
            groups.append({"结果": oname, "clv五分位": q + 1, "均值_bp": part.mean() * 1e4,
                           "NW_t": newey_west_t(part, HAC_LAGS), "天数": len(part)})
        for symbol in SYMBOLS:
            rows += [{"预测": "C1", **row} for row in _fit_all(
                f"{oname} ~ 自身 clv | {symbol}", per[outcome][symbol], [("clv", per["clv"][symbol])])]
        # IC − IH：同一天两条结果相减，对两品种 clv 回归；IC clv 系数减 IH clv 系数的 t 用 IC clv 与 (IC+IH)/2 的重参数化
        diff_y = per[outcome]["IC"] - per[outcome]["IH"]
        ic, ih = per["clv"]["IC"], per["clv"]["IH"]
        rows += [{"预测": "C1", **row} for row in _fit_all(
            f"{oname}：IC 收益 − IH 收益 ~ IC clv − IH clv", diff_y, [("clv差", ic - ih)])]
        keep = ~((m.index >= CRASH[0]) & (m.index <= CRASH[1]))
        rows += [{"预测": "C4", **row} for row in _fit_all(
            f"{oname} ~ clv | 剔除 2015-06-15..2016-02-29", y[keep], [("clv", m["clv"][keep])])]
        for year in sorted(set(m.index.year)):
            sel = m.index.year == year
            rows += [{"预测": "C4", **row}
                     for row in _fit_all(f"{oname} ~ clv | {year}", y[sel], [("clv", m["clv"][sel])])]
    return pd.DataFrame(rows), pd.DataFrame(groups)


def run_shape() -> None:
    protocol = load_shape_protocol()
    bars = load_daily_bars(list(SYMBOLS), directory=C.INDEX_RESEARCH_DIR)
    per = shape_panel(bars)
    market = shape_market(per)
    tests, groups = [], []
    for label, start in (("2015-04-16 起", SHAPE_START), ("2016-01-04 起", SHAPE_SUB_START)):
        t, g = shape_tests(market[market.index >= start], {k: v[v.index >= start] for k, v in per.items()})
        tests.append(t.assign(样本=label))
        groups.append(g.assign(样本=label))
    tests, groups = pd.concat(tests, ignore_index=True), pd.concat(groups, ignore_index=True)
    SHAPE_OUT.mkdir(parents=True, exist_ok=True)
    tests.to_csv(SHAPE_OUT / "tests.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(SHAPE_OUT / "clv_quintiles.csv", index=False, encoding="utf-8-sig")
    market.to_csv(SHAPE_OUT / "daily_market.csv", encoding="utf-8-sig", index_label="trading_date")
    (SHAPE_OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(tests[["样本", "预测", "检验", "变量", "系数_bp", "NW_t", "样本数"]].round(2).to_string(index=False))
    print(groups.round(2).to_string(index=False))
    print(f"结果：{SHAPE_OUT}")


# --------------------------------------------------------------------------
# 经济渠道（index_reversal_channels.json）
# --------------------------------------------------------------------------
CHANNELS_PATH = C.CONFIG_DIR / "research" / "index_reversal_channels.json"
CHANNELS_OUT = C.RUNS_DIR / "index" / "index_reversal_channels"
FLOWS_PATH = C.INDEX_DATA_ROOT / "external_rqdata" / "market_flows.csv"
SPOT_DAILY_PATH = C.INDEX_DATA_ROOT / "external_rqdata" / "spot_daily.csv"
SSE_CLOSING_AUCTION = pd.Timestamp("2018-08-20")
CUTS = (1000, 1130, 1430)


def load_channels_protocol(path: Path = CHANNELS_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022" not in sample or "不读 2026" not in sample
        or sorted(protocol.get("tests", {})) != [f"T{k}" for k in range(1, 8)]
    ):
        raise ValueError("经济渠道检验登记与样本边界或实现不一致")
    return protocol


def minute_path(minute: pd.DataFrame) -> pd.DataFrame:
    """单品种期货分钟表 → 每日路径：各时点复权价、14:30 前的高低点、原始开盘价、5 分钟已实现波动。"""
    m = minute.sort_index(kind="mergesort")
    day = pd.to_datetime(m["trading_date"]).dt.normalize()
    clock = m.index.hour * 100 + m.index.minute
    grouped = m.groupby(day)
    out = pd.DataFrame({
        "openw": grouped["openw"].first(), "open": grouped["open"].first(),
        "closew": grouped["closew"].last(), "highw": grouped["highw"].max(), "loww": grouped["loww"].min(),
    })
    for cut in CUTS:
        out[f"c{cut}"] = m["closew"].where(clock <= cut).groupby(day).last()
    early = clock <= 1430
    out["hi1430"] = m["highw"].where(early).groupby(day).max()
    out["lo1430"] = m["loww"].where(early).groupby(day).min()
    five = (m.groupby(day).cumcount() % 5 == 4).to_numpy()
    sampled, sampled_day = m[five], day[five]
    step = sampled["closew"].groupby(sampled_day).diff() / sampled["close"]
    out["rv"] = np.sqrt((step ** 2).groupby(sampled_day).sum(min_count=2))
    out.index.name = "trading_date"
    return out


def channel_panel(paths: dict[str, pd.DataFrame], spot: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """三品种等权的市场层面变量；所有解释变量在收盘 t（或次日 9:25 集合竞价）已知。"""
    def wide(field):
        return pd.DataFrame({s: paths[s][field] for s in SYMBOLS})

    dates = wide("closew").dropna().index
    dates = dates[(dates >= SHAPE_START) & (dates <= SCORE_END)]
    C.assert_no_holdout_dates(dates, what="经济渠道检验日期")
    f = {k: wide(k).reindex(dates) for k in ("openw", "open", "closew", "highw", "loww", "c1000", "c1130",
                                            "c1430", "hi1430", "lo1430", "rv")}
    base = f["open"].where(f["open"] > 0)
    seg = lambda a, b: ((b - a) / base).mean(axis=1, skipna=False)          # noqa: E731
    p = pd.DataFrame(index=dates)
    width = f["highw"] - f["loww"]
    p["clv"] = ((f["closew"] - f["loww"]) / width.where(width > 0)).mean(axis=1, skipna=False)
    w1430 = f["hi1430"] - f["lo1430"]
    p["clv_1430"] = ((f["c1430"] - f["lo1430"]) / w1430.where(w1430 > 0)).mean(axis=1, skipna=False)
    p["late_move"] = p["clv"] - p["clv_1430"]
    p["seg_am"] = seg(f["openw"], f["c1130"])
    p["seg_pm"] = seg(f["c1130"], f["c1430"])
    p["seg_tail"] = seg(f["c1430"], f["closew"])
    pieces = {"next_1": (f["openw"], f["c1000"]), "next_2": (f["c1000"], f["c1130"]),
              "next_3": (f["c1130"], f["c1430"]), "next_4": (f["c1430"], f["closew"])}
    for name, (a, b) in pieces.items():
        per = (b - a) / base
        p[name] = per.mean(axis=1, skipna=False).shift(-1)
        for s in ("IH", "IC"):
            p[f"{name}_{s}"] = per[s].shift(-1)
    p["next_intraday"] = (((f["closew"] - f["openw"]) / base).mean(axis=1, skipna=False)).shift(-1)
    for s in ("IF", "IH", "IC"):
        p[f"next_intraday_{s}"] = ((f["closew"][s] - f["openw"][s]) / base[s]).shift(-1)
    close_ret = (f["closew"].diff() / f["open"].shift(1)).mean(axis=1, skipna=False)
    p["past_return"] = close_ret.shift(1).rolling(20, min_periods=20).sum()
    p["realized_vol"] = f["rv"].mean(axis=1, skipna=False).shift(1).rolling(20, min_periods=20).mean()
    day = spot.assign(date=pd.to_datetime(spot["date"]).dt.normalize())
    C.assert_no_holdout_dates(day["date"], "现货指数日线")
    so = day.pivot(index="date", columns="symbol", values="open").reindex(columns=list(SYMBOLS))
    sc = day.pivot(index="date", columns="symbol", values="close").reindex(columns=list(SYMBOLS))
    gap = (so / sc.shift(1) - 1).mean(axis=1, skipna=False)
    following = pd.Series(list(dates[1:]) + [pd.NaT], index=dates)
    p["auction_gap"] = gap.reindex(following.to_numpy()).to_numpy()
    flows = flows.copy()
    flows.index = pd.to_datetime(flows.index).normalize()
    C.assert_no_holdout_dates(flows.index, "市场资金流")
    flows["margin_net"] = (flows["buy_on_margin_value"] - flows["margin_repayment"]) / flows["margin_balance"].shift(1)
    turnover = flows["north_buy"] + flows["north_sell"]
    flows["north_net"] = (flows["north_buy"] - flows["north_sell"]) / turnover.shift(1).rolling(20, min_periods=10).mean()
    p["margin_net"] = flows["margin_net"].reindex(dates)
    p["north_net"] = flows["north_net"].reindex(dates)
    p["north_net_next"] = p["north_net"].shift(-1)
    p["post_auction"] = (dates >= SSE_CLOSING_AUCTION).astype(float)
    return p.replace([np.inf, -np.inf], np.nan)


def _standard(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()


def channel_tests(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    add = lambda tag, label, y, preds: rows.extend(                      # noqa: E731
        {"检验": tag, **row} for row in _fit_all(label, y, preds))
    y = p["next_intraday"]
    add("T1", "次日盘中 ~ 上午 + 下午前段 + 尾盘", y,
        [("上午", p["seg_am"]), ("下午前段", p["seg_pm"]), ("尾盘", p["seg_tail"])])
    add("T1", "次日盘中 ~ clv_1430 + (clv − clv_1430)", y, [("clv_1430", p["clv_1430"]), ("尾盘位置变化", p["late_move"])])
    add("T1", "次日盘中 ~ clv", y, [("clv", p["clv"])])
    for k, name in enumerate(("开盘→10:00", "10:00→11:30", "11:30→14:30", "14:30→收盘"), start=1):
        add("T2", f"次日 {name} ~ clv", p[f"next_{k}"], [("clv", p["clv"])])
        for s in ("IH", "IC"):
            add("T2", f"次日 {name} ~ clv | {s}", p[f"next_{k}_{s}"], [("clv", p["clv"])])
    gap = p["auction_gap"]
    add("T3", "次日盘中 ~ 集合竞价跳空", y, [("跳空", gap)])
    add("T3", "次日盘中 ~ clv + 跳空", y, [("clv", p["clv"]), ("跳空", gap)])
    add("T3", "次日盘中 ~ clv + 跳空 + 交乘", y,
        [("clv", p["clv"]), ("跳空", gap), ("clv×跳空", (p["clv"] - p["clv"].mean()) * _standard(gap))])
    add("T3", "跳空 ~ clv", gap, [("clv", p["clv"])])
    for tag, name, z in (("T4", "过去20日收益", p["past_return"]), ("T4", "已实现波动", p["realized_vol"]),
                         ("T5", "融资净买入", p["margin_net"]), ("T6", "北向净买入", p["north_net"]),
                         ("T7", "上交所收盘集合竞价之后", p["post_auction"])):
        zs = z if tag == "T7" else _standard(z)
        add(tag, f"次日盘中 ~ clv + {name} + 交乘", y,
            [("clv", p["clv"]), (name, zs), (f"clv×{name}", (p["clv"] - p["clv"].mean()) * zs)])
    add("T5", "融资净买入（当日）~ clv", _standard(p["margin_net"]), [("clv", p["clv"])])
    add("T6", "北向净买入（当日）~ clv", _standard(p["north_net"]), [("clv", p["clv"])])
    add("T6", "北向净买入（次日）~ clv", _standard(p["north_net_next"]), [("clv", p["clv"])])
    for s in SYMBOLS:
        add("T7", f"次日盘中 ~ clv + 之后 + 交乘 | {s}", p[f"next_intraday_{s}"],
            [("clv", p["clv"]), ("之后", p["post_auction"]),
             ("clv×之后", (p["clv"] - p["clv"].mean()) * p["post_auction"])])
    diff = p["next_intraday_IH"] - p["next_intraday_IC"]
    add("T7", "IH − IC 次日盘中 ~ clv + 之后 + 交乘", diff,
        [("clv", p["clv"]), ("之后", p["post_auction"]), ("clv×之后", (p["clv"] - p["clv"].mean()) * p["post_auction"])])
    return pd.DataFrame(rows)


def run_channels() -> None:
    protocol = load_channels_protocol()
    from tfcta.data import load_shard

    paths = {}
    for symbol in SYMBOLS:
        minute = load_shard(symbol, directory=C.INDEX_RESEARCH_DIR,
                            columns=["open", "openw", "close", "closew", "highw", "loww", "trading_date"])
        C.assert_no_holdout_dates(pd.to_datetime(minute["trading_date"]), what=f"{symbol} 股指分钟")
        paths[symbol] = minute_path(minute)
    for path in (FLOWS_PATH, SPOT_DAILY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}")
    panel = channel_panel(paths, pd.read_csv(SPOT_DAILY_PATH), pd.read_csv(FLOWS_PATH, index_col="date"))
    tables = []
    for label, start in (("2016-01-04 起", SHAPE_SUB_START), ("2015-04-16 起", SHAPE_START)):
        tables.append(channel_tests(panel[panel.index >= start]).assign(样本=label))
    table = pd.concat(tables, ignore_index=True)
    quint = pd.qcut(panel["clv"], 5, labels=False) + 1
    sub = panel.index >= SHAPE_SUB_START
    profile = panel[sub].groupby(quint[sub])[["seg_am", "seg_pm", "seg_tail", "auction_gap", "next_1", "next_2",
                                              "next_3", "next_4", "margin_net", "north_net"]].mean()
    for column in profile.columns:
        if column not in ("margin_net", "north_net"):
            profile[column] *= 1e4
    CHANNELS_OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(CHANNELS_OUT / "tests.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(CHANNELS_OUT / "clv_quintile_profile.csv", encoding="utf-8-sig")
    panel.to_csv(CHANNELS_OUT / "daily_panel.csv", encoding="utf-8-sig", index_label="trading_date")
    (CHANNELS_OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    print(table.round(2).to_string(index=False))
    print(profile.round(3).to_string())
    print(f"结果：{CHANNELS_OUT}")


# --------------------------------------------------------------------------
# 流动性提供补偿：Nagel 型时间序列与成分股截面（index_liquidity_provision.json）
# --------------------------------------------------------------------------
PROVISION_PATH = C.CONFIG_DIR / "research" / "index_liquidity_provision.json"
PROVISION_OUT = C.RUNS_DIR / "index" / "index_liquidity_provision"
CONSTITUENT_DIR = C.INDEX_DATA_ROOT / "external_rqdata" / "constituents"
NORTHBOUND_START = pd.Timestamp("2017-03-17")


def load_provision_protocol(path: Path = PROVISION_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022" not in sample or "不读 2026" not in sample
        or sorted(protocol.get("tests", {})) != ["N1", "N2", "N3", "S1", "S2", "S3"]
    ):
        raise ValueError("流动性提供检验登记与样本边界或实现不一致")
    return protocol


def nagel_tests(p: pd.DataFrame, rv_daily: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, bins = [], []
    y = p["next_intraday"]
    centered = p["clv"] - p["clv"].mean()
    for window in (5, 20, 60):
        vol = rv_daily.shift(1).rolling(window, min_periods=window).mean().reindex(p.index)
        z = _standard(vol)
        rows += [{"检验": "N1", **r} for r in _fit_all(f"次日盘中 ~ clv + 波动{window}日 + 交乘", y,
                                                     [("clv", p["clv"]), (f"波动{window}日", z),
                                                      (f"clv×波动{window}日", centered * z)])]
    vol20 = rv_daily.shift(1).rolling(20, min_periods=20).mean().reindex(p.index)
    past = vol20.shift(1)
    edges = [past.rolling(250, min_periods=120).quantile(q) for q in (0.2, 0.4, 0.6, 0.8)]
    bucket = pd.Series(np.nan, index=p.index)
    valid = edges[0].notna() & vol20.notna()
    bucket[valid] = 1 + sum((vol20 > e).astype(int) for e in edges)[valid]
    slopes = []
    for k in range(1, 6):
        mask = bucket == k
        r = _fit_row(f"次日盘中 ~ clv | 波动第{k}档", y[mask], [("clv", p["clv"][mask])])
        bins.append({"档": k, "clv斜率_bp": r["系数"] * 1e4, "NW_t": r["NW_t"], "天数": r["样本数"],
                     "平均20日波动": float(vol20[mask].mean())})
        slopes.append(r["系数"])
    from scipy.stats import spearmanr
    rho = spearmanr(range(1, 6), slopes).correlation
    rows.append({"检验": "N2", "检验": "N2", "变量": "档位与斜率秩相关", "系数_bp": rho, "NW_t": np.nan,
                 "样本数": int(bucket.notna().sum())})
    gross = (0.5 - p["clv"]) * y
    rows += [{"检验": "N3", **r} for r in _fit_all("反转毛收益 ~ 过去20日波动", gross, [("波动20日", _standard(vol20))])]
    return pd.DataFrame(rows), pd.DataFrame(bins)


def constituent_panel(market_clv: pd.Series) -> pd.DataFrame:
    """个股日面板：信号日 t 的特征和 t+1 开盘到收盘收益；剔除涨跌停收盘和停牌。"""
    prices = pd.read_parquet(CONSTITUENT_DIR / "prices.parquet")
    margin = pd.read_parquet(CONSTITUENT_DIR / "margin.parquet")
    shares = pd.read_parquet(CONSTITUENT_DIR / "shares.parquet")
    north = pd.read_parquet(CONSTITUENT_DIR / "northbound.parquet").rename(columns={"trading_date": "date"})
    member = pd.read_csv(CONSTITUENT_DIR / "membership.csv", parse_dates=["month_start"])
    for frame, label in ((prices, "价格"), (margin, "融资"), (shares, "股本"), (north, "北向")):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        C.assert_no_holdout_dates(frame["date"], f"成分股{label}")
    p = prices.sort_values(["order_book_id", "date"]).copy()
    g = p.groupby("order_book_id")
    p["next_intraday"] = g["close"].shift(-1) / g["open"].shift(-1) - 1
    p["next_date"] = g["date"].shift(-1)
    width = p["high"] - p["low"]
    p["own_clv"] = (p["close"] - p["low"]) / width.where(width > 0)
    p["own_ret"] = p["close"] / g["close"].shift(1) - 1
    locked = (p["close"] >= p["limit_up"] - 1e-6) | (p["close"] <= p["limit_down"] + 1e-6)
    p = p[~locked & (p["total_turnover"] > 0)]
    p = p.merge(shares[["order_book_id", "date", "circulation_a"]], on=["order_book_id", "date"], how="left")
    p["float_cap"] = p["close"] * p["circulation_a"]
    p["log_cap"] = np.log(p["float_cap"])
    p["turnover"] = p["total_turnover"] / p["float_cap"]
    m = margin.sort_values(["order_book_id", "date"]).copy()
    m["prev_balance"] = m.groupby("order_book_id")["margin_balance"].shift(1)
    m["margin_repay_net"] = (m["margin_repayment"] - m["buy_on_margin_value"]) / m["prev_balance"]
    p = p.merge(m[["order_book_id", "date", "prev_balance", "margin_repay_net"]], on=["order_book_id", "date"], how="left")
    p["margin_share"] = p["prev_balance"] / p["float_cap"]
    n = north.sort_values(["order_book_id", "date"]).copy()
    n["north_change"] = n.groupby("order_book_id")["holding_ratio"].diff()
    p = p.merge(n[["order_book_id", "date", "north_change"]], on=["order_book_id", "date"], how="left")
    member["month"] = member["month_start"].dt.to_period("M")
    p["month"] = p["date"].dt.to_period("M")
    p = p.merge(member.drop_duplicates(["month", "order_book_id"])[["month", "order_book_id", "index"]],
                on=["month", "order_book_id"], how="inner")
    p["market_clv"] = market_clv.reindex(p["date"]).to_numpy()
    return p.replace([np.inf, -np.inf], np.nan)


CROSS_VARS = ("own_clv", "own_ret", "margin_share", "margin_repay_net", "north_change", "log_cap", "turnover")


def fama_macbeth(panel: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """逐日截面：次日盘中 ~ 标准化特征；返回每日系数。"""
    records = []
    for day, part in panel.groupby("date"):
        part = part.dropna(subset=["next_intraday", *columns])
        if len(part) < 100:
            continue
        x = (part[list(columns)] - part[list(columns)].mean()) / part[list(columns)].std()
        design = np.column_stack([np.ones(len(part)), x.to_numpy()])
        if not np.isfinite(design).all():
            continue
        beta, *_ = np.linalg.lstsq(design, part["next_intraday"].to_numpy(), rcond=None)
        records.append({"date": day, **dict(zip(columns, beta[1:]))})
    return pd.DataFrame(records).set_index("date")


def summarize_coefficients(daily: pd.DataFrame, label: str) -> list[dict]:
    return [{"检验": label, "变量": c, "系数_bp": daily[c].mean() * 1e4,
             "NW_t": newey_west_t(daily[c].dropna(), HAC_LAGS), "样本数": int(daily[c].notna().sum())}
            for c in daily.columns]


def run_provision() -> None:
    protocol = load_provision_protocol()
    from tfcta.data import load_shard

    paths = {}
    for symbol in SYMBOLS:
        minute = load_shard(symbol, directory=C.INDEX_RESEARCH_DIR,
                            columns=["open", "openw", "close", "closew", "highw", "loww", "trading_date"])
        C.assert_no_holdout_dates(pd.to_datetime(minute["trading_date"]), what=f"{symbol} 股指分钟")
        paths[symbol] = minute_path(minute)
    panel = channel_panel(paths, pd.read_csv(SPOT_DAILY_PATH), pd.read_csv(FLOWS_PATH, index_col="date"))
    rv_daily = pd.DataFrame({s: paths[s]["rv"] for s in SYMBOLS}).mean(axis=1, skipna=False)
    tables, bins = [], []
    for label, start in (("2016-01-04 起", SHAPE_SUB_START), ("2015-04-16 起", SHAPE_START)):
        t, b = nagel_tests(panel[panel.index >= start], rv_daily)
        tables.append(t.assign(样本=label))
        bins.append(b.assign(样本=label))
    stocks = constituent_panel(panel["clv"])
    for label, start in (("2016-01-04 起", SHAPE_SUB_START), ("2015-04-16 起", SHAPE_START)):
        part = stocks[(stocks["date"] >= start) & (stocks["date"] <= SCORE_END)]
        weak, strong = part[part["market_clv"] < 0.5], part[part["market_clv"] >= 0.5]
        no_north = tuple(c for c in CROSS_VARS if c != "north_change")
        rows = []
        for tag, data, cols in (("S1 弱收盘日（不含北向）", weak, no_north), ("S2 强收盘日（不含北向）", strong, no_north),
                                ("S1 弱收盘日（含北向，2017-03-17 起）", weak[weak["date"] >= NORTHBOUND_START], CROSS_VARS),
                                ("S2 强收盘日（含北向，2017-03-17 起）", strong[strong["date"] >= NORTHBOUND_START], CROSS_VARS)):
            daily = fama_macbeth(data, cols)
            rows += summarize_coefficients(daily, tag)
            if tag.startswith("S1"):
                weak_daily = daily
            else:
                diff = weak_daily.mean() - daily.mean()
                se = np.sqrt(weak_daily.var() / len(weak_daily) + daily.var() / len(daily))
                rows += [{"检验": tag.replace("S2", "S1 − S2"), "变量": c, "系数_bp": diff[c] * 1e4,
                          "NW_t": diff[c] / se[c], "样本数": len(weak_daily)} for c in daily.columns]
        gap = weak.groupby(["date", "index"])["next_intraday"].mean().unstack()
        rows.append({"检验": "S3 弱收盘日次日盘中：中证500成分 − 沪深300成分", "变量": "均值差",
                     "系数_bp": (gap["CSI500"] - gap["CSI300"]).mean() * 1e4,
                     "NW_t": newey_west_t((gap["CSI500"] - gap["CSI300"]).dropna(), HAC_LAGS), "样本数": len(gap)})
        tables.append(pd.DataFrame(rows).assign(样本=label))
    table = pd.concat(tables, ignore_index=True)
    bins = pd.concat(bins, ignore_index=True)
    PROVISION_OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROVISION_OUT / "tests.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(PROVISION_OUT / "vol_buckets.csv", index=False, encoding="utf-8-sig")
    (PROVISION_OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print(table.round(3).to_string(index=False))
    print(bins.round(4).to_string(index=False))
    print(f"个股面板 {len(stocks)} 行，{stocks['order_book_id'].nunique()} 只；结果：{PROVISION_OUT}")


# --------------------------------------------------------------------------
# 逻辑链第⑥环：指数层面的被迫卖出（index_forced_selling.json）
# --------------------------------------------------------------------------
FORCED_PATH = C.CONFIG_DIR / "research" / "index_forced_selling.json"
FORCED_OUT = C.RUNS_DIR / "index" / "index_forced_selling"
INDEX_OF = {"IF": "CSI300", "IH": "SSE50", "IC": "CSI500"}
MEASURES = ("F1_margin_repay", "F2_margin_repay_balance", "F3_abnormal_turnover", "F4_north_sell")


def load_forced_protocol(path: Path = FORCED_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022" not in sample or "不读 2026" not in sample
        or [k for k in protocol.get("measures", {}) if k.startswith("F")] != list(MEASURES)
        or sorted(protocol.get("tests", {})) != [f"K{k}" for k in range(1, 7)]
    ):
        raise ValueError("被迫卖出检验登记与样本边界或实现不一致")
    return protocol


def trailing_z(series: pd.Series, window: int = 250, minimum: int = 120) -> pd.Series:
    """相对自身过去 window 日（不含当天）的 z 值。"""
    past = series.shift(1).rolling(window, min_periods=minimum)
    return (series - past.mean()) / past.std()


def index_forced_selling(directory: Path = CONSTITUENT_DIR) -> dict[str, pd.DataFrame]:
    """每个指数成分股的日度被迫卖出度量（原始值，未标准化）。涨跌停日不剔除：被迫卖出正好集中在这些日子。"""
    prices = pd.read_parquet(directory / "prices.parquet", columns=["order_book_id", "date", "close", "total_turnover"])
    margin = pd.read_parquet(directory / "margin.parquet")
    north = pd.read_parquet(directory / "northbound.parquet", columns=["order_book_id", "trading_date", "shares_holding"])
    member = pd.read_csv(directory / "membership.csv")
    member["month_start"] = pd.to_datetime(member["month_start"], format="mixed")
    north = north.rename(columns={"trading_date": "date"})
    for frame, label in ((prices, "价格"), (margin, "融资"), (north, "北向")):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        C.assert_no_holdout_dates(frame["date"], f"成分股{label}")
    margin = margin.sort_values(["order_book_id", "date"])
    margin["prev_balance"] = margin.groupby("order_book_id")["margin_balance"].shift(1)
    margin["net_repay"] = margin["margin_repayment"] - margin["buy_on_margin_value"]
    base = prices.merge(margin[["order_book_id", "date", "net_repay", "prev_balance"]], on=["order_book_id", "date"],
                        how="left")
    base = base.merge(north, on=["order_book_id", "date"], how="left").sort_values(["order_book_id", "date"])
    base["shares_holding"] = base.groupby("order_book_id")["shares_holding"].ffill()
    base["north_value"] = base.groupby("order_book_id")["shares_holding"].diff() * base["close"]
    base["month"] = base["date"].dt.to_period("M")
    member["month"] = member["month_start"].dt.to_period("M")
    out = {}
    for name in INDEX_OF.values():
        names = member.loc[member["index"] == name, ["month", "order_book_id"]].drop_duplicates()
        part = base.merge(names, on=["month", "order_book_id"], how="inner")
        daily = part.groupby("date").agg(turnover=("total_turnover", "sum"), net_repay=("net_repay", "sum"),
                                         prev_balance=("prev_balance", "sum"), north_value=("north_value", "sum"),
                                         names=("order_book_id", "nunique"))
        daily = daily[daily["turnover"] > 0]
        frame = pd.DataFrame(index=daily.index)
        frame["F1_margin_repay"] = daily["net_repay"] / daily["turnover"]
        frame["F2_margin_repay_balance"] = daily["net_repay"] / daily["prev_balance"].where(daily["prev_balance"] > 0)
        frame["F3_abnormal_turnover"] = daily["turnover"] / daily["turnover"].shift(1).rolling(20, min_periods=20).mean()
        frame["F4_north_sell"] = (-daily["north_value"] / daily["turnover"]).where(daily.index > NORTHBOUND_START)
        frame["names"] = daily["names"]
        out[name] = frame.replace([np.inf, -np.inf], np.nan)
    return out


def pooled_by_date(y: pd.DataFrame, xs: list[pd.DataFrame], lags: int = HAC_LAGS) -> tuple[float, float, int]:
    """IF/IH/IC 堆叠的 OLS；得分按日期加总后做 Newey–West，等价于按日期聚类再考虑序列相关。返回第一列系数和 t。"""
    pieces = {}
    for symbol in y.columns:
        columns = [y[symbol].rename("y")] + [x[symbol].rename(f"x{i}") for i, x in enumerate(xs)]
        pieces[symbol] = pd.concat(columns, axis=1)
    stacked = pd.concat(pieces).dropna()
    dates = stacked.index.get_level_values(1)
    values = stacked.to_numpy()
    target, design = values[:, 0], np.column_stack([np.ones(len(values)), values[:, 1:]])
    bread = np.linalg.pinv(design.T @ design)
    beta = bread @ design.T @ target
    score = pd.DataFrame(design * (target - design @ beta)[:, None]).groupby(dates).sum().sort_index().to_numpy()
    meat = score.T @ score
    for lag in range(1, lags + 1):
        gamma = score[lag:].T @ score[:-lag]
        meat += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    var = bread @ meat @ bread
    return float(beta[1]), float(beta[1] / np.sqrt(var[1, 1])), len(stacked)


def within_year_z(series: pd.Series) -> pd.Series:
    grouped = series.groupby(series.index.year)
    return (series - grouped.transform("mean")) / grouped.transform("std")


def forced_tests(m: pd.DataFrame, per: dict[str, pd.DataFrame], z: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, groups = [], []
    centered = m["clv"] - m["clv"].mean()
    weak, strong = m["clv"] < 0.5, m["clv"] >= 0.5
    for outcome, oname in (("next_intraday", "次日盘中"), ("next_open_to_open", "账本下一段")):
        y = m[outcome]
        for f in MEASURES:
            x = m[f]
            rows += [{"预测": "K1", "度量": f, **r} for r in _fit_all(
                f"{oname} ~ clv + F + clv×F", y, [("clv×F", centered * x), ("clv", m["clv"]), ("F", x)])]
            rows += [{"预测": "K2", "度量": f, **r} for r in _fit_all(
                f"{oname} ~ F + clv | 弱收盘日", y[weak], [("F", x[weak]), ("clv", m["clv"][weak])])]
            rows += [{"预测": "K4", "度量": f, **r} for r in _fit_all(
                f"{oname} ~ F | 强收盘日（安慰剂）", y[strong], [("F", x[strong])])]
            tercile = m[f"{f}_bin"]
            for k, name in enumerate(("低", "中", "高")):
                part = y[weak & (tercile == k)].dropna()
                groups.append({"结果": oname, "度量": f, "弱收盘日F三分位": name, "均值_bp": part.mean() * 1e4,
                               "NW_t": newey_west_t(part, HAC_LAGS), "天数": len(part)})
            ys = pd.DataFrame({s: per[outcome][s] for s in SYMBOLS})
            clv_wide = pd.DataFrame({s: m["clv"] for s in SYMBOLS})
            fz = pd.DataFrame({s: z[s][f] for s in SYMBOLS}).reindex(m.index)
            for s in SYMBOLS:
                rows += [{"预测": "K3", "度量": f, **r} for r in _fit_all(
                    f"{oname} ~ clv + F + clv×F | {s}↔{INDEX_OF[s]}", ys[s],
                    [("clv×F", centered * fz[s]), ("clv", m["clv"]), ("F", fz[s])])]
            coef, t, n = pooled_by_date(ys, [(clv_wide - m["clv"].mean()) * fz, clv_wide, fz])
            rows.append({"预测": "K3", "度量": f, "检验式": f"{oname} ~ clv + F + clv×F | 三品种堆叠", "变量": "clv×F",
                         "系数_bp": coef * 1e4, "NW_t": t, "样本数": n})
        if "qvix_year" in m:
            rows += [{"预测": "K5", "度量": "F1_margin_repay", **r} for r in _fit_all(
                f"{oname} ~ clv + F1 + QVIX + 两个交乘", y,
                [("clv×F1", centered * m["F1_margin_repay"]), ("clv×QVIX", centered * m["qvix_year"]),
                 ("clv", m["clv"]), ("F1", m["F1_margin_repay"]), ("QVIX", m["qvix_year"])])]
    for f in MEASURES:
        fitted = _fit_all("F（同日，z）~ clv", m[f], [("clv", m["clv"])])
        for row in fitted:
            row["系数_bp"] /= 1e4
        rows += [{"预测": "K6", "度量": f, **row} for row in fitted]
    return pd.DataFrame(rows), pd.DataFrame(groups)


def run_forced() -> None:
    from scipy.stats import norm

    from download.index.getQvix import load_research_qvix

    protocol = load_forced_protocol()
    bars = load_daily_bars(list(SYMBOLS), directory=C.INDEX_RESEARCH_DIR)
    per = shape_panel(bars)
    raw = index_forced_selling()
    z = {s: pd.DataFrame({f: trailing_z(raw[INDEX_OF[s]][f]) for f in MEASURES}) for s in SYMBOLS}
    market = pd.DataFrame({"clv": per["clv"].mean(axis=1, skipna=False),
                           "next_intraday": per["next_intraday"].mean(axis=1, skipna=False),
                           "next_open_to_open": per["next_open_to_open"].mean(axis=1, skipna=False)})
    for f in MEASURES:
        market[f] = pd.concat([z[s][f] for s in SYMBOLS], axis=1).reindex(market.index).mean(axis=1, skipna=False)
        market[f"{f}_bin"] = trailing_tercile(market[f])
    qvix = load_research_qvix().set_index("date")["close"]
    market["qvix_year"] = within_year_z(qvix.reindex(market.index))
    C.assert_no_holdout_dates(market.index, what="被迫卖出检验日期")
    tables, groups = [], []
    for label, start in (("2016-01-04 起", SHAPE_SUB_START), ("2015-04-16 起", SHAPE_START)):
        sel = market.index >= start
        t, g = forced_tests(market[sel], {k: v[v.index >= start] for k, v in per.items()}, z)
        tables.append(t.assign(样本=label))
        groups.append(g.assign(样本=label))
    table, groups = pd.concat(tables, ignore_index=True), pd.concat(groups, ignore_index=True)
    k1 = (table["预测"] == "K1") & (table["变量"] == "clv×F")
    table.loc[k1, "Bonferroni_p"] = np.minimum(1.0, 8 * 2 * norm.sf(table.loc[k1, "NW_t"].abs()))
    FORCED_OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(FORCED_OUT / "tests.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(FORCED_OUT / "weak_close_terciles.csv", index=False, encoding="utf-8-sig")
    market.to_csv(FORCED_OUT / "daily_market.csv", encoding="utf-8-sig", index_label="trading_date")
    coverage = {name: {"首日": str(frame.index.min().date()), "末日": str(frame.index.max().date()),
                       "成分股数中位": int(frame["names"].median())} for name, frame in raw.items()}
    (FORCED_OUT / "protocol.json").write_text(json.dumps({"protocol": protocol, "coverage": coverage},
                                                         ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 400)
    print(table.round(3).to_string(index=False))
    print(groups.round(2).to_string(index=False))
    print(json.dumps(coverage, ensure_ascii=False))
    print(f"结果：{FORCED_OUT}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="IF/IH/IC L1 盘口与收盘位置机制诊断")
    parser.add_argument("--overwrite", action="store_true", help="覆盖同一预登记协议的输出")
    parser.add_argument("--shape", action="store_true", help="相对位置与绝对幅度、经济渠道检验")
    parser.add_argument("--channels", action="store_true", help="路径、跳空、资金流等经济渠道（index_reversal_channels.json）")
    parser.add_argument("--provision", action="store_true", help="流动性提供补偿（index_liquidity_provision.json）")
    parser.add_argument("--forced", action="store_true", help="指数层面被迫卖出与反弹（index_forced_selling.json）")
    args = parser.parse_args(argv)
    if args.forced:
        run_forced()
    elif args.provision:
        run_provision()
    elif args.channels:
        run_channels()
    elif args.shape:
        run_shape()
    else:
        run(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
