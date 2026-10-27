"""股指收盘位置与 L1 盘口机制诊断。"""
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="IF/IH/IC L1 盘口与收盘位置机制诊断")
    parser.add_argument("--overwrite", action="store_true", help="覆盖同一预登记协议的输出")
    args = parser.parse_args(argv)
    run(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
