"""日内时间因子在收盘位置之外的增量检验；只读研究期。

    python -m research.index.index_time_factor_increment

规格在 config/research/index_time_factor_increment.json。2016 年起逐品种结果必须与时间因子筛查一致。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from research.time_factor_screen import (
    _position,
    adjusted_return,
    controls_of,
    fair_prices,
    load_config as load_screen_config,
    ols_hac,
)
from tfcta import config as C
from tfcta.data import load_shard

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_time_factor_increment.json"
OUT = C.study_dir("index", "index_time_factor_increment")
# 筛查结果仍在旧布局 runs/commodity/time_factor_screen/，不在 study_dir 的 research/ 子目录下。
SCREEN_PANEL = C.RUNS_DIR / "commodity" / "time_factor_screen" / "panel.parquet"
EARLY_PATH = C.INDEX_DATA_ROOT / "if_before_ih_ic" / "minute_shards" / "IF.parquet"
SYMBOLS = ("IF", "IH", "IC")
FACTORS = ("ts_low_pm", "dfp_top3", "ts_high")
PRIOR = {"ts_low_pm": 1, "dfp_top3": 1, "ts_high": -1}
CONTROLS = ("ctrl_return", "ctrl_location", "ctrl_log_volume")
COLUMNS = ["trading_date", "open", "openw", "close", "closew", "highw", "loww", "volume", "total_turnover"]
COMMON_START, COMMON_END = pd.Timestamp("2015-04-16"), pd.Timestamp("2021-12-29")
SENSITIVITY_START = pd.Timestamp("2016-01-04")
EARLY_START, EARLY_END = pd.Timestamp("2011-01-04"), pd.Timestamp("2015-04-14")
SCREEN_START = pd.Timestamp("2016-01-04")
LOOKBACK, PERCENTILE = 250, 50.0
HAC_LAGS, MIN_YEAR_DAYS, WEAK_CLOSE = 5, 60, 0.4


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    duration = load_screen_config()["duration"]
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022–2025 样本外" not in sample
        or "不读 2026" not in sample
        or sorted(protocol.get("hypotheses", {})) != sorted(FACTORS)
        or any(protocol["hypotheses"][f].get("prior_sign") != PRIOR[f] for f in FACTORS)
        or "Holm" not in protocol.get("pass", {}).get("multiple_testing", "")
        or (int(duration["lookback"]), float(duration["percentile"])) != (LOOKBACK, PERCENTILE)
    ):
        raise ValueError("时间因子增量检验登记与实现或样本边界不一致")
    return protocol


def cash_clock(frame: pd.DataFrame) -> pd.DataFrame:
    """统一现金时段时钟：09:31–11:30 和 13:01–15:00；2015 年及以前的 09:16–09:30、15:01–15:15 去掉。"""
    hm = frame.index.strftime("%H:%M")
    keep = ((hm >= "09:31") & (hm <= "11:30")) | ((hm >= "13:01") & (hm <= "15:00"))
    return frame.loc[keep]


def symbol_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """单品种逐信号日的三个因子、控制变量和下一开盘到再下一开盘收益。只用信号日收盘前的 bar。"""
    frame = frame.sort_index(kind="mergesort").copy()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError("分钟索引必须是唯一的时间戳")
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(frame["trading_date"], "时间因子增量检验")
    first = frame.groupby("trading_date", sort=True).head(1).set_index("trading_date")
    clock = cash_clock(frame)
    fair, _ = fair_prices(clock, LOOKBACK, PERCENTILE, None)
    days = list(first.index)
    position = {day: k for k, day in enumerate(days)}
    rows = []
    for day, part in clock.groupby("trading_date", sort=True):
        if len(part) < 2 or part.index[-1].strftime("%H:%M") != "15:00":
            continue
        k = position[day]
        if k + 2 >= len(days):
            continue
        hold, after = first.loc[days[k + 1]], first.loc[days[k + 2]]
        afternoon = part.loc[part.index.strftime("%H:%M") >= "13:01"]
        rows.append({
            "trading_date": day,
            "ts_low_pm": _position(afternoon.loww.to_numpy(), "min") if len(afternoon) >= 2 else np.nan,
            "ts_high": _position(part.highw.to_numpy(), "max"),
            "dfp_top3": float(fair.at[day, "dfp_top3"]) if day in fair.index else np.nan,
            **controls_of(part),
            "y": adjusted_return(hold.openw, hold.open, after.openw),
        })
    panel = pd.DataFrame(rows).set_index("trading_date")
    panel.attrs["trading_days"] = days
    return panel


def reproduce_screen(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """2016 年起逐品种对照筛查保存的结果；任何一项对不上就报错。"""
    if not SCREEN_PANEL.is_file():
        raise FileNotFoundError(f"缺少筛查结果，无法核对：{SCREEN_PANEL}")
    screen = pd.read_parquet(SCREEN_PANEL)
    screen = screen[(screen.market == "index") & (screen.decision == "close") & screen.factor.isin(FACTORS)]
    rows = []
    for symbol in SYMBOLS:
        mine = panels[symbol]
        later = [day for day in mine.attrs["trading_days"] if day >= SCREEN_START]
        # 筛查的 dfp 阈值窗口在 2016 年初含 2015 年旧时钟 bar；只比较窗口完全在 2016 年起的日子。
        dfp_start = later[LOOKBACK]
        for factor in FACTORS:
            ref = screen[(screen.symbol == symbol) & (screen.factor == factor)].set_index("trading_date")
            if factor == "dfp_top3":
                ref = ref[ref.index >= dfp_start]
            ours = mine.reindex(ref.index)
            gaps = {
                factor: (ours[factor] - ref["value"]).abs().max(),
                "y": (ours["y"] - ref["gross"]).abs().max(),
                **{c: (ours[c] - ref[c]).abs().max() for c in CONTROLS},
            }
            worst = max(gaps.values())
            if ref.empty or ours[factor].isna().any() or not np.isfinite(worst) or worst > 1e-9:
                raise AssertionError(f"{symbol} {factor} 与时间因子筛查不一致：{gaps}")
            rows.append({"品种": symbol, "因子": factor, "核对天数": len(ref), "最大差": worst})
    return pd.DataFrame(rows)


def pooled(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """三品种等权；某个变量只要有一个品种缺失，当天该变量记缺失。"""
    columns = [*FACTORS, *CONTROLS, "y"]
    index = panels[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        index = index.union(panels[symbol].index)
    stacked = {c: pd.concat({s: panels[s][c].reindex(index) for s in SYMBOLS}, axis=1) for c in columns}
    return pd.DataFrame({c: frame.mean(axis=1, skipna=False) for c, frame in stacked.items()}, index=index)


def regression(frame: pd.DataFrame, factor: str, others=()) -> dict:
    cols = [factor, *others, *CONTROLS]
    data = frame[["y", *cols]].dropna()
    if len(data) < 30:
        return {"系数": np.nan, "t": np.nan, "p": np.nan, "bp_per_sd": np.nan, "天数": len(data)}
    coef, t = ols_hac(data["y"].to_numpy(), [data[c].to_numpy() for c in cols], HAC_LAGS)
    return {"系数": coef, "t": t, "p": float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else np.nan,
            "bp_per_sd": coef * float(data[factor].std()) * 1e4, "天数": len(data)}


def yearly_signs(frame: pd.DataFrame, factor: str, sign: float) -> str:
    years = [regression(part, factor)["系数"] for _, part in frame.groupby(frame.index.year)
             if part[[factor, "y", *CONTROLS]].dropna().shape[0] >= MIN_YEAR_DAYS]
    years = [v for v in years if np.isfinite(v)]
    return f"{sum(np.sign(v) == sign for v in years)}/{len(years)}"


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def verdict(factor: str, common: dict, holm_p: float, early: dict) -> str:
    sign = np.sign(common["系数"])
    if not (np.isfinite(holm_p) and holm_p < 0.05):
        return "不支持"
    if factor in ("ts_low_pm", "dfp_top3") and sign != PRIOR[factor]:
        return "不支持（方向与先验相反）"
    if np.isfinite(early["t"]) and np.sign(early["系数"]) == sign and abs(early["t"]) >= 1:
        return "支持增量"
    return "未复制"


def run() -> pd.DataFrame:
    protocol = load_protocol()
    panels = {}
    for symbol in SYMBOLS:
        frame = load_shard(symbol, directory=C.INDEX_RESEARCH_DIR, columns=COLUMNS)
        panels[symbol] = symbol_panel(frame)
        print(f"{symbol} 逐日因子完成", flush=True)
    check = reproduce_screen(panels)
    C.assert_research_only(EARLY_PATH)
    early_frame = pd.read_parquet(EARLY_PATH, columns=COLUMNS)
    if pd.to_datetime(early_frame["trading_date"]).max() >= pd.Timestamp("2015-04-17"):
        raise C.HoldoutViolation("早期 IF 分片越过 IH/IC 上市后的第一个交易日")
    early = symbol_panel(early_frame)
    early = early[(early.index >= EARLY_START) & (early.index <= EARLY_END)]
    common = pooled(panels)
    common = common[(common.index >= COMMON_START) & (common.index <= COMMON_END)]
    C.assert_no_holdout_dates(common.index, "时间因子增量检验信号日")

    samples = {
        "共同样本": common,
        "共同样本 2016 起": common[common.index >= SENSITIVITY_START],
        "早期 IF": early,
    }
    rows = []
    for name, frame in samples.items():
        weak = frame[frame["ctrl_location"] < WEAK_CLOSE]
        for factor in FACTORS:
            main = regression(frame, factor)
            sign = np.sign(main["系数"]) if np.isfinite(main["系数"]) else np.nan
            others = [f for f in FACTORS if f != factor]
            rows.append({
                "样本": name, "因子": factor, "先验方向": PRIOR[factor],
                "起点": str(frame[[factor]].dropna().index.min().date()),
                **{f"M1_{k}": v for k, v in main.items()},
                "逐年同号": yearly_signs(frame, factor, sign),
                **{f"M2弱收盘_{k}": v for k, v in regression(weak, factor).items() if k in ("t", "bp_per_sd", "天数")},
                "M3联合_t": regression(frame, factor, others)["t"],
            })
    table = pd.DataFrame(rows)
    primary = table[table["样本"] == "共同样本"].set_index("因子")
    adjusted = dict(zip(FACTORS, holm([primary.at[f, "M1_p"] for f in FACTORS])))
    table["M1_Holm_p"] = table.apply(lambda r: adjusted[r["因子"]] if r["样本"] == "共同样本" else np.nan, axis=1)
    early_rows = table[table["样本"] == "早期 IF"].set_index("因子")
    verdicts = pd.DataFrame([{
        "因子": f,
        "判定": verdict(f, {"系数": primary.at[f, "M1_系数"]}, adjusted[f],
                       {"系数": early_rows.at[f, "M1_系数"], "t": early_rows.at[f, "M1_t"]}),
        "共同样本_t": primary.at[f, "M1_t"], "Holm_p": adjusted[f],
        "早期IF_t": early_rows.at[f, "M1_t"],
    } for f in FACTORS])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    check.to_csv(OUT / "reproduction_check.csv", index=False, encoding="utf-8-sig")
    table.to_csv(OUT / "tests.csv", index=False, encoding="utf-8-sig")
    verdicts.to_csv(OUT / "verdict.csv", index=False, encoding="utf-8-sig")
    common.to_csv(OUT / "panel_common.csv", encoding="utf-8-sig")
    early.to_csv(OUT / "panel_early_if.csv", encoding="utf-8-sig")
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(check.to_string(index=False), flush=True)
        print(table.round(4).to_string(index=False), flush=True)
        print(verdicts.round(4).to_string(index=False), flush=True)
    print(f"结果：{OUT}", flush=True)
    return verdicts


if __name__ == "__main__":
    run()
