"""IF 日内承接成本假说的预登记机制检验；只读至 2021 年。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from research.time_factor_screen import ols_hac
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_intraday_cost_regime.json"
OUT = C.RUNS_DIR / "index" / "index_intraday_cost_regime"
EARLY_DIR = C.INDEX_DATA_ROOT / "if_before_ih_ic" / "minute_shards"
RESEARCH_DIR = C.INDEX_RESEARCH_DIR
EARLY_END = pd.Timestamp("2015-04-15")
OVERLAP_DAY = pd.Timestamp("2015-04-16")
CHEAP_END = pd.Timestamp("2015-08-25")
EXPENSIVE_START = pd.Timestamp("2015-09-07")
CRASH_FREE_END = pd.Timestamp("2015-05-29")
RESEARCH_END = pd.Timestamp("2021-12-31")
CHECKPOINTS = ("11:30", "14:30")
HAC_LAGS = 5


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    parameters = protocol.get("parameters", {})
    boundary = protocol.get("sample_boundary", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读2022" not in boundary
        or "不读2026" not in boundary
        or parameters.get("instrument") != "IF"
        or parameters.get("cheap_end") != str(CHEAP_END.date())
        or parameters.get("expensive_start") != str(EXPENSIVE_START.date())
        or parameters.get("checkpoints") != list(CHECKPOINTS)
        or parameters.get("session_end") != "15:00"
        or parameters.get("hac_lags") != HAC_LAGS
        or protocol.get("predictions", {}).get("P1_intraday") is None
    ):
        raise ValueError("IF 日内成本假说规格与实现或样本边界不一致")
    return protocol


def _read_partition(path: Path, *, research: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if research:
        C.assert_research_only(path)
    minute = pd.read_parquet(path)
    oi_path = path.parent / "open_interest" / path.name
    if research:
        C.assert_research_only(oi_path)
    oi = pd.read_parquet(oi_path)
    for label, frame in (("IF 分钟", minute), ("IF 持仓量", oi)):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
            raise ValueError(f"{label}必须有唯一的时间戳索引")
        if "trading_date" not in frame:
            raise KeyError(f"{label}缺少 trading_date")
        dates = pd.to_datetime(frame["trading_date"]).dt.normalize()
        C.assert_no_holdout_dates(dates, what=label)
        if not dates.is_monotonic_increasing:
            raise ValueError(f"{label}日期未按时间升序排列")
    if not minute.index.equals(oi.index):
        raise ValueError("IF 分钟行情与持仓量时间索引不一致")
    return minute.sort_index(kind="mergesort"), oi.sort_index(kind="mergesort")


def load_if_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """拼接 IF 单独期与股指研究期，核对 2015-04-16 重叠日。"""
    early, early_oi = _read_partition(EARLY_DIR / "IF.parquet", research=False)
    later, later_oi = _read_partition(RESEARCH_DIR / "IF.parquet", research=True)

    early_dates = pd.to_datetime(early["trading_date"]).dt.normalize()
    later_dates = pd.to_datetime(later["trading_date"]).dt.normalize()
    overlap_early = early.loc[early_dates == OVERLAP_DAY]
    overlap_later = later.loc[later_dates == OVERLAP_DAY]
    if not overlap_early.index.equals(overlap_later.index):
        raise ValueError("IF 两个分区的 2015-04-16 时间戳不一致")
    for field in ("open", "close", "openw", "closew", "highw", "loww"):
        if not np.allclose(
            overlap_early[field].to_numpy(dtype="float64"),
            overlap_later[field].to_numpy(dtype="float64"),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(f"IF 两个分区在 2015-04-16 的 {field} 不一致")
    overlap_early_oi = early_oi.loc[
        pd.to_datetime(early_oi["trading_date"]).dt.normalize() == OVERLAP_DAY
    ]
    overlap_later_oi = later_oi.loc[
        pd.to_datetime(later_oi["trading_date"]).dt.normalize() == OVERLAP_DAY
    ]
    if not overlap_early_oi.index.equals(overlap_later_oi.index):
        raise ValueError("IF 两个分区的 2015-04-16 持仓时间戳不一致")
    for field in ("open_interest", "dominant_id"):
        if not overlap_early_oi[field].equals(overlap_later_oi[field]):
            raise ValueError(f"IF 两个分区在 2015-04-16 的 {field} 不一致")

    early_mask = early_dates <= EARLY_END
    later_mask = (later_dates >= OVERLAP_DAY) & (later_dates <= RESEARCH_END)
    minute = pd.concat([early.loc[early_mask], later.loc[later_mask]]).sort_index()
    oi = pd.concat([early_oi.loc[early_mask], later_oi.loc[later_mask]]).sort_index()
    if minute.index.has_duplicates or not minute.index.is_monotonic_increasing:
        raise ValueError("IF 分区拼接后有重复或乱序的分钟")
    C.assert_no_holdout_dates(minute["trading_date"], what="IF 拼接分钟")
    if pd.to_datetime(minute["trading_date"]).max() > RESEARCH_END:
        raise C.HoldoutViolation("IF 分钟越过 2021-12-31")
    return minute, oi


def _clock_mask(index: pd.DatetimeIndex, full_session: bool) -> np.ndarray:
    minutes = index.hour * 60 + index.minute
    if full_session:
        return np.ones(len(index), dtype=bool)
    return ((minutes >= 9 * 60 + 31) & (minutes <= 11 * 60 + 30)) | (
        (minutes >= 13 * 60 + 1) & (minutes <= 15 * 60)
    )


def intraday_observations(minute: pd.DataFrame, *, full_session: bool = False) -> pd.DataFrame:
    """按分钟结束时刻计算检查点位置与后续收益。"""
    rows = []
    for day, group in minute.groupby(pd.to_datetime(minute["trading_date"]).dt.normalize(), sort=True):
        base = pd.Timestamp(day).normalize()
        group = group.sort_index(kind="mergesort")
        selected = group.loc[_clock_mask(group.index, full_session)]
        if selected.empty:
            continue
        end_time = selected.index[-1]
        if not full_session:
            end_time = base.replace(hour=15, minute=0)
            if end_time not in selected.index:
                continue
        end_row = selected.loc[end_time]
        for checkpoint in CHECKPOINTS:
            hour, minute_part = (int(value) for value in checkpoint.split(":"))
            stamp = base.replace(hour=hour, minute=minute_part)
            if stamp not in selected.index:
                continue
            prior = selected.loc[selected.index <= stamp]
            close = float(selected.at[stamp, "close"])
            open_px = float(prior["open"].iloc[0])
            if close <= 0 or open_px <= 0:
                raise ValueError(f"{day.date()} {checkpoint} IF 价格无效")
            width = float(prior["highw"].max() - prior["loww"].min())
            if not np.isfinite(width) or width <= 0:
                continue
            clv = (float(prior.at[stamp, "closew"]) - float(prior["loww"].min())) / width
            if not np.isfinite(clv) or not 0 <= clv <= 1:
                raise ValueError(f"{day.date()} {checkpoint} IF 日内位置超出 [0, 1]")
            next_bar = selected.loc[selected.index > stamp]
            if next_bar.empty:
                continue
            next_row = next_bar.iloc[0]
            outcome = (float(end_row["closew"]) - float(selected.at[stamp, "closew"])) / close
            skip_outcome = (float(end_row["closew"]) - float(next_row["closew"])) / float(next_row["close"])
            rows.append({
                "date": pd.Timestamp(day),
                "checkpoint": checkpoint,
                "location": clv,
                "return_bp": outcome * 1e4,
                "skip_one_bp": skip_outcome * 1e4,
                "open_to_checkpoint": (float(selected.at[stamp, "closew"]) - float(prior["openw"].iloc[0]))
                / open_px,
                "session_end": end_time,
                "clock": "full" if full_session else "cash",
            })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("IF 分钟数据未形成任何预登记检查点")
    return result


def daily_observations(minute: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, group in minute.groupby(pd.to_datetime(minute["trading_date"]).dt.normalize(), sort=True):
        group = group.sort_index(kind="mergesort")
        width = float(group["highw"].max() - group["loww"].min())
        if width <= 0:
            continue
        rows.append({
            "date": pd.Timestamp(day),
            "clv": (float(group["closew"].iloc[-1]) - float(group["loww"].min())) / width,
            "open": float(group["open"].iloc[0]),
            "openw": float(group["openw"].iloc[0]),
        })
    daily = pd.DataFrame(rows).set_index("date").sort_index()
    if daily.index.has_duplicates or daily[["clv", "open", "openw"]].isna().any().any():
        raise ValueError("IF 日线位置或开盘价缺失")
    daily["outcome_bp"] = (
        (daily["openw"].shift(-2) - daily["openw"].shift(-1))
        / daily["open"].shift(-1)
        * 1e4
    )
    daily["outcome_end"] = pd.Series(daily.index, index=daily.index).shift(-2)
    return daily


def fit(y: pd.Series, focus: pd.Series, controls: tuple[pd.Series, ...] = ()) -> dict:
    frame = pd.concat([y.rename("y"), focus.rename("focus"), *controls], axis=1).dropna()
    if len(frame) < len(controls) + 10:
        return {"系数_bp": np.nan, "HAC_t": np.nan, "样本数": len(frame)}
    coef, tstat = ols_hac(
        frame["y"].to_numpy(dtype="float64"),
        [frame["focus"].to_numpy(dtype="float64")]
        + [frame.iloc[:, i].to_numpy(dtype="float64") for i in range(2, frame.shape[1])],
        HAC_LAGS,
    )
    return {"系数_bp": coef, "HAC_t": tstat, "样本数": len(frame)}


def _regression_row(label: str, checkpoint: str, sample: str, result: dict, **extra) -> dict:
    return {
        "检验": label, "检查点": checkpoint, "样本": sample,
        **result, **extra,
    }


def _cheap(date: pd.Series) -> pd.Series:
    return (date >= pd.Timestamp("2010-01-01")) & (date <= CHEAP_END)


def _expensive(date: pd.Series) -> pd.Series:
    return date >= EXPENSIVE_START


def _p_two_sided(tstat: float) -> float:
    return float(2.0 * norm.sf(abs(tstat))) if np.isfinite(tstat) else np.nan


def _holm(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype="float64")
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted.tolist()
    order = finite[np.argsort(values[finite])]
    running = 0.0
    for rank, ix in enumerate(order):
        running = max(running, (len(order) - rank) * values[ix])
        adjusted[ix] = min(running, 1.0)
    return adjusted.tolist()


def _period_masks(dates: pd.Series) -> dict[str, pd.Series]:
    return {
        "便宜期": _cheap(dates),
        "昂贵期": _expensive(dates),
        "A_2015-09至2017-02": dates.between("2015-09-07", "2017-02-16"),
        "B_2017-02至2019-04": dates.between("2017-02-17", "2019-04-21"),
        "C_2019-04至2021": dates.between("2019-04-22", RESEARCH_END),
    }


def analyze(minute: pd.DataFrame, oi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    main = intraday_observations(minute)
    full = intraday_observations(minute, full_session=True)
    daily = daily_observations(minute)
    regressions = []
    interaction_rows = []

    for checkpoint in CHECKPOINTS:
        part = main.loc[main["checkpoint"] == checkpoint].copy()
        part["cheap"] = _cheap(part["date"])
        part["expensive"] = _expensive(part["date"])
        for label, mask in (("便宜期", part["cheap"]), ("昂贵期", part["expensive"])):
            sample = part.loc[mask]
            result = fit(sample["return_bp"], sample["location"])
            regressions.append(_regression_row("T1_主时段", checkpoint, label, result))
        eligible = part.loc[part["cheap"] | part["expensive"]].copy()
        cheap = eligible["cheap"].astype(float)
        interaction = eligible["location"] * cheap
        result = fit(eligible["return_bp"], interaction, (eligible["location"], cheap))
        interaction_rows.append(result)
        regressions.append(_regression_row("T2_制度交互", checkpoint, "便宜期对昂贵期", result))

        for label, mask in _period_masks(part["date"]).items():
            if label in ("便宜期", "昂贵期"):
                continue
            sample = part.loc[mask]
            result = fit(sample["return_bp"], sample["location"])
            regressions.append(_regression_row("T4_分段斜率", checkpoint, label, result))

        crash_free = part.loc[part["date"] <= CRASH_FREE_END]
        result = fit(crash_free["return_bp"], crash_free["location"])
        regressions.append(_regression_row("T5_去股灾便宜期", checkpoint, "截至2015-05-29", result))

        full_part = full.loc[full["checkpoint"] == checkpoint].copy()
        for label, mask in (("便宜期", _cheap(full_part["date"])), ("昂贵期", _expensive(full_part["date"]))):
            sample = full_part.loc[mask]
            result = fit(sample["return_bp"], sample["location"])
            regressions.append(_regression_row("T5_完整期货时段", checkpoint, label, result))

        skip_result = fit(part["skip_one_bp"], part["location"])
        regressions.append(_regression_row("T5_跳过检查点后一分钟", checkpoint, "全时期", skip_result))
        control_result = fit(
            part["return_bp"], part["location"], (part["open_to_checkpoint"],)
        )
        regressions.append(_regression_row("T5_控制开盘至检查点收益", checkpoint, "全时期", control_result))

    adjusted = _holm([_p_two_sided(row["HAC_t"]) for row in interaction_rows])
    for checkpoint, adjusted_p in zip(CHECKPOINTS, adjusted):
        for row in regressions:
            if row["检验"] == "T2_制度交互" and row["检查点"] == checkpoint:
                row["双侧p"] = _p_two_sided(row["HAC_t"])
                row["Holm_p"] = adjusted_p

    p2_frame = daily.reset_index()
    p2_frame["date"] = pd.to_datetime(p2_frame["date"])
    p2_frame["cheap"] = _cheap(p2_frame["date"])
    p2_frame["expensive"] = _expensive(p2_frame["date"])
    p2_frame = p2_frame.loc[
        (p2_frame["cheap"] & (p2_frame["outcome_end"] <= CHEAP_END))
        | p2_frame["expensive"]
    ].dropna(subset=["outcome_bp", "outcome_end"])
    cheap_flag = p2_frame["cheap"].astype(float)
    result = fit(
        p2_frame["outcome_bp"],
        p2_frame["clv"] * cheap_flag,
        (p2_frame["clv"], cheap_flag),
    )
    regressions.append(_regression_row("T3_下一开盘段制度交互", "全天CLV", "便宜期对昂贵期", result))
    for label, mask in _period_masks(p2_frame["date"]).items():
        sample = p2_frame.loc[mask]
        if sample.empty:
            continue
        result = fit(sample["outcome_bp"], sample["clv"])
        regressions.append(_regression_row("T4_P2分段斜率", "全天CLV", label, result))

    years = main.copy()
    years["year"] = years["date"].dt.year
    annual = []
    for (checkpoint, year), sample in years.groupby(["checkpoint", "year"], sort=True):
        if sample["date"].max() > CHEAP_END:
            sample = sample.loc[sample["date"] <= CHEAP_END]
        if sample.empty:
            continue
        result = fit(sample["return_bp"], sample["location"])
        if sample["date"].max() <= CHEAP_END:
            annual.append(_regression_row("T4_便宜期逐年", checkpoint, str(year), result))

    for year, group in minute.assign(
        trading_date=pd.to_datetime(minute["trading_date"]).dt.normalize()
    ).groupby(pd.to_datetime(minute["trading_date"]).dt.year, sort=True):
        day_volume = group.groupby("trading_date")["volume"].sum()
        oi_year = oi.loc[pd.to_datetime(oi["trading_date"]).dt.year == year].copy()
        oi_year["trading_date"] = pd.to_datetime(oi_year["trading_date"]).dt.normalize()
        closing_oi = oi_year.groupby("trading_date")["open_interest"].last()
        joined = pd.concat([day_volume.rename("volume"), closing_oi.rename("oi")], axis=1).dropna()
        ratio = joined["volume"] / joined["oi"].where(joined["oi"] > 0)
        annual.append({
            "检验": "D1_成交量持仓量", "检查点": "", "样本": str(year),
            "天数": len(joined), "成交量中位数": float(joined["volume"].median()),
            "收盘持仓量中位数": float(joined["oi"].median()),
            "成交量持仓量比中位数": float(ratio.median()),
        })
    verdict = _verdict(regressions)
    return pd.DataFrame(regressions), pd.DataFrame(annual), verdict


def _verdict(rows: list[dict]) -> str:
    t1 = {
        (row["检查点"], row["样本"]): row
        for row in rows if row["检验"] == "T1_主时段"
    }
    t2 = {row["检查点"]: row for row in rows if row["检验"] == "T2_制度交互"}
    t3 = next((row for row in rows if row["检验"] == "T3_下一开盘段制度交互"), None)
    crash = {
        row["检查点"]: row for row in rows if row["检验"] == "T5_去股灾便宜期"
    }
    for checkpoint in CHECKPOINTS:
        cheap = t1.get((checkpoint, "便宜期"), {})
        interaction = t2.get(checkpoint, {})
        crash_free = crash.get(checkpoint, {})
        if (
            cheap.get("系数_bp", np.nan) < 0
            and cheap.get("HAC_t", np.nan) <= -2
            and interaction.get("系数_bp", np.nan) < 0
            and interaction.get("Holm_p", np.nan) <= 0.05
            and t3 is not None
            and t3.get("系数_bp", np.nan) > 0
            and crash_free.get("系数_bp", np.nan) < 0
        ):
            return "支持（按预登记规则；仍是事后提出的机制假说）"
    if all(t1.get((point, "便宜期"), {}).get("系数_bp", np.nan) >= 0 for point in CHECKPOINTS):
        return "不支持：两个检查点的便宜期日内斜率均非负"
    if all(t2.get(point, {}).get("系数_bp", np.nan) >= 0 for point in CHECKPOINTS):
        return "不支持：两个检查点的制度交互项均非负"
    return "不确定"


def main() -> None:
    protocol = load_protocol()
    minute, oi = load_if_inputs()
    regressions, annual, verdict = analyze(minute, oi)
    OUT.mkdir(parents=True, exist_ok=True)
    regressions.to_csv(OUT / "regressions.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "annual_and_volume_oi.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"IF 日内承接成本假说：{verdict}", flush=True)
    print(regressions.to_string(index=False), flush=True)
    print(annual.to_string(index=False), flush=True)
    print(f"结果：{OUT}", flush=True)


if __name__ == "__main__":
    main()
