"""做多日的盈亏，和下单时已经知道的量价。研究期加上已经打开的 2022，不读更晚的年份。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import INDEX_SYMBOLS, read_index_2022_minutes
from research.time_factor_screen import ols_hac
from tfcta import config as C
from tfcta.data import daily_bars, load_daily_bars

SPEC = C.CONFIG_DIR / "research" / "index_long_day_split.json"
OUT = C.RUNS_DIR / "index" / "index_long_day_split"
FIELDS = ("open", "openw", "close", "closew", "highw", "loww", "volume")
HAC = 5
FEATURES = ("signal_clv", "signal_return", "signal_range", "signal_volume_ratio")


def load_protocol() -> dict:
    protocol = json.loads(SPEC.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2023–2025" not in sample
        or "不读 2026" not in sample
        or "2022 只读已经打开" not in sample
        or list(protocol.get("features", {})) != list(FEATURES)
    ):
        raise ValueError("做多日盈亏比较的登记与实现不一致")
    return protocol


def market_features(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """三品种等权。成交量比用各品种自己过去 20 日，不含当天。"""
    close = bars["close"].where(bars["close"] > 0)
    width = bars["highw"] - bars["loww"]
    clv = (bars["closew"] - bars["loww"]) / width.where(width > 0)
    session = (bars["closew"] - bars["openw"]) / bars["open"].where(bars["open"] > 0)
    span = width / close
    past = bars["volume"].shift(1).rolling(20, min_periods=20).mean()
    volume_ratio = bars["volume"] / past.where(past > 0)
    frame = pd.DataFrame({
        "clv": clv.mean(axis=1, skipna=False),
        "session": session.mean(axis=1, skipna=False),
        "range": span.mean(axis=1, skipna=False),
        "volume_ratio": volume_ratio.mean(axis=1, skipna=False),
        "intraday": session.mean(axis=1, skipna=False),
        "open_to_next": (
            (bars["openw"].shift(-1) - bars["openw"]) / bars["open"].where(bars["open"] > 0)
        ).mean(axis=1, skipna=False),
        "close_to_next_open": (
            (bars["openw"].shift(-1) - bars["closew"]) / close
        ).mean(axis=1, skipna=False),
    })
    return frame.replace([np.inf, -np.inf], np.nan)


def attach_signal(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for name, source in (
        ("signal_clv", "clv"), ("signal_return", "session"),
        ("signal_range", "range"), ("signal_volume_ratio", "volume_ratio"),
    ):
        out[name] = features[source].shift(1)
    return out


def year_bars() -> dict[str, pd.DataFrame]:
    columns = ["trading_date", *FIELDS]
    frames = {
        symbol: daily_bars(read_index_2022_minutes(symbol, columns=columns))
        for symbol in INDEX_SYMBOLS
    }
    index = pd.DatetimeIndex(sorted(set.intersection(*(set(frame.index) for frame in frames.values()))))
    C.assert_validation_2022_dates(index, "2022 做多日诊断")
    return {field: pd.DataFrame({symbol: frames[symbol][field].reindex(index) for symbol in INDEX_SYMBOLS})
            for field in FIELDS}


def combined_features() -> pd.DataFrame:
    research = load_daily_bars(list(INDEX_SYMBOLS), directory=C.INDEX_RESEARCH_DIR)
    C.assert_no_holdout_dates(research["close"].index, "研究期日线")
    live = year_bars()
    bars = {
        field: pd.concat([research[field], live[field]]).sort_index()
        for field in FIELDS
    }
    if bars["close"].index.has_duplicates:
        raise ValueError("研究期和 2022 的交易日重叠")
    return attach_signal(market_features(bars))


def long_frame(net: pd.Series, held: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
    frame = features.reindex(net.index).copy()
    frame["净收益"] = net
    frame["做多"] = held.astype(bool)
    return frame.loc[frame["做多"]].dropna(subset=["净收益"])


def summarize(label: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    win, lose = frame["净收益"] > 0, frame["净收益"] < 0
    rows, slopes = [], []
    for name in FEATURES:
        rows.append({
            "样本": label, "特征": name,
            "赚钱日均值": frame.loc[win, name].mean(),
            "亏钱日均值": frame.loc[lose, name].mean(),
            "差": frame.loc[win, name].mean() - frame.loc[lose, name].mean(),
            "赚钱天数": int(win.sum()), "亏钱天数": int(lose.sum()),
        })
        coef, t_stat = ols_hac(frame["净收益"].to_numpy(), [frame[name].to_numpy()], HAC)
        slopes.append({"样本": label, "特征": name, "系数": coef, "NW_t": t_stat, "做多天数": len(frame)})
    return pd.DataFrame(rows), pd.DataFrame(slopes)


def path_row(label: str, frame: pd.DataFrame, flat_gross: pd.Series, skipped: pd.Series) -> dict:
    return {
        "样本": label,
        "做多天数": len(frame),
        "做多净收益之和": float(frame["净收益"].sum()),
        "赚钱日盘中均值_bp": frame.loc[frame["净收益"] > 0, "intraday"].mean() * 1e4,
        "亏钱日盘中均值_bp": frame.loc[frame["净收益"] < 0, "intraday"].mean() * 1e4,
        "赚钱日隔夜均值_bp": frame.loc[frame["净收益"] > 0, "close_to_next_open"].mean() * 1e4,
        "亏钱日隔夜均值_bp": frame.loc[frame["净收益"] < 0, "close_to_next_open"].mean() * 1e4,
        "空仓天数": int(flat_gross.shape[0]),
        "空仓日市场毛收益均值_bp": float(flat_gross.mean() * 1e4) if len(flat_gross) else np.nan,
        "信号成立但未持仓天数": int(skipped.shape[0]),
        "未持仓日市场毛收益均值_bp": float(skipped.mean() * 1e4) if len(skipped) else np.nan,
    }


def main() -> None:
    protocol = load_protocol()
    features = combined_features()
    research = pd.read_csv(
        C.RUNS_DIR / "index" / "index_close_location_candidate" / "daily.csv",
        index_col=0, parse_dates=True,
    )
    C.assert_no_holdout_dates(research.index, "研究期候选日表")
    live = pd.read_csv(
        C.RUNS_DIR / "strategies" / "index_close_location_reversal" / "2022" / "daily.csv",
        parse_dates=["交易日"],
    ).set_index("交易日")
    C.assert_validation_2022_dates(live.index, "2022 账本")

    research_long = long_frame(
        research["index_close_location_reversal"], research["候选目标敞口"].gt(0.5), features,
    )
    live_long = long_frame(live["净收益"], live["敞口倍数"].gt(0.05), features)
    groups = pd.concat([
        summarize("研究期 2016 起", research_long)[0],
        summarize("2022", live_long)[0],
    ], ignore_index=True)
    slopes = pd.concat([
        summarize("研究期 2016 起", research_long)[1],
        summarize("2022", live_long)[1],
    ], ignore_index=True)

    research_flat = features.reindex(research.index[research["候选目标敞口"].le(0.5)])["open_to_next"].dropna()
    research_skip = features.reindex(research.index[
        features.reindex(research.index)["signal_clv"].lt(0.5) & research["候选目标敞口"].le(0.5)
    ])["open_to_next"].dropna()
    live_flat = features.reindex(live.index[live["敞口倍数"].le(0.05)])["open_to_next"].dropna()
    live_skip = features.reindex(live.index[
        features.reindex(live.index)["signal_clv"].lt(0.5) & live["敞口倍数"].le(0.05)
    ])["open_to_next"].dropna()
    paths = pd.DataFrame([
        path_row("研究期 2016 起", research_long, research_flat, research_skip),
        path_row("2022", live_long, live_flat, live_skip),
    ])
    worst = live_long.nsmallest(5, "净收益")
    concentration = pd.DataFrame([{
        "最差5天净收益之和": float(worst["净收益"].sum()),
        "全部做多日净收益之和": float(live_long["净收益"].sum()),
        "占比": float(worst["净收益"].sum() / live_long["净收益"].sum()),
        "日期": "，".join(day.date().isoformat() for day in worst.index),
    }])

    OUT.mkdir(parents=True, exist_ok=True)
    groups.to_csv(OUT / "groups.csv", index=False, encoding="utf-8-sig")
    slopes.to_csv(OUT / "slopes.csv", index=False, encoding="utf-8-sig")
    paths.to_csv(OUT / "paths.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(OUT / "concentration.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(groups.round(4).to_string(index=False))
    print(slopes.round(4).to_string(index=False))
    print(paths.round(2).to_string(index=False))
    print(concentration.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
