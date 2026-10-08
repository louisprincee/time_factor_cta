"""持续期公允价偏离：研究期，以及同一规则在 2022 的一次性验证。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tfcta import config as C
from tfcta.data import bars as B, sessions
from tfcta.data import universe as U
from tfcta.factors import cache, intraday
from tfcta.research import costs, engine, stats, study

CONFIG = ROOT / "config" / "dfp_quantile.json"
OUT = ROOT / "runs" / "dfp_quantile"
WARMUP = "2019-01-01"


def load_spec():
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {
        "factor": "dfp_top3", "sign": 1, "threshold_lookback": 250, "threshold_pct": 55.0,
        "quantile_window": 30, "quantile_low": 0.2, "quantile_high": 0.8,
        "vol_target": 0, "cap": 1, "days": 1, "slippage_ticks": 1,
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"规则已被改动：{key}={spec.get(key)}，写死的是 {value}")
    if spec["threshold_lookback"] != C.IC_REFERENCE_LOOKBACK or spec["threshold_pct"] != C.IC_REFERENCE_PCT:
        raise ValueError("持续期阈值与已固定的 250 日、55% 分位不一致")
    return spec


def quantile_signal(raw, window, low, high):
    """当日因子对照此前 window 日的分位；中间和分位重合都不持仓。"""
    history = raw.shift(1).rolling(window, min_periods=window)
    lo, hi = history.quantile(low), history.quantile(high)
    out = raw * 0.0
    out = out.mask(raw > hi, 1.0).mask(raw < lo, -1.0)
    return out.where(lo.notna() & hi.notna() & (hi > lo))


def run(data, raw, spec):
    signed = raw * spec["sign"]
    signal = quantile_signal(signed, spec["quantile_window"], spec["quantile_low"], spec["quantile_high"])
    position = engine.position(signal, vol_target=spec["vol_target"], cap=spec["cap"], days=spec["days"])
    result = engine.backtest(
        position.reindex_like(data.returns), data.returns, data.universe,
        data.open_fee, data.close_fee, data.slippage, data.rolls, data.roll_close_fee)
    exposure = engine.allocate(position.reindex_like(data.returns), data.universe)
    return result, exposure


def describe(result, exposure, label):
    net, gross = result.net, result.gross
    perf = stats.performance(net)
    row = {
        "区间": label,
        "夏普": stats.sharpe_ratio(net),
        "毛夏普": stats.sharpe_ratio(gross),
        "年化收益": perf["ann_return"],
        "年化波动": perf["ann_vol"],
        "最大回撤": perf["max_drawdown"],
        "日胜率": perf["win_rate"],
        "年化换手": float(result.turnover.sum() * 252 / len(result)) if len(result) else np.nan,
        "年化成本": float(result.cost.sum() * 252 / len(result)) if len(result) else np.nan,
        "持仓日占比": float((exposure.abs().sum(axis=1) > 0).mean()) if len(exposure) else np.nan,
        "平均持仓品种数": float((exposure != 0).sum(axis=1).mean()) if len(exposure) else np.nan,
        "交易日": int(len(result)),
    }
    return row


def yearly(result):
    rows = []
    for year, frame in result.groupby(result.index.year):
        perf = stats.performance(frame.net)
        rows.append({
            "年": int(year), "夏普": stats.sharpe_ratio(frame.net),
            "年化收益": perf["ann_return"], "最大回撤": perf["max_drawdown"], "交易日": int(len(frame)),
        })
    return pd.DataFrame(rows)


def side_sharpe(data, exposure):
    """多头、空头分开记账。传入的权重已经按品种池分过，不再被分配函数缩小。"""
    out = {}
    for name, part in (("多头", exposure.clip(lower=0)), ("空头", exposure.clip(upper=0))):
        gross = (part * data.returns).sum(axis=1)
        opened, closed = engine.trade_legs(part, data.rolls)
        net = gross - _charge(opened, closed, data)
        out[name] = stats.sharpe_ratio(net)
        out[name + "年化"] = stats.performance(net)["ann_return"]
    return out


def _charge(opened, closed, data):
    def cost(size, rate):
        return (size * rate.reindex_like(size)).where(size > 0, 0.0)
    closing = data.close_fee.where(~data.rolls.reindex_like(opened).fillna(False), data.roll_close_fee)
    return (cost(opened, data.open_fee) + cost(closed, closing) + cost(opened + closed, data.slippage)).sum(axis=1)


def fmt(row):
    return (f"{row['区间']}  夏普 {row['夏普']:.2f}  毛夏普 {row['毛夏普']:.2f}  "
            f"年化 {row['年化收益']:.2%}  波动 {row['年化波动']:.2%}  回撤 {row['最大回撤']:.2%}  "
            f"胜率 {row['日胜率']:.1%}  年化成本 {row['年化成本']:.2%}  "
            f"有仓日 {row['持仓日占比']:.1%}  平均持仓 {row['平均持仓品种数']:.1f} 个  {row['交易日']} 日")


def research(spec):
    data = study.load_research(spec["slippage_ticks"])
    raw = data.factors.raw(spec["factor"])
    result, exposure = run(data, raw, spec)
    C.assert_no_holdout_dates(result.index, "公允价偏离研究期")
    return data, result, exposure


def _symbol_2022(symbol, spec):
    cached = cache.load_symbol(symbol, spec["threshold_lookback"], spec["threshold_pct"])[spec["factor"]]
    cached = cached[cached.index < "2022-01-01"]
    if cached.empty:
        raise ValueError(f"{symbol} 的研究期因子缓存是空的")
    minutes = B.load_minutes(symbol, include_validation=True, columns=list(B.MINUTE_COLUMNS))
    dates = pd.to_datetime(minutes["trading_date"])
    tail = sessions.add_intraday_coords(minutes.loc[dates >= WARMUP])
    fresh = intraday.duration_factors(tail, spec["threshold_lookback"], spec["threshold_pct"])[spec["factor"]]
    overlap = cached.index.intersection(fresh.index)
    overlap = overlap[overlap >= "2021-06-01"]
    both = pd.concat([cached.reindex(overlap), fresh.reindex(overlap)], axis=1).dropna()
    if len(both) < 60 or not np.allclose(both.iloc[:, 0], both.iloc[:, 1], rtol=1e-6, atol=1e-8):
        raise ValueError(f"{symbol} 截断预热后的持续期偏离与因子缓存不一致，2022 数值不能用")
    latest = fresh[fresh.index >= "2022-01-01"]
    C.assert_validation_2022_dates(latest.index, f"{symbol} 公允价偏离")
    day = B.daily_bars(minutes)
    day = day[day.index >= "2021-01-01"]
    return cached, latest, day


def validation(spec):
    names = [n for n in U.validation_universe()[0] if n in costs.MULTIPLIER]
    missing = [n for n in names if shard_io_missing(n)]
    if missing:
        raise FileNotFoundError(f"2022 池里缺验证分片: {missing}")
    cached, latest, days = {}, {}, {}
    for i, symbol in enumerate(names, 1):
        print(f"2022 因子 {i}/{len(names)} {symbol}", flush=True)
        old, new, day = _symbol_2022(symbol, spec)
        cached[symbol], latest[symbol], days[symbol] = old, new, day
    raw = pd.concat(
        [pd.DataFrame(cached).sort_index(), pd.DataFrame(latest).sort_index()]
    )
    if raw.index.has_duplicates:
        raise ValueError("研究期与 2022 的因子日期重叠")
    bars = B.wide_by_field(days)
    prices = bars["open"]
    returns = (bars["openw"].shift(-1) - bars["openw"]) / prices
    index = returns.index[(returns.index.year == 2022) & returns.notna().any(axis=1)]
    C.assert_validation_2022_dates(index, "公允价偏离 2022")
    rolls = pd.DataFrame(False, index=index, columns=names)
    for symbol, frame in B.load_roll_calendar(names, "2022-12-31").items():
        rolls.loc[index.intersection(frame.trading_date), symbol] = True
    fees = pd.concat([costs.load_fees("research"), costs.load_fees("validation_2022")], ignore_index=True)
    opened, closed, roll_closed = costs.fee_tables(fees, prices)
    slip = costs.slippage_tables(costs.load_ticks(), prices.loc[index], spec["slippage_ticks"])
    data = study.StudyData(
        None, returns.loc[index], {2022: names}, rolls,
        opened.loc[index], closed.loc[index], roll_closed.loc[index], slip)
    result, exposure = run(data, raw, spec)
    return data, result, exposure


def shard_io_missing(symbol):
    from tfcta.data import shard_io
    return shard_io.find_shard(C.VALIDATION_DIR, symbol) is None


def main():
    spec = load_spec()
    OUT.mkdir(parents=True, exist_ok=True)
    print("规则：dfp_top3，过去30日 20/80 分位，中间空仓，次日开盘持有，等权，1跳", flush=True)
    research_data, research_result, research_exposure = research(spec)
    research_row = describe(research_result, research_exposure, "2016-2021")
    print(fmt(research_row), flush=True)
    print("研究期多空", side_sharpe(research_data, research_exposure), flush=True)
    years = yearly(research_result)
    print(years.to_string(index=False), flush=True)
    data_2022, result_2022, exposure_2022 = validation(spec)
    row_2022 = describe(result_2022, exposure_2022, "2022")
    print(fmt(row_2022), flush=True)
    print("2022 多空", side_sharpe(data_2022, exposure_2022), flush=True)
    research_result.to_csv(OUT / "research_daily.csv")
    result_2022.to_csv(OUT / "validation_2022_daily.csv")
    pd.DataFrame([research_row, row_2022]).to_csv(OUT / "summary.csv", index=False)
    years.to_csv(OUT / "research_yearly.csv", index=False)
    print(f"写出 {OUT}", flush=True)


if __name__ == "__main__":
    main()
