"""开盘过度反应回归：研究期评估，以及 2022 的一次性压力诊断。"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tfcta import config as C
from tfcta.data import universe as U
from tfcta.research import stats

FEATURES = ROOT / "runs" / "morning_features"
OUT = ROOT / "runs" / "morning_oor"
CONFIG = ROOT / "config" / "morning_oor.json"
SECTORS = {
    "农产品": "A B C CS M Y P JD AP CF SR RM OI".split(),
    "黑色": "I J JM RB HC SF SM ZC".split(),
    "有色": "CU AL ZN PB NI SN".split(),
    "贵金属": "AU AG".split(),
    "能源": "BU FU SC".split(),
    "化工": "L V PP EG RU SP TA FG MA".split(),
}
SECTOR_OF = {s: name for name, members in SECTORS.items() for s in members}
YEARS = {"research": (2016, 2021), "validation_2022": (2022, 2022)}


def add_history(table, relvol_window, sigma_window):
    """相对量能和仓位波动只用前一交易日及更早的值；表须按品种、日期排好。"""
    by = table.groupby("symbol")
    usual = by.pre_volume.transform(
        lambda s: s.shift(1).rolling(relvol_window, min_periods=relvol_window * 3 // 4).median())
    table["relvol"] = table.pre_volume / usual
    table["sigma"] = by.gross_1130.transform(
        lambda s: s.shift(1).rolling(sigma_window, min_periods=sigma_window * 3 // 4).std())
    return table


def prepare(partition, spec, years=None, pools=None):
    """读特征表，补上只用历史的派生量，并标出当年股票池成员。样本外须传入年份和各年品种池。"""
    if partition == "oos":
        C.assert_oos_research_locked()
        if years is None or pools is None:
            raise ValueError("样本外需要年份和品种池")
    table = pd.read_parquet(FEATURES / f"{partition}.parquet")
    table = table.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    if table.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("特征表品种/日期重复")
    table = add_history(table, spec["rule"]["relvol_window"], spec["sizing"]["sigma_window"])
    unknown = set(table.symbol) - set(SECTOR_OF)
    if unknown:
        raise ValueError(f"未登记板块: {sorted(unknown)}")
    table["sector"] = table.symbol.map(SECTOR_OF)
    first, last = years or YEARS[partition]
    table = table[table.trading_date.dt.year.between(first, last)].copy()
    if partition == "research":
        C.assert_no_holdout_dates(table.trading_date, "早盘研究")
        pools = U.load_universe()
    elif partition == "oos":
        C.assert_strict_oos_dates(table.trading_date, "早盘样本外")
    else:
        C.assert_validation_2022_dates(table.trading_date, "早盘 2022 诊断")
        names, _ = U.validation_universe()
        pools = {2022: list(names)}
    table["member"] = [s in pools.get(d.year, ()) for s, d in zip(table.symbol, table.trading_date)]
    return table


def add_sector_deviation(table, min_peers=2):
    """同板块其他品种当天 dev15 的均值，以及本品种减去该均值的残差。至少 min_peers 个其他品种。"""
    dev = table["dev15"]
    keys = [table["trading_date"], table["sector"]]
    total = dev.groupby(keys).transform("sum")
    count = dev.groupby(keys).transform("count")
    peers = count - dev.notna().astype(int)
    sector_dev = ((total - dev.fillna(0.0)) / peers).where(peers >= min_peers)
    table = table.copy()
    table["sector_dev"] = sector_dev
    table["resid_dev"] = dev - sector_dev
    return table


def sides(table, rule, candidate):
    dev = table.resid_dev if candidate.get("sector_residual") else table.dev15
    ok = (dev.abs() > rule["dev_threshold"]) & (table.est_cost < rule["max_est_cost"]) & table.member
    c = rule["clock"]
    if candidate["clock"] == "high":  # 原规则：两个方向都看高点时钟
        ok &= ((dev > 0) & (table.hclock_td >= c)) | ((dev < 0) & (table.hclock_td <= 1 - c))
    elif candidate["clock"] == "extreme":  # 上冲看高点出现得晚，下杀看低点出现得晚
        ok &= ((dev > 0) & (table.hclock_td >= c)) | ((dev < 0) & (table.lclock_td >= c))
    if candidate["night_aligned"]:
        ok &= np.sign(table.ret_night) == np.sign(dev)
    if candidate["volume"]:
        ok &= table.relvol < rule["relvol_max"]
    if candidate.get("sector_residual"):
        ok &= table.sector_dev.notna() & (table.sector_dev.abs() <= rule["dev_threshold"])
    if candidate.get("sector_follow"):
        ok &= table.sector_dev.notna() & (table.sector_dev.abs() > rule["dev_threshold"])
        ok &= np.sign(table.sector_dev) == np.sign(dev)
    side = np.sign(dev) if candidate.get("sector_follow") else -np.sign(dev)
    return side.where(ok, 0.0).fillna(0.0)


def weights(table, side, sizing):
    raw = side * np.minimum(sizing["max_weight"], sizing["risk_per_trade"] / table.sigma)
    raw = raw.where(side != 0, 0.0)
    if (side != 0).any() and not np.isfinite(raw[side != 0]).all():
        raise ValueError("有信号但缺少波动估计")
    frame = pd.DataFrame({"day": table.trading_date, "sector": table.sector, "w": raw})
    sector_gross = frame.w.abs().groupby([frame.day, frame.sector]).transform("sum")
    frame["w"] *= np.minimum(1.0, sizing["max_sector_gross"] / sector_gross.where(sector_gross > 0, 1.0))
    gross = frame.w.abs().groupby(frame.day).transform("sum")
    frame["w"] *= np.minimum(1.0, sizing["max_gross"] / gross.where(gross > 0, 1.0))
    return frame.w


SCENARIOS = {
    "主口径": lambda t: (t.gross_1130, t.cost_1130),
    "滑点2跳": lambda t: (t.gross_1130, t.cost_1130 + 2 * t.tick_frac),
    "手续费翻倍": lambda t: (t.gross_1130, 2 * t.cost_1130 - 2 * t.tick_frac),
    "晚一分钟进场": lambda t: (t.gross_late_1130, t.cost_1130),
    "持有到10:15": lambda t: (t.gross_1015, t.cost_1015),
    "持有到15:00": lambda t: (t.gross_1500, t.cost_1500),
}


def daily(table, weight, scenario, calendar):
    gross, cost = SCENARIOS[scenario](table)
    active = weight != 0
    if not (np.isfinite(gross[active]).all() and np.isfinite(cost[active]).all()):
        raise ValueError(f"{scenario}: 持仓缺行情或成本")
    pnl = (weight * gross - weight.abs() * cost).where(active, 0.0)
    return pnl.groupby(table.trading_date).sum().reindex(calendar, fill_value=0.0)


def summary(ret, table, weight, scenario):
    gross, cost = SCENARIOS[scenario](table)
    active = weight != 0
    trade_net = (np.sign(weight) * gross - cost)[active]
    perf = stats.performance(ret)
    return {
        "夏普": stats.sharpe_ratio(ret),
        "年化收益": perf["ann_return"],
        "年化波动": perf["ann_vol"],
        "最大回撤": perf["max_drawdown"],
        "交易笔数": int(active.sum()),
        "持仓日占比": float((ret != 0).mean()),
        "单笔净收益bp": float(trade_net.mean() * 1e4) if active.any() else np.nan,
        "胜率": float((trade_net > 0).mean()) if active.any() else np.nan,
        "平均总权重": float(weight.abs().groupby(table.trading_date).sum().mean()),
    }


def breaker(ret, halt_drawdown):
    """账面净值距高点回撤超过阈值时停手；账面净值停手期间照常按信号累计，回到阈值内恢复。"""
    paper = (1 + ret).cumprod()
    drawdown = paper / paper.cummax().clip(lower=1.0) - 1  # 高点含初始本金
    on = (drawdown.shift(1, fill_value=0.0) >= -halt_drawdown).astype(float)
    return ret * on


def bootstrap_sharpe(ret, draws=3000, block=20, seed=20261008):
    rng = np.random.default_rng(seed)
    values = ret.to_numpy()
    n = len(values)
    starts = rng.integers(0, n, size=(draws, int(np.ceil(n / block))))
    index = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    sample = values[index]
    sr = sample.mean(axis=1) / sample.std(axis=1, ddof=1) * np.sqrt(252)
    return np.nanquantile(sr, [0.025, 0.975]).tolist()


NEIGHBORS = {
    "dev_threshold": [0.0015, 0.002, 0.0025, 0.003],
    "clock": [0.6, 0.6667, 0.75],
    "relvol_max": [1.2, 1.5, 2.0],
}


def evaluate(table, spec, candidate, scenario="主口径", rule=None):
    rule = rule or spec["rule"]
    side = sides(table, rule, candidate)
    weight = weights(table, side, spec["sizing"])
    calendar = pd.DatetimeIndex(sorted(table.trading_date.unique()))
    ret = daily(table, weight, scenario, calendar)
    return ret, weight


def yearly(ret):
    rows = {}
    for year, r in ret.groupby(ret.index.year):
        perf = stats.performance(r)
        rows[year] = {"夏普": stats.sharpe_ratio(r), "收益": float((1 + r).prod() - 1),
                      "最大回撤": perf["max_drawdown"], "持仓日": int((r != 0).sum())}
    return pd.DataFrame(rows).T


def plot(curves, path, title):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    for name, ret in curves.items():
        nav = (1 + ret).cumprod()
        top.plot(nav.index, nav, label=name, lw=1.6 if name.startswith("主策略") and "去" not in name else 0.9)
        if name == "主策略":
            bottom.fill_between(nav.index, nav / nav.cummax() - 1, 0, color="tab:red", alpha=0.5)
    top.set_title(title)
    top.legend(loc="upper left", fontsize=8)
    top.grid(alpha=0.3)
    bottom.set_ylabel("主策略回撤")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-2022", action="store_true")
    args = parser.parse_args()
    partition = "validation_2022" if args.validation_2022 else "research"
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    table = prepare(partition, spec)
    out = OUT / partition
    out.mkdir(parents=True, exist_ok=True)
    main_name = spec["main"]
    main_candidate = spec["candidates"][main_name]

    curves, rows, years = {}, {}, {}
    for name, candidate in spec["candidates"].items():
        ret, weight = evaluate(table, spec, candidate)
        curves[name] = ret
        rows[name] = summary(ret, table, weight, "主口径")
        years[name] = yearly(ret)
    result = pd.DataFrame(rows).T
    ret, weight = evaluate(table, spec, main_candidate)
    low, high = bootstrap_sharpe(ret)
    result.loc[main_name, "夏普95%下限"], result.loc[main_name, "夏普95%上限"] = low, high
    guarded = breaker(ret, spec["breaker"]["halt_drawdown"])
    curves[f"{main_name}+熔断"] = guarded
    perf = stats.performance(guarded)
    result.loc[f"{main_name}+熔断", ["夏普", "年化收益", "年化波动", "最大回撤", "持仓日占比"]] = [
        stats.sharpe_ratio(guarded), perf["ann_return"], perf["ann_vol"], perf["max_drawdown"],
        float((guarded != 0).mean())]
    years[f"{main_name}+熔断"] = yearly(guarded)
    result.to_csv(out / "candidates.csv", encoding="utf-8-sig")
    yearly_table = pd.concat(years, names=["候选", "年份"])
    yearly_table.to_csv(out / "yearly.csv", encoding="utf-8-sig")

    stress = {}
    for scenario in SCENARIOS:
        r, w = evaluate(table, spec, main_candidate, scenario)
        stress[scenario] = summary(r, table, w, scenario)
    stress = pd.DataFrame(stress).T
    stress.to_csv(out / "stress.csv", encoding="utf-8-sig")

    neighbors = []
    for key, values in NEIGHBORS.items():
        for value in values:
            rule = {**spec["rule"], key: value}
            r, w = evaluate(table, spec, main_candidate, rule=rule)
            neighbors.append({"参数": key, "取值": value, **summary(r, table, w, "主口径")})
    neighbors = pd.DataFrame(neighbors)
    neighbors.to_csv(out / "neighborhood.csv", index=False, encoding="utf-8-sig")

    active = weight != 0
    pnl = (weight * table.gross_1130 - weight.abs() * table.cost_1130)[active]
    trades = table.loc[active, ["symbol", "sector", "trading_date"]].assign(
        weight=weight[active], net=(np.sign(weight) * table.gross_1130 - table.cost_1130)[active], pnl=pnl)
    trades.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    by_symbol = trades.groupby("symbol").agg(笔数=("net", "size"), 单笔净bp=("net", lambda x: x.mean() * 1e4),
                                             组合贡献=("pnl", "sum")).sort_values("组合贡献", ascending=False)
    by_symbol.to_csv(out / "by_symbol.csv", encoding="utf-8-sig")
    by_sector = trades.groupby("sector").agg(笔数=("net", "size"), 单笔净bp=("net", lambda x: x.mean() * 1e4),
                                             组合贡献=("pnl", "sum"))
    by_sector.to_csv(out / "by_sector.csv", encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(out / "daily.csv", encoding="utf-8-sig")
    first, last = YEARS[partition]
    span = str(first) if first == last else f"{first}–{last}"
    plot(curves, out / "nav.png", f"开盘过度反应回归 {span}（含现金日，扣费后）")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(result.round(4).to_string())
    print(yearly_table.round(4).to_string())
    print(stress.round(4).to_string())
    print(neighbors.round(4).to_string())
    print(by_sector.round(4).to_string())
    print(by_symbol.head(12).round(4).to_string())


if __name__ == "__main__":
    main()
