"""早盘策略。不带版本参数时跑冻结的主策略（亏损反手）；版本参数只跑对应的一本。

    python scripts/morning.py
    python scripts/morning.py --v1
    python scripts/morning.py --v2
    python scripts/morning.py --benchmark
    python scripts/morning.py --compare
    python scripts/morning.py --validation-2022
    python scripts/morning.py --grid

规则在 config/morning.json。冻结主策略的输出在 runs/morning/meta/<分区>/，
v1、v2、benchmark 在 runs/morning/morning_<版本>/<分区>/，三者并列在 runs/morning/comparison/。
2022 的账面从研究期末接着算（2021 的单进入 2022 年初的窗口）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tfcta import config as C
from tfcta.data import load_universe, validation_universe
from tfcta.research import deflated_sharpe, performance, pbo_cscv, plot_performance, sharpe_ratio

CONFIG = ROOT / "config" / "morning.json"
RUNS = ROOT / "runs" / "morning"
FEATURES = RUNS / "features"
OUT = RUNS / "meta"
GRID_OUT = RUNS / "meta_grid"
PROJECT_TRIALS = 300  # 整个项目在早盘上的粗略尝试次数
SECTORS = {
    "农产品": "A B C CS M Y P JD LH RR AP CF CY SR OI RM RS PK CJ WH PM RI LR JR".split(),
    "黑色": "I J JM RB HC WR SS SF SM ZC".split(),
    "有色": "CU AL ZN PB NI SN BC AO AD SI LC PS".split(),
    "贵金属": "AU AG".split(),
    "能源": "BU FU LU SC PG".split(),
    "化工": "L V PP EG EB RU NR BR SP TA MA FG SA UR PF PX PR SH".split(),
    "建材": "BB FB LG".split(),
    "航运": "EC".split(),
}
SECTOR_OF = {s: name for name, members in SECTORS.items() for s in members}
YEARS = {"research": (2016, 2021), "validation_2022": (2022, 2022)}
SCENARIOS = {
    "主口径": lambda t: (t.gross_1130, t.cost_1130),
    "滑点2跳": lambda t: (t.gross_1130, t.cost_1130 + 2 * t.tick_frac),
}


def load():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def add_history(table, relvol_window, sigma_window):
    """相对量能和仓位波动只用前一交易日及更早的值；表须按品种、日期排好。"""
    by = table.groupby("symbol")
    usual = by.pre_volume.transform(
        lambda s: s.shift(1).rolling(relvol_window, min_periods=relvol_window * 3 // 4).median())
    table["relvol"] = table.pre_volume / usual
    table["sigma"] = by.gross_1130.transform(
        lambda s: s.shift(1).rolling(sigma_window, min_periods=sigma_window * 3 // 4).std())
    return table


def annotate(table, spec, pools):
    """补历史量能和波动，并标出板块与当年品种池。表须已经按品种、日期排好。"""
    table = add_history(table, spec["rule"]["relvol_window"], spec["sizing"]["sigma_window"])
    unknown = set(table.symbol) - set(SECTOR_OF)
    if unknown:
        raise ValueError(f"未登记板块: {sorted(unknown)}")
    table["sector"] = table.symbol.map(SECTOR_OF)
    table["member"] = [s in pools.get(d.year, ()) for s, d in zip(table.symbol, table.trading_date)]
    return table


def prepare(partition, spec, years=None, pools=None):
    """读特征表，补上只用历史的派生量，并标出当年品种池成员。样本外须传入年份和各年品种池。"""
    if partition == "oos":
        C.assert_oos_research_locked()
        if years is None or pools is None:
            raise ValueError("样本外需要年份和品种池")
    table = pd.read_parquet(FEATURES / f"{partition}.parquet")
    table = table.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    if table.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("特征表品种/日期重复")
    if partition == "research":
        pools = load_universe()
    elif partition != "oos":
        names, _ = validation_universe()
        pools = {2022: list(names)}
    table = annotate(table, spec, pools)
    first, last = years or YEARS[partition]
    table = table[table.trading_date.dt.year.between(first, last)].copy()
    if partition == "research":
        C.assert_no_holdout_dates(table.trading_date, "早盘研究")
    elif partition == "oos":
        C.assert_strict_oos_dates(table.trading_date, "早盘样本外")
    else:
        C.assert_validation_2022_dates(table.trading_date, "早盘 2022 诊断")
    return table


def prepare_through_2022(spec, first_year):
    """研究期特征表接 2022 验证表，滚动量跨年连续；给需要用研究期训练、在 2022 测试的诊断用。"""
    research = pd.read_parquet(FEATURES / "research.parquet")
    C.assert_no_holdout_dates(research.trading_date, "早盘研究期")
    later = pd.read_parquet(FEATURES / "validation_2022.parquet")
    year = pd.to_datetime(later.trading_date).dt.year
    # 研究期特征表只含研究期品种池；2022 才入池的品种用 2022 表自带的 2021 预热，量能和波动才算得出
    later = later[(year == 2022) | ((year == 2021) & ~later.symbol.isin(set(research.symbol)))]
    C.assert_validation_2022_dates(later.trading_date[pd.to_datetime(later.trading_date).dt.year == 2022], "早盘 2022")
    C.assert_no_holdout_dates(later.trading_date[pd.to_datetime(later.trading_date).dt.year < 2022], "早盘 2022 预热")
    table = pd.concat([research, later], ignore_index=True).sort_values(["symbol", "trading_date"])
    table = table.reset_index(drop=True)
    if table.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("特征表品种/日期重复")
    pools = {**load_universe(), 2022: list(validation_universe()[0])}
    table = annotate(table, spec, pools)
    return table[table.trading_date.dt.year.between(first_year, 2022)].copy()


def sides(table, rule):
    """上冲且高点出现得晚则做空，下杀且低点出现得晚则做多；还要求夜盘同向、量能不高。"""
    dev = table.dev15
    c = rule["clock"]
    ok = ((dev.abs() > rule["dev_threshold"]) & (table.est_cost < rule["max_est_cost"]) & table.member
          & (((dev > 0) & (table.hclock_td >= c)) | ((dev < 0) & (table.lclock_td >= c)))
          & (np.sign(table.ret_night) == np.sign(dev))
          & (table.relvol < rule["relvol_max"]))
    return (-np.sign(dev)).where(ok, 0.0).fillna(0.0)


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


def daily(table, weight, scenario, calendar):
    gross, cost = SCENARIOS[scenario](table)
    return position_pnl(table, weight, gross, cost, calendar)


def base_books(table, spec):
    """早盘策略的基础反向单。"""
    return {"主策略": sides(table, spec["rule"])}


def position_pnl(table, weight, gross, cost, calendar):
    """任意毛收益和成本口径下的日收益（现金日记 0）。"""
    active = weight != 0
    if not (np.isfinite(gross[active]).all() and np.isfinite(cost[active]).all()):
        raise ValueError("持仓缺行情或成本")
    out = (weight * gross - weight.abs() * cost).where(active, 0.0)
    return out.groupby(table.trading_date).sum().reindex(calendar, fill_value=0.0)


def book_row(table, side, sizing, calendar, gross=None, cost=None):
    """一本单的主口径与 2 跳结果、逐年夏普，以及顺势 / 反向两条腿。"""
    gross = table.gross_1130 if gross is None else gross
    cost = table.cost_1130 if cost is None else cost
    weight = weights(table, side, sizing)
    ret = position_pnl(table, weight, gross, cost, calendar)
    stressed = position_pnl(table, weight, gross, cost + 2 * table.tick_frac, calendar)
    perf = performance(ret)
    active = weight != 0
    net = (np.sign(weight) * gross - cost)[active]
    follow = (np.sign(weight) == np.sign(table.dev15))[active]
    row = {"夏普": sharpe_ratio(ret), "夏普95%下限": bootstrap_sharpe(ret)[0], "滑点2跳夏普": sharpe_ratio(stressed),
           "年化收益": perf["ann_return"], "最大回撤": perf["max_drawdown"], "交易笔数": int(active.sum()),
           "单笔净bp": net.mean() * 1e4 if active.any() else np.nan,
           "顺势笔数": int(follow.sum()), "顺势净bp": net[follow].mean() * 1e4 if follow.any() else np.nan,
           "反向笔数": int((~follow).sum()), "反向净bp": net[~follow].mean() * 1e4 if (~follow).any() else np.nan}
    for year, part in ret.groupby(ret.index.year):
        row[f"{year}夏普"] = sharpe_ratio(part)
    return row, ret


def bootstrap_sharpe(ret, draws=3000, block=20, seed=20261008):
    rng = np.random.default_rng(seed)
    values = ret.to_numpy()
    n = len(values)
    starts = rng.integers(0, n, size=(draws, int(np.ceil(n / block))))
    index = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    sample = values[index]
    sr = sample.mean(axis=1) / sample.std(axis=1, ddof=1) * np.sqrt(252)
    return np.nanquantile(sr, [0.025, 0.975]).tolist()


def trailing_mean(table, side, window):
    """每个交易日可用的“最近 window 笔账面单笔均值”：只用前一交易日及以前的单，当天的单不算。"""
    active = side != 0
    trades = pd.DataFrame({"day": table.trading_date[active], "symbol": table.symbol[active],
                           "net": (side * table.gross_1130 - table.cost_1130)[active]})
    trades = trades.sort_values(["day", "symbol"], kind="mergesort")
    rolling = trades.net.rolling(window, min_periods=window).mean()
    at_close = rolling.groupby(trades.day).last()  # 当天收盘后已知
    calendar = pd.DatetimeIndex(sorted(table.trading_date.unique()))
    known = at_close.reindex(calendar).ffill().shift(1)  # 次日才用
    return table.trading_date.map(known)


def trailing_days_mean(table, side, days):
    """最近 days 个交易日内已平仓单的单笔均值，只用前一交易日及以前；窗口内没有单时为空。"""
    active = side != 0
    net = (side * table.gross_1130 - table.cost_1130).where(active)
    calendar = pd.DatetimeIndex(sorted(table.trading_date.unique()))
    total = net.groupby(table.trading_date).sum().reindex(calendar, fill_value=0.0)
    count = active.groupby(table.trading_date).sum().reindex(calendar, fill_value=0)
    rolling = total.rolling(days, min_periods=days).sum() / count.rolling(days, min_periods=days).sum().replace(0, np.nan)
    return table.trading_date.map(rolling.shift(1))


def switched(side, score, mode):
    """最近窗口账面均值为负时，把原方向反过来。其他模式已移除。"""
    if mode != "亏损反手":
        raise ValueError(f"只保留亏损反手，收到 {mode}")
    return side.where(~(score < 0), -side)


def loss_switch(table, side, window):
    """用这本单自己最近 window 笔已平仓的扣费收益。均值转负时，下一交易日把信号取反。"""
    return switched(side, trailing_mean(table, side, window), "亏损反手")


def v2_sides(table, rule):
    dev = table.dev15
    up, down = dev > rule["dev_threshold"], dev < -rule["dev_threshold"]
    side = pd.Series(0.0, index=table.index)
    side.loc[up & table.hclock_td.le(0.2)] = 1.0
    side.loc[up & table.hclock_td.ge(0.8)] = -1.0
    side.loc[down & table.lclock_td.le(0.2)] = -1.0
    side.loc[down & table.lclock_td.ge(0.8)] = 1.0
    eligible = (table.member & table.ret_night.notna()
                & table.est_cost.lt(rule["max_est_cost"])
                & table.relvol.lt(rule["relvol_max"])
                & table.hclock_td.notna() & table.lclock_td.notna()
                & table.gross_1130.notna() & table.cost_1130.notna()
                & np.isfinite(table.sigma) & table.sigma.gt(0)
                & (np.sign(table.ret_night) == np.sign(dev)))
    return side.where(eligible, 0.0)


def benchmark_sides(table, rule, threshold):
    """低点时钟减高点时钟。只在绝对差达到阈值时交易，方向跟着差的符号，不用偏离和夜盘方向。"""
    spread = table.lclock_td - table.hclock_td
    side = pd.Series(np.where(spread.ge(threshold), 1.0,
                              np.where(spread.le(-threshold), -1.0, 0.0)), index=table.index)
    eligible = (table.member & table.ret_night.notna()
                & table.est_cost.lt(rule["max_est_cost"])
                & table.relvol.lt(rule["relvol_max"])
                & table.hclock_td.notna() & table.lclock_td.notna()
                & table.gross_1130.notna() & table.cost_1130.notna()
                & np.isfinite(table.sigma) & table.sigma.gt(0))
    return side.where(eligible, 0.0)


def cost_attribution(table, side, sizing, calendar, strategy):
    position_weights = weights(table, side, sizing)
    active = position_weights.ne(0)
    direction = np.sign(position_weights)
    gross_per_signal = direction * table.gross_1130
    net_per_signal = gross_per_signal - table.cost_1130
    rows = []
    groups = [("全部信号", active), ("多头信号", active & direction.gt(0)),
              ("空头信号", active & direction.lt(0))]
    for sector in sorted(set(SECTOR_OF.values())):
        in_sector = table.sector.eq(sector)
        groups.extend([(f"{sector}·多头", active & in_sector & direction.gt(0)),
                       (f"{sector}·空头", active & in_sector & direction.lt(0))])
    for label, mask in groups:
        if mask.any():
            rows.append({"策略": strategy, "范围": label, "信号数": int(mask.sum()),
                         "方向毛收益bp": float(gross_per_signal[mask].mean() * 1e4),
                         "实际成本bp": float(table.cost_1130[mask].mean() * 1e4),
                         "方向净收益bp": float(net_per_signal[mask].mean() * 1e4)})

    zero_cost = pd.Series(0.0, index=table.index)
    gross_daily = position_pnl(table, position_weights, table.gross_1130, zero_cost, calendar)
    net_daily = position_pnl(table, position_weights, table.gross_1130, table.cost_1130, calendar)
    stressed = position_pnl(table, position_weights, table.gross_1130,
                            table.cost_1130 + 2 * table.tick_frac, calendar)
    rows.append({"策略": strategy, "范围": "风险加权组合", "信号数": int(active.sum()),
                 "扣费前Sharpe": sharpe_ratio(gross_daily), "扣费后Sharpe": sharpe_ratio(net_daily),
                 "加2跳Sharpe": sharpe_ratio(stressed),
                 "毛年化收益": performance(gross_daily)["ann_return"],
                 "净年化收益": performance(net_daily)["ann_return"],
                 "净最大回撤": performance(net_daily)["max_drawdown"],
                 "平均日成本bp": float((gross_daily - net_daily).mean() * 1e4),
                 "平均日额外2跳bp": float((net_daily - stressed).mean() * 1e4)})
    return pd.DataFrame(rows)


def run_versions(version, partition="research"):
    spec = load()
    table = (prepare_through_2022(spec, 2016) if partition == "validation_2022"
             else prepare("research", spec))
    keep = table.trading_date.dt.year.eq(2022) if partition == "validation_2022" else table.trading_date.dt.year.between(2016, 2021)
    if partition == "validation_2022":
        C.assert_validation_2022_dates(table.loc[keep, "trading_date"], "morning strategy validation")
    view = table.loc[keep].reset_index(drop=True)
    calendar = pd.DatetimeIndex(sorted(view.trading_date.unique()))

    window = 15  # v1、v2 的亏损反手。不带版本参数的冻结主策略仍用配置里的 30 笔。
    threshold = spec["benchmark"]["threshold"]
    bases = {"v1": base_books(table, spec)["主策略"], "v2": v2_sides(table, spec["rule"]),
             "benchmark": benchmark_sides(table, spec["rule"], threshold)}
    signals = {"v1": loss_switch(table, bases["v1"], window),
               "v2": loss_switch(table, bases["v2"], window),
               "benchmark": bases["benchmark"]}
    chosen = tuple(signals) if version == "compare" else (version,)
    if any(name not in signals for name in chosen):
        raise ValueError(f"未知策略版本: {version}")

    rows, curves, attribution = [], {}, []
    scopes = [("全板块", pd.Series(True, index=view.index)),
              ("农产品有色化工", view.sector.isin(("农产品", "有色", "化工")))]
    for name in chosen:
        signal = signals[name].loc[keep].reset_index(drop=True)
        base_signal = bases[name].loc[keep].reset_index(drop=True)
        for scope_name, scope_mask in scopes:
            scoped = view.loc[scope_mask].reset_index(drop=True)
            side = signal.loc[scope_mask].reset_index(drop=True)
            base_side = base_signal.loc[scope_mask].reset_index(drop=True)
            label = f"morning_{name}·{scope_name}"
            row, returns = book_row(scoped, side, spec["sizing"], calendar)
            active = side.ne(0) | base_side.ne(0)
            row.update({"组合": label, "切换比例": float((side != base_side)[active].mean()) if active.any() else 0.0,
                        "信号品种日": int(side.ne(0).sum()),
                        "信号交易日": int(scoped.loc[side.ne(0), "trading_date"].nunique()),
                        "涉及品种": int(scoped.loc[side.ne(0), "symbol"].nunique())})
            rows.append(row)
            curves[label] = returns
            attribution.append(cost_attribution(scoped, side, spec["sizing"], calendar, label))

    output_name = "comparison" if version == "compare" else f"morning_{version}"
    output = RUNS / output_name / partition
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).set_index("组合")
    result.to_csv(output / "books.csv", encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(output / "daily.csv", encoding="utf-8-sig")
    pd.concat(attribution, ignore_index=True).to_csv(
        output / "cost_attribution.csv", index=False, encoding="utf-8-sig")
    (output / "spec.json").write_text(json.dumps({
        "version": version, "partition": partition,
        "loss_switch_window": window,
        "v1_loss_switch": f"v1 自己最近 {window} 笔扣费收益均值 < 0 时，把 v1 信号取反",
        "v2_loss_switch": f"v2 自己最近 {window} 笔扣费收益均值 < 0 时，把 v2 信号取反",
        "benchmark_threshold": threshold,
        "benchmark_signal": spec["benchmark"]["signal"],
        "benchmark_uses_dev15": False, "benchmark_uses_night_direction": False,
        "benchmark_threshold_basis": spec["benchmark"]["note"],
        "oos_accessed": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    shown = "v1 / v2 / benchmark" if version == "compare" else f"morning_{version}"
    for scope_name in ("全板块", "农产品有色化工"):
        plot_series = {name: curves[f"morning_{name}·{scope_name}"] for name in chosen}
        plot_performance(plot_series, output / "plots" / scope_name,
                         f"{scope_name}：{shown}（{partition}）")
    print(result.round(3).to_string())
    print("结果：", output)


def run(partition):
    spec = load()
    if partition == "research":
        table = prepare("research", spec)
        C.assert_no_holdout_dates(table.trading_date, "早盘策略")
    else:
        table = prepare_through_2022(spec, 2016)
    books = base_books(table, spec)
    windows = [spec["window_trades"], *spec["neighborhood"]["window_trades"]]
    keep = table.trading_date.dt.year.between(*((2016, 2021) if partition == "research" else (2022, 2022)))
    view = table[keep].reset_index(drop=True)
    if partition != "research":
        C.assert_validation_2022_dates(view.trading_date, "早盘策略 2022")
    calendar = pd.DatetimeIndex(sorted(view.trading_date.unique()))
    if set(books) != {"主策略"} or set(spec["modes"]) != {"亏损反手"}:
        raise ValueError("早盘策略只保留主策略的亏损反手")
    side = books["主策略"]
    rows, curves = [], {}
    # 不切换的主策略只作对照，样本外核对要和已落盘的这一列逐日比较；它不是可选模式。
    plain = side[keep].reset_index(drop=True)
    row, ret = book_row(view, plain, spec["sizing"], calendar)
    rows.append({"组合": "主策略·原样", "窗口": spec["window_trades"], "切换比例": 0.0,
                 **row, "主设定": False})
    curves["主策略·原样"] = ret
    for window in windows:
        score = trailing_mean(table, side, window)[keep].reset_index(drop=True)
        base_side = side[keep].reset_index(drop=True)
        new = switched(base_side, score, "亏损反手")
        row, ret = book_row(view, new, spec["sizing"], calendar)
        losing = (score < 0)[base_side != 0]
        name = f"主策略·亏损反手·{window}笔"
        rows.append({"组合": name, "窗口": window, "切换比例": float(losing.mean()),
                     **row, "主设定": window == spec["window_trades"]})
        curves[name] = ret
    result = pd.DataFrame(rows).set_index("组合")
    if partition == "research":
        result["保留"] = result.主设定 & (result.夏普 > 0)  # 保留只看研究期
    out = OUT / partition
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "books.csv", encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(out / "daily.csv", encoding="utf-8-sig")
    main_names = [n for n in curves if result.loc[n, "主设定"]]
    print("图表：", plot_performance(curves, out / "plots" / ("research" if partition == "research" else "validation-2022"), f"早盘策略 {partition}（30 笔主设定，扣费后）", main_names))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(result.round(3).to_string())
    print("结果：", out)


def window_grid(spec):
    if spec["books"] != ["主策略"] or spec["modes"] != ["亏损反手"]:
        raise ValueError("参数面只保留主策略的亏损反手，变化的是回看窗口")
    for kind, sizes in spec["windows"].items():
        for size in sizes:
            yield "主策略", "亏损反手", kind, size


def run_grid():
    spec = load()
    grid = spec["grid"]
    first, _last = grid["evaluation"]["period"]
    table = prepare_through_2022(spec, first)
    C.assert_validation_2022_dates(table.trading_date[table.trading_date.dt.year == 2022], "早盘参数面 2022")
    books = base_books(table, spec)
    calendar = pd.DatetimeIndex(sorted(table.trading_date.unique()))
    rows, curves = [], {}
    for book, mode, kind, size in window_grid(grid):
        side = books[book]
        score = trailing_mean(table, side, size) if kind == "笔数" else trailing_days_mean(table, side, size)
        side = switched(side, score, mode)
        weight = weights(table, side, spec["sizing"])
        ret = daily(table, weight, "主口径", calendar)
        stressed = daily(table, weight, "滑点2跳", calendar)
        unit = {"笔数": "笔", "交易日": "日"}[kind]
        name = f"{book}·{mode}·{size}{unit}"
        dev, later = ret[ret.index.year <= 2021], ret[ret.index.year == 2022]
        rows.append({"组合": name, "基础单": book, "方式": mode, "口径": kind or "-", "窗口": size or 0,
                     "2016-2022夏普": sharpe_ratio(ret), "2016-2021夏普": sharpe_ratio(dev),
                     "2022夏普": sharpe_ratio(later), "2跳夏普": sharpe_ratio(stressed),
                     "交易笔数": int((weight != 0).sum()),
                     **{f"{y}夏普": sharpe_ratio(p) for y, p in ret.groupby(ret.index.year)}})
        curves[name] = ret
    result = pd.DataFrame(rows).set_index("组合")
    matrix = pd.DataFrame(curves)
    pbo = pbo_cscv(matrix, grid["evaluation"]["pbo_blocks"])
    best = result["2016-2022夏普"].idxmax()
    trial_sharpes = result["2016-2022夏普"].to_numpy()
    dsr_grid = deflated_sharpe(curves[best], trial_sharpes)
    dsr_project = deflated_sharpe(curves[best], trial_sharpes, PROJECT_TRIALS)
    frozen = "主策略·亏损反手·30笔"
    out = GRID_OUT
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "grid.csv", encoding="utf-8-sig")
    matrix.to_csv(out / "daily.csv", encoding="utf-8-sig")
    summary = {"最好的组合": best, "PBO": pbo, "DSR（网格内）": dsr_grid, f"DSR（约 {PROJECT_TRIALS} 次尝试）": dsr_project,
               "冻结组合": {"组合": frozen, **result.loc[frozen, ["2016-2021夏普", "2022夏普", "2016-2022夏普"]].to_dict()}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    top = list(dict.fromkeys([*result["2016-2022夏普"].nlargest(5).index, frozen]))
    print("图表：", plot_performance(matrix[top], out / "plots", "早盘参数面 2016–2022（前五与冻结组合）"))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    shown = ["交易笔数", "2016-2021夏普", "2022夏普", "2016-2022夏普", "2跳夏普"]
    print(result[shown].sort_values("2016-2022夏普", ascending=False).round(2).to_string())
    for book in grid["books"]:
        for mode in grid["modes"]:
            part = result[(result.基础单 == book) & (result.方式 == mode)]
            print(f"\n{book}·{mode}：窗口 → 2016-2021 / 2022 夏普")
            print(part.set_index(["口径", "窗口"])[["2016-2021夏普", "2022夏普"]].round(2).T.to_string())
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda v: round(float(v), 3)))
    print("结果：", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    versions = parser.add_mutually_exclusive_group()
    versions.add_argument("--v1", action="store_true", help="运行原始主策略")
    versions.add_argument("--v2", action="store_true", help="运行双端1/5时钟策略")
    versions.add_argument("--benchmark", action="store_true", help="运行纯时间戳差 benchmark")
    versions.add_argument("--compare", action="store_true", help="运行 v1/v2/benchmark 并输出对比图")
    parser.add_argument("--validation-2022", action="store_true")
    parser.add_argument("--grid", action="store_true", help="跑回看窗口的参数面（2016–2022）")
    args = parser.parse_args()
    if args.grid and args.validation_2022:
        parser.error("参数面固定看 2016–2022，不用再加 --validation-2022")
    if args.grid and any((args.v1, args.v2, args.benchmark, args.compare)):
        parser.error("--grid 不能与策略版本参数一起使用")
    if args.grid:
        run_grid()
    elif any((args.v1, args.v2, args.benchmark, args.compare)):
        version = ("compare" if args.compare else "benchmark" if args.benchmark
                   else "v2" if args.v2 else "v1")
        run_versions(version, "validation_2022" if args.validation_2022 else "research")
    else:
        run("validation_2022" if args.validation_2022 else "research")


if __name__ == "__main__":
    main()
