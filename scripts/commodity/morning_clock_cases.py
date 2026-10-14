"""高低点时钟：平均路径、典型正例/反例 K 线，以及时钟和价格位置的分解。只使用 2016–2021 研究期。

    python scripts/commodity/morning_clock_cases.py

样本和 morning_clock_return.py 相同：当年品种池、有夜盘、09:16 时钟和 09:17→11:30 收益都在。
「低减高」= 低点时钟 − 高点时钟。≥ 0.6 表示从夜盘高点一路走到 09:16 附近的低点（下杀），
≤ −0.6 表示一路冲高到 09:16 附近。研究里的关系是：下杀后上午偏反弹，冲高后上午偏回落。

  正例：关系成立，下杀后 09:17→11:30 上涨，或冲高后下跌。
  反例：关系不成立，下杀后继续跌，或冲高后继续涨。

输出在 runs/commodity/morning/clock_cases/：
  paths.png        三个时期的平均路径（以 09:16 收盘为 0）
  hit_rate.png     各年正例占比和按方向收益（hit_rate.csv）
  splits.csv       按时期×方向、09:16 是否仍在延伸、夜盘是否同向、夜盘幅度、板块拆开的按方向收益
    cases_*.png      四类典型个例，每类每年最多抽两天，含成交量子图
  decompose.csv    低减高与价格位置、各段收益的截面排序回归
  yearly_ic.csv    分年截面 IC、固定品种面板 IC 与年度异质性检验
  states.csv       事前状态（过去 20 天开盘移动幅度、共振度）三分组下的截面 IC
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "commodity"))
from tfcta.factors import day_codes_of, raw_price_path
import morning
import morning_clock_return as clock
import morning_features as features

OUT = morning.RUNS / "clock_cases"
SEED = 20261014
WIDE = 0.6      # 平均路径、正例占比用的 |低减高| 门槛，与 clock_return 的分档一致
STRONG = 0.8    # 个例只取路径更单边的日子
CASE_BAND = (0.5, 0.9)  # 个例的 |收益| 落在本类本年的这个分位区间：清楚但不是极端值
CASES_PER_YEAR = 2
PRE_POINTS = 21
PERIODS = {"2016–2017": (2016, 2017), "2018–2020": (2018, 2020), "2021": (2021, 2021)}
PERIOD_COLORS = {"2016–2017": "#2F5D7C", "2018–2020": "#C46B4A", "2021": "#1F7A4D"}
UP, DOWN = "#C0392B", "#1F7A4D"  # 国内习惯：红涨绿跌
CATEGORIES = {
    "下杀_正例": {"side": 1, "win": True, "title": "正例：夜盘一路下杀到 09:16，上午反弹"},
    "冲高_正例": {"side": -1, "win": True, "title": "正例：夜盘一路冲高到 09:16，上午回落"},
    "下杀_反例": {"side": 1, "win": False, "title": "反例：夜盘一路下杀到 09:16，上午继续跌"},
    "冲高_反例": {"side": -1, "win": False, "title": "反例：夜盘一路冲高到 09:16，上午继续涨"},
}


def session_minutes():
    """日盘 09:17 到 15:00 的分钟标签（K 线按结束时刻记）。"""
    spans = [((9, 17), (10, 15)), ((10, 31), (11, 30)), ((13, 31), (15, 0))]
    out = []
    for (h0, m0), (h1, m1) in spans:
        out += [(t // 60, t % 60) for t in range(h0 * 60 + m0, h1 * 60 + m1 + 1)]
    return out


DAY_GRID = session_minutes()
TICKS = ((9, 17), (10, 31), (11, 30), (14, 0), (15, 0))  # 11:30 和 13:31 相邻，只标一个


def pre_path_points(start, decision):
    """夜盘第一根到 09:16 按时钟等分取 PRE_POINTS 个点的位置，0 是第一根，最后一个是 09:16。"""
    return start + np.rint(np.linspace(0, 1, PRE_POINTS) * (decision - start)).astype(int)


def symbol_paths(symbol, wanted):
    """一个品种在 wanted 交易日上的路径和位置特征。价格都相对 09:16 收盘。
    复权价（closew/highw/loww）是加法复权，只能做差；比例一律除以原始价。"""
    frame = features.load_minutes(symbol, "research")
    raw = raw_price_path(frame)
    cw = frame.closew.to_numpy(float)
    hw, lw = frame.highw.to_numpy(float), frame.loww.to_numpy(float)
    idx = frame.index.asi8
    hour = frame.index.hour.to_numpy()
    night = (hour >= 20) | (hour <= 4)
    codes, days = day_codes_of(frame)
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    pos = {int(ts): i for i, ts in enumerate(idx)}
    rows, pre, post = [], [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        day = days[int(codes[s])]
        if day not in wanted or s == 0:
            continue
        d = pos.get(features.stamp(day, features.DECISION))
        day_bars = np.flatnonzero(~night[s:e]) + s
        if d is None or not s <= d < e or not day_bars.size or raw[d] <= 0:
            continue
        f = int(day_bars[0])
        high, low = hw[s:d + 1].max(), lw[s:d + 1].min()
        base = raw[d]
        rows.append({
            "symbol": symbol, "trading_date": day,
            "r_pre": (cw[d] - cw[s - 1]) / raw[s - 1],        # 昨收 → 09:16
            "r_am16": (cw[d] - cw[f - 1]) / raw[s - 1],       # 夜盘收盘 → 09:16
            "loc": (cw[d] - low) / (high - low) if high > low else np.nan,  # 09:16 在夜盘以来区间里的位置
            "rng": (high - low) / raw[s - 1],
        })
        line = (cw[pre_path_points(s, d)] - cw[d]) / base
        pre.append(np.r_[(cw[s - 1] - cw[d]) / base, line])  # 第一个点是昨收
        later = [pos.get(features.stamp(day, hm)) for hm in DAY_GRID]
        post.append(np.array([(cw[i] - cw[d]) / base if i is not None and d < i < e else np.nan for i in later]))
    return pd.DataFrame(rows), np.array(pre).reshape(-1, PRE_POINTS + 1), np.array(post).reshape(-1, len(DAY_GRID))


def build(table):
    parts, pre, post = [], [], []
    for symbol in sorted(set(table.symbol)):
        wanted = set(table.loc[table.symbol == symbol, "trading_date"])
        rows, a, b = symbol_paths(symbol, wanted)
        parts.append(rows)
        pre.append(a)
        post.append(b)
        print(symbol, len(rows), flush=True)
    rows = pd.concat(parts, ignore_index=True)
    rows["row"] = np.arange(len(rows))
    merged = table.merge(rows, on=["symbol", "trading_date"], how="inner", validate="one_to_one")
    merged["diff"] = merged.lclock_td - merged.hclock_td
    merged["year"] = merged.trading_date.dt.year
    merged["period"] = merged.year.map(period_of)
    return merged, np.vstack(pre), np.vstack(post)


def period_of(year):
    return next(name for name, (a, b) in PERIODS.items() if a <= year <= b)


def label_cases(table, threshold=WIDE):
    """side=+1 为下杀（低减高 ≥ 门槛），−1 为冲高；signed 是按预期方向的收益，>0 为正例。收益为 0 的日子不算。"""
    side = np.where(table["diff"] >= threshold, 1, np.where(table["diff"] <= -threshold, -1, 0))
    out = table.assign(side=side, signed=side * table.gross_1130)
    return out[(out.side != 0) & (out.gross_1130 != 0)]


def pick_cases(table, seed=SEED, threshold=STRONG, band=CASE_BAND):
    """四类个例，每类每年最多两天：|低减高| ≥ threshold，|收益| 在本类本年的 band 分位之间，固定种子随机抽。"""
    rng = np.random.default_rng(seed)
    labelled = label_cases(table, threshold)
    chosen = []
    for name, spec in CATEGORIES.items():
        part = labelled[(labelled.side == spec["side"]) & ((labelled.signed > 0) == spec["win"])]
        for year, cell in part.groupby("year", sort=True):
            size = cell.gross_1130.abs()
            lo, hi = size.quantile(band[0]), size.quantile(band[1])
            pool = cell[(size >= lo) & (size <= hi)]
            if pool.empty:
                continue
            indices = rng.choice(len(pool), size=min(CASES_PER_YEAR, len(pool)), replace=False)
            chosen.append(pool.iloc[indices].assign(类别=name))
    return pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()


def hit_rate_table(table):
    """|低减高| ≥ WIDE 的日子里，各年、各板块的正例占比和按方向收益（按交易日聚类的 95% 区间）。"""
    labelled = label_cases(table)
    rows = []
    for (year, sector), part in [((y, "全部"), g) for y, g in labelled.groupby("year")] + \
            [((y, s), g) for (y, s), g in labelled.groupby(["year", "sector"])]:
        mean = clock.bin_row("", "", "", "", part.trading_date.to_numpy(), part.signed.to_numpy())
        rows.append({"年份": int(year), "板块": sector, "笔数": len(part), "正例占比": float((part.signed > 0).mean()),
                     "下杀笔数": int((part.side == 1).sum()), "冲高笔数": int((part.side == -1).sum()),
                     "按方向收益bp": mean["均值bp"], "下限bp": mean["下限bp"], "上限bp": mean["上限bp"]})
    return pd.DataFrame(rows)


def split_table(table):
    """|低减高| ≥ WIDE 的日子按三种方式拆开看按方向收益：时期×方向；09:16 前 15 分钟是否仍在顺着极值方向走；
    夜盘收益是否和这段走势同向。dev15 < 0 在下杀日表示 09:16 仍在创新低（仍在延伸），> 0 表示已经回头。"""
    labelled = label_cases(table).dropna(subset=["dev15"])
    labelled = labelled.assign(
        时期=labelled.year.map(period_of),
        方向=np.where(labelled.side == 1, "下杀", "冲高"),
        到0916=np.where(-labelled.side * labelled.dev15 > 0, "仍在延伸", "已经回头"),
        夜盘=np.where(-labelled.side * labelled.ret_night > 0, "同向", "反向"),
        夜盘幅度=labelled.groupby(["symbol", "year"]).ret_night.transform(  # 品种内、年内按 |夜盘收益| 三分，避免变成品种波动差异
            lambda s: pd.qcut(s.abs().rank(method="first"), 3, labels=["小", "中", "大"])).astype(str))
    rows = []
    for split in (["时期", "方向"], ["到0916"], ["到0916", "year"], ["夜盘"], ["夜盘", "year"], ["夜盘幅度"], ["夜盘幅度", "year"], ["sector"], ["sector", "year"]):
        for key, part in labelled.groupby(split):
            key = key if isinstance(key, tuple) else (key,)
            mean = clock.bin_row("", "", "", "", part.trading_date.to_numpy(), part.signed.to_numpy())
            rows.append({"拆分": "×".join(split), "组": " ".join(str(k) for k in key), "笔数": len(part),
                         "正例占比": float((part.signed > 0).mean()), "按方向收益bp": mean["均值bp"],
                         "下限bp": mean["下限bp"], "上限bp": mean["上限bp"]})
    return pd.DataFrame(rows)


def xs_rank(frame, column):
    """当天截面百分位排名减 0.5，去掉“这一天整体涨跌”的成分。"""
    return frame.groupby("trading_date")[column].rank(pct=True) - 0.5


def rank_regression(frame, columns, target="gross_1130"):
    """截面排名回归，按交易日聚类的标准误。返回每个变量的系数和 t。"""
    data = frame.dropna(subset=list(columns) + [target])
    y = xs_rank(data, target).to_numpy()
    x = np.column_stack([np.ones(len(data))] + [xs_rank(data, c).to_numpy() for c in columns])
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    score = pd.DataFrame(x * (y - x @ beta)[:, None]).groupby(pd.factorize(data.trading_date)[0]).sum().to_numpy()
    groups = len(score)
    cov = bread @ (score.T @ score) @ bread * groups / (groups - 1)
    se = np.sqrt(np.diag(cov))
    return {c: (float(beta[i + 1]), float(beta[i + 1] / se[i + 1])) for i, c in enumerate(columns)}


DECOMPOSE_MODELS = [
    ("低减高", ["diff"]),
    ("09:16 区间位置", ["loc"]),
    ("昨收→09:16", ["r_pre"]),
    ("夜盘 + 早盘前16分钟", ["ret_night", "r_am16"]),
    ("低减高 + 区间位置", ["diff", "loc"]),
    ("低减高 + 昨收→09:16", ["diff", "r_pre"]),
    ("全部", ["diff", "ret_night", "r_am16", "loc"]),
]
NAMES = {"diff": "低减高", "loc": "09:16 区间位置", "r_pre": "昨收→09:16", "ret_night": "夜盘收益",
         "r_am16": "夜盘收盘→09:16"}


def decompose(table):
    rows = []
    for model, columns in DECOMPOSE_MODELS:
        for column, (coef, t_stat) in rank_regression(table, columns).items():
            rows.append({"模型": model, "变量": NAMES[column], "系数": coef, "聚类t": t_stat})
    corr = table[list(NAMES)].rename(columns=NAMES).corr("spearman")
    return pd.DataFrame(rows), corr


def spread_ic(frame):
    ic = clock.daily_ic(frame)
    return ic[ic.关系 == "低点时钟-高点时钟 → 多头收益"].set_index("trading_date").IC


def heterogeneity(daily):
    """各年 IC 均值是否相同：逆方差加权的 Cochran Q。"""
    g = daily.groupby(daily.index.year).agg(["mean", "std", "count"])
    weight = g["count"] / g["std"] ** 2
    pooled = (weight * g["mean"]).sum() / weight.sum()
    q = float((weight * (g["mean"] - pooled) ** 2).sum())
    return q, float(stats.chi2.sf(q, len(g) - 1))


def yearly_ic_table(table):
    full = spread_ic(table)
    years = table.groupby("symbol").year.nunique()
    panel = years[years == table.year.nunique()].index
    fixed = spread_ic(table[table.symbol.isin(panel)])
    rows = []
    for name, series in (("全部品种", full), (f"固定面板{len(panel)}个品种", fixed)):
        for year, part in series.groupby(series.index.year):
            rows.append({"样本": name, "年份": int(year), "IC均值": part.mean(),
                         "t": part.mean() / part.std() * np.sqrt(len(part)), "天数": len(part)})
        q, p = heterogeneity(series)
        rows.append({"样本": name, "年份": "Q检验", "IC均值": q, "t": p, "天数": len(series)})
    sector = (table.groupby(["sector", "year"])
              .apply(lambda d: stats.spearmanr(d["diff"], d.gross_1130).correlation, include_groups=False)
              .unstack())
    return pd.DataFrame(rows), sector, full, sorted(panel)


def state_table(table, daily):
    """事前状态：过去 20 个交易日（不含当天）的平均 |昨收→09:16| 和共振度 |均值|/均值|·|，三分组看当天 IC。"""
    day = table.groupby("trading_date").r_pre.agg(
        开盘移动幅度=lambda s: s.abs().mean(), 共振度=lambda s: abs(s.mean()) / s.abs().mean())
    day = day.join(daily.rename("IC"), how="inner")
    rows = []
    for column in ("开盘移动幅度", "共振度"):
        state = day[column].rolling(20, min_periods=15).mean().shift(1)
        for scheme, group in (("全期三分", pd.qcut(state, 3, labels=False)),
                              ("年内三分", state.groupby(state.index.year).transform(
                                  lambda s: pd.qcut(s, 3, labels=False)))):
            frame = day.assign(档=group).dropna(subset=["档"])
            for level, part in frame.groupby("档"):
                rows.append({"状态": column, "分组": scheme, "档": ["低", "中", "高"][int(level)], "年份": "全期",
                             "IC均值": part.IC.mean(), "t": part.IC.mean() / part.IC.std() * np.sqrt(len(part)),
                             "天数": len(part)})
                for year, cell in part.groupby(part.index.year):
                    rows.append({"状态": column, "分组": scheme, "档": ["低", "中", "高"][int(level)],
                                 "年份": int(year), "IC均值": cell.IC.mean(), "t": np.nan, "天数": len(cell)})
    monthly = daily.resample("ME").mean().dropna()
    return pd.DataFrame(rows), float(monthly.autocorr(1))


def plot_paths(table, pre, post, path):
    """两行：下杀日、冲高日。左栏是昨收和夜盘到 09:16（按时钟），右栏是 09:16 之后的日盘（按分钟）。"""
    clock.apply_style()
    labelled = label_cases(table)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2), gridspec_kw={"width_ratios": [1, 2.3]}, sharey="row")
    x_pre = np.r_[-0.18, np.linspace(0, 1, PRE_POINTS)]
    x_day = np.arange(len(DAY_GRID))
    ticks = [i for i, hm in enumerate(DAY_GRID) if hm in TICKS]
    exit_at = DAY_GRID.index((11, 30))
    for row, (side, name) in enumerate(((1, "低减高 ≥ 0.6：夜盘一路下杀到 09:16"), (-1, "低减高 ≤ -0.6：夜盘一路冲高到 09:16"))):
        left, right = axes[row]
        for period, color in PERIOD_COLORS.items():
            part = labelled[(labelled.side == side) & (labelled.period == period)]
            rows = part.row.to_numpy()
            win = float((part.signed > 0).mean())
            label = f"{period}  {len(part):,} 日  正例 {win:.1%}"
            left.plot(x_pre, np.nanmean(pre[rows], axis=0) * 1e4, color=color, lw=2, label=label)
            right.plot(x_day, np.nanmean(post[rows], axis=0) * 1e4, color=color, lw=2, label=label)
        for ax in (left, right):
            ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
            ax.grid(axis="y", zorder=0)
            ax.tick_params(length=0, labelsize=9)
        left.set_xticks([-0.18, 0, 0.5, 1], ["昨收", "夜盘开", "时钟 0.5", "09:16"])
        right.set_xticks(ticks, [f"{DAY_GRID[i][0]:02d}:{DAY_GRID[i][1]:02d}" for i in ticks])
        right.axvspan(0, exit_at, color="#F0F4F8", zorder=0)
        right.text(exit_at / 2, 0.02, "持仓 09:17 → 11:30", transform=right.get_xaxis_transform(),
                   ha="center", fontsize=9, color="#7B8794")
        left.set_ylabel("相对 09:16 收盘（bp）")
        left.set_title(name, loc="left", fontsize=12, color="#102A43")
        right.legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle("2016–2021  平均路径：极值出得晚的日子，09:16 之后怎么走", fontsize=15, color="#102A43",
                 x=0.06, ha="left", y=0.985)
    fig.text(0.06, 0.94, "所有品种日等权平均，未扣成本；每个品种日以自己 09:16 收盘为 0。正例指 09:17→11:30 收益与预期方向一致"
             "（下杀后涨、冲高后跌）。", fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.07, wspace=0.05, hspace=0.3)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_day_paths(table, post, path):
    """只看 09:16 之后：按预期方向翻转后的平均收益（下杀日不变，冲高日取负），三个时期对比。"""
    clock.apply_style()
    labelled = label_cases(table)
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    x_day = np.arange(len(DAY_GRID))
    ticks = [i for i, hm in enumerate(DAY_GRID) if hm in TICKS]
    exit_at = DAY_GRID.index((11, 30))
    ax.axvspan(0, exit_at, color="#F0F4F8", zorder=0)
    for period, color in PERIOD_COLORS.items():
        part = labelled[labelled.period == period]
        line = np.nanmean(post[part.row.to_numpy()] * part.side.to_numpy()[:, None], axis=0) * 1e4
        ax.plot(x_day, line, color=color, lw=2,
                label=f"{period}  {len(part):,} 日  11:30 {line[exit_at]:+.1f} bp  15:00 {line[-1]:+.1f} bp")
    ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
    ax.set_xticks(ticks, [f"{DAY_GRID[i][0]:02d}:{DAY_GRID[i][1]:02d}" for i in ticks])
    ax.set_ylabel("按预期方向的累计收益（bp）")
    ax.grid(axis="y", zorder=0)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle("2016–2021  |低减高| ≥ 0.6 的日子，09:16 之后按预期方向的平均收益", fontsize=14, color="#102A43",
                 x=0.07, ha="left", y=0.98)
    fig.text(0.07, 0.915, "下杀日做多、冲高日做空，以 09:16 收盘为 0，未扣成本。向上表示反转，向下表示延续。阴影是 09:17→11:30 持仓段。",
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.1)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_hit_rate(hits, path):
    clock.apply_style()
    whole = hits[hits.板块 == "全部"].sort_values("年份")
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = whole.年份.to_numpy()
    left.bar(x, whole.正例占比 * 100 - 50, bottom=50, color="#2F5D7C", width=0.6, zorder=2)
    left.axhline(50, color="#9AA5B1", lw=0.8)
    for xi, value, n in zip(x, whole.正例占比, whole.笔数):
        left.text(xi, value * 100 + (0.15 if value >= 0.5 else -0.15), f"{value:.1%}\n{n:,} 日",
                  ha="center", va="bottom" if value >= 0.5 else "top", fontsize=8.5, color="#243B53")
    reach = max(abs(whole.正例占比 * 100 - 50).max() + 1.5, 3)
    left.set_ylim(50 - reach, 50 + reach)
    left.set_ylabel("正例占比（%）")
    left.set_title("正例占比", loc="left", fontsize=12, color="#102A43")
    mean = whole.按方向收益bp.to_numpy()
    right.bar(x, mean, color=np.where(mean >= 0, "#1F7A4D", "#B23A48"), width=0.6, zorder=2)
    right.errorbar(x, mean, yerr=[mean - whole.下限bp, whole.上限bp - mean], fmt="none", ecolor="#52606D",
                   elinewidth=1, capsize=3, zorder=3)
    right.axhline(0, color="#9AA5B1", lw=0.8)
    right.set_ylabel("按方向的 09:17→11:30 收益（bp）")
    right.set_title("按方向收益，竖线为按交易日聚类的 95% 区间", loc="left", fontsize=12, color="#102A43")
    for ax in (left, right):
        ax.set_xticks(x)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    fig.suptitle("2016–2021  |低减高| ≥ 0.6：下杀后做多、冲高后做空", fontsize=14, color="#102A43",
                 x=0.07, ha="left", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.1, wspace=0.25)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def candles(frame, start, end):
    """原始价 5 分钟 K 线，切点对齐使 09:16 恰好是一根的结束。原始高低 = 复权高低 + (原始收 − 复权收)。"""
    part = frame.iloc[start:end]
    raw = raw_price_path(part)
    shift = raw - part.closew.to_numpy(float)
    bars = pd.DataFrame({"open": part.open.to_numpy(float), "high": part.highw.to_numpy(float) + shift,
                         "low": part.loww.to_numpy(float) + shift, "close": raw,
                         "volume": part.volume.to_numpy(float)}, index=part.index)
    out = bars.resample("5min", closed="right", label="right", origin="epoch", offset="1min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna()


def draw_case(ax, volume_ax, frame, case):
    raw_close = raw_price_path(frame)
    hw, lw = frame.highw.to_numpy(float), frame.loww.to_numpy(float)
    shift = raw_close - frame.closew.to_numpy(float)
    codes, days = day_codes_of(frame)
    day = case.trading_date
    rows = np.flatnonzero(days[codes] == day)
    s, e = int(rows[0]), int(rows[-1]) + 1
    d = int(frame.index.get_loc(pd.Timestamp(features.stamp(day, features.DECISION))))
    bars = candles(frame, s, e)
    x = np.arange(len(bars))
    color = np.where(bars.close >= bars.open, UP, DOWN)
    ax.vlines(x, bars.low, bars.high, color=color, lw=0.7, zorder=2)
    body_low = np.minimum(bars.open, bars.close)
    body = np.maximum(np.abs(bars.close - bars.open), (bars.high - bars.low).max() * 0.004)
    ax.bar(x, body, bottom=body_low, width=0.7, color=color, linewidth=0, zorder=3)
    times = bars.index
    cut = int(np.searchsorted(times, frame.index[d]))       # 09:16 所在那根
    exit_at = int(np.searchsorted(times, pd.Timestamp(features.stamp(day, features.EXIT))))
    ax.axvspan(cut + 0.5, min(exit_at, len(x) - 1) + 0.5, color="#F0F4F8", zorder=0)
    ax.axvline(cut + 0.5, color="#52606D", lw=0.9, ls="--", zorder=1)
    volume_ax.axvspan(cut + 0.5, min(exit_at, len(x) - 1) + 0.5, color="#F0F4F8", zorder=0)
    volume_ax.axvline(cut + 0.5, color="#52606D", lw=0.9, ls="--", zorder=1)
    hi = s + int(np.argmax(hw[s:d + 1]))
    lo = s + int(np.argmin(lw[s:d + 1]))
    span = float(bars.high.max() - bars.low.min()) or 1.0
    for at, price, marker, offset in ((hi, hw[hi] + shift[hi], "v", 0.04), (lo, lw[lo] + shift[lo], "^", -0.04)):
        k = int(np.searchsorted(times, frame.index[at]))
        ax.scatter([k], [price + offset * span], marker=marker, s=46, color="#102A43", zorder=4)
    gaps = np.flatnonzero(np.diff(times.asi8) > 5 * features.MINUTE) + 1
    ticks = sorted(set([0, *gaps.tolist()]))
    volume_ax.bar(x, bars.volume, width=0.7, color=color, linewidth=0, zorder=2)
    ax.set_xticks(ticks)
    ax.tick_params(labelbottom=False)
    volume_ax.set_xticks(ticks, [times[i].strftime("%H:%M") for i in ticks], fontsize=7)
    ax.set_xlim(-1, len(x))
    volume_ax.set_xlim(-1, len(x))
    ax.tick_params(length=0, labelsize=7.5)
    volume_ax.tick_params(length=0, labelsize=7)
    ax.grid(axis="y", zorder=0)
    volume_ax.grid(axis="y", zorder=0)
    volume_ax.set_ylabel("量", fontsize=7, rotation=0, labelpad=8)
    volume_ax.set_ylim(bottom=0)
    ax.set_title(f"{case.year}  {case.symbol}（{case.sector}）{day:%Y-%m-%d}\n"
                 f"低减高 {case['diff']:+.2f}   夜盘 {case.ret_night * 1e4:+.0f}bp   "
                 f"dev15 {case.dev15 * 1e4:+.0f}bp   上午 {case.gross_1130 * 1e4:+.0f}bp",
                 loc="left", fontsize=9.5, color="#102A43")


def plot_cases(cases, path_of):
    clock.apply_style()
    frames = {symbol: features.load_minutes(symbol, "research") for symbol in sorted(set(cases.symbol))}
    for name, spec in CATEGORIES.items():
        part = cases[cases.类别 == name].sort_values("year")
        if part.empty:
            continue
        fig, axes = plt.subplots(8, 3, figsize=(18, 15),
                                 gridspec_kw={"height_ratios": [3.2, 0.9] * 4})
        for index, (_, case) in enumerate(part.iterrows()):
            row, column = divmod(index, 3)
            draw_case(axes[row * 2, column], axes[row * 2 + 1, column], frames[case.symbol], case)
        for index in range(len(part), 12):
            row, column = divmod(index, 3)
            axes[row * 2, column].axis("off")
            axes[row * 2 + 1, column].axis("off")
        fig.suptitle(f"2016–2021  {spec['title']}", fontsize=15, color="#102A43", x=0.05, ha="left", y=0.985)
        fig.text(0.05, 0.945,
                 f"每类每年最多抽 {CASES_PER_YEAR} 天：|低减高| ≥ {STRONG}，上午收益位于本类本年 {CASE_BAND[0]:.0%}–{CASE_BAND[1]:.0%} 分位；固定种子。"
                 "原始价 5 分钟 K 线及成交量，红涨绿跌；▼ 夜盘以来高点，▲ 低点（到 09:16 为止）；虚线是 09:16，阴影是 09:17→11:30 持仓。",
                 fontsize=9, color="#7B8794")
        fig.subplots_adjust(left=0.05, right=0.99, top=0.87, bottom=0.05, hspace=0.6, wspace=0.18)
        fig.savefig(path_of(name), dpi=150, facecolor="white")
        plt.close(fig)


def main():
    table = clock.load_research()
    merged, pre, post = build(table)
    if len(merged) != len(table):
        print(f"路径缺 {len(table) - len(merged)} 个品种日（09:16 或昨收缺失）")
    OUT.mkdir(parents=True, exist_ok=True)
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)

    hits = hit_rate_table(merged)
    hits.to_csv(OUT / "hit_rate.csv", index=False, encoding="utf-8-sig")
    splits = split_table(merged)
    splits.to_csv(OUT / "splits.csv", index=False, encoding="utf-8-sig")
    plot_paths(merged, pre, post, plots / "paths.png")
    plot_day_paths(merged, post, plots / "paths_day.png")
    plot_hit_rate(hits, plots / "hit_rate.png")
    cases = pick_cases(merged)
    cases[["类别", "year", "symbol", "sector", "trading_date", "diff", "ret_night", "dev15", "gross_1130", "loc"]].to_csv(
        OUT / "cases.csv", index=False, encoding="utf-8-sig")
    plot_cases(cases, lambda name: plots / f"cases_{name}.png")

    reg, corr = decompose(merged)
    reg.to_csv(OUT / "decompose.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUT / "decompose_corr.csv", encoding="utf-8-sig")
    yearly, sector, daily, panel = yearly_ic_table(merged)
    yearly.to_csv(OUT / "yearly_ic.csv", index=False, encoding="utf-8-sig")
    sector.to_csv(OUT / "sector_year_spearman.csv", encoding="utf-8-sig")
    states, ar1 = state_table(merged, daily)
    states.to_csv(OUT / "states.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(corr.round(3).to_string())
    print(reg.round(4).to_string(index=False))
    print(yearly.round(4).to_string(index=False))
    print("固定面板:", " ".join(panel))
    print(sector.round(3).to_string())
    print(states[states.年份 == "全期"].round(4).to_string(index=False))
    print(f"月度 IC 一阶自相关 {ar1:.3f}")
    print(hits[hits.板块 == "全部"].round(4).to_string(index=False))
    print(splits.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
