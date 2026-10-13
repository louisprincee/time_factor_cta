"""高低点出现位置与 09:17 之后持仓收益的关系。只使用 2016–2021 研究期。

    python scripts/morning_clock_return.py
    python scripts/morning_clock_return.py --exit 1500
    python scripts/morning_clock_return.py --balance        # 涨跌各半重采样，输出到 clock_return_balanced/

时钟是本交易日夜盘第一根到 09:16 的相对位置（0 为夜盘开头，1 为 09:16）。
默认收益是同一交易日 09:17 开盘到 11:30 收盘的多头收益。--exit 1500 改成持有到 15:00 收盘。
夜盘在数据里记在当天的 trading_date 上，所以这里的「夜盘之后的日盘」就是当天。

三类位置关系都使用全部有效夜盘品种日，不按 dev15 正负筛选，也不翻转收益：高点时钟、低点时钟、低点时钟减高点时钟，分别与同一段多头收益比较。

显著性用按交易日聚类的 t 检验，同一天的品种不算独立样本。
输出在 runs/morning/clock_return/。
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tfcta import config as C
from tfcta.data import load_universe
import morning

OUT = morning.RUNS / "clock_return"
CLOCK = 2.0 / 3.0
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"]
DIFF_BIN_EDGES = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
DIFF_BIN_LABELS = ["-1 to -0.6", "-0.6 to -0.2", "-0.2 to 0.2", "0.2 to 0.6", "0.6 to 1"]
SEGMENT_LABELS = ["最早\n第1段", "偏早\n第2段", "中间\n第3段", "偏晚\n第4段", "最晚\n第5段"]
OLD_PLOTS = ("bins.png", "yearly.png", "samples.png", "year_sector.png", "by_symbol.png")


def cluster_slope(day, x, y):
    """y 对 x 的斜率。标准误按交易日聚类；交易日少于 3 天时不给 t。"""
    day = np.asarray(day)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    day, x, y = day[keep], x[keep], y[keep]
    empty = {"笔数": int(len(x)), "天数": int(pd.unique(day).size) if len(day) else 0,
             "斜率": np.nan, "标准误": np.nan, "t": np.nan, "双尾p": np.nan}
    if len(x) < 3 or np.unique(x).size < 2:
        return empty
    design = np.column_stack([np.ones(len(x)), x])
    try:
        bread = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return empty
    beta = bread @ (design.T @ y)
    resid = y - design @ beta
    order = np.argsort(day, kind="mergesort")
    day, design, resid = day[order], design[order], resid[order]
    _, starts, counts = np.unique(day, return_index=True, return_counts=True)
    meat = np.zeros((2, 2))
    for start, count in zip(starts, counts):
        score = design[start:start + count].T @ resid[start:start + count]
        meat += np.outer(score, score)
    groups = int(len(starts))
    if groups < 3:
        empty.update({"斜率": float(beta[1]), "天数": groups})
        return empty
    scale = (groups / (groups - 1)) * ((len(x) - 1) / (len(x) - 2))
    variance = float((bread @ meat @ bread * scale)[1, 1])
    se = np.sqrt(variance) if variance > 0 else np.nan
    t_stat = beta[1] / se if np.isfinite(se) and se > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(t_stat), groups - 1)) if np.isfinite(t_stat) else np.nan
    return {"笔数": int(len(x)), "天数": groups, "斜率": float(beta[1]), "标准误": float(se),
            "t": float(t_stat), "双尾p": p_value}


def association(day, x, y):
    """Spearman，以及原始收益对时钟的聚类斜率。斜率的单位随后再换成基点。"""
    day = np.asarray(day)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    day, x, y = day[keep], x[keep], y[keep]
    spearman = stats.spearmanr(x, y).correlation if len(x) >= 3 and np.unique(x).size > 1 else np.nan
    ranked = cluster_slope(day, stats.rankdata(x), stats.rankdata(y)) if len(x) else cluster_slope(day, x, y)
    raw = cluster_slope(day, x, y)
    return {"笔数": raw["笔数"], "天数": raw["天数"], "Spearman": float(spearman) if np.isfinite(spearman) else np.nan,
            "Spearman_t": ranked["t"], "Spearman_p": ranked["双尾p"],
            "斜率": raw["斜率"], "标准误": raw["标准误"], "斜率_t": raw["t"], "斜率_p": raw["双尾p"]}


def late_minus_early(day, clock, y, low=0.0, high=1.0):
    """取值区间 [low, high] 的上三分之一减下三分之一的平均收益，中间三分之一不进入对比。
    时钟是 [0, 1]：≥ 2/3 减 ≤ 1/3；低点时钟减高点时钟是 [−1, 1]：≥ 1/3 减 ≤ −1/3。"""
    clock = np.asarray(clock, dtype=float)
    y = np.asarray(y, dtype=float)
    day = np.asarray(day)
    lower = low + (high - low) * (1 - CLOCK)
    upper = low + (high - low) * CLOCK
    keep = np.isfinite(clock) & np.isfinite(y) & ((clock <= lower) | (clock >= upper))
    return cluster_slope(day[keep], (clock[keep] >= upper).astype(float), y[keep])


def describe(sample, scope, item):
    link = association(item["day"], item["x"], item["y"])
    span = item.get("range", (0.0, 1.0))
    gap = late_minus_early(item["day"], item["x"], item["y"], *span)
    crit = stats.t.ppf(0.975, link["天数"] - 1) if link["天数"] >= 3 else np.nan
    return {
        "样本": sample, "范围": scope, "关系": item["关系"], "预期方向": item["预期"],
        "笔数": link["笔数"], "天数": link["天数"],
        "Spearman": link["Spearman"], "Spearman聚类t": link["Spearman_t"], "Spearman双尾p": link["Spearman_p"],
        "时钟0到1的收益bp": link["斜率"] * 1e4, "斜率聚类t": link["斜率_t"], "斜率双尾p": link["斜率_p"],
        "斜率bp下限": (link["斜率"] - crit * link["标准误"]) * 1e4,
        "斜率bp上限": (link["斜率"] + crit * link["标准误"]) * 1e4,
        "晚减早bp": gap["斜率"] * 1e4, "晚减早t": gap["t"], "晚减早p": gap["双尾p"],
    }


def bin_means(sample, scope, item):
    clock = pd.Series(np.asarray(item["x"], dtype=float))
    group = pd.cut(clock, BIN_EDGES, include_lowest=True, labels=BIN_LABELS)
    day = np.asarray(item["day"])
    y = np.asarray(item["y"], dtype=float)
    return [bin_row(sample, scope, item["关系"], label, day[(group == label).to_numpy()], y[(group == label).to_numpy()])
            for label in BIN_LABELS]


def difference_bin_means(sample, scope, item):
    difference = pd.Series(np.asarray(item["x"], dtype=float))
    group = pd.cut(difference, DIFF_BIN_EDGES, include_lowest=True, labels=DIFF_BIN_LABELS)
    day = np.asarray(item["day"])
    y = np.asarray(item["y"], dtype=float)
    return [bin_row(sample, scope, item["关系"], label, day[(group == label).to_numpy()], y[(group == label).to_numpy()])
            for label in DIFF_BIN_LABELS]


def bin_row(sample, scope, name, label, day, y):
    """一个时钟区间里，收益均值和按日聚类的 95% 区间。"""
    day = np.asarray(day)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(y)
    day, y = day[keep], y[keep]
    if len(y) == 0:
        return {"样本": sample, "范围": scope, "关系": name, "时钟区间": label, "笔数": 0, "天数": 0,
                "均值bp": np.nan, "下限bp": np.nan, "上限bp": np.nan}
    order = np.argsort(day, kind="mergesort")
    day, y = day[order], y[order]
    _, starts, counts = np.unique(day, return_index=True, return_counts=True)
    groups = int(len(starts))
    mean = float(y.mean())
    if groups < 3:
        return {"样本": sample, "范围": scope, "关系": name, "时钟区间": label,
                "笔数": int(len(y)), "天数": groups, "均值bp": mean * 1e4, "下限bp": np.nan, "上限bp": np.nan}
    meat = 0.0
    for start, count in zip(starts, counts):
        meat += float(y[start:start + count].sum() - mean * count) ** 2
    se = np.sqrt(meat / len(y) ** 2 * (groups / (groups - 1)))
    crit = stats.t.ppf(0.975, groups - 1)
    return {"样本": sample, "范围": scope, "关系": name, "时钟区间": label, "笔数": int(len(y)), "天数": groups,
            "均值bp": mean * 1e4, "下限bp": (mean - crit * se) * 1e4, "上限bp": (mean + crit * se) * 1e4}


def attach_holding_gross(table, exit_hm, column):
    """按品种补上 09:17 开盘到指定收盘的多头收益。"""
    import morning_features as features
    parts = []
    for symbol in sorted(set(table.symbol)):
        gross = features.holding_gross(symbol, "research", exit_hm)
        gross["trading_date"] = pd.to_datetime(gross.trading_date)
        parts.append(gross)
        print(symbol, int(gross.gross.notna().sum()), flush=True)
    held = pd.concat(parts, ignore_index=True)
    merged = table.merge(held.rename(columns={"gross": column}), on=["symbol", "trading_date"], how="left")
    if merged.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("持仓收益并回特征表后品种/日期重复")
    return merged


def load_research(return_col="gross_1130", exit_hm=None):
    """研究期、当年品种池、有夜盘，并且 09:16 时钟和指定持仓收益都在。"""
    table = pd.read_parquet(morning.FEATURES / "research.parquet")
    table["trading_date"] = pd.to_datetime(table.trading_date)
    C.assert_no_holdout_dates(table.trading_date, "高低点与收益")
    table = table[table.trading_date.dt.year.between(2016, 2021)].copy()
    if return_col not in table.columns:
        if exit_hm is None:
            raise ValueError(f"特征表没有 {return_col}")
        table = attach_holding_gross(table, exit_hm, return_col)
    pools = load_universe()
    table["member"] = [symbol in pools.get(day.year, ()) for symbol, day in zip(table.symbol, table.trading_date)]
    unknown = set(table.symbol) - set(morning.SECTOR_OF)
    if unknown:
        raise ValueError(f"未登记板块: {sorted(unknown)}")
    table["sector"] = table.symbol.map(morning.SECTOR_OF)
    ready = (table.member & table.ret_night.notna() & table.hclock_td.notna() & table.lclock_td.notna()
             & table[return_col].notna())
    table = table.loc[ready].reset_index(drop=True)
    if table.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("特征表品种/日期重复")
    return table


def samples_of(table):
    return {"有夜盘": table}


def collect(table, return_col="gross_1130"):
    relations_rows, bin_rows, year_rows = [], [], []
    for sample, frame in samples_of(table).items():
        groups = [("全部", frame)] + [(name, frame[frame.sector == name]) for name in morning.SECTORS]
        for scope, part in groups:
            for item in study_items(part, return_col):
                relations_rows.append(describe(sample, scope, item))
                if scope == "全部" and item["关系"] != "低点时钟-高点时钟 → 多头收益":
                    bin_rows.extend(bin_means(sample, scope, item))
            year_rows.extend(yearly_pooled_summary(part, sample, scope, return_col).to_dict("records"))
    return pd.DataFrame(relations_rows), pd.DataFrame(bin_rows), pd.DataFrame(year_rows)


def position_items(frame, return_col="gross_1130"):
    """每个有夜盘的交易日：高点、低点各自在夜盘开盘到 09:16 的位置，对持仓做多收益。"""
    day = frame.trading_date.to_numpy()
    gross = frame[return_col].to_numpy(dtype=float)
    return [
        {"关系": "高点", "day": day, "x": frame.hclock_td.to_numpy(dtype=float), "y": gross},
        {"关系": "低点", "day": day, "x": frame.lclock_td.to_numpy(dtype=float), "y": gross},
    ]


def study_items(frame, return_col="gross_1130"):
    """位置变量与原始多头收益的关系，不按偏离方向拆样。"""
    high, low = position_items(frame, return_col)
    day = frame.trading_date.to_numpy()
    gross = frame[return_col].to_numpy(dtype=float)
    return [
        {**high, "关系": "高点时钟 → 多头收益", "预期": "负"},
        {**low, "关系": "低点时钟 → 多头收益", "预期": "正"},
        {"关系": "低点时钟-高点时钟 → 多头收益", "预期": "正", "day": day,
         "x": low["x"] - high["x"], "y": gross, "range": (-1.0, 1.0)},
    ]


def yearly_pooled_summary(frame, sample, scope, return_col="gross_1130"):
    """每个板块、每年合并所有品种日后计算 Spearman。"""
    rows = []
    for year, annual in frame.groupby(frame.trading_date.dt.year, sort=True):
        for item in study_items(annual, return_col):
            link = association(item["day"], item["x"], item["y"])
            rows.append({"样本": sample, "范围": scope, "年份": int(year), "关系": item["关系"],
                         "Spearman": link["Spearman"], "Spearman双尾p": link["Spearman_p"],
                         "笔数": link["笔数"], "天数": link["天数"]})
    return pd.DataFrame(rows)


def segment_table(frame, scope="全部", return_col="gross_1130"):
    rows = []
    summaries = []
    items = position_items(frame, return_col)
    items.append(study_items(frame, return_col)[2])
    for item in items:
        if item["关系"] == "低点时钟-高点时钟 → 多头收益":
            rows.extend(difference_bin_means("有夜盘", scope, item))
        else:
            rows.extend(bin_means("有夜盘", scope, item))
        link = association(item["day"], item["x"], item["y"])
        summaries.append({"关系": item["关系"], **link, "时钟0到1的收益bp": link["斜率"] * 1e4})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def sector_table(frame, return_col="gross_1130"):
    rows = []
    for sector in morning.SECTORS:
        part = frame[frame.sector == sector]
        if part.empty:
            continue
        for item in position_items(part, return_col):
            link = association(item["day"], item["x"], item["y"])
            rows.append({"板块": sector, "品种数": int(part.symbol.nunique()), "关系": item["关系"],
                         "Spearman": link["Spearman"], "Spearman双尾p": link["Spearman_p"],
                         "笔数": link["笔数"], "天数": link["天数"],
                         "时钟0到1的收益bp": link["斜率"] * 1e4, "斜率双尾p": link["斜率_p"]})
    return pd.DataFrame(rows)


def use_chinese_font():
    from matplotlib import font_manager
    have = {font.name for font in font_manager.fontManager.ttflist}
    chosen = [name for name in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "SimHei", "Noto Sans CJK SC")
              if name in have]
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def apply_style():
    use_chinese_font()
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E4E7EB",
        "axes.labelcolor": "#243B53",
        "axes.titlecolor": "#102A43",
        "xtick.color": "#52606D",
        "ytick.color": "#52606D",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#E4E7EB",
        "grid.linewidth": 0.7,
    })


def _p_text(p_value):
    if not np.isfinite(p_value):
        return "p —"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def plot_segments(segments, summary, path, scope="全部", hold_label="09:17 → 11:30"):
    """高点、低点位置及两者时钟差分档后的持仓做多收益。"""
    apply_style()
    names = ("高点", "低点", "低点时钟-高点时钟 → 多头收益")
    labels = {"高点": BIN_LABELS, "低点": BIN_LABELS, "低点时钟-高点时钟 → 多头收益": DIFF_BIN_LABELS}
    colors = {"高点": "#2F5D7C", "低点": "#C46B4A", "低点时钟-高点时钟 → 多头收益": "#1F7A4D"}
    titles = {"高点": "高点出现的位置", "低点": "低点出现的位置",
              "低点时钟-高点时钟 → 多头收益": "低点时钟 - 高点时钟"}
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.7), sharey=True)
    for ax, name in zip(axes, names):
        bin_labels = labels[name]
        part = segments[segments.关系 == name].set_index("时钟区间").loc[bin_labels]
        values = part.均值bp.to_numpy()
        low = part.下限bp.to_numpy()
        high = part.上限bp.to_numpy()
        positions = np.arange(len(bin_labels))
        ax.bar(positions, values, color=colors[name], width=0.66, zorder=2)
        ax.errorbar(positions, values, yerr=[values - low, high - values], fmt="none",
                    ecolor=colors[name], elinewidth=1.0, capsize=3.5, capthick=1.0, zorder=3)
        ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
        if name == names[2]:
            segment_names = ["差值最小", "偏小", "接近 0", "偏大", "差值最大"]
            ticks = [f"{segment}\n{label}\n{int(count):,}"
                     for segment, label, count in zip(segment_names, bin_labels, part.笔数)]
        else:
            ticks = [f"{label}\n{int(count):,}" for label, count in zip(SEGMENT_LABELS, part.笔数)]
        ax.set_xticks(positions, ticks)
        ax.grid(axis="y", zorder=0)
        ax.tick_params(length=0, labelsize=9)
        ax.set_axisbelow(True)
        link = summary[summary.关系 == name].iloc[0]
        ax.set_title(f"{titles[name]}\nSpearman {link.Spearman:+.3f}    {_p_text(link.Spearman_p)}",
                     loc="left", fontsize=12, color="#102A43", pad=8)
    axes[0].set_ylabel(f"{hold_label} 做多收益（bp）")
    scope_label = "全板块" if scope == "全部" else scope
    fig.suptitle(f"2016–2021  {scope_label}  每个交易日  {hold_label}", fontsize=15,
                 color="#102A43", x=0.07, ha="left", y=0.98)
    fig.text(0.07, 0.92,
             "竖线是按交易日聚类的 95% 区间。横轴数字是该段的品种日数量。位置从夜盘第一根到 09:16，含 09:16 这根已知 K 线。",
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.16, wspace=0.22)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def plot_sectors(sectors, path, hold_label="09:17 → 11:30"):
    """每个板块自己的 Spearman。两栏板块顺序相同。"""
    apply_style()
    order = [name for name in morning.SECTORS if name in set(sectors.板块)]
    ranked = sectors.set_index(["关系", "板块"])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.8), sharex=True)
    titles = {"高点": "高点位置  →  做多收益", "低点": "低点位置  →  做多收益"}
    y = np.arange(len(order))[::-1]
    reach = max(abs(float(sectors.Spearman.min())), abs(float(sectors.Spearman.max())), 0.02)
    for ax, name in zip(axes, ("高点", "低点")):
        values = np.array([ranked.loc[(name, sector), "Spearman"] for sector in order], dtype=float)
        p_values = np.array([ranked.loc[(name, sector), "Spearman双尾p"] for sector in order], dtype=float)
        counts = np.array([ranked.loc[(name, sector), "笔数"] for sector in order], dtype=int)
        significant = np.isfinite(p_values) & (p_values < 0.05)
        colors = np.where(~significant, "#C5CED6", np.where(values >= 0, "#1F7A4D", "#B23A48"))
        ax.hlines(y, 0, np.nan_to_num(values), color="#9FB3C8", lw=1.6, zorder=1)
        ax.scatter(values, y, c=colors, s=72, zorder=2, linewidths=0)
        ax.axvline(0, color="#9AA5B1", lw=0.8, zorder=1)
        tick_labels = [f"{sector}    {value:+.3f}\n{count:,} 日    {_p_text(p_value)}"
                       for sector, value, count, p_value in zip(order, values, counts, p_values)]
        ax.set_yticks(y, tick_labels, fontsize=9)
        ax.set_xlabel("Spearman 相关")
        ax.set_xlim(-reach * 1.35, reach * 1.35)
        ax.set_ylim(-0.65, len(order) - 0.35)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=4)
        n_sig = int(significant.sum())
        ax.set_title(f"{titles[name]}\n{n_sig} / {len(order)} 个板块  p < 0.05",
                     loc="left", fontsize=12, color="#102A43", pad=8)
    fig.suptitle(f"2016–2021  分板块  {hold_label}", fontsize=15, color="#102A43", x=0.18, ha="left", y=0.98)
    fig.text(0.18, 0.915, "同一板块左右对齐。绿为显著正相关，红为显著负相关，灰为不显著。每个点用该板块研究期的全部交易日。",
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.18, right=0.98, top=0.78, bottom=0.12, wspace=0.42)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def plot_yearly(yearly, path, scope="全部", hold_label="09:17 → 11:30"):
    """年度 pooled Spearman；三条线都只使用极值位置和多头收益。"""
    apply_style()
    frame = yearly[(yearly.样本 == "有夜盘") & (yearly.范围 == scope)]
    years = sorted(frame.年份.unique())
    fig, ax = plt.subplots(figsize=(11.8, 5.7))
    series = [
        ("高点时钟 → 多头收益", "高点时钟 → 多头收益", "#2F5D7C"),
        ("低点时钟 → 多头收益", "低点时钟 → 多头收益", "#C46B4A"),
        ("低点时钟-高点时钟 → 多头收益", "低点时钟 - 高点时钟 → 多头收益", "#1F7A4D"),
    ]
    for relation, label, color in series:
        if relation not in set(frame.关系):
            continue
        part = frame[frame.关系 == relation].sort_values("年份")
        ax.plot(part.年份.to_numpy(dtype=int), part.Spearman, marker="o", markersize=5, linewidth=2,
                color=color, label=label)
    ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
    ax.set_xticks(years)
    if years:
        ax.set_xlim(years[0] - 0.2, years[-1] + 0.2)
    ax.set_ylabel("年度 pooled Spearman")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="best", fontsize=9)
    values = frame.Spearman.dropna().to_numpy()
    reach = max(float(np.abs(values).max()), 0.02) * 1.15
    ax.set_ylim(-reach, reach)
    title = "全部品种" if scope == "全部" else scope
    fig.suptitle(f"2016–2021  夜盘样本分年 pooled Spearman · {title}  {hold_label}", fontsize=15,
                 color="#102A43", x=0.07, ha="left", y=0.98)
    note = "每年合并所有品种日计算；不按 dev15 分组，也不反转收益。"
    if scope != "全部":
        note += " 板块图纵轴单独缩放，不宜比较不同板块的幅度。"
    fig.text(0.07, 0.92, note,
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.13)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


MIN_CROSS_SECTION = 5  # 一天在范围内至少这么多个品种才算当天的截面 IC


def daily_ic(frame, return_col="gross_1130", min_names=MIN_CROSS_SECTION):
    """每个交易日在品种之间算一次 Spearman：当天的时钟排名对当天的收益排名。
    剔除了“这一天整个市场一起涨跌”的成分，只回答“同一天里，时钟靠后的品种是否收益更高”。"""
    rows = []
    for item in study_items(frame, return_col):
        x = pd.Series(item["x"], dtype=float)
        y = pd.Series(item["y"], dtype=float)
        day = pd.Series(item["day"])
        ok = x.notna() & y.notna()
        part = pd.DataFrame({"day": day[ok].to_numpy(), "x": x[ok].to_numpy(), "y": y[ok].to_numpy()})
        count = part.groupby("day").size()
        part = part[part.day.map(count) >= min_names]
        if part.empty:
            continue
        rank = part.groupby("day")[["x", "y"]].rank()
        part = part.assign(rx=rank.x, ry=rank.y)
        ic = part.groupby("day").apply(lambda d: d.rx.corr(d.ry), include_groups=False).dropna()
        rows.append(pd.DataFrame({"关系": item["关系"], "trading_date": pd.to_datetime(ic.index), "IC": ic.to_numpy(),
                                  "品种数": count.reindex(ic.index).to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["关系", "trading_date", "IC", "品种数"])


def ic_summary(ic, scope, group_year=True):
    """逐年（和全期）的截面 IC 均值、ICIR、Newey-West t 与双尾 p。"""
    from tfcta.research import timeseries_t
    rows = []
    keys = [("全期", ic)]
    if group_year:
        keys += [(int(y), g) for y, g in ic.groupby(ic.trading_date.dt.year)]
    for period, part in keys:
        for relation, g in part.groupby("关系", sort=False):
            series = g.sort_values("trading_date").IC
            test = timeseries_t(series)
            dof = max(int(test["n_periods"]) - 1, 1)
            p = float(2 * stats.t.sf(abs(test["t"]), dof)) if np.isfinite(test["t"]) else np.nan
            rows.append({"范围": scope, "年份": period, "关系": relation, "IC均值": float(series.mean()),
                         "IC标准差": float(series.std(ddof=1)), "ICIR": test["ic_ir"], "NW_t": test["t"], "双尾p": p,
                         "IC为正天数占比": float((series > 0).mean()), "天数": int(len(series)),
                         "平均品种数": float(g.品种数.mean())})
    return pd.DataFrame(rows)


def cross_section_tables(table, return_col="gross_1130"):
    """全部品种和各板块的逐日截面 IC 与逐年汇总。板块内每天至少 MIN_CROSS_SECTION 个品种。"""
    daily, summary = [], []
    scopes = [("全部", table)] + [(name, table[table.sector == name]) for name in morning.SECTORS]
    for scope, part in scopes:
        if part.empty:
            continue
        ic = daily_ic(part, return_col)
        if ic.empty:
            continue
        daily.append(ic.assign(范围=scope))
        summary.append(ic_summary(ic, scope))
    return pd.concat(daily, ignore_index=True), pd.concat(summary, ignore_index=True)


IC_SERIES = [
    ("高点时钟 → 多头收益", "高点时钟 → 多头收益", "#2F5D7C"),
    ("低点时钟 → 多头收益", "低点时钟 → 多头收益", "#C46B4A"),
    ("低点时钟-高点时钟 → 多头收益", "低点时钟 - 高点时钟 → 多头收益", "#1F7A4D"),
]


def plot_cross_section_yearly(summary, path, scope="全部", hold_label="09:17 → 11:30"):
    """逐年截面 IC 均值，竖线是 Newey-West 95% 区间；实心点 p < 0.05，空心点不显著。"""
    apply_style()
    frame = summary[(summary.范围 == scope) & (summary.年份 != "全期")].copy()
    frame["年份"] = frame.年份.astype(int)
    whole = summary[(summary.范围 == scope) & (summary.年份 == "全期")].set_index("关系")
    years = sorted(frame.年份.unique())
    fig, ax = plt.subplots(figsize=(11.8, 5.9))
    offsets = {name: shift for (name, _, _), shift in zip(IC_SERIES, (-0.12, 0.0, 0.12))}
    for relation, label, color in IC_SERIES:
        part = frame[frame.关系 == relation].sort_values("年份")
        if part.empty:
            continue
        x = part.年份.to_numpy(dtype=float) + offsets[relation]
        mean = part.IC均值.to_numpy()
        half = np.where(np.isfinite(part.NW_t) & (part.NW_t != 0), np.abs(mean / part.NW_t) * 1.96, np.nan)
        ax.errorbar(x, mean, yerr=half, fmt="none", ecolor=color, elinewidth=1.0, capsize=3, alpha=0.7, zorder=2)
        sig = part.双尾p.to_numpy() < 0.05
        ax.plot(x, mean, color=color, lw=1.6, zorder=3,
                label=f"{label}   全期 IC {whole.loc[relation, 'IC均值']:+.3f}  t {whole.loc[relation, 'NW_t']:+.2f}")
        ax.scatter(x[sig], mean[sig], color=color, s=46, zorder=4)
        ax.scatter(x[~sig], mean[~sig], facecolor="white", edgecolor=color, s=46, linewidths=1.4, zorder=4)
    ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
    ax.set_xticks(years)
    if years:
        ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.set_ylabel("年内日截面 IC 均值")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="best", fontsize=9)
    title = "全部品种" if scope == "全部" else scope
    fig.suptitle(f"2016–2021  逐日截面 IC 分年 · {title}  {hold_label}", fontsize=15,
                 color="#102A43", x=0.07, ha="left", y=0.98)
    names = int(round(float(frame.平均品种数.mean()))) if len(frame) else 0
    fig.text(0.07, 0.92,
             f"每个交易日在品种之间算一次 Spearman（每天至少 {MIN_CROSS_SECTION} 个品种，平均约 {names} 个），再按年平均。"
             "竖线为 Newey-West 95% 区间；实心 p < 0.05，空心不显著。",
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.11)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def plot_cross_section_cumulative(daily, path, scope="全部", hold_label="09:17 → 11:30"):
    """逐日截面 IC 的累计和：斜率稳定向上（或向下）说明关系持续存在，走平或拐头说明那段时间失效。"""
    apply_style()
    frame = daily[daily.范围 == scope]
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    for relation, label, color in IC_SERIES:
        part = frame[frame.关系 == relation].sort_values("trading_date")
        if part.empty:
            continue
        ax.plot(part.trading_date, part.IC.cumsum(), color=color, lw=1.6, label=label)
    ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=1)
    ax.set_ylabel("截面 IC 累计和")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    title = "全部品种" if scope == "全部" else scope
    fig.suptitle(f"2016–2021  截面 IC 累计 · {title}  {hold_label}", fontsize=15,
                 color="#102A43", x=0.07, ha="left", y=0.98)
    fig.text(0.07, 0.92, "线越陡关系越强；某段走平或反向，说明那段时间关系消失或反转。高点时钟预期向下，另两条预期向上。",
             fontsize=9, color="#7B8794")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.1)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def clear_old_plots(plot_dir):
    for name in OLD_PLOTS:
        path = plot_dir / name
        if path.exists():
            path.unlink()


EXITS = {
    "1130": {"hm": (11, 30), "column": "gross_1130", "label": "09:17 → 11:30", "out": morning.RUNS / "clock_return"},
    "1500": {"hm": (15, 0), "column": "gross_1500", "label": "09:17 → 15:00", "out": morning.RUNS / "clock_return_1500"},
}
BALANCE_SEED = 20261013
BALANCE_DRAWS = 200


def balance_up_down(table, return_col="gross_1130", seed=BALANCE_SEED):
    """涨跌各半重采样：去掉收益恰为 0 的品种日；在每个“年份 × 板块”里，把涨、跌中多的一边随机抽到和少的一边一样多。
    这样全样本、每一年、每个板块的涨跌笔数都相等，相关性不会来自样本里涨多或跌多。"""
    rng = np.random.default_rng(seed)
    sign = np.sign(table[return_col])
    frame = table[sign != 0]
    keep = []
    for _, cell in frame.groupby([frame.trading_date.dt.year, frame.sector], sort=True):
        up = cell.index[cell[return_col] > 0].to_numpy()
        down = cell.index[cell[return_col] < 0].to_numpy()
        n = min(len(up), len(down))
        keep += [rng.choice(up, n, replace=False), rng.choice(down, n, replace=False)]
    chosen = np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)
    return table.loc[chosen].reset_index(drop=True)


def balance_check(original, balanced, return_col="gross_1130"):
    """每个“年份 × 板块”重采样前后的涨、跌、平笔数。"""
    def counts(frame, tag):
        sign = np.sign(frame[return_col]).map({1.0: "涨", -1.0: "跌", 0.0: "平"})
        out = sign.groupby([frame.trading_date.dt.year.rename("年份"), frame.sector.rename("板块")]).value_counts()
        return out.unstack(fill_value=0).add_prefix(tag)
    return counts(original, "原样本").join(counts(balanced, "重采样后"), how="left").fillna(0).astype(int).reset_index()


def balance_draws(table, return_col="gross_1130", draws=BALANCE_DRAWS):
    """换 draws 个随机种子各重采样一次，看全样本 Spearman、聚类 p 和日截面 IC 的分布，确认结论不依赖某一次抽样。"""
    rows = []
    for k in range(draws):
        sample = balance_up_down(table, return_col, BALANCE_SEED + k)
        ic = daily_ic(sample, return_col)
        ic_mean = ic.groupby("关系").IC.mean()
        for item in study_items(sample, return_col):
            link = association(item["day"], item["x"], item["y"])
            rows.append({"抽样": k, "关系": item["关系"], "笔数": link["笔数"], "Spearman": link["Spearman"],
                         "Spearman双尾p": link["Spearman_p"], "日截面IC均值": ic_mean.get(item["关系"], np.nan)})
    return pd.DataFrame(rows)


def main(exit_name="1130", balance=False):
    spec = EXITS[exit_name]
    global OUT
    OUT = spec["out"]
    column, label = spec["column"], spec["label"]
    table = load_research(column, None if column == "gross_1130" else spec["hm"])
    if balance:
        OUT = OUT.with_name(OUT.name + "_balanced")
        OUT.mkdir(parents=True, exist_ok=True)
        draws = balance_draws(table, column)
        draws.to_csv(OUT / "balance_draws.csv", index=False, encoding="utf-8-sig")
        spread = draws.groupby("关系")[["Spearman", "Spearman双尾p", "日截面IC均值"]].quantile([.05, .5, .95]).unstack()
        print(f"{BALANCE_DRAWS} 次重采样的 5% / 50% / 95% 分位：")
        print(spread.round(4).to_string())
        balanced = balance_up_down(table, column)
        balance_check(table, balanced, column).to_csv(OUT / "balance_check.csv", index=False, encoding="utf-8-sig")
        table = balanced
        label = f"{label}（涨跌各半重采样）"
    result, bins, yearly = collect(table, column)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "relation.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(OUT / "bins.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT / "yearly.csv", index=False, encoding="utf-8-sig")
    night = samples_of(table)["有夜盘"]
    segments, segment_summary = segment_table(night, return_col=column)
    sectors = sector_table(night, column)
    segments.to_csv(OUT / "segments.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(OUT / "position.csv", index=False, encoding="utf-8-sig")
    sectors.to_csv(OUT / "by_sector.csv", index=False, encoding="utf-8-sig")
    leftover = OUT / "by_symbol.csv"
    if leftover.exists():
        leftover.unlink()
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    clear_old_plots(plot_dir)
    plot_segments(segments, segment_summary, plot_dir / "segments.png", hold_label=label)
    plot_sectors(sectors, plot_dir / "sectors.png", label)
    plot_yearly(yearly, plot_dir / "yearly.png", hold_label=label)
    yearly_sector_dir = plot_dir / "sectors"
    yearly_sector_dir.mkdir(parents=True, exist_ok=True)
    available_scopes = set(yearly.范围.unique())
    for sector in morning.SECTORS:
        if sector in available_scopes:
            plot_yearly(yearly, yearly_sector_dir / f"yearly_{sector}.png", scope=sector, hold_label=label)
    ic_daily, ic_yearly = cross_section_tables(night, column)
    ic_daily.to_csv(OUT / "ic_daily.csv", index=False, encoding="utf-8-sig")
    ic_yearly.to_csv(OUT / "ic_yearly.csv", index=False, encoding="utf-8-sig")
    plot_cross_section_yearly(ic_yearly, plot_dir / "ic_yearly.png", hold_label=label)
    plot_cross_section_cumulative(ic_daily, plot_dir / "ic_cumulative.png", hold_label=label)
    ic_sector_dir = plot_dir / "ic_sectors"
    ic_sector_dir.mkdir(parents=True, exist_ok=True)
    for sector in morning.SECTORS:
        if sector in set(ic_yearly.范围):
            plot_cross_section_yearly(ic_yearly, ic_sector_dir / f"ic_yearly_{sector}.png", scope=sector, hold_label=label)
    show = result[result.范围 == "全部"][
        ["样本", "关系", "笔数", "Spearman", "Spearman双尾p", "时钟0到1的收益bp", "斜率双尾p", "晚减早bp", "晚减早p"]]
    (OUT / "summary.json").write_text(show.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(show.round(4).to_string(index=False))
    print("\n分板块（夜盘样本，Spearman 与双尾 p）：")
    sector = result[result.样本 == "有夜盘"]
    wide = sector.pivot(index="范围", columns="关系", values=["Spearman", "Spearman双尾p"])
    print(wide.round(4).to_string())
    print("结果：", OUT)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit", choices=tuple(EXITS), default="1130", help="11:30 上午收盘，或 15:00 日盘收盘")
    parser.add_argument("--balance", action="store_true",
                        help="涨跌各半重采样后再算（每个年份×板块里涨、跌笔数相等），输出到 <输出目录>_balanced/")
    args = parser.parse_args()
    main(args.exit, args.balance)
