"""商品两本冻结策略的研究期图表报告：净值回撤、对比等权买入持有、只保留某一部分时的净值。

    python -m backtest.engine --research                       # 先跑两本商品冻结账本
    python -m research.report.strategy_report

股指熊市空头的净值回撤和对比买入持有图由 backtest.engine --market index --research 直接写出；
本模块的绘图工具（_two_panel、plot_nav_drawdown 等）也供那里使用。

规格在 config/research/strategy_report.json。完整策略按冻结规则重算并与 engine 账本逐日核对。
结果在 runs/reports/commodity/。只读研究期，不读 2022 及以后，不读 2026。
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from backtest.engine import (
    INITIAL_CAPITAL,
    MARGIN_RATE,
    ledger_inputs,
    max_drawdown,
    performance,
    sharpe_ratio,
    simulate_target_lots,
)
from research.commodity import market_clock_slow_short_20_60_120 as D
from strategies import commodity_close_location as R
from strategies import commodity_original as S
from tfcta import config as C

CONFIG_PATH = C.CONFIG_DIR / "research" / "strategy_report.json"
OUT = C.RUNS_DIR / "reports"
BOOK_IDS = (S.BOOK.ID, R.STRICT_SWAP_BOOK.ID)
COMMODITY_YEARS = tuple(range(C.EARLY_START.year, C.RESEARCH_END.year + 1))
BENCHMARK_DRAWDOWN = 0.99          # 买入持有基准不设熔断；引擎要求 (0, 1)，取 99% 等于不触发

# 已用 dataviz 校验脚本检查（浅色背景，相邻色 CVD ΔE ≥ 9.1）；后三色对比度 < 3:1，所以每条线都直接标注
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
BENCH = "#8a8984"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
COMPONENT_LABELS = {
    "full": "完整策略",
    "long_leg": "多头腿（信号且展期>0）",
    "short_leg": "空头腿单独",
    "signal_only": "时间信号单独",
    "carry_only": "展期>0单独",
}
COMPONENT_COLORS = dict(zip(COMPONENT_LABELS, SERIES))
OWN_CLV = "full_own_clv"
OWN_CLV_LABEL = "完整策略（多头用自身收盘位置）"


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or protocol.get("books") != list(BOOK_IDS)
        or "不读 2022" not in sample
        or "不读 2026" not in sample
    ):
        raise ValueError("策略报告登记与样本边界或策略清单不一致")
    return protocol


# --------------------------------------------------------------------------
# 目标仓位（收盘决策，后移一天到下一开盘执行）
# --------------------------------------------------------------------------
def component_decisions(signal: pd.DataFrame, lagged: pd.DataFrame, trend: pd.DataFrame,
                        bear: pd.Series) -> dict[str, pd.DataFrame]:
    """signal 是逐品种的多头时间信号（布尔）；lagged 是滞后一日展期。空头与多头不会同日冲突（展期符号相反）。"""
    ones = pd.DataFrame(1.0, index=lagged.index, columns=lagged.columns)
    signal = signal.reindex_like(ones).fillna(False).astype(bool)
    carry_ok = lagged.gt(0)
    bear_days = pd.DataFrame({c: bear.reindex(ones.index).eq(True) for c in ones.columns}, index=ones.index)
    short = trend.reindex_like(ones).lt(0) & lagged.lt(0) & bear_days
    long_leg = ones.where(signal & carry_ok, 0.0)
    return {
        "full": long_leg.where(~short, -1.0),
        "long_leg": long_leg,
        "short_leg": -short.astype(float),
        "signal_only": ones.where(signal, 0.0),
        "carry_only": ones.where(carry_ok, 0.0),
    }


def _broadcast(flag: pd.Series, columns) -> pd.DataFrame:
    return pd.DataFrame({c: flag.eq(True) for c in columns}, index=flag.index)


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def stats(daily: pd.DataFrame, benchmark: pd.Series | None = None) -> dict:
    net = daily["净收益"].fillna(0.0)
    row = {
        "起点": str(daily.index.min().date()),
        "终点": str(daily.index.max().date()),
        "年化净收益": performance(net)["ann_return"],
        "净Sharpe": sharpe_ratio(net),
        "最大回撤": max_drawdown(daily["权益"], float(daily["期初权益"].iloc[0])),
        "累计净收益": float(daily["权益"].iloc[-1] / daily["期初权益"].iloc[0] - 1),
        "年化成本占权益": float((daily["交易费用"] / daily["期初权益"]).mean() * C.TRADING_DAYS_PER_YEAR),
        "熔断次数": int((daily["停止新开仓"].astype(bool)
                     & ~daily["停止新开仓"].astype(bool).shift(1, fill_value=False)).sum()),
    }
    if benchmark is not None:
        bench = benchmark.reindex(net.index).fillna(0.0)
        row["相对买入持有beta"] = float(net.cov(bench) / bench.var())
    return row


def annual_returns(net: pd.Series) -> pd.Series:
    return (1 + net.fillna(0.0)).groupby(net.index.year).prod() - 1


def leg_contributions(daily: pd.DataFrame, positions: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    """完整策略逐日：多头腿、空头腿毛盈亏和成本除以期初权益；三者相加必须等于账本净收益。"""
    equity = daily["期初权益"]
    out = pd.DataFrame(0.0, index=daily.index, columns=["多头腿", "空头腿", "交易成本及清零调整"])
    if not positions.empty:
        held = positions.assign(交易日=pd.to_datetime(positions["交易日"]))
        moves = delta.stack()
        moves.index.names = ["交易日", "品种"]
        move = moves.reindex(pd.MultiIndex.from_frame(held[["交易日", "品种"]])).to_numpy(dtype=float)
        if np.isnan(move).any():
            raise ValueError("持仓归因缺少复权开盘变动")
        pnl = held["手数"].to_numpy(dtype=float) * held["合约乘数"].to_numpy(dtype=float) * move
        for side, mask in (("多头腿", held["手数"].to_numpy() > 0), ("空头腿", held["手数"].to_numpy() < 0)):
            by_day = pd.Series(pnl[mask], index=held["交易日"].to_numpy()[mask]).groupby(level=0).sum()
            out[side] = by_day.reindex(daily.index, fill_value=0.0)
    if not np.allclose(out[["多头腿", "空头腿"]].sum(axis=1), daily["毛盈亏"], rtol=1e-10, atol=1e-4):
        raise ArithmeticError("多空腿毛盈亏与账本不一致")
    out = out.div(equity, axis=0)
    out["交易成本及清零调整"] = (daily["权益清零调整"] - daily["交易费用"]) / equity
    if not np.allclose(out.sum(axis=1), daily["净收益"].fillna(0.0), rtol=1e-10, atol=1e-10):
        raise ArithmeticError("多空腿和成本贡献不能复算账本净收益")
    return out


def _engine_ledger(book_id: str) -> pd.DataFrame:
    path = C.RUNS_DIR / "strategies" / book_id / "research" / "daily.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结账本 {path}；先用 backtest.engine 跑研究期")
    return pd.read_csv(path, parse_dates=["交易日"]).set_index("交易日")


def check_against_engine(book_id: str, daily: pd.DataFrame) -> None:
    saved = _engine_ledger(book_id)
    if not saved.index.equals(daily.index) or not np.allclose(
        saved["净收益"].fillna(0.0), daily["净收益"].fillna(0.0), rtol=0, atol=1e-9,
    ):
        raise ArithmeticError(f"{book_id} 报告重算的净收益与 engine 账本不一致；先重跑 backtest.engine")


# --------------------------------------------------------------------------
# 商品
# --------------------------------------------------------------------------
def _commodity_simulate(target, inputs, drawdown_limit=S.DRAWDOWN_LIMIT):
    keys = ("open_price", "close_price", "adjusted_open", "adjusted_open_delta", "adjusted_high",
            "adjusted_low", "rolls", "contract_codes")
    return simulate_target_lots(
        target.reindex(index=inputs["index"], columns=inputs["columns"]).fillna(0.0),
        *(inputs[key] for key in keys), inputs["universe"], inputs["multipliers"],
        inputs["open_fee"], inputs["close_fee"], inputs["roll_close_fee"], inputs["slippage"],
        initial_capital=INITIAL_CAPITAL, margin_rate=MARGIN_RATE, exposure_limit=S.GROSS,
        drawdown_limit=drawdown_limit, halt_days=S.HALT_DAYS,
        contract_multipliers=inputs["contract_multipliers"],
    )


def run_commodity() -> dict:
    data = D._load(C.RESEARCH_END.year, C.EARLY_START.year)
    inputs = ledger_inputs(data, COMMODITY_YEARS, pd.Timestamp(C.RESEARCH_END))
    index = data.returns.index
    symbols = sorted({s for members in data.universe.values() for s in members})
    pool = S.pool_mask(data.universe, index, symbols)
    carry = S.roll_yield(index, symbols)
    S.require_carry_coverage(carry, pool)
    lagged = carry.reindex(index=index, columns=symbols).shift(1)
    trend = S.symbol_trend(data.factors.bars, symbols).reindex(index)
    bear = S.market_trend(S.close_return(data.factors.bars, symbols).reindex(index).where(pool).mean(axis=1)).lt(0)
    signals = {
        S.BOOK.ID: _broadcast(S.market_score(data.factors.signed, data.universe, index).gt(0), symbols),
        R.STRICT_SWAP_BOOK.ID: _broadcast(
            R.market_close_location(data.factors.bars, data.universe, index, symbols).lt(R.LONG_THRESHOLD), symbols),
    }
    frozen = {
        S.BOOK.ID: S.targets(data.factors.signed, data.factors.bars, data.universe, index, symbols, carry),
        R.STRICT_SWAP_BOOK.ID: R.targets(data.factors.bars, data.universe, index, symbols, carry=carry,
                                         long_carry=True),
    }
    bench_daily, _, _ = _commodity_simulate(pool.astype(float), inputs, BENCHMARK_DRAWDOWN)
    bench = bench_daily["净收益"].fillna(0.0)
    books = {}
    for book_id, signal in signals.items():
        decisions = component_decisions(signal, lagged, trend, bear)
        targets = {k: v.shift(1).fillna(0.0).where(pool, 0.0) for k, v in decisions.items()}
        if not np.allclose(targets["full"].to_numpy(), frozen[book_id].to_numpy()):
            raise ArithmeticError(f"{book_id} 报告的完整目标与冻结策略 targets 不一致")
        nets, rows = {}, []
        for key, target in targets.items():
            daily, positions, _ = _commodity_simulate(target, inputs)
            if key == "full":
                check_against_engine(book_id, daily)
                legs = leg_contributions(daily, positions, inputs["adjusted_open_delta"])
                full_daily = daily
            nets[key] = daily["净收益"].fillna(0.0)
            rows.append({"组成": COMPONENT_LABELS[key], **stats(daily, bench)})
        books[book_id] = {"nets": nets, "rows": rows, "legs": legs, "daily": full_daily}
    return {"books": books, "benchmark": bench, "benchmark_row": stats(bench_daily),
            "benchmark_label": "池内等权买入持有"}


# --------------------------------------------------------------------------
# 股指
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------
def _setup() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def _style(ax, percent: bool = False) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b2")
    ax.tick_params(length=3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=14, color=INK, weight="bold")
    fig.text(0.012, 0.947, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)


def _nav(net: pd.Series) -> pd.Series:
    return (1 + net.fillna(0.0)).cumprod()


def _drawdown(net: pd.Series) -> pd.Series:
    nav = _nav(net)
    return nav / nav.cummax().clip(lower=1.0) - 1


def _stat_text(row: dict) -> str:
    return f"Sharpe {row['净Sharpe']:.2f} · 年化 {row['年化净收益']:.1%} · 回撤 {row['最大回撤']:.1%}"


def _end_labels(ax, items: list[tuple[str, float, str]], log: bool) -> None:
    """线尾直接标注（名称 + 期末净值），按纵向最小间距错开，避免重叠。"""
    lo, hi = ax.get_ylim()
    span = (np.log10(hi) - np.log10(lo)) if log else (hi - lo)
    gap = span * 0.045
    pos = lambda v: np.log10(v) if log else v                 # noqa: E731
    back = lambda v: 10 ** v if log else v                    # noqa: E731
    placed = []
    for text, value, color in sorted(items, key=lambda x: x[1]):
        y = pos(value)
        if placed and y < placed[-1] + gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(f"{text}  {value:.2f}", xy=(1.0, back(y)), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=color,
                    weight="bold", annotation_clip=False)


def _log_axis(ax, navs) -> None:
    ax.set_yscale("log")
    low = min(float(s.min()) for s in navs)
    high = max(float(s.max()) for s in navs)
    candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
    ticks = [t for t in candidates if low * 0.95 <= t <= high * 1.05] or [1.0]
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter([f"{t:g}" for t in ticks]))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(low * 0.93, high * 1.07)


def plot_nav_drawdown(ax_nav, ax_dd, series: list[tuple[str, pd.Series, str, dict, dict]], legend: bool = True):
    """series：(名称, 日净收益, 颜色, 统计, 线型参数)。上方对数净值，下方回撤。"""
    navs = []
    for i, (name, net, color, row, style) in enumerate(series):
        nav = _nav(net)
        navs.append(nav)
        dates = nav.index.to_numpy(dtype="datetime64[ns]")
        ax_nav.plot(dates, nav.to_numpy(), color=color, lw=style.get("lw", 2.0), ls=style.get("ls", "-"),
                    label=f"{name}　{_stat_text(row)}", zorder=3 if i == 0 else 2)
        dd = _drawdown(net)
        ax_dd.plot(dates, dd.to_numpy(), color=color, lw=1.1, ls=style.get("ls", "-"))
        if i == 0:
            ax_dd.fill_between(dates, dd.to_numpy(), 0, color=color, alpha=0.14, lw=0)
    ax_nav.axhline(1.0, color="#b9b8b2", lw=0.8, zorder=1)
    _log_axis(ax_nav, navs)
    _end_labels(ax_nav, [(n, float(_nav(r).iloc[-1]), c) for n, r, c, _, _ in series], log=True)
    ax_nav.set_ylabel("净值（对数轴）")
    ax_dd.set_ylabel("回撤")
    _style(ax_nav)
    _style(ax_dd, percent=True)
    ax_dd.set_ylim(min(float(_drawdown(r).min()) for _, r, _, _, _ in series) * 1.15, 0.005)
    if legend:
        ax_nav.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)


def _two_panel(width=12.5, height=7.2, cols=1):
    fig, axes = plt.subplots(2, cols, figsize=(width, height), sharex="col", height_ratios=[3, 1],
                             squeeze=False)
    fig.subplots_adjust(left=0.07 if cols == 1 else 0.05, right=0.86 if cols == 1 else 0.92,
                        top=0.86, bottom=0.07, hspace=0.08, wspace=0.2 if cols == 1 else 0.5)
    return fig, axes


def _first_column_labels(axes) -> None:
    for ax in (*axes[0, 1:], *axes[1, 1:]):
        ax.set_ylabel("")


def _save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


COSTS = "1 亿元、1 倍敞口按池宽分配、历史手续费加每边 1 跳滑点；研究期样本内结果"


def plot_commodity(result: dict, out) -> None:
    books = result["books"]
    names = {S.BOOK.ID: S.BOOK.NAME, R.STRICT_SWAP_BOOK.ID: R.STRICT_SWAP_BOOK.NAME}
    colors = {S.BOOK.ID: SERIES[0], R.STRICT_SWAP_BOOK.ID: SERIES[1]}
    row_of = lambda b: books[b]["rows"][0]                    # noqa: E731
    first, last = books[S.BOOK.ID]["daily"].index[[0, -1]]
    span = f"{first:%Y-%m-%d} 至 {last:%Y-%m-%d}"

    fig, axes = _two_panel()
    plot_nav_drawdown(axes[0, 0], axes[1, 0], [
        (names[b], books[b]["nets"]["full"], colors[b], row_of(b), {}) for b in books])
    _title(fig, "商品两本冻结策略：净值与回撤", f"{span}｜{COSTS}")
    _save(fig, out / "nav_drawdown.png")

    fig, axes = _two_panel()
    plot_nav_drawdown(axes[0, 0], axes[1, 0], [
        *[(names[b], books[b]["nets"]["full"], colors[b], row_of(b), {}) for b in books],
        (result["benchmark_label"], result["benchmark"], BENCH, result["benchmark_row"], {"lw": 1.6})])
    _title(fig, "商品策略与池内等权买入持有",
           f"{span}｜基准：当年池内品种等权 1 倍做多、同样成本、不设熔断、不再平衡｜{COSTS}")
    _save(fig, out / "vs_buy_hold.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.84, bottom=0.08, wspace=0.36)
    navs = [_nav(n) for b in books for n in books[b]["nets"].values()]
    signal_names = {S.BOOK.ID: "时间信号 = 五因子分数 > 0", R.STRICT_SWAP_BOOK.ID: "时间信号 = 池内收盘位置 < 0.5"}
    for ax, book_id in zip(axes, books):
        _plot_components(ax, books[book_id], navs)
        ax.set_title(f"{names[book_id]}（{signal_names[book_id]}）", loc="left", fontsize=11, color=INK, pad=10)
    axes[0].set_ylabel("净值（对数轴）")
    _title(fig, "商品：只保留某一部分时的净值",
           f"{span}｜每条线是单独一本 1 亿元账（保留 10% 熔断），互相不能相加，不是因果拆分｜{COSTS}")
    _save(fig, out / "components.png")


def _plot_components(ax, book: dict, navs_for_axis, keys=None, legend=True) -> None:
    keys = keys or list(book["nets"])
    items = []
    for i, key in enumerate(keys):
        label = OWN_CLV_LABEL if key == OWN_CLV else COMPONENT_LABELS[key]
        color = SERIES[5] if key == OWN_CLV else COMPONENT_COLORS[key]
        nav = _nav(book["nets"][key])
        row = book["rows"][list(book["nets"]).index(key)]
        ax.plot(nav.index.to_numpy(dtype="datetime64[ns]"), nav.to_numpy(), color=color,
                lw=2.4 if key == "full" else 1.5, ls="--" if key == OWN_CLV else "-",
                zorder=4 if key == "full" else 2, label=f"{label}　Sharpe {row['净Sharpe']:.2f}")
        items.append((label.split("（")[0] if key != OWN_CLV else "自身收盘位置版", float(nav.iloc[-1]), color))
    ax.axhline(1.0, color="#b9b8b2", lw=0.8, zorder=1)
    _log_axis(ax, navs_for_axis)
    _style(ax)
    _end_labels(ax, items, log=True)
    if legend:
        ax.legend(loc="upper left", frameon=False, fontsize=8.3, labelcolor=INK)


# --------------------------------------------------------------------------
# 表
# --------------------------------------------------------------------------
def _tables(groups: dict[str, dict], out) -> pd.DataFrame:
    """groups：名称 → {nets, rows, legs, benchmark, benchmark_row, benchmark_label}。"""
    summary, yearly, legs = [], [], []
    for group, book in groups.items():
        for key, row in zip(book["nets"], book["rows"]):
            summary.append({"账本": group, **row})
        summary.append({"账本": group, "组成": book["benchmark_label"], **book["benchmark_row"]})
        table = pd.DataFrame({row["组成"]: annual_returns(book["nets"][key])
                              for key, row in zip(book["nets"], book["rows"])})
        table[book["benchmark_label"]] = annual_returns(book["benchmark"])
        yearly.append(table.rename_axis("年份").reset_index().assign(账本=group))
        part = book["legs"].groupby(book["legs"].index.year).sum()
        part["合计（不复利）"] = part.sum(axis=1)
        legs.append(part.rename_axis("年份").reset_index().assign(账本=group))
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(summary)
    frame.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(yearly, ignore_index=True).to_csv(out / "annual.csv", index=False, encoding="utf-8-sig")
    pd.concat(legs, ignore_index=True).to_csv(out / "leg_contributions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({f"{g}|{k}": v for g, b in groups.items() for k, v in b["nets"].items()}).to_csv(
        out / "daily_returns.csv", encoding="utf-8-sig")
    return frame


def run() -> None:
    protocol = load_protocol()
    for book in (S.BOOK, R.STRICT_SWAP_BOOK):
        book.frozen()
    _setup()
    commodity = run_commodity()
    names = {S.BOOK.ID: S.BOOK.NAME, R.STRICT_SWAP_BOOK.ID: R.STRICT_SWAP_BOOK.NAME}
    commodity_groups = {
        names[b]: {**v, "benchmark": commodity["benchmark"], "benchmark_row": commodity["benchmark_row"],
                   "benchmark_label": commodity["benchmark_label"]}
        for b, v in commodity["books"].items()
    }
    pd.set_option("display.width", 240)
    columns = ["账本", "组成", "起点", "年化净收益", "净Sharpe", "最大回撤", "相对买入持有beta", "熔断次数"]
    for market, groups, plot, result in (("commodity", commodity_groups, plot_commodity, commodity),):
        out = OUT / market
        frame = _tables(groups, out)
        plot(result, out / "plots")
        (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
        print(frame[columns].round(3).to_string(index=False), flush=True)
    print(f"结果：{OUT}", flush=True)


if __name__ == "__main__":
    run()
