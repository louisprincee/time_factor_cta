"""商品收盘位置反转（探索期）：登记的三个变体和冻结策略同账本对照。只跑研究期。

    python -m research.commodity_close_location_reversal
    python -m research.commodity_close_location_reversal --three-book-comparison

规格在 strategies/commodity_close_location_reversal/commodity_close_location_reversal.json。
结果在 runs/research/commodity_close_location_reversal/。不读 2022 及以后，不读 2026。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.engine import INITIAL_CAPITAL, ledger_inputs, newey_west_t, plot_performance
from research import index_market_clock as M
from research import market_clock_slow_short_20_60_120 as D
from strategies import commodity_close_location_reversal as R
from strategies import index_close_location_reversal as IX
from strategies import market_clock_slow_short_20_60_120 as S
from tfcta import config as C

OUT = C.RUNS_DIR / "research" / R.BOOK.ID
WINDOWS = ((2011, 2021), (2016, 2021))
VARIANTS = {"main": {}, "long_carry": {"long_carry": True}, "long_only": {"long_only": True}}


def run() -> None:
    config = R.BOOK.load_config()
    if list(config["variants"]) != list(VARIANTS):
        raise ValueError("登记的变体与代码不一致")
    rows, years, paired = [], [], []
    for start, end in WINDOWS:
        data = D._load(end, start)
        span = tuple(range(start, end + 1))
        inputs = ledger_inputs(data, span, pd.Timestamp(f"{end}-12-31"))
        every = sorted({s for members in data.universe.values() for s in members})
        index, columns = inputs["index"], inputs["columns"]
        market = D.pool_open_return(data, inputs["universe"], index)
        carry = S.roll_yield(data.returns.index, every)
        S.require_carry_coverage(carry, S.pool_mask(data.universe, data.returns.index, every))
        books = {"frozen": S.BOOK.targets(data, every)}
        for key, options in VARIANTS.items():
            books[key] = R.targets(data.factors.bars, data.universe, data.returns.index, every, carry=carry, **options)
        nets = {}
        for key, target in books.items():
            target = target.reindex(index=index, columns=columns).fillna(0.0)
            daily, positions, _ = D.simulate(target, inputs, S.BOOK)
            row, annual = D.summarize(key, daily, positions, market, inputs["adjusted_open_delta"], D.sectors())
            row["区间"] = f"{start}-{end}"
            row["多头品种日"] = int((target > 0).sum().sum())
            row["空头品种日"] = int((target < 0).sum().sum())
            rows.append(row)
            years += [{**a, "区间": f"{start}-{end}"} for a in annual]
            nets[key] = daily["净收益"].fillna(0.0)
        for key in VARIANTS:
            diff = nets[key] - nets["frozen"]
            paired.append({"区间": f"{start}-{end}", "变体": key, "相对冻结日超额年化": float(diff.mean() * 242),
                           "配对NW_t": newey_west_t(diff, D.NW_LAGS),
                           "与冻结日收益相关": float(nets[key].corr(nets["frozen"]))})
    OUT.mkdir(parents=True, exist_ok=True)
    summary, annual, pair = pd.DataFrame(rows), pd.DataFrame(years), pd.DataFrame(paired)
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(OUT / "paired.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 240)
    cols = ["区间", "变体", "年化净收益", "净Sharpe", "最大回撤", "相对池子beta", "相对池子年化alpha", "alpha_t",
            "为正年份", "熔断次数", "年化成本占权益", "多头品种日", "空头品种日"]
    print(summary[cols].round(3).to_string(index=False))
    print(pair.round(3).to_string(index=False))
    print(annual.pivot_table(index=["区间", "年份"], columns="变体", values="净收益").round(3).to_string())
    print(f"结果：{OUT}")


# 三策略严格替换比较与多空、品种贡献归因。
CONFIG_PATH = C.CONFIG_DIR / "research" / "three_strategy_clv_comparison.json"
FOLLOWUP_CONFIG_PATH = C.CONFIG_DIR / "research" / "three_strategy_contribution_followup.json"
COMPARISON_OUT = C.RUNS_DIR / "research" / "three_strategy_clv_comparison"
FOLLOWUP_OUT = COMPARISON_OUT / "component_followup"
COMMODITY_START = 2011
END_YEAR = 2021
INDEX_START = pd.Timestamp("2015-04-16")
INDEX_END = pd.Timestamp("2021-12-31")
BOOK_LABELS = {
    "commodity_original": "商品原策略",
    "commodity_clv_swap": "商品策略（仅时间换收盘位置）",
    "index_clv_swap": "股指策略（仅时间换收盘位置）",
}
COLORS = {
    "commodity_original": "#315A7D",
    "commodity_clv_swap": "#E07A3F",
    "index_clv_swap": "#2A8C82",
    "long": "#2A8C82",
    "short": "#E07A3F",
    "cost": "#7D8490",
    "IF": "#315A7D",
    "IH": "#E07A3F",
    "IC": "#2A8C82",
}


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or protocol.get("books")
        != [
            "market_clock_slow_short_20_60_120",
            "commodity_close_location_reversal_strict_swap",
            "index_close_location_reversal_strict_swap",
        ]
        or "Do not read 2022" not in protocol.get("sample", "")
        or "2023-2025 locked partitions" not in protocol.get("sample", "")
        or "or 2026" not in protocol.get("sample", "")
        or "2015-04-16" not in protocol.get("sample", "")
        or protocol.get("rules", {}).get("close_location")
        != (
            "For each instrument, (closew - loww) / (highw - loww), with additive-adjusted "
            "price differences only; use the equal-weight mean across the active commodity "
            "pool or IF/IH/IC."
        )
    ):
        raise ValueError("三策略收盘位置对照登记与样本边界或规则不一致")
    return protocol


def load_followup_protocol() -> dict:
    protocol = json.loads(FOLLOWUP_CONFIG_PATH.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    components = protocol.get("commodity_components", {})
    products = protocol.get("index_products", {})
    if (
        protocol.get("status") != "post_hoc_research_only"
        or "Do not read 2022" not in sample
        or "2023-2025 locked partitions" not in sample
        or "or 2026" not in sample
        or components.get("books") != [S.BOOK.ID, R.STRICT_SWAP_BOOK.ID]
        or products.get("instruments") != list(M.SYMBOLS)
        or "own close location < 0.5" not in products.get("individual_books", "")
    ):
        raise ValueError("组成部分后续检验规格与样本边界或实现不一致")
    return protocol


def index_close_location_targets(
    index: pd.DatetimeIndex,
    bars: dict[str, pd.DataFrame],
    carry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """用收盘位置替换股指迁移策略的五因子分数，其余信号保持不变。"""
    market_clv = IX.market_close_location(bars, index)
    targets = IX.STRICT_SWAP_BOOK.targets(market_clv, bars, carry)
    return targets, market_clv


def attribute_positions(
    daily: pd.DataFrame,
    positions: pd.DataFrame,
    adjusted_open_delta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按多空方向和品种拆毛盈亏，实际交易成本单列，并核对账本净收益。"""
    index = daily.index
    equity = daily["期初权益"].reindex(index)
    if (equity <= 0).any() or not np.isfinite(equity.to_numpy()).all():
        raise ValueError("收益归因要求每日期初权益为正且有限")

    side_pnl = pd.DataFrame(0.0, index=index, columns=["多头腿", "空头腿"])
    symbol_pnl = pd.DataFrame(0.0, index=index, columns=adjusted_open_delta.columns)
    if not positions.empty:
        held = positions.copy()
        held["交易日"] = pd.to_datetime(held["交易日"])
        key = pd.MultiIndex.from_frame(held[["交易日", "品种"]])
        delta = adjusted_open_delta.stack()
        delta.index.names = ["交易日", "品种"]
        moves = delta.reindex(key)
        if moves.isna().any():
            raise ValueError("持仓收益归因缺少复权开盘变动")
        pnl = (
            held["手数"].to_numpy(dtype="float64")
            * held["合约乘数"].to_numpy(dtype="float64")
            * moves.to_numpy(dtype="float64")
        )
        values = pd.DataFrame(
            {
                "交易日": held["交易日"].to_numpy(),
                "品种": held["品种"].to_numpy(),
                "手数": held["手数"].to_numpy(dtype="float64"),
                "毛盈亏": pnl,
            }
        )
        for side, selected in (
            ("多头腿", values["手数"] > 0),
            ("空头腿", values["手数"] < 0),
        ):
            by_day = values.loc[selected].groupby("交易日")["毛盈亏"].sum()
            side_pnl[side] = by_day.reindex(index, fill_value=0.0)
        by_symbol = values.groupby(["交易日", "品种"])["毛盈亏"].sum().unstack("品种")
        symbol_pnl = by_symbol.reindex(index=index, columns=adjusted_open_delta.columns).fillna(0.0)

    expected_gross = daily["毛盈亏"].reindex(index)
    if not np.allclose(side_pnl.sum(axis=1), expected_gross, rtol=1e-10, atol=1e-5):
        raise ArithmeticError("按多空腿拆分的毛盈亏与账本不一致")

    contribution = side_pnl.div(equity, axis=0)
    contribution["交易成本及清零调整"] = (
        -daily["交易费用"].reindex(index) + daily["权益清零调整"].reindex(index)
    ).div(equity)
    if not np.allclose(
        contribution.sum(axis=1),
        daily["净收益"].reindex(index),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ArithmeticError("多空及成本贡献无法复算账本净收益")

    instrument_contribution = symbol_pnl.div(equity, axis=0)
    instrument_contribution["交易成本及清零调整"] = contribution["交易成本及清零调整"]
    if not np.allclose(
        instrument_contribution.sum(axis=1),
        daily["净收益"].reindex(index),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ArithmeticError("按品种及成本拆分的贡献无法复算账本净收益")
    return contribution, instrument_contribution


def _annual_rows(strategy_id: str, daily: pd.DataFrame, contribution: pd.DataFrame) -> list[dict]:
    rows = []
    for year, part in daily.groupby(daily.index.year):
        annual_return = float((1.0 + part["净收益"].fillna(0.0)).prod() - 1.0)
        components = contribution.loc[part.index].sum()
        rows.append(
            {
                "策略ID": strategy_id,
                "策略": BOOK_LABELS[strategy_id],
                "年份": int(year),
                "净收益": annual_return,
                "日收益贡献求和": float(components.sum()),
                "多头腿贡献": float(components.get("多头腿", 0.0)),
                "空头腿贡献": float(components.get("空头腿", 0.0)),
                "交易成本贡献": float(components.get("交易成本及清零调整", 0.0)),
            }
        )
    return rows


def _commodity_component_targets(data, symbols, carry) -> dict[str, pd.DataFrame]:
    index = data.returns.index
    pool = S.pool_mask(data.universe, index, symbols)
    lagged = carry.reindex(index=index, columns=symbols).shift(1)
    ones = pd.DataFrame(1.0, index=index, columns=symbols)
    score = S.market_score(data.factors.signed, data.universe, index)
    close_location = R.market_close_location(
        data.factors.bars, data.universe, index, symbols
    )
    trend = S.symbol_trend(data.factors.bars, symbols).reindex(index)
    market_daily = S.close_return(data.factors.bars, symbols).where(pool).mean(axis=1)
    bear = S.market_trend(market_daily).lt(0)
    short = -(
        (trend.lt(0) & lagged.lt(0)).mul(bear.eq(True), axis=0).astype(float)
    )
    decisions = {
        "score_only": ones.mul(score.gt(0).astype(float), axis=0),
        "clv_only": ones.mul(close_location.lt(R.LONG_THRESHOLD).astype(float), axis=0),
        "carry_only": ones.where(lagged.gt(0), 0.0),
        "score_and_carry_long": S.long_weights(score, lagged),
        "clv_and_carry_long": ones.where(lagged.gt(0), 0.0).mul(
            close_location.lt(R.LONG_THRESHOLD).astype(float), axis=0
        ),
        "short_only": short,
    }
    return {
        name: decision.shift(1).fillna(0.0).where(pool, 0.0)
        for name, decision in decisions.items()
    }


def _run_commodity_component_followup(data, inputs, carry, symbols, market) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    protocol = load_followup_protocol()
    output = FOLLOWUP_OUT / "commodity"
    output.mkdir(parents=True, exist_ok=True)
    rows, annual, daily_returns = [], [], {}
    labels = {
        "score_only": "分数单独（无展期筛选）",
        "clv_only": "CLV 单独（无展期筛选）",
        "carry_only": "正展期单独（无时间信号）",
        "score_and_carry_long": "分数 + 正展期（实际多头规则）",
        "clv_and_carry_long": "CLV + 正展期（替换后多头规则）",
        "short_only": "趋势 + 负展期空头",
    }
    for key, target in _commodity_component_targets(data, symbols, carry).items():
        aligned = target.reindex(index=inputs["index"], columns=inputs["columns"]).fillna(0.0)
        daily, positions, _ = D.simulate(aligned, inputs, S.BOOK)
        row, _ = D.summarize(
            labels[key],
            daily,
            positions,
            market,
            inputs["adjusted_open_delta"],
            D.sectors(),
        )
        row.update({"组件": key, "解释": labels[key]})
        rows.append(row)
        net = daily["净收益"].fillna(0.0)
        daily_returns[key] = net
        for year, part in net.groupby(net.index.year):
            annual.append(
                {
                    "组件": key,
                    "解释": labels[key],
                    "年份": int(year),
                    "净收益": float((1.0 + part).prod() - 1.0),
                    "日收益贡献求和_非复利": float(part.sum()),
                }
            )
    summary = pd.DataFrame(rows)
    annual_frame = pd.DataFrame(annual)
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    annual_frame.to_csv(output / "annual.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_returns).to_csv(
        output / "daily_returns.csv", encoding="utf-8-sig"
    )
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(12, 6.5))
    colors = {
        "score_only": "#315A7D",
        "clv_only": "#2A8C82",
        "carry_only": "#E2A23A",
        "score_and_carry_long": "#5D76B5",
        "clv_and_carry_long": "#44A39B",
        "short_only": "#D66B4D",
    }
    for key, values in daily_returns.items():
        axis.plot(
            values.index,
            values.cumsum(),
            color=colors[key],
            linewidth=1.8,
            label=labels[key],
        )
    axis.axhline(0, color="#77818A", linewidth=0.8)
    axis.set_title(
        "商品组件反事实组合：累计日净收益贡献（非复利、不可相加）",
        loc="left",
        color="#24313D",
        fontsize=13,
    )
    axis.set_ylabel("累计日净收益贡献")
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(axis)
    axis.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(output / "component_counterfactuals.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _style(axis) -> None:
    axis.grid(axis="y", color="#E5E8EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.spines["left"].set_color("#AAB1B8")
    axis.spines["bottom"].set_color("#AAB1B8")
    axis.tick_params(colors="#59636E", labelsize=9)


def _plot_results(
    daily_by_book: dict[str, pd.DataFrame],
    contribution_by_book: dict[str, pd.DataFrame],
    instrument_by_book: dict[str, pd.DataFrame],
    annual_contribution: pd.DataFrame,
    out: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    comparison_dir = out / "plots" / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    commodity_keys = ("commodity_original", "commodity_clv_swap")
    for key in commodity_keys:
        nav = (1.0 + daily_by_book[key]["净收益"].fillna(0.0)).cumprod()
        axes[0].plot(
            nav.index,
            nav,
            color=COLORS[key],
            linewidth=2.1,
            label=BOOK_LABELS[key],
        )
    axes[0].set_title("商品：净值对比 | 2011–2021", color="#24313D", loc="left", fontsize=12)
    axes[0].set_ylabel("净值（期初=1）")
    axes[0].legend(frameon=False, loc="upper left")
    _style(axes[0])

    key = "index_clv_swap"
    nav = (1.0 + daily_by_book[key]["净收益"].fillna(0.0)).cumprod()
    axes[1].plot(
        nav.index,
        nav,
        color=COLORS[key],
        linewidth=2.1,
        label=BOOK_LABELS[key],
    )
    axes[1].set_title("股指：净值 | IF/IH/IC 共同上市期起", color="#24313D", loc="left", fontsize=12)
    axes[1].set_ylabel("净值（期初=1）")
    axes[1].legend(frameon=False, loc="upper left")
    _style(axes[1])
    fig.suptitle("三策略研究期净值（1亿元、历史手续费、每边1跳）", color="#18232D", fontsize=15, x=0.08, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(comparison_dir / "nav_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for key, daily in daily_by_book.items():
        plot_performance(
            daily["净收益"].rename(BOOK_LABELS[key]),
            out / "plots" / key,
            title=f"{BOOK_LABELS[key]} | {daily.index.min():%Y-%m-%d} 至 {daily.index.max():%Y-%m-%d}",
        )

    # 年度贡献按逐日 PnL / 当日期初权益累加，与净收益的逐日简单收益口径相同。
    commodity_keys = ("commodity_original", "commodity_clv_swap")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, key in zip(axes, commodity_keys):
        table = annual_contribution[annual_contribution["策略ID"] == key].set_index("年份")
        x = table.index.to_numpy(dtype=int)
        bottom_pos = np.zeros(len(x))
        bottom_neg = np.zeros(len(x))
        for column, color, label in (
            ("多头腿贡献", COLORS["long"], "多头腿"),
            ("空头腿贡献", COLORS["short"], "空头腿"),
            ("交易成本贡献", COLORS["cost"], "交易成本"),
        ):
            values = table[column].to_numpy(dtype=float)
            bottoms = np.where(values >= 0, bottom_pos, bottom_neg)
            axis.bar(x, values, bottom=bottoms, width=0.72, color=color, label=label, edgecolor="white", linewidth=0.5)
            bottom_pos += np.maximum(values, 0.0)
            bottom_neg += np.minimum(values, 0.0)
        axis.plot(
            x,
            table["日收益贡献求和"],
            color="#24313D",
            marker="o",
            linewidth=1.5,
            markersize=4,
            label="贡献合计（未复利）",
        )
        axis.axhline(0, color="#68737D", linewidth=0.8)
        axis.set_title(BOOK_LABELS[key], color="#24313D", loc="left", fontsize=11)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        _style(axis)
    axes[0].legend(frameon=False, ncol=4, loc="upper left")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(year) for year in x])
    fig.suptitle("商品各组成部分年度收益贡献（逐日求和，未复利）", color="#18232D", fontsize=14, x=0.08, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "plots" / "commodity_annual_contributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    table = annual_contribution[annual_contribution["策略ID"] == "index_clv_swap"].set_index("年份")
    index_symbols = list(M.SYMBOLS)
    instrument_table = instrument_by_book["index_clv_swap"]
    yearly_symbols = instrument_table.groupby(instrument_table.index.year).sum()
    yearly_cost = table["交易成本贡献"]
    years = table.index.to_numpy(dtype=int)
    fig, axis = plt.subplots(figsize=(12, 6))
    bottom_pos = np.zeros(len(years))
    bottom_neg = np.zeros(len(years))
    for symbol in index_symbols:
        values = yearly_symbols[symbol].reindex(years, fill_value=0.0).to_numpy(dtype=float)
        bottoms = np.where(values >= 0, bottom_pos, bottom_neg)
        axis.bar(
            years,
            values,
            bottom=bottoms,
            width=0.68,
            color=COLORS[symbol],
            label=f"{symbol} 毛贡献",
            edgecolor="white",
            linewidth=0.5,
        )
        bottom_pos += np.maximum(values, 0.0)
        bottom_neg += np.minimum(values, 0.0)
    axis.bar(
        years + 0.38,
        yearly_cost.reindex(years).to_numpy(dtype=float),
        width=0.28,
        color=COLORS["cost"],
        alpha=0.9,
        label="交易成本",
    )
    axis.plot(
        years,
        table["日收益贡献求和"].to_numpy(dtype=float),
        color="#18232D",
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="贡献合计（未复利）",
    )
    axis.axhline(0, color="#68737D", linewidth=0.8)
    axis.set_xticks(years)
    axis.set_xticklabels([str(year) for year in years])
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axis.set_title("股指年度贡献：IF / IH / IC 与交易成本", color="#24313D", loc="left", fontsize=12)
    axis.legend(frameon=False, ncol=5, loc="upper left")
    _style(axis)
    fig.tight_layout()
    fig.savefig(out / "plots" / "index_annual_symbol_contributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for axis, key in zip(axes, ("commodity_original", "commodity_clv_swap", "index_clv_swap")):
        contribution = contribution_by_book[key].cumsum()
        for column, color in (
            ("多头腿", COLORS["long"]),
            ("空头腿", COLORS["short"]),
            ("交易成本及清零调整", COLORS["cost"]),
        ):
            axis.plot(contribution.index, contribution[column], color=color, linewidth=1.7, label=column)
        axis.set_title(BOOK_LABELS[key], color="#24313D", loc="left", fontsize=11)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        _style(axis)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("多空腿及交易成本的累计贡献（逐日收益贡献求和，未复利）", color="#18232D", fontsize=14, x=0.08, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "plots" / "cumulative_leg_contributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run_commodity():
    data = D._load(END_YEAR, COMMODITY_START)
    years = tuple(range(COMMODITY_START, END_YEAR + 1))
    inputs = D.ledger_inputs(data, years, pd.Timestamp(f"{END_YEAR}-12-31"))
    all_symbols = sorted({symbol for members in data.universe.values() for symbol in members})
    market = D.pool_open_return(data, inputs["universe"], inputs["index"])
    carry = S.roll_yield(data.returns.index, all_symbols)
    S.require_carry_coverage(
        carry,
        S.pool_mask(data.universe, data.returns.index, all_symbols),
    )

    books = {
        "commodity_original": S.BOOK.ID,
        "commodity_clv_swap": R.STRICT_SWAP_BOOK.ID,
    }
    results = {}
    summary_rows, annual_rows, position_rows, event_rows = [], [], [], []
    contributions, instruments = {}, {}
    for key, book_id in books.items():
        result_dir = C.RUNS_DIR / "strategies" / book_id / "commodity" / "research"
        required = ("daily.csv", "positions.csv", "events.csv", "summary.csv")
        missing = [name for name in required if not (result_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{BOOK_LABELS[key]} 账本须先由 backtest.engine 运行；"
                f"缺少 {', '.join(missing)}：{result_dir}"
            )
        daily = pd.read_csv(
            result_dir / "daily.csv", parse_dates=["交易日"]
        ).set_index("交易日")
        positions = pd.read_csv(
            result_dir / "positions.csv", parse_dates=["交易日"]
        )
        events = pd.read_csv(result_dir / "events.csv", parse_dates=["交易日"])
        expected_index = inputs["index"][inputs["index"] >= daily.index.min()]
        if not daily.index.equals(expected_index):
            raise ValueError(f"{book_id} engine 账本日期与本次研究归因输入不一致")
        row, _ = D.summarize(
            BOOK_LABELS[key],
            daily,
            positions,
            market,
            inputs["adjusted_open_delta"],
            D.sectors(),
        )
        row.update(
            {
                "策略ID": key,
                "策略": BOOK_LABELS[key],
                "市场": "商品",
                "计分起点": str(daily.index.min().date()),
                "计分终点": str(daily.index.max().date()),
            }
        )
        summary_rows.append(row)
        contribution, instrument_contribution = attribute_positions(
            daily, positions, inputs["adjusted_open_delta"]
        )
        annual_rows.extend(_annual_rows(key, daily, contribution))
        results[key] = daily
        contributions[key] = contribution
        instruments[key] = instrument_contribution
        position_rows.append(positions.assign(策略ID=key))
        event_rows.append(events.assign(策略ID=key))
    _run_commodity_component_followup(data, inputs, carry, all_symbols, market)
    return results, contributions, instruments, summary_rows, annual_rows, position_rows, event_rows


def _run_index():
    index, bars, _, _, _ = M._load_index_inputs()
    returns = (
        (bars["openw"].shift(-1) - bars["openw"])
        / bars["open"].where(bars["open"] > 0)
    )
    calendar = returns.index[(returns.index >= INDEX_START) & (returns.index <= INDEX_END)]
    returns = returns.reindex(calendar).dropna(how="any")
    if returns.empty:
        raise ValueError("IF/IH/IC 共同上市期没有完整的下一开盘收益")
    C.assert_no_holdout_dates(returns.index, what="股指 CLV 严格替换收益")
    all_returns = returns.copy()
    carry = M.load_index_carry(index)
    M.require_index_carry_coverage(carry, returns.index)
    target, market_clv = index_close_location_targets(index, bars, carry)
    first_nonzero = target.index[target.ne(0.0).any(axis=1)]
    if first_nonzero.empty:
        raise ValueError("股指 CLV 严格替换在窗口内没有非零目标")
    first_target_date = first_nonzero[0]
    returns = returns.loc[returns.index >= first_target_date]
    if returns.empty:
        raise ValueError("股指首次非零目标后没有可用收益")

    result_dir = (
        C.RUNS_DIR
        / "strategies"
        / IX.STRICT_SWAP_BOOK.ID
        / "index"
        / "research"
    )
    required = ("daily.csv", "positions.csv", "events.csv", "summary.csv")
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "股指账本须先由 backtest.engine 运行；"
            f"缺少 {', '.join(missing)}：{result_dir}"
        )
    daily = pd.read_csv(result_dir / "daily.csv", parse_dates=["交易日"]).set_index(
        "交易日"
    )
    positions = pd.read_csv(
        result_dir / "positions.csv", parse_dates=["交易日"]
    )
    events = pd.read_csv(result_dir / "events.csv", parse_dates=["交易日"])
    expected_index = returns.index[returns.index >= daily.index.min()]
    if not daily.index.equals(expected_index):
        raise ValueError("engine 中股指账本日期与本次研究归因输入不一致")
    market = returns.mean(axis=1).reindex(daily.index)
    summary = M._summary_row(
        "index_clv_swap",
        BOOK_LABELS["index_clv_swap"],
        daily,
        market,
    )
    summary.update(
        {
            "策略ID": "index_clv_swap",
            "市场": "股指",
            "计分起点": str(daily.index.min().date()),
            "计分终点": str(daily.index.max().date()),
            "信号首日": str(first_target_date.date()),
            "年化成本占权益": summary["年化交易成本"],
        }
    )
    adjusted_delta = (bars["openw"].shift(-1) - bars["openw"]).reindex(
        index=daily.index, columns=list(M.SYMBOLS)
    )
    contribution, instrument_contribution = attribute_positions(
        daily, positions, adjusted_delta
    )
    annual = _annual_rows("index_clv_swap", daily, contribution)
    annual_index = instrument_contribution.groupby(instrument_contribution.index.year).sum()
    for row in annual:
        year = row["年份"]
        for symbol in M.SYMBOLS:
            row[f"{symbol}毛贡献"] = float(annual_index.at[year, symbol])
    _run_index_individual(bars, all_returns, instrument_contribution)
    return (
        {"index_clv_swap": daily},
        {"index_clv_swap": contribution},
        {"index_clv_swap": instrument_contribution},
        [summary],
        annual,
        [positions.assign(策略ID="index_clv_swap")],
        [events.assign(策略ID="index_clv_swap")],
        market_clv.reindex(daily.index),
    )


def _run_index_individual(
    bars,
    all_returns,
    joint_contributions,
) -> None:
    protocol = load_followup_protocol()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    output = FOLLOWUP_OUT / "index_individual"
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summaries, annual_rows, daily_returns, contributions = [], [], {}, []
    position_rows = []
    for symbol in M.SYMBOLS:
        engine_dir = (
            C.RUNS_DIR
            / "strategies"
            / IX.STRICT_SWAP_BOOK.ID
            / "index"
            / "individual"
            / symbol
            / "research"
        )
        required = ("daily.csv", "positions.csv", "summary.csv")
        missing = [name for name in required if not (engine_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{symbol} 单品种账本须先由 backtest.engine --index-individuals 运行；"
                f"缺少 {', '.join(missing)}：{engine_dir}"
            )
        daily = pd.read_csv(
            engine_dir / "daily.csv", parse_dates=["交易日"]
        ).set_index("交易日")
        positions = pd.read_csv(
            engine_dir / "positions.csv", parse_dates=["交易日"]
        )
        if not daily.index.isin(all_returns.index).all():
            raise ValueError(f"{symbol} 单品种账本日期超出股指研究收益区间")
        start = daily.index.min()
        market = all_returns[symbol].reindex(daily.index)
        row = M._summary_row(
            f"index_{symbol}_own_clv",
            f"{symbol} 独立 CLV + 展期 + 空头",
            daily,
            market,
        )
        row["信号首日"] = str(start.date())
        summaries.append(row)
        adjusted_delta = (
            bars["openw"].shift(-1) - bars["openw"]
        ).reindex(index=daily.index, columns=[symbol])
        _, instrument_contribution = attribute_positions(
            daily, positions, adjusted_delta
        )
        joint = joint_contributions[symbol].reindex(daily.index).fillna(0.0)
        contribution = instrument_contribution[symbol].rename(
            f"{symbol}_standalone_gross"
        )
        contributions.append(
            pd.concat(
                [
                    contribution,
                    instrument_contribution["交易成本及清零调整"].rename(
                        f"{symbol}_standalone_cost"
                    ),
                    joint.rename(f"{symbol}_joint_gross"),
                ],
                axis=1,
            )
        )
        daily_returns[f"{symbol}_standalone_net"] = daily["净收益"].fillna(0.0)
        for year, part in daily.groupby(daily.index.year):
            annual_rows.append(
                {
                    "品种": symbol,
                    "年份": int(year),
                    "独立策略净收益": float(
                        daily.loc[part.index, "权益"].iloc[-1]
                        / daily.loc[part.index, "期初权益"].iloc[0]
                        - 1.0
                    ),
                    "独立策略日净收益贡献求和": float(
                        daily.loc[part.index, "净收益"].fillna(0.0).sum()
                    ),
                    "共同组合该品种毛贡献": float(joint.reindex(part.index).sum()),
                }
            )
        position_rows.append(positions.assign(独立策略=symbol))

        nav = (1.0 + daily["净收益"].fillna(0.0)).cumprod()
        peak = daily["权益"].cummax().clip(lower=INITIAL_CAPITAL)
        drawdown = daily["权益"] / peak - 1.0
        joint_cumulative = joint.cumsum()
        fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True)
        axes[0].plot(nav.index, nav, color=COLORS[symbol], linewidth=2.0)
        axes[0].set_ylabel("独立策略净值")
        axes[0].set_title(
            f"{symbol}：独立 CLV 策略净值",
            loc="left",
            color="#24313D",
            fontsize=12,
        )
        axes[1].plot(
            joint_cumulative.index,
            joint_cumulative,
            color="#7B6FA8",
            linewidth=1.8,
        )
        axes[1].axhline(0, color="#77818A", linewidth=0.8)
        axes[1].set_ylabel("共同组合毛贡献")
        axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[1].set_title(
            f"{symbol}：在 IF/IH/IC 共同策略中的累计毛贡献（非复利）",
            loc="left",
            color="#24313D",
            fontsize=11,
        )
        axes[2].fill_between(
            drawdown.index,
            drawdown.to_numpy(),
            0.0,
            color="#D66B4D",
            alpha=0.25,
            linewidth=0,
        )
        axes[2].plot(drawdown.index, drawdown, color="#D66B4D", linewidth=1.2)
        axes[2].set_ylabel("独立策略回撤")
        axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        for axis in axes:
            _style(axis)
        fig.suptitle(
            f"{symbol} | 2015共同上市期起；独立策略与共同组合贡献口径不同",
            x=0.08,
            ha="left",
            color="#18232D",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(plot_dir / f"{symbol}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(summaries).to_csv(
        output / "summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(annual_rows).to_csv(
        output / "annual.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(daily_returns).to_csv(
        output / "daily_returns.csv", encoding="utf-8-sig"
    )
    pd.concat(contributions, axis=1).to_csv(
        output / "contributions.csv", encoding="utf-8-sig"
    )
    pd.concat(position_rows, ignore_index=True).to_csv(
        output / "positions.csv", index=False, encoding="utf-8-sig"
    )
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_three_book_comparison(*, overwrite: bool = False) -> None:
    protocol = load_protocol()
    load_followup_protocol()
    if COMPARISON_OUT.exists():
        saved_protocol = COMPARISON_OUT / "protocol.json"
        if not overwrite or not saved_protocol.is_file():
            raise FileExistsError(f"结果目录已存在；仅可显式刷新本研究输出：{COMPARISON_OUT}")
        saved = json.loads(saved_protocol.read_text(encoding="utf-8"))
        if saved != protocol:
            raise FileExistsError(f"已有结果对应不同研究规格，不覆盖：{COMPARISON_OUT}")
    C.assert_research_only(C.RESEARCH_DIR)
    C.assert_research_only(C.INDEX_RESEARCH_DIR)
    commodity = _run_commodity()
    index_result = _run_index()

    daily_by_book = {**commodity[0], **index_result[0]}
    contribution_by_book = {**commodity[1], **index_result[1]}
    instrument_by_book = {**commodity[2], **index_result[2]}
    summaries = commodity[3] + index_result[3]
    annual_rows = commodity[4] + index_result[4]
    positions = commodity[5] + index_result[5]
    events = commodity[6] + index_result[6]
    annual_contribution = pd.DataFrame(annual_rows)

    COMPARISON_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(COMPARISON_OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual_contribution.to_csv(COMPARISON_OUT / "annual_contributions.csv", index=False, encoding="utf-8-sig")
    pd.concat(
        [daily["净收益"].rename(key) for key, daily in daily_by_book.items()],
        axis=1,
    ).to_csv(COMPARISON_OUT / "daily_returns.csv", encoding="utf-8-sig")
    pd.concat(
        [
            frame.assign(策略ID=key)
            for key, frame in contribution_by_book.items()
        ]
    ).to_csv(COMPARISON_OUT / "daily_leg_contributions.csv", encoding="utf-8-sig")
    pd.concat(
        [
            frame.assign(策略ID=key)
            for key, frame in instrument_by_book.items()
        ]
    ).to_csv(COMPARISON_OUT / "daily_instrument_contributions.csv", encoding="utf-8-sig")
    position_frames = [position for position in positions if not position.empty]
    event_frames = [event for event in events if not event.empty]
    if position_frames:
        pd.concat(position_frames, ignore_index=True).to_csv(
            COMPARISON_OUT / "positions.csv", index=False, encoding="utf-8-sig"
        )
    if event_frames:
        pd.concat(event_frames, ignore_index=True).to_csv(
            COMPARISON_OUT / "events.csv", index=False, encoding="utf-8-sig"
        )
    pd.Series(index_result[7], name="股指市场收盘位置").to_csv(
        COMPARISON_OUT / "index_market_close_location.csv", encoding="utf-8-sig"
    )
    (COMPARISON_OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _plot_results(
        daily_by_book,
        contribution_by_book,
        instrument_by_book,
        annual_contribution,
        COMPARISON_OUT,
    )
    pd.set_option("display.width", 240)
    columns = [
        "策略",
        "市场",
        "计分起点",
        "计分终点",
        "年化净收益",
        "净Sharpe",
        "最大回撤",
        "年化成本占权益",
    ]
    print(pd.DataFrame(summaries)[columns].round(3).to_string(index=False), flush=True)
    print(annual_contribution.round(4).to_string(index=False), flush=True)
    print(f"研究结果和图表：{COMPARISON_OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="商品收盘位置研究")
    parser.add_argument("--three-book-comparison", action="store_true", help="运行三策略收盘位置严格替换比较")
    parser.add_argument("--overwrite", action="store_true", help="仅刷新相同规格的三策略比较结果")
    args = parser.parse_args()
    if args.three_book_comparison:
        run_three_book_comparison(overwrite=args.overwrite)
    else:
        if args.overwrite:
            parser.error("--overwrite 仅适用于 --three-book-comparison")
        run()
