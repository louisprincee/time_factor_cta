"""冻结策略的组成部分、日内振幅与流动性代理：判断分数择时是不是伪相关。只读商品研究分片 2010–2021。

    python -m research.market_clock_mechanism

规则在 config/research/market_clock_mechanism.json，先登记后运行。只诊断，不改冻结策略。
结果在 runs/research/market_clock_mechanism/，图在其下 plots/。不读 2022 及以后。
"""
from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.engine import PERIODS, ledger_inputs, load_research, newey_west_t, sharpe_ratio
from research.market_clock_slow_short_20_60_120 import pool_open_return, sectors, simulate, summarize
from research.time_factor_screen import ols_hac
from strategies import market_clock_slow_short_20_60_120 as S
from tfcta import config as C
from tfcta.data import load_shard

CONFIG_PATH = C.CONFIG_DIR / "research" / "market_clock_mechanism.json"
OUT = C.RUNS_DIR / "research" / "market_clock_mechanism"
FIRST_YEAR = C.EARLY_START.year
YEARS = tuple(range(FIRST_YEAR, C.RESEARCH_END.year + 1))
LAGS, ALPHA_LAGS, MIN_NAMES = 5, 10, 10
TAIL_BARS = 60
ILLIQ_WINDOW = 250
BOOKS = ("full", "long_leg", "short_leg", "carry_only", "score_only", "pool")
LABELS = {"full": "冻结策略", "long_leg": "只有多头腿", "short_leg": "只有空头腿",
          "carry_only": "只看展期>0", "score_only": "只看分数>0", "pool": "全池常驻"}
# 参考调色板前六个类别色，按固定顺序分配；对比度不足 3:1 的几种靠线尾直接标注和 CSV 表补足
COLORS = {"full": "#2a78d6", "long_leg": "#eb6834", "short_leg": "#1baf7a", "carry_only": "#eda100",
          "score_only": "#e87ba4", "pool": "#6250d6"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def load_protocol(path=CONFIG_PATH) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "declared_test_not_a_strategy":
        raise ValueError("检验登记必须声明这不是策略")
    if "不读 2022" not in protocol.get("sample", "") or "不读 2026" not in protocol.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    if [b["id"] for b in protocol["components"]["books"]] != list(BOOKS):
        raise ValueError("登记的组成部分与实现不一致")
    return protocol


# --------------------------------------------------------------------------
# 收盘 t 已知的面板
# --------------------------------------------------------------------------
def day_shape(bars: dict, columns) -> dict:
    """振幅、当天收盘收益和收盘位置。复权列只做差，分母用原始收盘。"""
    close = bars["close"].reindex(columns=columns)
    high, low, closew = (bars[k].reindex(columns=columns) for k in ("highw", "loww", "closew"))
    width = high - low
    return {
        "range": width / close.where(close > 0),
        "same_day": S.close_return(bars, columns),
        "clv": ((closew - low) / width.where(width > 0)).clip(0, 1),
    }


def legs(score, lagged, trend, bear, pool) -> dict:
    """各组成部分的收盘决策（未后移）。"""
    ones = pd.DataFrame(1.0, index=lagged.index, columns=lagged.columns)
    long_side = S.long_weights(score, lagged)
    full = S.decision_weights(score, lagged, trend, bear)
    return {
        "full": full,
        "long_leg": long_side,
        "short_leg": full.clip(upper=0.0),
        "carry_only": ones.where(lagged.gt(0), 0.0),
        "score_only": ones.mul(score.gt(0).astype(float), axis=0),
        "pool": ones,
    }


def minute_liquidity(minute: pd.DataFrame, tail: int = TAIL_BARS) -> pd.DataFrame:
    """单品种分钟表 → 每个交易日的 Abdi–Ranaldo 价差估计、全天和尾盘 tick-rule 不平衡度。收盘 t 已知。"""
    m = minute.sort_index()
    day = pd.to_datetime(m["trading_date"]).dt.normalize()
    offset = m["closew"] - m["close"]                       # 加法复权偏移，日内不变；换回原始价再取对数
    hi, lo, cl = m["highw"] - offset, m["loww"] - offset, m["close"]
    ok = (hi > 0) & (lo > 0) & (cl > 0)
    eta = (np.log(hi.where(ok)) + np.log(lo.where(ok))) / 2
    c = np.log(cl.where(ok))
    eta_next = eta.groupby(day).shift(-1)
    term = 4 * (c - eta) * (c - eta_next)
    spread = np.sqrt(term.groupby(day).mean().clip(lower=0))
    step = np.sign(cl.groupby(day).diff())
    vol = m["volume"].where(step.notna())
    signed = step * vol
    rank_from_end = m.groupby(day).cumcount(ascending=False)
    last = rank_from_end < tail
    out = pd.DataFrame({
        "spread": spread,
        "imbalance": signed.groupby(day).sum() / vol.groupby(day).sum().replace(0, np.nan),
        "tail_imbalance": signed.where(last).groupby(day).sum() / vol.where(last).groupby(day).sum().replace(0, np.nan),
    })
    out.index.name = "trading_date"
    return out


def trailing_median_above(series: pd.Series, window: int = ILLIQ_WINDOW) -> pd.Series:
    """高于自己过去 window 天（不含当天）的中位数；历史不足为缺失。"""
    median = series.shift(1).rolling(window, min_periods=window // 2).median()
    return (series > median).where(median.notna())


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def slope(y: pd.Series, xs: list[pd.Series]) -> tuple[float, float]:
    frame = pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 30:
        return np.nan, np.nan
    return ols_hac(frame.iloc[:, 0].to_numpy(), [frame.iloc[:, i].to_numpy() for i in range(1, frame.shape[1])], LAGS)


def daily_ic(x: pd.DataFrame, y: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    ok = x.notna() & y.notna() & valid
    ic = x.where(ok).rank(axis=1).corrwith(y.where(ok).rank(axis=1), axis=1)
    return ic[ok.sum(axis=1) >= MIN_NAMES].dropna()


def yearly_sign(series_by_year: pd.Series) -> int:
    return int((series_by_year > 0).sum())


def run(protocol=None) -> None:
    protocol = protocol or load_protocol()
    data = load_research(1.0, first_year=FIRST_YEAR)
    C.assert_no_holdout_dates(data.returns.index, "机制检验收益")
    index = data.returns.index
    columns = sorted({s for members in data.universe.values() for s in members})
    pool = S.pool_mask(data.universe, index, columns)
    carry = S.roll_yield(index, columns)
    S.require_carry_coverage(carry, pool)
    lagged = carry.shift(1)
    bars = data.factors.bars
    score = S.market_score(data.factors.signed, data.universe, index)
    trend = S.symbol_trend(bars, columns).reindex(index)
    pool_close = S.close_return(bars, columns).reindex(index).where(pool).mean(axis=1)
    bear = S.market_trend(pool_close).lt(0)
    forward = data.returns.reindex(index=index, columns=columns).shift(-1).where(pool)
    shape = {k: v.reindex(index).where(pool) for k, v in day_shape(bars, columns).items()}
    next_close = shape["same_day"].shift(-1)
    pool_fwd, pool_next_close = forward.mean(axis=1), next_close.mean(axis=1)
    keep = index[index >= pd.Timestamp(f"{FIRST_YEAR}-01-01")]

    # ---- 1. 组成部分账本 ----
    inputs = ledger_inputs(data, YEARS, pd.Timestamp(C.RESEARCH_END))
    market = pool_open_return(data, inputs["universe"], inputs["index"])
    decisions = legs(score, lagged, trend, bear, pool)
    book_rows, yearly_rows, nets, attribution = [], [], {}, None
    for key in BOOKS:
        target = decisions[key].shift(1).fillna(0.0).where(pool, 0.0).reindex(
            index=inputs["index"], columns=inputs["columns"]).fillna(0.0)
        daily, positions, _ = simulate(target, inputs, S.BOOK)
        row, annual = summarize(LABELS[key], daily, positions, market, inputs["adjusted_open_delta"], sectors())
        net = daily["净收益"].fillna(0.0)
        nets[key] = net
        by_year = (1 + net).groupby(net.index.year).prod() - 1
        sharpe_y = net.groupby(net.index.year).apply(sharpe_ratio)
        book_rows.append({"组成": key, "名称": LABELS[key], **{k: row[k] for k in (
            "净Sharpe", "年化净收益", "年化波动", "最大回撤", "相对池子beta", "相对池子年化alpha", "alpha_t",
            "年化成本占权益", "平均敞口倍数")}, "为正年份": int((by_year > 0).sum()), "年数": len(by_year),
            "逐年Sharpe最小": float(sharpe_y.min()), "逐年Sharpe标准差": float(sharpe_y.std())})
        for year in by_year.index:
            yearly_rows.append({"组成": key, "名称": LABELS[key], "年份": int(year),
                                "净收益": float(by_year[year]), "Sharpe": float(sharpe_y[year])})
        if key == "full" and not positions.empty:
            moves = inputs["adjusted_open_delta"].stack()
            moves.index.names = ["交易日", "品种"]
            held = positions.set_index(["交易日", "品种"])
            pnl = held["手数"] * held["合约乘数"] * moves.reindex(held.index).to_numpy()
            side = np.where(held["手数"] > 0, "多头腿", "空头腿")
            equity = daily["期初权益"]
            frame = pd.DataFrame({"pnl": pnl.to_numpy(), "side": side,
                                  "day": held.index.get_level_values("交易日")})
            frame["ret"] = frame["pnl"] / frame["day"].map(equity).to_numpy()
            attribution = frame.groupby([frame["day"].dt.year, "side"])["ret"].sum().unstack(fill_value=0.0)
            attribution["成本"] = -(daily["交易费用"] / daily["期初权益"]).groupby(daily.index.year).sum()
    books = pd.DataFrame(book_rows)
    yearly = pd.DataFrame(yearly_rows)

    # ---- 2. 振幅、收盘位置 ----
    range_rows = []
    mkt = {k: v.mean(axis=1) for k, v in shape.items()}
    for name in ("range", "clv"):
        x = mkt[name].reindex(keep)
        for target_name, y in (("下一段", pool_fwd), ("次日收盘", pool_next_close)):
            b, t = slope(y.reindex(keep), [x])
            by_year = pd.Series({yr: slope(y.reindex(keep)[keep.year == yr], [x[keep.year == yr]])[0] for yr in YEARS})
            range_rows.append({"层面": "市场", "变量": name, "目标": target_name, "斜率": b, "NW_t": t,
                               "斜率为正年数": yearly_sign(by_year), "年数": len(YEARS)})
        ic = daily_ic(shape[name].reindex(keep), forward.reindex(keep), pool.reindex(keep))
        range_rows.append({"层面": "截面", "变量": name, "目标": "下一段", "斜率": float(ic.mean()),
                           "NW_t": newey_west_t(ic, LAGS),
                           "斜率为正年数": yearly_sign(ic.groupby(ic.index.year).mean()), "年数": len(YEARS)})
    same = mkt["same_day"].reindex(keep)
    contemporaneous = {
        "振幅与当天收益相关": float(mkt["range"].reindex(keep).corr(same)),
        "振幅与当天收益绝对值相关": float(mkt["range"].reindex(keep).corr(same.abs())),
        "收盘位置与当天收益相关": float(mkt["clv"].reindex(keep).corr(same)),
        "分数与当天收益相关": float(score.reindex(keep).corr(same)),
        "分数与振幅相关": float(score.reindex(keep).corr(mkt["range"].reindex(keep))),
        "分数与收盘位置相关": float(score.reindex(keep).corr(mkt["clv"].reindex(keep))),
    }

    # ---- 3. 伪相关：分数择时加控制 ----
    y = pool_fwd.reindex(keep)
    bull = score.reindex(keep).gt(0).astype(float).where(score.reindex(keep).notna())
    controls = {"当天池子收益": same, "池内平均振幅": mkt["range"].reindex(keep), "池内平均收盘位置": mkt["clv"].reindex(keep)}
    spurious = []
    for label, xs in (("只有分数", []), *(((f"加{k}", [v]) for k, v in controls.items())), ("全部控制", list(controls.values()))):
        b, t = slope(y, [bull, *xs])
        spurious.append({"模型": label, "分数项bp": b * 1e4, "分数项NW_t": t})
    full_by_year = {}
    for yr in YEARS:
        m = keep.year == yr
        full_by_year[yr] = slope(y[m], [bull[m], *(v[m] for v in controls.values())])[0]
    full_by_year = pd.Series(full_by_year)
    for label, x in controls.items():
        b, t = slope(y, [x])
        spurious.append({"模型": f"只有{label}", "分数项bp": np.nan, "分数项NW_t": np.nan, "控制项斜率": b, "控制项NW_t": t})
    spurious = pd.DataFrame(spurious)

    # ---- 4. 分钟数据流动性代理 ----
    liq = {k: {} for k in ("spread", "imbalance", "tail_imbalance")}
    for symbol in columns:
        minute = load_shard(symbol, columns=["closew", "close", "highw", "loww", "volume", "trading_date"])
        C.assert_no_holdout_dates(pd.to_datetime(minute["trading_date"]), f"{symbol} 分钟行情")
        daily_liq = minute_liquidity(minute)
        for k in liq:
            liq[k][symbol] = daily_liq[k]
    liq = {k: pd.DataFrame(v).reindex(index=index, columns=columns).where(pool) for k, v in liq.items()}
    pool_spread = liq["spread"].mean(axis=1)
    illiq = trailing_median_above(pool_spread).reindex(keep)
    tail_imb = liq["tail_imbalance"].mean(axis=1).reindex(keep)
    on = bull.eq(1)
    h1 = tail_imb[on].dropna().mean() - tail_imb[~on & bull.notna()].dropna().mean()
    h1_b, h1_t = slope(tail_imb, [bull])
    inter = (bull * illiq.astype(float)).where(illiq.notna())
    h2 = slope(y, [inter, bull, illiq.astype(float).where(illiq.notna())])
    h2_sub = {state: slope(y[illiq == state], [bull[illiq == state]]) for state in (True, False)}
    h3 = slope(y, [tail_imb])
    h3_joint = slope(y, [bull, tail_imb])
    liquidity = pd.DataFrame([
        {"假说": "H1 分数>0 日尾盘不平衡度更低", "值": h1, "NW_t": h1_t, "预期": "<0"},
        {"假说": "H2 分数×不流动 交乘项（下一段收益）", "值": h2[0] * 1e4, "NW_t": h2[1], "预期": ">0，bp/天"},
        {"假说": "H2 不流动日的分数择时差", "值": h2_sub[True][0] * 1e4, "NW_t": h2_sub[True][1], "预期": "bp/天"},
        {"假说": "H2 流动日的分数择时差", "值": h2_sub[False][0] * 1e4, "NW_t": h2_sub[False][1], "预期": "bp/天"},
        {"假说": "H3 尾盘不平衡度对下一段收益", "值": h3[0] * 1e4, "NW_t": h3[1], "预期": "<0，每单位 bp"},
        {"假说": "H3 同时放分数后，尾盘不平衡度斜率", "值": slope(y, [tail_imb, bull])[0] * 1e4,
         "NW_t": slope(y, [tail_imb, bull])[1], "预期": "<0"},
        {"假说": "H3 同时放尾盘不平衡度后，分数项", "值": h3_joint[0] * 1e4, "NW_t": h3_joint[1], "预期": "bp/天"},
    ])
    liq_desc = {"池内平均价差估计中位": float(pool_spread.reindex(keep).median()),
                "尾盘不平衡度均值": float(tail_imb.mean()), "尾盘不平衡度标准差": float(tail_imb.std()),
                "不流动日占比": float(illiq.mean())}

    # ---- 写结果 ----
    OUT.mkdir(parents=True, exist_ok=True)
    books.to_csv(OUT / "components.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT / "components_yearly.csv", index=False, encoding="utf-8-sig")
    attribution.to_csv(OUT / "full_book_attribution.csv", encoding="utf-8-sig")
    pd.DataFrame(nets).to_csv(OUT / "components_daily.csv", encoding="utf-8-sig")
    pd.DataFrame(range_rows).to_csv(OUT / "range_tests.csv", index=False, encoding="utf-8-sig")
    spurious.to_csv(OUT / "spurious_tests.csv", index=False, encoding="utf-8-sig")
    full_by_year.rename("全控制下分数项").to_csv(OUT / "spurious_by_year.csv", encoding="utf-8-sig")
    liquidity.to_csv(OUT / "liquidity_tests.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps({
        "protocol": protocol, "window": f"{keep.min().date()}..{keep.max().date()}",
        "contemporaneous": contemporaneous, "liquidity_descriptive": liq_desc,
        "score_full_control_positive_years": yearly_sign(full_by_year),
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    plot_components(pd.DataFrame(nets), yearly, attribution, OUT / "plots")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(books.round(3).to_string(index=False))
    print(yearly.pivot(index="年份", columns="名称", values="净收益")[[LABELS[k] for k in BOOKS]].round(3).to_string())
    print(attribution.round(4).to_string())
    print(pd.DataFrame(range_rows).round(4).to_string(index=False))
    print({k: round(v, 3) for k, v in contemporaneous.items()})
    print(spurious.round(3).to_string(index=False))
    print("全控制下分数项逐年（bp）:", (full_by_year * 1e4).round(1).to_dict())
    print(liquidity.round(3).to_string(index=False))
    print({k: round(v, 5) for k, v in liq_desc.items()})
    print(f"结果：{OUT}")


# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------
def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_components(nets: pd.DataFrame, yearly: pd.DataFrame, attribution: pd.DataFrame, out_dir) -> None:
    """累计净值（对数轴）、逐年净收益小多图、冻结策略多空腿逐年贡献。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 6))
    dates = nets.index.to_numpy(dtype="datetime64[ns]")
    ends = {}
    for key in BOOKS:
        nav = (1 + nets[key]).cumprod()
        ax.plot(dates, nav.to_numpy(), color=COLORS[key], lw=2.4 if key == "full" else 1.6, label=LABELS[key])
        ends[key] = float(nav.iloc[-1])
    # 线尾标注按对数间距错开，避免终值相近的两条重叠
    placed, gap = {}, 0.035
    for key in sorted(ends, key=ends.get):
        y = np.log10(ends[key])
        lower = max(placed.values(), default=-np.inf)
        placed[key] = max(y, lower + gap)
    for key, y in placed.items():
        ax.annotate(f"{LABELS[key]} {ends[key]:.2f}", (dates[-1], 10 ** y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ticks = [0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_title("各组成部分的累计净值（2011–2021，1 亿元，1 跳加手续费，对数轴）", color=INK, fontsize=12, loc="left")
    ax.set_ylabel("净值", color=MUTED)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _style(ax)
    ax.set_xlim(dates[0], dates[-1] + np.timedelta64(700, "D"))
    fig.tight_layout()
    fig.savefig(out_dir / "components_nav.png", dpi=150)
    plt.close(fig)

    table = yearly.pivot(index="年份", columns="组成", values="净收益")[list(BOOKS)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
    years = table.index.to_numpy()
    for ax, key in zip(axes.ravel(), BOOKS):
        values = table[key].to_numpy()
        ax.bar(years, values, color=COLORS[key], width=0.72, edgecolor="#fcfcfb", linewidth=1)
        ax.axhline(0, color=MUTED, lw=0.8)
        positive = int((values > 0).sum())
        ax.set_title(f"{LABELS[key]}：{positive}/{len(values)} 年为正", color=INK, fontsize=10, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        _style(ax)
    fig.suptitle("各组成部分的逐年净收益（2011–2021）", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "components_yearly.png", dpi=150)
    plt.close(fig)

    parts = [c for c in ("多头腿", "空头腿", "成本") if c in attribution.columns]
    fig, ax = plt.subplots(figsize=(11, 5))
    years = attribution.index.to_numpy()
    width = 0.26
    part_colors = {"多头腿": COLORS["long_leg"], "空头腿": COLORS["short_leg"], "成本": MUTED}
    for i, part in enumerate(parts):
        ax.bar(years + (i - 1) * width, attribution[part].to_numpy(), width=width - 0.02,
               color=part_colors[part], label=part)
    total = attribution[parts].sum(axis=1)
    ax.plot(years, total.to_numpy(), color=COLORS["full"], lw=2, marker="o", ms=8, label="合计（近似净收益）")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("冻结策略里多头腿、空头腿和成本的逐年贡献（占期初权益之和）", color=INK, fontsize=12, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=4)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "full_book_attribution.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run()
