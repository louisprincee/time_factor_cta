"""CLV 纯择时与因子倾斜的研究期复现入口。

    python -m research.index factor strategy-candidates

规则规格在 config/research/index_tactical_factor_candidates.json；策略冻结状态以各自配置为准。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.index.factors.index_time_factor_increment import (
    FACTORS,
    LOOKBACK,
    PERCENTILE,
    SYMBOLS,
    cash_clock,
)
from research.time_factor_screen import _position, fair_prices
from strategies.index_common import close_vol_scale, market_close_location
from tfcta import config as C
from tfcta.data import find_shard, read_frame

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_tactical_factor_candidates.json"
OUT = C.study_dir("index", "index_tactical_factor_candidates")
START, END = pd.Timestamp("2015-04-16"), pd.Timestamp("2021-12-30")
STANDARDIZATION_WINDOW = 250
RALLY_MULTIPLIER = 2.0
INITIAL_CAPITAL = 10_000_000.0
FACTOR_DIRECTIONS = {
    "IF": {"dfp_top3": 1},
    "IH": {},
    "IC": {"dfp_top3": 1, "ts_high": 1},
}
FACTOR_MINUTE_COLUMNS = [
    "trading_date", "open", "openw", "close", "closew",
    "highw", "loww", "volume", "total_turnover",
]


@dataclass(frozen=True)
class Candidate:
    ID: str
    NAME: str
    DETAIL: str
    GROSS: float
    DRAWDOWN_LIMIT: float = 0.10
    HALT_DAYS: int = 10
    SYMBOLS: tuple[str, ...] = SYMBOLS


TACTICAL_ONLY = Candidate(
    ID="index_tactical_clv_only",
    NAME="股指CLV纯择时（无动量底仓）",
    DETAIL="市场CLV<0.4时全预算等权择时；冲高×2；无动量底仓",
    GROSS=1.0,
)
FACTOR_TILT = Candidate(
    ID="index_tactical_clv_factor_select",
    NAME="股指CLV三个正向因子仓位倾斜择时",
    DETAIL="市场CLV<0.4时三品种均参与，仅计正向t>1的三个因子票，权重按1+票数倾斜；冲高×2",
    GROSS=2.0,
)


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    candidates = protocol.get("candidates", {})
    parameters = protocol.get("parameters", {})
    factor_directions = candidates.get(FACTOR_TILT.ID, {}).get("factor_directions", {})
    if (
        protocol.get("status") != "frozen_research_strategies"
        or "不运行 OOS" not in protocol.get("authorization", "")
        or "不读取 2022–2025" not in protocol.get("shared_basis", {}).get("sample", "")
        or sorted(candidates) != sorted((TACTICAL_ONLY.ID, FACTOR_TILT.ID))
        or tuple(FACTOR_DIRECTIONS) != SYMBOLS
        or FACTOR_DIRECTIONS != {
            "IF": {"dfp_top3": 1},
            "IH": {},
            "IC": {"dfp_top3": 1, "ts_high": 1},
        }
        or factor_directions != FACTOR_DIRECTIONS
        or parameters.get("standardization_window") != STANDARDIZATION_WINDOW
        or parameters.get("market_clv_threshold") != 0.4
        or parameters.get("rally_multiplier") != RALLY_MULTIPLIER
        or parameters.get("factor_allocation_rule") != "各品种预算份额 ∝ 1 + 有利因子票数；因子缺失按0票"
        or parameters.get("max_favorable_votes_by_symbol") != {"IF": 1, "IH": 0, "IC": 2}
        or parameters.get("candidate_factor_cells") != 3
    ):
        raise ValueError("纯择时/因子候选登记与实现不一致")
    return protocol


def symbol_factor_panel(
    frame: pd.DataFrame, *, partition: str
) -> pd.DataFrame:
    """只从单日已收盘分钟数据构造仓位倾斜所需因子。"""
    if partition not in ("research", "oos", "research_oos"):
        raise ValueError(f"未知因子数据分区：{partition}")
    frame = frame.sort_index(kind="mergesort").copy()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError("分钟索引必须是唯一的时间戳")
    if "trading_date" not in frame:
        raise KeyError("股指分钟分片缺少 trading_date")
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    if partition == "research":
        C.assert_no_holdout_dates(frame["trading_date"], "研究期因子输入")
    elif partition == "oos":
        C.assert_oos_dates(frame["trading_date"], "样本外因子输入")
        if frame["trading_date"].max() > pd.Timestamp(C.DEFAULT_OOS_END):
            raise C.HoldoutViolation("样本外因子输入超出2025年")
    else:
        research_dates = frame.loc[
            frame["trading_date"] < pd.Timestamp(C.OOS_START), "trading_date"
        ]
        oos_dates = frame.loc[
            frame["trading_date"] >= pd.Timestamp(C.OOS_START), "trading_date"
        ]
        C.assert_no_holdout_dates(research_dates, "OOS因子研究期预热")
        C.assert_oos_dates(oos_dates, "OOS因子样本外数据")
        if (
            research_dates.empty
            or oos_dates.empty
            or research_dates.max() > pd.Timestamp(C.RESEARCH_END)
            or oos_dates.max() > pd.Timestamp(C.DEFAULT_OOS_END)
        ):
            raise C.HoldoutViolation("OOS因子预热/计分日期超出研究期或样本外范围")

    clock = cash_clock(frame)
    if clock.empty:
        raise ValueError("股指分钟分片不含有效日盘时段")
    fair, _ = fair_prices(clock, LOOKBACK, PERCENTILE, None)
    rows = []
    for day, part in clock.groupby("trading_date", sort=True):
        if len(part) < 2 or part.index[-1].strftime("%H:%M") != "15:00":
            continue
        afternoon = part.loc[part.index.strftime("%H:%M") >= "13:01"]
        rows.append({
            "trading_date": day,
            "ts_low_pm": (
                _position(afternoon.loww.to_numpy(), "min")
                if len(afternoon) >= 2 else np.nan
            ),
            "ts_high": _position(part.highw.to_numpy(), "max"),
            "dfp_top3": float(fair.at[day, "dfp_top3"]) if day in fair.index else np.nan,
        })
    if not rows:
        raise ValueError("股指分钟分片未生成任何有效收盘因子")
    return pd.DataFrame(rows).set_index("trading_date")


def load_oos_factor_panels(end) -> dict[str, pd.DataFrame]:
    """用研究期末250个交易日预热指标，再单独计算请求范围内的OOS因子。"""
    end = pd.Timestamp(end).normalize()
    if end < pd.Timestamp(C.OOS_START) or end > pd.Timestamp(C.DEFAULT_OOS_END):
        raise C.HoldoutViolation(f"股指因子 OOS 截止日超出范围：{end.date()}")
    panels = {}
    for symbol in SYMBOLS:
        research_path = find_shard(C.INDEX_RESEARCH_DIR, symbol)
        if research_path is None:
            raise FileNotFoundError(
                f"找不到 {symbol} 股指研究期预热分片：{C.INDEX_RESEARCH_DIR}"
            )
        C.assert_research_only(research_path)
        if research_path.suffix == ".parquet":
            research_frame = read_frame(
                research_path,
                FACTOR_MINUTE_COLUMNS,
                filters=[
                    ("trading_date", ">=", pd.Timestamp("2020-01-01")),
                    ("trading_date", "<=", pd.Timestamp(C.RESEARCH_END)),
                ],
            )
        else:
            research_frame = read_frame(research_path, FACTOR_MINUTE_COLUMNS)
            research_frame = research_frame.loc[
                pd.to_datetime(research_frame["trading_date"]).dt.normalize()
                >= pd.Timestamp("2020-01-01")
            ]
            research_frame = research_frame.loc[
                pd.to_datetime(research_frame["trading_date"]).dt.normalize()
                <= pd.Timestamp(C.RESEARCH_END)
            ]
        research_dates = pd.to_datetime(research_frame["trading_date"]).dt.normalize()
        C.assert_no_holdout_dates(research_dates, f"{symbol} 股指研究期因子预热")
        research_days = pd.DatetimeIndex(research_dates.unique()).sort_values()
        if len(research_days) < LOOKBACK:
            raise ValueError(
                f"{symbol} OOS因子预热不足{LOOKBACK}个研究交易日：{len(research_days)}"
            )
        research_warmup = research_frame.loc[
            research_dates.isin(research_days[-LOOKBACK:])
        ]

        path = find_shard(C.INDEX_OOS_DIR, symbol)
        if path is None:
            raise FileNotFoundError(f"找不到 {symbol} 股指 OOS 分片：{C.INDEX_OOS_DIR}")
        C.assert_index_oos_only(path)
        if path.suffix != ".parquet":
            raise ValueError(f"股指 OOS 分片必须使用 Parquet：{path}")
        frame = read_frame(
            path,
            FACTOR_MINUTE_COLUMNS,
            filters=[
                ("trading_date", ">=", pd.Timestamp(C.OOS_START)),
                ("trading_date", "<=", end),
            ],
        )
        if "trading_date" not in frame:
            raise KeyError(f"{symbol} 股指 OOS 分片缺少 trading_date")
        dates = pd.to_datetime(frame["trading_date"]).dt.normalize()
        C.assert_oos_dates(dates, f"{symbol} 股指 OOS 因子")
        if dates.empty or dates.max() > end:
            raise ValueError(f"{symbol} 股指 OOS 因子日期为空或超过截止日")
        combined = pd.concat([research_warmup, frame]).sort_index(kind="mergesort")
        if combined.index.has_duplicates:
            raise ValueError(f"{symbol} 研究预热与 OOS 因子分钟索引重复")
        full_panel = symbol_factor_panel(combined, partition="research_oos")
        panels[symbol] = full_panel.loc[full_panel.index >= pd.Timestamp(C.OOS_START)]
        if panels[symbol].empty or panels[symbol].index.max() > end:
            raise ValueError(f"{symbol} OOS 收盘因子为空或超过截止日")
        print(f"{symbol} OOS 收盘因子完成", flush=True)
    return panels


def factor_votes(
    panels: dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成收盘时可用的逐品种方向性因子投票结果。"""
    scores = pd.DataFrame(0, index=index, columns=SYMBOLS, dtype="int64")
    available = pd.DataFrame(0, index=index, columns=SYMBOLS, dtype="int64")
    for symbol in SYMBOLS:
        for factor, direction in FACTOR_DIRECTIONS[symbol].items():
            values = panels[symbol][factor].reindex(index).astype("float64")
            history = values.shift(1)
            mean = history.rolling(
                STANDARDIZATION_WINDOW, min_periods=STANDARDIZATION_WINDOW
            ).mean()
            scale = history.rolling(
                STANDARDIZATION_WINDOW, min_periods=STANDARDIZATION_WINDOW
            ).std(ddof=1)
            z = (values - mean) / scale.where(scale > 0)
            valid = z.notna()
            favorable = (z * direction).gt(0)
            scores[symbol] += (valid & favorable).astype("int64")
            available[symbol] += valid.astype("int64")
    return scores, available


def factor_budget_shares(scores: pd.DataFrame) -> pd.DataFrame:
    """所有品种保留一份基础权重，再按有利因子票数倾斜。"""
    if tuple(scores.columns) != SYMBOLS:
        raise ValueError("因子票数必须包含 IF/IH/IC")
    values = scores.to_numpy(dtype="float64")
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > len(FACTORS)).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("因子票数必须是 0 至 3 的非负整数")
    raw = scores.astype("float64") + 1.0
    return raw.div(raw.sum(axis=1), axis=0)


def target_weights(
    candidate_id: str,
    market_clv: pd.Series,
    scores: pd.DataFrame,
    rally_z: pd.Series,
    vol_scale: pd.Series,
) -> pd.DataFrame:
    """生成信号目标并滞后一日，在下一实际开盘执行。"""
    index = market_clv.index
    if not scores.index.equals(index) or tuple(scores.columns) != SYMBOLS:
        raise ValueError("因子票数必须与市场 CLV 日期和品种对齐")
    if not rally_z.index.equals(index) or not vol_scale.index.equals(index):
        raise ValueError("冲高或波动缩放信号未与市场 CLV 日期对齐")
    gate = market_clv.lt(0.4)
    multiplier = pd.Series(
        np.where(rally_z.gt(0), RALLY_MULTIPLIER, 1.0), index=index
    )
    budget = gate.astype(float) * multiplier * vol_scale

    if candidate_id == TACTICAL_ONLY.ID:
        unshifted = pd.DataFrame(
            {symbol: budget for symbol in SYMBOLS}, index=index
        )
    elif candidate_id == FACTOR_TILT.ID:
        # 账本按三品种等分账户；exposure_limit=2，预算份额 1 对应目标权重 1.5。
        shares = factor_budget_shares(scores)
        unshifted = shares.mul(1.5 * budget, axis=0)
    else:
        raise ValueError(f"未知研究候选：{candidate_id}")
    return unshifted.shift(1).fillna(0.0)


def _write_candidate(
    engine,
    candidate: Candidate,
    target: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    contracts: pd.DataFrame,
    ticks: pd.DataFrame,
    scored: pd.DatetimeIndex,
    slippage,
    benchmark: pd.Series,
):
    daily, positions, events = engine._simulate_index_targets(
        target,
        scored,
        bars,
        contracts,
        ticks,
        END.year,
        SYMBOLS,
        exposure_limit=candidate.GROSS,
        drawdown_limit=candidate.DRAWDOWN_LIMIT,
        halt_days=candidate.HALT_DAYS,
        initial_capital=INITIAL_CAPITAL,
        open_limits=True,
        open_slippage=slippage,
    )
    summary, annual = engine.write_results(
        "index_research",
        candidate,
        daily,
        positions,
        events,
        list(SYMBOLS),
        initial_capital=INITIAL_CAPITAL,
        market="index",
        benchmark=benchmark,
    )
    result_dir = C.backtest_dir("index", candidate.ID, "research")
    target.reindex(scored).to_csv(
        result_dir / "targets.csv", encoding="utf-8-sig"
    )
    return daily, positions, summary, annual


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = load_protocol()
    from backtest import engine
    from research.index.factors.index_clv_cross_product_factors import load_panels
    from research.index.shared.index_execution_10m import load_opening_quotes, l1_slippage

    C.assert_research_only(C.INDEX_RESEARCH_DIR)
    index, bars, contracts, rally_z, ticks = engine._load_index_research_inputs()
    if index.min() < START or index.max() > pd.Timestamp("2021-12-31"):
        raise C.HoldoutViolation("候选回测行情超出股指研究分区")

    panels, factor_clv = load_panels()
    market_clv = market_close_location(bars, index)
    comparable_clv = market_clv.reindex(factor_clv.index)
    if comparable_clv.isna().any() or not np.allclose(
        comparable_clv.to_numpy(), factor_clv.mean(axis=1).to_numpy()
    ):
        raise ArithmeticError("生产口径市场 CLV 与因子面板计算不一致")

    open_to_open = (
        (bars["openw"].shift(-1) - bars["openw"]) / bars["open"]
    ).mean(axis=1, skipna=False)
    scale = close_vol_scale(open_to_open).fillna(1.0).reindex(index)
    rally_z = rally_z.reindex(index)
    scores, available = factor_votes(panels, index)

    targets = {
        TACTICAL_ONLY.ID: target_weights(
            TACTICAL_ONLY.ID, market_clv, scores, rally_z, scale
        ),
        FACTOR_TILT.ID: target_weights(
            FACTOR_TILT.ID, market_clv, scores, rally_z, scale
        ),
    }
    if any(not np.isfinite(target.to_numpy()).all() for target in targets.values()):
        raise ValueError("候选目标权重包含非有限值")
    open_returns = (
        (bars["openw"].shift(-1) - bars["openw"]) / bars["open"]
    ).dropna(how="any")
    scored = open_returns.index[
        (open_returns.index >= START) & (open_returns.index <= END)
    ]
    if scored.empty or scored.min() < START or scored.max() > END:
        raise ValueError("候选账本计分日期为空或越过研究样本")

    quotes = load_opening_quotes()
    slippage, _, _ = l1_slippage(quotes, bars, contracts, scored)
    held = pd.DataFrame(1.0, index=scored, columns=SYMBOLS)
    benchmark_daily, _, _ = engine._simulate_index_targets(
        held,
        scored,
        bars,
        contracts,
        ticks,
        END.year,
        SYMBOLS,
        exposure_limit=1.0,
        drawdown_limit=0.99,
        halt_days=10,
        initial_capital=INITIAL_CAPITAL,
        open_limits=True,
        open_slippage=slippage,
    )
    benchmark = benchmark_daily["净收益"]

    candidate_specs = (TACTICAL_ONLY, FACTOR_TILT)
    outputs = {}
    summary_rows, annual_rows = [], []
    for candidate in candidate_specs:
        daily, positions, summary, annual = _write_candidate(
            engine,
            candidate,
            targets[candidate.ID].reindex(scored).fillna(0.0),
            bars,
            contracts,
            ticks,
            scored,
            slippage,
            benchmark,
        )
        summary["平均实际敞口"] = float(daily["敞口倍数"].mean())
        outputs[candidate.ID] = (daily, positions, summary)
        summary_rows.append(summary)
        annual_rows.extend(annual)
        print(f"{candidate.NAME} 完成", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_table = pd.DataFrame(summary_rows)
    annual_table = pd.DataFrame(annual_rows)
    summary_table.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    annual_table.to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")

    base_daily = outputs[TACTICAL_ONLY.ID][0]["净收益"]
    factor_daily = outputs[FACTOR_TILT.ID][0]["净收益"]
    paired = pd.DataFrame({
        "纯择时净收益": base_daily,
        "因子选品种净收益": factor_daily,
    })
    paired["因子候选减纯择时_bp"] = (
        paired["因子选品种净收益"] - paired["纯择时净收益"]
    ) * 1e4
    paired.to_csv(OUT / "daily_comparison.csv", encoding="utf-8-sig")

    weak_close = market_clv.lt(0.4)
    shares = factor_budget_shares(scores)
    allocation_rows = []
    for symbol in SYMBOLS:
        allocation_rows.append({
            "品种": symbol,
            "市场弱收盘日": int(weak_close.sum()),
            "可用候选因子数": len(FACTOR_DIRECTIONS[symbol]),
            "平均有效因子数_弱收盘日": float(available.loc[weak_close, symbol].mean()),
            "有利因子票数均值_弱收盘日": float(scores.loc[weak_close, symbol].mean()),
            "平均预算份额_pct_弱收盘日": float(shares.loc[weak_close, symbol].mean() * 100),
        })
    allocation = pd.DataFrame(allocation_rows)
    allocation.to_csv(OUT / "factor_allocation.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 260, "display.max_columns", 35):
        print("\n候选策略汇总", flush=True)
        print(summary_table.round(4).to_string(index=False), flush=True)
        print("\n年度结果", flush=True)
        print(annual_table.round(4).to_string(index=False), flush=True)
        print("\n因子仓位份额", flush=True)
        print(allocation.round(4).to_string(index=False), flush=True)
    print(f"研究候选结果：{OUT}", flush=True)
    return summary_table, annual_table


if __name__ == "__main__":
    run()
