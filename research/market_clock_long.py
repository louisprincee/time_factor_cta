"""只做多全市场时钟：登记变体、稳健性网格和经济机制检验。只读商品研究期。

    python -m research.market_clock_long

规则在 config/research/market_clock_long_robustness.json，先登记后运行。不读 2022，不读 2023 年以后，不读 2026。
结果写到 runs/research/market_clock_long/。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import (
    PERIODS, RESEARCH_YEARS, deflated_sharpe, ledger_inputs, load_research, pbo_cscv,
    plot_performance, sharpe_ratio, simulate_target_lots,
)
from research.time_factor_screen import ols_hac
from strategies import market_clock_long as M
from strategies.book import _adx
from tfcta import config as C
from tfcta.costs import fee_tables, load_fees, load_ticks, slippage_tables
from tfcta.data import find_shard, load_shard, tag_session

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = C.CONFIG_DIR / "research" / "market_clock_long_robustness.json"
SECTOR_PATH = C.CONFIG_DIR / "research" / "time_factor_screen.json"
OUT = C.RUNS_DIR / "research" / "market_clock_long"
HAC_LAGS = 5
FORWARD_DAYS = 5
STATE_LOOKBACK = 250


def load_protocol(path=PROTOCOL_PATH) -> dict:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("status") != "exploratory_predeclared":
        raise ValueError("稳健性登记必须是事先登记")
    if "不读取2022及2023年以后" not in protocol.get("sample", "") or "不读 2026" not in protocol.get("sample", ""):
        raise ValueError("样本口径必须写明不读 2022 以后和 2026")
    ids = [v["id"] for v in protocol["variants"]]
    if protocol["candidate"] not in ids or len(set(ids)) != len(ids):
        raise ValueError("候选必须在登记变体里，且变体 id 不能重复")
    return protocol


def spec_of(variant: dict) -> dict:
    return {key: variant[key] for key in ("score", "vol_scale", "trend_cap", "weights")}


def sectors() -> dict[str, str]:
    table = json.loads(SECTOR_PATH.read_text(encoding="utf-8"))["sectors"]
    return {symbol: name for name, members in table.items() for symbol in members}


def day_session_prices(inputs: dict, symbols) -> dict:
    """每个交易日第一根日盘分钟的原始和复权开盘价，替换入场与估值价格，费用和滑点按它重算。"""
    index, columns = inputs["index"], inputs["columns"]
    raw, adjusted = {}, {}
    for symbol in symbols:
        path = find_shard(C.RESEARCH_DIR, symbol)
        if path is None:
            raise FileNotFoundError(f"{symbol} 缺少研究期分钟分片")
        minute = load_shard(symbol, columns=["trading_date", "open", "openw"])
        minute = minute[tag_session(minute.index).to_numpy() == C.SESSION_AM]
        first = minute.groupby(pd.to_datetime(minute["trading_date"]).dt.normalize()).first()
        raw[symbol], adjusted[symbol] = first["open"], first["openw"]
    raw = pd.DataFrame(raw).reindex(index=index, columns=columns)
    adjusted_full = pd.DataFrame(adjusted).reindex(columns=columns)
    opened, closed, rolled = fee_tables(load_fees(), raw)
    return {
        **inputs,
        "open_price": raw,
        "adjusted_open": adjusted_full.reindex(index),
        "adjusted_open_delta": (adjusted_full.shift(-1) - adjusted_full).reindex(index),
        "open_fee": opened, "close_fee": closed, "roll_close_fee": rolled,
        "slippage": slippage_tables(load_ticks(), raw, 1.0),
    }


def run_book(data, inputs: dict, spec: dict, universe: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = universe or inputs["universe"]
    columns = inputs["columns"]
    target = M.targets(data.factors.signed, data.factors.bars, universe, inputs["index"], columns, spec)
    keys = ("open_price", "close_price", "adjusted_open", "adjusted_open_delta", "adjusted_high",
            "adjusted_low", "rolls", "contract_codes")
    daily, positions, _ = simulate_target_lots(
        target, *(inputs[k] for k in keys), universe, inputs["multipliers"],
        inputs["open_fee"], inputs["close_fee"], inputs["roll_close_fee"], inputs["slippage"],
        initial_capital=100_000_000.0, margin_rate=0.20, exposure_limit=1.5, drawdown_limit=0.20,
    )
    return daily, positions


def pool_open_return(data, universe: dict, index) -> pd.Series:
    """池内开盘到开盘收益的等权平均，与账本的逐日净收益同一口径。"""
    pool = M.pool_mask(universe, index, data.returns.columns)
    return data.returns.reindex(index).where(pool).mean(axis=1).fillna(0.0)


def summarize(label: str, daily: pd.DataFrame, positions: pd.DataFrame, market: pd.Series,
              delta: pd.DataFrame, sector_of: dict) -> tuple[dict, list[dict]]:
    net = daily["净收益"].fillna(0.0)
    gross = (daily["毛盈亏"] / daily["期初权益"]).fillna(0.0)
    market = market.reindex(net.index).fillna(0.0)
    beta = float(net.cov(market) / market.var())
    alpha = net - beta * market
    years = sorted(set(net.index.year))
    month = net.index.to_period("M")
    strat_m = (1 + net).groupby(month).prod() - 1
    pool_m = (1 + market).groupby(month).prod() - 1
    up, down = pool_m > 0, pool_m < 0
    equity = daily["权益"]
    peak = np.maximum.accumulate(np.r_[100_000_000.0, equity.to_numpy()])[1:]
    share = {}
    if not positions.empty:
        moves = delta.stack()
        moves.index.names = ["交易日", "品种"]
        pnl = positions.set_index(["交易日", "品种"])["手数"] * positions.set_index(
            ["交易日", "品种"])["合约乘数"] * moves.reindex(positions.set_index(["交易日", "品种"]).index).to_numpy()
        by_sector = pnl.groupby(pnl.index.get_level_values("品种").map(sector_of)).sum()
        share = (by_sector / by_sector.abs().sum()).to_dict()
    annual = []
    for year in years:
        part = net[net.index.year == year]
        annual.append({"变体": label, "年份": year, "净收益": float((1 + part).prod() - 1),
                       "Sharpe": sharpe_ratio(part), "相对池子alpha": float(alpha[alpha.index.year == year].sum())})
    row = {
        "变体": label,
        "净Sharpe": sharpe_ratio(net),
        "毛Sharpe": sharpe_ratio(gross),
        "年化净收益": float((equity.iloc[-1] / 1e8) ** (PERIODS / len(net)) - 1),
        "年化波动": float(net.std() * np.sqrt(PERIODS)),
        "最大回撤": float((equity.to_numpy() / peak - 1).min()),
        "年化成本占权益": float((daily["交易费用"] / daily["期初权益"]).mean() * PERIODS),
        "平均敞口倍数": float(daily["敞口倍数"].mean()),
        "最大敞口倍数": float(daily["敞口倍数"].max()),
        "为正年份": int(sum(a["净收益"] > 0 for a in annual)),
        "相对池子beta": beta,
        "相对池子年化alpha": float(alpha.mean() * PERIODS),
        "alpha为正年份": int(sum(a["相对池子alpha"] > 0 for a in annual)),
        "上涨月捕获": float(strat_m[up].mean() / pool_m[up].mean()),
        "下跌月捕获": float(strat_m[down].mean() / pool_m[down].mean()),
        "熔断次数": int((daily["停止新开仓"].astype(bool) & ~daily["停止新开仓"].astype(bool).shift(1, fill_value=False)).sum()),
        "最大板块毛盈亏占比": float(max(share.values())) if share else np.nan,
        "最大占比板块": max(share, key=share.get) if share else "",
    }
    return row, annual


def gate_result(row: dict, gates: dict) -> dict:
    checks = {
        "净Sharpe": row["净Sharpe"] >= gates["net_sharpe_min"],
        "为正年份": row["为正年份"] >= gates["positive_years_min"],
        "相对池子alpha": row["相对池子年化alpha"] > 0 and row["alpha为正年份"] >= 5,
        "下跌月捕获": row["下跌月捕获"] <= gates["down_capture_max"],
        "板块占比": row["最大板块毛盈亏占比"] <= gates["max_sector_share_of_gross"],
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return {**checks, "全部通过": all(checks.values())}


def regimes(data, inputs, net: pd.Series, market: pd.Series, label: str) -> pd.DataFrame:
    """按事前状态（前一收盘已知）分档的年化收益和相对池子 alpha。分档切点用全样本三分位，只作诊断。"""
    index, universe = inputs["index"], inputs["universe"]
    close_ret = M.pool_return(data.factors.bars, universe, index)
    beta = float(net.cov(market) / market.var())
    alpha = net - beta * market
    adx = _adx(data.factors.bars).reindex(index=index, columns=inputs["columns"]).where(
        M.pool_mask(universe, index, inputs["columns"])).mean(axis=1)
    states = {
        "过去60日池子涨跌": close_ret.rolling(60).sum().shift(1),
        "过去20日池子波动": close_ret.rolling(20).std().shift(1),
        "池内平均ADX": adx.shift(1),
    }
    rows = []
    for name, state in states.items():
        bins = pd.qcut(state, 3, labels=["低", "中", "高"])
        for level in ["低", "中", "高"]:
            days = bins.index[bins.eq(level)]
            rows.append({"变体": label, "状态": name, "档": level, "天数": len(days),
                         "年化净收益": float(net.reindex(days).mean() * PERIODS),
                         "池子年化": float(market.reindex(days).mean() * PERIODS),
                         "相对池子年化alpha": float(alpha.reindex(days).mean() * PERIODS),
                         "Sharpe": sharpe_ratio(net.reindex(days))})
    return pd.DataFrame(rows)


def open_interest_change(symbols, index) -> pd.DataFrame:
    """主力合约收盘持仓量的日变化率；当天或前一天换过主力时记为缺失。"""
    out = {}
    for symbol in symbols:
        path = find_shard(C.RESEARCH_DIR / "open_interest", symbol)
        if path is None:
            continue
        C.assert_research_only(path)
        frame = pd.read_parquet(path, columns=["open_interest", "dominant_id", "trading_date"])
        last = frame.groupby(pd.to_datetime(frame["trading_date"]).dt.normalize()).last()
        C.assert_no_holdout_dates(last.index, f"{symbol} 持仓量")
        same = last["dominant_id"].eq(last["dominant_id"].shift(1))
        out[symbol] = (last["open_interest"] / last["open_interest"].shift(1) - 1).where(same)
    return pd.DataFrame(out).reindex(index)


def mechanism(data, inputs, protocol) -> pd.DataFrame:
    index, universe, columns = inputs["index"], inputs["universe"], inputs["columns"]
    pool = M.pool_mask(universe, index, columns)
    score = M.market_score(data.factors.signed, universe, index)
    bullish = score.gt(0)
    market_open = pool_open_return(data, universe, index)
    forward = sum(market_open.shift(-k) for k in range(1, FORWARD_DAYS + 1))
    outcome = forward - forward.mean()
    bars = data.factors.bars
    close, closew = bars["close"].reindex(columns=columns), bars["closew"].reindex(columns=columns)
    daily = ((closew - closew.shift(1)) / close.shift(1)).reindex(index).where(pool)
    mean = daily.mean(axis=1)
    volume = bars["volume"].reindex(columns=columns)
    surge = np.log(volume / volume.shift(1).rolling(20, min_periods=20).mean()).reindex(index).where(pool)
    states = {
        "co_movement": np.sign(daily).eq(np.sign(mean), axis=0).where(daily.notna()).mean(axis=1),
        "volume_surge": surge.replace([np.inf, -np.inf], np.nan).mean(axis=1),
        "oi_change": open_interest_change(columns, index).where(pool).mean(axis=1),
    }
    expected = {"co_movement": 1, "volume_surge": 1, "oi_change": -1}
    rows = []
    for name, state in states.items():
        median = state.shift(1).rolling(STATE_LOOKBACK, min_periods=STATE_LOOKBACK // 2).median()
        high = state.gt(median).where(state.notna() & median.notna())
        keep = bullish & high.notna() & outcome.notna()
        y, flag = outcome[keep], high[keep].astype(float)
        coef, t_value = ols_hac(y.to_numpy(), [flag.to_numpy()], HAC_LAGS)
        rows.append({
            "状态": name, "预期方向": "高>低" if expected[name] > 0 else "低>高",
            "看多日数": int(keep.sum()), "高半天数": int(flag.sum()),
            "高半5日超额": float(y[flag.eq(1)].mean()), "低半5日超额": float(y[flag.eq(0)].mean()),
            "高减低": coef, "NW_t": t_value,
            "符合预期且显著": bool(np.sign(coef) == expected[name] and abs(t_value) >= 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    protocol = load_protocol()
    M.load_config()
    data = load_research(1.0)
    C.assert_no_holdout_dates(data.returns.index, "全市场时钟研究收益")
    years = tuple(RESEARCH_YEARS)
    inputs = ledger_inputs(data, years, pd.Timestamp(f"{years[-1]}-12-31"))
    index, universe = inputs["index"], inputs["universe"]
    sector_of = sectors()
    delta = inputs["adjusted_open_delta"]
    OUT.mkdir(parents=True, exist_ok=True)

    rows, annual, daily_returns, regime_tables = [], [], {}, []
    for variant in protocol["variants"]:
        daily, positions = run_book(data, inputs, spec_of(variant))
        market = pool_open_return(data, universe, daily.index)
        row, years_rows = summarize(variant["id"], daily, positions, market, delta, sector_of)
        rows.append({**row, "角色": variant["role"]})
        annual.extend(years_rows)
        daily_returns[variant["id"]] = daily["净收益"].fillna(0.0)
        if variant["id"] in ("base", protocol["candidate"]):
            regime_tables.append(regimes(data, inputs, daily["净收益"].fillna(0.0), market, variant["id"]))
        print(f"{variant['id']}: 净Sharpe {row['净Sharpe']:.2f}", flush=True)
    variants = pd.DataFrame(rows)
    candidate = variants.set_index("变体").loc[protocol["candidate"]].to_dict()
    gates = gate_result(candidate, protocol["gates"])
    returns = pd.DataFrame(daily_returns)
    trial_sharpes = variants["净Sharpe"].tolist()
    deflated = {n: deflated_sharpe(returns[protocol["candidate"]], trial_sharpes, n)
                for n in protocol["robustness"]["trials_for_deflated_sharpe"]}
    pbo = pbo_cscv(returns, blocks=16)

    grid = protocol["robustness"]
    base_spec = spec_of(next(v for v in protocol["variants"] if v["id"] == protocol["candidate"]))
    cases = [("z窗口", str(w), {**base_spec, "z_window": w}, None) for w in grid["z_window"]]
    cases += [("持有日数", str(h), {**base_spec, "holding_days": h}, None) for h in grid["holding_days"]]
    cases += [("去掉因子", f, {**base_spec, "factors": tuple(x for x in M.FACTORS if x != f)}, None)
              for f in M.FACTORS]
    for sector in grid["drop_one_sector"]:
        kept = {y: [s for s in m if sector_of.get(s) != sector] for y, m in universe.items()}
        cases.append(("去掉板块", sector, base_spec, kept))
    rng = np.random.default_rng(grid["random_half_pool"]["seed"])
    symbols = list(inputs["columns"])
    for draw in range(grid["random_half_pool"]["draws"]):
        half = set(rng.choice(symbols, size=len(symbols) // 2, replace=False))
        cases.append(("随机半池", str(draw), base_spec, {y: [s for s in m if s in half] for y, m in universe.items()}))
    robust = []
    for group, value, spec, kept in cases:
        daily, positions = run_book(data, inputs, spec, kept)
        market = pool_open_return(data, kept or universe, daily.index)
        row, _ = summarize(f"{group}:{value}", daily, positions, market, delta, sector_of)
        robust.append({"组": group, "取值": value, **row})
    session = day_session_prices(inputs, symbols)
    daily, positions = run_book(data, session, base_spec)
    market = pool_open_return(data, universe, daily.index)
    row, _ = summarize("日盘开盘入场", daily, positions, market, session["adjusted_open_delta"], sector_of)
    robust.append({"组": "日盘开盘入场", "取值": "AM", **row})
    robust = pd.DataFrame(robust)
    tests = mechanism(data, inputs, protocol)

    variants.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.DataFrame(annual).to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    robust.to_csv(OUT / "robustness.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    tests.to_csv(OUT / "mechanism.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.concat(regime_tables).to_csv(OUT / "regimes.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    returns.to_csv(OUT / "daily_net_returns.csv", encoding="utf-8-sig", float_format="%.10f")
    plot_performance(returns, OUT / "plots", title="全市场时钟登记变体：研究期净值与回撤")
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(M.__file__), ROOT / "backtest" / "engine.py", PROTOCOL_PATH)}
    (OUT / "protocol.json").write_text(json.dumps({
        "protocol": protocol, "candidate_gates": gates, "deflated_sharpe": deflated, "pbo": pbo,
        "code_sha256": code, "window": "2016-2021", "costs": "历史手续费 + 上一年 tick 每边 1 跳",
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(variants.round(3).to_string(index=False))
    print("候选门槛：", gates)
    print("紧缩夏普：", {n: round(v["dsr"], 3) for n, v in deflated.items()}, "PBO：", round(pbo["pbo"], 3))
    print(tests.round(4).to_string(index=False))
    print(f"结果：{OUT}")


if __name__ == "__main__":
    main()
