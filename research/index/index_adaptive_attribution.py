"""历史股指部件拆分与当前双半仓底仓对照；只读研究期。

    python -m research.index.index_adaptive_attribution
    python -m research.index.index_adaptive_attribution --base-momentum
    python -m research.index.index_adaptive_attribution --budget-transfer

默认入口复核历史自适应账本；--base-momentum 对照当前底仓，分别读取相应登记。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from backtest.engine import (
    INDEX_DEFAULT_CAPITAL,
    PERIODS,
    _load_index_research_inputs,
    _simulate_index_targets,
    deflated_sharpe,
    max_drawdown,
    newey_west_t,
    pbo_cscv,
    performance,
    plot_performance,
    sharpe_ratio,
)
from research.index.index_execution_10m import l1_slippage, load_opening_quotes
from research.index._adaptive_signal_helpers import (
    DRAWDOWN_LIMIT,
    GROSS,
    HALT_DAYS,
    ONE_SIDED_UP_BODY_MIN,
    ONE_SIDED_UP_CLV_MIN,
    RALLY_MULTIPLIER,
    adaptive_targets,
    blended_vol_scale,
    one_sided_down,
    one_sided_up,
)
from research.time_factor_screen import ols_hac
from strategies.index_allocation import allocation_multipliers
from strategies.index_common import SYMBOLS, THRESHOLD, close_vol_scale, high_open_z, market_close_location
from tfcta import config as C
from tfcta.data import daily_bars, load_shard

CONFIG_PATH = C.CONFIG_DIR / "research" / "index_adaptive_attribution.json"
OUT = C.study_dir("index", "index_adaptive_attribution")
EARLY_IF = C.INDEX_DATA_ROOT / "if_before_ih_ic" / "minute_shards" / "IF.parquet"
FIELDS = ("open", "openw", "close", "closew", "highw", "loww")
PARTS = ("rally", "skip_down", "one_sided_up", "rank_allocation", "ewma")
LADDER = (("核心", ()), ("+冲高", PARTS[:1]), ("+跳过单边下跌", PARTS[:2]),
          ("+单边上涨", PARTS[:3]), ("+收益排名分配", PARTS[:4]), ("+EWMA（完整）", PARTS))
LABELS = {"rally": "冲高", "skip_down": "跳过单边下跌", "one_sided_up": "单边上涨",
          "rank_allocation": "收益排名分配", "ewma": "EWMA"}
REGIME_WINDOW, VOL_HISTORY = 20, 250


def load_protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sample = protocol.get("sample", "")
    ladder = [step[0] for step in protocol.get("components", {}).get("ladder", [])]
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or "不读 2022 至 2025 样本外" not in sample
        or "不读 2026" not in sample
        or ladder != ["core", "+rally", "+skip_down", "+one_sided_up", "+rank_allocation", "+ewma"]
        or sorted(protocol.get("tests", {})) != ["T1_reversal_core", "T2_not_microstructure",
                                                  "T3_compensation_scales_with_risk",
                                                  "T4_timing_beyond_exposure", "T5_selection_accounting"]
    ):
        raise ValueError("部件拆分协议与实现或样本边界不一致")
    return protocol


def component_targets(parts, clv, frames, rally_z, scale20, scale_blend, rank):
    """按部件组合目标；全部部件等于冻结策略。"""
    parts = set(parts)
    index = clv.index
    weak = clv.lt(THRESHOLD)
    if "skip_down" in parts:
        weak = weak & ~one_sided_down(frames, index)
    long = weak
    if "one_sided_up" in parts:
        long = long | one_sided_up(frames, index, clv, ONE_SIDED_UP_BODY_MIN, ONE_SIDED_UP_CLV_MIN)
    tilt = pd.Series(np.where(rally_z.gt(0), RALLY_MULTIPLIER, 1.0), index=index) if "rally" in parts else 1.0
    scale = scale_blend if "ewma" in parts else scale20
    multipliers = rank if "rank_allocation" in parts else pd.DataFrame(1.0, index=index, columns=list(SYMBOLS))
    gross = long.astype(float) * scale * tilt
    return multipliers.shift(1).mul(gross.shift(1), axis=0).fillna(0.0)


def regime_labels(market_close: pd.Series) -> pd.DataFrame:
    """执行日的状态只用信号日收盘及以前：20 日收益方向、20 日波动相对此前 250 日中位数。"""
    momentum = market_close.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).sum()
    vol = market_close.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).std()
    median = vol.shift(1).rolling(VOL_HISTORY, min_periods=120).median()
    up, high = momentum.gt(0), vol.gt(median)
    label = pd.Series(np.select([up & ~high, up & high, ~up & ~high, ~up & high],
                                ["平稳上涨", "震荡上涨", "平稳下跌", "恐慌下跌"], default=""),
                      index=market_close.index)
    label = label.where(momentum.notna() & median.notna(), "")
    pct = vol.rolling(VOL_HISTORY + 1, min_periods=121).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    return pd.DataFrame({"状态": label.shift(1), "波动分位": pct.shift(1)})


def _ledger(target, scored, bars, contracts, ticks, slip):
    daily, _, _ = _simulate_index_targets(
        target.reindex(scored), scored, bars, contracts, ticks, 2021, tuple(SYMBOLS),
        exposure_limit=GROSS, drawdown_limit=DRAWDOWN_LIMIT, halt_days=HALT_DAYS,
        initial_capital=INDEX_DEFAULT_CAPITAL, open_limits=True, open_slippage=slip,
    )
    return daily


def _summary(name, daily, bench):
    net = daily["净收益"].fillna(0.0)
    m = performance(net)
    excess = net - bench
    row = {"账本": name, "年化收益": m["ann_return"], "年化波动": m["ann_vol"],
           "最大回撤": max_drawdown(daily["权益"], INDEX_DEFAULT_CAPITAL), "Sharpe": sharpe_ratio(net),
           "Calmar": m["calmar"], "平均敞口": float(daily["敞口倍数"].fillna(0.0).mean()),
           "成本占初始资金": float(daily["交易费用"].sum() / INDEX_DEFAULT_CAPITAL),
           "相对买入持有超额年化": float(excess.mean() * PERIODS),
           "超额NW_t": newey_west_t(excess.to_numpy(), lags=10)}
    for year, part in daily.groupby(daily.index.year):
        row[f"{year}"] = float(part["权益"].iloc[-1] / part["期初权益"].iloc[0] - 1.0)
    return row


def _signal_test(y, x, lags=5):
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    coef, t = ols_hac(frame["y"].to_numpy(), [frame["x"].to_numpy()], lags)
    years = frame.groupby(frame.index.year).apply(lambda g: np.polyfit(g["x"], g["y"], 1)[0] if g["x"].std() > 0 else np.nan)
    return coef, t, len(frame), int((years > 0).sum()), int(years.notna().sum())


def _intraday_segments(index, weak_exec):
    """执行日 开盘→09:31、09:31→15:00、收盘→次日开盘 三段，三品种等权，统一 09:30–15:00 时钟。"""
    parts = []
    for symbol in SYMBOLS:
        m = load_shard(symbol, directory=C.INDEX_RESEARCH_DIR, columns=["datetime", "trading_date", "open", "openw", "closew"])
        if "datetime" not in m.columns:
            m = m.reset_index()
        m["trading_date"] = pd.to_datetime(m["trading_date"]).dt.normalize()
        C.assert_no_holdout_dates(m["trading_date"], f"{symbol} 分钟")
        hm = m["datetime"].dt.strftime("%H:%M")
        day = m[(hm >= "09:30") & (hm <= "15:00")]
        g = day.groupby("trading_date")
        first_w, first = g["openw"].first(), g["open"].first()
        at_931 = day[day["datetime"].dt.strftime("%H:%M") <= "09:31"].groupby("trading_date")["closew"].last()
        close_w = g["closew"].last()
        frame = pd.DataFrame({
            "开盘到0931": (at_931 - first_w) / first,
            "0931到收盘": (close_w - at_931) / first,
            "收盘到次日开盘": (first_w.shift(-1) - close_w) / first,
        })
        parts.append(frame)
    seg = sum(p.reindex(index) for p in parts) / len(parts)
    return seg.loc[weak_exec.reindex(seg.index).fillna(False)]


def run() -> None:
    protocol = load_protocol()
    _, bars, contracts, _, ticks = _load_index_research_inputs()
    index = bars["close"].index
    frames = {f: bars[f].reindex(index=index, columns=list(SYMBOLS)) for f in FIELDS}
    clv = market_close_location(frames, index)
    rally_z = high_open_z(frames)
    o2o = ((frames["openw"].shift(-1) - frames["openw"]) / frames["open"]).mean(axis=1, skipna=False)
    scale20 = close_vol_scale(o2o).fillna(1.0)
    scale_blend = blended_vol_scale(o2o)
    rank = allocation_multipliers(frames, "relative_return_rank_20d")
    returns = ((frames["openw"].shift(-1) - frames["openw"]) / frames["open"]).dropna(how="any")
    scored = returns.loc[(returns.index >= "2015-04-16") & (returns.index <= "2021-12-31")].index
    C.assert_no_holdout_dates(scored, "部件拆分计分日")
    slip, bad, _ = l1_slippage(load_opening_quotes(), bars, contracts, scored)
    print(f"L1 不匹配品种日 {bad}", flush=True)

    build = lambda parts: component_targets(parts, clv, frames, rally_z, scale20, scale_blend, rank)
    full_target = build(PARTS)
    pd.testing.assert_frame_equal(full_target, adaptive_targets(clv, bars, rally_z=rally_z).reindex(columns=list(SYMBOLS)),
                                  check_names=False)
    held =pd.DataFrame(1.0, index=scored, columns=list(SYMBOLS))
    bench_daily, _, _ = _simulate_index_targets(
        held, scored, bars, contracts, ticks, 2021, tuple(SYMBOLS), exposure_limit=1.0, drawdown_limit=0.99,
        halt_days=HALT_DAYS, initial_capital=INDEX_DEFAULT_CAPITAL, open_limits=True, open_slippage=slip)
    bench = bench_daily["净收益"].fillna(0.0)

    books = {name: build(parts) for name, parts in LADDER}
    for part in PARTS:
        books[f"完整−{LABELS[part]}"] = build(tuple(p for p in PARTS if p != part))
    dailies = {name: _ledger(t, scored, bars, contracts, ticks, slip) for name, t in books.items()}
    full = dailies["+EWMA（完整）"]
    # 1000 万整数手下 0.37 倍常数仓位约每品种 1 手，指数上涨后取整为 0，账本口径做不出对照；
    # 改用按实现平均敞口缩放的买入持有日收益（买入持有本身的敞口也因取整低于 1）。
    exposure = float(full["敞口倍数"].fillna(0.0).mean())
    bench_exposure = float(bench_daily["敞口倍数"].fillna(0.0).mean())
    dailies["等权买入持有"] = bench_daily
    nets = {name: d["净收益"].fillna(0.0) for name, d in dailies.items()}
    nets["同敞口买入持有"] = bench * (exposure / bench_exposure)

    summary = pd.DataFrame([_summary(name, d, bench) for name, d in dailies.items()])
    scaled = nets["同敞口买入持有"]
    m = performance(scaled)
    summary = pd.concat([summary, pd.DataFrame([{
        "账本": "同敞口买入持有", "年化收益": m["ann_return"], "年化波动": m["ann_vol"],
        "最大回撤": max_drawdown((1 + scaled).cumprod(), 1.0), "Sharpe": sharpe_ratio(scaled),
        "Calmar": m["calmar"], "平均敞口": exposure, "相对买入持有超额年化": float((scaled - bench).mean() * PERIODS),
        **{f"{y}": float((1 + part).prod() - 1) for y, part in scaled.groupby(scaled.index.year)}}])], ignore_index=True)
    steps = []
    names = [name for name, _ in LADDER]
    for prev, cur in zip(names[:-1], names[1:]):
        d = nets[cur] - nets[prev]
        steps.append({"比较": f"{cur} − {prev}", "年化差": float(d.mean() * PERIODS),
                      "NW_t": newey_west_t(d.to_numpy(), lags=10),
                      "逐年为正": f"{int((d.groupby(d.index.year).sum() > 0).sum())}/7"})
    for part in PARTS:
        d = nets["+EWMA（完整）"] - nets[f"完整−{LABELS[part]}"]
        steps.append({"比较": f"完整 − 去掉{LABELS[part]}", "年化差": float(d.mean() * PERIODS),
                      "NW_t": newey_west_t(d.to_numpy(), lags=10),
                      "逐年为正": f"{int((d.groupby(d.index.year).sum() > 0).sum())}/7"})
    steps = pd.DataFrame(steps)

    # 市场状态
    market_close = ((frames["closew"] - frames["closew"].shift(1)) / frames["close"].shift(1)).mean(axis=1)
    reg = regime_labels(market_close).reindex(scored)
    held_long = full_target.reindex(scored).gt(0).any(axis=1)
    regime_rows = []
    for label in ("平稳上涨", "震荡上涨", "平稳下跌", "恐慌下跌"):
        mask = reg["状态"].eq(label)
        excess = (nets["+EWMA（完整）"] - bench)[mask]
        regime_rows.append({"状态": label, "天数": int(mask.sum()),
                            "策略年化": float(nets["+EWMA（完整）"][mask].mean() * PERIODS),
                            "买入持有年化": float(bench[mask].mean() * PERIODS),
                            "超额年化": float(excess.mean() * PERIODS),
                            "超额NW_t": newey_west_t(excess.to_numpy(), lags=10),
                            "同敞口买入持有超额年化": float((nets["同敞口买入持有"] - bench)[mask].mean() * PERIODS),
                            "平均敞口": float(full["敞口倍数"].fillna(0.0)[mask].mean()),
                            "持多日占比": float(held_long[mask].mean())})
    regimes = pd.DataFrame(regime_rows)

    # 机制检验
    y = returns.mean(axis=1).shift(-1).reindex(index)          # 信号日 t → 执行日 t+1 的开盘到再下一开盘
    common = (index >= "2015-04-16") & (index <= "2021-12-29")
    weak = clv.lt(THRESHOLD).astype(float)
    tests = []
    coef, t, n, pos, yrs = _signal_test(y[common], weak[common])
    q = pd.qcut(clv[common], 5, labels=False)
    tests.append({"检验": "T1 共同样本：弱收盘虚拟变量", "系数_bp": coef * 1e4, "t": t, "n": n, "逐年为正": f"{pos}/{yrs}",
                  "补充": f"最低五分位减最高五分位 {(y[common][q == 0].mean() - y[common][q == 4].mean()) * 1e4:.1f}bp"})
    C.assert_research_only(EARLY_IF)
    early = daily_bars(pd.read_parquet(EARLY_IF, columns=["trading_date", "open", "openw", "close", "closew", "highw", "loww", "datetime"]))
    early = early[(early.index >= "2011-01-04") & (early.index <= "2015-04-15")]
    C.assert_no_holdout_dates(early.index, "早期 IF")
    early_clv = (early["closew"] - early["loww"]) / (early["highw"] - early["loww"])
    early_y = ((early["openw"].shift(-2) - early["openw"].shift(-1)) / early["open"].shift(-1))
    mask = early.index <= "2015-04-14"
    coef, t, n, pos, yrs = _signal_test(early_y[mask], early_clv.lt(THRESHOLD).astype(float)[mask])
    tests.append({"检验": "T1 早期 IF：弱收盘虚拟变量", "系数_bp": coef * 1e4, "t": t, "n": n, "逐年为正": f"{pos}/{yrs}", "补充": "单品种，信号层面复制"})

    weak_exec = clv.lt(THRESHOLD).shift(1).reindex(scored).fillna(False)
    segments = _intraday_segments(scored, weak_exec)
    for col in segments:
        r = segments[col].dropna()
        tests.append({"检验": f"T2 弱收盘后 {col}", "系数_bp": r.mean() * 1e4, "t": newey_west_t(r.to_numpy(), lags=5),
                      "n": len(r), "逐年为正": f"{int((r.groupby(r.index.year).mean() > 0).sum())}/{r.index.year.nunique()}", "补充": "执行日分段均值"})

    weak_days = common & clv.lt(THRESHOLD).to_numpy()
    pct = regime_labels(market_close)["波动分位"].shift(-1).reindex(index)  # 信号日收盘时已知的分位
    coef, t, n, pos, yrs = _signal_test(y[weak_days], pct[weak_days])
    hi = y[weak_days & pct.gt(0.5).to_numpy()].mean(); lo = y[weak_days & pct.le(0.5).to_numpy()].mean()
    tests.append({"检验": "T3 弱收盘日：收益对波动分位", "系数_bp": coef * 1e4, "t": t, "n": n, "逐年为正": f"{pos}/{yrs}",
                  "补充": f"高波动半 {hi * 1e4:.1f}bp，低波动半 {lo * 1e4:.1f}bp"})

    d = nets["+EWMA（完整）"] - nets["同敞口买入持有"]
    tests.append({"检验": "T4 完整 − 同敞口买入持有", "系数_bp": d.mean() * 1e4, "t": newey_west_t(d.to_numpy(), lags=10),
                  "n": len(d), "逐年为正": f"{int((d.groupby(d.index.year).sum() > 0).sum())}/7",
                  "补充": f"策略平均敞口 {exposure:.3f}，买入持有 {bench_exposure:.3f}，缩放 {exposure / bench_exposure:.3f}"})
    strat, b = nets["+EWMA（完整）"], bench
    beta = float(strat.cov(b) / b.var())
    resid = strat - beta * b
    tests.append({"检验": "T4 相对买入持有 alpha（日）", "系数_bp": resid.mean() * 1e4, "t": newey_west_t(resid.to_numpy(), lags=10),
                  "n": len(resid), "逐年为正": f"{int((resid.groupby(resid.index.year).sum() > 0).sum())}/7",
                  "补充": f"beta {beta:.3f}，年化 alpha {resid.mean() * PERIODS:.2%}"})

    trial_names = [name for name in nets if name not in ("等权买入持有", "同敞口买入持有")]
    trials = [sharpe_ratio(nets[name]) for name in trial_names]
    dsr = deflated_sharpe(nets["+EWMA（完整）"], trials, n_trials=len(trials))
    pbo = pbo_cscv(pd.DataFrame({name: nets[name] for name in trial_names}), blocks=16)
    tests.append({"检验": "T5 Deflated Sharpe", "系数_bp": np.nan, "t": np.nan, "n": len(trials), "逐年为正": "",
                  "补充": f"DSR {dsr['dsr']:.3f}，观测 Sharpe {dsr['sr']:.2f}，选择门槛 {dsr['sr0']:.2f}"})
    tests.append({"检验": "T5 PBO（CSCV 16 段）", "系数_bp": np.nan, "t": np.nan, "n": len(trials), "逐年为正": "",
                  "补充": f"PBO {pbo.get('pbo', np.nan):.3f}"})
    tests = pd.DataFrame(tests)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary.to_csv(OUT / "components.csv", index=False, encoding="utf-8-sig")
    steps.to_csv(OUT / "component_steps.csv", index=False, encoding="utf-8-sig")
    regimes.to_csv(OUT / "regimes.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUT / "mechanism_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(nets).to_csv(OUT / "daily_net.csv", encoding="utf-8-sig")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(summary.round(4).to_string(index=False))
        print(steps.round(4).to_string(index=False))
        print(regimes.round(4).to_string(index=False))
        print(tests.round(3).to_string(index=False))
    print(f"结果：{OUT}", flush=True)


def base_target_components(clv, frames, rally_z, base_sign, *, scaled=True):
    """两半先缩放再同向净额，最后整体移到下一开盘；仅用于登记对照。"""
    market_open = ((frames["openw"].shift(-1) - frames["openw"]) / frames["open"]).mean(axis=1, skipna=False)
    scale = close_vol_scale(market_open).fillna(1.0) if scaled else 1.0
    tactical = clv.lt(THRESHOLD).astype(float) * 0.5 * np.where(rally_z.gt(0), 1.5, 1.0)
    parts = pd.DataFrame({"底仓": scale * 0.5 * base_sign, "择时": scale * tactical})
    parts["合计"] = parts.sum(axis=1)
    return parts.shift(1).fillna(0.0)


def _prior_momentum_sign(returns):
    """保留上一版63/126/252多数表决，仅用于已登记的历史对照。"""
    votes = []
    for window in (63, 126, 252):
        momentum = (1.0 + returns).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
        votes.append(pd.Series(np.where(momentum.isna() | momentum.gt(0), 1.0, -1.0), index=returns.index))
    return pd.Series(np.where(pd.concat(votes, axis=1).sum(axis=1).gt(0), 1.0, -1.0), index=returns.index)


def exposure_audit(daily, positions, target, raw_parts, bars, contracts):
    """同一账本逐日敞口桥接；每日重新取整一栏是诊断，不是另一条回测。"""
    from tfcta.costs import INDEX_MULTIPLIER

    index = daily.index
    target = target.reindex(index=index, columns=list(SYMBOLS))
    if target.lt(0).any().any() or target.isna().any().any():
        raise ValueError("多头敞口审计需要完整非负目标")
    unit = bars["open"].reindex(index=index, columns=list(SYMBOLS)).mul(pd.Series(INDEX_MULTIPLIER).reindex(SYMBOLS))
    lots = positions.pivot(index="交易日", columns="品种", values="手数").reindex(index=index, columns=list(SYMBOLS)).fillna(0.0)
    nominal = (lots.abs() * unit).sum(axis=1)
    actual = nominal / daily["期初权益"]
    reported = nominal / daily["权益"]
    if not np.allclose(nominal, daily["名义敞口"], rtol=1e-10, atol=1e-5) or not np.allclose(reported, daily["敞口倍数"], rtol=1e-10, atol=1e-12):
        raise ArithmeticError("逐日名义敞口无法由实际手数、原始开盘和乘数复算")
    halted = daily["停止新开仓"].astype(bool)
    live_target = target.mul((~halted).astype(float), axis=0)
    rounded = np.floor(live_target.mul(daily["期初权益"], axis=0) / len(SYMBOLS) / unit)
    clean = (
        target.ne(target.shift()).all(axis=1)
        & ~halted & ~halted.shift(1, fill_value=False)
        & daily["受阻委托数"].eq(0) & daily["开仓受限数"].eq(0)
        & daily["受阻委托数"].shift(1, fill_value=0).eq(0)
        & daily["开仓受限数"].shift(1, fill_value=0).eq(0)
        & daily["换月笔数"].eq(0)
        & ~contracts.reindex(index).ne(contracts.reindex(index).shift()).any(axis=1)
    )
    if not np.array_equal(lots.loc[clean].to_numpy(), rounded.loc[clean].to_numpy()):
        raise ArithmeticError("清洁执行日实际手数不等于目标预算向下取整")
    bridge = pd.DataFrame({
        "未缩放底仓": raw_parts["底仓"].reindex(index),
        "未缩放CLV": raw_parts["择时"].reindex(index),
        "缩放后目标": target.mean(axis=1),
        "停机后目标": live_target.mean(axis=1),
        "假设每日重算整数手": (rounded * unit).sum(axis=1) / daily["期初权益"],
        "实际开盘敞口": actual, "实际期末口径": reported,
    })
    bridge["未缩放总目标"] = bridge["未缩放底仓"] + bridge["未缩放CLV"]
    ratio = raw_parts["底仓"].reindex(index).div(bridge["未缩放总目标"].replace(0, np.nan)).fillna(0.0)
    bridge["缩放后底仓"] = target.mean(axis=1) * ratio
    bridge["缩放后CLV"] = target.mean(axis=1) * (1.0 - ratio)
    bridge["按目标比例分摊底仓"] = reported * ratio
    bridge["按目标比例分摊CLV"] = reported * (1.0 - ratio)
    checks = {"rows": len(index), "clean_execution_days": int(clean.sum()),
              "notional_and_exposure_match": True, "clean_lots_match": True,
              "halted_days": int(halted.sum()), "means": {k: float(v) for k, v in bridge.mean().items()},
              "interpretation": "同一历史账本的算术桥接，非因果归因；比例分摊不是独立底仓/CLV实际手数；目标不变保持手数可使实际不同于每日重算整数手"}
    return bridge, checks


def _macd_control_sign(returns):
    """原MACD底仓仅作本次已登记对照，不注册成策略。"""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    dif = wealth.ewm(span=12, adjust=False, min_periods=12).mean() - wealth.ewm(span=26, adjust=False, min_periods=26).mean()
    valid = dif.dropna()
    signal = pd.Series(np.nan, index=returns.index)
    if not valid.empty:
        signal.loc[valid.index] = valid.ewm(span=9, adjust=False, min_periods=9).mean()
    return pd.Series(np.where(signal.isna() | dif.gt(signal), 1.0, -1.0), index=returns.index)


def run_base_momentum(*, budget_transfer=False):
    """一个新规则、两个已知底仓对照；不得据结果选长度。"""
    from strategies.index_clv_rally_base_long_2x import (
        _base_budget,
        _close_to_close,
        budget_transfer_components,
    )

    study_id = "index_base_budget_transfer" if budget_transfer else "index_base_momentum"
    path = C.CONFIG_DIR / "research" / f"{study_id}.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    names = ("多头预算转移", "原中期多空", "常驻多头") if budget_transfer else ("中期动量多数表决", "原MACD", "常驻多头")
    windows = [21, 63, 126] if budget_transfer else [63, 126, 252]
    variants = (["long_only_budget_transfer_21_63_126", "prior_long_short_63_126_252_control", "always_long_half_control"]
                if budget_transfer else ["momentum_63_126_252_majority", "macd_12_26_9_control", "always_long_control"])
    if (
        protocol.get("status") != "declared_test_not_a_strategy"
        or protocol.get("design", {}).get("momentum_windows") != windows
        or protocol.get("variants") != variants
        or "不读2022至2025" not in protocol.get("sample", "")
        or "不读2026" not in protocol.get("sample", "")
    ):
        raise ValueError("底仓动量研究登记不一致，拒绝读取行情")
    index, bars, contracts, _, ticks = _load_index_research_inputs()
    C.assert_no_holdout_dates(index, "底仓动量研究")
    if index.min() != pd.Timestamp("2015-04-16") or index.max() != pd.Timestamp("2021-12-31"):
        raise ValueError("底仓动量共同样本覆盖不完整")
    clv, rally_z = market_close_location(bars, index), high_open_z(bars)
    market_close = _close_to_close(bars)
    signs = {
        names[0]: _base_budget(market_close) if budget_transfer else _prior_momentum_sign(market_close),
        names[1]: _prior_momentum_sign(market_close) if budget_transfer else _macd_control_sign(market_close),
        names[2]: pd.Series(1.0, index=index),
    }
    components = {name: base_target_components(clv, bars, rally_z, sign) for name, sign in signs.items()}
    if budget_transfer:
        components[names[0]] = budget_transfer_components(
            clv, bars, rally_z=rally_z, rally_multiplier=1.5
        )
    targets = {
        name: pd.DataFrame({symbol: part["合计"] for symbol in SYMBOLS}, index=index)
        for name, part in components.items()
    }
    if budget_transfer:
        expected = budget_transfer_components(
            clv, bars, rally_z=rally_z, rally_multiplier=1.5
        )
        expected_targets = pd.DataFrame(
            {symbol: expected["合计"] for symbol in SYMBOLS}, index=index
        )
        pd.testing.assert_frame_equal(targets[names[0]], expected_targets)
    market_open = ((bars["openw"].shift(-1) - bars["openw"]) / bars["open"]).mean(axis=1, skipna=False)
    scored = market_open.dropna().index
    if scored[-1] != pd.Timestamp("2021-12-30"):
        raise ValueError("底仓研究计分末日不符")
    slip, bad, _ = l1_slippage(load_opening_quotes(), bars, contracts, scored)
    out = C.study_dir("index", study_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"仅研究期 {scored[0].date()}..{scored[-1].date()}；L1不匹配品种日 {bad}", flush=True)

    held = pd.DataFrame(1.0, index=scored, columns=list(SYMBOLS))
    bench_daily, _, _ = _simulate_index_targets(
        held, scored, bars, contracts, ticks, 2021, tuple(SYMBOLS), exposure_limit=GROSS,
        drawdown_limit=0.99, halt_days=HALT_DAYS, initial_capital=INDEX_DEFAULT_CAPITAL,
        open_limits=True, open_slippage=slip,
    )
    bench = bench_daily["净收益"].fillna(0.0)
    dailies, summary, annual, gross_rows, audits = {}, [], [], [], {}
    for name, target in targets.items():
        print(f"计算 {name}", flush=True)
        daily, positions, events = _simulate_index_targets(
            target.reindex(scored), scored, bars, contracts, ticks, 2021, tuple(SYMBOLS),
            exposure_limit=GROSS, drawdown_limit=DRAWDOWN_LIMIT, halt_days=HALT_DAYS,
            initial_capital=INDEX_DEFAULT_CAPITAL, open_limits=True, open_slippage=slip,
        )
        dailies[name] = daily
        row = _summary(name, daily, bench)
        annual_net = [row[str(year)] for year in range(2015, 2022)]
        part = components[name].reindex(scored)
        signed = (positions["名义敞口"] * np.sign(positions["手数"])).groupby(positions["交易日"]).sum().reindex(scored, fill_value=0.0)
        row.update({
            "正收益年数": int(sum(value > 0 for value in annual_net)),
            "最差年度": min(annual_net), "年度收益标准差": float(np.std(annual_net, ddof=1)),
            "净目标非零日占比": float(part["合计"].ne(0).mean()),
            "平均目标绝对敞口": float(part["合计"].abs().mean()),
            "平均实现净敞口": float((signed / daily["权益"]).mean()),
            "底仓调档次数" if budget_transfer else "底仓翻向次数": int(signs[name].loc[scored].diff().abs().gt(0).sum()),
            "实际持仓日占比": float(daily["持仓总手数"].gt(0).mean()),
            "底仓预热日数": (0 if name == names[2] else int(market_close.rolling(
                max(windows) if name == names[0] else (252 if budget_transfer else 33),
                min_periods=max(windows) if name == names[0] else (252 if budget_transfer else 33),
            ).mean().reindex(scored).isna().sum())),
        })
        summary.append(row)
        for year, daily_year in daily.groupby(daily.index.year):
            annual.append({"账本": name, "年份": int(year), "净收益": row[str(year)],
                           "Sharpe": sharpe_ratio(daily_year["净收益"].fillna(0)),
                           "年度内最大回撤": max_drawdown(daily_year["权益"], float(daily_year["期初权益"].iloc[0]))})
        gross = part[["底仓", "择时"]].mul(market_open.reindex(scored), axis=0)
        gross["合计"] = gross.sum(axis=1)
        for year, values in gross.groupby(gross.index.year):
            gross_rows.append({"账本": name, "年份": int(year),
                               **{column: float(values[column].sum()) for column in gross.columns}})
        folder = out / name
        folder.mkdir(exist_ok=True)
        for filename, table in (("daily.csv", daily), ("targets.csv", target.reindex(scored)),
                                ("components.csv", part), ("gross_diagnostic.csv", gross)):
            table.to_csv(folder / filename, encoding="utf-8-sig")
        positions.to_csv(folder / "positions.csv", index=False, encoding="utf-8-sig")
        events.to_csv(folder / "events.csv", index=False, encoding="utf-8-sig")
        if budget_transfer and name in (names[0], names[2]):
            raw = (part[["未缩放底仓", "未缩放择时"]].rename(columns={"未缩放底仓": "底仓", "未缩放择时": "择时"})
                   if name == names[0] else base_target_components(clv, bars, rally_z, signs[name], scaled=False).reindex(scored))
            bridge, checks = exposure_audit(daily, positions, target, raw, bars, contracts)
            bridge.to_csv(folder / "exposure_audit.csv", encoding="utf-8-sig")
            audits[name] = checks

    summary = pd.DataFrame(summary)
    stats = summary.set_index("账本")
    comparisons = []
    for name in (*names[1:], "等权买入持有"):
        other = bench if name == "等权买入持有" else dailies[name]["净收益"]
        diff = dailies[names[0]]["净收益"] - other
        years_positive = int(diff.groupby(diff.index.year).sum().gt(0).sum())
        t = newey_west_t(diff.to_numpy(), lags=10)
        comparisons.append({"比较": f"{names[0]} - {name}", "日均差_bp": float(diff.mean() * 1e4),
                            "年化算术差": float(diff.mean() * PERIODS), "NW_t": t,
                            "年度差为正": years_positive, "增益显著": bool(t >= 2 and years_positive >= 5)})
    improvement = bool(
        stats.loc[names[0], "Sharpe"] > stats.loc[names[1], "Sharpe"]
        and stats.loc[names[0], "最大回撤"] >= stats.loc[names[1], "最大回撤"]
        and stats.loc[names[0], "最差年度"] >= stats.loc[names[1], "最差年度"]
        and (not budget_transfer or stats.loc[names[0], "平均敞口"] > stats.loc[names[1], "平均敞口"])
    )
    verification = {"sample_start": str(scored[0].date()), "sample_end": str(scored[-1].date()),
                    "new_rule_equals_default_targets": bool(budget_transfer), "descriptive_improvement_pass": improvement,
                    "oos_read": False, "variants": list(names),
                    "gross_diagnostic": "风控前目标乘等权开盘段收益的年度算术和；无费用、取整、停机，不是净收益或独立两账户归因"}
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(out / "annual.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(out / "comparisons.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gross_rows).to_csv(out / "gross_diagnostic_annual.csv", index=False, encoding="utf-8-sig")
    (out / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if audits:
        (out / "exposure_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    nets = pd.DataFrame({**{name: daily["净收益"] for name, daily in dailies.items()}, "等权买入持有": bench})
    nets.to_csv(out / "daily_net.csv", encoding="utf-8-sig")
    plot_performance(nets, out, title=f"2015–2021：{'多头预算转移' if budget_transfer else '历史多空底仓'}对照（研究期）")
    print(summary.round(4).to_string(index=False))
    print(pd.DataFrame(annual).round(4).to_string(index=False))
    print(pd.DataFrame(comparisons).round(4).to_string(index=False))
    print(f"描述改进门槛通过：{improvement}；结果：{out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--base-momentum", action="store_true", help="复核登记的历史多空底仓对照")
    group.add_argument("--budget-transfer", action="store_true", help="多头底仓动态预算转移及仓位审计")
    args = parser.parse_args()
    run_base_momentum(budget_transfer=args.budget_transfer) if args.base_momentum or args.budget_transfer else run()
