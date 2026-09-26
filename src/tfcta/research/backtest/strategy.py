"""自选因子、自选板块的等权周频书。研究期回测、2022 验证、样本外测试共用这一层。

因子写法与 ``config/validation_2022_plan.json`` 一致：``名字:符号``，符号乘在**原始值**上。
已定向因子可以只写名字，取 ``library.SIGNED_PRIORS`` 里的先验方向；
无方向因子（外部数据、截面候选等）必须显式给出 ``+1`` 或 ``-1``。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config as C
from ...data import bars as B
from ...data import sectors
from ...factors import library
from ..analysis import stats
from . import engine

REBALANCE = "weekly_last_trading_day"
EXECUTION = "next_trading_day_open"
DEFAULT_MIN_SHARPE = 0.5
DEFAULT_MIN_ANN_RETURN = 0.0


@dataclass
class BookConfig:
    factors: dict[str, float]
    pools: list[str] = field(default_factory=list)
    merge_pools: bool = False
    symbols: list[str] | None = None
    fee_rate: float = C.FEE_BASE
    slippage_ticks: float = C.SLIPPAGE_TICKS
    min_net_sharpe: float = DEFAULT_MIN_SHARPE
    min_net_ann_return: float = DEFAULT_MIN_ANN_RETURN

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------
def _parse_sign(value, name: str) -> float:
    try:
        sign = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"因子 {name} 的符号 {value!r} 不是 +1 或 -1") from None
    if sign not in (1.0, -1.0):
        raise ValueError(f"因子 {name} 的符号只能是 +1 或 -1，收到 {value!r}")
    return sign


def parse_factor_specs(specs) -> dict[str, float]:
    """``["ts_high", "carry_main_sub_yield:+1"]`` 或 ``{"ts_high": -1}`` → {名字: 原始值上的符号}。"""
    if isinstance(specs, dict):
        items = [(str(k), v) for k, v in specs.items()]
    else:
        items = []
        for spec in specs:
            name, _, sign = str(spec).partition(":")
            items.append((name.strip(), sign.strip() or None))
    out: dict[str, float] = {}
    for name, sign in items:
        if not name:
            raise ValueError("因子名为空")
        if sign is None:
            if name not in library.SIGNED_PRIORS:
                raise ValueError(f"{name} 没有事前方向，必须写成 {name}:+1 或 {name}:-1")
            sign = library.SIGNED_PRIORS[name]
        if name in out:
            raise ValueError(f"因子 {name} 重复")
        out[name] = _parse_sign(sign, name)
    if not out:
        raise ValueError("至少要选一个因子")
    return out


def add_book_args(parser, *, with_criteria: bool = False) -> None:
    parser.add_argument("--config", default=None,
                        help="JSON 配置文件；命令行参数覆盖文件里的同名项")
    parser.add_argument("--factors", nargs="+", default=None,
                        help="等权因子，写法 名字 或 名字:+1/-1（符号乘在原始值上）")
    parser.add_argument("--pools", nargs="*", default=None,
                        help=f"板块，可多选，每个单独成书；{sectors.ALL_POOL} 表示不分板块")
    parser.add_argument("--merge-pools", action="store_true", default=None,
                        help="多个板块之外再合成一本")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="与板块取交集；不传板块时这些品种合成一本")
    parser.add_argument("--fee-rate", type=float, default=None)
    parser.add_argument("--slippage-ticks", type=float, default=None)
    if with_criteria:
        parser.add_argument("--min-sharpe", type=float, default=None,
                            help=f"验证通过所需的扣费后 Sharpe，默认 {DEFAULT_MIN_SHARPE}")
        parser.add_argument("--min-ann-return", type=float, default=None,
                            help=f"验证通过所需的扣费后年化，默认 {DEFAULT_MIN_ANN_RETURN}")
    parser.add_argument("--list-pools", action="store_true", help="打印板块分类后退出")
    parser.add_argument("--list-factors", action="store_true",
                        help="打印第 3 步因子目录后退出")


def config_from_args(args) -> BookConfig:
    raw: dict = {}
    if args.config:
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    criteria = raw.get("pass_criteria", {})

    def pick(cli, key, default):
        return cli if cli is not None else raw.get(key, default)

    def criterion(cli, key, default):
        return cli if cli is not None else criteria.get(key, default)

    factors = pick(args.factors, "factors", dict(C.FACTOR_SIGNS))
    symbols = pick(args.symbols, "symbols", None)
    return BookConfig(
        factors=parse_factor_specs(factors),
        pools=sectors.parse_pools(pick(args.pools, "pools", [])),
        merge_pools=bool(pick(args.merge_pools, "merge_pools", False)),
        symbols=[s.upper() for s in symbols] if symbols else None,
        fee_rate=float(pick(args.fee_rate, "fee_rate", C.FEE_BASE)),
        slippage_ticks=float(pick(args.slippage_ticks, "slippage_ticks", C.SLIPPAGE_TICKS)),
        min_net_sharpe=float(criterion(getattr(args, "min_sharpe", None),
                                       "min_net_sharpe", DEFAULT_MIN_SHARPE)),
        min_net_ann_return=float(criterion(getattr(args, "min_ann_return", None),
                                           "min_net_ann_return", DEFAULT_MIN_ANN_RETURN)),
    )


def factor_catalog_path() -> Path:
    return C.RESEARCH_OUT_DIR / "factor_catalog.csv"


def print_catalog() -> int:
    path = factor_catalog_path()
    if not path.exists():
        print(f"找不到因子目录 {path}，请先运行 step3_build_factors.py")
        return 2
    table = pd.read_csv(path)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.to_string(index=False))
    return 0


def handle_listing(args) -> int | None:
    if args.list_pools:
        print(sectors.catalog())
        return 0
    if args.list_factors:
        return print_catalog()
    return None


# --------------------------------------------------------------------------
# 池子与信号
# --------------------------------------------------------------------------
def case_labels(cfg: BookConfig) -> list[str]:
    if not cfg.pools:
        return ["指定品种" if cfg.symbols else sectors.ALL_POOL]
    labels = list(cfg.pools)
    if cfg.merge_pools and len(cfg.pools) > 1:
        labels.append("+".join(cfg.pools))
    return labels


def resolve_cases(cfg: BookConfig, available: list[str],
                  labels: list[str] | None = None) -> dict[str, list[str]]:
    """每个板块单独成书。不传板块时，全部可用品种或命令行指定品种合成一本。"""
    allowed = set(available)
    if cfg.symbols:
        allowed &= set(cfg.symbols)
    cases = {}
    for label in (case_labels(cfg) if labels is None else labels):
        if label in (sectors.ALL_POOL, "指定品种"):
            cases[label] = sorted(allowed)
        else:
            cases[label] = sorted({s for name in label.split("+")
                                   for s in sectors.symbols_in(name) if s in allowed})
    return cases


def candidate_symbols(cfg: BookConfig) -> list[str]:
    """验证与样本外需要加载分钟数据的候选品种（尚未按时点品种池筛选）。"""
    pool = {s for name in (cfg.pools or [sectors.ALL_POOL]) for s in sectors.symbols_in(name)}
    pool &= set(C.COMMODITY_SYMBOLS)
    if cfg.symbols:
        pool &= set(cfg.symbols)
    return sorted(pool)


def equal_weight_signal(signal_set: library.SignalSet, factors: dict[str, float]) -> pd.DataFrame:
    """每个因子乘上符号后做事前 z 分数，再等权，截到 [-1, 1]。"""
    frames = [library.trail_z(signal_set.raw(name) * sign) for name, sign in factors.items()]
    return library.average_signals(frames).clip(-1, 1)


def fingerprint(cfg: BookConfig, case: str) -> str:
    """同一本书在验证台账和样本外台账里的唯一标识。通过门槛不参与指纹。"""
    payload = {
        "factors": {k: cfg.factors[k] for k in sorted(cfg.factors)},
        "case": case,
        "symbols": sorted(cfg.symbols) if cfg.symbols else None,
        "fee_rate": cfg.fee_rate,
        "slippage_ticks": cfg.slippage_ticks,
        "lookback": C.IC_REFERENCE_LOOKBACK,
        "pct": C.IC_REFERENCE_PCT,
        "z_window": library.Z_WINDOW,
        "z_min": library.Z_MIN,
        "rebalance": REBALANCE,
        "execution": EXECUTION,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def factor_label(factors: dict[str, float]) -> str:
    return ",".join(f"{name}:{int(sign):+d}" for name, sign in factors.items())


# --------------------------------------------------------------------------
# 回测与汇总
# --------------------------------------------------------------------------
def portfolio(signal: pd.DataFrame,
              day_ret: pd.DataFrame,
              universe: dict,
              members: list[str],
              fee: float,
              slippage: pd.DataFrame | None):
    selected = [s for s in members if s in day_ret.columns]
    scoped = {int(y): [s for s in syms if s in selected] for y, syms in universe.items()}
    position = engine.execute_position(engine.weekly(signal.reindex(columns=selected)))
    gross = engine.run_book(position, day_ret, scoped, 0.0)
    net = engine.run_book(position, day_ret, scoped, fee, slippage=slippage)
    return selected, scoped, position, gross, net


def period_row(case: str, period: str, gross: pd.Series, net: pd.Series,
               universe: dict, position: pd.DataFrame,
               turnover_years: list[int], n_symbols: int) -> dict:
    gross_perf = stats.performance(gross)
    net_perf = stats.performance(net)
    return {
        "universe": case,
        "period": period,
        "n_symbols": n_symbols,
        "gross_ann_return": gross_perf["ann_return"],
        "gross_sharpe": stats.sharpe_ratio(gross),
        "gross_max_drawdown": gross_perf["max_drawdown"],
        "net_ann_return": net_perf["ann_return"],
        "net_ann_vol": net_perf["ann_vol"],
        "net_sharpe": stats.sharpe_ratio(net),
        "net_ret_risk": net_perf["ret_risk"],
        "net_max_drawdown": net_perf["max_drawdown"],
        "net_calmar": net_perf["calmar"],
        "net_n_days": net_perf["n_days"],
        "turnover": engine.annual_turnover(position, universe, turnover_years),
    }


def evaluate_cases(signal: pd.DataFrame,
                   day_ret: pd.DataFrame,
                   universe: dict,
                   cases: dict[str, list[str]],
                   years: list[int],
                   fee: float,
                   slippage: pd.DataFrame | None,
                   overall_label: str,
                   window: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> pd.DataFrame:
    """每个池子一本书：整段一行 + 逐年一行，整段那一行附上合成信号的时序 IC。"""
    fwd = B.forward_return(day_ret)
    rows = []
    for case, members in cases.items():
        if not members:
            rows.append({"universe": case, "period": overall_label, "n_symbols": 0})
            continue
        _, scoped, position, gross, net = portfolio(
            signal, day_ret, universe, members, fee, slippage)
        if window is not None:
            lo, hi = window
            gross = gross.loc[(gross.index >= lo) & (gross.index <= hi)]
            net = net.loc[(net.index >= lo) & (net.index <= hi)]
        gross_all = engine.stitch_test_years(gross, years)
        net_all = engine.stitch_test_years(net, years)
        in_pool = sorted({s for y in years for s in scoped.get(int(y), [])})
        row = period_row(case, overall_label, gross_all, net_all, scoped, position,
                         years, len(in_pool))
        ic = stats.factor_ic_table(signal, fwd, scoped, case, years)
        tail = ic[ic["fold"] == "mean_of_folds"].iloc[0]
        row.update({"ic": tail["ic"], "ic_ts": tail["ic_ts"], "ic_t": tail["t"],
                    "members": " ".join(in_pool)})
        rows.append(row)
        if len(years) > 1:
            for year in years:
                rows.append(period_row(
                    case, str(year),
                    gross_all[gross_all.index.year == int(year)],
                    net_all[net_all.index.year == int(year)],
                    scoped, position, [int(year)], len(scoped.get(int(year), []))))
    return pd.DataFrame(rows)


def passes(row: dict | pd.Series, cfg: BookConfig) -> bool:
    sharpe = row.get("net_sharpe", np.nan)
    ann = row.get("net_ann_return", np.nan)
    if not (np.isfinite(sharpe) and np.isfinite(ann)):
        return False
    return bool(sharpe >= cfg.min_net_sharpe and ann > cfg.min_net_ann_return)


def print_table(table: pd.DataFrame, columns: list[str]) -> None:
    cols = [c for c in columns if c in table.columns]
    with pd.option_context("display.width", 240, "display.max_rows", 200,
                           "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:+.4f}"):
        print(table[cols].to_string(index=False))
