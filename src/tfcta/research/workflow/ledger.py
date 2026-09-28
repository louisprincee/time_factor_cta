"""验证期与样本外的使用台账。一本书（按配置指纹）在 2022 上只测一次，在同一个样本外窗口上也只测一次。

旧版 step6 的冻结方案和 step10 的回溯诊断在改版前已经看过 2022，这里把它们登记为已消耗。
除了精确指纹，每条记录还带 ``book_key``（等效因子权重 + 池 + 品种，不含成本与执行口径）；
step6 按 ``book_key`` 拦截，换调仓节奏、波动率目标或成本再测同一组因子也不放行。
冻结方案只要 ``config/validation_2022_plan.json`` 在就登记，不依赖旧绩效表是否还在。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config as C
from ...data import sectors
from ..backtest import strategy
from .context import clean_json

VALIDATION_ROOT = C.DATA_ROOT / "validation_2022"
OOS_ROOT = C.DATA_ROOT / "oos"


def validation_path() -> Path:
    return VALIDATION_ROOT / "ledger.jsonl"


def oos_path() -> Path:
    return OOS_ROOT / "ledger.jsonl"


def read(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def append(path: Path, entries: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(clean_json(entry), ensure_ascii=False) + "\n")


def legacy_validation_entries() -> list[dict]:
    """改版前已经在 2022 上跑过、看过结果的书。"""
    entries = []
    plan_path = C.CONFIG_DIR / "validation_2022_plan.json"
    perf_path = VALIDATION_ROOT / "performance.csv"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        perf = (pd.read_csv(perf_path).set_index("strategy") if perf_path.exists()
                else pd.DataFrame())
        for name, members in plan.get("strategies", {}).items():
            # 旧版口径：周五单批、不做波动率缩放。
            cfg = strategy.BookConfig(
                factors=strategy.parse_factor_specs(members),
                fee_rate=float(plan["fee_rate"]),
                slippage_ticks=float(plan["slippage_ticks"]),
                tranches=0, vol_target=0.0)
            note = f"旧 step6 冻结方案 {name}"
            if name in perf.index:
                row = perf.loc[name]
                # 最早的表没有算术 Sharpe，只有几何年化 / 波动（列名 net_ret_risk）。
                # 2026-09-28 补回的表同时有 net_sharpe（日均 / 日波动 × sqrt(252)）。
                sharpe = row.get("net_sharpe")
                if sharpe is None or pd.isna(sharpe):
                    sharpe = row.get("net_ret_risk")
                n_symbols = row.get("n_symbols")
                if n_symbols is None or pd.isna(n_symbols):
                    n_symbols = row.get("symbols")
                net = {"net_ann_return": row.get("net_ann_return"),
                       "net_sharpe": sharpe,
                       "net_max_drawdown": row.get("net_max_drawdown"),
                       "n_symbols": n_symbols,
                       "turnover": row.get("turnover")}
            else:
                net = {}
                note += "（旧绩效表已不在，按已看过 2022、未通过登记）"
            entry = _legacy(cfg, sectors.ALL_POOL, note, net)
            if name in perf.index:
                logged_at = row.get("run_at") if "run_at" in row.index else None
                result_dir = row.get("run_dir") if "run_dir" in row.index else None
                if isinstance(logged_at, str) and logged_at.strip():
                    entry["run_at"] = logged_at.strip()
                if isinstance(result_dir, str) and result_dir.strip():
                    entry["run_dir"] = result_dir.strip()
            entries.append(entry)

    for params_path in sorted(C.RUNS_DIR.glob("*_retro_2022_sector_combo/params.json")):
        perf_path = params_path.with_name("performance.csv")
        if not perf_path.exists():
            continue
        params = json.loads(params_path.read_text(encoding="utf-8"))
        cfg = strategy.BookConfig(factors=strategy.parse_factor_specs(params["factors"]),
                                  tranches=0, vol_target=0.0)
        perf = pd.read_csv(perf_path)
        full = perf[perf["universe"] == "冻结2022全品种池"]
        if full.empty:
            continue
        row = full.iloc[0]
        net = {"net_ann_return": row.get("net_ann_return"),
               "net_sharpe": row.get("net_sharpe"),
               "net_max_drawdown": row.get("net_max_drawdown"),
               "n_symbols": row.get("n_symbols"),
               "turnover": row.get("turnover")}
        entries.append(_legacy(cfg, sectors.ALL_POOL,
                               f"旧 step10 回溯诊断 {params_path.parent.name}", net))
    return entries


def _legacy(cfg: strategy.BookConfig, case: str, note: str, net: dict) -> dict:
    row = {k: float(net[k]) if net.get(k) is not None and pd.notna(net[k]) else np.nan
           for k in ("net_ann_return", "net_sharpe")}
    entry = {
        "fingerprint": strategy.fingerprint(cfg, case),
        "book_key": strategy.book_key(cfg, case),
        "case": case,
        "factors": strategy.factor_label(cfg.factors),
        "config": cfg.to_dict(),
        "legacy": True,
        "note": note,
        "passed": bool(net) and strategy.passes(row, cfg),
        "criteria": {"min_net_sharpe": cfg.min_net_sharpe,
                     "min_net_ann_return": cfg.min_net_ann_return},
        **row,
    }
    for key in ("net_max_drawdown", "turnover"):
        if net.get(key) is not None and pd.notna(net[key]):
            entry[key] = float(net[key])
    if net.get("n_symbols") is not None and pd.notna(net["n_symbols"]):
        entry["n_symbols"] = int(net["n_symbols"])
    return entry


def validation_entries() -> list[dict]:
    return legacy_validation_entries() + read(validation_path())


def validation_log_path() -> Path:
    return C.PROJECT_ROOT / "docs" / "Validation2022Log.md"


def write_validation_log(path: Path | None = None) -> Path:
    path = Path(path) if path is not None else validation_log_path()
    entries = validation_entries()
    unique = {}
    for entry in entries:
        unique[entry["fingerprint"]] = entry

    def value(entry: dict, key: str, percent: bool = False) -> str:
        item = entry.get(key)
        if item is None or pd.isna(item):
            return "-"
        return f"{float(item):.2%}" if percent else f"{float(item):.3f}"

    lines = [
        "# 2022 验证结果登记簿",
        "",
        "本表由 `scripts/step6_validate_2022.py` 自动重建，包含旧冻结方案与当前验证台账。",
        "运行 step6 时会先同步已有记录；新结果写入 JSONL 台账后会再次同步本表。",
        "防重复以 `data/validation_2022/ledger.jsonl` 与旧冻结方案为准，本表是便于检索的可读索引。",
        "Fingerprint 表示相同因子与方向、池、费率、滑点及执行配置（调仓批数、波动率目标）。",
        "step6 另按 book_key（等效因子权重 + 池 + 品种）拦截：同一组因子在同一池子上看过 2022 后，",
        "换成本或执行口径也不能再验证。`time_combo:+1` 与四个时间因子等权是同一个 book_key。",
        "",
        "通过标准：净算术 Sharpe >= 0.5 且净年化收益 > 0%。Sharpe = 日均收益 / 日波动 × sqrt(252)。净绩效已扣手续费和滑点。",
        "旧冻结方案若只有几何年化 / 波动（`net_ret_risk`），Sharpe 列填的是那个数。",
        "2026-09-28 补回的四行是按冻结口径（周五单批、波动目标 0）用当前代码重算的算术 Sharpe。",
        "这些旧方案行保持 legacy，数字过门槛也不进入第 7 步。",
        "",
        "| 时间/来源 | 池 | 因子（乘在原始值上的方向） | 品种数 | 费率 | 滑点(tick) | 净年化 | 净 Sharpe | 净最大回撤 | 通过 | Fingerprint | 结果 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- |",
    ]
    for entry in unique.values():
        config = entry.get("config") or {}
        run_at = entry.get("run_at") or entry.get("note", "旧版记录")
        n_symbols = entry.get("n_symbols")
        n_symbols_text = "-" if n_symbols is None or pd.isna(n_symbols) else str(int(n_symbols))
        fee = config.get("fee_rate")
        fee_text = "-" if fee is None else f"{float(fee):.5f}"
        slippage = config.get("slippage_ticks")
        slippage_text = "-" if slippage is None else f"{float(slippage):.1f}"
        result = entry.get("note", "")
        run_dir = entry.get("run_dir")
        if run_dir:
            try:
                relative = Path(run_dir).resolve().relative_to(C.PROJECT_ROOT.resolve())
                result = f"[运行结果](../{relative.as_posix()}/performance.csv)"
            except ValueError:
                result = run_dir
        lines.append(
            f"| {run_at} | {entry.get('case', '-')} | `{entry.get('factors', '-')}` | "
            f"{n_symbols_text} | {fee_text} | {slippage_text} | "
            f"{value(entry, 'net_ann_return', percent=True)} | {value(entry, 'net_sharpe')} | "
            f"{value(entry, 'net_max_drawdown', percent=True)} | "
            f"{'通过' if entry.get('passed') else '未通过'} | `{entry['fingerprint']}` | {result} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def lookup(entries: list[dict], fp: str, **match) -> list[dict]:
    return [e for e in entries if e.get("fingerprint") == fp
            and all(e.get(k) == v for k, v in match.items())]


def stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
