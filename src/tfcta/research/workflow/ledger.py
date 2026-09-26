"""验证期与样本外的使用台账。一本书（按配置指纹）在 2022 上只测一次，在同一个样本外窗口上也只测一次。

旧版 step6 的冻结方案和 step10 的回溯诊断在改版前已经看过 2022，这里按同样的指纹
把它们登记为已消耗，不能用新脚本再测一遍。
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
    if plan_path.exists() and perf_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        perf = pd.read_csv(perf_path).set_index("strategy")
        for name, members in plan.get("strategies", {}).items():
            if name not in perf.index:
                continue
            cfg = strategy.BookConfig(
                factors=strategy.parse_factor_specs(members),
                fee_rate=float(plan["fee_rate"]),
                slippage_ticks=float(plan["slippage_ticks"]))
            row = perf.loc[name]
            net = {"net_ann_return": row.get("net_ann_return"),
                   "net_sharpe": row.get("net_ret_risk")}
            entries.append(_legacy(cfg, sectors.ALL_POOL, f"旧 step6 冻结方案 {name}", net))

    for params_path in sorted(C.RUNS_DIR.glob("*_retro_2022_sector_combo/params.json")):
        perf_path = params_path.with_name("performance.csv")
        if not perf_path.exists():
            continue
        params = json.loads(params_path.read_text(encoding="utf-8"))
        cfg = strategy.BookConfig(factors=strategy.parse_factor_specs(params["factors"]))
        perf = pd.read_csv(perf_path)
        full = perf[perf["universe"] == "冻结2022全品种池"]
        if full.empty:
            continue
        row = full.iloc[0]
        net = {"net_ann_return": row.get("net_ann_return"), "net_sharpe": row.get("net_sharpe")}
        entries.append(_legacy(cfg, sectors.ALL_POOL,
                               f"旧 step10 回溯诊断 {params_path.parent.name}", net))
    return entries


def _legacy(cfg: strategy.BookConfig, case: str, note: str, net: dict) -> dict:
    row = {k: float(v) if v is not None else np.nan for k, v in net.items()}
    return {
        "fingerprint": strategy.fingerprint(cfg, case),
        "case": case,
        "factors": strategy.factor_label(cfg.factors),
        "legacy": True,
        "note": note,
        "passed": strategy.passes(row, cfg),
        "criteria": {"min_net_sharpe": cfg.min_net_sharpe,
                     "min_net_ann_return": cfg.min_net_ann_return},
        **row,
    }


def validation_entries() -> list[dict]:
    return legacy_validation_entries() + read(validation_path())


def lookup(entries: list[dict], fp: str, **match) -> list[dict]:
    return [e for e in entries if e.get("fingerprint") == fp
            and all(e.get(k) == v for k, v in match.items())]


def stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
