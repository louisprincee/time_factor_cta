"""验证期与样本外的使用台账。一本书（按配置指纹）在 2022 上只测一次，在同一个样本外窗口上也只测一次。

除了精确指纹，每条记录还带 ``book_key``（等效因子权重 + 池 + 品种，不含成本与执行口径）；
step6 按 ``book_key`` 拦截，换调仓节奏、波动率目标或成本再测同一组因子也不放行。

作废：输入数据有缺陷（例如外部分区没覆盖检验窗口）时，追加一条 ``void`` 记录（``void_entry``），
写明原因。作废之前的记录不再拦截、不进可读表，只在作废清单里留痕；规格不能改，只能重跑同一本书。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ... import config as C
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


def validation_entries() -> list[dict]:
    return read(validation_path())


def validation_log_path() -> Path:
    return C.PROJECT_ROOT / "docs" / "ResearchNotes.md"


LOG_BEGIN = "<!-- 验证登记簿：由 ledger.write_validation_log 自动生成，勿手改 -->"
LOG_END = "<!-- 验证登记簿结束 -->"


def _num(entry: dict, key: str, percent: bool = False) -> str:
    item = entry.get(key)
    if item is None or pd.isna(item):
        return "-"
    return f"{float(item):.2%}" if percent else f"{float(item):.3f}"


def _result_link(entry: dict) -> str:
    run_dir = entry.get("run_dir")
    if not run_dir:
        return entry.get("note", "")
    if not (Path(run_dir) / "performance.csv").is_file():
        note = entry.get("note", "").strip()
        status = "运行明细已删除，仅保留台账摘要"
        return f"{note}（{status}）" if note else status
    try:
        relative = Path(run_dir).resolve().relative_to(C.PROJECT_ROOT.resolve())
    except ValueError:
        return run_dir
    return f"[运行结果](../{relative.as_posix()}/performance.csv)"


def _cell(text) -> str:
    """表格单元里的竖线要转义，否则 ``orb30|eod`` 会被拆成两列。"""
    return str(text).replace("|", "\\|")


def _live(entries: list[dict]) -> list[dict]:
    """去掉作废记录，以及同一指纹在最后一次作废之前的记录。"""
    last_void = {e["fingerprint"]: i for i, e in enumerate(entries) if e.get("void")}
    return [e for i, e in enumerate(entries)
            if not e.get("void") and i > last_void.get(e["fingerprint"], -1)]


def _unique(entries: list[dict]) -> list[dict]:
    return list({e["fingerprint"]: e for e in _live(entries)}.values())


def void_entry(fp: str, reason: str, run_dir=None) -> dict:
    return {"fingerprint": fp, "book_key": fp, "void": True, "reason": reason,
            "voided_run_dir": None if run_dir is None else str(run_dir), "run_at": stamp()}


def validation_log_lines() -> list[str]:
    """2022 验证台账与样本外台账的可读表，按指纹去重。"""
    lines = [
        "### 2022 验证",
        "",
        "防重复以 `data/validation_2022/ledger.jsonl` 为准，本表是可读索引。",
        "Fingerprint = 因子与方向、池、费率、滑点及执行配置；step6 另按 book_key（等效因子权重 + 池 + 品种）拦截，",
        "同一组因子在同一池子上看过 2022 后，换成本或执行口径也不能再验证。",
        "通过标准：净算术 Sharpe >= 0.5 且净年化 > 0。",
        "",
        "| 时间/来源 | 池 | 因子（乘在原始值上的方向） | 品种数 | 费率 | 滑点(tick) | 净年化 | 净 Sharpe | 净最大回撤 | 通过 | Fingerprint | 结果 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- |",
    ]
    for entry in _unique(validation_entries()):
        config = entry.get("config") or {}
        n_symbols = entry.get("n_symbols")
        fee, slippage = config.get("fee_rate"), config.get("slippage_ticks")
        lines.append(
            f"| {entry.get('run_at') or entry.get('note', '-')} | {entry.get('case', '-')} | "
            f"`{_cell(entry.get('factors', '-'))}` | "
            f"{'-' if n_symbols is None or pd.isna(n_symbols) else int(n_symbols)} | "
            f"{'-' if fee is None else f'{float(fee):.5f}'} | "
            f"{'-' if slippage is None else f'{float(slippage):.1f}'} | "
            f"{_num(entry, 'net_ann_return', True)} | {_num(entry, 'net_sharpe')} | "
            f"{_num(entry, 'net_max_drawdown', True)} | "
            f"{'通过' if entry.get('passed') else '未通过'} | `{entry['fingerprint']}` | "
            f"{_result_link(entry)} |")
    lines += [
        "",
        "### 样本外（2023 起）",
        "",
        "台账 `data/oos/ledger.jsonl`，只有 2022 通过的书才会出现在这里，每本书在同一窗口只测一次。",
        "",
        "| 时间 | 池 | 因子 | 窗口 | 净年化 | 净 Sharpe | 净最大回撤 | Fingerprint | 结果 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for entry in _unique(read(oos_path())):
        window = f"{entry.get('oos_start', '-')}..{entry.get('oos_end', '-')}"
        lines.append(
            f"| {entry.get('run_at', '-')} | {entry.get('case', '-')} | `{_cell(entry.get('factors', '-'))}` | "
            f"{window} | {_num(entry, 'net_ann_return', True)} | {_num(entry, 'net_sharpe')} | "
            f"{_num(entry, 'net_max_drawdown', True)} | "
            f"`{entry['fingerprint']}` | {_result_link(entry)} |")
    voided = [e for e in [*validation_entries(), *read(oos_path())] if e.get("void")]
    if voided:
        lines += ["", "### 作废记录", "",
                  "输入数据有缺陷的运行，作废后按同一规格重跑一次，上面两张表只列重跑结果。", "",
                  "| 时间 | Fingerprint | 原因 | 作废的运行 |", "| --- | --- | --- | --- |"]
        for e in voided:
            lines.append(f"| {e.get('run_at', '-')} | `{e['fingerprint']}` | {_cell(e.get('reason', '-'))} | "
                         f"{_cell(e.get('voided_run_dir') or '-')} |")
    return lines


def write_validation_log(path: Path | None = None) -> Path:
    """把登记簿写进研究笔记里 ``LOG_BEGIN``..``LOG_END`` 之间，笔记其余部分不动；没有标记就追加到末尾。"""
    path = Path(path) if path is not None else validation_log_path()
    block = "\n".join([LOG_BEGIN, *validation_log_lines(), LOG_END])
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if LOG_BEGIN in text and LOG_END in text:
        head, rest = text.split(LOG_BEGIN, 1)
        text = head + block + rest.split(LOG_END, 1)[1]
    else:
        text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def lookup(entries: list[dict], fp: str, **match) -> list[dict]:
    """指纹的有效记录（跳过作废的）。"""
    return [e for e in _live(entries) if e.get("fingerprint") == fp
            and all(e.get(k) == v for k, v in match.items())]


def stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
