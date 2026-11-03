"""下载 50ETF 期权波动率指数 QVIX，并按研究期 / OOS 切开。

    python -m download.index.getQvix

来源是期权论坛的公开日线。研究期写到 data/index/external_rqdata/volatility/qvix_50etf.csv，
只保留 2015-02-09 至 2021-12-31。OOS 写到 data/index/external_rqdata/oos/volatility/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tfcta import config as C  # noqa: E402

URL = "http://1.optbbs.com/d/csv/d/k.csv"
RESEARCH_OUT = C.INDEX_QVIX_RESEARCH_PATH
OOS_OUT = C.INDEX_QVIX_OOS_DIR / "qvix_50etf.csv"
MANIFEST = C.INDEX_QVIX_OOS_DIR / "qvix_manifest.json"
COLUMNS = ("date", "open", "high", "low", "close")
SERIES_START = pd.Timestamp("2015-02-09")


def parse_qvix(raw: pd.DataFrame) -> pd.DataFrame:
    """只取 50ETF 的开高低收。其它品种的列不保留。"""
    if raw.shape[1] < 5:
        raise ValueError("QVIX 文件缺少 50ETF 的开高低收")
    frame = raw.iloc[:, :5].copy()
    frame.columns = list(COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for name in COLUMNS[1:]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    if frame.empty:
        raise ValueError("QVIX 文件没有有效行")
    if frame["date"].duplicated().any():
        raise ValueError("QVIX 日期重复")
    if frame["date"].min() > SERIES_START:
        raise ValueError("QVIX 研究期起点晚于 50ETF 期权上市日")
    return frame.reset_index(drop=True)


def split_qvix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2015-02-09 至 2021-12-31 留在研究期；OOS 仅保留 2022–2025。"""
    research = frame[(frame["date"] >= SERIES_START) & (frame["date"] <= pd.Timestamp(C.RESEARCH_END))].copy()
    oos = frame[frame["date"].between(
        pd.Timestamp(C.OOS_START), pd.Timestamp(C.DEFAULT_OOS_END)
    )].copy()
    if research.empty or oos.empty:
        raise ValueError("QVIX 切分后研究期或 OOS 为空")
    C.assert_no_holdout_dates(research["date"], "QVIX 研究期")
    return research.reset_index(drop=True), oos.reset_index(drop=True)


def load_research_qvix(path: Path | None = None) -> pd.DataFrame:
    """只读研究期文件。OOS 目录直接拒绝。"""
    path = Path(path or RESEARCH_OUT)
    C.assert_research_only(path)
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    C.assert_no_holdout_dates(frame["date"], "QVIX 研究期")
    if list(frame.columns) != list(COLUMNS):
        raise ValueError("QVIX 研究期文件列与下载口径不一致")
    return frame


def _write_oos(oos: pd.DataFrame) -> None:
    """保存 OOS 数据与不含行情值的来源清单。"""
    OOS_OUT.parent.mkdir(parents=True, exist_ok=True)
    oos.to_csv(OOS_OUT, index=False, encoding="utf-8-sig")
    MANIFEST.write_text(json.dumps({
        "source": URL,
        "partition": "oos",
        "start": C.OOS_START.isoformat(),
        "end": C.DEFAULT_OOS_END.isoformat(),
        "rule": "2022-01-01 <= date <= 2025-12-31",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download() -> pd.DataFrame:
    raw = pd.read_csv(URL, encoding="gbk")
    research, oos = split_qvix(parse_qvix(raw))
    del raw
    RESEARCH_OUT.parent.mkdir(parents=True, exist_ok=True)
    research.to_csv(RESEARCH_OUT, index=False, encoding="utf-8-sig")
    _write_oos(oos)
    del oos
    loaded = load_research_qvix()
    print(RESEARCH_OUT, loaded["date"].min().date(), loaded["date"].max().date(), len(loaded), flush=True)
    print(f"OOS 已写入 {OOS_OUT}", flush=True)
    return loaded


def main() -> int:
    download()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
