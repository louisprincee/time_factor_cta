"""下载 IF/IH/IC 主力合约历史 L1 tick；锁定样本只存原始数据。

    python -m download.index.getTickL1Close
    python -m download.index.getTickL1Close --symbols IF
    python -m download.index.getTickL1Close --partition oos_locked --start 2022-01-01 --end 2025-12-31

共同区间为 2015-04-16 至 2021-12-31，不含 IM。每个交易日取主力合约，
保存完整日内 L1 tick，并计算日盘与 14:30–15:00 的价差、买一卖一深度、
Lee–Ready 成交不平衡及收盘位置。按批次写 Parquet，可中断续跑。
锁定样本模式将 2022–2025 原始 tick 单独保存，不计算/保存日汇总或策略结果。
RiceQuant 历史期指 tick 是 L1，不包含可用的多档 L2 深度。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.rq_auth import rqdata_credentials  # noqa: E402

SYMBOLS = ("IF", "IH", "IC")
START = date(2015, 4, 16)
END = C.RESEARCH_END
OUT = C.INDEX_DATA_ROOT / "external_rqdata" / "l1_close"
RAW_OUT = C.INDEX_DATA_ROOT / "external_rqdata" / "l1_ticks"
LOCKED_RAW_OUT = C.INDEX_DATA_ROOT / "external_rqdata" / "holdout_locked" / "l1_ticks"
DAY_OPEN, LAST_WINDOW, DAY_CLOSE = "09:30:00", "14:30:00", "15:00:00.999"
RAW_BATCH_DAYS = 10
LOCKED_START, LOCKED_END = C.HOLDOUT_START, C.DEFAULT_OOS_END


def _flow_imbalance(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    direction = np.sign(frame["last"] - frame["prev_mid"])
    tick = np.sign(frame["last"].diff()).replace(0, np.nan).ffill()
    direction = direction.where(direction != 0, tick).fillna(0.0)
    volume = frame["dvol"]
    total = volume.sum()
    return float((direction * volume).sum() / total) if total > 0 else np.nan


def l1_day_stats(ticks: pd.DataFrame, trading_date) -> dict:
    """一个股指交易日的 tick → 日盘与尾盘流动性汇总。"""
    day = pd.Timestamp(trading_date).normalize()
    t = ticks.sort_values("datetime").copy()
    t["dvol"] = t["volume"].diff().fillna(t["volume"]).clip(lower=0)
    quote_ok = (t["a1"] > 0) & (t["b1"] > 0) & (t["a1"] >= t["b1"])
    t["mid"] = ((t["a1"] + t["b1"]) / 2).where(quote_ok)
    t["rel_spread"] = ((t["a1"] - t["b1"]) / t["mid"]).where(quote_ok)
    t["depth"] = (t["a1_v"] + t["b1_v"]).where(quote_ok)
    t["prev_mid"] = t["mid"].shift(1)
    stamp = pd.to_datetime(t["datetime"])
    day_session = (
        (stamp >= pd.Timestamp(f"{day.date()} {DAY_OPEN}"))
        & (stamp <= pd.Timestamp(f"{day.date()} {DAY_CLOSE}"))
    )
    tail_session = day_session & (stamp >= pd.Timestamp(f"{day.date()} {LAST_WINDOW}"))
    session, tail = t[day_session], t[tail_session]
    out = {"trading_date": day, "n_ticks": int(day_session.sum())}
    if session.empty:
        return out
    final = session.iloc[-1]
    final_quote = session[session["mid"].notna()]
    mid_final = float(final_quote["mid"].iloc[-1]) if len(final_quote) else np.nan
    high, low = float(final["high"]), float(final["low"])
    width = high - low
    out.update({
        "spread_day": float(session["rel_spread"].mean()),
        "spread_last30": float(tail["rel_spread"].mean()),
        "depth_day": float(session["depth"].mean()),
        "depth_last30": float(tail["depth"].mean()),
        "imb_day": _flow_imbalance(session),
        "imb_last30": _flow_imbalance(tail),
        "close_last": float(final["last"]),
        "mid_final": mid_final,
        "close_vs_mid": (
            (float(final["last"]) - mid_final) / mid_final if mid_final else np.nan
        ),
        "clv_last": (float(final["last"]) - low) / width if width > 0 else np.nan,
        "clv_mid": (mid_final - low) / width if width > 0 else np.nan,
    })
    return out


def prepare_raw_ticks(ticks: pd.DataFrame, symbol: str, contract: str) -> pd.DataFrame:
    """校验并标记一日完整 tick，保留接口返回的全部字段。"""
    frame = ticks.reset_index()
    required = {
        "datetime", "trading_date", "open", "last", "high", "low", "volume",
        "a1", "b1", "a1_v", "b1_v",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"历史 tick 缺少字段：{missing}")
    frame["symbol"] = symbol
    frame["contract"] = contract
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="下载 IF/IH/IC 主力合约完整历史 L1 tick")
    parser.add_argument(
        "--partition",
        choices=("research", "oos_locked"),
        default="research",
        help="oos_locked 只下载 2022–2025 原始 tick，不生成汇总/分析结果",
    )
    parser.add_argument("--symbols", nargs="*", default=None, help="只下载指定品种，可分批")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    args = parser.parse_args(argv)
    default_start, default_end = (
        (START, END) if args.partition == "research" else (LOCKED_START, LOCKED_END)
    )
    start = pd.Timestamp(args.start).date() if args.start else default_start
    end = pd.Timestamp(args.end).date() if args.end else default_end
    validate_window(start, end, args.partition)
    names = [name.upper() for name in args.symbols] if args.symbols else list(SYMBOLS)
    invalid = sorted(set(names) - set(SYMBOLS))
    if invalid or not names:
        raise ValueError(f"只支持 IF/IH/IC，收到无效品种：{invalid or names}")
    if args.partition == "oos_locked":
        return download_locked(names, start, end)
    return download(names, start, end)


def validate_window(start: date, end: date, partition: str) -> None:
    if start > end:
        raise ValueError(f"开始日期 {start} 晚于结束日期 {end}")
    if partition == "research":
        if start < START or end > END:
            raise C.HoldoutViolation(f"研究期股指 L1 只下 {START}..{END}，收到 {start}..{end}")
        return
    if start < LOCKED_START or end > LOCKED_END:
        raise C.HoldoutViolation(
            f"锁定股指 L1 只允许 {LOCKED_START}..{LOCKED_END}，收到 {start}..{end}"
        )


def _prepare_dominant(rq, symbol: str, start: date, end: date) -> pd.Series:
    dominant = rq.futures.get_dominant(symbol, start, end, rule=0)
    if dominant is None or dominant.empty:
        raise ValueError(f"{symbol} 在请求区间没有 RiceQuant 主力合约数据")
    dominant.index = pd.to_datetime(dominant.index).normalize()
    return dominant.sort_index()


def _write_tick_batch(raw_root: Path, records: list[pd.DataFrame]) -> tuple[Path, pd.Timestamp, pd.Timestamp]:
    batch = pd.concat(records, ignore_index=True)
    dates = pd.to_datetime(batch["trading_date"]).dt.normalize()
    first_day, last_day = dates.min(), dates.max()
    target = raw_root / f"{first_day.year}" / f"{first_day:%Y%m%d}_{last_day:%Y%m%d}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    batch.to_parquet(temporary, index=False)
    temporary.replace(target)
    return target, first_day, last_day


def _manifest_write(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_locked(symbols, start: date, end: date) -> int:
    """保存锁定期原始 tick；不计算每日统计、不读取收益或因子。"""
    validate_window(start, end, "oos_locked")
    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证；请检查 config/rqdata.env")
    import rqdatac as rq

    rq.init(*credentials)
    LOCKED_RAW_OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"锁定下载 {', '.join(symbols)}，{start}..{end}；仅写原始 L1 tick，不生成统计",
        flush=True,
    )
    for symbol in symbols:
        raw_root = LOCKED_RAW_OUT / symbol
        raw_root.mkdir(parents=True, exist_ok=True)
        manifest_path = raw_root / "download_manifest.json"
        manifest = {
            "source": "RiceQuant",
            "partition": "holdout_locked",
            "start": str(start),
            "end": str(end),
            "symbol": symbol,
            "content_not_loaded_for_analysis": True,
            "chunks": [],
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                previous.get("partition") != "holdout_locked"
                or previous.get("symbol") != symbol
                or previous.get("start") != str(start)
                or previous.get("end") != str(end)
            ):
                raise ValueError(f"{symbol} 已有不同日期范围的锁定下载清单：{manifest_path}")
            manifest = previous
        existing_parts = [
            path for path in raw_root.rglob("*.parquet")
            if "_" in path.stem and path.stem[:8].isdigit()
        ]
        completed_chunks = {path.name for path in existing_parts}
        manifest["chunks"] = sorted(
            path.relative_to(raw_root).as_posix() for path in existing_parts
        )
        completed_ranges = []
        for name in completed_chunks:
            first_text, last_text = name.removesuffix(".parquet").split("_", maxsplit=1)
            completed_ranges.append(
                (pd.Timestamp(first_text), pd.Timestamp(last_text))
            )
        dominant = _prepare_dominant(rq, symbol, start, end)
        if dominant.index.min().date() < start or dominant.index.max().date() > end:
            raise ValueError(f"{symbol} 主力日期超出锁定请求窗口")
        records: list[pd.DataFrame] = []
        batch_start = None

        def flush_locked_batch():
            nonlocal records, batch_start
            if not records:
                return
            target, first_day, last_day = _write_tick_batch(raw_root, records)
            manifest["chunks"].append(target.relative_to(raw_root).as_posix())
            _manifest_write(manifest_path, manifest)
            records = []
            batch_start = None
            print(f"{symbol}: 已保存锁定 tick 分块 {first_day.date()}..{last_day.date()}", flush=True)

        for day, contract in dominant.items():
            current_day = pd.Timestamp(day).normalize()
            if any(first <= current_day <= last for first, last in completed_ranges):
                continue
            if records and current_day.year != batch_start.year:
                flush_locked_batch()
            ticks = rq.get_price(
                contract,
                start_date=current_day.date(),
                end_date=current_day.date(),
                frequency="tick",
            )
            if ticks is None or ticks.empty:
                flush_locked_batch()
                raise ValueError(
                    f"{symbol} {current_day.date()} {contract} 没有 tick 行情；"
                    "锁定模式已保留此前分块，可修复后续跑"
                )
            frame = prepare_raw_ticks(ticks, symbol, contract)
            returned_dates = pd.to_datetime(frame["trading_date"]).dt.normalize()
            if returned_dates.min().date() < start or returned_dates.max().date() > end:
                flush_locked_batch()
                raise ValueError(f"{symbol} 返回 tick 日期超出锁定窗口")
            if not returned_dates.eq(current_day).all():
                flush_locked_batch()
                raise ValueError(f"{symbol} {current_day.date()} API 返回了其他交易日")
            records.append(frame)
            if batch_start is None:
                batch_start = current_day
            if len(records) >= RAW_BATCH_DAYS:
                flush_locked_batch()
        flush_locked_batch()
        requested_dates = pd.DatetimeIndex(dominant.index).normalize()
        written_ranges = [
            tuple(pd.Timestamp(text) for text in Path(chunk).stem.split("_"))
            for chunk in manifest["chunks"]
        ]
        coverage = sum(
            any(first <= day <= last for first, last in [*completed_ranges, *written_ranges])
            for day in requested_dates
        )
        if coverage != len(requested_dates):
            raise ValueError(f"{symbol} 锁定 L1 tick 分块覆盖不完整：{coverage}/{len(requested_dates)} 天")
        _manifest_write(manifest_path, manifest)
        print(f"{symbol}: 锁定窗口下载完成 {len(requested_dates)}/{len(requested_dates)} 天", flush=True)
    return 0


def download(symbols, start: date, end: date) -> int:
    C.assert_no_holdout_dates(pd.DatetimeIndex([pd.Timestamp(end)]), "股指 L1 下载截止日")
    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError("缺少 RiceQuant 凭证；请检查 config/rqdata.env")
    import rqdatac as rq

    rq.init(*credentials)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(symbols)} 个品种，{start}..{end}，下载每日主力合约的全部历史 L1 tick",
        flush=True,
    )
    for symbol in symbols:
        path = OUT / f"{symbol}.csv"
        done = pd.read_csv(path, parse_dates=["trading_date"]) if path.is_file() else pd.DataFrame()
        if len(done):
            C.assert_no_holdout_dates(done["trading_date"], f"{symbol} 已保存的 L1 汇总")
        summary_seen = set(done["trading_date"].dt.normalize()) if len(done) else set()
        raw_root = RAW_OUT / symbol
        raw_root.mkdir(parents=True, exist_ok=True)
        raw_seen = set()
        for part in raw_root.rglob("*.parquet"):
            raw_dates = pd.read_parquet(part, columns=["trading_date"])["trading_date"]
            C.assert_no_holdout_dates(raw_dates, f"{symbol} 已保存的历史 L1 tick")
            raw_seen.update(pd.to_datetime(raw_dates).dt.normalize())
        dominant = rq.futures.get_dominant(symbol, start, end, rule=0)
        if dominant is None or dominant.empty:
            raise ValueError(f"{symbol} 在请求区间没有 RiceQuant 主力合约数据")
        dominant.index = pd.to_datetime(dominant.index).normalize()
        C.assert_no_holdout_dates(dominant.index, f"{symbol} 股指主力表")
        requested = dominant.sort_index()
        records = []
        summaries = []
        batch_start = None

        def flush_batch():
            nonlocal records, summaries, done, batch_start
            if records:
                batch = pd.concat(records, ignore_index=True)
                dates = pd.to_datetime(batch["trading_date"]).dt.normalize()
                first_day, last_day = dates.min(), dates.max()
                target = (
                    raw_root
                    / f"{first_day.year}"
                    / f"{first_day:%Y%m%d}_{last_day:%Y%m%d}.parquet"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".tmp")
                batch.to_parquet(temporary, index=False)
                temporary.replace(target)
                raw_seen.update(dates)
                records = []
            if summaries:
                done = pd.concat([done, pd.DataFrame(summaries)], ignore_index=True)
                done = done.drop_duplicates(["trading_date"], keep="last").sort_values("trading_date")
                done.to_csv(path, index=False, encoding="utf-8-sig")
                summary_seen.update(pd.to_datetime(pd.DataFrame(summaries)["trading_date"]).dt.normalize())
                summaries = []
            if batch_start is not None:
                completed = len(raw_seen.intersection(requested.index.normalize()))
                print(
                    f"{symbol}: L1 tick {completed}/{len(requested)} 天已保存",
                    flush=True,
                )
                batch_start = None

        for day, contract in requested.items():
            if day.normalize() in raw_seen:
                continue
            if records and day.year != batch_start.year:
                flush_batch()
            ticks = rq.get_price(
                contract, start_date=day.date(), end_date=day.date(), frequency="tick"
            )
            if ticks is None or ticks.empty:
                flush_batch()
                raise ValueError(
                    f"{symbol} {day.date()} {contract} 没有 tick 行情，"
                    "已保存此前批次，可修复后续跑"
                )
            try:
                frame = prepare_raw_ticks(ticks, symbol, contract)
            except ValueError:
                flush_batch()
                raise
            records.append(frame)
            if batch_start is None:
                batch_start = day
            if day.normalize() not in summary_seen:
                stats = l1_day_stats(frame, day)
                if stats["n_ticks"] == 0:
                    flush_batch()
                    raise ValueError(f"{symbol} {day.date()} {contract} 没有日盘 tick")
                summaries.append({"symbol": symbol, "contract": contract, **stats})
            if len(records) >= RAW_BATCH_DAYS:
                flush_batch()
        flush_batch()
        requested_dates = set(requested.index.normalize())
        raw_coverage = len(raw_seen.intersection(requested_dates)) / len(requested_dates)
        summary_dates = set(pd.to_datetime(done["trading_date"]).dt.normalize()) if len(done) else set()
        summary_coverage = len(summary_dates.intersection(requested_dates)) / len(requested_dates)
        if raw_coverage < 1 or summary_coverage < 1:
            raise ValueError(
                f"{symbol} L1 覆盖不完整：tick {raw_coverage:.1%}，汇总 {summary_coverage:.1%}"
            )
        print(
            f"{symbol}: tick 与汇总各覆盖 {raw_coverage:.1%}；"
            f"原始 tick 分批保存在 {raw_root}，汇总在 {path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
