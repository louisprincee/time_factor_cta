"""第 1 步：分钟单体 pickle 按品种、按 2022-01-01 切成 parquet，落盘后验收。本脚本是唯一允许写 holdout 的地方。

写 ``validation_2022/`` 分片有两种模式，验证分片已存在时都不覆盖：

- ``--validation-from-monolith``：从切研究期用的同一个单体文件里按 ``trading_date`` 取 2022 年。
  和切 holdout 一样只在本脚本里过滤，2023 年及以后的行不落盘、不参与任何计算。
  同一份文件保证复权基准与研究期一致。
- ``--validation-source <文件>``：独立的 2022-only 源文件，出现 2022 年以外的日期就整份拒收，不做过滤。

    python scripts/step1_shard_minutes.py --validation-from-monolith
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C        # noqa: E402
from tfcta.data import sessions
from tfcta.data import shard_io       # noqa: E402
from tfcta.data import validation_import as VI   # noqa: E402


def _run_dir() -> Path:
    d = C.RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_step1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_roll_dates(sub: pd.DataFrame) -> pd.DataFrame | None:
    """从 dominant_id 的跳变提取换月日（验收项 5）。"""
    if C.ROLL_FIELD not in sub.columns:
        return None
    s = sub[C.ROLL_FIELD].dropna()
    if s.empty:
        return None
    td = pd.to_datetime(sub.loc[s.index, 'trading_date']).dt.normalize()
    daily = pd.Series(s.to_numpy(), index=td).groupby(level=0).last()
    changed = daily.ne(daily.shift())
    changed.iloc[0] = False          # 首日不算换月
    out = pd.DataFrame({'trading_date': daily.index[changed],
                        'new_contract': daily[changed].to_numpy(),
                        'prev_contract': daily.shift()[changed].to_numpy()})
    return out


def clean_symbol(sub: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """单品种只留分片字段，规整 trading_date，丢掉全空行。返回 (表, 缺的因子字段, 缺的价格字段)。"""
    # 因子字段是计算所必需的；开盘价是收益口径所必需的。后者缺失仍允许分片
    # （因子可以先算），但会在 info 里标明，第 4 步读到时会直接报错而不是用收盘价顶替。
    wanted = list(dict.fromkeys([*C.FACTOR_FIELDS, *C.PRICE_FIELDS]))
    keep = [c for c in wanted if c in sub.columns]
    missing = [c for c in C.FACTOR_FIELDS if c not in sub.columns]
    missing_price = [c for c in C.PRICE_FIELDS if c not in sub.columns]
    if 'trading_date' not in keep:
        return None, missing, missing_price

    df = sub[keep].copy()
    df['trading_date'] = pd.to_datetime(df['trading_date']).dt.normalize()
    df = df[df['trading_date'].notna()].sort_index(kind='mergesort')

    # 全字段皆空的行直接丢（未上市期间）
    val_cols = [c for c in keep if c != 'trading_date']
    df = df[df[val_cols].notna().any(axis=1)]
    return df, missing, missing_price


def shard_one(panel: pd.DataFrame, sym: str, write: bool = True,
              fmt: str = 'auto') -> dict:
    """切出单品种，按 trading_date 分成研究期与样本外两份。"""
    if sym not in panel.columns.get_level_values(0):
        return {'symbol': sym, 'status': 'absent'}

    sub = panel[sym]
    df, missing, missing_price = clean_symbol(sub)
    if df is None:
        return {'symbol': sym, 'status': 'no_trading_date', 'missing': missing}
    rolls = extract_roll_dates(sub)

    cut = pd.Timestamp(C.HOLDOUT_START)
    research = df[df['trading_date'] < cut]
    holdout = df[df['trading_date'] >= cut]

    info = {
        'symbol': sym, 'status': 'ok', 'missing_fields': missing,
        'missing_price_fields': missing_price,
        'research_rows': int(len(research)),
        'holdout_rows': int(len(holdout)),
        'research_days': int(research['trading_date'].nunique()),
        'research_start': str(research['trading_date'].min().date()) if len(research) else None,
        'research_end': str(research['trading_date'].max().date()) if len(research) else None,
        'roll_count': int(len(rolls)) if rolls is not None else 0,
    }

    if write:
        C.ensure_dirs()
        if len(research):
            p = shard_io.save_shard(research, C.RESEARCH_DIR, sym, fmt=fmt)
            info['research_mb'] = round(p.stat().st_size / 1e6, 1)
            info['format'] = p.suffix
        if len(holdout):
            # 唯一允许写 holdout 的地方；写完即锁，不读不看
            shard_io.save_shard(holdout, C.HOLDOUT_DIR, sym, fmt=fmt)
        if rolls is not None and len(rolls):
            rolls.to_csv(C.ROLL_DIR / f"{sym}.csv", index=False)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=None,
                    help='要分片的品种；默认商品期货全量')
    ap.add_argument('--all', action='store_true', help='含金融期货共 83 个')
    ap.add_argument('--dry-run', action='store_true', help='只报告不落盘')
    ap.add_argument('--monolith', default=str(C.MINUTE_MONOLITH))
    ap.add_argument('--format', default='auto', choices=['auto', 'parquet', 'pickle'],
                    help='分片格式；auto 表示有 pyarrow 就用 parquet，否则 pickle')
    ap.add_argument('--validation-source', default=None,
                    help='独立的 2022-only 分钟单体 pickle；给了就只写 validation_2022/ 分片')
    ap.add_argument('--validation-from-monolith', action='store_true',
                    help='从 --monolith 按 trading_date 取 2022 年，只写 validation_2022/ 分片')
    args = ap.parse_args()
    fmt = shard_io.resolve_format(args.format)
    if args.validation_source and args.validation_from_monolith:
        print("--validation-source 与 --validation-from-monolith 只能选一个。")
        return 2
    if args.validation_source:
        return shard_validation(Path(args.validation_source), args, fmt, slice_2022=False)
    if args.validation_from_monolith:
        return shard_validation(Path(args.monolith), args, fmt, slice_2022=True)

    src = Path(args.monolith)
    if not src.exists():
        print(f"找不到单体文件: {src}")
        return 2

    syms = args.symbols or (C.ALL_SYMBOLS if args.all else C.COMMODITY_SYMBOLS)
    if fmt == 'pickle':
        print("未检测到 pyarrow/fastparquet，退回 pickle 分片。"
              "装上 pyarrow 可获得更小体积与按列读取。")

    t0 = time.time()
    with open(src, 'rb') as f:
        panel = pickle.load(f)

    have = set(panel.columns.get_level_values(0))
    fields = sorted(set(panel.columns.get_level_values(1)))
    absent = [s for s in syms if s not in have]
    results = [shard_one(panel, sym, write=not args.dry_run, fmt=fmt) for sym in syms]

    run = _run_dir()
    (run / 'manifest.json').write_text(json.dumps({
        'source': str(src), 'generated': datetime.now().isoformat(),
        'holdout_start': str(C.HOLDOUT_START), 'dry_run': args.dry_run,
        'load_seconds': round(time.time() - t0, 1),
        'panel_shape': list(panel.shape), 'fields': fields,
        'absent': absent, 'results': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    ok = [r for r in results if r['status'] == 'ok']
    no_price = [r['symbol'] for r in ok if r.get('missing_price_fields')]
    notes = []
    if absent:
        notes.append(f"面板里没有这些品种，已跳过: {absent}")
    if no_price:
        notes.append(f"{len(no_price)} 个品种缺少 open/openw，day_ret 算不出来: {no_price}")
    if args.dry_run:
        C.report_step(1, passed=True, next_step=None, paths=[
            (run / 'manifest.json', '本次 dry-run 的面板结构和逐品种切分统计（未落盘）'),
        ], note="dry-run，没有落盘，也没有跑验收。"
        + ((" " + " ".join(notes)) if notes else ""))
        return 0
    return run_verify(syms if args.symbols else None, check_all=not args.symbols,
                      run=run, notes=notes)


VERIFY_SYMBOLS = ['RB', 'CU', 'AU', 'JD', 'M', 'TA']
PASS, FAIL, WARN = '通过', '不通过', '注意'


def load_research(sym: str) -> pd.DataFrame:
    return shard_io.load_shard(sym, C.RESEARCH_DIR)


def check_1_time_bound(df: pd.DataFrame) -> tuple[str, str]:
    cut = pd.Timestamp(C.HOLDOUT_START)
    # 用 date + timedelta，避免 Timestamp + Timedelta 触发 numpy generic unit 警告
    limit = pd.Timestamp(C.HOLDOUT_START + timedelta(days=1))
    bad_td = df['trading_date'].max() >= cut
    bad_idx = df.index.max() >= limit
    msg = f"trading_date max={df['trading_date'].max().date()}, index max={df.index.max()}"
    return (FAIL if (bad_td or bad_idx) else PASS), msg


def check_2_night_ownership(df: pd.DataFrame) -> tuple[str, str]:
    d = sessions.add_intraday_coords(df)
    night = d[d['session'] == C.SESSION_NIGHT]
    if night.empty:
        return WARN, "该品种无夜盘，此项不适用"
    late = pd.DatetimeIndex(night.index).hour >= C.NIGHT_START_HOUR
    if not late.any():
        return WARN, "夜盘 bar 全部在 00:00 之后，无法验证跨日归属"
    wall = pd.DatetimeIndex(night.index).normalize()
    ok_cross = bool((wall[late] < night['trading_date'][late]).all())
    mon = night[(night['trading_date'].dt.dayofweek == 0) & late]
    detail = ""
    if not mon.empty:
        ts = mon.index[0]
        td = mon['trading_date'].iloc[0]
        gap = (td.normalize() - ts.normalize()).days
        ok_mon = (ts.dayofweek <= 4) and gap >= 3
        detail = (f"；周一专项: bar {ts} -> trading_date {td.date()} "
                  f"(间隔 {gap} 天) {'正确' if ok_mon else '异常'}")
        ok_cross = ok_cross and ok_mon
    else:
        detail = "；无周一夜盘样本可查"
    frac = float((wall[late] < night['trading_date'][late]).mean())
    return (PASS if ok_cross else FAIL), f"跨日归属正确比例 {frac:.4f}{detail}"


def check_3_bar_counts(df: pd.DataFrame) -> tuple[str, str]:
    d = sessions.add_intraday_coords(df)
    counts = d.groupby('trading_date').size()
    cls = sessions.classify_night_length(sessions.day_bar_counts(d))
    expect = C.EXPECTED_BARS_PER_DAY[cls]
    med = float(counts.median())
    near = float(((counts - expect).abs() <= 15).mean())
    top = counts.value_counts().head(4).to_dict()
    status = PASS if (abs(med - expect) <= 20 and near >= 0.70) else WARN
    return status, (f"判定类别 {cls}（预期 {expect}）；中位数 {med:.0f}；"
                    f"±15 内占比 {near:.2f}；最常见 {top}")


def check_4_nan_by_year(df: pd.DataFrame) -> tuple[str, str]:
    yr = df['trading_date'].dt.year
    nan_ratio = df['closew'].isna().groupby(yr).mean().round(4)
    bad = nan_ratio[nan_ratio > 0.05]
    lines = ', '.join(f"{y}:{v:.3f}" for y, v in nan_ratio.items())
    status = PASS if bad.empty else WARN
    extra = f"；NaN>5% 的年份: {list(bad.index)}" if not bad.empty else ""
    return status, lines + extra


def check_5_roll_dates(sym: str) -> tuple[str, str]:
    p = Path(C.ROLL_DIR) / f"{sym}.csv"
    if not p.exists():
        return FAIL, f"缺少换月日文件 {p}"
    r = pd.read_csv(p)
    if r.empty:
        return WARN, "换月日文件为空"
    per_year = r.assign(y=pd.to_datetime(r['trading_date']).dt.year).groupby('y').size()
    return PASS, f"{len(r)} 次换月；逐年 {per_year.to_dict()}"


def check_6_inventory() -> tuple[str, str]:
    names = shard_io.list_shards(C.RESEARCH_DIR)
    total = sum(p.stat().st_size for p in Path(C.RESEARCH_DIR).iterdir()
                if p.is_file()) / 1e9 if names else 0.0
    status = PASS if names else FAIL
    return status, (f"research/ 共 {len(names)} 个分片，合计 {total:.2f} GB"
                    f"（预期 <= 83，商品 {len(C.COMMODITY_SYMBOLS)} 个）")


def check_7_no_holdout_leak() -> tuple[str, str]:
    try:
        C.assert_research_only(C.HOLDOUT_DIR / 'RB.parquet')
    except C.HoldoutViolation:
        return PASS, "assert_research_only 正确拦截 holdout_locked/ 访问"
    return FAIL, "守卫失效：holdout 路径未被拦截，这是严重问题"


def run_verify(symbols=None, check_all: bool = False,
               run: Path | None = None, notes: list[str] | None = None) -> int:
    available = shard_io.list_shards(C.RESEARCH_DIR)
    if not available:
        print(f"research/ 下没有分片\n路径: {C.RESEARCH_DIR}")
        return 2
    syms = available if check_all else (list(symbols) if symbols else VERIFY_SYMBOLS)
    rows, failures, critical = [], [], []

    def record(scope: str, name: str, status: str, detail: str) -> None:
        rows.append({'scope': scope, 'check': name, 'status': status, 'detail': detail})
        if status != FAIL:
            return
        key = f"{scope}/{name}"
        (critical if name in ('holdout 守卫', '验收1 时间边界', '验收2 夜盘归属')
         else failures).append(key)

    s, m = check_7_no_holdout_leak()
    record('全局', 'holdout 守卫', s, m)
    s, m = check_6_inventory()
    record('全局', '验收6 清单', s, m)
    for sym in syms:
        if sym not in available:
            record(sym, '分片', FAIL, '无分片')
            continue
        df = load_research(sym)
        for name, (s, m) in {
            '验收1 时间边界': check_1_time_bound(df),
            '验收2 夜盘归属': check_2_night_ownership(df),
            '验收3 bar 数': check_3_bar_counts(df),
            '验收4 NaN 逐年': check_4_nan_by_year(df),
            '验收5 换月日': check_5_roll_dates(sym),
        }.items():
            record(sym, name, s, m)

    verify_path = (run / 'verify.csv') if run is not None else C.RUNS_DIR / 'step1_verify.csv'
    pd.DataFrame(rows).to_csv(verify_path, index=False, encoding='utf-8-sig')
    paths = [
        (C.RESEARCH_DIR, '研究期分钟分片（按品种，到 2021-12-31）'),
        (C.ROLL_DIR, '各品种换月日'),
        (verify_path, '分片验收逐项结果（时间边界、夜盘归属、bar 数、NaN、换月日）'),
    ]
    if run is not None:
        paths.append((run / 'manifest.json', '本次运行参数与逐品种切分行数'))
    extra = ' '.join(notes or [])
    if critical or failures:
        print(f"未通过: {critical + failures}")
        C.report_step(1, passed=False, paths=paths, note=extra)
        return 1
    C.report_step(1, passed=True, next_step=2, paths=paths, note=extra)
    return 0


def _research_boundary(sym: str) -> pd.DataFrame | None:
    """研究期分片的末尾几天（只要 close/closew），用来核对复权偏移的衔接。"""
    if shard_io.find_shard(C.RESEARCH_DIR, sym) is None:
        return None
    df = shard_io.load_shard(sym, C.RESEARCH_DIR, columns=['close', 'closew', 'trading_date'])
    since = pd.Timestamp(df['trading_date'].max().date() - timedelta(days=10))
    return df[df['trading_date'] >= since]


def only_2022(df: pd.DataFrame) -> pd.DataFrame:
    """按 trading_date 只留 2022 年（2021-12-31 夜盘归属 2022-01-04，随之保留）。"""
    td = df['trading_date']
    out = df[(td >= pd.Timestamp(C.HOLDOUT_START)) & (td < pd.Timestamp(C.STRICT_OOS_START))]
    C.assert_validation_2022_dates(out['trading_date'], what="2022 切片")
    return out


def shard_validation(src: Path, args, fmt: str, slice_2022: bool) -> int:
    """写验证分片：检查全部通过才落盘，落盘后按验证 loader 读回验收。

    ``slice_2022`` 为真时源文件是含全样本的单体文件，逐品种按 trading_date 取 2022；
    否则源文件必须本身只含 2022，整份检查、不做过滤。
    """
    if not src.exists():
        print(f"找不到 2022 源文件: {src}")
        return 2
    existing = shard_io.list_shards(C.VALIDATION_DIR)
    if existing:
        print(f"{C.VALIDATION_DIR} 已有 {len(existing)} 个分片，不覆盖。"
              "验证数据只写一次；确需重建请先人工移走原目录并在研究笔记里记录原因。")
        return 2

    t0 = time.time()
    with open(src, 'rb') as f:
        panel = pickle.load(f)
    problems = [] if slice_2022 else VI.source_window_problems(panel)
    if problems:
        print("源文件不是 2022-only，整份拒收（不做过滤）：\n  " + "\n  ".join(problems))
        return 1

    syms = args.symbols or (C.ALL_SYMBOLS if args.all else C.COMMODITY_SYMBOLS)
    have = set(panel.columns.get_level_values(0))
    research_syms = set(shard_io.list_shards(C.RESEARCH_DIR))
    frames, results, boundary = {}, [], {}
    for sym in syms:
        if sym not in have:
            results.append({'symbol': sym, 'status': 'absent',
                            'has_research': sym in research_syms})
            continue
        df, missing, missing_price = clean_symbol(panel[sym])
        if df is not None and slice_2022:
            df = only_2022(df)
        if df is None or df.empty:
            results.append({'symbol': sym, 'status': 'empty', 'missing': missing,
                            'has_research': sym in research_syms})
            continue
        status, detail = VI.boundary_check(_research_boundary(sym), df)
        boundary[sym] = status
        frames[sym] = df
        results.append({
            'symbol': sym, 'status': 'ok', 'missing_fields': missing,
            'missing_price_fields': missing_price, 'rows': int(len(df)),
            'days': int(df['trading_date'].nunique()),
            'start': str(df['trading_date'].min().date()),
            'end': str(df['trading_date'].max().date()),
            'boundary': status, 'boundary_detail': detail,
        })
    ok_boundary, boundary_note = VI.boundary_verdict(boundary)

    run = _run_dir()
    (run / 'manifest.json').write_text(json.dumps({
        'mode': 'validation_2022', 'sliced_from_monolith': slice_2022, 'source': str(src), 'generated': datetime.now().isoformat(),
        'dry_run': args.dry_run, 'load_seconds': round(time.time() - t0, 1),
        'panel_shape': list(panel.shape), 'boundary': boundary_note, 'results': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    paths = [(run / 'manifest.json', '2022 源文件检查与逐品种统计（含复权衔接）')]

    absent = [r['symbol'] for r in results if r['status'] != 'ok' and r.get('has_research')]
    no_price = [r['symbol'] for r in results
                if r['status'] == 'ok' and r['missing_price_fields']]
    notes = [boundary_note]
    if absent:
        notes.append(f"研究期有、2022 源文件里没有的品种（2022 仍在池里的话第 6 步会报错）: {absent}")
    if no_price:
        notes.append(f"缺 open/openw，2022 收益算不出来: {no_price}")
    if not frames or not ok_boundary:
        C.report_step(1, passed=False, paths=paths, note=' '.join(notes) + " 没有落盘。")
        return 1
    if args.dry_run:
        C.report_step(1, passed=True, paths=paths,
                      note="dry-run，检查通过但没有落盘。 " + ' '.join(notes))
        return 0

    for sym, df in frames.items():
        shard_io.save_shard(df, C.VALIDATION_DIR, sym, fmt=fmt)
    return verify_validation(list(frames), run, paths, notes)


def verify_validation(syms: list[str], run: Path, paths: list, notes: list[str]) -> int:
    """按验证 loader 检查分片：日期限于 2022，夜盘归属与 bar 数同研究期口径。"""
    rows, failures = [], []
    try:
        C.assert_validation_only(C.RESEARCH_DIR / 'RB.parquet')
        rows.append({'scope': '全局', 'check': '验证目录守卫', 'status': FAIL,
                     'detail': '研究期路径没有被验证 loader 拦截'})
        failures.append('全局/验证目录守卫')
    except C.HoldoutViolation:
        rows.append({'scope': '全局', 'check': '验证目录守卫', 'status': PASS,
                     'detail': 'assert_validation_only 拦截了 validation_2022/ 以外的路径'})
    for sym in syms:
        try:
            df = shard_io.load_validation_shard(sym)
        except (C.HoldoutViolation, FileNotFoundError, KeyError) as exc:
            rows.append({'scope': sym, 'check': '读回', 'status': FAIL, 'detail': str(exc)})
            failures.append(f"{sym}/读回")
            continue
        for name, (s, m) in {
            '验收2 夜盘归属': check_2_night_ownership(df),
            '验收3 bar 数': check_3_bar_counts(df),
            '验收4 NaN 逐年': check_4_nan_by_year(df),
        }.items():
            rows.append({'scope': sym, 'check': name, 'status': s, 'detail': m})
            if s == FAIL:
                failures.append(f"{sym}/{name}")
    verify_path = run / 'verify.csv'
    pd.DataFrame(rows).to_csv(verify_path, index=False, encoding='utf-8-sig')
    paths = [(C.VALIDATION_DIR, '2022 验证分片（按品种，仅 2022）'),
             (verify_path, '读回验收逐项结果'), *paths]
    extra = ' '.join(notes)
    if failures:
        print(f"未通过: {failures}")
        C.report_step(1, passed=False, paths=paths, note=extra)
        return 1
    C.report_step(1, passed=True, next_step=6, paths=paths, note=extra)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
