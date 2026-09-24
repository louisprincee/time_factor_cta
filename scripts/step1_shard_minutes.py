"""第 1 步：分钟单体 pickle 按品种、按 2022-01-01 切成 parquet，落盘后验收。本脚本是唯一允许写 holdout 的地方。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C        # noqa: E402
from tfcta.data import sessions
from tfcta.data import shard_io       # noqa: E402


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


def shard_one(panel: pd.DataFrame, sym: str, write: bool = True,
              fmt: str = 'auto') -> dict:
    """切出单品种，按 trading_date 分成研究期与样本外两份。"""
    if sym not in panel.columns.get_level_values(0):
        return {'symbol': sym, 'status': 'absent'}

    sub = panel[sym]
    # 因子字段是计算所必需的；开盘价是收益口径所必需的。后者缺失仍允许分片
    # （因子可以先算），但会在 info 里标明，第 4 步读到时会直接报错而不是用收盘价顶替。
    wanted = list(dict.fromkeys([*C.FACTOR_FIELDS, *C.PRICE_FIELDS]))
    keep = [c for c in wanted if c in sub.columns]
    missing = [c for c in C.FACTOR_FIELDS if c not in sub.columns]
    missing_price = [c for c in C.PRICE_FIELDS if c not in sub.columns]
    if 'trading_date' not in keep:
        return {'symbol': sym, 'status': 'no_trading_date', 'missing': missing}

    rolls = extract_roll_dates(sub)
    df = sub[keep].copy()
    df['trading_date'] = pd.to_datetime(df['trading_date']).dt.normalize()
    df = df[df['trading_date'].notna()].sort_index(kind='mergesort')

    # 全字段皆空的行直接丢（未上市期间）
    val_cols = [c for c in keep if c != 'trading_date']
    df = df[df[val_cols].notna().any(axis=1)]

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
    args = ap.parse_args()
    fmt = shard_io.resolve_format(args.format)

    src = Path(args.monolith)
    if not src.exists():
        print(f"找不到单体文件: {src}")
        return 2

    syms = args.symbols or (C.ALL_SYMBOLS if args.all else C.COMMODITY_SYMBOLS)
    print(f"单体文件 {src.name}  {src.stat().st_size / 1e9:.1f} GB")
    print(f"目标品种 {len(syms)} 个；模式 {'dry-run' if args.dry_run else '落盘'}；"
          f"分片格式 {fmt}")
    if fmt == 'pickle':
        print("  注意：未检测到 pyarrow/fastparquet，退回 pickle 分片。"
              "装上 pyarrow 可获得更小体积与按列读取。")
    print("载入 pickle（峰值内存约 12-15 GB，请耐心）...", flush=True)

    t0 = time.time()
    with open(src, 'rb') as f:
        panel = pickle.load(f)
    print(f"载入完成 {time.time() - t0:.0f}s  shape={panel.shape}", flush=True)

    have = set(panel.columns.get_level_values(0))
    fields = sorted(set(panel.columns.get_level_values(1)))
    print(f"面板含 {len(have)} 个品种，{len(fields)} 个字段")
    print(f"字段: {fields}")
    absent = [s for s in syms if s not in have]
    if absent:
        print(f"警告：以下品种不在面板中，将跳过: {absent}")

    results = []
    for i, sym in enumerate(syms, 1):
        info = shard_one(panel, sym, write=not args.dry_run, fmt=fmt)
        results.append(info)
        print(f"[{i}/{len(syms)}] {sym:<4} {info.get('status'):<14} "
              f"research={info.get('research_rows', 0):>9,} "
              f"days={info.get('research_days', 0):>5} "
              f"{info.get('research_start')}..{info.get('research_end')}", flush=True)

    run = _run_dir()
    (run / 'manifest.json').write_text(json.dumps({
        'source': str(src), 'generated': datetime.now().isoformat(),
        'holdout_start': str(C.HOLDOUT_START), 'dry_run': args.dry_run,
        'panel_shape': list(panel.shape), 'fields': fields,
        'results': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    ok = [r for r in results if r['status'] == 'ok']
    no_price = [r['symbol'] for r in ok if r.get('missing_price_fields')]
    if no_price:
        print(f"警告：{len(no_price)} 个品种的分片缺少 open/openw，"
              f"第 8.3 节的 day_ret 将无法计算: {no_price}")
    print(f"\n完成 {len(ok)}/{len(syms)}；清单写入 {run / 'manifest.json'}")
    if args.dry_run:
        print("dry-run 不跑验收。")
        return 0
    print("开始分片验收。")
    return run_verify(syms if args.symbols else None, check_all=not args.symbols)


VERIFY_SYMBOLS = ['RB', 'CU', 'AU', 'JD', 'M', 'TA']
PASS, FAIL, WARN = '通过', '不通过', '注意'


def load_research(sym: str) -> pd.DataFrame:
    return shard_io.load_shard(sym, C.RESEARCH_DIR)


def check_1_time_bound(df: pd.DataFrame) -> tuple[str, str]:
    cut = pd.Timestamp(C.HOLDOUT_START)
    bad_td = df['trading_date'].max() >= cut
    bad_idx = df.index.max() >= cut + pd.Timedelta(days=1)
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


def run_verify(symbols=None, check_all: bool = False) -> int:
    available = shard_io.list_shards(C.RESEARCH_DIR)
    if not available:
        print(f"research/ 下没有分片\n路径: {C.RESEARCH_DIR}")
        return 2
    syms = available if check_all else (list(symbols) if symbols else VERIFY_SYMBOLS)
    failures, critical = [], []
    s, m = check_7_no_holdout_leak()
    print(f"[守卫] {s}  {m}")
    if s == FAIL:
        critical.append('holdout 守卫')
    s, m = check_6_inventory()
    print(f"[验收6] {s}  {m}")
    if s == FAIL:
        failures.append('验收6')
    for sym in syms:
        if sym not in available:
            print(f"{sym}: 无分片，跳过")
            continue
        df = load_research(sym)
        print(f"{sym}  {len(df):,} 行  {df['trading_date'].nunique():,} 交易日  "
              f"{df['trading_date'].min().date()} .. {df['trading_date'].max().date()}")
        for name, (s, m) in {
            '验收1 时间边界': check_1_time_bound(df),
            '验收2 夜盘归属': check_2_night_ownership(df),
            '验收3 bar 数  ': check_3_bar_counts(df),
            '验收4 NaN 逐年': check_4_nan_by_year(df),
            '验收5 换月日  ': check_5_roll_dates(sym),
        }.items():
            print(f"  [{name}] {s}  {m}")
            if s == FAIL:
                (critical if '验收1' in name or '验收2' in name else failures).append(
                    f"{sym}/{name.split()[0]}")
    if critical:
        print(f"存在致命问题，不得进入第 2 步: {critical}")
        return 1
    if failures:
        print(f"存在需处理的问题: {failures}")
        return 1
    print("分片验收通过，可以进入第 2 步。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
