"""第 3 步：先抽查持续期，通过后再计算并落盘日频因子。
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C            # noqa: E402
from tfcta.data import sessions           # noqa: E402
from tfcta.data import shard_io           # noqa: E402
from tfcta.data import universe as U      # noqa: E402
from tfcta.factors import duration as D   # noqa: E402
from tfcta.factors import factor_cache as FC  # noqa: E402

MIN_NON_NULL = 0.90
SKEW_MIN = 3.0
PROBE_COLS = ['closew']
PROBE_SYMBOLS = ['RB', 'CU', 'M']
PROBE_YEARS = [2016, 2018, 2021]


def probe(values, day_codes, keep, lookback: int, pct: float) -> dict:
    """持续期分布只在 keep 为真的行上汇总。预热行可以参与阈值，但不进入统计。"""
    values = np.asarray(values, dtype='float64')
    codes = np.asarray(day_codes)
    keep = np.asarray(keep, dtype=bool)
    diff = D.intraday_abs_diff(values, codes)
    thr = D.rolling_threshold(diff, codes, lookback, pct)
    dur = D.duration_series(values, codes, thr)
    d = dur[keep]
    finite = d[np.isfinite(d)]
    n_rows = int(keep.sum())
    n = int(finite.size)
    kept_diff = diff[keep]
    fin_diff = kept_diff[np.isfinite(kept_diff)]
    zero_ratio = float((fin_diff == 0).mean()) if fin_diff.size else np.nan
    if n == 0:
        p50 = p95 = mx = skew = np.nan
    else:
        p50 = float(np.percentile(finite, 50))
        p95 = float(np.percentile(finite, 95))
        mx = float(finite.max())
        skew = p95 / p50 if p50 else np.nan
    return {
        'n_rows': n_rows,
        'n': n,
        'nan_ratio': 1.0 if n_rows == 0 else 1.0 - n / n_rows,
        'all_nan': n == 0,
        'p50': p50,
        'p95': p95,
        'max': mx,
        'skew_ratio': skew,
        'zero_ratio': zero_ratio,
    }


def run_duration_probe() -> int:
    """抽查少数品种与年份。右偏不足或整段缺失时返回 1，拦住后面的因子计算。"""
    lookback = C.IC_REFERENCE_LOOKBACK
    pct = C.IC_REFERENCE_PCT
    print(f"持续期抽查  品种 {PROBE_SYMBOLS}  年份 {PROBE_YEARS}"
          f"  N={lookback}  M={pct}")
    failed = False
    for sym in PROBE_SYMBOLS:
        try:
            df = shard_io.load_shard(sym, columns=['trading_date', *PROBE_COLS])
        except FileNotFoundError:
            print(f"  {sym} 没有分片，请先运行 step1_shard_minutes.py")
            return 2
        df = sessions.add_intraday_coords(df)
        years = df['trading_date'].dt.year
        codes = df['trading_date'].factorize()[0]
        for year in PROBE_YEARS:
            keep = (years == year).to_numpy()
            if not keep.any():
                print(f"  {sym} {year} 无数据，跳过")
                continue
            for col in PROBE_COLS:
                r = probe(df[col].to_numpy(dtype='float64'), codes, keep,
                          lookback, pct)
                flag = ''
                if r['all_nan'] or (np.isfinite(r['skew_ratio']) and r['skew_ratio'] < SKEW_MIN):
                    flag = '  未通过'
                    failed = True
                skew = r['skew_ratio']
                skew_s = f'{skew:.2f}' if np.isfinite(skew) else 'nan'
                print(f"  {sym} {year} {col:<8} n={r['n']:<6} "
                      f"nan={r['nan_ratio']:.2%} zero={r['zero_ratio']:.2%} "
                      f"p95/p50={skew_s}{flag}")
    if failed:
        print("持续期抽查未通过，因子计算已停止。")
        return 1
    print("持续期抽查通过。")
    return 0


def pick_symbols(explicit: list[str] | None) -> tuple[list[str], str]:
    """默认用品种池的并集——只算池内品种，池外品种算了也不会进任何组合。"""
    have = shard_io.list_shards(C.RESEARCH_DIR)
    if explicit:
        return [s for s in explicit if s in have], '命令行指定'
    try:
        uni = U.load_universe()
        pooled = sorted({s for syms in uni.values() for s in syms})
        got = [s for s in pooled if s in have]
        if got:
            return got, f"品种池并集（{len(uni)} 个年份）"
    except FileNotFoundError:
        pass
    return have, '全部研究期分片（未找到品种池，退回全量）'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=None)
    ap.add_argument('--combos', nargs='*', default=None,
                    help='形如 N250_M55；省略则只算周频书用的那一组')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--health-combo', default=None,
                    help='用哪个组合做因子健康度验收，默认取中间那组')
    args = ap.parse_args()
    probe_code = run_duration_probe()
    if probe_code != 0:
        return probe_code

    all_combos = FC.combo_grid()
    if args.combos:
        want = set(args.combos)
        combos = [c for c in all_combos if FC.combo_name(*c) in want]
        unknown = want - {FC.combo_name(*c) for c in all_combos}
        if unknown:
            print(f"未知组合 {sorted(unknown)}；可用: "
                  f"{[FC.combo_name(*c) for c in all_combos]}")
            return 2
    else:
        combos = all_combos

    syms, how = pick_symbols(args.symbols)
    if not syms:
        print(f"没有可用分片，请先运行 step1_shard_minutes.py\n路径: {C.RESEARCH_DIR}")
        return 2

    print("=" * 92)
    print(f"日频因子缓存  {len(syms)} 个品种 × {len(combos)} 个参数组合"
          f"（{'覆盖重算' if args.overwrite else '断点续跑'}）")
    print(f"品种来源: {how}")
    print(f"落盘: {C.FACTOR_DAILY_DIR}  格式: {shard_io.resolve_format('auto')}")
    print("=" * 92)

    C.ensure_dirs()
    t0 = time.time()
    infos = []
    for i, s in enumerate(syms, 1):
        t1 = time.time()
        info = FC.build_symbol(s, combos, overwrite=args.overwrite)
        infos.append(info)
        tag = ('跳过（已存在）' if info['skipped']
               else f"{info['combos_written']} 组"
                    f"{' + 时间戳' if info.get('timestamp_written') else ''}")
        print(f"[{i}/{len(syms)}] {s:<5} {tag:<22} {time.time() - t1:6.1f}s", flush=True)
    print(f"\n计算完成，总耗时 {time.time() - t0:.0f}s")

    # ---------------- 验收 ----------------
    health_combo = combos[len(combos) // 2]
    if args.health_combo:
        match = [c for c in combos if FC.combo_name(*c) == args.health_combo]
        if not match:
            print(f"--health-combo {args.health_combo} 不在本次组合内")
            return 2
        health_combo = match[0]

    print("\n" + "-" * 92)
    print(f"因子健康度与验收（组合 {FC.combo_name(*health_combo)}，"
          f"非空门槛 {MIN_NON_NULL:.0%}）")
    print("-" * 92)
    panel = FC.load_panel(*health_combo, symbols=syms)
    if panel.empty:
        print("面板为空，无法验收")
        return 1
    health = FC.panel_health(panel)
    table = FC.check_acceptance(health, MIN_NON_NULL)

    with pd.option_context('display.width', 220, 'display.max_columns', 30,
                           'display.max_rows', 80):
        print(table[['scope', 'n', 'non_null_ratio', 'nunique',
                     'min', 'p50', 'max', 'passed', 'reason']])

    # ---------------- 留痕 ----------------
    run = C.RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_step3"
    run.mkdir(parents=True, exist_ok=True)
    table.to_csv(run / 'factor_health.csv', encoding='utf-8-sig')
    pd.DataFrame(infos).to_csv(run / 'build_log.csv', index=False, encoding='utf-8-sig')
    FC.write_manifest(run / 'manifest.json', {
        'generated': datetime.now().isoformat(),
        'symbols': syms, 'symbol_source': how,
        'combos': [FC.combo_name(*c) for c in combos],
        'health_combo': FC.combo_name(*health_combo),
        'min_non_null': MIN_NON_NULL,
        'factor_daily_dir': str(C.FACTOR_DAILY_DIR),
        'format': shard_io.resolve_format('auto'),
        'research_end': str(C.RESEARCH_END),
        'n_factors': int(panel.shape[1]),
        'n_rows': int(panel.shape[0]),
    })
    FC.write_manifest(C.FACTOR_DAILY_DIR / 'manifest.json', {
        'generated': datetime.now().isoformat(),
        'combos': [FC.combo_name(*c) for c in combos],
        'symbols': syms,
        'timestamp_dir': FC.TIMESTAMP_DIR_NAME,
        'note': '时间戳族不依赖 (N, M)，单独一份；持续期族每个组合一份。',
    })

    failed = table.index[~table['passed']].tolist()
    print(f"\n留痕 {run}")
    print(f"因子 {panel.shape[1]} 个，日频观测 {panel.shape[0]:,} 行")
    if failed:
        print(f"验收不通过的因子 {len(failed)} 个:")
        for f in failed:
            print(f"    {f:<18} {table.loc[f, 'reason']}")
        return 1
    print(f"全部 {len(table)} 个因子通过验收，可以进入第 4 步。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
