"""第 3 步：构造全部因子。

1. 持续期抽查，不过就停；
2. 时间戳、持续期因子按品种落盘（分钟级，最慢）；
3. 外部数据因子按时间分区落盘（研究期、2022 验证期；样本外须 --oos-end 且该日已过）；
4. 装配全部日频因子，写因子目录 factor_catalog.csv，第 5-7 步按目录选因子。

以后新因子都在这一步构造，不要再另开构造脚本：
    分钟级因子写进 factors/intraday.py（落盘在 factors/cache.py），
    外部数据因子写进 factors/external.py，
    日频量价/慢信号写进 factors/daily.py 并在 factors/library.py::assemble 登记。
    三处都会自动进入因子目录。
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
from tfcta.factors import cache as FC     # noqa: E402
from tfcta.factors import external as EXT  # noqa: E402
from tfcta.factors import intraday as D   # noqa: E402
from tfcta.factors import library         # noqa: E402
from tfcta.research.backtest import engine, strategy  # noqa: E402

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


def build_external(partitions: list[str], symbols: list[str] | None,
                   oos_end) -> int:
    print("\n" + "-" * 92)
    print(f"外部因子  分区 {partitions}"
          f"{f'  样本外截止 {oos_end}' if oos_end else ''}")
    print("-" * 92)
    failed = False
    for partition in partitions:
        try:
            results = EXT.build_partition(
                partition, symbols, end=oos_end if partition == 'holdout_locked' else None)
        except Exception as exc:
            print(f"{partition} 构建失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed = True
            continue
        written = sum(item.get('rows', 0) > 0 for item in results.values())
        print(f"{partition}: {written}/{len(results)} 个品种已写入")
    print(f"外部因子目录: {EXT.EXTERNAL_FACTOR_ROOT}")
    return 1 if failed else 0


def build_catalog(symbols: list[str], run: Path) -> pd.DataFrame:
    """研究期全部因子的来源、方向和池内覆盖率。只读研究期。"""
    sig = library.load(symbols)
    universe = U.load_universe()
    index = sig.bars['close'].index
    study = (index >= pd.Timestamp(C.STUDY_START)) & (index <= pd.Timestamp(C.RESEARCH_END))
    columns = sig.bars['close'].columns
    mask = engine.universe_mask(index, columns, universe).to_numpy() & study[:, None]
    covered = int(mask.sum())
    rows = []
    for name in [*sig.signed, *sig.unsigned]:
        wide = sig.signed.get(name, sig.unsigned.get(name))
        wide = wide.reindex(index=index, columns=columns)
        present = int((wide.notna().to_numpy() & mask).sum())
        valid = wide.dropna(how='all').index
        if name in C.FACTOR_SIGNS:
            source = '分钟缓存'
        elif name in library.EXTERNAL_FAMILIES:
            source = '外部数据'
        else:
            source = '日频装配'
        prior = library.SIGNED_PRIORS.get(name) if name in sig.signed else None
        rows.append({
            'factor': name,
            'family': sig.family[name],
            'source': source,
            'directed': name in sig.signed,
            'prior_sign': prior,
            'spec': name if prior is not None else f'{name}:+1 或 {name}:-1',
            'in_pool_coverage': present / covered if covered else np.nan,
            'n_symbols': int(wide.notna().any().sum()),
            'first': valid.min().date().isoformat() if len(valid) else None,
        })
    table = pd.DataFrame(rows)
    C.ensure_dirs()
    table.to_csv(strategy.factor_catalog_path(), index=False, encoding='utf-8-sig')
    table.to_csv(run / 'factor_catalog.csv', index=False, encoding='utf-8-sig')
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=None)
    ap.add_argument('--combos', nargs='*', default=None,
                    help='形如 N250_M55；省略则只算周频书用的那一组')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no-timestamp', action='store_true',
                    help='不重算时间戳族（只改了持续期族公式时配合 --overwrite 用）')
    ap.add_argument('--health-combo', default=None,
                    help='用哪个组合做因子健康度验收，默认取中间那组')
    ap.add_argument('--external-partitions', nargs='*', choices=EXT.PARTITIONS,
                    default=['research', 'validation_2022'],
                    help='外部因子分区；holdout_locked 需要同时给 --oos-end')
    ap.add_argument('--oos-end', default=None,
                    help='样本外外部因子的截止日期，必须已经过去')
    ap.add_argument('--no-external', action='store_true', help='跳过外部因子')
    ap.add_argument('--no-catalog', action='store_true', help='跳过因子目录')
    ap.add_argument('--skip-minute', action='store_true',
                    help='跳过持续期抽查和分钟级因子，只重建外部因子和因子目录')
    args = ap.parse_args()

    oos_end = None
    if 'holdout_locked' in args.external_partitions and not args.no_external:
        if not args.oos_end:
            print('构造 holdout_locked 外部因子必须给出 --oos-end。')
            return 2
        oos_end = C.to_date(args.oos_end)
        try:
            C.assert_test_window_closed(oos_end)
        except C.HoldoutViolation as exc:
            print(exc)
            return 1

    C.ensure_dirs()
    run = C.RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_step3"
    run.mkdir(parents=True, exist_ok=True)

    minute_code = 0
    if not args.skip_minute:
        minute_code, stop = build_minute_factors(args, run)
        if stop:
            return minute_code

    external_code = 0
    if not args.no_external:
        symbols = [s.upper() for s in args.symbols] if args.symbols else None
        external_code = build_external(args.external_partitions, symbols, oos_end)

    if not args.no_catalog:
        syms, _ = pick_symbols(args.symbols)
        print("\n" + "-" * 92)
        print(f"因子目录（研究期，{len(syms)} 个品种）")
        print("-" * 92)
        catalog = build_catalog(syms, run)
        with pd.option_context('display.width', 200, 'display.max_rows', 200,
                               'display.float_format', lambda v: f'{v:.3f}'):
            print(catalog[['factor', 'family', 'source', 'prior_sign',
                           'in_pool_coverage', 'n_symbols', 'first']].to_string(index=False))
        print(f"目录: {strategy.factor_catalog_path()}")

    print(f"\n留痕 {run}")
    if minute_code or external_code:
        return 1
    print("第 3 步完成，可以进入第 4 步。")
    return 0


def build_minute_factors(args, run: Path) -> tuple[int, bool]:
    """返回 (退出码, 是否停止后续)。抽查不过或前置缺失时停止；健康度不过则继续但最终返回 1。"""
    probe_code = run_duration_probe()
    if probe_code != 0:
        return probe_code, True

    all_combos = FC.combo_grid()
    if args.combos:
        want = set(args.combos)
        combos = [c for c in all_combos if FC.combo_name(*c) in want]
        unknown = want - {FC.combo_name(*c) for c in all_combos}
        if unknown:
            print(f"未知组合 {sorted(unknown)}；可用: "
                  f"{[FC.combo_name(*c) for c in all_combos]}")
            return 2, True
    else:
        combos = all_combos

    syms, how = pick_symbols(args.symbols)
    if not syms:
        print(f"没有可用分片，请先运行 step1_shard_minutes.py\n路径: {C.RESEARCH_DIR}")
        return 2, True

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
        info = FC.build_symbol(s, combos, overwrite=args.overwrite,
                               timestamp=not args.no_timestamp)
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
            return 2, True
        health_combo = match[0]

    print("\n" + "-" * 92)
    print(f"因子健康度与验收（组合 {FC.combo_name(*health_combo)}，"
          f"非空门槛 {MIN_NON_NULL:.0%}）")
    print("-" * 92)
    panel = FC.load_panel(*health_combo, symbols=syms)
    if panel.empty:
        print("面板为空，无法验收")
        return 1, True
    health = FC.panel_health(panel)
    table = FC.check_acceptance(health, MIN_NON_NULL)

    with pd.option_context('display.width', 220, 'display.max_columns', 30,
                           'display.max_rows', 80):
        print(table[['scope', 'n', 'non_null_ratio', 'nunique',
                     'min', 'p50', 'max', 'passed', 'reason']])

    # ---------------- 留痕 ----------------
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
    print(f"因子 {panel.shape[1]} 个，日频观测 {panel.shape[0]:,} 行")
    if failed:
        print(f"验收不通过的因子 {len(failed)} 个:")
        for f in failed:
            print(f"    {f:<18} {table.loc[f, 'reason']}")
        return 1, False
    print(f"分钟级因子 {len(table)} 项全部通过验收。")
    return 0, False


if __name__ == '__main__':
    raise SystemExit(main())
