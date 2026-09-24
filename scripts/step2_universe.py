"""第 2 步：逐年品种池。只用到 2021，并标出成交额塌缩的品种供人工核对。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C        # noqa: E402
from tfcta.data import universe as U  # noqa: E402


def _run_dir() -> Path:
    d = C.RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_step2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=None)
    ap.add_argument('--lookback', type=int, default=C.UNIVERSE_LOOKBACK_YEARS)
    ap.add_argument('--min-turnover-yi', type=float, default=C.MIN_DAILY_TURNOVER / 1e8,
                    help='日均成交额门槛，单位亿元')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    print("=" * 78)
    print(f"品种池筛选  研究期终点 {C.RESEARCH_END}  回看 {args.lookback} 年  "
          f"成交额门槛 {args.min_turnover_yi:.0f} 亿元")
    print(f"完整度门槛 有效日占比 >= {C.MIN_VALID_DAY_RATIO:.0%} 且有效日 >= {C.MIN_VALID_DAYS}")
    print("=" * 78)

    stats = U.collect_stats(args.symbols, verbose=not args.quiet)
    if stats.empty:
        print(f"没有可用分片，请先运行 step1_shard_minutes.py\n路径: {C.RESEARCH_DIR}")
        return 2

    max_year = int(stats['year'].max())
    print(f"\n统计完成：{stats['symbol'].nunique()} 个品种，"
          f"年份范围 {int(stats['year'].min())}..{max_year}")
    if max_year >= C.HOLDOUT_START.year:
        print(f"致命：统计量含 {max_year} 年，样本外被污染")
        return 1

    years = [y for y in C.UNIVERSE_YEARS if y <= max_year + 1]
    uni, detail = U.build_universe(stats, years=years,
                                  lookback_years=args.lookback,
                                  min_turnover=args.min_turnover_yi * 1e8)

    print("\n" + "-" * 78)
    print("逐年品种池")
    print("-" * 78)
    ee = U.entries_and_exits(uni)
    for _, r in ee.iterrows():
        print(f"  {r['year']}  {r['n']:>3} 个")
        if r['entered']:
            print(f"        进: {r['entered']}")
        if r['exited']:
            print(f"        出: {r['exited']}")

    print("\n" + "-" * 78)
    print("成交额年度轨迹（亿元），按 末年/峰值 升序——表首即塌缩型品种")
    print("-" * 78)
    traj = U.universe_turnover_report(stats)
    with pd.option_context('display.width', 200, 'display.max_columns', 30,
                           'display.max_rows', 100):
        print(traj.head(15))

    # 塌缩品种是否被挡在后期池子外——这是本步最实质的验收
    print("\n" + "-" * 78)
    print("塌缩品种核对（末年/峰值 < 0.2 且峰值 >= 门槛的品种）")
    print("-" * 78)
    floor = args.min_turnover_yi
    suspects = traj[(traj['末年/峰值'] < 0.2) & (traj['峰值'] >= floor)].index.tolist()
    if not suspects:
        print("  研究期内没有出现先活跃后塌缩的品种")
    for sym in suspects:
        yrs = [y for y in sorted(uni) if sym in uni[y]]
        span = f"{min(yrs)}-{max(yrs)}" if yrs else "从未入池"
        print(f"  {sym:<4} 峰值 {traj.loc[sym, '峰值']:>7.1f} 亿 -> "
              f"末年 {traj.loc[sym, '末年']:>7.1f} 亿；入池年份 {span}")

    # 夜盘分组（逐年，不是逐品种）
    last = stats[stats['year'] == max_year]
    pool_last = set(uni.get(max(uni), []))
    no_night = sorted(last.loc[~last['has_night'].astype(bool), 'symbol'])
    print("\n" + "-" * 78)
    print(f"{max_year} 年夜盘分组（夜盘类因子在无夜盘品种上必须为 NaN 而非 0）")
    print("-" * 78)
    grp = last.groupby('night_class')['symbol'].apply(lambda s: ', '.join(sorted(s)))
    for cls, syms in grp.items():
        print(f"  {cls:<11} {syms}")
    in_pool = sorted(set(no_night) & pool_last)
    print(f"  其中在池内的无夜盘品种: {', '.join(in_pool) if in_pool else '（无）'}")

    # 落盘
    C.ensure_dirs()
    run = _run_dir()
    out_json = {str(y): uni[y] for y in sorted(uni)}
    for d in (C.UNIVERSE_DIR, run):
        (d / 'universe_by_year.json').write_text(
            json.dumps(out_json, ensure_ascii=False, indent=2), encoding='utf-8')
        stats.to_csv(d / 'yearly_stats.csv', index=False, encoding='utf-8-sig')
        detail.to_csv(d / 'screen_detail.csv', index=False, encoding='utf-8-sig')
        traj.to_csv(d / 'turnover_trajectory.csv', encoding='utf-8-sig')
        ee.to_csv(d / 'entries_exits.csv', index=False, encoding='utf-8-sig')

    fixed = uni.get(max(uni), [])
    (C.UNIVERSE_DIR / 'universe_fixed.txt').write_text('\n'.join(fixed) + '\n',
                                                       encoding='utf-8')
    (run / 'params.json').write_text(json.dumps({
        'generated': datetime.now().isoformat(),
        'lookback_years': args.lookback,
        'min_daily_turnover': args.min_turnover_yi * 1e8,
        'min_valid_day_ratio': C.MIN_VALID_DAY_RATIO,
        'min_valid_days': C.MIN_VALID_DAYS,
        'research_end': str(C.RESEARCH_END),
        'years': years,
        'symbols_scanned': int(stats['symbol'].nunique()),
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\n落盘: {C.UNIVERSE_DIR}  留痕: {run}")
    print(f"固定池（{max(uni)} 年，{len(fixed)} 个）: {' '.join(fixed)}")
    print("\n请人工核对上面的成交额轨迹与塌缩品种核对两节，确认无僵尸品种后进入第 3 步。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
