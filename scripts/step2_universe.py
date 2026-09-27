"""第 2 步：逐年品种池。只用到 2021，并标出成交额塌缩的品种供人工核对。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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

    stats = U.collect_stats(args.symbols, verbose=False)
    if stats.empty:
        print(f"没有可用分片，请先运行 step1_shard_minutes.py\n路径: {C.RESEARCH_DIR}")
        return 2

    max_year = int(stats['year'].max())
    if max_year >= C.HOLDOUT_START.year:
        print(f"致命：统计量含 {max_year} 年，样本外被污染")
        return 1

    years = [y for y in C.UNIVERSE_YEARS if y <= max_year + 1]
    uni, detail = U.build_universe(stats, years=years,
                                  lookback_years=args.lookback,
                                  min_turnover=args.min_turnover_yi * 1e8)
    ee = U.entries_and_exits(uni)
    traj = U.universe_turnover_report(stats)

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

    C.report_step(2, passed=True, next_step=3, paths=[
        (C.UNIVERSE_DIR / 'universe_by_year.json', '逐年时点品种池（第 y 年只用 y-1 年统计）'),
        (C.UNIVERSE_DIR / 'entries_exits.csv', '各年进池、出池品种'),
        (C.UNIVERSE_DIR / 'turnover_trajectory.csv', '各品种成交额年度轨迹，表首是塌缩型'),
        (C.UNIVERSE_DIR / 'yearly_stats.csv', '各品种逐年成交额、有效日、夜盘类别'),
        (C.UNIVERSE_DIR / 'screen_detail.csv', '筛选门槛的逐品种逐年轻重'),
        (run, '以上文件的本次快照，另有 params.json'),
    ], note="请打开成交额轨迹和进出池记录，确认没有僵尸品种。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
