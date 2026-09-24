"""第 5 步：参数扫描，扫完后用中心点选出唯一参数。滑点 0 只是对照，不得用于选参。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C                          # noqa: E402
from tfcta.factors import factor_cache as FC           # noqa: E402
from tfcta.research import backtest, center, folds, freeze, paths  # noqa: E402
from tfcta.research import runtime                     # noqa: E402


def threshold_combos(name: str,
                     lookbacks: list[int],
                     pcts: list[float]) -> list[tuple[int, float]]:
    """时间戳因子只读一组阈值文件。持续期因子扫完整的 (N, M)。"""
    if name in C.TIMESTAMP_FACTORS:
        n = lookbacks[len(lookbacks) // 2]
        m = pcts[len(pcts) // 2]
        return [(int(n), float(m))]
    return FC.combo_grid(lookbacks, pcts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--factors', nargs='*', default=None)
    ap.add_argument('--lookbacks', nargs='*', type=int, default=None)
    ap.add_argument('--pcts', nargs='*', type=float, default=None)
    ap.add_argument('--windows', nargs='*', type=int, default=None)
    ap.add_argument('--bands', nargs='*', default=None, help='形如 25:75')
    ap.add_argument('--fee', type=float, default=C.FEE_BASE)
    ap.add_argument('--slippage-ticks', type=float, default=C.SLIPPAGE_TICKS,
                    help='每次换手穿越几个最小变动价位；0 表示不计滑点')
    ap.add_argument('--rebuild-ticks', action='store_true',
                    help='重扫分钟数据重估 tick 表')
    ap.add_argument('--std-window', type=int, default=C.STD_WINDOW)
    ap.add_argument('--symbols', nargs='*', default=None)
    args = ap.parse_args()

    reason = runtime.not_ready_reason()
    if reason:
        print(reason)
        return 2

    try:
        bands = runtime.parse_bands(args.bands)
    except ValueError as e:
        print(e)
        return 2

    names = list(args.factors) if args.factors else list(C.DURATION_FACTORS) + list(C.TIMESTAMP_FACTORS)
    lookbacks = args.lookbacks or list(C.THRESHOLD_LOOKBACKS)
    pcts = args.pcts or list(C.THRESHOLD_PCTS)
    windows = args.windows or list(C.SIGNAL_WINDOWS)
    years = [f['test_year'] for f in folds.walk_forward_folds()]

    universe, symbols, day_ret = runtime.load_context(args.symbols)
    if not symbols or day_ret.empty:
        print("品种池与分片没有交集，或日收益为空。")
        return 2

    slip, slip_note = runtime.load_slippage(
        symbols, day_ret.index, args.slippage_ticks, rebuild=args.rebuild_ticks)

    print(f"参数扫描  {len(names)} 个因子  手续费 {args.fee:g}  "
          f"品种 {len(symbols)}  {slip_note}")
    print(f"W={windows}  分位轨={bands}  扫完后按中心点选参。")

    C.ensure_dirs()
    paths.scan_dir().mkdir(parents=True, exist_ok=True)
    run = paths.run_dir('step5')
    wrote = 0
    for name in names:
        combos = threshold_combos(name, lookbacks, pcts)
        parts = []
        err = None
        for n, m in combos:
            try:
                wide = runtime.load_factor(name, n, m, symbols)
            except (FileNotFoundError, KeyError) as e:
                err = str(e).splitlines()[0]
                break
            part = backtest.evaluate_signal_grid(
                wide, name, day_ret, universe, windows, bands,
                fee=args.fee, std_window=args.std_window, test_years=years,
                lookback=n, pct=m, slippage=slip)
            part.insert(0, 'factor', name)
            part.insert(1, 'depends_on_threshold', name in C.DURATION_FACTORS)
            parts.append(part)
        if err or not parts:
            print(f"  {name:<16} 跳过：{err or '没有组合'}")
            continue
        tab = pd.concat(parts, ignore_index=True)
        tab.to_csv(paths.scan_dir() / f'{name}.csv', index=False, encoding='utf-8-sig')
        tab.to_csv(run / f'{name}.csv', index=False, encoding='utf-8-sig')
        wrote += 1
        finite = int(tab['ret_risk'].notna().sum())
        print(f"  {name:<16} {len(tab):4d} 组参数，其中 {finite} 组有收益风险比")

    print(f"\n写出 {wrote}/{len(names)} 个因子  {paths.scan_dir()}")
    print(f"留痕 {run}")
    if wrote == 0:
        print("没有写出任何扫描。请先运行 step3_build_factors.py。")
        return 2
    code = select_center()
    if code != 0:
        return code
    print("可以进入第 6 步（分族回测、费率与持仓对照）。")
    return 0


def select_center() -> int:
    """扫描目录里每张表各选一个中心点。argmax 只对照，不采用。"""
    d = paths.scan_dir()
    files = sorted(d.glob('*.csv'))
    if not files:
        print(f"{d} 下没有扫描结果。")
        return 2

    selection = {}
    failed = []
    print("中心点选参  先 z-score 再算距离。argmax 只作对照。")
    for p in files:
        tab = pd.read_csv(p)
        name = p.stem
        try:
            chosen, argmax, scored = center.select_center(tab)
        except center.NoCandidate as e:
            failed.append(name)
            print(f"  {name:<16} {e}")
            continue
        family = 'duration' if name in C.DURATION_FACTORS else 'timestamp'
        selection[name] = {
            'family': family,
            'n_candidates': int(len(scored.dropna(subset=['distance']))),
            'center': freeze.clean_record(chosen, freeze.PARAM_KEYS),
            'argmax_ret_risk': freeze.clean_record(argmax, freeze.PARAM_KEYS),
        }
        c, a = selection[name]['center'], selection[name]['argmax_ret_risk']
        same = (c.get('window') == a.get('window') and c.get('q_low') == a.get('q_low')
                and c.get('lookback') == a.get('lookback') and c.get('pct') == a.get('pct'))
        print(f"  {name:<16} 中心点 N={c.get('lookback')} M={c.get('pct')} "
              f"W={c.get('window')} Q={c.get('q_low')}/{c.get('q_high')} "
              f"SR={c.get('ret_risk')}"
              f"{'  （与 argmax 相同）' if same else '  ≠ argmax'}")

    if not any(k for k in selection):
        print("没有任何因子选出参数。")
        return 1
    try:
        freeze.assert_research_selection(selection)
    except freeze.FreezeError as e:
        print(e)
        return 1

    C.ensure_dirs()
    paths.dump_json(paths.selection_path(), selection)
    run = paths.run_dir('step5')
    paths.dump_json(run / 'selection.json', selection)
    print(f"写出 {paths.selection_path()}")
    if failed:
        print(f"未能选参的因子: {failed}")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
