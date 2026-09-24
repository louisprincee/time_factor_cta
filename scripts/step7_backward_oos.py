"""第 7 步：2010-2014 只看 IC 符号。不选参，不写 selection.json。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C                       # noqa: E402
from tfcta.research import folds, ic, paths         # noqa: E402
from tfcta.research import returns, runtime        # noqa: E402


def _threshold(name: str, selection: dict | None,
               fallback_n: int, fallback_m: float) -> tuple[int, float]:
    if selection and name in selection and 'center' in selection[name]:
        c = selection[name]['center']
        if c.get('lookback') is not None and c.get('pct') is not None:
            return int(c['lookback']), float(c['pct'])
    return int(fallback_n), float(fallback_m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--factors', nargs='*', default=None)
    ap.add_argument('--min-obs', type=int, default=C.IC_MIN_OBS)
    ap.add_argument('--lookback', type=int, default=C.IC_REFERENCE_LOOKBACK)
    ap.add_argument('--pct', type=float, default=C.IC_REFERENCE_PCT)
    args = ap.parse_args()

    reason = runtime.not_ready_reason()
    if reason:
        print(reason)
        return 2

    names = list(args.factors) if args.factors else list(C.PRIOR_FACTORS)
    universe, symbols, day_ret = runtime.load_context(None)
    years = folds.backward_years()
    window = day_ret.loc[(day_ret.index.year >= years[0]) & (day_ret.index.year <= years[-1])]
    print("=" * 88)
    print(f"向后时间外 {years[0]}-{years[-1]}  只看符号，不选参")
    print("夜盘类因子在这个窗口里大多没有定义，IC 缺失是预期结果。")
    print("=" * 88)
    if window.dropna(how='all').empty:
        print(f"研究期分片在 {years[0]}-{years[-1]} 没有日收益。"
              "早期年份尚未入库时这是预期状态，本步跳过，不改任何参数。")
        return 0

    selection = paths.load_json(paths.selection_path()) if paths.selection_path().exists() else None
    # 不用时点池。每个向后年份都用研究期池子的并集，并在表上标明。
    flat_universe = {y: list(symbols) for y in years}
    fwd = returns.forward_return(day_ret)

    tables = []
    for name in names:
        n, m = _threshold(name, selection, args.lookback, args.pct)
        try:
            wide = runtime.load_factor(name, n, m, symbols)
        except (FileNotFoundError, KeyError) as e:
            print(f"  {name:<16} 跳过：{str(e).splitlines()[0]}")
            continue
        tab = ic.factor_ic_table(wide, fwd, flat_universe, name, years, args.min_obs)
        tab.insert(1, 'lookback', n)
        tab.insert(2, 'pct', m)
        tables.append(tab)

    if not tables:
        print("没有读到任何因子。")
        return 2

    all_tab = pd.concat(tables, ignore_index=True)
    with pd.option_context('display.width', 220, 'display.max_rows', 400,
                           'display.float_format', lambda v: f'{v: .4f}'):
        print(all_tab.to_string(index=False))
    C.ensure_dirs()
    all_tab.to_csv(paths.backward_path(), index=False, encoding='utf-8-sig')
    run = paths.run_dir('step7')
    all_tab.to_csv(run / 'backward_ic.csv', index=False, encoding='utf-8-sig')
    print(f"\n留痕 {run}")
    print("本步不修改 selection.json，也不进入选参。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
