"""第 4 步：单因子时序 IC。方向看 ic，显著性看时序 t，不在这里选参。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C                 # noqa: E402
from tfcta.research import folds, ic, paths   # noqa: E402
from tfcta.research import returns            # noqa: E402
from tfcta.research import runtime            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--factors', nargs='*', default=None,
                    help='默认只检验有先验的因子')
    ap.add_argument('--lookback', type=int, default=C.IC_REFERENCE_LOOKBACK)
    ap.add_argument('--pct', type=float, default=C.IC_REFERENCE_PCT)
    ap.add_argument('--min-obs', type=int, default=C.IC_MIN_OBS)
    ap.add_argument('--symbols', nargs='*', default=None)
    args = ap.parse_args()

    reason = runtime.not_ready_reason()
    if reason:
        print(reason)
        return 2

    names = list(args.factors) if args.factors else list(C.PRIOR_FACTORS)
    universe, symbols, day_ret = runtime.load_context(args.symbols)
    if not symbols or day_ret.empty:
        print("品种池与分片没有交集，或日收益为空。")
        return 2
    fwd = returns.forward_return(day_ret)
    years = [f['test_year'] for f in folds.walk_forward_folds()]

    print(f"时序 IC  N={args.lookback} M={args.pct:g}  "
          f"最少观测 {args.min_obs}  品种 {len(symbols)}  测试年 {years}")
    print("方向看 ic，显著性看时序 t。t_cross 只供排序。")

    tables = []
    missing = []
    for name in names:
        try:
            wide = runtime.load_factor(name, args.lookback, args.pct, symbols)
        except (FileNotFoundError, KeyError) as e:
            missing.append((name, str(e).splitlines()[0]))
            print(f"  {name:<16} 跳过：{missing[-1][1]}")
            continue
        tab = ic.factor_ic_table(wide, fwd, universe, name, years, args.min_obs)
        tables.append(tab)

    if not tables:
        print("\n没有读到任何因子。请先运行 step3_build_factors.py。")
        for name, msg in missing:
            print(f"    {name}: {msg}")
        return 2

    all_tab = pd.concat(tables, ignore_index=True)
    show = all_tab.copy()
    with pd.option_context('display.width', 220, 'display.max_rows', 400,
                           'display.float_format', lambda v: f'{v: .4f}'):
        print(show.to_string(index=False))

    gate = all_tab[(all_tab['fold'] == 'mean_of_folds')
                   & (all_tab['factor'].isin(C.PRIOR_FACTORS))]
    flipped = gate.loc[gate['sign'] == 'flip', 'factor'].tolist()
    weak = gate.loc[gate['sign'] == 'flip_weak', 'factor'].tolist()
    pending = gate.loc[gate['sign'] == 'inconclusive', 'factor'].tolist()

    run = paths.run_dir('step4')
    all_tab.to_csv(run / 'ic_by_fold.csv', index=False, encoding='utf-8-sig')
    C.ensure_dirs()
    all_tab.to_csv(paths.ic_path(), index=False, encoding='utf-8-sig')
    print(f"\n留痕 {run}")

    if flipped:
        print(f"符号与先验**显著**相反的因子（|t| >= {C.SIGN_T_MIN:g}）: {flipped}")
        print("方向不允许按这个结果翻转。先核对实现，再决定是否进入第 5 步。")
        return 1
    if len(pending) == len(gate) and len(gate):
        print("有先验的因子都没有足够样本算出 IC，验收还做不了。")
        return 2
    if pending:
        print(f"样本不足、暂不判定的因子: {pending}")
    if weak:
        print(f"方向相反但测不出显著性的因子（|t| < {C.SIGN_T_MIN:g}）: {weak}")
        print("这不是方向错了，是在商品上没有可测的 IC——论文的因子没迁移过来。")
        print("不拦；但它们不该进主合成，报告里要单独写明。")
    print("有先验因子的折间平均 IC 没有出现显著反向，可以进入第 5 步。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
