"""第 6 步：分族等权回测，接着做费率网格和高 IC 因子持仓对照。都不改选参。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C                              # noqa: E402
from tfcta.research import backtest, combo, costs, folds, metrics, paths  # noqa: E402
from tfcta.research import runtime, signal                 # noqa: E402

DEFAULTS = ['COMBO_DUR', 'COMBO_TS', 'COMBO_ALL', 'COMBO_AGG']


def _one(name: str, port: pd.Series, years: list[int],
         fee: float, slip_ticks: float) -> list[dict]:
    rows = []
    stitched = folds.stitch_test_years(port, years)
    rec = metrics.performance(stitched)
    rec.update(combo=name, fold='stitched', fee=fee, slippage_ticks=slip_ticks)
    rows.append(rec)
    for y in years:
        fm = metrics.performance(port.loc[port.index.year == int(y)])
        fm.update(combo=name, fold=str(y), fee=fee, slippage_ticks=slip_ticks)
        rows.append(fm)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--combo', action='append', default=None,
                    help='NAME=f1,f2；可重复。省略则跑 DUR/TS/ALL/AGG')
    ap.add_argument('--fee', type=float, default=C.FEE_BASE)
    ap.add_argument('--slippage-ticks', type=float, default=C.SLIPPAGE_TICKS,
                    help='每次换手穿越几个最小变动价位；0 表示不计滑点')
    ap.add_argument('--std-window', type=int, default=C.STD_WINDOW)
    ap.add_argument('--symbols', nargs='*', default=None)
    args = ap.parse_args()

    reason = runtime.not_ready_reason()
    if reason:
        print(reason)
        return 2
    sel_path = paths.selection_path()
    if not sel_path.exists():
        print(f"找不到 {sel_path}\n请先运行 step5_param_scan.py。")
        return 2
    try:
        groups = runtime.parse_combos(args.combo, DEFAULTS)
    except ValueError as e:
        print(e)
        return 2

    selection = paths.load_json(sel_path)
    universe, symbols, day_ret = runtime.load_context(args.symbols)
    years = [f['test_year'] for f in folds.walk_forward_folds()]

    def loader(name, lookback, pct):
        return runtime.load_factor(name, lookback, pct, symbols)

    slip, slip_note = runtime.load_slippage(
        symbols, day_ret.index, args.slippage_ticks)

    print(f"分族等权  手续费 {args.fee:g}  {slip_note}")
    print("成员参数来自第 5 步。AGG 是对照组。")

    rows = []
    for name, members in groups.items():
        try:
            sig = combo.combo_signal_from_specs(loader, members, selection, args.std_window)
        except KeyError as e:
            print(f"  {name:<12} 跳过：{e}")
            continue
        port = backtest.run_book(signal.execute_position(sig), day_ret, universe,
                                 args.fee, slippage=slip)
        rows.extend(_one(name, port, years, args.fee, args.slippage_ticks))
        st = rows[-1 - len(years)]
        print(f"  {name:<12} 成员 {len(members):2d}  "
              f"年化 {st['ann_return'] if st['ann_return'] == st['ann_return'] else float('nan'): .2%}  "
              f"收益风险比 {st['ret_risk'] if st['ret_risk'] == st['ret_risk'] else float('nan'): .3f}  "
              f"最大回撤 {st['max_drawdown'] if st['max_drawdown'] == st['max_drawdown'] else float('nan'): .2%}")

    if not rows:
        print("没有跑成任何组合。")
        return 2

    tab = pd.DataFrame(rows)
    C.ensure_dirs()
    tab.to_csv(paths.combo_path(), index=False, encoding='utf-8-sig')
    run = paths.run_dir('step6')
    tab.to_csv(run / 'combo_metrics.csv', index=False, encoding='utf-8-sig')
    stitched = tab[tab['fold'] == 'stitched']
    cols = ['combo', 'ann_return', 'ann_vol', 'ret_risk', 'calmar', 'win_rate', 'max_drawdown', 'n_days']
    with pd.option_context('display.width', 160, 'display.float_format', lambda v: f'{v: .4f}'):
        print('\n拼接测试年:')
        print(stitched[cols].to_string(index=False))
    print(f"\n留痕 {run}")
    fee_code = fee_grid(groups, selection, universe, symbols, day_ret, years, args.std_window)
    if fee_code != 0:
        return fee_code
    return hold_diag(selection, args.std_window)


FOCUS = ['ts_high', 'ts_low', 'dfp_max', 'dfp_top3']
BP_MAX = 5.0


def fee_grid(groups, selection, universe, symbols, day_ret, years, std_window) -> int:
    fees = list(C.FEE_GRID)
    ticks = list(C.SLIPPAGE_TICK_GRID)

    def loader(name, lookback, pct):
        return runtime.load_factor(name, lookback, pct, symbols)

    slips = {float(k): runtime.load_slippage(symbols, day_ret.index, k) for k in ticks}
    print(f"成本敏感性  手续费 {fees}  滑点 {ticks} 个 tick")
    for k in ticks:
        print(f"  {slips[float(k)][1]}")
    print("滑点 0 是对照。写结论看 1 个 tick。")

    books = {}
    for name, members in groups.items():
        try:
            sig = combo.combo_signal_from_specs(loader, members, selection, std_window)
        except KeyError as e:
            print(f"  {name:<12} 跳过：{e}")
            continue
        books[name] = signal.execute_position(sig)
    if not books:
        print("没有跑成任何组合。")
        return 2

    rows = []
    for name, pos in books.items():
        for k in ticks:
            slip = slips[float(k)][0]
            srs = {}
            for fee in fees:
                port = backtest.run_book(pos, day_ret, universe, fee, slippage=slip)
                stitched = folds.stitch_test_years(port, years)
                rec = metrics.performance(stitched)
                rec.update(combo=name, fee=float(fee), slippage_ticks=float(k))
                rows.append(rec)
                srs[float(fee)] = rec['ret_risk']
            print(f"  {name:<12} 滑点 {float(k):g} tick: {metrics.fee_flip_text(pd.Series(srs))}")

    tab = pd.DataFrame(rows)
    tab.to_csv(paths.fee_path(), index=False, encoding='utf-8-sig')
    run = paths.run_dir('step6')
    tab.to_csv(run / 'fee_sensitivity.csv', index=False, encoding='utf-8-sig')
    cols = ['combo', 'fee', 'slippage_ticks', 'ann_return', 'ret_risk',
            'calmar', 'max_drawdown', 'win_rate']
    with pd.option_context('display.width', 200, 'display.max_rows', 400,
                           'display.float_format', lambda v: f'{v: .4f}'):
        print(tab[cols].to_string(index=False))
        print('收益风险比网格（行=手续费，列=滑点 tick 数）:')
        for name in books:
            sub = tab[tab['combo'] == name]
            print(f"\n  {name}")
            print(sub.pivot(index='fee', columns='slippage_ticks', values='ret_risk').to_string())
    print(f"写出 {paths.fee_path()}")
    return 0


def _turnover(pos, universe, years) -> float:
    cur = pos.fillna(0.0)
    to = (cur - cur.shift(1).fillna(0.0)).abs()
    per_year = []
    for y in years:
        cols = [s for s in universe.get(int(y), []) if s in to.columns]
        if not cols:
            continue
        block = to.loc[to.index.year == int(y), cols]
        if not block.empty:
            per_year.append(float(block.sum().mean()))
    return float(np.mean(per_year)) if per_year else float('nan')


def hold_diag(selection, std_window) -> int:
    missing = [m for m in FOCUS if m not in selection or 'center' not in selection[m]]
    if missing:
        print(f"缺少中心点参数: {missing}")
        return 2
    universe, symbols, day_ret = runtime.load_context(None)
    years = [f['test_year'] for f in folds.walk_forward_folds()]
    slip, slip_note = runtime.load_slippage(symbols, day_ret.index, C.SLIPPAGE_TICKS)
    frames = []
    for name in FOCUS:
        spec = selection[name]['center']
        raw = runtime.load_factor(name, spec.get('lookback'), spec.get('pct'), symbols)
        frames.append(backtest.member_signal(
            raw, name, spec['window'], spec['q_low'], spec['q_high'], std_window))
    held = [signal.hold_signal(f) for f in frames]
    confirmed = {k: [signal.confirm_signal(f, k) for f in frames] for k in (3, 5)}
    books = [(n, 'flat', f) for n, f in zip(FOCUS, frames)]
    books += [(n, 'hold', f) for n, f in zip(FOCUS, held)]
    for k, frames_k in confirmed.items():
        books += [(n, f'confirm{k}', f) for n, f in zip(FOCUS, frames_k)]
    books.append(('IC4', 'flat', backtest.average_signals(frames)))
    books.append(('IC4', 'hold', backtest.average_signals(held)))
    for k, frames_k in confirmed.items():
        books.append(('IC4', f'confirm{k}', backtest.average_signals(frames_k)))
    summary = costs.cost_summary(costs.load_tick_table(symbols), 1.0)
    cheap = set(summary.index[summary['bp_per_turnover'] <= BP_MAX])
    dropped = sorted(set(symbols) - cheap)
    uni_cheap = {y: [s for s in syms if s in cheap] for y, syms in universe.items()}
    books.append(('IC4_cheap', 'hold', backtest.average_signals(held)))

    rows = []
    for name, rule, sig in books:
        uni = uni_cheap if name == 'IC4_cheap' else universe
        n_sym = len({s for vs in uni.values() for s in vs})
        pos = signal.execute_position(sig)
        for fee, ticks, slip_i in ((0.0, 0.0, None), (C.FEE_BASE, C.SLIPPAGE_TICKS, slip)):
            port = backtest.run_book(pos, day_ret, uni, fee, slippage=slip_i)
            stitched = folds.stitch_test_years(port, years)
            rec = metrics.performance(stitched)
            rec.update(name=name, rule=rule, fee=fee, slippage_ticks=ticks,
                       turnover=round(_turnover(pos, uni, years), 1), n_symbols=n_sym)
            rows.append(rec)
    tab = pd.DataFrame(rows)
    out = C.RESEARCH_OUT_DIR / 'hold_diag.csv'
    tab.to_csv(out, index=False, encoding='utf-8-sig')
    run = paths.run_dir('step6')
    tab.to_csv(run / 'hold_diag.csv', index=False, encoding='utf-8-sig')
    show = ['name', 'rule', 'fee', 'slippage_ticks', 'ann_return', 'ann_vol',
            'ret_risk', 'turnover', 'n_symbols']
    print(f"持仓对照  {slip_note}")
    print(f"IC4_cheap 去掉单 tick > {BP_MAX:g}bp 的品种"
          f"（{len(dropped)} 个）: {' '.join(dropped) if dropped else '无'}")
    with pd.option_context('display.width', 160, 'display.float_format', lambda v: f'{v: .4f}'):
        print(tab[show].to_string(index=False))
    print(f"写出 {out}")
    print("未改 selection.json。可以进入第 7 步，或直接第 8 步冻结。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
