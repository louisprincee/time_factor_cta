"""第 4 步：单因子时序 IC、符号闸门、逻辑组合、稳定性与板块异质性。只读 2016–2021。

四部分：
1. 有先验的时间戳/持续期因子逐折 IC，做符号闸门（显著反向则返回 1）；
2. 全部因子（量价、慢信号、反转代理、无方向指标、外部数据）的事前 IC 汇总；
3. 按经济逻辑分组的等权组合，周频调仓、扣费回测；
4. 年度 IC 稳定性筛选，以及通过筛选的因子在五大板块、单品种上的异质性。

方向一律取事前先验，不按 IC 定；不在这里选参。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C                               # noqa: E402
from tfcta.data import bars as B                            # noqa: E402
from tfcta.factors import library                           # noqa: E402
from tfcta.research.analysis import screen, stats           # noqa: E402
from tfcta.research.backtest import costs, engine           # noqa: E402
from tfcta.research.workflow import context                 # noqa: E402

LOGIC = {
    '时间戳': ['ts_high', 'ts_low'],
    '持续期': ['dfp_max', 'dfp_top3'],
    '趋势形态': ['er_signed'],
    '均线偏离': ['po', 'bias'],
    '区间位置': ['rsv', 'rsi'],
    '资金流': ['obv', 'pvt'],
    '反转代理': ['neg_clv', 'neg_ret_day'],
    '慢信号': ['tsmom', 'carry'],
}
PV_LOGICS = ['趋势形态', '均线偏离', '区间位置', '资金流']


def summary_row(tab: pd.DataFrame) -> dict:
    row = tab[tab['fold'] == 'mean_of_folds'].iloc[0]
    by_year = tab[tab['fold'] != 'mean_of_folds'].set_index('fold')['ic_ts']
    out = {'ic': row['ic'], 'ic_ts': row['ic_ts'], 't': row['t'],
           'n_periods': row['n_periods']}
    out.update({f'ic_ts_{y}': v for y, v in by_year.items()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=None)
    ap.add_argument('--no-combos', action='store_true', help='跳过第 3 部分的组合回测')
    ap.add_argument('--no-heterogeneity', action='store_true',
                    help='跳过第 4 部分的稳定性与板块异质性')
    args = ap.parse_args()

    reason = context.not_ready_reason()
    if reason:
        print(reason)
        return 2
    universe, symbols, day_ret = context.load_context(args.symbols)
    if not symbols or day_ret.empty:
        print("品种池与分片没有交集，或日收益为空。")
        return 2
    years = [f['test_year'] for f in engine.walk_forward_folds()]
    fwd = B.forward_return(day_ret)
    try:
        sig = library.load(symbols)
    except (FileNotFoundError, KeyError) as e:
        print(f"读不到因子缓存：{str(e).splitlines()[0]}\n请先运行 step3_build_factors.py。")
        return 2

    # 1. 符号闸门
    gate_tabs = []
    for name in C.PRIOR_FACTORS:
        gate_tabs.append(stats.factor_ic_table(sig.raw(name), fwd, universe, name, years))
    gate_all = pd.concat(gate_tabs, ignore_index=True)

    # 2. 全部因子
    rows = []
    for name, wide in {**sig.signed, **sig.unsigned}.items():
        if sig.family[name].startswith('截面'):
            tab = stats.cross_sectional_ic_table(
                wide, fwd, universe, name, years)
        else:
            tab = stats.factor_ic_table(wide, fwd, universe, name, years)
        direction = ('先验已定向' if name in sig.signed else '无方向，仅报 IC')
        rows.append({'factor': name, 'family': sig.family[name], 'direction': direction,
                     **summary_row(tab)})
    all_tab = pd.DataFrame(rows)

    # 3. 逻辑组合
    combo_tab = pd.DataFrame()
    if not args.no_combos:
        slip, _ = costs.research_slippage(symbols, day_ret.index, C.SLIPPAGE_TICKS)
        signed_z = {n: library.trail_z(sig.signed[n]) for vs in LOGIC.values() for n in vs}

        def pack(label: str, names: list[str]) -> dict:
            wide = library.average_signals([signed_z[n] for n in names])
            tab = stats.factor_ic_table(wide, fwd, universe, label, years)
            row = tab[tab['fold'] == 'mean_of_folds'].iloc[0]
            pos = engine.execute_position(engine.weekly(wide.clip(-1, 1)))
            port = engine.run_book(pos, day_ret, universe, C.FEE_BASE, slippage=slip)
            net = engine.stitch_test_years(port, years)
            perf = stats.performance(net)
            return {'combo': label, 'members': ','.join(names), 'n': len(names),
                    'ic_ts': row['ic_ts'], 't': row['t'],
                    'ann_return': perf['ann_return'], 'ann_vol': perf['ann_vol'],
                    'sharpe': stats.sharpe_ratio(net), 'ret_risk': perf['ret_risk'],
                    'max_drawdown': perf['max_drawdown'],
                    'turnover': engine.annual_turnover(pos, universe, years)}

        ts, dur = LOGIC['时间戳'], LOGIC['持续期']
        pv_all = [n for g in PV_LOGICS for n in LOGIC[g]]
        plan = [(g, LOGIC[g]) for g in LOGIC]
        plan.append(('时间戳+持续期', ts + dur))
        for factor in LOGIC['反转代理']:
            plan.append((f'时间戳+持续期+{factor}', ts + dur + [factor]))
        plan.append(('时间戳+持续期+反转代理', ts + dur + LOGIC['反转代理']))
        for factor in LOGIC['慢信号']:
            plan.append((f'时间戳+持续期+{factor}', ts + dur + [factor]))
        plan.append(('时间戳+持续期+慢信号', ts + dur + LOGIC['慢信号']))
        for g in PV_LOGICS:
            plan += [(f'时间戳+{g}', ts + LOGIC[g]), (f'持续期+{g}', dur + LOGIC[g]),
                     (f'时间戳+持续期+{g}', ts + dur + LOGIC[g])]
        plan += [('全部量价', pv_all), ('三类全加', ts + dur + pv_all)]
        for g in PV_LOGICS:
            plan.append((f'三类去掉{g}', ts + dur + [n for o in PV_LOGICS if o != g
                                                   for n in LOGIC[o]]))
        singles = {n for vs in LOGIC.values() for n in vs if len(vs) > 1}
        plan += [(n, [n]) for n in sorted(singles)]
        combo_tab = pd.DataFrame([pack(lab, names) for lab, names in plan]).sort_values(
            'ann_return', ascending=False)

    C.ensure_dirs()
    run = context.run_dir('step4')
    outputs = [(gate_all, 'ic_by_fold.csv'), (all_tab, 'factor_ic_all.csv')]
    if not combo_tab.empty:
        outputs.append((combo_tab, 'logic_combo_ic.csv'))
    for tab, fname in outputs:
        tab.to_csv(run / fname, index=False, encoding='utf-8-sig')
        tab.to_csv(C.RESEARCH_OUT_DIR / fname, index=False, encoding='utf-8-sig')

    # 4. 稳定性与异质性
    if not args.no_heterogeneity:
        screen.write_heterogeneity(sig, fwd, universe, symbols, years, run)

    labels = {
        'ic_by_fold.csv': '有先验因子的逐折时序 IC 与符号闸门',
        'factor_ic_all.csv': '全部因子的事前 IC 汇总',
        'logic_combo_ic.csv': '逻辑组合周频扣费回测',
        'factor_stability.csv': '年度 IC 同号与时序 t 的稳定性筛选',
        'sector_factor_ic.csv': '五大板块上的因子 IC 与是否通过',
        'symbol_factor_ic.csv': '单品种因子 IC（多重比较，只作候选）',
    }
    paths = [(C.RESEARCH_OUT_DIR / name, labels[name]) for _, name in outputs]
    if not args.no_heterogeneity:
        paths += [(C.RESEARCH_OUT_DIR / n, labels[n]) for n in
                  ('factor_stability.csv', 'sector_factor_ic.csv', 'symbol_factor_ic.csv')]
    paths.append((run, '以上表格的本次快照'))

    gate = gate_all[gate_all['fold'] == 'mean_of_folds']
    flipped = gate.loc[gate['sign'] == 'flip', 'factor'].tolist()
    weak = gate.loc[gate['sign'] == 'flip_weak', 'factor'].tolist()
    pending = gate.loc[gate['sign'] == 'inconclusive', 'factor'].tolist()
    if flipped:
        print(f"符号与先验显著相反（|t| >= {C.SIGN_T_MIN:g}）: {flipped}。"
              "方向不允许按这个结果翻转。")
        C.report_step(4, passed=False, paths=paths)
        return 1
    if len(pending) == len(gate) and len(gate):
        print("有先验的因子都没有足够样本算出 IC，验收还做不了。")
        return 2
    notes = []
    if pending:
        notes.append(f"样本不足、暂不判定: {pending}")
    if weak:
        notes.append(f"方向相反但测不出显著性（|t| < {C.SIGN_T_MIN:g}）: {weak}")
    C.report_step(4, passed=True, next_step=5, paths=paths, note=" ".join(notes))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
