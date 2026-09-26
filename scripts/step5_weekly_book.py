"""第 5 步：周频等权，两个时序因子加两个持续期因子。

ts_high × -1，ts_low × +1，dfp_max × +1，dfp_top3 × +1。
符号来自研究期时序 IC。252 日 z 分数等权，每周最后一个交易日更新，次日开盘成交。
不读 2022 及以后。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C                              # noqa: E402
from tfcta.research import book, execution, jobs, panel, protocol, signals, stats  # noqa: E402

MEMBERS = {'ts_high': -1.0, 'ts_low': +1.0, 'dfp_max': +1.0, 'dfp_top3': +1.0}


def main() -> int:
    reason = jobs.not_ready_reason()
    if reason:
        print(reason)
        return 2
    universe, symbols, day_ret = jobs.load_context(None)
    C.assert_no_holdout_dates(day_ret.index, what='日收益')
    years = [f['test_year'] for f in protocol.walk_forward_folds()]
    frames = []
    for name, sign in MEMBERS.items():
        raw = jobs.load_factor(name, C.IC_REFERENCE_LOOKBACK, C.IC_REFERENCE_PCT, symbols)
        frames.append(signals.trail_z(raw) * sign)
    sig = execution.weekly(book.average_signals(frames).clip(-1, 1))
    slip, note = jobs.load_slippage(symbols, day_ret.index, C.SLIPPAGE_TICKS)
    pos = panel.execute_position(sig)
    port = book.run_book(pos, day_ret, universe, C.FEE_BASE, slippage=slip)
    stitched = protocol.stitch_test_years(port, years)
    rec = stats.performance(stitched)
    rec.update(name='ts_dfp', cost='net', fee=C.FEE_BASE,
               slippage_ticks=C.SLIPPAGE_TICKS,
               turnover=round(stats.annual_turnover(pos, universe, years), 1),
               members=','.join(f'{n}×{int(s):+d}' for n, s in MEMBERS.items()))
    for y in years:
        fm = stats.performance(port.loc[port.index.year == int(y)])
        rec[f'y{y}_ann_return'] = fm['ann_return']
        rec[f'y{y}_ret_risk'] = fm['ret_risk']
    tab = pd.DataFrame([rec])
    C.ensure_dirs()
    out = C.RESEARCH_OUT_DIR / 'weekly_book.csv'
    tab.to_csv(out, index=False, encoding='utf-8-sig')
    print(note)
    print(f"扣费后年化 {rec['ann_return']: .2%}  收益风险比 {rec['ret_risk']: .3f}  "
          f"换手 {rec['turnover']: .1f}")
    print(f"写出 {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
