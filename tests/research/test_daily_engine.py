import numpy as np
import pandas as pd
import pytest

from tfcta.research.backtest import engine as E
from tfcta.factors import library as L


def frame(values, index=None):
    return pd.DataFrame(values, index=index if index is not None else pd.bdate_range('2021-01-04', periods=len(next(iter(values.values())))))


def test_roll_closes_old_and_opens_new_even_when_target_unchanged():
    pos = frame({'A':[1.,1.,1.]})
    rolls = frame({'A':[False,True,False]})
    opened, closed = E.trade_legs(pos, rolls)
    assert opened['A'].tolist() == [1.,1.,0.]
    assert closed['A'].tolist() == [0.,1.,0.]


def test_reversal_and_reduction_charge_correct_legs():
    opened, closed = E.trade_legs(frame({'A':[1.,.5,-1.,0.]}))
    assert opened['A'].tolist() == [1.,0.,1.,0.]
    assert closed['A'].tolist() == [0.,.5,.5,1.]


def test_pool_change_trades_survivors_and_closes_exiting_symbols():
    idx = pd.to_datetime(['2020-12-31','2021-01-04'])
    p = frame({'A':[1.,1.], 'B':[1.,1.]}, idx)
    exp = E.allocate(p, {2020:['A','B'],2021:['A']})
    np.testing.assert_allclose(exp.to_numpy(), [[.5,.5],[1.,0.]])
    opened, closed = E.trade_legs(exp)
    np.testing.assert_allclose(opened.to_numpy(), [[.5,.5],[.5,0.]])
    np.testing.assert_allclose(closed.to_numpy(), [[0.,0.],[0.,.5]])


def test_missing_signal_stays_cash_instead_of_upweighting_other_symbol():
    exp = E.allocate(frame({'A':[1.], 'B':[np.nan]}), {2021:['A','B']})
    assert exp.iloc[0].tolist() == [.5,0.]


def test_missing_market_return_for_a_live_position_is_an_error():
    p = frame({'A':[1.,1.]})
    with pytest.raises(ValueError, match='收益'):
        E.backtest(p, frame({'A':[.01,np.nan]}), {2021:['A']}, .001, .002, .0001)


def test_net_return_and_turnover_count_roll_and_asymmetric_fees():
    pos = frame({'A':[1.,1.,0.]})
    ret = frame({'A':[.01,.02,0.]})
    result = E.backtest(pos, ret, {2021:['A']}, .001, .002, .0001,
                        rolls=frame({'A':[False,True,False]}))
    assert result['net'].tolist() == pytest.approx([.0089,.0168,-.0021])
    assert result['turnover'].tolist() == [1.,2.,1.]


def test_single_rebalance_uses_only_scheduled_signal():
    sig = frame({'A':[1.,-1.,-.5,.5,1.]})
    assert E.rebalance(sig, days=3, mode='single', phase=0)['A'].tolist() == [1.,1.,1.,.5,.5]
    missing = frame({'A':[1.,0.,0.,np.nan,1.]})
    assert E.rebalance(missing, days=3)['A'].iloc[3] == 0.


def test_different_phases_and_staggering_are_actual_distinct_holdings():
    sig = frame({'A':[1.,-1.,-1.,1.]})
    assert E.rebalance(sig, 2, 'single', 0)['A'].tolist() == [1.,1.,-1.,-1.]
    assert E.rebalance(sig, 2, 'single', 1)['A'].tolist() == [0.,-1.,-1.,1.]
    assert E.rebalance(sig, 2, 'staggered')['A'].tolist() == [.5,0.,-1.,0.]


def test_economic_trend_and_positive_carry_never_turn_short_after_scaling():
    raw = frame({'A':np.r_[np.ones(130),np.full(30,.1)]})
    assert (L.to_signal('tsmom', raw).dropna() > 0).all().all()
    assert (L.to_signal('carry_ms', raw).dropna() > 0).all().all()


def test_combination_methods_express_different_hypotheses():
    a, b = frame({'A':[1.,1.]}), frame({'A':[-1.,.5]})
    assert L.combine([a,b], 'mean')['A'].tolist() == [0.,.75]
    assert L.combine([a,b], 'agree')['A'].tolist() == [0.,.75]
    assert L.combine([a,b], 'filter')['A'].tolist() == [0.,1.]
    assert L.combine([a, b*.0], 'filter')['A'].tolist() == [0.,0.]


def test_missing_combination_member_does_not_increase_other_weight():
    a, b = frame({'A':[1.]}), frame({'A':[np.nan]})
    assert L.combine([a,b], 'mean').iloc[0,0] == .5


def test_close_signal_is_executed_next_open_not_same_day():
    signal = frame({'A':[1.,0.,0.]})
    position = E.position(signal,vol_target=0.)
    result = E.backtest(position,frame({'A':[.10,-.02,.05]}),{2021:['A']})
    assert result.gross.tolist() == [0.,-.02,0.]


def test_timing_and_carry_signals_do_not_use_future_values():
    raw = frame({'A':np.linspace(-.3,.3,300)})
    modified = raw.copy()
    modified.iloc[250:,0] = 1000.
    for name in ('ts_low','carry_ms'):
        for timing in ('z','quantile'):
            pd.testing.assert_frame_equal(L.to_signal(name,raw,timing).iloc[:250],
                                           L.to_signal(name,modified,timing).iloc[:250])
