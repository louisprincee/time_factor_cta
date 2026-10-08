"""Regression cases from the audit; all market inputs are synthetic."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
morning = importlib.import_module('plot_morning_rule')
book = importlib.import_module('select_main_book')


@pytest.mark.parametrize('gross,cost', [(np.nan, .001), (.01, np.nan), (np.inf, .001)])
def test_held_trade_requires_finite_return_and_cost(gross, cost):
    with pytest.raises(ValueError, match='持仓'):
        morning.daily_pnl(np.array([[.2]]), np.array([[gross]]), np.array([[cost]]))


def test_cash_can_have_missing_quotes_without_hiding_real_cost():
    result = morning.daily_pnl(np.array([[0., -.2]]),
        np.array([[np.nan, -.01]]), np.array([[np.nan, .001]]))
    assert result[0] == pytest.approx(.0018)


def test_missing_signal_is_cash_and_not_counted_in_allocation():
    np.testing.assert_allclose(morning.allocate(np.array([[np.nan, 1.]]), None), [[0., 1.]])


def test_infinite_signal_is_rejected():
    with pytest.raises(ValueError):
        morning.allocate(np.array([[np.inf, 1.]]), .2)


def test_drawdown_includes_initial_capital():
    daily = pd.DataFrame({'a': [-.1, .02]})
    assert morning.drawdown_of(daily).iloc[0, 0] == pytest.approx(-.1)


@pytest.mark.parametrize('implementation', ['plot', 'book'])
@pytest.mark.parametrize('missing', ['entry', 'exit', 'exit_price'])
def test_future_missing_execution_never_erases_a_live_signal(monkeypatch, implementation, missing):
    day = pd.Timestamp('2021-06-15')
    idx = pd.date_range(day.replace(hour=9, minute=1), periods=17, freq='min').append(
        pd.DatetimeIndex([day.replace(hour=11, minute=30)]))
    close = np.r_[np.full(15, 100.), 103., 103., 102.]
    frame = pd.DataFrame({'close': close, 'open': close, 'highw': close,
        'trading_date': day}, index=idx)
    if missing in ('entry', 'exit'):
        hm = (9, 17) if missing == 'entry' else (11, 30)
        frame = frame.drop(day.replace(hour=hm[0], minute=hm[1]))
    else:
        frame.loc[day.replace(hour=11, minute=30), 'close'] = np.nan
    fees = pd.DataFrame({'symbol': ['A'], 'trading_date': [day],
        'commission_type': ['by_money'], 'open_commission': [0.], 'close_commission_today': [0.]})
    ticks = pd.DataFrame({'symbol': ['A'], 'year': [2020], 'tick': [.01]})
    monkeypatch.setattr(morning.costs, 'load_fees', lambda *args: fees)
    monkeypatch.setattr(morning.costs, 'load_ticks', lambda: ticks)
    monkeypatch.setattr(morning.shard_io, 'load_shard', lambda *args, **kwargs: frame.copy())
    master = pd.DatetimeIndex([day])
    if implementation == 'plot':
        run = lambda: morning.build(master, {2021: ['A']}, ['A'])
    else:
        monkeypatch.setattr(book, 'calendar_and_members', lambda p: (master, {2021: ['A']}, ['A'], []))
        monkeypatch.setattr(book, 'load_minutes', lambda *args: frame.copy())
        run = lambda: book.build('research', {'AM': ((9, 16), (9, 17), (11, 30))})
    with pytest.raises(ValueError, match='成交|退出'):
        run()
