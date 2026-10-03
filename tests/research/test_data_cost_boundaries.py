import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import shard_io
from tfcta.factors import external
from tfcta.research.backtest import costs


def test_carry_research_cache_cannot_hide_future_dates(tmp_path):
    shard_io.save_shard(pd.DataFrame({'carry_main_sub_annualized':[.1]},
        index=pd.to_datetime(['2023-01-04'])),tmp_path/'research','A','pickle')
    with pytest.raises(C.HoldoutViolation):
        external.load_panel(['A'],root=tmp_path)


def test_carry_uses_preceding_day_and_expires_stale_data():
    dates = pd.bdate_range('2021-12-20',periods=10)
    source = pd.Series([.1],index=dates[:1])
    aligned = external.align_asof(source,dates)
    assert np.isnan(aligned.iloc[0])
    assert aligned.iloc[1] == .1
    assert np.isnan(aligned.iloc[-1])


def test_historical_fees_split_open_close_and_old_contract_at_roll():
    dates = pd.bdate_range('2021-01-04',periods=3)
    source = pd.DataFrame({'symbol':'A','trading_date':dates,
        'contract':['A2105','A2109','A2109'],'commission_type':'by_volume',
        'open_commission':[1.,10.,10.],'close_commission':[2.,20.,20.]})
    prices = pd.DataFrame({'A':100.},index=dates)
    opened,closed,rolled = costs.fee_tables(source,prices)
    assert opened['A'].tolist() == pytest.approx([.00101,.01001,.01001])
    assert closed['A'].tolist() == pytest.approx([.00201,.02001,.02001])
    assert rolled['A'].iloc[1] == pytest.approx(.00201)


def test_fee_does_not_backfill_from_future_first_observation():
    dates = pd.bdate_range('2021-01-04',periods=3)
    source = pd.DataFrame({'symbol':['A'],'trading_date':dates[-1:],
        'contract':['A2105'],'commission_type':['by_money'],
        'open_commission':[.0001],'close_commission':[.0002]})
    opened,_,_ = costs.fee_tables(source,pd.DataFrame({'A':100.},index=dates))
    assert opened['A'].iloc[:2].isna().all()
    assert opened['A'].iloc[2] == pytest.approx(.000101)


def test_slippage_uses_past_tick_and_current_execution_price():
    dates = pd.to_datetime(['2020-01-02','2021-01-04'])
    table = pd.DataFrame({'symbol':['A','A'],'year':[2019,2020],'tick':[1.,2.]})
    prices = pd.DataFrame({'A':[100.,200.]},index=dates)
    assert costs.slippage_tables(table,prices)['A'].tolist() == [.01,.01]
    future = pd.concat([table,pd.DataFrame({'symbol':['A'],'year':[2021],'tick':[50.]})])
    pd.testing.assert_frame_equal(costs.slippage_tables(table,prices),costs.slippage_tables(future,prices))
