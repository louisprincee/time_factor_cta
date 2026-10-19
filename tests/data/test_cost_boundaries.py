import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import save_shard
from tfcta.factors import align_asof, load_panel
from tfcta.costs import fee_tables, slippage_tables


def test_carry_research_cache_cannot_hide_future_dates(tmp_path):
    save_shard(pd.DataFrame({'carry_main_sub_annualized':[.1]},
        index=pd.to_datetime(['2023-01-04'])),tmp_path/'research','A','pickle')
    with pytest.raises(C.HoldoutViolation):
        load_panel(['A'],root=tmp_path)


def test_carry_uses_preceding_day_and_expires_stale_data():
    dates = pd.bdate_range('2021-12-20',periods=10)
    source = pd.Series([.1],index=dates[:1])
    aligned = align_asof(source,dates)
    assert np.isnan(aligned.iloc[0])
    assert aligned.iloc[1] == .1
    assert np.isnan(aligned.iloc[-1])


def test_historical_fees_split_open_close_and_old_contract_at_roll():
    dates = pd.bdate_range('2021-01-04',periods=3)
    source = pd.DataFrame({'symbol':'A','trading_date':dates,
        'contract':['A2105','A2109','A2109'],'commission_type':'by_volume',
        'open_commission':[1.,10.,10.],'close_commission':[2.,20.,20.]})
    prices = pd.DataFrame({'A':100.},index=dates)
    opened,closed,rolled = fee_tables(source,prices)
    assert opened['A'].tolist() == pytest.approx([.00101,.01001,.01001])
    assert closed['A'].tolist() == pytest.approx([.00201,.02001,.02001])
    assert rolled['A'].iloc[1] == pytest.approx(.00201)


def test_fee_does_not_backfill_from_future_first_observation():
    dates = pd.bdate_range('2021-01-04',periods=3)
    source = pd.DataFrame({'symbol':['A'],'trading_date':dates[-1:],
        'contract':['A2105'],'commission_type':['by_money'],
        'open_commission':[.0001],'close_commission':[.0002]})
    opened,_,_ = fee_tables(source,pd.DataFrame({'A':100.},index=dates))
    assert opened['A'].iloc[:2].isna().all()
    assert opened['A'].iloc[2] == pytest.approx(.000101)


def test_slippage_uses_past_tick_and_current_execution_price():
    dates = pd.to_datetime(['2020-01-02','2021-01-04'])
    table = pd.DataFrame({'symbol':['A','A'],'year':[2019,2020],'tick':[1.,2.]})
    prices = pd.DataFrame({'A':[100.,200.]},index=dates)
    assert slippage_tables(table,prices)['A'].tolist() == [.01,.01]
    future = pd.concat([table,pd.DataFrame({'symbol':['A'],'year':[2021],'tick':[50.]})])
    pd.testing.assert_frame_equal(slippage_tables(table,prices),slippage_tables(future,prices))


def test_fee_table_stops_reusing_a_rate_after_the_file_ends():
    dates = pd.bdate_range('2021-01-04', periods=60)
    source = pd.DataFrame({'symbol': 'A', 'trading_date': dates[:5], 'contract': 'A2105',
        'commission_type': 'by_volume', 'open_commission': 1., 'close_commission': 1.})
    opened, _, _ = fee_tables(source, pd.DataFrame({'A': 100.}, index=dates))
    assert opened['A'].iloc[:20].notna().all()
    assert opened['A'].iloc[-1:].isna().all()


def test_slippage_stops_reusing_a_tick_after_the_table_ends():
    dates = pd.to_datetime(['2018-01-02', '2019-01-02', '2020-01-02'])
    table = pd.DataFrame({'symbol': ['A'], 'year': [2017], 'tick': [1.]})
    slip = slippage_tables(table, pd.DataFrame({'A': [100., 100., 100.]}, index=dates))['A']
    assert slip.iloc[0] == pytest.approx(.01)   # 上一年的 tick
    assert slip.iloc[1] == pytest.approx(.01)   # 上一年缺，最多再沿用一年
    assert np.isnan(slip.iloc[2])
