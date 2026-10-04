import importlib.util
from pathlib import Path
import pandas as pd
import pytest
import sys


def module():
    path = Path(__file__).resolve().parents[2]/'scripts/compare_intraday.py'
    spec = importlib.util.spec_from_file_location('comparison', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_agreement_rule_does_not_trade_when_time_and_price_conflict():
    source = pd.DataFrame({'ret_15':[-.02,.02,-.02], 'prefix_dfp_top3':[.01,.01,float('nan')],
        'prefix_low_time':[.8,.8,.8], 'prefix_high_time':[.2,.2,.2], 'range_ratio':.01})
    prediction = module().rule_prediction(source, 'duration_price_agree')
    assert prediction.tolist() == [.015,0.,0.]


def test_selection_cannot_use_later_research_confirmation_scores():
    table = pd.DataFrame({'id':['a'], 'period':['2020-2021'], 'net_sharpe':[100.],
                          'positive_years':[2], 'trade_count':[1000]})
    with pytest.raises(ValueError, match='2017-2019'):
        module().select_candidates(table, [{'id':'a'}])


def test_selection_prioritizes_positive_years_and_keeps_predeclared_controls():
    specs = [dict(id=x, group=x, model='ridge',kind='ml', hold_minutes=30) for x in ['control','fragile','stable']]
    table = pd.DataFrame({'id':['control','fragile','stable'],'period':'2017-2019',
        'net_sharpe':[-1.,2.,1.], 'positive_years':[0,1,3], 'trade_count':[1000]*3})
    ids = module().select_candidates(table,specs,maximum=2,controls=['control'])
    assert ids == ['control','stable']


def test_summary_uses_explicit_pool_weights_for_symbol_concentration():
    dates=pd.to_datetime(['2017-01-03','2017-01-03','2018-01-02','2018-01-02'])
    trades=pd.DataFrame(dict(trading_date=dates,symbol=['A','B','A','B'],position=1.,
                             net=[.03,.01,-.01,.01],estimated_cost=.001))
    daily=pd.DataFrame(dict(net=[.02,0.],gross=[.021,.001],cost=.001,turnover=2.),index=dates.unique())
    result=module().metrics({'id':'test'},trades,daily,{2017:['A','B'],2018:['A','B']},2017,2019,'2017-2019')
    assert result['top_symbol_positive_share'] == pytest.approx(.5)
    assert result['trade_count'] == 4


def test_comparison_entry_point_finishes_all_stages_on_synthetic_samples(tmp_path,monkeypatch):
    mod=module()
    dates=pd.to_datetime([f'{year}-06-01' for year in range(2016,2022)])
    decisions=dates+pd.Timedelta(570,unit='min')
    sample=pd.DataFrame(dict(symbol='A',trading_date=dates,decision_time=decisions,
        entry_time=decisions+pd.Timedelta(1,unit='min'),exit_time=decisions+pd.Timedelta(15,unit='min'),
        gross_return=.01,estimated_cost=.001,actual_cost=.001,half_actual_cost=.0005,risk_scale=.01))
    for name in mod.F.FEATURES:
        sample[name]=.001
    monkeypatch.setattr(mod.U,'load_universe',lambda:{y:['A'] for y in range(2016,2022)})
    monkeypatch.setattr(mod.context,'run_dir',lambda _:tmp_path)
    monkeypatch.setattr(mod,'build_samples',lambda *_:({h:sample.copy() for h in (15,30,120)},list(dates),[]))
    monkeypatch.setattr(sys,'argv',['compare_intraday.py'])
    mod.main()
    assert len(pd.read_csv(tmp_path/'discovery.csv')) == 72
    confirmation=pd.read_csv(tmp_path/'confirmation.csv')
    assert confirmation.id.nunique() == 15
    assert set(confirmation.period)=={'2020-2021','2020-2021 half_tick_fixed_positions'}
