import numpy as np
import pandas as pd
import pytest
import importlib.util
import json
from pathlib import Path

from tfcta import config as C
from tfcta.research.intraday import failed_breakout as fb
from tfcta.research.intraday import ml


def day(prices, start='2017-01-03 09:01', volume=10.):
    idx = pd.date_range(start, periods=len(prices), freq='min')
    return pd.DataFrame({'raw_close':prices, 'volume':volume}, index=idx)


def detect(source):
    return fb.events_from_day(source, .0031622776601683794,
                              pd.Series(10., index=source.index), .2)


@pytest.mark.parametrize('tail,direction,reference', [([102.,103.,100.4],-1,100.4),
                                                    ([98.,97.,100.],1,100.)])
def test_breakout_return_freezes_old_center_and_has_opposite_direction(tail,direction,reference):
    source=day([100.,100.4]*5+tail)
    events=detect(source)
    assert len(events)==1
    e=events.iloc[0]
    assert e.decision_time==pd.Timestamp('2017-01-03 09:13')
    assert e.direction==direction
    assert e.anchor_center==pytest.approx(100.2)
    assert e.reference_price==reference
    assert e.return_minutes==2
    assert e.anchor_minutes==10
    assert e.anchor_volume_ratio==1


def test_low_volume_stall_is_not_a_valid_consensus_anchor():
    assert detect(day([100.,100.4]*5+[102.,103.,100.4],volume=2.)).empty


def test_a_new_liquid_stable_region_outside_anchor_invalidates_old_breakout():
    assert detect(day([100.,100.4]*5+[103.]*10+[100.4])).empty


def test_newer_region_is_available_for_a_breakout_on_the_very_next_minute():
    source=day([99.5,100.5]*5+[100.8]*10+[102.,100.85])
    events=detect(source)
    assert len(events)==1
    e=events.iloc[0]
    assert e.decision_time==pd.Timestamp('2017-01-03 09:22')
    assert e.anchor_start==pd.Timestamp('2017-01-03 09:11')
    assert e.anchor_end==pd.Timestamp('2017-01-03 09:20')
    assert e.anchor_center==pytest.approx(100.8)
    assert e.direction==-1


def test_future_prices_and_volume_cannot_change_an_already_observed_event():
    source=day([100.,100.4]*5+[102.,103.,100.4]+[101.]*20)
    original=detect(source)
    altered=source.copy()
    altered.loc[altered.index>pd.Timestamp('2017-01-03 09:13'),'raw_close']=500.
    altered.loc[altered.index>pd.Timestamp('2017-01-03 09:13'),'volume']=1000.
    pd.testing.assert_frame_equal(original.iloc[:1],detect(altered).iloc[:1])


def test_a_missing_minute_or_session_break_cannot_be_part_of_a_stable_window():
    source=day([100.,100.4]*5+[102.,103.,100.4])
    assert detect(source.drop(source.index[4])).empty
    before=day([100.,100.4]*5+[102.],start='2017-01-03 10:05')
    after=day([100.4],start='2017-01-03 10:31')
    assert detect(pd.concat([before,after])).empty


def test_return_after_the_predeclared_timeout_is_not_a_reversal_candidate():
    # Alternating outside prices prevent a new stable anchor; return after 31 bars expires.
    source=day([100.,100.4]*5+[102.]+[102.,105.]*15+[100.4])
    assert detect(source).empty


def history():
    parts=[]
    for date in pd.bdate_range('2016-01-04',periods=23):
        idx=fb.trading_minutes(date)
        prices=100.+np.arange(len(idx))%2*.4
        prices[10:13]=[102.,103.,100.4]
        parts.append(pd.DataFrame({'close':prices,'open':prices-.01,'volume':10.,
                                  'trading_date':date},index=idx))
    return pd.concat(parts)


def test_calibration_is_prior_day_only_and_volume_is_clock_matched():
    source=history()
    first=fb.calibrate(source,lookback=20)
    altered=source.copy()
    last=altered.trading_date==altered.trading_date.max()
    altered.loc[last,'volume']=999.
    altered.loc[last,'close']=999.
    second=fb.calibrate(altered,lookback=20)
    assert first.iloc[:20].sigma.isna().all()
    assert first.iloc[20].volume_baseline.loc[541]==10.
    assert first.iloc[-1].sigma>0
    assert first.iloc[-1].sigma==second.iloc[-1].sigma
    pd.testing.assert_series_equal(first.iloc[-1].volume_baseline,second.iloc[-1].volume_baseline)


def test_sample_labels_count_scheduled_trading_minutes_across_recess():
    source=history()
    # Event returns at 10:10; next open 10:11, 30th holding bar closes 10:55.
    last_date=source.trading_date.max()
    idx=fb.trading_minutes(last_date)
    p=100.+np.arange(len(idx))%2*.4
    p[:57]=np.arange(57)+200.
    p[57:67]=[100.,100.4]*5
    p[67:70]=[102.,103.,100.4]
    source.loc[source.trading_date.eq(last_date),'close']=p
    source.loc[source.trading_date.eq(last_date),'open']=p-.01
    ticks=pd.DataFrame({'symbol':['A'],'year':[2015],'tick':[.2]})
    samples=fb.samples_from_minutes('A',source,ticks,30)
    event=samples[samples.decision_time.eq(last_date+pd.Timedelta(610,unit='min'))].iloc[0]
    assert event.entry_time==last_date+pd.Timedelta(611,unit='min')
    assert event.exit_time==last_date+pd.Timedelta(655,unit='min')
    assert event.entry_price==pytest.approx(99.99)
    assert event.oriented_return==pytest.approx(-event.gross_return)
    altered=source.drop(last_date+pd.Timedelta(640,unit='min'))
    with pytest.raises(ValueError,match='缺失'):
        fb.samples_from_minutes('A',altered,ticks,30)


def test_late_event_is_rejected_by_known_close_calendar_and_night_is_separate():
    source=history()
    date=source.trading_date.max()
    idx=fb.trading_minutes(date)
    p=200.+np.arange(len(idx))
    p[-23:-13]=[100.,100.4]*5
    p[-13:-10]=[102.,103.,100.4]  # 14:50 return, cannot hold 30 trading minutes.
    source.loc[source.trading_date.eq(date),'close']=p
    source.loc[source.trading_date.eq(date),'open']=p-.01
    tick=pd.DataFrame(dict(symbol=['A'],year=[2015],tick=[.2]))
    first=fb.samples_from_minutes('A',source,tick,30)
    assert not first.trading_date.eq(date).any()
    night_idx=pd.date_range(date-pd.Timedelta(3,unit='h'),periods=13,freq='min')
    prices=[100.,100.4]*5+[102.,103.,100.4]
    night=pd.DataFrame(dict(close=prices,open=prices,volume=999.,trading_date=date),index=night_idx)
    second=fb.samples_from_minutes('A',pd.concat([source,night]).sort_index(),tick,30)
    pd.testing.assert_frame_equal(first,second)


def test_event_builder_rejects_holdout_even_if_all_prices_would_be_filtered():
    source=history()
    source['trading_date']=pd.Timestamp('2023-01-03')
    with pytest.raises(C.HoldoutViolation):
        fb.samples_from_minutes('A',source,pd.DataFrame(),30)


def candidate_panel():
    dates=pd.to_datetime(['2016-06-01','2016-07-01','2017-01-03','2017-01-03','2017-01-03'])
    decisions=dates+pd.to_timedelta([570,570,570,590,610],unit='min')
    frame=pd.DataFrame(dict(symbol='A',trading_date=dates,decision_time=decisions,
        entry_time=decisions+pd.Timedelta(1,unit='min'),exit_time=decisions+pd.Timedelta(30,unit='min'),
        gross_return=[-.02,.02,.01,.01,.01],direction=[-1,1,-1,1,-1],
        oriented_return=[.02]*5,opportunity=[.02]*5,estimated_cost=[.01]*5,actual_cost=[.012]*5))
    for name in fb.FEATURES:
        frame[name]=1.
    return frame


def test_ridge_learns_direction_oriented_labels_using_train_only():
    source=candidate_panel()
    first,folds=ml.walk_forward(source,fb.FEATURES,min_train=1,label='oriented_return')
    assert first.prediction.iloc[2:].tolist()==pytest.approx([.02]*3)
    changed=source.copy()
    changed.loc[2:,'oriented_return']=100.
    second,_=ml.walk_forward(changed,fb.FEATURES,min_train=1,label='oriented_return')
    np.testing.assert_allclose(first.prediction,second.prediction,equal_nan=True)
    assert (folds.train_last_exit<folds.test_first_decision).all()


def test_first_passing_event_is_causal_model_never_flips_reversal_direction():
    source=candidate_panel().iloc[2:].copy()
    source['prediction']=[-.1,.03,.99]
    trades,daily,audit=fb.evaluate(source,{2017:['A','B']},calendar=[pd.Timestamp('2017-01-03'),pd.Timestamp('2017-01-04')])
    assert len(trades)==1
    assert trades.decision_time.iloc[0]==pd.Timestamp('2017-01-03 09:50')
    assert trades.position.iloc[0]==1.
    assert daily.loc['2017-01-03','net']==pytest.approx(-.001)
    assert daily.loc['2017-01-04','net']==0.
    assert audit.selected.tolist()==[False,True,False]
    source.loc[source.index[-1],'prediction']=1000.
    assert fb.evaluate(source,{2017:['A','B']})[0].decision_time.iloc[0]==pd.Timestamp('2017-01-03 09:50')


def test_profit_space_and_cost_both_gate_and_negative_direction_is_preserved():
    source=candidate_panel().iloc[2:].copy()
    source['prediction']=[.03,.03,.03]
    source['opportunity']=[.005,.005,.02]
    trades,daily,_=fb.evaluate(source,{2017:['A']})
    assert trades.position.tolist()==[-1.]
    assert daily.net.iloc[0]==pytest.approx(-.022)
    source['estimated_cost']=np.nan
    assert fb.evaluate(source,{2017:['A']})[0].empty


def test_research_entry_runs_real_event_training_and_keeps_cash_days(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('failed_entry',C.PROJECT_ROOT/'scripts/research_failed_breakout.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    parts=[]
    for year in range(2016,2022):
        for date in pd.bdate_range(f'{year}-01-04',periods=28):
            idx=fb.trading_minutes(date)
            prices=100.+np.arange(len(idx))%2*.4
            prices[10:13]=[102.,103.,100.4]
            # One deliberately eventless day: must remain in the daily denominator/calendar.
            if date==pd.Timestamp('2017-01-04'):
                prices=np.arange(len(idx))*.2+100.
            parts.append(pd.DataFrame(dict(close=prices,open=prices-.01,volume=10.,trading_date=date),index=idx))
    minute=pd.concat(parts)
    research=tmp_path/'research';research.mkdir()
    universe_path=tmp_path/'universe';universe_path.mkdir()
    pd.DataFrame(dict(symbol=['A'],trading_date=['2015-01-01'],commission_type=['by_volume'],
        open_commission=[0.],close_commission=[0.],close_commission_today=[0.])).to_csv(research/'fee_history.csv',index=False)
    pd.DataFrame(dict(symbol=['A'],year=[2015],tick=[.01])).to_csv(research/'tick_size.csv',index=False)
    (universe_path/'universe_by_year.json').write_text(json.dumps({str(y):['A'] for y in range(2016,2022)}))
    shard=tmp_path/'A.parquet';minute.to_parquet(shard)
    monkeypatch.setattr(C,'RESEARCH_OUT_DIR',research)
    monkeypatch.setattr(C,'UNIVERSE_DIR',universe_path)
    monkeypatch.setattr(C,'RUNS_DIR',tmp_path/'runs')
    # The boundary mock substitutes an isolated synthetic shard; all events, fits and costs are real.
    monkeypatch.setattr(entry.shard_io,'load_shard',lambda symbol,columns:minute[columns].copy())
    monkeypatch.setattr(entry.shard_io,'find_shard',lambda directory,symbol:shard)
    entry.main(['--min-train','1'])
    run=next((tmp_path/'runs').iterdir())
    table=pd.read_csv(run/'performance.csv')
    assert table.id.nunique()==12
    assert table.trade_count.max()>0
    folds=pd.read_csv(run/'ridge_time_h30_folds.csv')
    assert folds.fitted.any()
    daily=pd.read_csv(run/'ridge_time_h30_daily.csv',index_col=0,parse_dates=True)
    assert len(daily)==168
    assert daily.loc['2017-01-04','net']==0
    definitions=json.loads((run/'definition.json').read_text())
    assert definitions['validation_2022']=='not_run'
    saved=pd.read_parquet(run/'samples_h30.parquet')
    assert len(saved)>100
    assert saved.trading_date.dt.year.max()==2021
