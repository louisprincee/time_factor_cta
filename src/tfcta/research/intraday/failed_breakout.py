"""Causal day-session stable regions and failed-breakout reversal candidates.

Close-price stability is a hypothesis about acceptance, not measured fair value.
All thresholds use earlier trading days; every event freezes its old region.
"""
import numpy as np
import pandas as pd

from ... import config as C
from ...factors.intraday import raw_price_path

PRICE_FEATURES = ('distance_to_center', 'oriented_ret_5', 'oriented_ret_15', 'volatility_ratio')
QUALITY_FEATURES = ('anchor_minutes', 'anchor_volume_ratio', 'anchor_age')
TIME_FEATURES = ('return_minutes', 'excursion_size', 'return_fraction',
                 'deviation_quality', 'deviation_return_time')
FEATURES = PRICE_FEATURES + QUALITY_FEATURES + TIME_FEATURES
GROUPS = dict(price=PRICE_FEATURES, quality=PRICE_FEATURES+QUALITY_FEATURES,
              time=FEATURES)
EVENT_COLUMNS = ('decision_time','direction','reference_price','anchor_center',
                 'anchor_start','anchor_end','break_time','opportunity','scale_return')+FEATURES
SAMPLE_COLUMNS = ('symbol','trading_date','entry_time','exit_time','entry_price',
                  'exit_price','gross_return','oriented_return','risk_scale')+EVENT_COLUMNS


def trading_minutes(date):
    """Known common commodity day-session bar-end calendar, never observed-bar count."""
    date = pd.Timestamp(date).normalize()
    return pd.DatetimeIndex(np.concatenate([pd.date_range(date+pd.Timedelta(a,unit='min'),
        date+pd.Timedelta(b,unit='min'),freq='min').to_numpy()
        for a,b in [(541,615),(631,690),(811,900)]]))


def calibrate(source, lookback=20):
    """Prior-day median minute sigma and clock-specific median minute volume."""
    frame=source.copy()
    frame['trading_date']=pd.to_datetime(frame.trading_date).dt.normalize()
    C.assert_no_holdout_dates(frame.trading_date,'失败突破历史校准')
    frame['slot']=frame.index.hour*60+frame.index.minute
    same_date=frame.index.normalize()==pd.DatetimeIndex(frame.trading_date)
    slots=set(trading_minutes('2016-01-04').hour*60+trading_minutes('2016-01-04').minute)
    frame=frame.loc[same_date & frame.slot.isin(slots)].sort_index()
    frame['raw_close']=raw_price_path(frame)
    ret=frame.raw_close.pct_change(fill_method=None)
    continuous=frame.index.to_series().diff().eq(pd.Timedelta(1,unit='min'))
    ret=ret.where(continuous)
    sigma=ret.groupby(frame.trading_date).std().rolling(lookback,min_periods=lookback).median().shift(1)
    volume=frame.pivot(index='trading_date',columns='slot',values='volume')
    baseline=volume.rolling(lookback,min_periods=lookback).median().shift(1)
    return pd.DataFrame({'sigma':sigma,
        'volume_baseline':[baseline.loc[d] for d in sigma.index]},index=sigma.index)


def events_from_day(day, sigma, volume_baseline, tick, stable_minutes=10,
                    max_return_minutes=30, max_anchor_age=60):
    """Only observed prefixes; gaps/recesses reset all region and breakout state."""
    if not np.isfinite([sigma,tick]).all() or sigma<=0 or tick<=0:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    p=day.raw_close.to_numpy(dtype=float)
    v=day.volume.to_numpy(dtype=float)
    base=volume_baseline.reindex(day.index).to_numpy(dtype=float)
    if not np.isfinite(np.r_[p,v]).all() or (p<=0).any() or (v<0).any():
        raise ValueError('日盘报价或成交量无效')
    ratio=np.divide(v,base,out=np.full(len(v),np.nan),where=np.isfinite(base)&(base>0))
    rows=[]; anchor=None; pending=None; floor=0
    for i in range(len(p)):
        if i and day.index[i]-day.index[i-1]!=pd.Timedelta(1,unit='min'):
            anchor=pending=None; floor=i
        start=i-stable_minutes+1
        delta=max(p[i]*sigma*np.sqrt(stable_minutes),tick)
        stable=(start>=floor and np.isfinite(ratio[start:i+1]).all()
                and np.ptp(p[start:i+1])<=delta and ratio[start:i+1].mean()>=1.)
        if anchor is not None and i-anchor['end']>max_anchor_age:
            anchor=pending=None
        if pending is not None:
            if i-pending['index']>max_return_minutes:
                anchor=pending=None; floor=i; continue
            # A newly accepted region wholly outside the old frozen band vetoes reversal.
            if stable and start>anchor['end'] and (
                    p[start:i+1].min()>anchor['upper'] or p[start:i+1].max()<anchor['lower']):
                anchor=pending=None
            else:
                pending['peak']=max(pending['peak'],abs(p[i]-anchor['center']))
                if anchor['lower']<=p[i]<=anchor['upper']:
                    direction=-pending['sign']
                    scale=anchor['delta']/p[i]
                    opportunity=direction*(anchor['center']/p[i]-1.)
                    prefix_start=max(floor,i-20)
                    prefix=p[prefix_start:i+1]
                    realized=np.std(np.diff(prefix)/prefix[:-1]) if len(prefix)>1 else 0.
                    age=i-anchor['end']; duration=anchor['end']-anchor['start']+1
                    distance=opportunity/scale
                    elapsed=i-pending['index']
                    rows.append(dict(decision_time=day.index[i],direction=direction,
                        reference_price=p[i],anchor_center=anchor['center'],
                        anchor_start=day.index[anchor['start']],anchor_end=day.index[anchor['end']],
                        break_time=day.index[pending['index']],opportunity=opportunity,scale_return=scale,
                        distance_to_center=distance,
                        oriented_ret_5=direction*(p[i]/p[max(floor,i-5)]-1.),
                        oriented_ret_15=direction*(p[i]/p[max(floor,i-15)]-1.),
                        volatility_ratio=realized/sigma,anchor_minutes=duration,
                        anchor_volume_ratio=anchor['volume_ratio'],anchor_age=age,
                        return_minutes=elapsed,excursion_size=pending['peak']/anchor['delta'],
                        return_fraction=1.-abs(p[i]-anchor['center'])/pending['peak'],
                        deviation_quality=distance*np.log1p(duration)*anchor['volume_ratio'],
                        deviation_return_time=distance/(1.+elapsed)))
                    anchor=pending=None; floor=i+1; continue
        if anchor is None and stable:
            anchor=dict(start=start,end=i,center=p[start:i+1].mean(),
                lo=p[start:i+1].min(),hi=p[start:i+1].max(),delta=delta,
                volume_ratio=ratio[start:i+1].mean())
            anchor.update(lower=anchor['lo']-tick/2,upper=anchor['hi']+tick/2)
            continue
        if anchor is not None and pending is None:
            if p[i]>anchor['upper']+anchor['delta']/2 or p[i]<anchor['lower']-anchor['delta']/2:
                if anchor['volume_ratio']>=1:
                    pending=dict(index=i,sign=1 if p[i]>anchor['upper'] else -1,
                                 peak=abs(p[i]-anchor['center']))
                continue
            # Extend the same contiguous region rather than duplicate overlapping windows.
            a=anchor['start']
            if (i==anchor['end']+1 and max(anchor['hi'],p[i])-min(anchor['lo'],p[i])<=anchor['delta']
                    and np.isfinite(ratio[a:i+1]).all() and ratio[a:i+1].mean()>=1):
                anchor.update(end=i,center=p[a:i+1].mean(),lo=min(anchor['lo'],p[i]),
                    hi=max(anchor['hi'],p[i]),volume_ratio=ratio[a:i+1].mean())
                anchor.update(lower=anchor['lo']-tick/2,upper=anchor['hi']+tick/2)
            elif stable and start>anchor['end']:
                # Confirmation itself establishes the replacement; the next bar may break it.
                anchor=dict(start=start,end=i,center=p[start:i+1].mean(),
                    lo=p[start:i+1].min(),hi=p[start:i+1].max(),delta=delta,
                    volume_ratio=ratio[start:i+1].mean())
                anchor.update(lower=anchor['lo']-tick/2,upper=anchor['hi']+tick/2)
    return pd.DataFrame(rows,columns=EVENT_COLUMNS)


def samples_from_minutes(symbol, minute, ticks, hold_minutes=30, eligible_years=None):
    return build_panels(symbol,minute,ticks,(hold_minutes,),eligible_years)[hold_minutes]


def build_panels(symbol, minute, ticks, holds=(30,60), eligible_years=None):
    """Detect once and label predeclared horizons; no horizon-dependent features."""
    if not holds or set(holds)-{30,60}:
        raise ValueError('首版持仓只接受 30/60 个有效交易分钟')
    if not isinstance(minute.index,pd.DatetimeIndex) or minute.index.has_duplicates:
        raise ValueError('分钟时间戳必须唯一')
    source=minute.sort_index().copy()
    source['trading_date']=pd.to_datetime(source.trading_date).dt.normalize()
    C.assert_no_holdout_dates(source.trading_date,'失败突破研究分钟')
    if not source.trading_date.is_monotonic_increasing:
        raise ValueError('交易日顺序错误')
    if ticks.duplicated(['symbol','year']).any():
        raise ValueError('tick 历史重复')
    C.assert_no_holdout_dates(pd.to_datetime(ticks.year.astype(str)+'-01-01'),'失败突破 tick 历史')
    calibration=calibrate(source)
    source['raw_close']=raw_price_path(source)
    source['raw_open']=raw_price_path(pd.DataFrame({'close':source.open},index=source.index))
    tick_history=ticks[ticks.symbol.eq(symbol)].sort_values('year')
    rows={h:[] for h in holds}
    for date,whole_day in source.groupby('trading_date',sort=False):
        if date<pd.Timestamp(C.STUDY_START) or (eligible_years is not None and date.year not in eligible_years):
            continue
        prior=tick_history[tick_history.year<date.year]
        if prior.empty or date not in calibration.index or not np.isfinite(calibration.loc[date,'sigma']):
            continue
        calendar=trading_minutes(date)
        day=whole_day.loc[whole_day.index.isin(calendar)]
        baseline=calibration.loc[date,'volume_baseline']
        aligned=pd.Series(baseline.reindex(day.index.hour*60+day.index.minute).to_numpy(),index=day.index)
        sigma=calibration.loc[date,'sigma']
        events=events_from_day(day,sigma,aligned,float(prior.iloc[-1].tick.round(2)))
        for event in events.to_dict('records'):
            i=calendar.get_loc(event['decision_time'])
            for hold in holds:
                if i+hold>=len(calendar) or calendar[i+1]-calendar[i]!=pd.Timedelta(1,unit='min'):
                    continue  # Known clock boundary, not a test using future prices.
                required=calendar[i+1:i+hold+1]
                if not required.isin(day.index).all():
                    raise ValueError(f'{symbol} {date.date()} 预定持仓分钟缺失')
                entry,exit=required[0],required[-1]
                entry_price=float(day.loc[entry,'raw_open']);exit_price=float(day.loc[exit,'raw_close'])
                if not np.isfinite([entry_price,exit_price]).all() or min(entry_price,exit_price)<=0:
                    raise ValueError('成交报价无效')
                gross=exit_price/entry_price-1.
                rows[hold].append(dict(symbol=symbol,trading_date=date,entry_time=entry,exit_time=exit,
                    entry_price=entry_price,exit_price=exit_price,gross_return=gross,
                    oriented_return=event['direction']*gross,risk_scale=sigma*np.sqrt(hold),**event))
    return {h:pd.DataFrame(rows[h],columns=SAMPLE_COLUMNS) for h in holds}


def evaluate(predictions, universe, calendar=None):
    """First passing event each symbol/day. Negative predictions never flip direction."""
    from . import ml
    audit=predictions.copy().reset_index(drop=True)
    audit['trading_date']=pd.to_datetime(audit.trading_date)
    audit['decision_time']=pd.to_datetime(audit.decision_time)
    C.assert_no_holdout_dates(audit.trading_date,'失败突破候选选择')
    if not audit.direction.isin([-1,1]).all():
        raise ValueError('反转候选方向必须为正负一')
    if audit.duplicated(['symbol','decision_time']).any():
        raise ValueError('同品种事件时间重复')
    known_cost=np.isfinite(audit.estimated_cost)&audit.estimated_cost.ge(0)
    member=pd.Series([s in universe.get(d.year,[]) for s,d in zip(audit.symbol,audit.trading_date)],index=audit.index,dtype=bool)
    audit['cost_known']=known_cost
    audit['space_pass']=known_cost & np.isfinite(audit.opportunity) & audit.opportunity.gt(audit.estimated_cost)
    audit['model_pass']=known_cost & np.isfinite(audit.prediction) & audit.prediction.gt(audit.estimated_cost)
    eligible=audit.space_pass & audit.model_pass & member
    selected=audit[eligible].sort_values(['decision_time','symbol']).drop_duplicates(['symbol','trading_date'])
    audit['selected']=audit.index.isin(selected.index)
    signed=selected.copy()
    signed['oriented_prediction']=signed.prediction
    signed['prediction']=signed.direction*signed.prediction
    dates=audit.trading_date.unique() if calendar is None else calendar
    trades,daily=ml.evaluate(signed,universe,dates)
    return trades,daily,audit
