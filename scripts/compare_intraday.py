"""Predeclared intraday factor/model study; selection uses 2017–2019 only.

Fixed 09:30 decision, 15/30/120 wall-clock minutes holding, same-day exit.
2016 trains; 2020–2021 is internal research confirmation for the selected list.
No 2022 or strict OOS reads. No hyperparameter grid or automatic sign flips.
"""
import argparse
import hashlib
import platform
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tfcta import config as C
from tfcta.data import shard_io, universe as U
from tfcta.factors.intraday import raw_price_path
from tfcta.research.intraday import features as F, ml
from tfcta.research.backtest import costs
from tfcta.research.analysis import stats
from tfcta.research.workflow import context

GROUPS = dict(time=F.TIME_FEATURES, price=F.PRICE_FEATURES, all=F.FEATURES,
    duration_price=F.PRICE_FEATURES+F.TIME_FEATURES[:2],
    extremes_price=F.PRICE_FEATURES+F.TIME_FEATURES[2:],
    flow_time=F.TIME_FEATURES+('ret_5','ret_15','close_vwap_proxy_dev','volume_last5_share'))
RULES = ('momentum','price_reversion','duration_reversion','time_reversion',
         'duration_price_agree','duration_time_agree')
CONTROLS = ('ml_price_ridge_h30','ml_price_hgb_h30','rule_price_reversion_h30')


def definitions():
    specs = []
    for hold in (15,30,120):
        for group in GROUPS:
            for model in ('ridge','hgb'):
                specs.append(dict(id=f'ml_{group}_{model}_h{hold}', kind='ml', group=group,
                    model=model, hold_minutes=hold, training_years=None, refit='annual', target='raw'))
        for rule in RULES:
            specs.append(dict(id=f'rule_{rule}_h{hold}',kind='rule',group=rule,
                              model='rule',hold_minutes=hold,rule=rule))
    for group in ('time','price','all'):
        for model in ('ridge','hgb'):
            for name,years,refit,target in [('roll2',2,'annual','raw'),
                    ('roll2_quarter',2,'quarterly','raw'),('scaled',None,'annual','risk_scaled')]:
                specs.append(dict(id=f'ml_{group}_{model}_h30_{name}', kind='ml',group=group,
                    model=model,hold_minutes=30,training_years=years,refit=refit,target=target))
    return specs


def rule_prediction(data, name):
    momentum = data.ret_15
    duration = data.prefix_dfp_top3
    pressure = (data.prefix_low_time-data.prefix_high_time)*data.range_ratio
    values = dict(momentum=momentum, price_reversion=-momentum,
                  duration_reversion=duration, time_reversion=pressure)
    if name in values:
        return values[name].fillna(0.)
    other = -momentum if name=='duration_price_agree' else pressure
    if name not in ('duration_price_agree','duration_time_agree'):
        raise ValueError('未知经济规则')
    return ((duration+other)/2).where(np.sign(duration)==np.sign(other),0.).fillna(0.)


def select_candidates(table, specs, maximum=15, controls=CONTROLS):
    if not table.period.eq('2017-2019').all():
        raise ValueError('筛选只接受 2017-2019 成绩，禁止使用后段确认成绩')
    by_id = {s['id']:s for s in specs}
    selected = [s for s in controls if s in set(table.id)]
    ranked = table.assign(eligible=(table.positive_years>=2)&(table.trade_count>=300)&(table.net_sharpe>0))
    ranked = ranked.sort_values(['eligible','net_sharpe','id'],ascending=[False,False,True],na_position='last')
    groups,models,holds = Counter(),Counter(),Counter()
    for candidate in selected:
        s=by_id[candidate]; groups[s['group']]+=1; models[s['model']]+=1; holds[s['hold_minutes']]+=1
    for candidate in ranked.id:
        if len(selected)>=maximum:
            break
        if candidate in selected:
            continue
        s=by_id[candidate]
        if groups[s['group']]>=3 or models[s['model']]>=9 or holds[s['hold_minutes']]>=9:
            continue
        selected.append(candidate)
        groups[s['group']]+=1; models[s['model']]+=1; holds[s['hold_minutes']]+=1
    return selected


def build_samples(universe, run):
    frames = {h:[] for h in (15,30,120)}
    calendar=set()
    fee,ticks=costs.load_fees(),costs.load_ticks()
    inputs=[C.UNIVERSE_DIR/'universe_by_year.json',C.RESEARCH_OUT_DIR/'fee_history.csv',C.RESEARCH_OUT_DIR/'tick_size.csv']
    symbols=sorted({s for ss in universe.values() for s in ss})
    for i,symbol in enumerate(symbols,1):
        minute=shard_io.load_shard(symbol,columns=F.MINUTE_COLUMNS)
        years=[y for y,ss in universe.items() if symbol in ss]
        dates=pd.DatetimeIndex(minute.trading_date.unique())
        calendar.update(dates[dates.year.isin(years)])
        base=F.samples_from_minutes(symbol,minute,30,15,eligible_years=years)
        raw=pd.Series(raw_price_path(minute),index=minute.index)
        for hold in frames:
            sample=base.copy()
            sample['exit_time']=sample.decision_time+pd.Timedelta(hold,unit='min')
            sample['exit_price']=raw.reindex(sample.exit_time).to_numpy()
            if sample.exit_price.isna().any() or (sample.exit_price<=0).any():
                raise ValueError(f'{symbol} 预定出场价缺失')
            sample['gross_return']=sample.exit_price/sample.entry_price-1
            sample['risk_scale']=(sample.realized_vol*np.sqrt(hold)).clip(lower=.0001)
            charge=costs.intraday_costs(sample,fee,ticks,1.)
            sample=sample.join(charge)
            sample['half_actual_cost']=costs.intraday_costs(sample,fee,ticks,.5).actual_cost
            frames[hold].append(sample)
        inputs.append(shard_io.find_shard(C.RESEARCH_DIR,symbol))
        print(f'Data {i}/{len(symbols)}: {symbol}',flush=True)
    panels={h:pd.concat(ss,ignore_index=True).sort_values(['decision_time','symbol']).reset_index(drop=True)
            for h,ss in frames.items()}
    for hold,panel in panels.items():
        panel.to_parquet(run/f'samples_h{hold}.parquet',index=False)
    return panels,sorted(calendar),inputs


def run_spec(data,spec,universe,calendar):
    if spec['kind']=='rule':
        predictions=data.assign(prediction=rule_prediction(data,spec['rule']))
        folds=pd.DataFrame()
    else:
        predictions,folds=ml.walk_forward(data,GROUPS[spec['group']],spec['model'],spec.get('min_train',1000),
            spec['training_years'],spec['refit'],spec['target'],spec.get('scope','pooled'))
    trades,daily=ml.evaluate(predictions,universe,calendar)
    return trades,daily,folds


def metrics(spec,trades,daily,universe,lo,hi,label):
    part=daily.loc[daily.index.year.to_series(index=daily.index).between(lo,hi)]
    t=trades.loc[trades.trading_date.dt.year.between(lo,hi)]
    row={**spec,'period':label}
    for name in ('gross','net'):
        row.update({f'{name}_{k}':v for k,v in stats.performance(part[name]).items()})
        row[f'{name}_sharpe']=stats.sharpe_ratio(part[name])
    yearly=part.net.groupby(part.index.year).sum()
    row.update(positive_years=int((yearly>0).sum()),trade_count=int(t.position.ne(0).sum()),
        annual_cost=part.cost.mean()*252,annual_turnover=part.turnover.mean()*252,
        decision_activity=float(t.position.ne(0).mean()),missing_cost_fraction=float(t.estimated_cost.isna().mean()))
    pnl=t.assign(contribution=t.net/[len(universe[d.year]) for d in t.trading_date]).groupby('symbol').contribution.sum()
    positive=pnl.clip(lower=0)
    row['top_symbol_positive_share']=positive.max()/positive.sum() if positive.sum()>0 else np.nan
    return row


def factor_ic(panels,lo,hi):
    rows=[]
    for hold,data in panels.items():
        frame=data[data.trading_date.dt.year.between(lo,hi)]
        for name in F.FEATURES:
            correlations=[]
            for _,sample in frame.groupby('symbol'):
                sample=sample[[name,'gross_return']].dropna()
                if len(sample)>=60 and sample[name].nunique()>1:
                    correlations.append(sample[name].corr(sample.gross_return,method='spearman'))
            rows.append(dict(hold_minutes=hold,feature=name,period=f'{lo}-{hi}',
                mean_symbol_spearman=np.nanmean(correlations) if correlations else np.nan,n_symbols=len(correlations)))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols',nargs='+')
    args=parser.parse_args()
    universe={y:ss for y,ss in U.load_universe().items() if 2016<=y<=2021}
    if args.symbols:
        known={s for ss in universe.values() for s in ss}
        if set(args.symbols)-known:
            parser.error('研究池内无此品种')
        universe={y:[s for s in ss if s in args.symbols] for y,ss in universe.items()}
    specs=definitions()
    run=context.run_dir('intraday_comparison')
    code=list((C.PROJECT_ROOT/'src').rglob('*.py'))+list((C.PROJECT_ROOT/'scripts').glob('*.py'))
    context.dump_json(run/'protocol.json',dict(specs=specs,decision_time='09:30',
        discovery='2017-2019',confirmation='2020-2021 internal research, not strict OOS',
        selection='3 controls + rank by >=2 positive years, >=300 trades, positive net Sharpe, then Sharpe; group/model/horizon caps; max 15',
        cost_ticks_per_side=1.,sensitivity='0.5 ticks per side, fixed primary positions; no regating',
        thresholds='250 preceding trading days pooled minute changes, 55th percentile',
        risk_target='known prefix realized volatility * sqrt(wall-clock hold), floor 0.0001',
        ml_settings='Ridge alpha10; HGB depth3/100 iterations/lr0.05/early_stopping=False',
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        code_sha256={str(p.relative_to(C.PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        validation_2022='not_run',oos_2023_2025='locked'))
    panels,calendar,inputs=build_samples(universe,run)
    context.dump_json(run/'inputs.json',dict(universe=universe,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}))
    factor_ic(panels,2017,2019).to_csv(run/'factor_ic_discovery.csv',index=False)
    discovery,years=[],[]
    folder=run/'discovery';folder.mkdir()
    for i,spec in enumerate(specs,1):
        data=panels[spec['hold_minutes']]
        data=data[data.trading_date.dt.year<=2019].copy()
        dates=[d for d in calendar if d.year<=2019]
        trades,daily,folds=run_spec(data,spec,universe,dates)
        discovery.append(metrics(spec,trades,daily,universe,2017,2019,'2017-2019'))
        years.extend(metrics(spec,trades,daily,universe,y,y,str(y)) for y in (2017,2018,2019))
        daily.to_csv(folder/f"{spec['id']}_daily.csv")
        trades.to_parquet(folder/f"{spec['id']}_predictions.parquet",index=False)
        folds.to_csv(folder/f"{spec['id']}_folds.csv",index=False)
        print(f'Discovery {i}/{len(specs)}: {spec["id"]}',flush=True)
    table=pd.DataFrame(discovery)
    table.to_csv(run/'discovery.csv',index=False)
    pd.DataFrame(years).to_csv(run/'discovery_years.csv',index=False)
    chosen=select_candidates(table,specs)
    context.dump_json(run/'selected_before_confirmation.json',dict(ids=chosen,selection_window='2017-2019 only'))
    confirm=[]
    folder=run/'confirmation';folder.mkdir()
    for i,spec in enumerate(s for s in specs if s['id'] in chosen):
        trades,daily,folds=run_spec(panels[spec['hold_minutes']],spec,universe,calendar)
        confirm.append(metrics(spec,trades,daily,universe,2020,2021,'2020-2021'))
        years.extend(metrics(spec,trades,daily,universe,y,y,str(y)) for y in (2020,2021))
        daily.to_csv(folder/f"{spec['id']}_daily.csv")
        trades.to_parquet(folder/f"{spec['id']}_predictions.parquet",index=False)
        folds.to_csv(folder/f"{spec['id']}_folds.csv",index=False)
        t=trades.assign(cost=trades.position.abs()*trades.half_actual_cost.fillna(0.))
        t['net']=t.gross-t.cost
        d=t.groupby('trading_date')[['gross','cost','net','turnover']].sum().reindex(daily.index,fill_value=0)
        d=d.div(pd.Series({day:len(universe[day.year]) for day in d.index}),axis=0)
        alternate=metrics(spec,t,d,universe,2020,2021,'2020-2021 half_tick_fixed_positions')
        confirm.append(alternate)
        print(f'Confirmation {i+1}/{len(chosen)}: {spec["id"]}',flush=True)
    pd.DataFrame(confirm).to_csv(run/'confirmation.csv',index=False)
    pd.DataFrame(years).to_csv(run/'yearly.csv',index=False)
    print(f'Complete: {run}',flush=True)


if __name__=='__main__':
    main()
