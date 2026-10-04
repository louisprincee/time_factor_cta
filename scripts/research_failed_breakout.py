"""Fixed 12 failed-breakout rule/Ridge comparisons on 2016–2021 only."""
import argparse
import hashlib
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tfcta import config as C
from tfcta.data import shard_io, universe as U
from tfcta.research.intraday import failed_breakout as F, ml
from tfcta.research.backtest import costs
from tfcta.research.analysis import stats
from tfcta.research.workflow import context


def definitions():
    return [dict(id=f'{model}_{group}_h{hold}',model=model,group=group,hold_minutes=hold,
        feature_names=list(names),training='expanding annual',label='direction * gross_return')
        for hold in (30,60) for group,names in F.GROUPS.items() for model in ('rule','ridge')]


def rule_predictions(samples, group):
    pred=samples.opportunity.copy()
    if group in ('quality','time'):
        pred=pred.where((samples.anchor_minutes>=20)&(samples.anchor_age<=15),0.)
    if group=='time':
        pred=pred.where((samples.return_minutes<=5)&(samples.return_fraction>=.8),0.)
    return samples.assign(prediction=pred)


def summarize(spec,trades,daily,audit,lo,hi):
    part=daily.loc[daily.index.year.to_series(index=daily.index).between(lo,hi)]
    t=trades[trades.trading_date.dt.year.between(lo,hi)]
    events=audit[audit.trading_date.dt.year.between(lo,hi)]
    result=dict(id=spec['id'],model=spec['model'],group=spec['group'],hold_minutes=spec['hold_minutes'],
        period=f'{lo}-{hi}',candidate_count=len(events),trade_count=len(t),
        cost_known=int(events.cost_known.sum()),space_pass=int(events.space_pass.sum()),
        model_pass=int(events.model_pass.sum()),positive_years=int((part.net.groupby(part.index.year).sum()>0).sum()),
        annual_cost=part.cost.mean()*252,annual_turnover=part.turnover.mean()*252)
    for name in ('gross','net'):
        result.update({name+'_'+k:v for k,v in stats.performance(part[name]).items()})
        result[name+'_sharpe']=stats.sharpe_ratio(part[name])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols',nargs='+')
    parser.add_argument('--min-train',type=int,default=1000)
    args=parser.parse_args(argv)
    if args.min_train<1:
        parser.error('训练样本下限必须为正')
    universe={y:ss for y,ss in U.load_universe().items() if 2016<=y<=2021}
    symbols=sorted({s for ss in universe.values() for s in ss})
    if args.symbols:
        if set(args.symbols)-set(symbols):
            parser.error('品种不在研究池中')
        universe={y:[s for s in ss if s in args.symbols] for y,ss in universe.items()}
        symbols=sorted({s for ss in universe.values() for s in ss})
    run=context.run_dir('failed_breakout')
    specs=definitions()
    code=list((C.PROJECT_ROOT/'src').rglob('*.py'))+list((C.PROJECT_ROOT/'scripts').glob('*.py'))
    context.dump_json(run/'definition.json',dict(status='research_only_unfrozen',specs=specs,
        arguments=vars(args),universe=universe,cost_ticks_per_side=1.,
        calibration='20 preceding trading-day median minute sigma and clock-matched volume; no current day',
        region='10 consecutive closes, range <= price*sigma*sqrt(10), floor prior tick; mean volume/baseline >=1',
        event='half-scale outside frozen band then return within 30 minutes; new liquid region veto; anchor age<=60',
        execution='next consecutive minute raw open; fixed 30/60 scheduled trading minutes; max1 trade/symbol/day; fixed1/N',
        gating='positive direction-oriented prediction and frozen center opportunity both exceed estimated cost',
        rule_quality='anchor duration>=20, age<=15',rule_time='quality plus return minutes<=5 and recovery fraction>=0.8',
        ridge='alpha10; training-only imputer/scaler; annual label availability purge; cannot reverse rule direction',
        research='2016 initial train; 2017-2019 early; 2020-2021 already seen/reused internal research',
        validation_2022='not_run',oos_2023_2025='locked',
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        code_sha256={str(p.relative_to(C.PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code}))
    fee,ticks=costs.load_fees(),costs.load_ticks()
    inputs=[C.UNIVERSE_DIR/'universe_by_year.json',C.RESEARCH_OUT_DIR/'fee_history.csv',C.RESEARCH_OUT_DIR/'tick_size.csv']
    frames={30:[],60:[]};calendar=set()
    for n,symbol in enumerate(symbols,1):
        minute=shard_io.load_shard(symbol,columns=['open','close','volume','trading_date'])
        years=[y for y,ss in universe.items() if symbol in ss]
        dates=pd.DatetimeIndex(minute.trading_date.unique())
        calendar.update(dates[dates.year.isin(years)])
        panels=F.build_panels(symbol,minute,ticks,eligible_years=years)
        for hold,samples in panels.items():
            if not samples.empty:
                frames[hold].append(samples.join(costs.intraday_costs(samples,fee,ticks,1.)))
        inputs.append(shard_io.find_shard(C.RESEARCH_DIR,symbol))
        print(f'Prepared {n}/{len(symbols)} {symbol}: '+str({h:len(s) for h,s in panels.items()}),flush=True)
    context.dump_json(run/'inputs.json',dict(sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}))
    results=[]
    for hold,parts in frames.items():
        if not parts:
            parser.error(f'{hold} 分钟无候选，运行定义已留痕：{run}')
        panel=pd.concat(parts,ignore_index=True).sort_values(['decision_time','symbol']).reset_index(drop=True)
        panel.to_parquet(run/f'samples_h{hold}.parquet',index=False)
        for spec in [s for s in specs if s['hold_minutes']==hold]:
            if spec['model']=='ridge':
                predictions,folds=ml.walk_forward(panel,tuple(spec['feature_names']),min_train=args.min_train,label='oriented_return')
            else:
                predictions=rule_predictions(panel,spec['group']);folds=pd.DataFrame()
            trades,daily,audit=F.evaluate(predictions,universe,sorted(calendar))
            trades.to_parquet(run/f"{spec['id']}_trades.parquet",index=False)
            audit.to_parquet(run/f"{spec['id']}_candidates.parquet",index=False)
            daily.to_csv(run/f"{spec['id']}_daily.csv")
            folds.to_csv(run/f"{spec['id']}_folds.csv",index=False)
            for lo,hi in [(2017,2019),(2020,2021),(2017,2021)]+[(y,y) for y in range(2017,2022)]:
                results.append(summarize(spec,trades,daily,audit,lo,hi))
            print(f"Evaluated {spec['id']}: {len(trades)} trades including 2016 rule warmup",flush=True)
    pd.DataFrame(results).to_csv(run/'performance.csv',index=False)
    print(f'Complete: {run}',flush=True)
    return run


if __name__=='__main__':
    main()
