"""Shared research workflow: explicit hypotheses, one daily engine, inspectable outputs."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..data import bars as B, universe as U, shard_io
from ..factors import library, cache
from .backtest import engine, costs
from .analysis import stats


@dataclass
class StudyData:
    factors: library.SignalSet
    returns: pd.DataFrame
    universe: dict
    rolls: pd.DataFrame
    open_fee: pd.DataFrame
    close_fee: pd.DataFrame
    roll_close_fee: pd.DataFrame
    slippage: pd.DataFrame


def load_research(n_ticks=1.):
    universe = {y:s for y,s in U.load_universe().items() if 2016<=y<=2021}
    symbols = sorted({s for ss in universe.values() for s in ss})
    absent = set(symbols)-set(shard_io.list_shards(C.RESEARCH_DIR))
    if absent:
        raise FileNotFoundError(f"研究期品种池缺分钟分片: {sorted(absent)}")
    factors = library.load(symbols)
    prices = factors.bars['open']
    returns = (factors.bars['openw'].shift(-1)-factors.bars['openw'])/prices
    C.assert_no_holdout_dates(returns.index,'研究收益')
    # Final open has no next open in this partition; never borrow 2022 prices.
    index = returns.index[(returns.index>=pd.Timestamp(C.STUDY_START)) & returns.notna().any(axis=1)]
    returns = returns.loc[index]
    rolls = pd.DataFrame(False,index=index,columns=symbols)
    for symbol,frame in B.load_roll_calendar(symbols,C.RESEARCH_END).items():
        days = index.intersection(frame.trading_date)
        rolls.loc[days,symbol] = True
    o,c,r = costs.fee_tables(costs.load_fees(),prices)
    slip = costs.slippage_tables(costs.load_ticks(),prices.loc[index],n_ticks)
    return StudyData(factors,returns,universe,rolls,o.loc[index],c.loc[index],r.loc[index],slip)


def validate_specs(specs):
    seen = set()
    allowed = {'id','factors','timing','combine','days','mode','phase','vol_target','cap','hypothesis'}
    for spec in specs:
        if set(spec)-allowed:
            raise ValueError(f"未知候选字段 {set(spec)-allowed}")
        if spec['id'] in seen:
            raise ValueError('候选 id 重复')
        seen.add(spec['id'])
        if not spec.get('factors'):
            raise ValueError('候选没有因子')
        for name,weight in spec['factors'].items():
            if name not in library.SIGNED_PRIORS or not np.isfinite(weight) or weight==0:
                raise ValueError(f'无效因子/权重: {name}:{weight}')
        if spec.get('timing','z') not in ('z','quantile') or spec.get('combine','mean') not in ('mean','agree','filter'):
            raise ValueError('无效信号/组合方法')
        days,phase = spec.get('days',1),spec.get('phase',0)
        if not isinstance(days,int) or days<1 or not isinstance(phase,int) or not 0<=phase<days:
            raise ValueError('无效调仓周期/相位')
        if spec.get('mode','single') not in ('single','staggered'):
            raise ValueError('无效调仓模式')
        if spec.get('mode','single') == 'staggered' and phase != 0:
            raise ValueError('错开持有已经平均全部相位，不接受额外相位选择')
        if not 0 < spec.get('cap',1.) <= 1. or not np.isfinite(spec.get('vol_target',.20)) or spec.get('vol_target',.20)<0:
            raise ValueError('本研究只支持最大一倍名义暴露，非负波动目标')


def signal_for(data,spec):
    frames,weights = [],[]
    for name,weight in spec['factors'].items():
        # JSON weights multiply the already economically directed factor.
        direction = library.SIGNED_PRIORS[name]*np.sign(weight)
        frames.append(library.to_signal(name,data.factors.raw(name),spec.get('timing','z'),direction))
        weights.append(abs(weight))
    return library.combine(frames,spec.get('combine','mean'),weights)


def run_spec(data,spec):
    signal = signal_for(data,spec)
    position = engine.position(signal,data.factors.vol,spec.get('vol_target',.20),
        spec.get('cap',1.),spec.get('days',1),spec.get('mode','single'),spec.get('phase',0))
    return engine.backtest(position.reindex_like(data.returns),data.returns,data.universe,
        data.open_fee,data.close_fee,data.slippage,data.rolls,data.roll_close_fee)


def performance_rows(result,spec):
    periods = [('2016-2021',result)]
    periods += [(str(year),result.loc[result.index.year==year]) for year in range(2016,2022)]
    rows = []
    for label,frame in periods:
        row = {'id':spec['id'],'period':label}
        for kind in ('gross','net'):
            row.update({kind+'_'+k:v for k,v in stats.performance(frame[kind]).items()})
            row[kind+'_sharpe'] = stats.sharpe_ratio(frame[kind])
        row['annual_turnover'] = frame.turnover.sum()*252/len(frame) if len(frame) else np.nan
        row['annual_cost'] = frame.cost.sum()*252/len(frame) if len(frame) else np.nan
        rows.append(row)
    return rows


def factor_diagnostics(data,horizons=(1,3,5,10)):
    """Raw factor IC only: Spearman and normalized predictive score are labelled separately.

    No second z-scoring of a composite. Same monthly Newey-West sampling for
    each horizon; p-values are exploratory, not multiple-testing-adjusted proof.
    """
    rows = []
    historical = (data.factors.bars['openw'].shift(-1)-data.factors.bars['openw'])/data.factors.bars['open']
    for name in data.factors.signed:
        raw = data.factors.raw(name)
        for h in horizons:
            forward = B.holding_forward_return(historical,h)
            function = stats.cross_sectional_ic_table if name.startswith('cs_') else stats.factor_ic_table
            table = function(raw,forward,data.universe,name,list(range(2016,2022)))
            table = table.rename(columns={'ic':'spearman_ic','ic_ts':'predictive_score','t':'score_nw_t'})
            table.insert(0,'horizon',h)
            rows.extend(table.to_dict('records'))
    return pd.DataFrame(rows)


def provenance(specs,n_ticks):
    files = sorted((C.PROJECT_ROOT/'src').rglob('*.py'))
    files += sorted((C.PROJECT_ROOT/'scripts').glob('*.py'))
    hashes = {str(p.relative_to(C.PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source_hashes = {}
    for p in [C.UNIVERSE_DIR/'universe_by_year.json',C.RESEARCH_OUT_DIR/'fee_history.csv',
              C.RESEARCH_OUT_DIR/'tick_size.csv']:
        source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    for directory in (cache.timestamp_dir(),cache.combo_dir(C.IC_REFERENCE_LOOKBACK,C.IC_REFERENCE_PCT),
                      C.FACTOR_DAILY_DIR/'external'/'research'):
        for p in sorted(directory.glob('*.parquet')) + sorted(directory.glob('*.pkl')):
            source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {'status':'research_only_unfrozen','candidates':specs,'slippage_ticks_per_side':n_ticks,
        'cache_version':cache.CACHE_VERSION,'code_sha256':hashes,'input_sha256':source_hashes,
        'window':'2016-2021','execution':'close signal, next trading-date open, open-to-open returns',
        'roll_fee':'old-contract close rate approximated by preceding dominant rate',
        'tick_source':'previous-year estimates, current raw open denominator',
        'phase_checks':'all single-book phases, reported separately; none automatically selected',
        'missing_factor':'fixed weight left in cash','validation_2022':'not_run','oos_2023_2025':'locked'}
