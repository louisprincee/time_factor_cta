"""Four pre-registered morning mechanisms; never load 2023-2025.

Research candidates are selected before the 2022 shard is read. All outcomes
are reported, including failed hypotheses. This is continuous-notional research.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from tfcta import config as C
from tfcta.data import shard_io
from tfcta.factors.intraday import raw_price_path, day_codes_of
from tfcta.research import costs, stats
import plot_morning_rule as morning
import select_main_book as book

PROTOCOL = ROOT / 'config/robust_morning_20261005.json'
OUT = ROOT / 'runs/robustness_20261005'
GROUPS = {
    'agriculture': 'A B C CS M Y P JD AP CF SR RM OI'.split(),
    'black': 'I J JM RB HC SF SM ZC'.split(),
    'metals': 'CU AL ZN PB NI SN'.split(),
    'precious': 'AU AG'.split(),
    'energy': 'BU FU SC'.split(),
    'chemicals': 'L V PP EG RU SP TA FG MA'.split(),
}


def add_history(frame):
    frame = frame.sort_values(['symbol', 'trading_date']).reset_index(drop=True).copy()
    if frame.duplicated(['symbol', 'trading_date']).any():
        raise ValueError('重复品种/交易日')
    if pd.to_datetime(frame.trading_date).max() >= pd.Timestamp('2023-01-01'):
        raise C.HoldoutViolation('候选历史混入封存年份')
    for _, indices in frame.groupby('symbol', sort=False).groups.items():
        block = frame.loc[indices]
        dev = block.dev16.abs().shift(1).rolling(60, min_periods=40)
        frame.loc[indices, 'q80'] = dev.quantile(.8).to_numpy()
        frame.loc[indices, 'q90'] = dev.quantile(.9).to_numpy()
        frame.loc[indices, 'median_range'] = block.range15.shift(1).rolling(60, min_periods=40).median().to_numpy()
        frame.loc[indices, 'median_volume'] = block.volume20.shift(1).rolling(60, min_periods=40).median().to_numpy()
        frame.loc[indices, 'prior_sigma'] = block.gross17.shift(1).rolling(60, min_periods=30).std().to_numpy() * np.sqrt(252)
    return frame


def make_signals(frame):
    f = frame
    dev = f.dev16.to_numpy(float)
    side = -np.sign(dev)
    base = (np.isfinite(dev) & f.estimated16.lt(.0006) & f.estimated16.ge(0)).to_numpy()
    margin = np.abs(dev) > 3 * f.estimated16.to_numpy(float)
    q80 = base & margin & (np.abs(dev) > f.q80.to_numpy(float))
    q90 = base & margin & (np.abs(dev) > f.q90.to_numpy(float))
    clock = ((dev > 0) & f.high_clock.ge(2/3).to_numpy()) | ((dev < 0) & f.low_clock.ge(2/3).to_numpy())
    old_clock = ((dev > 0) & f.whole_clock.ge(2/3).to_numpy()) | ((dev < 0) & f.whole_clock.le(1/3).to_numpy())
    confirm = (((dev > 0) & f.close20.lt(f.close16).to_numpy() & f.high_after.le(f.high16).to_numpy())
        | ((dev < 0) & f.close20.gt(f.close16).to_numpy() & f.low_after.ge(f.low16).to_numpy()))
    confirm &= (f.estimated20.lt(.0006) & f.estimated20.ge(0)).to_numpy()
    confirm &= np.abs(dev) > 3 * f.estimated20.to_numpy(float)
    def reverse(mask):
        return np.where(mask, side, 0.)
    result = {
        'R_original_fixedrisk': reverse(base & (np.abs(dev) > .002) & old_clock),
        'R_fixeddev_no_clock': reverse(base & (np.abs(dev) > .002)),
        'R_std80_no_clock': reverse(q80),
        'A_std80_amclock': reverse(q80 & clock),
        'A_std90_amclock': reverse(q90 & clock),
        'B_er40_clock': reverse(q80 & clock & f.er.le(.4).to_numpy()),
        'B_er60_clock': reverse(q80 & clock & f.er.le(.6).to_numpy()),
        'B_er40_noclock': reverse(q80 & f.er.le(.4).to_numpy()),
        'B_er60_noclock': reverse(q80 & f.er.le(.6).to_numpy()),
        'C_confirm80': reverse(q80 & clock & confirm),
        'C_confirm90': reverse(q90 & clock & confirm),
        'C_confirm80_noclock': reverse(q80 & confirm),
    }
    long = f.close20.gt(f.high15 + f.tick).to_numpy()
    short = f.close20.lt(f.low15 - f.tick).to_numpy()
    orb = np.where(long, 1., np.where(short, -1., 0.))
    allowed = (f.estimated20.lt(.0006) & f.estimated20.ge(0)
        & f.range15.ge(3 * f.estimated20)).to_numpy()
    compressed = f.range15.le(f.median_range).to_numpy()
    volume = f.volume20.ge(1.5 * f.median_volume).to_numpy()
    result.update(D_orb_control=np.where(allowed, orb, 0.),
        D_orb_compression=np.where(allowed & compressed, orb, 0.),
        D_orb_compression_volume=np.where(allowed & compressed & volume, orb, 0.))
    return result


def risk_weights(signal, sigma, universe):
    if not signal.index.equals(sigma.index) or not signal.columns.equals(sigma.columns):
        raise ValueError('风险与信号没有对齐')
    if not np.isfinite(signal.to_numpy()).all():
        raise ValueError('信号必须有限')
    multiplier = (.10 / sigma.clip(lower=.05)).clip(upper=2.)
    multiplier = multiplier.where(np.isfinite(sigma) & sigma.ge(0), 0.)
    weight = signal * 0.
    for year in sorted(set(signal.index.year)):
        members = [s for s in universe.get(int(year), []) if s in signal.columns]
        if not members:
            continue
        rows = signal.index.year == year
        weight.loc[rows, members] = (signal.loc[rows, members] * multiplier.loc[rows, members] / len(members)).clip(-.10, .10)
    mapped = {s for members in GROUPS.values() for s in members}
    if set(signal.columns) - mapped:
        raise ValueError(f'未登记板块: {set(signal.columns) - mapped}')
    for members in GROUPS.values():
        columns = [s for s in members if s in weight.columns]
        if columns:
            total = weight[columns].abs().sum(axis=1)
            weight.loc[:, columns] = weight[columns].mul((.25 / total.replace(0, np.nan)).clip(upper=1).fillna(1), axis=0)
    total = weight.abs().sum(axis=1)
    return weight.mul((.60 / total.replace(0, np.nan)).clip(upper=1).fillna(1), axis=0)


def extract(partition):
    master, universe, symbols, skipped = book.calendar_and_members(partition)
    fees, ticks = book.fees_for(partition), costs.load_ticks()
    rows = []
    columns = ['open', 'close', 'closew', 'highw', 'loww', 'volume', 'trading_date']
    for symbol in symbols:
        loader = shard_io.load_shard if partition == 'research' else shard_io.load_validation_shard
        frame = loader(symbol, columns=columns).sort_index(kind='mergesort')
        if frame.index.has_duplicates:
            raise ValueError(f'{symbol}: 重复分钟')
        # Only the isolated research partition can supply warm-up.
        frame = frame[pd.to_datetime(frame.trading_date) >= pd.Timestamp('2015-01-01')]
        frame['trading_date'] = pd.to_datetime(frame.trading_date).dt.normalize()
        close = raw_price_path(frame)
        op = raw_price_path(pd.DataFrame({'close': frame.open.to_numpy()}, index=frame.index))
        high = np.round(close + frame.highw.to_numpy(float) - frame.closew.to_numpy(float), 2)
        low = np.round(close + frame.loww.to_numpy(float) - frame.closew.to_numpy(float), 2)
        whole_high = frame.highw.to_numpy(float)
        volume = frame.volume.to_numpy(float)
        idx = frame.index.asi8
        codes, days = day_codes_of(frame)
        bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
        positions = {ts: i for i, ts in enumerate(frame.index.to_numpy())}
        fee = fees[fees.symbol.eq(symbol)].set_index('trading_date').sort_index()
        tick = ticks[ticks.symbol.eq(symbol)].set_index('year').tick.sort_index().round(2)
        prior_tick = tick.rename(index=lambda y: int(y)+1)
        prior_tick = prior_tick.reindex(sorted(set(prior_tick.index) | set(range(2015, 2023)))).ffill()
        for start, stop in zip(bounds[:-1], bounds[1:]):
            day = pd.Timestamp(days[int(codes[start])])
            r = {'trading_date': day, 'symbol': symbol}
            p = {m: positions.get(morning.stamp(day, (9, m))) for m in (1, 15, 16, 17, 18, 20, 21, 22)}
            def prefix_ok(m):
                a, b = p[1], p[m]
                return (a is not None and b is not None and start <= a <= b < stop
                    and b-a == m-1 and np.all(np.diff(idx[a:b+1]) == 60_000_000_000)
                    and np.isfinite(close[a:b+1]).all() and (close[a:b+1] > 0).all())
            good16, good20 = prefix_ok(16), prefix_ok(20)
            for key in ['dev16', 'er', 'high_clock', 'low_clock', 'whole_clock',
                        'close16', 'close20', 'high15', 'low15', 'range15',
                        'high16', 'low16', 'high_after', 'low_after', 'volume20']:
                r[key] = np.nan
            if good16:
                a, b = p[1], p[16]
                r['dev16'] = close[b] / close[a:b].mean() - 1
                r['close16'] = close[b]
                path = np.r_[op[a], close[a:b+1]]
                travel = np.abs(np.diff(path)).sum()
                r['er'] = abs(path[-1]-path[0])/travel if travel > 0 else 0.
                if np.isfinite(high[a:b+1]).all() and np.isfinite(low[a:b+1]).all():
                    r['high_clock'] = np.argmax(high[a:b+1]) / 15
                    r['low_clock'] = np.argmin(low[a:b+1]) / 15
                    r['high16'], r['low16'] = high[a:b+1].max(), low[a:b+1].min()
                    r['high15'], r['low15'] = high[a:b].max(), low[a:b].min()
                    r['range15'] = (r['high15']-r['low15'])/close[b]
                r['whole_clock'] = morning.clock_of(whole_high, start, b+1)
            if good20:
                r['close20'] = close[p[20]]
                r['high_after'] = high[p[16]+1:p[20]+1].max()
                r['low_after'] = low[p[16]+1:p[20]+1].min()
                v = volume[p[1]:p[20]+1]
                if np.isfinite(v).all() and (v >= 0).all():
                    r['volume20'] = v.sum()
            t = float(prior_tick.loc[day.year]) if day.year in prior_tick.index else np.nan
            r['tick'] = t
            fr = fee.reindex([day], method='ffill').iloc[0] if not fee.empty else None
            def fee_at(ref, entry, exit_price):
                if fr is None or not np.isfinite([ref, entry, exit_price, t]).all() or min(ref, entry, exit_price, t) <= 0:
                    return np.nan
                return (morning.one_rate(fr.commission_type, fr.open_commission, entry, ref, symbol)
                    + morning.one_rate(fr.commission_type, fr.close_commission_today, exit_price, ref, symbol) + 2*t/ref)
            r['estimated16'] = fee_at(r['close16'], r['close16'], r['close16'])
            r['estimated20'] = fee_at(r['close20'], r['close20'], r['close20'])
            exit_bar = positions.get(morning.stamp(day, (11, 30)))
            # Store outcomes, never consult them in make_signals.
            for m, decision_m in [(17,16), (18,16), (21,20), (22,20)]:
                q = morning.execution_prices(op, close, p[decision_m] if p[decision_m] is not None else -1,
                    p[m], exit_bar, start, stop, False)
                r[f'gross{m}'] = np.nan if q is None else q[1]/q[0]-1
                r[f'cost{m}'] = np.nan if q is None else fee_at(q[0], q[0], q[1])
            rows.append(r)
        print(f'extracted {partition} {symbol}', flush=True)
    frame = pd.DataFrame(rows)
    return frame, master, universe, skipped


def metric(daily):
    p = stats.performance(daily)
    return dict(n_days=len(daily), ann_return=p['ann_return'], sharpe=stats.sharpe_ratio(daily),
        max_drawdown=p['max_drawdown'], ann_vol=p['ann_vol'], cumulative=float((1+daily).prod()-1))


def evaluate(features, master, universe):
    all_signal = make_signals(features)
    sigma = features.pivot(index='trading_date', columns='symbol', values='prior_sigma').reindex(master)
    daily, stress, delayed, ntrades, exposures, attribution = {}, {}, {}, {}, {}, {}
    for name, values in all_signal.items():
        s = features[['trading_date','symbol']].assign(signal=values).pivot(index='trading_date', columns='symbol', values='signal').reindex(master).fillna(0.)
        weight = risk_weights(s, sigma, universe)
        entry = 21 if name[0] in ('C','D') else 17
        def panel(field):
            return features.pivot(index='trading_date', columns='symbol', values=field).reindex(index=master, columns=weight.columns).to_numpy(float)
        w, g, c = weight.to_numpy(float), panel(f'gross{entry}'), panel(f'cost{entry}')
        daily[name] = morning.daily_pnl(w, g, c)
        stress[name] = morning.daily_pnl(w, g, c*1.5)
        delayed[name] = morning.daily_pnl(w, panel(f'gross{entry+1}'), panel(f'cost{entry+1}'))
        ntrades[name] = (w != 0).sum(axis=1)
        exposures[name] = np.abs(w).sum(axis=1)
        active = w != 0
        contrib = np.zeros_like(w)
        contrib[active] = w[active]*g[active] - np.abs(w[active])*c[active]
        attribution[name] = pd.DataFrame(contrib, index=master, columns=weight.columns)
    return tuple(pd.DataFrame(x, index=master) for x in (daily, stress, delayed, ntrades, exposures)) + (attribution,)


def walkforward(daily, stress, delayed, trades):
    paths, paths_stress, paths_delay, paths_trades, choices = {}, {}, {}, {}, []
    for family in 'ABCD':
        candidates = [n for n in daily if n.startswith(family+'_')]
        pieces, pieces_s, pieces_d, pieces_n = [], [], [], []
        for year in (2019,2020,2021):
            train = (daily.index.year >= 2016) & (daily.index.year < year)
            test = daily.index.year == year
            scored = [(float(stats.sharpe_ratio(stress.loc[train,n])), n) for n in candidates if trades.loc[train,n].sum() >= 30]
            scored = [(score,n) for score,n in scored if np.isfinite(score) and score > 0]
            winner = sorted(scored, key=lambda x: (-x[0], x[1]))[0][1] if scored else None
            choices.append(dict(family=family, test_year=year, train_end=year-1, candidate=winner))
            for source, dest in [(daily,pieces),(stress,pieces_s),(delayed,pieces_d),(trades,pieces_n)]:
                dest.append(source.loc[test,winner] if winner else pd.Series(0., index=daily.index[test]))
        paths[family], paths_stress[family], paths_delay[family], paths_trades[family] = [pd.concat(p) for p in (pieces,pieces_s,pieces_d,pieces_n)]
    paths['CD_equal'] = .5*paths['C'] + .5*paths['D']
    paths_stress['CD_equal'] = .5*paths_stress['C'] + .5*paths_stress['D']
    paths_delay['CD_equal'] = .5*paths_delay['C'] + .5*paths_delay['D']
    # Number of sleeve fills; overlapping fills are conservatively charged twice.
    paths_trades['CD_equal'] = paths_trades['C'] + paths_trades['D']
    return [pd.DataFrame(p) for p in (paths, paths_stress, paths_delay, paths_trades)], choices


def summarize(evaluation, period):
    daily, stress, delayed, trades, exposure, _ = evaluation
    rows = []
    for name in daily:
        periods = [(period, np.ones(len(daily), dtype=bool))]
        if period == 'research':
            periods += [(str(y), daily.index.year == y) for y in range(2016,2022)]
        for label, mask in periods:
            rows.append(dict(candidate=name, period=label, **metric(daily.loc[mask,name]),
                stress_sharpe=stats.sharpe_ratio(stress.loc[mask,name]), delay_sharpe=stats.sharpe_ratio(delayed.loc[mask,name]),
                trades=int(trades.loc[mask,name].sum()), mean_exposure=float(exposure.loc[mask,name].mean()),
                cost_arithmetic=float(2*(daily.loc[mask,name]-stress.loc[mask,name]).sum()),
                gross_arithmetic=float((3*daily.loc[mask,name]-2*stress.loc[mask,name]).sum())))
    return pd.DataFrame(rows)


def bootstrap(paths):
    rng = np.random.default_rng(20261005)
    n, block = len(paths), 20
    starts = rng.integers(0, n, size=(3000, int(np.ceil(n/block))))
    indices = ((starts[:,:,None] + np.arange(block)) % n).reshape(3000,-1)[:,:n]
    out = {}
    for name in paths:
        samples = paths[name].to_numpy()[indices]
        means, sd = samples.mean(axis=1), samples.std(axis=1, ddof=1)
        sr = np.divide(means*np.sqrt(252),sd,out=np.full(len(sd),np.nan),where=sd>0)
        out[name] = dict(arithmetic_annual_mean_interval=np.quantile(means*252,[.025,.975]).tolist(),
            sharpe_interval=np.nanquantile(sr,[.025,.975]).tolist() if np.isfinite(sr).any() else [None,None])
    return out


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-features', action='store_true', help='Reuse only this registered experiment\'s allowed-year cache')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    protocol_copy = OUT/'registered_protocol.json'
    if protocol_copy.exists() and json.loads(protocol_copy.read_text()) != protocol:
        raise ValueError('禁止覆盖已经登记的协议')
    write_json(protocol_copy, protocol)
    def get(partition):
        path = OUT/f'{partition}_features.parquet'
        if args.reuse_features and path.exists():
            f = pd.read_parquet(path)
            if partition == 'research': C.assert_no_holdout_dates(f.trading_date)
            else: C.assert_validation_2022_dates(f.trading_date)
            master, universe, _, skipped = book.calendar_and_members(partition)
        else:
            f, master, universe, skipped = extract(partition)
            f.to_parquet(path, index=False)
        return f, master, universe, skipped
    raw, master, universe, skipped = get('research')
    f = add_history(raw)
    if set(make_signals(f)) != set(protocol['candidates']):
        raise ValueError('候选与登记清单不一致')
    research = evaluate(f, master, universe)
    summary = summarize(research, 'research')
    summary.to_csv(OUT/'candidate_metrics_research.csv', index=False)
    research[0].to_csv(OUT/'candidate_daily_research.csv')
    research[1].to_csv(OUT/'candidate_stress_research.csv')
    research[2].to_csv(OUT/'candidate_delay_research.csv')
    wf, choices = walkforward(*research[:4])
    wf_metrics = []
    for family in wf[0]:
        row = dict(family=family, **metric(wf[0][family]), stress_sharpe=stats.sharpe_ratio(wf[1][family]),
            delay_sharpe=stats.sharpe_ratio(wf[2][family]), trades=int(wf[3][family].sum()))
        row['admitted'] = bool(row['sharpe'] >= 1.5 and row['stress_sharpe'] >= 1 and row['delay_sharpe'] > 0 and row['trades'] >= 100)
        wf_metrics.append(row)
    pd.DataFrame(wf_metrics).to_csv(OUT/'walkforward_metrics.csv', index=False)
    for i, label in enumerate(['daily','stress','delay','trades']):
        wf[i].to_csv(OUT/f'walkforward_{label}.csv')
    # Fix the 2022 variant selection before reading its feature shard.
    selected = {}
    for family in 'ABCD':
        names = [n for n in research[0] if n.startswith(family+'_')]
        eligible = [(float(stats.sharpe_ratio(research[1][n])), n) for n in names if research[3][n].sum() >= 30]
        eligible = [(s,n) for s,n in eligible if np.isfinite(s) and s > 0]
        selected[family] = sorted(eligible, key=lambda x:(-x[0],x[1]))[0][1] if eligible else None
    write_json(OUT/'research_selection_before_2022.json', dict(choices=choices, selected_for_2022_diagnostic=selected,
        admitted_families=[r['family'] for r in wf_metrics if r['admitted']], frozen_strategy=None))
    write_json(OUT/'walkforward_bootstrap.json', bootstrap(wf[0]))
    valraw, valmaster, valuniverse, valskipped = get('validation_2022')
    combined = add_history(pd.concat([raw, valraw], ignore_index=True))
    validation = evaluate(combined, valmaster, valuniverse)
    summarize(validation, '2022_diagnostic').to_csv(OUT/'candidate_metrics_2022.csv', index=False)
    validation[0].to_csv(OUT/'candidate_daily_2022.csv')
    # Evaluate only the research-selected sleeve choices, never optimize here.
    selpaths = {family: validation[0][name] if name else pd.Series(0., index=valmaster) for family,name in selected.items()}
    selpaths['CD_equal'] = .5*selpaths['C'] + .5*selpaths['D']
    pd.DataFrame(selpaths).to_csv(OUT/'selected_diagnostic_2022.csv')
    for partition, evaluation in [('research',research),('2022',validation)]:
        contributions = []
        removal = []
        for name, matrix in evaluation[-1].items():
            for symbol, value in matrix.sum().items():
                contributions.append(dict(candidate=name, symbol=symbol, sector=next(k for k,v in GROUPS.items() if symbol in v), arithmetic_net_contribution=value))
            # Sensitivity of the existing weighted book; never redistribute or select.
            for sector, members in GROUPS.items():
                remaining = matrix.drop(columns=[s for s in members if s in matrix]).sum(axis=1)
                removal.append(dict(candidate=name, excluded_sector=sector, **metric(remaining)))
        pd.DataFrame(contributions).to_csv(OUT/f'symbol_contributions_{partition}.csv', index=False)
        pd.DataFrame(removal).to_csv(OUT/f'sector_removal_diagnostic_{partition}.csv', index=False)
    paths = [PROTOCOL, Path(__file__), ROOT/'scripts/plot_morning_rule.py', ROOT/'scripts/select_main_book.py',
        ROOT/'src/tfcta/data/shard_io.py', ROOT/'src/tfcta/config.py'] + list(OUT.glob('*.parquet'))
    write_json(OUT/'experiment_provenance.json', dict(code_and_features_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        skipped_research=skipped, skipped_2022=valskipped, oos_read=False,
        limitations=['continuous notional, no integer lots/margin/limits/impact', 'historical tick inferred; raw intrabar highs/lows reconstructed from adjusted offsets and rounded to 0.01', 'bootstrap intervals unadjusted for multiple trials', 'known development and diagnostic history; no forward performance claim']))
    print(pd.DataFrame(wf_metrics).to_string(index=False), flush=True)
    print('Completed all registered candidates. No OOS read; no strategy frozen.', flush=True)


if __name__ == '__main__':
    main()
