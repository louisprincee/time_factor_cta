"""One explicitly configured intraday ML experiment on 2016–2021 only."""
import argparse
import hashlib
import math
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from tfcta import config as C
from tfcta.data import shard_io, universe as U
from tfcta.research.intraday import features, ml
from tfcta.research.backtest import costs
from tfcta.research.analysis import stats
from tfcta.research.workflow import context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ridge','hgb'], default='ridge')
    parser.add_argument('--features', choices=['time','price','all'], default='all')
    parser.add_argument('--decision-minutes', type=int, default=30)
    parser.add_argument('--hold-minutes', type=int, default=30)
    parser.add_argument('--threshold-lookback', type=int, default=250)
    parser.add_argument('--threshold-pct', type=float, default=55.)
    parser.add_argument('--min-train', type=int, default=1000)
    parser.add_argument('--slippage-ticks', type=float, default=1.)
    parser.add_argument('--symbols', nargs='+', help='明确缩小研究品种池，现金分配按缩小后的年度池计算')
    args = parser.parse_args()
    if not math.isfinite(args.slippage_ticks) or args.slippage_ticks < 0:
        parser.error('滑点必须非负且有限')
    if args.min_train < 1:
        parser.error('min-train 必须为正数')
    universe = {y:ss for y,ss in U.load_universe().items() if 2016<=y<=2021}
    available = sorted({s for ss in universe.values() for s in ss})
    if args.symbols:
        if set(args.symbols)-set(available):
            parser.error('指定品种必须在 2016–2021 年度研究池内')
        universe = {y:[s for s in ss if s in args.symbols] for y,ss in universe.items()}
    symbols = sorted({s for ss in universe.values() for s in ss})
    frames, calendar = [], set()
    input_paths = [C.UNIVERSE_DIR/'universe_by_year.json', C.RESEARCH_OUT_DIR/'fee_history.csv',
                   C.RESEARCH_OUT_DIR/'tick_size.csv']
    for symbol in symbols:
        print(f'Building decision-time features: {symbol}', flush=True)
        minute = shard_io.load_shard(symbol, columns=features.MINUTE_COLUMNS)
        eligible = [y for y,ss in universe.items() if symbol in ss]
        dates = pd.DatetimeIndex(minute.trading_date.unique())
        calendar.update(dates[dates.year.isin(eligible)])
        frames.append(features.samples_from_minutes(symbol, minute, args.decision_minutes,
            args.hold_minutes, args.threshold_lookback, args.threshold_pct, eligible))
        input_paths.append(shard_io.find_shard(C.RESEARCH_DIR, symbol))
    if not frames or not any(len(f) for f in frames):
        parser.error('没有可训练的研究样本')
    samples = pd.concat(frames, ignore_index=True).sort_values(['decision_time','symbol']).reset_index(drop=True)
    fee = costs.intraday_costs(samples, costs.load_fees(), costs.load_ticks(), args.slippage_ticks)
    samples = samples.join(fee)
    names = {'time':features.TIME_FEATURES, 'price':features.PRICE_FEATURES, 'all':features.FEATURES}[args.features]
    predictions, folds = ml.walk_forward(samples, names, args.model, args.min_train)
    if folds.empty or not folds.fitted.any():
        parser.error('此前年份训练样本不足；没有拟合任何模型')
    trades, daily = ml.evaluate(predictions, universe, calendar)
    run = context.run_dir('intraday_ml')
    code_paths = list((C.PROJECT_ROOT/'src').rglob('*.py')) + list((C.PROJECT_ROOT/'scripts').glob('*.py'))
    hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    context.dump_json(run/'definition.json', dict(status='research_only_unfrozen',
        arguments=vars(args), feature_names=names, universe=universe,
        code_sha256={str(p.relative_to(C.PROJECT_ROOT)):hash_file(p) for p in code_paths},
        input_sha256={str(p):hash_file(p) for p in input_paths},
        window='2016 train; 2017–2021 expanding annual research folds',
        execution='bar-end timestamps; next minute open; scheduled wall-clock close; same-day commission',
        cost='historical commissions plus specified ticks PER SIDE; tick estimates from preceding years',
        validation_2022='not_run_this_experiment; historically used', oos_2023_2025='locked'))
    folds.to_csv(run/'folds.csv', index=False)
    trades.to_csv(run/'predictions.csv', index=False)
    daily.to_csv(run/'daily.csv')
    metrics = []
    for label, part in [('2017-2021',daily.loc[daily.index.year>=2017])] + [
            (str(y),daily.loc[daily.index.year==y]) for y in range(2017,2022)]:
        row = {'period':label, 'days':len(part)}
        for kind in ('gross','net'):
            row.update({kind+'_'+k:v for k,v in stats.performance(part[kind]).items()})
            row[kind+'_sharpe'] = stats.sharpe_ratio(part[kind])
        row['annual_cost'] = part.cost.mean()*252
        row['annual_turnover'] = part.turnover.mean()*252
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(run/'performance.csv', index=False)
    print(f'Research complete: {run}; {len(samples)} samples; {int(folds.fitted.sum())} fitted folds', flush=True)


if __name__=='__main__':
    main()
