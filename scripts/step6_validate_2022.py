"""按冻结方案评估 2022；2023+ 严格锁定，不读取、不生成任何统计。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tfcta import config as C  # noqa: E402
from tfcta.data import shard_io, universe as U  # noqa: E402
from tfcta.factors import slow, tech  # noqa: E402
from tfcta.factors.factors import symbol_daily_factors  # noqa: E402
from tfcta.data import sessions  # noqa: E402
from tfcta.research import book, costs, execution, panel, stats  # noqa: E402

MINUTE_COLUMNS = list(dict.fromkeys([
    *C.FACTOR_FIELDS, *C.PRICE_FIELDS,
]))
DAILY_AGG = {
    'open': 'first', 'openw': 'first', 'close': 'last', 'closew': 'last',
    'highw': 'max', 'loww': 'min', 'volume': 'sum',
}


def load_plan() -> dict:
    path = C.CONFIG_DIR / 'validation_2022_plan.json'
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan.get('status') != 'frozen' or plan.get('validation_year') != 2022:
        raise ValueError(f'验证方案未冻结或年份不正确: {path}')
    if plan.get('strict_oos_start') != '2023-01-01':
        raise ValueError('严格 OOS 边界必须保持为 2023-01-01')
    return plan


def assert_validation_unused(result_dir: Path) -> None:
    outputs = [Path(result_dir) / 'performance.csv', Path(result_dir) / 'ic_2022.csv']
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            '2022 一次性验证已有结果，拒绝重跑: ' + ', '.join(existing))


def load_2022_universe(plan: dict) -> tuple[dict[int, list[str]], pd.DataFrame]:
    stats_path = C.PROJECT_ROOT / plan['universe']['source']
    yearly = pd.read_csv(stats_path)
    cfg = plan['universe']
    universe, detail = U.build_universe(
        yearly,
        years=[2022],
        lookback_years=int(cfg['lookback_years']),
        min_turnover=float(cfg['min_daily_turnover']),
        min_valid_ratio=float(cfg['min_valid_day_ratio']),
        min_valid_days=int(cfg['min_valid_days']),
    )
    return universe, detail


def load_history(symbols: list[str], lookback: int,
                 pct: float) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    factor_by_symbol: dict[str, pd.DataFrame] = {}
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        research = shard_io.load_shard(
            symbol, directory=C.RESEARCH_DIR, columns=MINUTE_COLUMNS)
        validation = shard_io.load_validation_shard(symbol, columns=MINUTE_COLUMNS)
        minute = pd.concat([research, validation]).sort_index(kind='mergesort')
        if minute.index.has_duplicates:
            raise ValueError(f'{symbol} research 与 2022 分片有重复时间戳')
        C.assert_no_holdout_dates(research['trading_date'], what=f'{symbol} 研究期')
        C.assert_validation_2022_dates(validation['trading_date'],
                                       what=f'{symbol} 验证期')

        coords = sessions.add_intraday_coords(minute)
        factor_by_symbol[symbol] = symbol_daily_factors(
            coords, lookback=lookback, pct=pct, with_coords=True)
        td = pd.to_datetime(minute['trading_date']).dt.normalize()
        bars_by_symbol[symbol] = minute.groupby(td, sort=True).agg(DAILY_AGG)
        del research, validation, minute, coords
    return factor_by_symbol, bars_by_symbol


def wide_by_field(bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    fields = next(iter(bars_by_symbol.values())).columns
    return {
        field: pd.DataFrame({s: frame[field] for s, frame in bars_by_symbol.items()})
        .sort_index()
        for field in fields
    }


def build_signals(plan: dict,
                  factor_by_symbol: dict[str, pd.DataFrame],
                  bars_by_symbol: dict[str, pd.DataFrame]) -> tuple[
                      dict[str, pd.DataFrame], pd.DataFrame]:
    fields = next(iter(factor_by_symbol.values())).columns
    raw = {
        name: pd.DataFrame({s: frame[name] for s, frame in factor_by_symbol.items()})
        .sort_index()
        for name in fields
    }
    bars = wide_by_field(bars_by_symbol)
    prices = panel.multiplicative_prices(bars)
    raw.update(tech.build(prices['close'], prices['high'], prices['low'], bars['volume']))
    raw['tsmom'] = slow.tsmom(bars['close'], bars['closew'])
    raw['carry'] = slow.carry(bars['close'], bars['closew'])
    raw['neg_ret_day'] = -((bars['closew'] - bars['openw']) /
                           bars['open'].where(bars['open'] > 0))

    normal = plan['normalization']
    window, minimum = int(normal['window']), int(normal['min_periods'])
    strategies = {}
    for strategy, members in plan['strategies'].items():
        standardized = []
        for name, sign in members.items():
            if name not in raw:
                raise KeyError(f'验证策略引用了不可用因子: {name}')
            standardized.append(stats.exante_z(raw[name] * float(sign), window, minimum))
        strategies[strategy] = book.average_signals(standardized)

    returns = {}
    for symbol, frame in bars_by_symbol.items():
        px = frame[['open', 'openw']]
        returns[symbol] = panel.day_return_from_prices(px)
    day_ret = pd.DataFrame(returns).sort_index()
    return strategies, day_ret


def evaluate(plan: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    universe_2022, screen = load_2022_universe(plan)
    available = set(shard_io.list_shards(C.RESEARCH_DIR)) & set(
        shard_io.list_shards(C.VALIDATION_DIR))
    symbols = sorted(set(universe_2022.get(2022, [])) & available)
    if not symbols:
        raise ValueError('2022 时点品种池与研究/验证分片没有交集')
    universe_2022 = {2022: symbols}

    factor_by_symbol, bars_by_symbol = load_history(
        symbols,
        int(plan['duration']['lookback']),
        float(plan['duration']['percentile']),
    )
    strategy_signals, day_ret = build_signals(
        plan, factor_by_symbol, bars_by_symbol)
    validation_index = day_ret.index[day_ret.index.year == 2022]
    C.assert_validation_2022_dates(validation_index, what='验证收益索引')
    forward = panel.forward_return(day_ret)

    old_universe = U.load_universe()
    full_universe = {**old_universe, **universe_2022}
    tick_table = costs.load_tick_table(symbols)
    slippage = costs.slippage_wide(
        tick_table, day_ret.index, symbols, float(plan['slippage_ticks']))

    perf_rows, ic_rows = [], []
    for name, signal in strategy_signals.items():
        weekly_signal = execution.weekly(signal.clip(-1, 1))
        position = panel.execute_position(weekly_signal)
        gross = book.run_book(position, day_ret, full_universe, 0.0)
        net = book.run_book(position, day_ret, full_universe,
                            float(plan['fee_rate']), slippage=slippage)
        gross_2022 = gross.loc[gross.index.year == 2022]
        net_2022 = net.loc[net.index.year == 2022]
        C.assert_validation_2022_dates(gross_2022.index, what='验证组合收益')
        row = {'strategy': name, 'symbols': len(symbols),
               'turnover': stats.annual_turnover(position, full_universe, [2022])}
        row.update({f'gross_{k}': v for k, v in stats.performance(gross_2022).items()})
        row.update({f'net_{k}': v for k, v in stats.performance(net_2022).items()})
        perf_rows.append(row)

        tab = stats.factor_ic_table(signal, forward, universe_2022,
                                    name, [2022])
        ic_rows.append(tab[tab['fold'] == '2022'])

    return (pd.DataFrame(perf_rows), pd.concat(ic_rows, ignore_index=True),
            {'symbols': symbols, 'screen': screen, 'tick_table': tick_table})


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    plan = load_plan()

    if not shard_io.list_shards(C.VALIDATION_DIR):
        print(f"缺少隔离后的 2022 验证分片: {C.VALIDATION_DIR}")
        return 2

    result_dir = C.DATA_ROOT / 'validation_2022'
    assert_validation_unused(result_dir)
    perf, ic_table, context = evaluate(plan)
    run = C.RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_step6_validation2022"
    run.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    perf.to_csv(run / 'performance.csv', index=False, encoding='utf-8-sig')
    ic_table.to_csv(run / 'ic_2022.csv', index=False, encoding='utf-8-sig')
    perf.to_csv(result_dir / 'performance.csv', index=False, encoding='utf-8-sig')
    ic_table.to_csv(result_dir / 'ic_2022.csv', index=False, encoding='utf-8-sig')
    context['screen'].to_csv(run / 'universe_2022_screen.csv', encoding='utf-8-sig')
    context['tick_table'].to_csv(run / 'research_tick_table.csv', index=False,
                                 encoding='utf-8-sig')
    (run / 'params.json').write_text(
        json.dumps({'plan': plan, 'symbols': context['symbols']},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    print(perf.to_string(index=False))
    print(f"\n2022 验证结果: {result_dir}")
    print(f"运行快照: {run}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())