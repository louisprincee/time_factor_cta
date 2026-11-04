from __future__ import annotations
# 合并补充经济机制检验；每项子研究仍使用独立协议和结果目录。
# L1 流动性与期现基差。
import json as liquidity__json
import warnings as liquidity__warnings
from pathlib import Path as liquidity__Path
import numpy as liquidity__np
import pandas as liquidity__pd
from backtest.engine import newey_west_t as liquidity__newey_west_t
from research.index import index_market_clock as liquidity__I
from research.time_factor_screen import ols_hac as liquidity__ols_hac
from tfcta import config as liquidity__C
liquidity__CONFIG_PATH = liquidity__C.CONFIG_DIR / 'research' / 'index_liquidity_mechanism.json'
liquidity__OUT = liquidity__C.study_dir('index', 'index_liquidity_mechanism')
liquidity__TICK_ROOT = liquidity__C.INDEX_L1_TICKS_DIR
liquidity__SPOT_PATH = liquidity__C.INDEX_SPOT_DIR / 'spot_daily.csv'
liquidity__SYMBOLS = ('IF', 'IH', 'IC')
liquidity__START, liquidity__END = (liquidity__pd.Timestamp('2015-04-16'), liquidity__pd.Timestamp(liquidity__C.RESEARCH_END))
liquidity__SUB_START = liquidity__pd.Timestamp('2016-01-04')
liquidity__LAGS = 5
liquidity__TAIL = liquidity__pd.Timedelta(30, unit='min')
liquidity__MIN_MINUTES = 60
liquidity__IMPACT_WINDOW = 250

def liquidity__load_protocol(path: liquidity__Path=liquidity__CONFIG_PATH) -> dict:
    protocol = liquidity__json.loads(path.read_text(encoding='utf-8'))
    if protocol.get('status') != 'declared_test_not_a_strategy':
        raise ValueError('检验登记必须声明这不是策略')
    if '不读 2022' not in protocol.get('sample', '') or '不读 2026' not in protocol.get('sample', ''):
        raise ValueError('样本口径必须写明不读 2022 以后和 2026')
    return protocol

def liquidity__tick_day_features(ticks: liquidity__pd.DataFrame) -> dict:
    """一个合约日的日盘逐笔（volume 为当日累计）→ 中间价、成交流量的价格冲击、尾盘和开盘深度。"""
    t = ticks.sort_values('datetime')
    ok = (t['a1'] > 0) & (t['b1'] > 0) & (t['a1'] >= t['b1'])
    t = t.assign(mid=((t['a1'] + t['b1']) / 2).where(ok), depth=(t['a1_v'] + t['b1_v']).where(ok))
    t['dvol'] = t['volume'].diff().fillna(t['volume']).clip(lower=0)
    prev_mid = t['mid'].ffill().shift(1)
    direction = liquidity__np.sign(t['last'] - prev_mid)
    tick_rule = liquidity__np.sign(t['last'].diff()).replace(0, liquidity__np.nan).ffill()
    t['sv'] = direction.where(direction != 0, tick_rule).fillna(0.0) * t['dvol']
    quotes = t[t['mid'].notna()]
    out = {'n_ticks': len(t)}
    if len(quotes) < 2:
        return out
    stamp = quotes['datetime']
    first, last = (stamp.iloc[0], stamp.iloc[-1])
    day = liquidity__pd.Timestamp(first).normalize()
    mid_open, mid_close = (float(quotes['mid'].iloc[0]), float(quotes['mid'].iloc[-1]))
    at_1500 = quotes[stamp <= day + liquidity__pd.Timedelta(15, unit='h')]
    in_first30 = quotes[stamp <= first + liquidity__TAIL]
    tail = quotes[stamp >= last - liquidity__TAIL]
    head = quotes[stamp <= first + liquidity__TAIL]
    out.update({'mid_open': mid_open, 'mid_close': mid_close, 'mid_1500': float(at_1500['mid'].iloc[-1]) if len(at_1500) else liquidity__np.nan, 'mid_first30': float(in_first30['mid'].iloc[-1]), 'depth_head': float(head['depth'].mean()), 'depth_tail': float(tail['depth'].mean())})
    minute = t.set_index('datetime').resample('1min')
    sv = minute['sv'].sum()
    mid_end = minute['mid'].last()
    frame = liquidity__pd.DataFrame({'sv': sv, 'mid': mid_end}).dropna(subset=['mid'])
    frame['dm'] = liquidity__np.log(frame['mid']).diff()
    frame = frame.dropna(subset=['dm'])
    if len(frame) < liquidity__MIN_MINUTES or frame['sv'].std() == 0:
        return out
    lam = float(liquidity__np.polyfit(frame['sv'].to_numpy(), frame['dm'].to_numpy(), 1)[0])
    tail_minutes = frame.index >= (last - liquidity__TAIL).floor('min')
    total = float(liquidity__np.log(mid_close / mid_open))
    flow = lam * float(frame['sv'].sum())
    out.update({'lambda': lam, 'flow': flow, 'flow_tail': lam * float(frame.loc[tail_minutes, 'sv'].sum()), 'resid': total - flow, 'intraday_mid': total, 'impact': lam * float(frame['sv'].abs().median())})
    return out

def liquidity__build_tick_features(symbols=liquidity__SYMBOLS) -> liquidity__pd.DataFrame:
    rows = []
    for symbol in symbols:
        root = liquidity__TICK_ROOT / symbol
        liquidity__C.assert_research_only(root)
        for path in sorted(root.glob('*/*.parquet')):
            frame = liquidity__pd.read_parquet(path, columns=['datetime', 'trading_date', 'last', 'volume', 'a1', 'b1', 'a1_v', 'b1_v', 'contract'])
            frame['trading_date'] = liquidity__pd.to_datetime(frame['trading_date']).dt.normalize()
            liquidity__C.assert_no_holdout_dates(frame['trading_date'], f'{symbol} L1 逐笔')
            for day, part in frame.groupby('trading_date', sort=True):
                if not liquidity__START <= day <= liquidity__END:
                    continue
                rows.append({'symbol': symbol, 'trading_date': day, 'contract': part['contract'].iloc[-1], **liquidity__tick_day_features(part)})
    out = liquidity__pd.DataFrame(rows)
    if out.duplicated(['symbol', 'trading_date']).any():
        raise ValueError('L1 逐笔特征出现重复的品种-日期')
    return out

def liquidity__market_panel(features: liquidity__pd.DataFrame, spot: liquidity__pd.DataFrame, bars: dict, dates: liquidity__pd.DatetimeIndex) -> liquidity__pd.DataFrame:
    """逐品种量，三品种等权（任一缺失则缺失）。次日量都对齐到 t。"""

    def wide(frame, field):
        return frame.pivot(index='trading_date', columns='symbol', values=field).reindex(index=dates, columns=list(liquidity__SYMBOLS))
    contract = wide(features, 'contract')
    same_next = contract.eq(contract.shift(-1))
    m = {f: wide(features, f) for f in ('mid_open', 'mid_close', 'mid_1500', 'mid_first30', 'depth_head', 'depth_tail', 'flow', 'flow_tail', 'resid', 'intraday_mid', 'impact')}
    s_close, s_high, s_low = (wide(spot, k) for k in ('close', 'high', 'low'))
    basis = (m['mid_1500'] - s_close) / s_close
    close, high, low = (bars[k].reindex(index=dates, columns=list(liquidity__SYMBOLS)) for k in ('closew', 'highw', 'loww'))
    width = high - low
    per = {'clv_futures': (close - low) / width.where(width > 0), 'clv_spot': (s_close - s_low) / (s_high - s_low).where(s_high > s_low), 'basis': basis, 'd_basis_next': (basis.shift(-1) - basis).where(same_next), 'next_spot': s_close.shift(-1) / s_close - 1, 'next_futures_cc': (m['mid_1500'].shift(-1) / m['mid_1500'] - 1).where(same_next), 'next_intraday': liquidity__np.log(m['mid_close'].shift(-1) / m['mid_open'].shift(-1)), 'next_first30': liquidity__np.log(m['mid_first30'].shift(-1) / m['mid_open'].shift(-1)), 'flow': m['flow'], 'flow_tail': m['flow_tail'], 'resid': m['resid'], 'impact': m['impact'], 'depth_recovery': m['depth_head'].shift(-1) / m['depth_tail'], 'basis_dev20': basis - basis.shift(1).rolling(20, min_periods=20).mean()}
    out = liquidity__pd.DataFrame({k: v.mean(axis=1, skipna=False) for k, v in per.items()}, index=dates)
    out['next_rest'] = out['next_intraday'] - out['next_first30']
    impact_median = out['impact'].shift(1).rolling(liquidity__IMPACT_WINDOW, min_periods=liquidity__IMPACT_WINDOW // 2).median()
    out['high_impact'] = (out['impact'] > impact_median).astype(float).where(impact_median.notna())
    return out.replace([liquidity__np.inf, -liquidity__np.inf], liquidity__np.nan)

def liquidity__fit(y: liquidity__pd.Series, xs: list[liquidity__pd.Series]) -> tuple[float, float, float, int]:
    frame = liquidity__pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 60:
        return (liquidity__np.nan, liquidity__np.nan, liquidity__np.nan, len(frame))
    b, t = liquidity__ols_hac(frame.iloc[:, 0].to_numpy(), [frame.iloc[:, i].to_numpy() for i in range(1, frame.shape[1])], liquidity__LAGS)
    return (b, t, abs(b / t) if t else liquidity__np.nan, len(frame))

def liquidity__verdict(value, t, expected) -> str:
    if liquidity__pd.isna(t):
        return '不确定'
    right = value < 0 if expected == '<0' else value > 0
    if abs(t) >= 2:
        return '支持' if right else '否定'
    return '不确定'

def liquidity__row(tag, label, value, t, se, n, expected) -> dict:
    return {'预测': tag, '内容': label, '系数': value, 'NW_t': t, '天数': n, '95%下限': value - 1.96 * se if liquidity__pd.notna(se) else liquidity__np.nan, '95%上限': value + 1.96 * se if liquidity__pd.notna(se) else liquidity__np.nan, '80%功效最小可检出': 2.8 * se if liquidity__pd.notna(se) else liquidity__np.nan, '预期': expected, '判断': liquidity__verdict(value, t, expected)}

def liquidity__yearly_signs(y, x) -> str:
    signs = []
    for year in sorted(set(y.dropna().index.year)):
        b = liquidity__fit(y[y.index.year == year], [x[x.index.year == year]])[0]
        signs.append(f"{year}:{('+' if b > 0 else '−' if b < 0 else '?')}")
    return ' '.join(signs)

def liquidity__tests(p: liquidity__pd.DataFrame) -> liquidity__pd.DataFrame:
    rows = []
    clv, y = (p['clv_futures'], p['next_intraday'])

    def add(tag, label, target, xs, expected):
        b, t, se, n = liquidity__fit(target, xs)
        rows.append(liquidity__row(tag, label, b, t, se, n, expected))
        return (b, t, se)
    b_cc = add('A0', '期货次日收盘到收盘 ~ 期货收盘位置', p['next_futures_cc'], [clv], '<0')[0]
    add('A1', '次日基差变化 ~ 期货收盘位置', p['d_basis_next'], [clv], '<0')
    b_spot = add('A2', '现货次日收益 ~ 期货收盘位置', p['next_spot'], [clv], '<0')[0]
    rows.append({'预测': 'A2b', '内容': '现货斜率 / 期货收盘到收盘斜率', '系数': b_spot / b_cc if b_cc else liquidity__np.nan, '预期': '远小于 1 支持套利限制', '判断': '见数值'})
    add('A3', '期货次日收盘到收盘 ~ 期货收盘位置 + 现货收盘位置：期货项', p['next_futures_cc'], [clv, p['clv_spot']], '<0')
    add('A3b', '同上：现货收盘位置项', p['next_futures_cc'], [p['clv_spot'], clv], '<0')
    add('A3c', '现货次日收益 ~ 现货收盘位置', p['next_spot'], [p['clv_spot']], '<0')
    low = (clv <= clv.quantile(0.2)).astype(float).where(clv.notna())
    add('A4', '收盘基差偏离 20 日均值 ~ 1{低位收盘}', p['basis_dev20'], [low], '<0')
    add('B0', '次日盘中 ~ 期货收盘位置', y, [clv], '<0')
    add('B1', '次日盘中 ~ 流量部分 F', y, [p['flow']], '<0')
    bf, tf, sef = add('B2', '次日盘中 ~ F + R：F 项', y, [p['flow'], p['resid']], '<0')
    br, tr, ser = add('O1', '次日盘中 ~ F + R：R 项（非流量部分）', y, [p['resid'], p['flow']], '<0')
    bd, td, sed, nd = liquidity__fit(y, [p['flow'], p['flow'] + p['resid']])
    rows.append(liquidity__row('B2b', 'F 系数 − R 系数（<0 表示流量部分回复更多）', bd, td, sed, nd, '<0'))
    add('B2c', '次日盘中 ~ 尾盘流量部分', y, [p['flow_tail']], '<0')
    add('B2d', '次日盘中 ~ 收盘位置 + F + R：收盘位置项', y, [clv, p['flow'], p['resid']], '<0')
    b30 = add('B3', '次日开盘后 30 分钟 ~ 期货收盘位置', p['next_first30'], [clv], '<0')[0]
    brest = add('B3b', '次日其余时段 ~ 期货收盘位置', p['next_rest'], [clv], '<0')[0]
    rows.append({'预测': 'B3c', '内容': '开盘后 30 分钟占次日盘中斜率的比例', '系数': b30 / (b30 + brest) if b30 + brest else liquidity__np.nan, '预期': '> 0.5', '判断': '支持' if b30 / (b30 + brest) > 0.5 else '不支持'})
    hi = p['high_impact']
    add('B4', '次日盘中 ~ 收盘位置 × 高冲击日（交乘）', y, [clv * hi, clv, hi], '<0')
    add('B5', '次日深度恢复 ~ 1{低位收盘}', p['depth_recovery'], [low], '>0')
    lows = low.eq(1)
    corr = p.loc[lows, ['depth_recovery', 'next_intraday']].dropna()
    rows.append({'预测': 'B5b', '内容': '低位收盘日：深度恢复与次日盘中收益的相关', '系数': float(corr.corr().iloc[0, 1]), '天数': len(corr), '预期': '>0', '判断': '见数值'})
    out = liquidity__pd.DataFrame(rows)
    out.loc[out['预测'].isin(['A0', 'A1', 'A2', 'B0', 'B1', 'B3', 'O1']), '逐年符号'] = [liquidity__yearly_signs(target, x) for target, x in ((p['next_futures_cc'], clv), (p['d_basis_next'], clv), (p['next_spot'], clv), (y, clv), (y, p['flow']), (p['next_first30'], clv), (y, p['resid']))]
    return out

def liquidity__post_hoc(p: liquidity__pd.DataFrame, features: liquidity__pd.DataFrame, spot: liquidity__pd.DataFrame) -> liquidity__pd.DataFrame:
    """看过预登记结果后加的拆分（登记文件 post_hoc 一节）：基差回归发生在隔夜还是次日盘中，现货自己怎么走。"""

    def wide(frame, field):
        return frame.pivot(index='trading_date', columns='symbol', values=field).reindex(index=p.index, columns=list(liquidity__SYMBOLS))
    contract = wide(features, 'contract')
    same_next = contract.eq(contract.shift(-1))
    s_open, s_close = (wide(spot, 'open'), wide(spot, 'close'))
    b_close = (wide(features, 'mid_1500') - s_close) / s_close
    b_open = (wide(features, 'mid_open') - s_open) / s_open
    series = {'基差 收盘→次日开盘': (b_open.shift(-1) - b_close).where(same_next), '基差 次日开盘→次日收盘': b_close.shift(-1) - b_open.shift(-1), '现货 收盘→次日开盘': s_open.shift(-1) / s_close - 1, '现货 次日开盘→收盘': s_close.shift(-1) / s_open.shift(-1) - 1}
    series = {k: v.mean(axis=1, skipna=False) for k, v in series.items()}
    series['期货 次日盘中（中间价）'] = p['next_intraday']
    series['期货 次日开盘后30分钟'] = p['next_first30']
    rows = []
    for label, start in (('全样本', p.index.min()), ('2016+', liquidity__SUB_START)):
        m = p.index >= start
        for xname in ('clv_futures', 'clv_spot'):
            x = p.loc[m, xname]
            q5 = liquidity__pd.qcut(x, 5, labels=False) + 1
            for name, y in series.items():
                b, t, se, n = liquidity__fit(y[m], [x])
                rows.append({'样本': label, '预测变量': xname, '结果': name, '斜率': b, 'NW_t': t, '天数': n, **{f'Q{k}均值bp': float(y[m].groupby(q5).mean().get(k, liquidity__np.nan) * 10000.0) for k in range(1, 6)}})
    return liquidity__pd.DataFrame(rows)

def liquidity__run() -> None:
    protocol = liquidity__load_protocol()
    dates, bars, *_ = liquidity__I._load_index_inputs()
    dates = dates[(dates >= liquidity__START) & (dates <= liquidity__END)]
    liquidity__C.assert_no_holdout_dates(dates, '股指机制日期')
    spot = liquidity__pd.read_csv(liquidity__SPOT_PATH, parse_dates=['date']).rename(columns={'date': 'trading_date'})
    liquidity__C.assert_no_holdout_dates(spot['trading_date'], '现货指数日线')
    liquidity__OUT.mkdir(parents=True, exist_ok=True)
    cache = liquidity__OUT / 'tick_features.csv'
    if cache.is_file():
        features = liquidity__pd.read_csv(cache, parse_dates=['trading_date'])
    else:
        features = liquidity__build_tick_features()
        features.to_csv(cache, index=False, encoding='utf-8-sig')
    liquidity__C.assert_no_holdout_dates(features['trading_date'], 'L1 逐笔特征')
    coverage = features.groupby('symbol')['trading_date'].agg(['min', 'max', 'count'])
    panel = liquidity__market_panel(features, spot, bars, dates)
    full = liquidity__tests(panel)
    sub = liquidity__tests(panel[panel.index >= liquidity__SUB_START])
    full.insert(0, '样本', f'{dates.min().date()}..{dates.max().date()}')
    sub.insert(0, '样本', f'{liquidity__SUB_START.date()}..{dates.max().date()}')
    table = liquidity__pd.concat([full, sub], ignore_index=True)
    table.to_csv(liquidity__OUT / 'predictions.csv', index=False, encoding='utf-8-sig')
    extra = liquidity__post_hoc(panel, features, spot)
    extra.to_csv(liquidity__OUT / 'post_hoc_split.csv', index=False, encoding='utf-8-sig')
    panel.to_csv(liquidity__OUT / 'daily_panel.csv', encoding='utf-8-sig')
    desc = {'逐笔覆盖': {k: {'min': str(v['min'].date()), 'max': str(v['max'].date()), 'days': int(v['count'])} for k, v in coverage.iterrows()}, 'lambda 有值的品种日占比': float(features['lambda'].notna().mean()), '收盘基差均值': float(panel['basis'].mean()), '收盘基差中位': float(panel['basis'].median()), '期货收盘位置与现货收盘位置相关': float(panel['clv_futures'].corr(panel['clv_spot'])), 'F 与当日收益相关': float(panel['flow'].corr(panel['flow'] + panel['resid'])), 'F 与收盘位置相关': float(panel['flow'].corr(panel['clv_futures'])), 'R 与收盘位置相关': float(panel['resid'].corr(panel['clv_futures'])), '高冲击日占比': float(panel['high_impact'].mean())}
    (liquidity__OUT / 'protocol.json').write_text(liquidity__json.dumps({'protocol': protocol, 'descriptive': desc}, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    liquidity__pd.set_option('display.width', 260)
    liquidity__pd.set_option('display.max_columns', 20)
    liquidity__pd.set_option('display.max_colwidth', 44)
    print(table.drop(columns=['逐年符号']).round(5).to_string(index=False))
    print(full[['预测', '逐年符号']].dropna().to_string(index=False))
    print(extra.round(5).to_string(index=False))
    print(liquidity__json.dumps(desc, ensure_ascii=False, indent=1, default=float))
    print(f'结果：{liquidity__OUT}')
# 会员持仓与未吸收卖压。
import argparse as positioning__argparse
import json as positioning__json
import warnings as positioning__warnings
from pathlib import Path as positioning__Path
import numpy as positioning__np
import pandas as positioning__pd
from research.index import index_market_clock as positioning__I
from research.time_factor_screen import ols_hac as positioning__ols_hac
from tfcta import config as positioning__C
positioning__CONFIG_PATH = positioning__C.CONFIG_DIR / 'research' / 'index_positioning_mechanism.json'
positioning__UNABSORBED_PATH = positioning__C.CONFIG_DIR / 'research' / 'index_unabsorbed_pressure.json'
positioning__UNABSORBED_OUT = positioning__C.study_dir('index', 'index_unabsorbed_pressure')
positioning__OUT = positioning__C.study_dir('index', 'index_positioning_mechanism')
positioning__RANK_PATH = positioning__C.INDEX_MEMBER_RANK_PATH
positioning__CONTRACT_DIR = positioning__C.INDEX_CONTRACTS_DIR
positioning__SYMBOLS = ('IF', 'IH', 'IC')
positioning__START, positioning__END = (positioning__pd.Timestamp('2015-04-16'), positioning__pd.Timestamp(positioning__C.RESEARCH_END))
positioning__SUB_START = positioning__pd.Timestamp('2016-01-04')
positioning__LAGS = 5
positioning__STATE_WINDOW = 250

def positioning__load_protocol(path: positioning__Path=positioning__CONFIG_PATH) -> dict:
    protocol = positioning__json.loads(path.read_text(encoding='utf-8'))
    if protocol.get('status') != 'declared_test_not_a_strategy':
        raise ValueError('检验登记必须声明这不是策略')
    if '不读 2022' not in protocol.get('sample', '') or '不读 2026' not in protocol.get('sample', ''):
        raise ValueError('样本口径必须写明不读 2022 以后和 2026')
    return protocol

def positioning__above_own_median(series: positioning__pd.Series, window: int=positioning__STATE_WINDOW) -> positioning__pd.Series:
    """高于自身过去 window 天（不含当天，至少一半）的中位数为 1，否则 0；历史不足为缺失。"""
    median = series.shift(1).rolling(window, min_periods=window // 2).median()
    return (series > median).astype(float).where(median.notna() & series.notna())

def positioning__positioning_panel(rank: positioning__pd.DataFrame, contract_daily: dict, dates: positioning__pd.DatetimeIndex) -> dict:
    """逐品种：总持仓、持仓对数变化、前 20 名多空当日变化占前一日总持仓的比例。"""
    oi, d_oi, short_flow, long_flow = ({}, {}, {}, {})
    for symbol in positioning__SYMBOLS:
        daily = contract_daily[symbol]
        total = daily.groupby('date')['open_interest'].sum().reindex(dates)
        prev = total.shift(1)
        oi[symbol] = total
        d_oi[symbol] = positioning__np.log(total / prev)
        part = rank[rank['symbol'] == symbol]
        flows = part.groupby(['side', 'trading_date'])['volume_change'].sum().unstack('side')
        flows = flows.reindex(dates)
        short_flow[symbol] = flows['short'] / prev
        long_flow[symbol] = flows['long'] / prev
    frame = lambda d: positioning__pd.DataFrame(d, index=dates)[list(positioning__SYMBOLS)]
    return {'oi': frame(oi), 'd_oi': frame(d_oi), 'short_flow': frame(short_flow), 'long_flow': frame(long_flow)}

def positioning__expiry_mask(dates: positioning__pd.DatetimeIndex, maturity: positioning__pd.Series) -> positioning__pd.Series:
    """当天或下一交易日是股指合约到期日为真。"""
    expiries = set(positioning__pd.DatetimeIndex(maturity).normalize())
    today = positioning__pd.Series(dates.isin(list(expiries)), index=dates)
    return today | today.shift(-1, fill_value=False)

def positioning__fit(y: positioning__pd.Series, xs: list[positioning__pd.Series]):
    frame = positioning__pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 60:
        return (positioning__np.nan, positioning__np.nan, positioning__np.nan, len(frame))
    b, t = positioning__ols_hac(frame.iloc[:, 0].to_numpy(), [frame.iloc[:, i].to_numpy() for i in range(1, frame.shape[1])], positioning__LAGS)
    return (b, t, abs(b / t) if t else positioning__np.nan, len(frame))

def positioning__row(tag, label, y, xs, expected, scale=1.0):
    b, t, se, n = positioning__fit(y, xs)
    right = b < 0 if expected == '<0' else b > 0
    verdict = '不确定' if positioning__pd.isna(t) or abs(t) < 2 else '支持' if right else '否定'
    years = []
    for year in sorted(set(y.dropna().index.year)):
        m = y.index.year == year
        by = positioning__fit(y[m], [x[m] for x in xs])[0]
        years.append('+' if by > 0 else '−' if by < 0 else '?')
    return {'预测': tag, '内容': label, '系数': b * scale, 'NW_t': t, '天数': n, '95%下限': (b - 1.96 * se) * scale if positioning__pd.notna(se) else positioning__np.nan, '95%上限': (b + 1.96 * se) * scale if positioning__pd.notna(se) else positioning__np.nan, '80%功效最小可检出': 2.8 * se * scale if positioning__pd.notna(se) else positioning__np.nan, '预期': expected, '判断': verdict, '逐年符号': ''.join(years)}

def positioning__tests(p: positioning__pd.DataFrame) -> positioning__pd.DataFrame:
    clv, y = (p['clv'], p['next_intraday'])
    hi_net_short = positioning__above_own_median(-p['net_flow'])
    oi_up = (p['d_oi'] > 0).astype(float).where(p['d_oi'].notna())
    oi_down = 1.0 - oi_up
    rows = [positioning__row('基准', '次日盘中 ~ 收盘位置（bp/单位）', y, [clv], '<0', 10000.0), positioning__row('基准', '账本下一段 ~ 收盘位置（bp/单位）', p['fwd'], [clv], '<0', 10000.0), positioning__row('H1', '当天前20空头变化/持仓 ~ 收盘位置（bp）', p['short_flow'], [clv], '<0', 10000.0), positioning__row('H1', '当天前20净多变化/持仓 ~ 收盘位置（bp）', p['net_flow'], [clv], '>0', 10000.0), positioning__row('H2', '次日前20空头变化 ~ 当天收盘位置（bp）', p['short_flow_next'], [clv], '>0', 10000.0), positioning__row('H2', '次日前20净多变化 ~ 当天收盘位置（bp）', p['net_flow_next'], [clv], '<0', 10000.0), positioning__row('H3', '次日盘中 ~ 净多变化 + 收盘位置：净多变化项', y, [p['net_flow'], clv], '<0'), positioning__row('H3', '次日盘中 ~ 净多变化 + 收盘位置：收盘位置项（bp）', y, [clv, p['net_flow']], '<0', 10000.0), positioning__row('H3', '次日盘中 ~ 收盘位置 × 1{大会员净加空高}（交乘，bp）', y, [clv * hi_net_short, clv, hi_net_short], '<0', 10000.0), positioning__row('H4', '次日盘中 ~ 收盘位置 × 1{增仓}（交乘，bp）', y, [clv * oi_up, clv, oi_up], '<0', 10000.0), positioning__row('L1', '次日盘中 ~ 收盘位置 × 1{减仓}（交乘，bp）', y, [clv * oi_down, clv, oi_down], '<0', 10000.0), positioning__row('L2', '当天前20多头变化/持仓 ~ 收盘位置（bp）', p['long_flow'], [clv], '>0', 10000.0), positioning__row('描述', '当天持仓对数变化 ~ 收盘位置（bp）', p['d_oi'], [clv], '<0', 10000.0)]
    for state, mask in (('增仓日', oi_up == 1), ('减仓日', oi_down == 1), ('大会员净加空高', hi_net_short == 1), ('大会员净加空低', hi_net_short == 0)):
        rows.append(positioning__row('分组', f'{state}：次日盘中 ~ 收盘位置（bp）', y[mask], [clv[mask]], '<0', 10000.0))
    return positioning__pd.DataFrame(rows)

def positioning__unabsorbed_terms(clv: positioning__pd.Series, absorption: positioning__pd.Series) -> positioning__pd.DataFrame:
    """P = max(0, 0.5 − clv)；U = P × 1{A ≤ 0}（大会员也在卖），B = P × 1{A > 0}（大会员在接）。"""
    pressure = (0.5 - clv).clip(lower=0.0)
    taking = absorption > 0
    valid = clv.notna() & absorption.notna()
    return positioning__pd.DataFrame({'P': pressure, 'U': pressure.where(~taking, 0.0), 'B': pressure.where(taking, 0.0)}).where(valid)

def positioning__unabsorbed_tests(p: positioning__pd.DataFrame) -> positioning__pd.DataFrame:
    terms = positioning__unabsorbed_terms(p['clv'], p['net_flow'])
    rows = []
    for outcome, label in (('next_intraday', '次日盘中'), ('fwd', '账本下一段')):
        y = p[outcome]
        rows += [positioning__row('U1', f'{label} ~ U（bp/单位）', y, [terms['U']], '>0', 10000.0), positioning__row('U2', f'{label} ~ U + B：U 项（bp）', y, [terms['U'], terms['B']], '>0', 10000.0), positioning__row('U2', f'{label} ~ U + B：B 项（bp）', y, [terms['B'], terms['U']], '>0', 10000.0), positioning__row('U2b', f'{label}：U 系数 − B 系数（bp）', y, [terms['U'], terms['U'] + terms['B']], '>0', 10000.0), positioning__row('U3', f'{label} ~ U + clv：U 项（bp）', y, [terms['U'], p['clv']], '>0', 10000.0), positioning__row('U3', f'{label} ~ U + clv：clv 项（bp）', y, [p['clv'], terms['U']], '<0', 10000.0), positioning__row('参照', f'{label} ~ P（bp）', y, [terms['P']], '>0', 10000.0)]
    return positioning__pd.DataFrame(rows)

def positioning__run(unabsorbed: bool=False) -> None:
    protocol = positioning__load_protocol()
    if unabsorbed:
        extra = positioning__json.loads(positioning__UNABSORBED_PATH.read_text(encoding='utf-8'))
        if '不读 2022' not in extra.get('sample', '') or '不读 2026' not in extra.get('sample', ''):
            raise ValueError('样本口径必须写明不读 2022 以后和 2026')
    dates, bars, *_ = positioning__I._load_index_inputs()
    dates = dates[(dates >= positioning__START) & (dates <= positioning__END)]
    positioning__C.assert_no_holdout_dates(dates, '股指持仓检验日期')
    rank = positioning__pd.read_csv(positioning__RANK_PATH, parse_dates=['trading_date'])
    positioning__C.assert_no_holdout_dates(rank['trading_date'], '会员持仓')
    contract_daily = {}
    for symbol in positioning__SYMBOLS:
        daily = positioning__pd.read_pickle(positioning__CONTRACT_DIR / f'{symbol}_contract_daily.pkl')
        daily['date'] = positioning__pd.to_datetime(daily['date'])
        positioning__C.assert_no_holdout_dates(daily['date'], f'{symbol} 合约日线')
        contract_daily[symbol] = daily
    maturity = positioning__pd.read_csv(positioning__CONTRACT_DIR / 'contract_maturity.csv', parse_dates=['maturity_date'])['maturity_date']
    pos = positioning__positioning_panel(rank, contract_daily, dates)
    close, high, low = (bars[k].reindex(index=dates, columns=list(positioning__SYMBOLS)) for k in ('closew', 'highw', 'loww'))
    width = high - low
    o, ow, cw = (bars[k].reindex(index=dates, columns=list(positioning__SYMBOLS)) for k in ('open', 'openw', 'closew'))
    per = {'clv': (close - low) / width.where(width > 0), 'next_intraday': ((cw - ow) / o.where(o > 0)).shift(-1), 'fwd': ((ow.shift(-1) - ow) / o.where(o > 0)).shift(-1), 'd_oi': pos['d_oi'], 'short_flow': pos['short_flow'], 'long_flow': pos['long_flow'], 'net_flow': pos['long_flow'] - pos['short_flow']}
    panel = positioning__pd.DataFrame({k: v.mean(axis=1, skipna=False) for k, v in per.items()}, index=dates)
    panel['short_flow_next'] = panel['short_flow'].shift(-1)
    panel['net_flow_next'] = panel['net_flow'].shift(-1)
    excluded = positioning__expiry_mask(dates, maturity)
    panel = panel[~excluded.to_numpy()].replace([positioning__np.inf, -positioning__np.inf], positioning__np.nan)
    if unabsorbed:
        full, sub = (positioning__unabsorbed_tests(panel), positioning__unabsorbed_tests(panel[panel.index >= positioning__SUB_START]))
        full.insert(0, '样本', f'{panel.index.min().date()}..{panel.index.max().date()}')
        sub.insert(0, '样本', f'{positioning__SUB_START.date()}..{panel.index.max().date()}')
        table = positioning__pd.concat([full, sub], ignore_index=True)
        positioning__UNABSORBED_OUT.mkdir(parents=True, exist_ok=True)
        table.to_csv(positioning__UNABSORBED_OUT / 'predictions.csv', index=False, encoding='utf-8-sig')
        terms = positioning__unabsorbed_terms(panel['clv'], panel['net_flow'])
        desc = {'收在中点以下的日子': int((terms['P'] > 0).sum()), '其中大会员也在卖（U>0）': int((terms['U'] > 0).sum()), '其中大会员在接（B>0）': int((terms['B'] > 0).sum())}
        (positioning__UNABSORBED_OUT / 'protocol.json').write_text(positioning__json.dumps({'protocol': extra, 'descriptive': desc}, ensure_ascii=False, indent=2), encoding='utf-8')
        positioning__pd.set_option('display.width', 260)
        positioning__pd.set_option('display.max_colwidth', 40)
        print(table[['样本', '预测', '内容', '系数', 'NW_t', '天数', '95%下限', '95%上限', '判断', '逐年符号']].round(3).to_string(index=False))
        print(desc)
        return
    full = positioning__tests(panel)
    sub = positioning__tests(panel[panel.index >= positioning__SUB_START])
    full.insert(0, '样本', f'{panel.index.min().date()}..{panel.index.max().date()}')
    sub.insert(0, '样本', f'{positioning__SUB_START.date()}..{panel.index.max().date()}')
    table = positioning__pd.concat([full, sub], ignore_index=True)
    positioning__OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(positioning__OUT / 'predictions.csv', index=False, encoding='utf-8-sig')
    panel.to_csv(positioning__OUT / 'daily_panel.csv', encoding='utf-8-sig')
    top20_share = {s: float((rank[(rank.symbol == s) & (rank.side == 'short')].groupby('trading_date')['volume'].sum() / contract_daily[s].groupby('date')['open_interest'].sum()).median()) for s in positioning__SYMBOLS}
    desc = {'排除的到期相关日数': int(excluded.sum()), '可用天数': len(panel), '前20空头占总持仓中位': top20_share, '收盘位置与当天持仓变化相关': float(panel['clv'].corr(panel['d_oi'])), '收盘位置与前20净多变化相关': float(panel['clv'].corr(panel['net_flow']))}
    (positioning__OUT / 'protocol.json').write_text(positioning__json.dumps({'protocol': protocol, 'descriptive': desc}, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    positioning__pd.set_option('display.width', 260)
    positioning__pd.set_option('display.max_colwidth', 40)
    print(table[['样本', '预测', '内容', '系数', 'NW_t', '天数', '95%下限', '95%上限', '判断', '逐年符号']].round(3).to_string(index=False))
    print(positioning__json.dumps(desc, ensure_ascii=False, indent=1, default=float))
    print(f'结果：{positioning__OUT}')
# 隐含波动率与反转。
import json as qvix__json
from pathlib import Path as qvix__Path
import numpy as qvix__np
import pandas as qvix__pd
from scipy.stats import spearmanr as qvix__spearmanr
from research.time_factor_screen import ols_hac as qvix__ols_hac
from strategies.index_common import market_close_location as qvix__market_close_location
from tfcta import config as qvix__C
qvix__CONFIG_PATH = qvix__C.CONFIG_DIR / 'research' / 'index_qvix_reversal.json'
qvix__OUT = qvix__C.study_dir('index', 'index_qvix_reversal')
qvix__SYMBOLS = ('IF', 'IH', 'IC')
qvix__LAGS = 5

def qvix__load_protocol(path: qvix__Path=qvix__CONFIG_PATH) -> dict:
    protocol = qvix__json.loads(path.read_text(encoding='utf-8'))
    parameters = protocol.get('parameters') or {}
    sample = protocol.get('sample', '')
    if protocol.get('status') != 'declared_test_not_a_strategy' or '不读 2022' not in sample or '不读 2026' not in sample or (list(protocol.get('predictions', {})) != ['Q1', 'Q2', 'Q3']) or (parameters.get('primary_start') != '2016-01-04') or (parameters.get('full_start') != '2015-04-16') or (parameters.get('outcome_end') != '2021-12-31') or (parameters.get('hac_lags') != qvix__LAGS) or (parameters.get('windows') != [1, 5, 20, 60]) or (parameters.get('overlap_min_corr') != 0.99):
        raise ValueError('QVIX 反转检验登记与实现或样本边界不一致')
    return protocol

def qvix___fit(y: qvix__pd.Series, xs: list[qvix__pd.Series]) -> tuple[float, float, int]:
    frame = qvix__pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 60:
        return (qvix__np.nan, qvix__np.nan, len(frame))
    coef, tstat = qvix__ols_hac(frame.iloc[:, 0].to_numpy(), [frame.iloc[:, i].to_numpy() for i in range(1, frame.shape[1])], qvix__LAGS)
    return (coef, tstat, len(frame))

def qvix___verdict(coef: float, tstat: float, expected: str) -> str:
    if qvix__pd.isna(tstat):
        return '不确定'
    right = coef < 0 if expected == '<0' else coef > 0
    if abs(tstat) >= 2:
        return '支持' if right else '否定'
    return '不确定'

def qvix___yearly(y: qvix__pd.Series, xs: list[qvix__pd.Series]) -> str:
    signs = []
    for year in sorted(set(y.dropna().index.year)):
        mask = y.index.year == year
        coef = qvix___fit(y[mask], [x[mask] for x in xs])[0]
        signs.append(f"{year}:{('+' if coef > 0 else '−' if coef < 0 else '?')}")
    return ' '.join(signs)

def qvix__futures_panel() -> qvix__pd.DataFrame:
    """信号日收盘位置，以及下一交易日开盘到收盘的等权收益。"""
    from tfcta.data import load_daily_bars
    bars = load_daily_bars(list(qvix__SYMBOLS), directory=qvix__C.INDEX_RESEARCH_DIR)
    width = bars['highw'] - bars['loww']
    ok = (width.gt(0) & bars['closew'].notna() & bars['open'].gt(0)).all(axis=1)
    index = ok.index[ok]
    index = index[(index >= qvix__pd.Timestamp('2015-04-16')) & (index <= qvix__pd.Timestamp('2021-12-31'))]
    clv = qvix__market_close_location(bars, index)
    intra = ((bars['closew'] - bars['openw']) / bars['open']).mean(axis=1, skipna=False)
    panel = qvix__pd.DataFrame({'clv': clv, 'next_intraday': intra.reindex(index).shift(-1)})
    qvix__C.assert_no_holdout_dates(panel.index, 'QVIX 反转检验日期')
    return panel.dropna(subset=['clv'])

def qvix__attach_qvix(panel: qvix__pd.DataFrame, qvix: qvix__pd.DataFrame) -> qvix__pd.DataFrame:
    """信号日用当天 QVIX 收盘。下一交易日的 QVIX 不进入这一行。"""
    level = qvix.set_index('date')['close'].sort_index()
    qvix__C.assert_no_holdout_dates(level.index, 'QVIX 研究期')
    out = panel.copy()
    out['qvix'] = level.reindex(out.index)
    return out

def qvix__vol_on_signal_day(level: qvix__pd.Series, window: int) -> qvix__pd.Series:
    """水平，或截至信号日的均值。滚动窗口含当天、不含未来。"""
    if window == 1:
        return level
    return level.rolling(window, min_periods=window).mean()

def qvix__assert_matches_official(qvix: qvix__pd.DataFrame, minimum_corr: float, low: float, high: float) -> dict:
    """研究期内和官方中国波指重叠的日子必须对得上。"""
    from tfcta.rq_auth import rqdata_credentials
    import rqdatac as rq
    credentials = rqdata_credentials()
    if credentials is None:
        raise RuntimeError('缺少 RiceQuant 凭证，无法核对官方中国波指')
    rq.init(*credentials)
    official = rq.get_price('000188.XSHG', start_date='2016-11-28', end_date='2018-02-14', frequency='1d', fields=['close'])
    if official is None or official.empty:
        raise RuntimeError('官方中国波指没有返回研究期价格')
    official = official.reset_index()
    official['date'] = qvix__pd.to_datetime(official['date']).dt.normalize()
    qvix__C.assert_no_holdout_dates(official['date'], '官方中国波指')
    merged = qvix.merge(official[['date', 'close']], on='date', suffixes=('_qvix', '_official'))
    if len(merged) < 60:
        raise ValueError(f'QVIX 与官方中国波指只重叠 {len(merged)} 天')
    corr = float(merged['close_qvix'].corr(merged['close_official']))
    ratio = float((merged['close_qvix'] / merged['close_official']).median())
    if corr < minimum_corr or not low <= ratio <= high:
        raise ValueError(f'QVIX 与官方中国波指不一致：相关 {corr:.4f}，中位比 {ratio:.4f}')
    return {'重叠天数': len(merged), '相关': corr, '中位比': ratio}

def qvix__tests(panel: qvix__pd.DataFrame, windows: list[int]) -> tuple[qvix__pd.DataFrame, qvix__pd.DataFrame]:
    y = panel['next_intraday']
    centered = panel['clv'] - panel['clv'].mean()
    rows = []
    for window in windows:
        vol = qvix__vol_on_signal_day(panel['qvix'], window)
        z = (vol - vol.mean()) / vol.std()
        label = 'QVIX水平' if window == 1 else f'QVIX{window}日均值'
        specs = (('clv', [panel['clv'], z, centered * z], ''), (label, [z, panel['clv'], centered * z], ''), (f'clv×{label}', [centered * z, panel['clv'], z], '<0'))
        for name, xs, expected in specs:
            coef, tstat, nobs = qvix___fit(y, xs)
            rows.append({'检验': 'Q1', '变量': name, '系数_bp': coef * 10000.0, 'NW_t': tstat, '天数': nobs, '预期': expected or '—', '判断': qvix___verdict(coef, tstat, expected) if expected else '—', '逐年符号': qvix___yearly(y, xs) if expected else ''})
    level = panel['qvix']
    past = level.shift(1)
    edges = [past.rolling(250, min_periods=120).quantile(q) for q in (0.2, 0.4, 0.6, 0.8)]
    valid = edges[0].notna() & level.notna()
    bucket = qvix__pd.Series(qvix__np.nan, index=panel.index)
    bucket.loc[valid] = 1 + sum(((level.loc[valid] > edge.loc[valid]).astype(int) for edge in edges))
    bins = []
    slopes = []
    for k in range(1, 6):
        mask = bucket == k
        coef, tstat, nobs = qvix___fit(y[mask], [panel.loc[mask, 'clv']])
        bins.append({'档': k, '天数': nobs, 'QVIX均值': float(level[mask].mean()) if mask.any() else qvix__np.nan, 'clv斜率_bp': coef * 10000.0, 'NW_t': tstat})
        slopes.append(coef)
    rho = float(qvix__spearmanr(range(1, 6), slopes).correlation) if qvix__np.isfinite(slopes).all() else qvix__np.nan
    rows.append({'检验': 'Q2', '变量': '档位与斜率秩相关', '系数_bp': rho, 'NW_t': qvix__np.nan, '天数': int(bucket.notna().sum()), '预期': '<0', '判断': '见数值', '逐年符号': ''})
    gross = (0.5 - panel['clv']) * y
    z_level = (level - level.mean()) / level.std()
    coef, tstat, nobs = qvix___fit(gross, [z_level])
    rows.append({'检验': 'Q3', '变量': 'QVIX水平', '系数_bp': coef * 10000.0, 'NW_t': tstat, '天数': nobs, '预期': '>0', '判断': qvix___verdict(coef, tstat, '>0'), '逐年符号': qvix___yearly(gross, [z_level])})
    return (qvix__pd.DataFrame(rows), qvix__pd.DataFrame(bins))

def qvix__run() -> None:
    from download.index.getQvix import load_research_qvix
    protocol = qvix__load_protocol()
    parameters = protocol['parameters']
    qvix = load_research_qvix()
    gate = qvix__assert_matches_official(qvix, parameters['overlap_min_corr'], parameters['overlap_ratio_low'], parameters['overlap_ratio_high'])
    panel = qvix__attach_qvix(qvix__futures_panel(), qvix)
    panel = panel[panel.index <= qvix__pd.Timestamp(parameters['outcome_end'])]
    tables, bins = ([], [])
    for label, start in (('2016-01-04 起', parameters['primary_start']), ('2015-04-16 起', parameters['full_start'])):
        part = panel[panel.index >= qvix__pd.Timestamp(start)]
        table, bucket = qvix__tests(part, parameters['windows'])
        tables.append(table.assign(样本=label))
        bins.append(bucket.assign(样本=label))
    table = qvix__pd.concat(tables, ignore_index=True)
    buckets = qvix__pd.concat(bins, ignore_index=True)
    qvix__OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(qvix__OUT / 'tests.csv', index=False, encoding='utf-8-sig')
    buckets.to_csv(qvix__OUT / 'buckets.csv', index=False, encoding='utf-8-sig')
    panel.to_csv(qvix__OUT / 'daily_panel.csv', encoding='utf-8-sig')
    (qvix__OUT / 'protocol.json').write_text(qvix__json.dumps({'protocol': protocol, '官方核对': gate}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    qvix__pd.set_option('display.width', 220)
    qvix__pd.set_option('display.max_rows', 80)
    print(table.round(3).to_string(index=False), flush=True)
    print(buckets.round(3).to_string(index=False), flush=True)
    print(qvix__json.dumps(gate, ensure_ascii=False), flush=True)
    print(f'结果：{qvix__OUT}', flush=True)
# IF 盘中承接成本制度。
import json as intraday_cost__json
from pathlib import Path as intraday_cost__Path
import numpy as intraday_cost__np
import pandas as intraday_cost__pd
from scipy.stats import norm as intraday_cost__norm
from research.time_factor_screen import ols_hac as intraday_cost__ols_hac
from tfcta import config as intraday_cost__C
intraday_cost__CONFIG_PATH = intraday_cost__C.CONFIG_DIR / 'research' / 'index_intraday_cost_regime.json'
intraday_cost__OUT = intraday_cost__C.study_dir('index', 'index_intraday_cost_regime')
intraday_cost__EARLY_DIR = intraday_cost__C.INDEX_DATA_ROOT / 'if_before_ih_ic' / 'minute_shards'
intraday_cost__RESEARCH_DIR = intraday_cost__C.INDEX_RESEARCH_DIR
intraday_cost__EARLY_END = intraday_cost__pd.Timestamp('2015-04-15')
intraday_cost__OVERLAP_DAY = intraday_cost__pd.Timestamp('2015-04-16')
intraday_cost__CHEAP_END = intraday_cost__pd.Timestamp('2015-08-25')
intraday_cost__EXPENSIVE_START = intraday_cost__pd.Timestamp('2015-09-07')
intraday_cost__CRASH_FREE_END = intraday_cost__pd.Timestamp('2015-05-29')
intraday_cost__RESEARCH_END = intraday_cost__pd.Timestamp('2021-12-31')
intraday_cost__CHECKPOINTS = ('11:30', '14:30')
intraday_cost__HAC_LAGS = 5

def intraday_cost__load_protocol(path: intraday_cost__Path=intraday_cost__CONFIG_PATH) -> dict:
    protocol = intraday_cost__json.loads(path.read_text(encoding='utf-8'))
    parameters = protocol.get('parameters', {})
    boundary = protocol.get('sample_boundary', '')
    if protocol.get('status') != 'declared_test_not_a_strategy' or '不读2022' not in boundary or '不读2026' not in boundary or (parameters.get('instrument') != 'IF') or (parameters.get('cheap_end') != str(intraday_cost__CHEAP_END.date())) or (parameters.get('expensive_start') != str(intraday_cost__EXPENSIVE_START.date())) or (parameters.get('checkpoints') != list(intraday_cost__CHECKPOINTS)) or (parameters.get('session_end') != '15:00') or (parameters.get('hac_lags') != intraday_cost__HAC_LAGS) or (protocol.get('predictions', {}).get('P1_intraday') is None):
        raise ValueError('IF 日内成本假说规格与实现或样本边界不一致')
    return protocol

def intraday_cost___read_partition(path: intraday_cost__Path, *, research: bool) -> tuple[intraday_cost__pd.DataFrame, intraday_cost__pd.DataFrame]:
    if research:
        intraday_cost__C.assert_research_only(path)
    minute = intraday_cost__pd.read_parquet(path)
    oi = minute[['open_interest', 'dominant_id', 'trading_date']].copy()
    for label, frame in (('IF 分钟', minute), ('IF 持仓量', oi)):
        if not isinstance(frame.index, intraday_cost__pd.DatetimeIndex) or frame.index.has_duplicates:
            raise ValueError(f'{label}必须有唯一的时间戳索引')
        if 'trading_date' not in frame:
            raise KeyError(f'{label}缺少 trading_date')
        dates = intraday_cost__pd.to_datetime(frame['trading_date']).dt.normalize()
        intraday_cost__C.assert_no_holdout_dates(dates, what=label)
        if not dates.is_monotonic_increasing:
            raise ValueError(f'{label}日期未按时间升序排列')
    if not minute.index.equals(oi.index):
        raise ValueError('IF 分钟行情与持仓量时间索引不一致')
    return (minute.sort_index(kind='mergesort'), oi.sort_index(kind='mergesort'))

def intraday_cost__load_if_inputs() -> tuple[intraday_cost__pd.DataFrame, intraday_cost__pd.DataFrame]:
    """拼接 IF 单独期与股指研究期，核对 2015-04-16 重叠日。"""
    early, early_oi = intraday_cost___read_partition(intraday_cost__EARLY_DIR / 'IF.parquet', research=False)
    later, later_oi = intraday_cost___read_partition(intraday_cost__RESEARCH_DIR / 'IF.parquet', research=True)
    early_dates = intraday_cost__pd.to_datetime(early['trading_date']).dt.normalize()
    later_dates = intraday_cost__pd.to_datetime(later['trading_date']).dt.normalize()
    overlap_early = early.loc[early_dates == intraday_cost__OVERLAP_DAY]
    overlap_later = later.loc[later_dates == intraday_cost__OVERLAP_DAY]
    if not overlap_early.index.equals(overlap_later.index):
        raise ValueError('IF 两个分区的 2015-04-16 时间戳不一致')
    for field in ('open', 'close', 'openw', 'closew', 'highw', 'loww'):
        if not intraday_cost__np.allclose(overlap_early[field].to_numpy(dtype='float64'), overlap_later[field].to_numpy(dtype='float64'), rtol=0.0, atol=1e-08):
            raise ValueError(f'IF 两个分区在 2015-04-16 的 {field} 不一致')
    overlap_early_oi = early_oi.loc[intraday_cost__pd.to_datetime(early_oi['trading_date']).dt.normalize() == intraday_cost__OVERLAP_DAY]
    overlap_later_oi = later_oi.loc[intraday_cost__pd.to_datetime(later_oi['trading_date']).dt.normalize() == intraday_cost__OVERLAP_DAY]
    if not overlap_early_oi.index.equals(overlap_later_oi.index):
        raise ValueError('IF 两个分区的 2015-04-16 持仓时间戳不一致')
    for field in ('open_interest', 'dominant_id'):
        if not overlap_early_oi[field].equals(overlap_later_oi[field]):
            raise ValueError(f'IF 两个分区在 2015-04-16 的 {field} 不一致')
    early_mask = early_dates <= intraday_cost__EARLY_END
    later_mask = (later_dates >= intraday_cost__OVERLAP_DAY) & (later_dates <= intraday_cost__RESEARCH_END)
    minute = intraday_cost__pd.concat([early.loc[early_mask], later.loc[later_mask]]).sort_index()
    oi = intraday_cost__pd.concat([early_oi.loc[early_mask], later_oi.loc[later_mask]]).sort_index()
    if minute.index.has_duplicates or not minute.index.is_monotonic_increasing:
        raise ValueError('IF 分区拼接后有重复或乱序的分钟')
    intraday_cost__C.assert_no_holdout_dates(minute['trading_date'], what='IF 拼接分钟')
    if intraday_cost__pd.to_datetime(minute['trading_date']).max() > intraday_cost__RESEARCH_END:
        raise intraday_cost__C.HoldoutViolation('IF 分钟越过 2021-12-31')
    return (minute, oi)

def intraday_cost___clock_mask(index: intraday_cost__pd.DatetimeIndex, full_session: bool) -> intraday_cost__np.ndarray:
    minutes = index.hour * 60 + index.minute
    if full_session:
        return intraday_cost__np.ones(len(index), dtype=bool)
    return (minutes >= 9 * 60 + 31) & (minutes <= 11 * 60 + 30) | (minutes >= 13 * 60 + 1) & (minutes <= 15 * 60)

def intraday_cost__intraday_observations(minute: intraday_cost__pd.DataFrame, *, full_session: bool=False) -> intraday_cost__pd.DataFrame:
    """按分钟结束时刻计算检查点位置与后续收益。"""
    rows = []
    for day, group in minute.groupby(intraday_cost__pd.to_datetime(minute['trading_date']).dt.normalize(), sort=True):
        base = intraday_cost__pd.Timestamp(day).normalize()
        group = group.sort_index(kind='mergesort')
        selected = group.loc[intraday_cost___clock_mask(group.index, full_session)]
        if selected.empty:
            continue
        end_time = selected.index[-1]
        if not full_session:
            end_time = base.replace(hour=15, minute=0)
            if end_time not in selected.index:
                continue
        end_row = selected.loc[end_time]
        for checkpoint in intraday_cost__CHECKPOINTS:
            hour, minute_part = (int(value) for value in checkpoint.split(':'))
            stamp = base.replace(hour=hour, minute=minute_part)
            if stamp not in selected.index:
                continue
            prior = selected.loc[selected.index <= stamp]
            close = float(selected.at[stamp, 'close'])
            open_px = float(prior['open'].iloc[0])
            if close <= 0 or open_px <= 0:
                raise ValueError(f'{day.date()} {checkpoint} IF 价格无效')
            width = float(prior['highw'].max() - prior['loww'].min())
            if not intraday_cost__np.isfinite(width) or width <= 0:
                continue
            clv = (float(prior.at[stamp, 'closew']) - float(prior['loww'].min())) / width
            if not intraday_cost__np.isfinite(clv) or not 0 <= clv <= 1:
                raise ValueError(f'{day.date()} {checkpoint} IF 日内位置超出 [0, 1]')
            next_bar = selected.loc[selected.index > stamp]
            if next_bar.empty:
                continue
            next_row = next_bar.iloc[0]
            outcome = (float(end_row['closew']) - float(selected.at[stamp, 'closew'])) / close
            skip_outcome = (float(end_row['closew']) - float(next_row['closew'])) / float(next_row['close'])
            rows.append({'date': intraday_cost__pd.Timestamp(day), 'checkpoint': checkpoint, 'location': clv, 'return_bp': outcome * 10000.0, 'skip_one_bp': skip_outcome * 10000.0, 'open_to_checkpoint': (float(selected.at[stamp, 'closew']) - float(prior['openw'].iloc[0])) / open_px, 'session_end': end_time, 'clock': 'full' if full_session else 'cash'})
    result = intraday_cost__pd.DataFrame(rows)
    if result.empty:
        raise ValueError('IF 分钟数据未形成任何预登记检查点')
    return result

def intraday_cost__daily_observations(minute: intraday_cost__pd.DataFrame) -> intraday_cost__pd.DataFrame:
    rows = []
    for day, group in minute.groupby(intraday_cost__pd.to_datetime(minute['trading_date']).dt.normalize(), sort=True):
        group = group.sort_index(kind='mergesort')
        width = float(group['highw'].max() - group['loww'].min())
        if width <= 0:
            continue
        rows.append({'date': intraday_cost__pd.Timestamp(day), 'clv': (float(group['closew'].iloc[-1]) - float(group['loww'].min())) / width, 'open': float(group['open'].iloc[0]), 'openw': float(group['openw'].iloc[0])})
    daily = intraday_cost__pd.DataFrame(rows).set_index('date').sort_index()
    if daily.index.has_duplicates or daily[['clv', 'open', 'openw']].isna().any().any():
        raise ValueError('IF 日线位置或开盘价缺失')
    daily['outcome_bp'] = (daily['openw'].shift(-2) - daily['openw'].shift(-1)) / daily['open'].shift(-1) * 10000.0
    daily['outcome_end'] = intraday_cost__pd.Series(daily.index, index=daily.index).shift(-2)
    return daily

def intraday_cost__fit(y: intraday_cost__pd.Series, focus: intraday_cost__pd.Series, controls: tuple[intraday_cost__pd.Series, ...]=()) -> dict:
    frame = intraday_cost__pd.concat([y.rename('y'), focus.rename('focus'), *controls], axis=1).dropna()
    if len(frame) < len(controls) + 10:
        return {'系数_bp': intraday_cost__np.nan, 'HAC_t': intraday_cost__np.nan, '样本数': len(frame)}
    coef, tstat = intraday_cost__ols_hac(frame['y'].to_numpy(dtype='float64'), [frame['focus'].to_numpy(dtype='float64')] + [frame.iloc[:, i].to_numpy(dtype='float64') for i in range(2, frame.shape[1])], intraday_cost__HAC_LAGS)
    return {'系数_bp': coef, 'HAC_t': tstat, '样本数': len(frame)}

def intraday_cost___regression_row(label: str, checkpoint: str, sample: str, result: dict, **extra) -> dict:
    return {'检验': label, '检查点': checkpoint, '样本': sample, **result, **extra}

def intraday_cost___cheap(date: intraday_cost__pd.Series) -> intraday_cost__pd.Series:
    return (date >= intraday_cost__pd.Timestamp('2010-01-01')) & (date <= intraday_cost__CHEAP_END)

def intraday_cost___expensive(date: intraday_cost__pd.Series) -> intraday_cost__pd.Series:
    return date >= intraday_cost__EXPENSIVE_START

def intraday_cost___p_two_sided(tstat: float) -> float:
    return float(2.0 * intraday_cost__norm.sf(abs(tstat))) if intraday_cost__np.isfinite(tstat) else intraday_cost__np.nan

def intraday_cost___holm(pvalues: list[float]) -> list[float]:
    values = intraday_cost__np.asarray(pvalues, dtype='float64')
    adjusted = intraday_cost__np.full(values.shape, intraday_cost__np.nan)
    finite = intraday_cost__np.flatnonzero(intraday_cost__np.isfinite(values))
    if not len(finite):
        return adjusted.tolist()
    order = finite[intraday_cost__np.argsort(values[finite])]
    running = 0.0
    for rank, ix in enumerate(order):
        running = max(running, (len(order) - rank) * values[ix])
        adjusted[ix] = min(running, 1.0)
    return adjusted.tolist()

def intraday_cost___period_masks(dates: intraday_cost__pd.Series) -> dict[str, intraday_cost__pd.Series]:
    return {'便宜期': intraday_cost___cheap(dates), '昂贵期': intraday_cost___expensive(dates), 'A_2015-09至2017-02': dates.between('2015-09-07', '2017-02-16'), 'B_2017-02至2019-04': dates.between('2017-02-17', '2019-04-21'), 'C_2019-04至2021': dates.between('2019-04-22', intraday_cost__RESEARCH_END)}

def intraday_cost__analyze(minute: intraday_cost__pd.DataFrame, oi: intraday_cost__pd.DataFrame) -> tuple[intraday_cost__pd.DataFrame, intraday_cost__pd.DataFrame, str]:
    main = intraday_cost__intraday_observations(minute)
    full = intraday_cost__intraday_observations(minute, full_session=True)
    daily = intraday_cost__daily_observations(minute)
    regressions = []
    interaction_rows = []
    for checkpoint in intraday_cost__CHECKPOINTS:
        part = main.loc[main['checkpoint'] == checkpoint].copy()
        part['cheap'] = intraday_cost___cheap(part['date'])
        part['expensive'] = intraday_cost___expensive(part['date'])
        for label, mask in (('便宜期', part['cheap']), ('昂贵期', part['expensive'])):
            sample = part.loc[mask]
            result = intraday_cost__fit(sample['return_bp'], sample['location'])
            regressions.append(intraday_cost___regression_row('T1_主时段', checkpoint, label, result))
        eligible = part.loc[part['cheap'] | part['expensive']].copy()
        cheap = eligible['cheap'].astype(float)
        interaction = eligible['location'] * cheap
        result = intraday_cost__fit(eligible['return_bp'], interaction, (eligible['location'], cheap))
        interaction_rows.append(result)
        regressions.append(intraday_cost___regression_row('T2_制度交互', checkpoint, '便宜期对昂贵期', result))
        for label, mask in intraday_cost___period_masks(part['date']).items():
            if label in ('便宜期', '昂贵期'):
                continue
            sample = part.loc[mask]
            result = intraday_cost__fit(sample['return_bp'], sample['location'])
            regressions.append(intraday_cost___regression_row('T4_分段斜率', checkpoint, label, result))
        crash_free = part.loc[part['date'] <= intraday_cost__CRASH_FREE_END]
        result = intraday_cost__fit(crash_free['return_bp'], crash_free['location'])
        regressions.append(intraday_cost___regression_row('T5_去股灾便宜期', checkpoint, '截至2015-05-29', result))
        full_part = full.loc[full['checkpoint'] == checkpoint].copy()
        for label, mask in (('便宜期', intraday_cost___cheap(full_part['date'])), ('昂贵期', intraday_cost___expensive(full_part['date']))):
            sample = full_part.loc[mask]
            result = intraday_cost__fit(sample['return_bp'], sample['location'])
            regressions.append(intraday_cost___regression_row('T5_完整期货时段', checkpoint, label, result))
        skip_result = intraday_cost__fit(part['skip_one_bp'], part['location'])
        regressions.append(intraday_cost___regression_row('T5_跳过检查点后一分钟', checkpoint, '全时期', skip_result))
        control_result = intraday_cost__fit(part['return_bp'], part['location'], (part['open_to_checkpoint'],))
        regressions.append(intraday_cost___regression_row('T5_控制开盘至检查点收益', checkpoint, '全时期', control_result))
    adjusted = intraday_cost___holm([intraday_cost___p_two_sided(row['HAC_t']) for row in interaction_rows])
    for checkpoint, adjusted_p in zip(intraday_cost__CHECKPOINTS, adjusted):
        for row in regressions:
            if row['检验'] == 'T2_制度交互' and row['检查点'] == checkpoint:
                row['双侧p'] = intraday_cost___p_two_sided(row['HAC_t'])
                row['Holm_p'] = adjusted_p
    p2_frame = daily.reset_index()
    p2_frame['date'] = intraday_cost__pd.to_datetime(p2_frame['date'])
    p2_frame['cheap'] = intraday_cost___cheap(p2_frame['date'])
    p2_frame['expensive'] = intraday_cost___expensive(p2_frame['date'])
    p2_frame = p2_frame.loc[p2_frame['cheap'] & (p2_frame['outcome_end'] <= intraday_cost__CHEAP_END) | p2_frame['expensive']].dropna(subset=['outcome_bp', 'outcome_end'])
    cheap_flag = p2_frame['cheap'].astype(float)
    result = intraday_cost__fit(p2_frame['outcome_bp'], p2_frame['clv'] * cheap_flag, (p2_frame['clv'], cheap_flag))
    regressions.append(intraday_cost___regression_row('T3_下一开盘段制度交互', '全天CLV', '便宜期对昂贵期', result))
    for label, mask in intraday_cost___period_masks(p2_frame['date']).items():
        sample = p2_frame.loc[mask]
        if sample.empty:
            continue
        result = intraday_cost__fit(sample['outcome_bp'], sample['clv'])
        regressions.append(intraday_cost___regression_row('T4_P2分段斜率', '全天CLV', label, result))
    years = main.copy()
    years['year'] = years['date'].dt.year
    annual = []
    for (checkpoint, year), sample in years.groupby(['checkpoint', 'year'], sort=True):
        if sample['date'].max() > intraday_cost__CHEAP_END:
            sample = sample.loc[sample['date'] <= intraday_cost__CHEAP_END]
        if sample.empty:
            continue
        result = intraday_cost__fit(sample['return_bp'], sample['location'])
        if sample['date'].max() <= intraday_cost__CHEAP_END:
            annual.append(intraday_cost___regression_row('T4_便宜期逐年', checkpoint, str(year), result))
    for year, group in minute.assign(trading_date=intraday_cost__pd.to_datetime(minute['trading_date']).dt.normalize()).groupby(intraday_cost__pd.to_datetime(minute['trading_date']).dt.year, sort=True):
        day_volume = group.groupby('trading_date')['volume'].sum()
        oi_year = oi.loc[intraday_cost__pd.to_datetime(oi['trading_date']).dt.year == year].copy()
        oi_year['trading_date'] = intraday_cost__pd.to_datetime(oi_year['trading_date']).dt.normalize()
        closing_oi = oi_year.groupby('trading_date')['open_interest'].last()
        joined = intraday_cost__pd.concat([day_volume.rename('volume'), closing_oi.rename('oi')], axis=1).dropna()
        ratio = joined['volume'] / joined['oi'].where(joined['oi'] > 0)
        annual.append({'检验': 'D1_成交量持仓量', '检查点': '', '样本': str(year), '天数': len(joined), '成交量中位数': float(joined['volume'].median()), '收盘持仓量中位数': float(joined['oi'].median()), '成交量持仓量比中位数': float(ratio.median())})
    verdict = intraday_cost___verdict(regressions)
    return (intraday_cost__pd.DataFrame(regressions), intraday_cost__pd.DataFrame(annual), verdict)

def intraday_cost___verdict(rows: list[dict]) -> str:
    t1 = {(row['检查点'], row['样本']): row for row in rows if row['检验'] == 'T1_主时段'}
    t2 = {row['检查点']: row for row in rows if row['检验'] == 'T2_制度交互'}
    t3 = next((row for row in rows if row['检验'] == 'T3_下一开盘段制度交互'), None)
    crash = {row['检查点']: row for row in rows if row['检验'] == 'T5_去股灾便宜期'}
    for checkpoint in intraday_cost__CHECKPOINTS:
        cheap = t1.get((checkpoint, '便宜期'), {})
        interaction = t2.get(checkpoint, {})
        crash_free = crash.get(checkpoint, {})
        if cheap.get('系数_bp', intraday_cost__np.nan) < 0 and cheap.get('HAC_t', intraday_cost__np.nan) <= -2 and (interaction.get('系数_bp', intraday_cost__np.nan) < 0) and (interaction.get('Holm_p', intraday_cost__np.nan) <= 0.05) and (t3 is not None) and (t3.get('系数_bp', intraday_cost__np.nan) > 0) and (crash_free.get('系数_bp', intraday_cost__np.nan) < 0):
            return '支持（按预登记规则；仍是事后提出的机制假说）'
    if all((t1.get((point, '便宜期'), {}).get('系数_bp', intraday_cost__np.nan) >= 0 for point in intraday_cost__CHECKPOINTS)):
        return '不支持：两个检查点的便宜期日内斜率均非负'
    if all((t2.get(point, {}).get('系数_bp', intraday_cost__np.nan) >= 0 for point in intraday_cost__CHECKPOINTS)):
        return '不支持：两个检查点的制度交互项均非负'
    return '不确定'

def intraday_cost__main() -> None:
    protocol = intraday_cost__load_protocol()
    minute, oi = intraday_cost__load_if_inputs()
    regressions, annual, verdict = intraday_cost__analyze(minute, oi)
    intraday_cost__OUT.mkdir(parents=True, exist_ok=True)
    regressions.to_csv(intraday_cost__OUT / 'regressions.csv', index=False, encoding='utf-8-sig')
    annual.to_csv(intraday_cost__OUT / 'annual_and_volume_oi.csv', index=False, encoding='utf-8-sig')
    (intraday_cost__OUT / 'protocol.json').write_text(intraday_cost__json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'IF 日内承接成本假说：{verdict}', flush=True)
    print(regressions.to_string(index=False), flush=True)
    print(annual.to_string(index=False), flush=True)
    print(f'结果：{intraday_cost__OUT}', flush=True)

def main() -> None:
    import argparse
    import warnings
    parser = argparse.ArgumentParser(description="Supplementary index economic-mechanism studies")
    parser.add_argument("study", choices=("liquidity", "positioning", "qvix", "intraday-cost"))
    parser.add_argument("--unabsorbed", action="store_true", help="Test selling not absorbed by top-20 members")
    args = parser.parse_args()
    if args.study in {"liquidity", "positioning"}:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if args.study == "liquidity":
                liquidity__run()
            else:
                positioning__run(unabsorbed=args.unabsorbed)
    elif args.study == "qvix":
        qvix__run()
    else:
        intraday_cost__main()


if __name__ == "__main__":
    main()
