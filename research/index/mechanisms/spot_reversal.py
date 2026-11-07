from __future__ import annotations
# 合并早期现货反转与次日开盘压力检验；保留各自的协议与结果目录。
# 现货长期样本。
import json as spot_long__json
from pathlib import Path as spot_long__Path
import numpy as spot_long__np
import pandas as spot_long__pd
from research.time_factor_screen import ols_hac as spot_long__ols_hac
from tfcta import config as spot_long__C
from tfcta.data import daily_bars as spot_long__daily_bars
spot_long__CONFIG_PATH = spot_long__C.CONFIG_DIR / 'research' / 'index_spot_long_sample.json'
spot_long__OUT = spot_long__C.study_dir('index', 'index_spot_long_sample')
spot_long__EARLY_PATH = spot_long__C.INDEX_SPOT_DIR / 'spot_daily_2005_20150415.csv'
spot_long__LATE_PATH = spot_long__C.INDEX_SPOT_DIR / 'spot_daily.csv'
spot_long__IF_MINUTE = spot_long__C.INDEX_DATA_ROOT / 'if_before_ih_ic' / 'minute_shards' / 'IF.parquet'
spot_long__SYMBOLS = ('IF', 'IH', 'IC')
spot_long__LISTING = {'IH': '2004-01-02', 'IF': '2005-04-08', 'IC': '2007-01-15'}

def spot_long__load_protocol(path: spot_long__Path=spot_long__CONFIG_PATH) -> dict:
    protocol = spot_long__json.loads(path.read_text(encoding='utf-8'))
    parameters = protocol.get('parameters') or {}
    sample = protocol.get('sample', '')
    reading = protocol.get('reading') or {}
    if protocol.get('status') != 'declared_test_not_a_strategy' or '不读 2022' not in sample or '不读 2026' not in sample or ('不产生新策略' not in protocol.get('pass', '')) or ('不确定' not in reading.get('otherwise', '')) or (parameters.get('instruments') != list(spot_long__SYMBOLS)) or (parameters.get('listing') != spot_long__LISTING) or (parameters.get('threshold') != 0.4) or (parameters.get('early_start') != '2005-01-04') or (parameters.get('early_end') != '2015-04-15') or (parameters.get('late_start') != '2016-01-04') or (parameters.get('late_end') != '2021-12-31') or (parameters.get('overlap_start') != '2010-04-16') or (parameters.get('hac_lags') != 5) or (parameters.get('min_year_days') != 60) or (parameters.get('min_negative_years') != 8) or (parameters.get('open_equal_max_share') != 0.05) or (parameters.get('open_round') != 0.01):
        raise ValueError('现货长样本登记与实现或样本边界不一致')
    return protocol

def spot_long__read_spot(path: spot_long__Path, what: str) -> spot_long__pd.DataFrame:
    frame = spot_long__pd.read_csv(path, parse_dates=['date'])
    frame['date'] = spot_long__pd.to_datetime(frame['date']).dt.normalize()
    spot_long__C.assert_no_holdout_dates(frame['date'], what)
    if frame.duplicated(['symbol', 'date']).any():
        raise ValueError(f'{what} 同一指数的日期重复')
    return frame

def spot_long__close_location(frame: spot_long__pd.DataFrame) -> spot_long__pd.Series:
    """单个指数的收盘位置。高低价无效的日子留空。"""
    width = frame['high'] - frame['low']
    out = (frame['close'] - frame['low']) / width.where(width > 0)
    return out.where(frame[['open', 'high', 'low', 'close']].gt(0).all(axis=1))

def spot_long__outcomes_from(open_: spot_long__pd.Series, close: spot_long__pd.Series) -> spot_long__pd.DataFrame:
    """信号日 t 上的三种下一日收益。结果日落在序列外则为空。"""
    return spot_long__pd.DataFrame({'开盘到收盘': close.shift(-1) / open_.shift(-1) - 1.0, '开盘到再下一开盘': open_.shift(-2) / open_.shift(-1) - 1.0, '收盘到收盘': close.shift(-1) / close - 1.0}, index=open_.index)

def spot_long___symbol_frames(daily: spot_long__pd.DataFrame) -> dict[str, spot_long__pd.DataFrame]:
    frames = {}
    for symbol in spot_long__SYMBOLS:
        one = daily[daily['symbol'] == symbol].set_index('date').sort_index()
        one = one[one.index >= spot_long__pd.Timestamp(spot_long__LISTING[symbol])]
        one = one.copy()
        one['clv'] = spot_long__close_location(one)
        frames[symbol] = one
    return frames

def spot_long__spot_panel(daily: spot_long__pd.DataFrame, start: str, end: str) -> spot_long__pd.DataFrame:
    """信号日等权收盘位置，以及这些指数各自收益的等权。不先平均点位。"""
    start_ts, end_ts = (spot_long__pd.Timestamp(start), spot_long__pd.Timestamp(end))
    frames = spot_long___symbol_frames(daily)
    index = frames['IH'].index.union(frames['IF'].index).union(frames['IC'].index).sort_values()
    index = index[(index >= start_ts) & (index <= end_ts)]
    clv = spot_long__pd.DataFrame({symbol: frame['clv'].reindex(index) for symbol, frame in frames.items()})
    returns = {name: spot_long__pd.DataFrame(index=index, columns=list(spot_long__SYMBOLS), dtype='float64') for name in ('开盘到收盘', '开盘到再下一开盘', '收盘到收盘')}
    for symbol, frame in frames.items():
        own = spot_long__outcomes_from(frame['open'], frame['close']).reindex(index)
        for name in returns:
            returns[name][symbol] = own[name]
    available = clv.notna()
    out = spot_long__pd.DataFrame({'clv': clv.mean(axis=1, skipna=True), '只数': available.sum(axis=1).astype(int)}, index=index)
    for name, wide in returns.items():
        out[name] = wide.where(available).mean(axis=1, skipna=True)
    out = out.dropna(subset=['clv'])
    spot_long__C.assert_no_holdout_dates(out.index, '现货长样本信号日')
    return out

def spot_long__open_quality(daily: spot_long__pd.DataFrame, end: str, max_share: float, step: float) -> spot_long__pd.DataFrame:
    """上市日至截止日。开盘四舍五入后等于前收的占比超过门槛，则开盘价不可靠。"""
    rows = []
    for symbol in spot_long__SYMBOLS:
        one = daily[daily['symbol'] == symbol].set_index('date').sort_index()
        one = one[one.index >= spot_long__pd.Timestamp(spot_long__LISTING[symbol])]
        one = one[one.index <= spot_long__pd.Timestamp(end)]
        previous = one['close'].shift(1)
        comparable = previous.notna() & one['open'].notna()
        rounded_equal = spot_long__pd.Series(spot_long__np.round(one['open'] / step) == spot_long__np.round(previous / step), index=one.index) & comparable
        invalid = one[['open', 'high', 'low', 'close']].isna().any(axis=1) | one[['open', 'high', 'low', 'close']].le(0).any(axis=1) | one['high'].lt(one['low']) | one['open'].lt(one['low']) | one['open'].gt(one['high'])
        share = float(rounded_equal.sum() / comparable.sum()) if comparable.any() else spot_long__np.nan
        rows.append({'指数': symbol, '起始': one.index.min().date().isoformat() if len(one) else '', '截止': one.index.max().date().isoformat() if len(one) else '', '天数': int(len(one)), '无效行情天数': int(invalid.sum()), '开盘等于前收占比': share, '开盘价不可靠': bool(invalid.any() or share > max_share)})
    return spot_long__pd.DataFrame(rows)

def spot_long__fit_slope(y: spot_long__pd.Series, x: spot_long__pd.Series, lags: int) -> tuple[float, float, int]:
    frame = spot_long__pd.concat([y.rename('y'), x.rename('x')], axis=1).dropna()
    coef, tstat = spot_long__ols_hac(frame['y'].to_numpy(), [frame['x'].to_numpy()], lags)
    return (coef, tstat, int(len(frame)))

def spot_long__yearly_slopes(y: spot_long__pd.Series, x: spot_long__pd.Series, lags: int, min_days: int) -> spot_long__pd.DataFrame:
    rows = []
    for year in sorted(set(y.dropna().index.year)):
        mask = y.index.year == year
        coef, tstat, n = spot_long__fit_slope(y[mask], x[mask], lags)
        rows.append({'年份': int(year), '天数': n, '斜率': coef, 't': tstat, '符号': '−' if spot_long__np.isfinite(coef) and coef < 0 else '+' if spot_long__np.isfinite(coef) and coef > 0 else '', '计入年份条件': bool(n >= min_days and spot_long__np.isfinite(coef))})
    return spot_long__pd.DataFrame(rows)

def spot_long__slope_gap(early: spot_long__pd.DataFrame, late: spot_long__pd.DataFrame, column: str, lags: int) -> tuple[float, float, int]:
    """后段斜率减前段斜率：把收盘位置乘后段虚拟变量放在第一列。"""

    def piece(frame: spot_long__pd.DataFrame, is_late: float) -> spot_long__pd.DataFrame:
        out = frame[['clv', column]].dropna().copy()
        out['后段'] = is_late
        out['交乘'] = out['clv'] * is_late
        return out
    stacked = spot_long__pd.concat([piece(early, 0.0), piece(late, 1.0)], ignore_index=True)
    coef, tstat = spot_long__ols_hac(stacked[column].to_numpy(), [stacked['交乘'].to_numpy(), stacked['clv'].to_numpy(), stacked['后段'].to_numpy()], lags)
    return (coef, tstat, int(len(stacked)))

def spot_long__designated(quality: spot_long__pd.DataFrame) -> str:
    if quality['开盘价不可靠'].any():
        return '收盘到收盘'
    return '开盘到收盘'

def spot_long__stable(slope: float, tstat: float, yearly: spot_long__pd.DataFrame, min_years: int) -> bool:
    counted = yearly[yearly['计入年份条件']]
    negative = int((counted['斜率'] < 0).sum())
    return bool(spot_long__np.isfinite(slope) and slope < 0 and (tstat <= -2) and (negative >= min_years))

def spot_long__read_verdict(early_ok: bool, late_ok: bool, overlap_spot: float, if_slope: float, if_t: float) -> str:
    if_rebounds = bool(spot_long__np.isfinite(if_slope) and if_slope < 0 and (if_t <= -2))
    spot_overlap_rebounds = bool(spot_long__np.isfinite(overlap_spot) and overlap_spot < 0)
    if early_ok and spot_overlap_rebounds and (not if_rebounds):
        return '早期现货有反弹，IF 没有'
    if not early_ok and late_ok:
        return '2016 年以后才出现'
    return '不确定'

def spot_long__if_panel() -> spot_long__pd.DataFrame:
    if not spot_long__IF_MINUTE.is_file():
        raise FileNotFoundError('缺少 IF 上市至 IH/IC 上市前的分钟分片，不在这次重新下载')
    minute = spot_long__pd.read_parquet(spot_long__IF_MINUTE)
    minute['trading_date'] = spot_long__pd.to_datetime(minute['trading_date']).dt.normalize()
    spot_long__C.assert_no_holdout_dates(minute['trading_date'], 'IF 早期分钟')
    if minute['trading_date'].max() >= spot_long__pd.Timestamp('2015-04-17'):
        raise ValueError('IF 早期分钟越过 IH/IC 上市日的下一交易日')
    daily = spot_long__daily_bars(minute)
    width = daily['highw'] - daily['loww']
    clv = (daily['closew'] - daily['loww']) / width.where(width > 0)
    out = spot_long__pd.DataFrame({'clv': clv, '开盘到收盘': daily['close'].shift(-1) / daily['open'].shift(-1) - 1.0, '开盘到再下一开盘': (daily['openw'].shift(-2) - daily['openw'].shift(-1)) / daily['open'].shift(-1)})
    return out.sort_index()

def spot_long__run() -> None:
    protocol = spot_long__load_protocol()
    parameters = protocol['parameters']
    if not spot_long__EARLY_PATH.is_file():
        raise FileNotFoundError('缺少早期现货日线，先运行 python -m download.index.getIndexSpot --early')
    early_daily = spot_long__read_spot(spot_long__EARLY_PATH, '早期现货日线')
    if early_daily['date'].min().date().isoformat() < parameters['early_start']:
        raise ValueError('早期现货日线早于 2005-01-04')
    if early_daily['date'].max().date().isoformat() > parameters['early_end']:
        raise ValueError('早期现货日线晚于 2015-04-15')
    late_daily = spot_long__read_spot(spot_long__LATE_PATH, '研究期现货日线')
    late_daily = late_daily[late_daily['date'] <= spot_long__pd.Timestamp(parameters['late_end'])]
    quality = spot_long__open_quality(early_daily, parameters['early_end'], parameters['open_equal_max_share'], parameters['open_round'])
    column = spot_long__designated(quality)
    early = spot_long__spot_panel(early_daily, parameters['early_start'], parameters['early_end'])
    late = spot_long__spot_panel(late_daily, parameters['late_start'], parameters['late_end'])
    bridge = spot_long__spot_panel(late_daily, '2015-04-16', '2015-12-31')
    lags, min_days = (parameters['hac_lags'], parameters['min_year_days'])
    early_slope, early_t, early_n = spot_long__fit_slope(early[column], early['clv'], lags)
    late_slope, late_t, late_n = spot_long__fit_slope(late[column], late['clv'], lags)
    gap, gap_t, gap_n = spot_long__slope_gap(early, late, column, lags)
    early_years = spot_long__yearly_slopes(early[column], early['clv'], lags, min_days)
    late_years = spot_long__yearly_slopes(late[column], late['clv'], lags, min_days)
    early_years.insert(0, '样本', '2005–2015-04')
    late_years.insert(0, '样本', '2016–2021')
    futures = spot_long__if_panel()
    overlap_start = spot_long__pd.Timestamp(parameters['overlap_start'])
    common = early.index.intersection(futures.dropna(subset=['clv', '开盘到再下一开盘']).index)
    common = common[common >= overlap_start]
    spot_overlap = early.loc[common]
    fut_overlap = futures.loc[common]
    spot_o, spot_ot, spot_on = spot_long__fit_slope(spot_overlap[column], spot_overlap['clv'], lags)
    if_oc, if_oct, if_ocn = spot_long__fit_slope(fut_overlap['开盘到收盘'], fut_overlap['clv'], lags)
    if_oo, if_oot, if_oon = spot_long__fit_slope(fut_overlap['开盘到再下一开盘'], fut_overlap['clv'], lags)
    early_ok = spot_long__stable(early_slope, early_t, early_years, parameters['min_negative_years'])
    late_ok = bool(spot_long__np.isfinite(late_slope) and late_slope < 0 and (late_t <= -2))
    verdict = spot_long__read_verdict(early_ok, late_ok, spot_o, if_oo, if_oot)
    bridge_slope, bridge_t, bridge_n = spot_long__fit_slope(bridge[column], bridge['clv'], lags)

    def row(name, outcome, slope, tstat, n, note):
        return {'检验': name, '结果': outcome, '斜率': slope, 't': tstat, '天数': n, '说明': note}
    table = spot_long__pd.DataFrame([row('早期现货', column, early_slope, early_t, early_n, '被指定的结果'), row('2016年后现货', column, late_slope, late_t, late_n, '已看过的对照'), row('后段减前段', column, gap, gap_t, gap_n, '斜率差'), row('并排现货', column, spot_o, spot_ot, spot_on, '2010-04 至 2015-04'), row('IF开盘到收盘', '开盘到收盘', if_oc, if_oct, if_ocn, '同一批日期'), row('IF开盘到再下一开盘', '开盘到再下一开盘', if_oo, if_oot, if_oon, '同一批日期'), row('2015年4月16日后的桥', column, bridge_slope, bridge_t, bridge_n, '不进入解读')])
    for extra in ('开盘到收盘', '开盘到再下一开盘', '收盘到收盘'):
        if extra == column:
            continue
        slope, tstat, n = spot_long__fit_slope(early[extra], early['clv'], lags)
        note = '不解释' if column == '收盘到收盘' else '对照'
        table = spot_long__pd.concat([table, spot_long__pd.DataFrame([row(f'早期现货{extra}', extra, slope, tstat, n, note)])], ignore_index=True)
    spot_long__OUT.mkdir(parents=True, exist_ok=True)
    quality.to_csv(spot_long__OUT / 'open_quality.csv', index=False, encoding='utf-8-sig')
    table.to_csv(spot_long__OUT / 'slopes.csv', index=False, encoding='utf-8-sig')
    spot_long__pd.concat([early_years, late_years], ignore_index=True).to_csv(spot_long__OUT / 'yearly.csv', index=False, encoding='utf-8-sig')
    (spot_long__OUT / 'verdict.txt').write_text(verdict + '\n', encoding='utf-8')
    (spot_long__OUT / 'protocol.json').write_text(spot_long__json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(quality.to_string(index=False))
    print(table.round(6).to_string(index=False))
    print(early_years.round(4).to_string(index=False))
    print(late_years.round(4).to_string(index=False))
    print(verdict)
# 现货次日开盘压力。
import json as spot_pressure__json
from pathlib import Path as spot_pressure__Path
import numpy as spot_pressure__np
import pandas as spot_pressure__pd
from research.time_factor_screen import ols_hac as spot_pressure__ols_hac
from strategies.index_common import market_close_location as spot_pressure__market_close_location
from tfcta import config as spot_pressure__C
spot_pressure__CONFIG_PATH = spot_pressure__C.CONFIG_DIR / 'research' / 'index_spot_open_pressure.json'
spot_pressure__OUT = spot_pressure__C.study_dir('index', 'index_spot_open_pressure')
spot_pressure__MINUTE_PATH = spot_pressure__C.INDEX_SPOT_DIR / 'spot_minute.parquet'
spot_pressure__DAILY_PATH = spot_pressure__C.INDEX_SPOT_DIR / 'spot_daily.csv'
spot_pressure__SYMBOLS = ('IF', 'IH', 'IC')
spot_pressure__TAGS = ('P1', 'P2', 'P3', 'P4a', 'P4b', 'P4', 'P5', 'D_IF', 'D_IH', 'D_IC', 'R0')
spot_pressure__OUTCOME_FIELDS = ('open_30', 'rest', 'vol_share', 'open_30_IF', 'open_30_IH', 'open_30_IC', 'gap')

def spot_pressure__load_protocol(path: spot_pressure__Path=spot_pressure__CONFIG_PATH) -> dict:
    protocol = spot_pressure__json.loads(path.read_text(encoding='utf-8'))
    parameters = protocol.get('parameters') or {}
    sample = protocol.get('sample', '')
    if protocol.get('status') != 'declared_test_not_a_strategy' or '不读 2022' not in sample or '不读 2026' not in sample or (list(protocol.get('predictions', {})) != [tag for tag in spot_pressure__TAGS if tag != 'P4']) or (parameters.get('instruments') != list(spot_pressure__SYMBOLS)) or (parameters.get('open_bars') != 30) or (parameters.get('min_morning_bars') != 120) or (parameters.get('min_day_bars') != 230) or (parameters.get('hac_lags') != 5) or (parameters.get('primary_outcome_start') != '2016-01-04') or (parameters.get('outcome_end') != '2021-12-31') or (parameters.get('mediation_shrink') != 0.5) or (parameters.get('bar30_minute_min') != '09:55') or (parameters.get('bar30_minute_max') != '10:05') or (parameters.get('open_match_tolerance') != 0.0001) or (parameters.get('max_mismatch_share') != 0.01):
        raise ValueError('现货开盘卖压登记与实现或样本边界不一致')
    return protocol

def spot_pressure___clock_minute(stamp: spot_pressure__pd.Series) -> spot_pressure__pd.Series:
    stamp = spot_pressure__pd.to_datetime(stamp)
    return stamp.dt.hour * 60 + stamp.dt.minute

def spot_pressure__session_paths(minutes: spot_pressure__pd.DataFrame, open_bars: int=30, min_morning: int=120, min_day: int=230) -> spot_pressure__pd.DataFrame:
    """每个指数的完整交易日：开盘前 30 根的收益、其后收益、这 30 根的成交量占比。"""
    frame = minutes.copy()
    if 'datetime' not in frame.columns or 'symbol' not in frame.columns:
        raise KeyError('现货分钟需要 symbol 和 datetime')
    frame['datetime'] = spot_pressure__pd.to_datetime(frame['datetime'])
    spot_pressure__C.assert_no_holdout_dates(frame['datetime'], '现货指数分钟')
    frame['trading_date'] = frame['datetime'].dt.normalize()
    rows = []
    for (symbol, day), group in frame.groupby(['symbol', 'trading_date'], sort=True):
        group = group.sort_values('datetime')
        morning = group[group['datetime'].dt.hour < 12]
        if len(morning) < min_morning or len(group) < min_day:
            continue
        window = morning.iloc[:open_bars]
        open_px = float(window['open'].iloc[0])
        px30 = float(window['close'].iloc[-1])
        close_px = float(group['close'].iloc[-1])
        volume = float(group['volume'].sum())
        if open_px <= 0 or px30 <= 0 or close_px <= 0 or (volume <= 0):
            continue
        rows.append({'symbol': symbol, 'trading_date': spot_pressure__pd.Timestamp(day), 'open_30': px30 / open_px - 1.0, 'rest': close_px / px30 - 1.0, 'vol_share': float(window['volume'].sum()) / volume, 'day_open': open_px, 'bar30_time': window['datetime'].iloc[-1]})
    if not rows:
        raise ValueError('现货分钟没有完整交易日')
    out = spot_pressure__pd.DataFrame(rows)
    if out.duplicated(['symbol', 'trading_date']).any():
        raise ValueError('现货分钟路径有重复的品种-日期')
    return out

def spot_pressure__market_outcomes(paths: spot_pressure__pd.DataFrame) -> spot_pressure__pd.DataFrame:
    """三指数都有完整路径的日子，等权合成；任一指数缺失则该日不用。"""
    fields = ('open_30', 'rest', 'vol_share', 'day_open')
    wide = {field: paths.pivot(index='trading_date', columns='symbol', values=field).reindex(columns=list(spot_pressure__SYMBOLS)) for field in fields}
    out = spot_pressure__pd.DataFrame({field: frame.mean(axis=1, skipna=False) for field, frame in wide.items()})
    opens = wide['open_30']
    out['open_30_IF'] = opens['IF']
    out['open_30_IH'] = opens['IH']
    out['open_30_IC'] = opens['IC']
    out['ic_minus_ih'] = opens['IC'] - opens['IH']
    return out.dropna(how='any').sort_index()

def spot_pressure__with_prior_clv(outcomes: spot_pressure__pd.DataFrame, clv: spot_pressure__pd.Series, gap: spot_pressure__pd.Series) -> spot_pressure__pd.DataFrame:
    """结果日用期货交易日序列的前一天收盘位置。当天收盘位置不进回归。

    前一天可以没有完整的现货分钟路径（例如熔断提前收盘），只要期货收盘位置有值。
    """
    if clv.index.has_duplicates:
        raise ValueError('期货收盘位置日期必须唯一')
    clv = clv.sort_index()
    previous = spot_pressure__pd.Series(clv.index[:-1].to_numpy(), index=clv.index[1:])
    out = outcomes.loc[outcomes.index.isin(previous.index)].copy()
    if len(out) < 2:
        raise ValueError('现货路径与期货收盘位置的重叠不足')
    out['clv_prior'] = clv.reindex(previous.reindex(out.index).to_numpy()).to_numpy()
    out['gap'] = gap.reindex(out.index)
    spot_pressure__C.assert_no_holdout_dates(out.index, '现货开盘卖压结果日')
    return out.sort_index().dropna(subset=['clv_prior'])

def spot_pressure__check_bar_clock(paths: spot_pressure__pd.DataFrame, low: str='09:55', high: str='10:05') -> dict:
    """第 30 根若不是 10:00 前后，这 30 根就不是开盘半小时。"""
    low_m = int(low[:2]) * 60 + int(low[3:])
    high_m = int(high[:2]) * 60 + int(high[3:])
    summary = {}
    for symbol, group in paths.groupby('symbol'):
        minute = int(spot_pressure___clock_minute(group['bar30_time']).median())
        summary[symbol] = f'{minute // 60:02d}:{minute % 60:02d}'
        if not low_m <= minute <= high_m:
            raise ValueError(f'{symbol} 第 30 根时刻中位数是 {summary[symbol]}，不在 {low}–{high}')
    return summary

def spot_pressure__mismatched_opens(paths: spot_pressure__pd.DataFrame, daily: spot_pressure__pd.DataFrame, tolerance: float, max_share: float) -> spot_pressure__pd.DataFrame:
    """日线开盘对不上的品种日。中位数或占比超标说明窗口定义错了，直接拒绝。"""
    day = daily.rename(columns={'date': 'trading_date'})
    day['trading_date'] = spot_pressure__pd.to_datetime(day['trading_date']).dt.normalize()
    spot_pressure__C.assert_no_holdout_dates(day['trading_date'], '现货指数日线')
    merged = paths.merge(day[['symbol', 'trading_date', 'open']], on=['symbol', 'trading_date'], how='inner')
    if len(merged) != len(paths):
        raise ValueError('现货分钟路径有对不上现货日线的品种日')
    gap = (merged['day_open'] - merged['open']).abs() / merged['open']
    bad = merged.loc[gap > tolerance, ['symbol', 'trading_date']].assign(相对偏差=gap[gap > tolerance].to_numpy())
    if float(gap.median()) > tolerance or float((gap > tolerance).mean()) > max_share:
        raise ValueError(f'分钟开盘与日线开盘中位相对偏差 {float(gap.median()):.6g}，超标占比 {float((gap > tolerance).mean()):.4f}，窗口定义不对')
    return bad.reset_index(drop=True)

def spot_pressure__fit(y: spot_pressure__pd.Series, xs: list[spot_pressure__pd.Series], lags: int) -> tuple[float, float, float, int]:
    frame = spot_pressure__pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 60:
        return (spot_pressure__np.nan, spot_pressure__np.nan, spot_pressure__np.nan, len(frame))
    coef, tstat = spot_pressure__ols_hac(frame.iloc[:, 0].to_numpy(), [frame.iloc[:, i].to_numpy() for i in range(1, frame.shape[1])], lags)
    se = abs(coef / tstat) if tstat else spot_pressure__np.nan
    return (coef, tstat, se, len(frame))

def spot_pressure__verdict(value, tstat, expected: str) -> str:
    if spot_pressure__pd.isna(tstat):
        return '不确定'
    right = value > 0 if expected == '>0' else value < 0
    if abs(tstat) >= 2:
        return '支持' if right else '否定'
    return '不确定'

def spot_pressure__yearly_signs(y: spot_pressure__pd.Series, x: spot_pressure__pd.Series, lags: int) -> str:
    signs = []
    for year in sorted(set(y.dropna().index.year)):
        mask = y.index.year == year
        coef = spot_pressure__fit(y[mask], [x[mask]], lags)[0]
        signs.append(f"{year}:{('+' if coef > 0 else '−' if coef < 0 else '?')}")
    return ' '.join(signs)

def spot_pressure__channel_verdict(rest_on_open: float, rest_on_open_t: float, controlled: float, alone: float, shrink: float) -> str:
    """开盘卖压解释后段：后段顺着开盘收益走，且收盘位置斜率至少掉一半。"""
    if spot_pressure__pd.isna(rest_on_open_t) or spot_pressure__pd.isna(alone) or alone == 0 or spot_pressure__pd.isna(controlled):
        return '不确定'
    open_supports = rest_on_open < 0 and rest_on_open_t <= -2
    shrunk = abs(controlled) <= shrink * abs(alone) and alone < 0
    if open_supports and shrunk:
        return '支持'
    if rest_on_open > 0 and rest_on_open_t >= 2 or (alone < 0 and abs(controlled) > abs(alone)):
        return '否定'
    return '不确定'

def spot_pressure__tests(panel: spot_pressure__pd.DataFrame, lags: int, shrink: float) -> spot_pressure__pd.DataFrame:
    clv = panel['clv_prior']
    specs = [('P1', '开盘半小时收益 ~ 前一日收盘位置', 'open_30', [clv], '>0', True), ('P2', '其后收益 ~ 前一日收盘位置', 'rest', [clv], '<0', True), ('P3', '开盘成交量占比 ~ 前一日收盘位置', 'vol_share', [clv], '<0', True), ('P4a', '其后收益 ~ 开盘半小时收益 + 前一日收盘位置', 'rest', [panel['open_30'], clv], '<0', False), ('P4b', '其后收益 ~ 前一日收盘位置 + 开盘半小时收益', 'rest', [clv, panel['open_30']], '<0', False), ('P5', '中证500减上证50的开盘半小时 ~ 前一日收盘位置', 'ic_minus_ih', [clv], '>0', True), ('D_IF', '沪深300开盘半小时 ~ 前一日收盘位置', 'open_30_IF', [clv], '>0', False), ('D_IH', '上证50开盘半小时 ~ 前一日收盘位置', 'open_30_IH', [clv], '>0', False), ('D_IC', '中证500开盘半小时 ~ 前一日收盘位置', 'open_30_IC', [clv], '>0', False), ('R0', '隔夜跳空 ~ 前一日收盘位置（已知复述）', 'gap', [clv], '>0', True)]
    rows = []
    fitted = {}
    for tag, label, target, xs, expected, signs in specs:
        coef, tstat, se, nobs = spot_pressure__fit(panel[target], xs, lags)
        fitted[tag] = (coef, tstat)
        rows.append({'预测': tag, '内容': label, '系数': coef, 'NW_t': tstat, '天数': nobs, '预期': expected, '判断': spot_pressure__verdict(coef, tstat, expected), '逐年符号': spot_pressure__yearly_signs(panel[target], xs[0], lags) if signs else ''})
    p2, p4a, p4b = (fitted['P2'], fitted['P4a'], fitted['P4b'])
    ratio = abs(p4b[0]) / abs(p2[0]) if p2[0] else spot_pressure__np.nan
    rows.append({'预测': 'P4', '内容': '控制开盘半小时后，收盘位置斜率绝对值相对 P2 的比例', '系数': ratio, 'NW_t': spot_pressure__np.nan, '天数': int(panel[['rest', 'open_30', 'clv_prior']].dropna().shape[0]), '预期': f'<={shrink} 且 P4a 支持', '判断': spot_pressure__channel_verdict(p4a[0], p4a[1], p4b[0], p2[0], shrink), '逐年符号': ''})
    return spot_pressure__pd.DataFrame(rows)

def spot_pressure__quintiles(panel: spot_pressure__pd.DataFrame) -> spot_pressure__pd.DataFrame:
    group = spot_pressure__pd.qcut(panel['clv_prior'], 5, labels=False) + 1
    rows = []
    for name, part in panel.groupby(group):
        rows.append({'分组': int(name), '天数': len(part), '前一日收盘位置': float(part['clv_prior'].mean()), '开盘半小时bp': float(part['open_30'].mean() * 10000.0), '其后bp': float(part['rest'].mean() * 10000.0), '成交量占比': float(part['vol_share'].mean()), '隔夜bp': float(part['gap'].mean() * 10000.0)})
    return spot_pressure__pd.DataFrame(rows)

def spot_pressure__futures_clv() -> spot_pressure__pd.Series:
    from tfcta.data import load_daily_bars
    bars = load_daily_bars(list(spot_pressure__SYMBOLS), directory=spot_pressure__C.INDEX_RESEARCH_DIR)
    width = bars['highw'] - bars['loww']
    ok = (width.gt(0) & bars['closew'].notna() & bars['highw'].notna() & bars['loww'].notna()).all(axis=1)
    index = ok.index[ok]
    index = index[(index >= spot_pressure__pd.Timestamp('2015-04-16')) & (index <= spot_pressure__pd.Timestamp('2021-12-31'))]
    return spot_pressure__market_close_location(bars, index)

def spot_pressure__spot_gap(daily: spot_pressure__pd.DataFrame) -> spot_pressure__pd.Series:
    day = daily.copy()
    day['date'] = spot_pressure__pd.to_datetime(day['date']).dt.normalize()
    spot_pressure__C.assert_no_holdout_dates(day['date'], '现货指数日线')
    wide_open = day.pivot(index='date', columns='symbol', values='open').reindex(columns=list(spot_pressure__SYMBOLS))
    wide_close = day.pivot(index='date', columns='symbol', values='close').reindex(columns=list(spot_pressure__SYMBOLS))
    gap = wide_open / wide_close.shift(1) - 1.0
    return gap.mean(axis=1, skipna=False).rename('gap')

def spot_pressure__run() -> None:
    protocol = spot_pressure__load_protocol()
    parameters = protocol['parameters']
    if not spot_pressure__MINUTE_PATH.is_file():
        raise FileNotFoundError('缺少现货分钟线，先运行 python -m download.index.getIndexSpot --minute')
    minutes = spot_pressure__pd.read_parquet(spot_pressure__MINUTE_PATH)
    spot_pressure__C.assert_no_holdout_dates(spot_pressure__pd.to_datetime(minutes['datetime']), '现货指数分钟')
    daily = spot_pressure__pd.read_csv(spot_pressure__DAILY_PATH, parse_dates=['date'])
    paths = spot_pressure__session_paths(minutes, parameters['open_bars'], parameters['min_morning_bars'], parameters['min_day_bars'])
    clocks = spot_pressure__check_bar_clock(paths, parameters['bar30_minute_min'], parameters['bar30_minute_max'])
    bad = spot_pressure__mismatched_opens(paths, daily, parameters['open_match_tolerance'], parameters['max_mismatch_share'])
    if len(bad):
        marked = paths.merge(bad.assign(_bad=1), on=['symbol', 'trading_date'], how='left')
        paths = marked.loc[marked['_bad'].isna()].drop(columns=['_bad', '相对偏差'])
    outcomes = spot_pressure__market_outcomes(paths)
    panel = spot_pressure__with_prior_clv(outcomes, spot_pressure__futures_clv(), spot_pressure__spot_gap(daily))
    panel = panel[panel.index <= spot_pressure__pd.Timestamp(parameters['outcome_end'])]
    lags, shrink = (parameters['hac_lags'], parameters['mediation_shrink'])
    full = spot_pressure__tests(panel, lags, shrink)
    primary = spot_pressure__tests(panel[panel.index >= spot_pressure__pd.Timestamp(parameters['primary_outcome_start'])], lags, shrink)
    full.insert(0, '样本', f'{panel.index.min().date()}..{panel.index.max().date()}')
    primary.insert(0, '样本', f"{parameters['primary_outcome_start']}..{panel.index.max().date()}")
    table = spot_pressure__pd.concat([primary, full], ignore_index=True)
    q_primary = spot_pressure__quintiles(panel[panel.index >= parameters['primary_outcome_start']])
    q_primary.insert(0, '样本', '2016-01-04 起')
    q_full = spot_pressure__quintiles(panel)
    q_full.insert(0, '样本', '全样本')
    spot_pressure__OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(spot_pressure__OUT / 'predictions.csv', index=False, encoding='utf-8-sig')
    spot_pressure__pd.concat([q_primary, q_full], ignore_index=True).to_csv(spot_pressure__OUT / 'quintiles.csv', index=False, encoding='utf-8-sig')
    panel.to_csv(spot_pressure__OUT / 'daily_panel.csv', encoding='utf-8-sig')
    kept = paths.groupby('symbol')['trading_date'].agg(['min', 'max', 'count'])
    desc = {'第30根时刻中位数': clocks, '开盘价对不上日线的品种日': [{'symbol': row.symbol, 'trading_date': str(spot_pressure__pd.Timestamp(row.trading_date).date()), '相对偏差': float(row.相对偏差)} for row in bad.itertuples(index=False)], '完整交易日': {symbol: {'min': str(row['min'].date()), 'max': str(row['max'].date()), 'days': int(row['count'])} for symbol, row in kept.iterrows()}, '进入回归的结果日': int(len(panel)), '主样本结果日': int((panel.index >= parameters['primary_outcome_start']).sum())}
    (spot_pressure__OUT / 'protocol.json').write_text(spot_pressure__json.dumps({'protocol': protocol, 'descriptive': desc}, ensure_ascii=False, indent=2), encoding='utf-8')
    spot_pressure__pd.set_option('display.width', 220)
    spot_pressure__pd.set_option('display.max_colwidth', 60)
    print(table.round(4).to_string(index=False), flush=True)
    print(spot_pressure__pd.concat([q_primary, q_full], ignore_index=True).round(2).to_string(index=False), flush=True)
    print(spot_pressure__json.dumps(desc, ensure_ascii=False, indent=2), flush=True)
    print(f'结果：{spot_pressure__OUT}', flush=True)

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Index spot weak-close reversal research")
    parser.add_argument("study", choices=("long-sample", "open-pressure"))
    args = parser.parse_args()
    (spot_long__run if args.study == "long-sample" else spot_pressure__run)()


designated = spot_long__designated
load_protocol = spot_long__load_protocol
open_quality = spot_long__open_quality
spot_panel = spot_long__spot_panel


if __name__ == "__main__":
    main()
