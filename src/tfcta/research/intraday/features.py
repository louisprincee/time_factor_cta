"""One scheduled day-session decision per symbol/day; features use its observed prefix.

Minute timestamps label bar ends. Entry uses the following bar's open and exit
uses a predeclared wall-clock bar's close, never an outcome-dependent stop.
"""
import numpy as np
import pandas as pd

from ... import config as C
from ...factors import intraday as I

TIME_FEATURES = ('duration_last', 'prefix_dfp_top3', 'prefix_high_time', 'prefix_low_time')
PRICE_FEATURES = ('ret_5', 'ret_15', 'range_ratio', 'close_location',
                  'close_vwap_proxy_dev', 'realized_vol', 'volume_last5_share')
FEATURES = TIME_FEATURES + PRICE_FEATURES
MINUTE_COLUMNS = ['open', 'close', 'closew', 'highw', 'loww', 'volume', 'trading_date']
SAMPLE_COLUMNS = ['symbol', 'trading_date', 'decision_time', 'entry_time', 'exit_time',
                  'reference_price', 'entry_price', 'exit_price', 'gross_return'] + list(FEATURES)


def samples_from_minutes(symbol, minute, decision_minutes=30, hold_minutes=30,
                         lookback=250, pct=55, eligible_years=None):
    """Decision at 09:00 + decision_minutes; prefix includes earlier night bars.

    Lookback warmup may precede 2016; only 2016–2021 samples are emitted. DFP
    unavailable before threshold warmup stays NaN for train-only imputation.
    Missing scheduled fills fail explicitly; never replace them by future bars.
    """
    if (not isinstance(decision_minutes, int) or not 1 <= decision_minutes <= 120
            or not isinstance(hold_minutes, int) or hold_minutes < 1
            or not isinstance(lookback, int) or lookback < 1 or not 0 < pct < 100):
        raise ValueError('无效决策时间、持有分钟或持续期阈值参数')
    if not isinstance(minute.index, pd.DatetimeIndex) or minute.index.has_duplicates:
        raise ValueError('分钟索引必须为不重复的时间戳')
    C.assert_no_holdout_dates(pd.to_datetime(minute.trading_date), '日内 ML 分片')
    source = minute.sort_index(kind='mergesort').copy()
    source['trading_date'] = pd.to_datetime(source.trading_date).dt.normalize()
    codes, _ = I.day_codes_of(source)
    if np.any(np.diff(codes) < 0):
        raise ValueError('交易日顺序与分钟顺序不一致')
    source['raw_close'] = I.raw_price_path(source)
    source['raw_open'] = I.raw_price_path(pd.DataFrame({'close':source.open}, index=source.index))
    thresholds = I.rolling_threshold(I.intraday_abs_diff(source.raw_close.to_numpy(), codes),
                                     codes, lookback, pct)
    rows = []
    for code, day in source.groupby(codes, sort=True):
        td = day.trading_date.iloc[0]
        if td < pd.Timestamp(C.STUDY_START) or (eligible_years is not None and td.year not in eligible_years):
            continue
        decision = td + pd.Timedelta(9*60+decision_minutes, unit='min')
        entry = decision + pd.Timedelta(1, unit='min')
        exit_time = decision + pd.Timedelta(hold_minutes, unit='min')
        if exit_time > td + pd.Timedelta(15, unit='h'):
            raise ValueError('持有期超过当日日盘收盘')
        if decision not in day.index:
            continue  # No decision bar: this symbol's allocated weight stays cash.
        if entry not in day.index or exit_time not in day.index:
            raise ValueError(f'{symbol} {td.date()} 预定成交分钟缺失（可能落在休市时段）')
        prefix = day.loc[:decision]
        p = prefix.raw_close.to_numpy(dtype=float)
        adjusted = prefix.closew.to_numpy(dtype=float)
        high, low = prefix.highw.to_numpy(dtype=float), prefix.loww.to_numpy(dtype=float)
        volume = prefix.volume.to_numpy(dtype=float)
        prices = [p[-1], float(day.loc[entry, 'raw_open']), float(day.loc[exit_time, 'raw_close'])]
        if not np.isfinite(prices).all() or min(prices) <= 0:
            raise ValueError(f'{symbol} {td.date()} 决策或预定成交价格无效')
        if (not np.isfinite(np.r_[p, adjusted, high, low, volume]).all()
                or (p <= 0).any() or (volume < 0).any()):
            raise ValueError(f'{symbol} {td.date()} 决策前分钟数据无效')
        duration = I.duration_one_day(p, thresholds.loc[code])
        good = np.isfinite(duration)
        dfp = ((p[np.argsort(-duration, kind='mergesort')[:3]].mean()-p[-1])/p[-1]
               if good.any() else np.nan)
        width = high.max()-low.min()
        total_volume = volume.sum()
        rows.append(dict(symbol=symbol, trading_date=td, decision_time=decision,
            entry_time=entry, exit_time=exit_time, reference_price=prices[0],
            entry_price=prices[1], exit_price=prices[2], gross_return=prices[2]/prices[1]-1,
            duration_last=duration[-1], prefix_dfp_top3=dfp,
            prefix_high_time=np.argmax(high)/max(len(p)-1, 1),
            prefix_low_time=np.argmin(low)/max(len(p)-1, 1),
            ret_5=(p[-1]-p[max(0,len(p)-6)])/p[max(0,len(p)-6)],
            ret_15=(p[-1]-p[max(0,len(p)-16)])/p[max(0,len(p)-16)],
            range_ratio=width/p[-1], close_location=(adjusted[-1]-low.min())/width if width>0 else 0.5,
            close_vwap_proxy_dev=p[-1]/np.average(p, weights=volume)-1 if total_volume>0 else np.nan,
            realized_vol=np.std(np.diff(p)/p[:-1]) if len(p)>1 else 0.,
            volume_last5_share=volume[-5:].sum()/total_volume if total_volume>0 else np.nan))
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)
