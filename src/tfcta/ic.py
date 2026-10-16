"""因子预测性：截面相关、事前标准化和 Newey–West t。不是策略回测。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfcta import config as C
from tfcta.factors import exante_z


def _pair(factor: pd.Series, fwd: pd.Series) -> pd.DataFrame:
    return pd.concat([factor.rename('f'), fwd.rename('r')], axis=1).dropna()

def _by_period(series: pd.Series, period: str | None = None):
    """按自然月（或 ``period``）分组。"""
    period = C.IC_PERIOD if period is None else str(period)
    freq = {'ME': 'M', 'MS': 'M'}.get(period, period)
    return series.groupby(pd.DatetimeIndex(series.index).to_period(freq))

def _period_mean(series: pd.Series, min_obs: int, period: str | None = None) -> pd.Series:
    g = _by_period(series, period)
    out = g.mean().where(g.count() >= min_obs)
    if isinstance(out.index, pd.PeriodIndex):
        out.index = out.index.to_timestamp(how='end').normalize()
    return out

def _corr(df: pd.DataFrame, min_obs: int) -> float:
    if len(df) < int(min_obs):
        return np.nan
    if df['f'].nunique(dropna=True) < 2 or df['r'].nunique(dropna=True) < 2:
        return np.nan
    return float(df['f'].corr(df['r'], method='spearman'))

def spearman_ic(factor: pd.Series, fwd: pd.Series, min_obs: int) -> float:
    return _corr(_pair(factor, fwd), min_obs)

def summarize_ics(ics: list[float]) -> dict:
    """把一折内各品种的 IC 汇总成 (ic, t_cross, n_symbols)。"""
    arr = np.asarray([x for x in ics if np.isfinite(x)], dtype='float64')
    out = {'ic': np.nan, 't_cross': np.nan, 'n_symbols': int(arr.size)}
    if arr.size == 0:
        return out
    out['ic'] = float(arr.mean())
    if arr.size >= 2:
        sd = float(arr.std(ddof=1))
        if sd > 0:
            out['t_cross'] = float(out['ic'] / (sd / np.sqrt(arr.size)))
    return out

def horizon_of(fwd: pd.DataFrame, horizon: int | None = None) -> int:
    if horizon is not None:
        return int(horizon)
    return int(getattr(fwd, 'attrs', {}).get('horizon', 1))

def exante_scaled_return(fwd: pd.DataFrame,
                         window: int | None = None,
                         min_periods: int | None = None,
                         horizon: int | None = None) -> pd.DataFrame:
    """未来收益除以 t 日已知的同期限波动。"""
    window = C.IC_VOL_WINDOW if window is None else int(window)
    min_periods = C.IC_Z_MIN if min_periods is None else int(min_periods)
    h = horizon_of(fwd, horizon)
    vol = fwd.shift(h + 1).rolling(window, min_periods=min_periods).std()
    return fwd / vol.where(vol > 0)

def ic_period_series(factor: pd.DataFrame,
                     fwd: pd.DataFrame,
                     symbols: list[str],
                     min_obs: int | None = None,
                     period: str | None = None,
                     prepared: bool = False,
                     horizon: int | None = None) -> pd.Series:
    """逐期（默认逐月）的事前 IC 序列，时序显著性检验的输入。"""
    period = C.IC_PERIOD if period is None else str(period)
    min_obs = C.IC_PERIOD_MIN_OBS if min_obs is None else int(min_obs)
    if not prepared:
        factor = exante_z(factor)
        fwd = exante_scaled_return(fwd, horizon=horizon)
    cols = {}
    for s in symbols:
        if s not in factor.columns or s not in fwd.columns:
            continue
        df = _pair(factor[s], fwd[s])
        if df.empty:
            continue
        prod = df['f'] * df['r']
        cols[s] = _period_mean(prod, min_obs, period)
    if not cols:
        return pd.Series(dtype='float64')
    out = pd.DataFrame(cols).mean(axis=1, skipna=True).dropna()
    out.name = 'ic_period'
    return out

def nw_lag(n: int) -> int:
    """Newey-West 的自动截断滞后 floor(4·(n/100)^(2/9))。"""
    if n < 2:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))

def _nw_se(x: np.ndarray, lag: int) -> float:
    """样本均值的 Newey-West 标准误（Bartlett 权）。"""
    n = x.size
    d = x - x.mean()
    s = float(d @ d) / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * float(d[k:] @ d[:-k]) / n
    scale = float(x @ x) / n
    if not np.isfinite(s) or s <= 1e-16 * max(scale, 1.0):
        return np.nan
    return float(np.sqrt(s / n))

def timeseries_t(ic_series: pd.Series, lag: int | None = None) -> dict:
    """IC 时序均值的显著性：``ic_ts`` / ``t`` / ``ic_ir`` / ``n_periods`` / ``nw_lag``。"""
    x = np.asarray(pd.Series(ic_series, dtype='float64').dropna(), dtype='float64')
    out = {'ic_ts': np.nan, 't': np.nan, 'ic_ir': np.nan,
           'n_periods': int(x.size), 'nw_lag': 0}
    if x.size < C.IC_PERIOD_MIN_COUNT:
        return out
    k = nw_lag(x.size) if lag is None else int(lag)
    out['nw_lag'] = int(k)
    out['ic_ts'] = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd > 0:
        out['ic_ir'] = float(x.mean() / sd)
    se = _nw_se(x, k)
    if np.isfinite(se) and se > 0:
        out['t'] = float(x.mean() / se)
    return out

def sign_status(ic: float, name: str, t: float | None = None) -> str:
    """ok / flip / flip_weak / inconclusive / no_prior。"""
    if name not in C.PRIOR_FACTORS:
        return 'no_prior'
    ic = float(ic) if ic is not None else np.nan
    if not np.isfinite(ic) or ic == 0:
        return 'inconclusive'
    if np.sign(ic) == np.sign(C.FACTOR_SIGNS[name]):
        return 'ok'
    if t is None or not np.isfinite(t) or abs(float(t)) >= C.SIGN_T_MIN:
        return 'flip'
    return 'flip_weak'

IC_COLUMNS = ['factor', 'fold', 'ic', 'ic_ts', 't', 'ic_ir', 'n_periods',
              'nw_lag', 't_cross', 'n_symbols', 'n_folds', 'sign']

def _slice_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    return df.loc[idx.year == int(year)]

def factor_ic_table(factor: pd.DataFrame,
                    fwd: pd.DataFrame,
                    universe: dict,
                    name: str,
                    years: list[int],
                    min_obs: int | None = None,
                    horizon: int | None = None) -> pd.DataFrame:
    """逐折一行，最后再加一行 fold=mean_of_folds。"""
    min_obs = C.IC_MIN_OBS if min_obs is None else int(min_obs)
    z_all, r_all = exante_z(factor), exante_scaled_return(fwd, horizon=horizon)
    rows, series = [], []
    for y in years:
        y = int(y)
        syms = [s for s in universe.get(y, [])
                if s in factor.columns and s in fwd.columns]
        f_y, r_y = _slice_year(factor, y), _slice_year(fwd, y)
        rec = summarize_ics([spearman_ic(f_y[s], r_y[s], min_obs) for s in syms])
        ser = ic_period_series(_slice_year(z_all, y), _slice_year(r_all, y), syms,
                               prepared=True)
        series.append(ser)
        rec.update(timeseries_t(ser))
        rec.update(factor=name, fold=str(y), n_folds=1,
                   sign=sign_status(rec['ic_ts'], name, rec['t']))
        rows.append(rec)

    good = [r for r in rows if np.isfinite(r['ic'])]
    mean_ic = float(np.mean([r['ic'] for r in good])) if good else np.nan
    # n_symbols 取有效折的品种数均值。折之间品种池会变，所以累加得到的是
    # "品种×折"计数，写在 n_symbols 这一格会被读成品种数。折数单列 n_folds。
    n_sym = int(round(np.mean([r['n_symbols'] for r in good]))) if good else 0
    # 总的 t 值用**拼起来的逐期 IC 序列**算（2016-2021 约 72 个月），不是折间 t 的
    # 平均。折只有 6 个、每折 12 期，逐折 t 的自由度低得可怜；拼成一条长序列既提高
    # 自由度，也让 Newey-West 的滞后项真正吃到跨折的自相关。折与折在时间上不重叠，
    # 直接 concat 不会有重复索引。
    nonempty_series = [item for item in series if not item.empty]
    pooled = (pd.concat(nonempty_series).sort_index() if nonempty_series
              else pd.Series(dtype='float64'))
    tail = {'factor': name, 'fold': 'mean_of_folds', 'ic': mean_ic,
            't_cross': np.nan, 'n_symbols': n_sym, 'n_folds': len(good)}
    tail.update(timeseries_t(pooled))
    # 方向判定读拼起来那条长序列的 t（约 72 期），不是逐折 t。逐折只有 12 期，
    # 用它判方向会因为自由度太低而在 flip / flip_weak 之间反复摇摆。
    tail['sign'] = sign_status(tail['ic_ts'], name, tail['t'])
    rows.append(tail)
    return pd.DataFrame(rows).reindex(columns=IC_COLUMNS)

def cross_sectional_ic_table(factor: pd.DataFrame,
                             fwd: pd.DataFrame,
                             universe: dict,
                             name: str,
                             years: list[int],
                             min_symbols: int = 5,
                             min_days_per_period: int | None = None
                             ) -> pd.DataFrame:
    """每日跨品种 Spearman IC，按月平均后以时序 Newey-West t 检验。"""
    min_days = (C.IC_PERIOD_MIN_OBS if min_days_per_period is None
                else int(min_days_per_period))
    rows, monthly_series = [], []
    for year in years:
        year = int(year)
        symbols = [s for s in universe.get(year, [])
                   if s in factor.columns and s in fwd.columns]
        factor_year = _slice_year(factor, year).reindex(columns=symbols)
        return_year = _slice_year(fwd, year).reindex(columns=symbols)
        dates = factor_year.index.intersection(return_year.index).sort_values()
        daily_ic, daily_n = [], []
        for date in dates:
            pair = pd.concat([factor_year.loc[date], return_year.loc[date]], axis=1)
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            daily_n.append(len(pair))
            if (len(pair) < int(min_symbols) or pair.iloc[:, 0].nunique() < 2
                    or pair.iloc[:, 1].nunique() < 2):
                daily_ic.append(np.nan)
            else:
                daily_ic.append(float(pair.iloc[:, 0].corr(
                    pair.iloc[:, 1], method='spearman')))
        daily = pd.Series(daily_ic, index=dates, dtype='float64')
        monthly = _period_mean(daily, min_days).dropna()
        monthly_series.append(monthly)
        finite_daily = daily.dropna()
        rec = {
            'ic': float(finite_daily.mean()) if not finite_daily.empty else np.nan,
            't_cross': np.nan,
            'n_symbols': int(round(np.mean([n for n in daily_n if n])))
            if any(daily_n) else 0,
        }
        rec.update(timeseries_t(monthly))
        rec.update(factor=name, fold=str(year), n_folds=1,
                   sign='no_prior')
        rows.append(rec)

    valid_rows = [row for row in rows if np.isfinite(row['ic'])]
    nonempty_series = [item for item in monthly_series if not item.empty]
    pooled = (pd.concat(nonempty_series).sort_index() if nonempty_series
              else pd.Series(dtype='float64'))
    tail = {
        'factor': name,
        'fold': 'mean_of_folds',
        'ic': float(np.mean([row['ic'] for row in valid_rows]))
        if valid_rows else np.nan,
        't_cross': np.nan,
        'n_symbols': int(round(np.mean([row['n_symbols'] for row in valid_rows])))
        if valid_rows else 0,
        'n_folds': len(valid_rows),
        'sign': 'no_prior',
    }
    tail.update(timeseries_t(pooled))
    rows.append(tail)
    return pd.DataFrame(rows).reindex(columns=IC_COLUMNS)
