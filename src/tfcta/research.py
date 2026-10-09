"""研究评估：绩效、成本、回测和运行流程。"""
from __future__ import annotations

import hashlib
import json
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tfcta import config as C
from tfcta.data import bars as B
from tfcta.data import shard_io
from tfcta.data import universe as U
from tfcta.factors import library, cache, exante_z


def run_dir(step: str) -> Path:
    d = C.RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{step}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_json(o):
    """numpy 标量转 Python，NaN/inf 转 None。"""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2), encoding='utf-8')


MULTIPLIER: dict[str, float] = {
    "A": 10, "B": 10, "C": 10, "CS": 10, "M": 10, "Y": 10, "P": 10, "JD": 10,
    "L": 5, "V": 5, "PP": 5, "EG": 10, "I": 100, "J": 100, "JM": 60,
    "CU": 5, "AL": 5, "ZN": 5, "PB": 5, "NI": 1, "SN": 1, "AU": 1000, "AG": 15,
    "RB": 10, "HC": 10, "RU": 10, "BU": 10, "FU": 10, "SP": 10, "SC": 1000,
    "AP": 10, "CF": 5, "SR": 10, "TA": 5, "RM": 10, "OI": 10, "FG": 20,
    "SF": 5, "SM": 5, "MA": 10, "ZC": 100,
}


def fee_path(partition):
    paths = {"research": C.RESEARCH_OUT_DIR / "fee_history.csv",
             "validation_2022": C.DATA_ROOT / "validation_2022" / "fee_history.csv",
             "oos": C.DATA_ROOT / "oos" / "fee_history.csv"}
    if partition not in paths:
        raise C.HoldoutViolation(f"未知手续费分区 {partition}")
    return paths[partition]


def load_fees(partition="research"):
    path = fee_path(partition)
    if partition == "oos":
        end = pd.Timestamp(C.final_evaluation_end())  # 未打开最终评估时这里就拒绝
    table = pd.read_csv(path, parse_dates=["trading_date"])
    if partition == "oos":
        table = table[table.trading_date <= end]
        C.assert_strict_oos_dates(table.trading_date, what="手续费缓存")
    else:
        guard = C.assert_no_holdout_dates if partition == "research" else C.assert_validation_2022_dates
        guard(table.trading_date, what="手续费缓存")
    if table.duplicated(["symbol","trading_date"]).any():
        raise ValueError("手续费缓存有重复品种/日期")
    return table


def fee_tables(table, prices):
    """开仓费和隔夜平仓费，按当日原始名义金额的比例计算。"""
    opened, closed, roll_closed = [pd.DataFrame(np.nan, index=prices.index,
        columns=prices.columns) for _ in range(3)]
    for symbol in prices.columns:
        sub = table[table.symbol == symbol].set_index("trading_date").sort_index()
        if sub.empty:
            raise KeyError(f"手续费缓存缺 {symbol}")
        if sub.index.has_duplicates:
            raise ValueError(f"手续费缓存 {symbol} 日期重复")
        q = sub.reindex(prices.index, method="ffill")  # never backfill
        notional = prices[symbol] * MULTIPLIER[symbol]
        def rate(source, field):
            money = source.commission_type.eq("by_money")
            valid = source.commission_type.isin(["by_money", "by_volume"])
            values = source[field]
            result = np.where(money, values*1.01, (values+.01)/notional)
            # Even zero ad-valorem exchange fee has broker's .01 yuan fee.
            result = np.where(money & values.eq(0), .01/notional, result)
            return pd.Series(result, index=prices.index).where(valid)
        opened[symbol] = rate(q, "open_commission")
        closed[symbol] = rate(q, "close_commission")
        # The previous dominant rate is only substituted when contract changes.
        old = q.shift(1).where(q.contract.ne(q.contract.shift(1)), q)
        roll_closed[symbol] = rate(old, "close_commission")
    return opened, closed, roll_closed


def load_ticks():
    table = pd.read_csv(C.RESEARCH_OUT_DIR / "tick_size.csv")
    C.assert_no_holdout_dates(pd.to_datetime(table.year.astype(str)+"-01-01"), "tick 缓存")
    return table


def slippage_tables(ticks, prices, n_ticks=1.):
    """用上一年的最小变动价位除以成交日的原始开盘价。"""
    result = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for symbol in prices.columns:
        sub = ticks[ticks.symbol.eq(symbol)].set_index("year").tick.sort_index()
        if sub.empty:
            raise KeyError(f"tick 缓存缺 {symbol}")
        # Remove float32 quote noise in the existing estimates.
        sub = sub.round(2)
        years = sorted(set(prices.index.year) | set(sub.index+1))
        prior = sub.rename(index=lambda y: int(y)+1).reindex(years).ffill()
        result[symbol] = prior.reindex(prices.index.year).to_numpy()/prices[symbol] * n_ticks
    return result


def rebalance(signal, days=1, mode="single", phase=0):
    """收盘目标仓位：单一周期组合，或等权错开的几组。"""
    if int(days) != days or days < 1 or mode not in ("single","staggered"):
        raise ValueError("调仓周期须为正整数，模式为 single/staggered")
    if not 0 <= phase < days:
        raise ValueError("调仓相位越界")
    order = np.arange(len(signal)) % days
    if mode == "staggered":
        return sum(rebalance(signal, days, "single", k) for k in range(days))/days
    scheduled = pd.Series(order == phase, index=signal.index)
    return signal.fillna(0.).where(scheduled, axis=0).ffill().fillna(0.)


def position(signal, vol=None, vol_target=.20, cap=1., days=1, mode="single", phase=0):
    if vol_target:
        if vol is None:
            raise ValueError("波动率目标需要事前波动率")
        ann = vol.reindex_like(signal)*np.sqrt(252)
        signal = signal * (vol_target/ann.where(ann>0)).clip(upper=cap)
    return rebalance(signal.clip(-cap,cap), days, mode, phase).shift(1).fillna(0.)


def allocate(position, universe):
    out = position*0.
    for year, members in universe.items():
        members = list(dict.fromkeys(members))
        present = [s for s in members if s in position.columns]
        if members:
            mask = position.index.year == int(year)
            out.loc[mask,present] = position.loc[mask,present].fillna(0.)/len(members)
    return out.fillna(0.)


def trade_legs(exposure, rolls=None):
    previous = exposure.shift(1).fillna(0.)
    same = exposure*previous > 0
    retained = pd.DataFrame(np.minimum(exposure.abs(), previous.abs()),
        index=exposure.index, columns=exposure.columns).where(same,0.)
    if rolls is not None:
        retained = retained.where(~rolls.reindex_like(exposure).fillna(False).astype(bool),0.)
    return exposure.abs()-retained, previous.abs()-retained


def backtest(position, returns, universe, open_fee=0., close_fee=0., slippage=0.,
             rolls=None, roll_close_fee=None):
    position = position.reindex_like(returns)
    exposure = allocate(position, universe)
    opened, closed = trade_legs(exposure, rolls)
    if ((exposure != 0) & ~np.isfinite(returns)).any().any():
        raise ValueError("持仓对应的市场收益缺失，不能跳过品种或填零")
    def cost(size, rate):
        if isinstance(rate,pd.DataFrame):
            rate = rate.reindex_like(size)
        values = (size*rate).where(size>0,0.)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("实际成交对应的成本缺失")
        return values
    closing = close_fee
    if rolls is not None and roll_close_fee is not None:
        closing = close_fee.where(~rolls.reindex_like(exposure).fillna(False), roll_close_fee)
    charge = cost(opened,open_fee)+cost(closed,closing)+cost(opened+closed,slippage)
    gross = (exposure*returns.fillna(0.)).sum(axis=1)
    return pd.DataFrame({"gross":gross, "net":gross-charge.sum(axis=1),
        "turnover":(opened+closed).sum(axis=1), "cost":charge.sum(axis=1)})


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


PERIODS = 252
METRIC_KEYS = ['ann_return', 'ann_vol', 'ret_risk', 'calmar', 'win_rate', 'max_drawdown']


def _empty(n: int) -> dict:
    out = {k: np.nan for k in METRIC_KEYS}
    out['n_days'] = int(n)
    return out


def performance(ret: pd.Series, periods: int = PERIODS) -> dict:
    r = pd.Series(ret, dtype='float64').dropna()
    n = int(len(r))
    if n < 2:
        return _empty(n)
    growth = (1.0 + r).to_numpy()
    if np.any(growth <= 0):
        return _empty(n)
    nav = np.cumprod(growth)
    ann_return = float(nav[-1] ** (periods / n) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods))
    ret_risk = ann_return / ann_vol if ann_vol > 0 else np.nan
    peak = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    dd = nav / peak - 1.0
    max_dd = float(dd.min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    win = float((r.to_numpy() > 0).mean())
    return {
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'ret_risk': float(ret_risk) if np.isfinite(ret_risk) else np.nan,
        'calmar': float(calmar) if np.isfinite(calmar) else np.nan,
        'win_rate': win,
        'max_drawdown': max_dd,
        'n_days': n,
    }


def sharpe_ratio(returns: pd.Series, periods: int = PERIODS) -> float:
    """日均值 / 日标准差 × sqrt(252)，无风险利率 0。"""
    values = pd.Series(returns, dtype='float64').dropna()
    if len(values) < 2:
        return np.nan
    vol = float(values.std(ddof=1))
    if not np.isfinite(vol) or vol <= 0:
        return np.nan
    return float(values.mean() / vol * np.sqrt(periods))


def deflated_sharpe(returns: pd.Series, trial_sharpes, n_trials: int | None = None) -> dict:
    """Bailey & López de Prado (2014) 的 DSR：扣掉“试了 n_trials 次取最好”带来的期望最大夏普后，
    观测夏普仍大于 0 的概率。trial_sharpes 是各次尝试的年化夏普（用来估计尝试之间的离散程度）。"""
    from scipy.stats import kurtosis, norm, skew
    r = pd.Series(returns, dtype='float64').dropna()
    trials = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype='float64')
    n = int(n_trials or len(trials))
    if len(r) < 30 or len(trials) < 2 or n < 2:
        return {'dsr': np.nan, 'sr0': np.nan, 'sr': np.nan, 'n_trials': n}
    scale = np.sqrt(PERIODS)
    sr = r.mean() / r.std(ddof=1)                      # 日频，不年化
    spread = np.std(trials / scale, ddof=1)
    gamma = 0.5772156649
    sr0 = spread * ((1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * np.e)))
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    z = (sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return {'dsr': float(norm.cdf(z)), 'sr0': float(sr0 * scale), 'sr': float(sr * scale), 'n_trials': n}


def pbo_cscv(matrix: pd.DataFrame, blocks: int = 16) -> dict:
    """Bailey 等 (2017) 的组合对称交叉验证：把日收益按时间切成 blocks 段，每次取一半做样本内选夏普最高的方案，
    看它在另一半里排在第几。PBO = 样本内最好的方案在样本外排到中位数以下的比例。"""
    from itertools import combinations
    m = pd.DataFrame(matrix).dropna(how='all').fillna(0.0)
    if blocks % 2 or m.shape[1] < 2 or len(m) < blocks * 5:
        raise ValueError("需要偶数段、至少两个方案、每段至少 5 天")
    edges = np.linspace(0, len(m), blocks + 1).astype(int)
    parts = [m.iloc[a:b].to_numpy() for a, b in zip(edges[:-1], edges[1:])]

    def sharpe(rows):
        sd = rows.std(axis=0, ddof=1)
        return np.where(sd > 0, rows.mean(axis=0) / np.where(sd > 0, sd, 1.0), -np.inf)

    logits, degradation = [], []
    for train in combinations(range(blocks), blocks // 2):
        test = [k for k in range(blocks) if k not in train]
        inside = sharpe(np.vstack([parts[k] for k in train]))
        outside = sharpe(np.vstack([parts[k] for k in test]))
        best = int(np.argmax(inside))
        rank = (outside < outside[best]).sum() + 0.5 * ((outside == outside[best]).sum() - 1) + 1
        omega = rank / (m.shape[1] + 1)
        logits.append(np.log(omega / (1 - omega)))
        degradation.append((inside[best] * np.sqrt(PERIODS), outside[best] * np.sqrt(PERIODS)))
    logits = np.asarray(logits)
    pairs = np.asarray(degradation)
    return {'pbo': float((logits <= 0).mean()), 'n_splits': int(len(logits)),
            'is_best_sharpe': float(pairs[:, 0].mean()), 'oos_of_best_sharpe': float(pairs[:, 1].mean()),
            'oos_of_best_below_zero': float((pairs[:, 1] < 0).mean())}


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


def combined(data,factors,timing='z',method='mean'):
    frames,weights = [],[]
    for name,weight in factors.items():
        # JSON weights multiply the already economically directed factor.
        direction = library.SIGNED_PRIORS[name]*np.sign(weight)
        frames.append(library.to_signal(name,data.factors.raw(name),timing,direction))
        weights.append(abs(weight))
    return library.combine(frames,method,weights)


def signal_for(data,spec):
    timing = spec.get('timing','z')
    return combined(data,spec['factors'],timing,spec.get('combine','mean'))


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
    """只算原始因子的 IC，Spearman 和标准化预测分数分开记录。"""
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
    for directory in (cache.timestamp_dir(),cache.report_dir(),
                      cache.combo_dir(C.IC_REFERENCE_LOOKBACK,C.IC_REFERENCE_PCT),
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


def plot_performance(returns, out_dir, title="净值与最大回撤", series=None) -> Path:
    """日收益（列为策略）→ 净值和回撤同一张图，写到 out_dir/performance.png。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(returns).apply(pd.to_numeric, errors="coerce")
    if series:
        missing = set(series) - set(frame.columns)
        if missing:
            raise KeyError(f"结果中没有这些序列: {sorted(missing)}")
        frame = frame[list(series)]
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all").fillna(0.0)
    if frame.empty:
        raise ValueError("没有可绘制的日收益数据")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    nav = (1 + frame).cumprod()
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1
    fig, (nav_axis, drawdown_axis) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    nav.plot(ax=nav_axis, lw=1.2)
    nav_axis.set_title(title)
    nav_axis.set_ylabel("净值")
    nav_axis.grid(alpha=0.3)
    nav_axis.legend(loc="upper left", fontsize=8)
    drawdown.plot(ax=drawdown_axis, lw=1.0, legend=False)
    drawdown_axis.fill_between(drawdown.index, drawdown.min(axis=1), 0, alpha=0.2)
    drawdown_axis.set_ylabel("回撤")
    drawdown_axis.grid(alpha=0.3)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "performance.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


context = types.SimpleNamespace(run_dir=run_dir, clean_json=clean_json, dump_json=dump_json, plot_performance=plot_performance)
costs = types.SimpleNamespace(fee_path=fee_path, MULTIPLIER=MULTIPLIER, load_fees=load_fees, fee_tables=fee_tables, load_ticks=load_ticks, slippage_tables=slippage_tables)
engine = types.SimpleNamespace(rebalance=rebalance, position=position, allocate=allocate, trade_legs=trade_legs, backtest=backtest)
stats = types.SimpleNamespace(deflated_sharpe=deflated_sharpe, pbo_cscv=pbo_cscv, _pair=_pair, _by_period=_by_period, _period_mean=_period_mean, _corr=_corr, spearman_ic=spearman_ic, summarize_ics=summarize_ics, horizon_of=horizon_of, exante_scaled_return=exante_scaled_return, ic_period_series=ic_period_series, nw_lag=nw_lag, _nw_se=_nw_se, timeseries_t=timeseries_t, sign_status=sign_status, IC_COLUMNS=IC_COLUMNS, _slice_year=_slice_year, factor_ic_table=factor_ic_table, cross_sectional_ic_table=cross_sectional_ic_table, PERIODS=PERIODS, METRIC_KEYS=METRIC_KEYS, _empty=_empty, performance=performance, sharpe_ratio=sharpe_ratio)
study = types.SimpleNamespace(StudyData=StudyData, load_research=load_research, validate_specs=validate_specs, combined=combined, signal_for=signal_for, run_spec=run_spec, performance_rows=performance_rows, factor_diagnostics=factor_diagnostics, provenance=provenance)
