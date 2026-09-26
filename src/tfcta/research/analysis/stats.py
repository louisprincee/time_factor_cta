"""时序 IC 与六项绩效。

IC 是每个品种单独做 Spearman 再对品种取平均，不是横截面相关。
绩效与论文同一组指标，年化按 252 个交易日。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ... import config as C
from ...factors.library import exante_z


def _pair(factor: pd.Series, fwd: pd.Series) -> pd.DataFrame:
    return pd.concat([factor.rename('f'), fwd.rename('r')], axis=1).dropna()


def _corr(df: pd.DataFrame, min_obs: int) -> float:
    if len(df) < int(min_obs):
        return np.nan
    if df['f'].nunique(dropna=True) < 2 or df['r'].nunique(dropna=True) < 2:
        return np.nan
    return float(df['f'].corr(df['r'], method='spearman'))


def spearman_ic(factor: pd.Series, fwd: pd.Series, min_obs: int) -> float:
    return _corr(_pair(factor, fwd), min_obs)


def summarize_ics(ics: list[float]) -> dict:
    """把一折内各品种的 IC 汇总成 (ic, t_cross, n_symbols)。

    ``t_cross`` 的口径是**跨品种**：分母是品种间 IC 的标准误，回答的是"这个因子在
    多少品种上同向"，不是"这个因子在时间上有多稳"。它把相关性很高的商品当成了
    独立样本，所以绝对值偏大，只能用来排序、**不能当显著性检验**。
    显著性看 ``t``，那是 :func:`timeseries_t` 给的时序 Newey-West t 值。
    """
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


def exante_scaled_return(fwd: pd.DataFrame,
                         window: int | None = None,
                         min_periods: int | None = None) -> pd.DataFrame:
    """未来收益除以 t 日已知的波动。

    ``fwd[t] = day_ret[t+1]``，``fwd.shift(2)[t] = day_ret[t-1]``，后者在 t 日收盘前
    已经实现（t 日开盘时就知道了）。用 ``shift(1)`` 会用到 t+1 开盘的价格。
    """
    window = C.IC_VOL_WINDOW if window is None else int(window)
    min_periods = C.IC_Z_MIN if min_periods is None else int(min_periods)
    vol = fwd.shift(2).rolling(window, min_periods=min_periods).std()
    return fwd / vol.where(vol > 0)


def ic_period_series(factor: pd.DataFrame,
                     fwd: pd.DataFrame,
                     symbols: list[str],
                     min_obs: int | None = None,
                     period: str | None = None,
                     prepared: bool = False) -> pd.Series:
    """逐期（默认逐月）的事前 IC 序列，时序显著性检验的输入。

    每期的统计量是 ``mean(z_t · r̃_{t+1})``：``z`` 是事前标准化的因子，``r̃`` 是
    除以事前波动的未来收益。两者都近似单位方差，所以量级与相关系数可比。

    **不能在期内算相关系数。** 期内相关要在期内去均值，而对 RSI、均线乖离这类
    日间高度持续的因子，20 个观测的期内去均值会带来 Stambaugh 型的小样本负偏差：
    纯随机游走上的 RSI 用期内 Spearman 能得到 IC≈-0.20、t≈-49。事前标准化的
    ``z_t`` 在 t 日可测，收益不可预测时 ``E[z_t · r̃_{t+1}] = 0`` 严格成立。

    次序是**先在品种内按期求平均，再在期内对品种取平均**。一期只贡献一个观测，
    商品之间的同期相关被期内平均吸收，分母只来自时间上的变异。

    ``prepared=True`` 表示传入的已经是 ``z`` 与 ``r̃``（调用方需要在切片之前用完整
    历史做标准化，否则每折开头的滚动窗口会空掉）。
    """
    period = C.IC_PERIOD if period is None else str(period)
    min_obs = C.IC_PERIOD_MIN_OBS if min_obs is None else int(min_obs)
    if not prepared:
        factor = exante_z(factor)
        fwd = exante_scaled_return(fwd)
    cols = {}
    for s in symbols:
        if s not in factor.columns or s not in fwd.columns:
            continue
        df = _pair(factor[s], fwd[s])
        if df.empty:
            continue
        prod = df['f'] * df['r']
        g = prod.groupby(pd.Grouper(freq=period))
        m = g.mean().where(g.count() >= min_obs)
        cols[s] = m
    if not cols:
        return pd.Series(dtype='float64')
    out = pd.DataFrame(cols).mean(axis=1, skipna=True).dropna()
    out.name = 'ic_period'
    return out


def nw_lag(n: int) -> int:
    """Newey-West 的自动截断滞后 floor(4·(n/100)^(2/9))。

    月度 IC 的自相关本来不高，但它不是 0：分位轨与标准化窗口都是滚动的，
    相邻月份的信号有重叠。取 0 会低估标准误、把 t 值抬高。
    """
    if n < 2:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _nw_se(x: np.ndarray, lag: int) -> float:
    """样本均值的 Newey-West 标准误（Bartlett 权）。lag=0 即普通标准误。

    退化判定用**相对**门槛而不是 ``s <= 0``：常数序列的离均差是 1e-17 级的浮点残渣，
    平方后长期方差约 1e-34，绝对值判定会让它过关并给出 t=2e16 这种荒唐的"显著"。
    与均方对比之后，只有真的一点变异都没有才会落进这个分支。
    """
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
    """IC 时序均值的显著性：``ic_ts`` / ``t`` / ``ic_ir`` / ``n_periods`` / ``nw_lag``。

    ``ic_ts`` 是逐期 IC 的均值，与 ``ic``（逐品种全窗口 IC 的均值）算的是同一件事
    但加权不同，两者差得多说明 IC 在年内极不均匀，值得在报告里说一句。
    ``ic_ir`` = 均值 / 标准差，不做年化——月度与日度的年化因子不同，写成年化容易被
    误读成"信息比率"。

    期数少于 ``IC_PERIOD_MIN_COUNT`` 时 t 值留空。逐期 IC 完全没有变异（标准误为 0）
    时也留空而不给 inf：那是自造数据或退化样本，不是显著。
    """
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
    """ok / flip / flip_weak / inconclusive / no_prior。

    给了 ``t``（时序 Newey-West t）之后，方向相反再分两档：

    * ``flip``      —— 反向且 ``|t| >= C.SIGN_T_MIN``。这是要去查实现的信号，照样拦。
    * ``flip_weak`` —— 反向但测不出来。pmt 在商品上就是这样：IC = +0.0008、t = 0.22，
      六折里四折"反向"两折"同向"，量级全在 0.016 以内。把它判成 flip 等于宣称
      "发现了一个方向错误"，而真实结论是"这个因子在商品上没有可测的 IC"——论文的
      因子没有迁移过来，是个研究结论，不是 bug。两者的处置不同，标签就得分开。

    不给 ``t`` 时退回严格口径（反向即 ``flip``），这样直接调用它的地方不会被悄悄放松。
    """
    if name not in C.PRIOR_FACTORS:
        return 'no_prior'
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
                    min_obs: int | None = None) -> pd.DataFrame:
    """逐折一行，最后再加一行 fold=mean_of_folds。

    每折**既切品种池也切时间**。切时间这件事一度漏了——只按 `universe[y]` 换品种、
    ``factor[s]`` 和 ``fwd[s]`` 传的是完整历史，于是六折"逐折 IC"其实是同一个全样本
    IC 的六个品种池变体，折间一致性看起来好得离谱。时序 t 值的前提就是"一折 = 一段
    时间"，不切时间它连定义都不成立。

    两个 t 值并列输出，含义完全不同：

    * ``t`` —— 逐期 IC 序列均值的 Newey-West t，**这才是显著性**。
    * ``t_cross`` —— 品种间 IC 的横截面 t，只回答"多少品种同向"，仅供排序。

    ``ic`` 保持"逐品种全折窗口 Spearman 的均值"不变，因为方向验收 ``sign_status``
    读的是它；``ic_ts`` 是逐期 IC 的均值，两者差得多说明 IC 在折内极不均匀。
    """
    min_obs = C.IC_MIN_OBS if min_obs is None else int(min_obs)
    z_all, r_all = exante_z(factor), exante_scaled_return(fwd)
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
                   sign=sign_status(rec['ic'], name, rec['t']))
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
    tail['sign'] = sign_status(mean_ic, name, tail['t'])
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
        grouped = daily.groupby(pd.Grouper(freq=C.IC_PERIOD))
        monthly = grouped.mean().where(grouped.count() >= min_days).dropna()
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
    peak = np.maximum.accumulate(nav)
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
