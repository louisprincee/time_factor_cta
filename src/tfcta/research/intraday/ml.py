"""Fixed model settings, expanding annual training, train-only preprocessing.

    2016 is initial training, 2017–2021 are forward research folds. No automatic
    feature/hyperparameter selection, no 2022 validation and no strict OOS reads.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ... import config as C
from .features import FEATURES


def walk_forward(samples, feature_names=FEATURES, model='ridge', min_train=1000):
    if not feature_names or set(feature_names)-set(FEATURES):
        raise ValueError('只接受决策时已知的白名单特征')
    if len(set(feature_names)) != len(feature_names):
        raise ValueError('特征重复')
    if model not in ('ridge', 'hgb') or not isinstance(min_train, int) or min_train < 1:
        raise ValueError('无效模型或最小训练样本数')
    data = samples.copy().reset_index(drop=True)
    data['trading_date'] = pd.to_datetime(data.trading_date)
    for name in ('decision_time', 'entry_time', 'exit_time'):
        data[name] = pd.to_datetime(data[name])
    C.assert_no_holdout_dates(data.trading_date, '日内 ML 样本')
    C.assert_no_holdout_dates(data.exit_time, '日内 ML 标签截止')
    if (data.trading_date < pd.Timestamp(C.STUDY_START)).any():
        raise ValueError('ML 训练从 2016 开始，预热数据不作为训练样本')
    if data[['trading_date','decision_time','entry_time','exit_time']].isna().any().any():
        raise ValueError('样本时间缺失')
    if (data.decision_time >= data.entry_time).any() or (data.entry_time > data.exit_time).any():
        raise ValueError('决策、入场、出场时间顺序错误')
    X = data[list(feature_names)].astype(float)
    if np.isinf(X.to_numpy()).any() or not np.isfinite(data.gross_return).all():
        raise ValueError('特征含无穷或训练标签无效')
    data['prediction'] = np.nan
    folds = []
    for year in range(2017, 2022):
        test = data.trading_date.dt.year.eq(year)
        if not test.any():
            continue
        cutoff = data.loc[test, 'decision_time'].min()
        train = data.trading_date.dt.year.lt(year) & data.exit_time.lt(cutoff)
        row = dict(test_year=year, train_rows=int(train.sum()), test_rows=int(test.sum()),
            train_last_exit=data.loc[train,'exit_time'].max(), test_first_decision=cutoff,
            fitted=int(train.sum()) >= min_train)
        folds.append(row)
        if not row['fitted']:
            continue
        estimator = (Ridge(alpha=10.) if model=='ridge' else HistGradientBoostingRegressor(
            max_iter=100, max_depth=3, learning_rate=.05, random_state=0))
        pipeline = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                 StandardScaler(), estimator)
        pipeline.fit(X.loc[train], data.loc[train,'gross_return'])
        data.loc[test,'prediction'] = pipeline.predict(X.loc[test])
    return data, pd.DataFrame(folds)


def evaluate(predictions, universe, calendar=None):
    """One ±1 notional position per day; annual pool allocates fixed 1/N slots.

    Gate uses decision-time estimated roundtrip cost; P&L charges realized
    historical commission + fixed per-side ticks once. Missing signals = cash.
    """
    trades = predictions.copy()
    trades['trading_date'] = pd.to_datetime(trades.trading_date)
    C.assert_no_holdout_dates(trades.trading_date, '日内组合')
    if trades.duplicated(['symbol','trading_date']).any():
        raise ValueError('同品种同交易日重复仓位')
    if calendar is None:
        calendar = trades.trading_date.unique()
    dates = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
    C.assert_no_holdout_dates(dates, '日内组合日历')
    trades['position'] = 0.
    active = trades.prediction.abs().gt(trades.estimated_cost) & trades.estimated_cost.notna()
    member = pd.Series([s in universe.get(d.year, []) for s,d in
                       zip(trades.symbol, trades.trading_date)], index=trades.index)
    active &= member
    if not np.isfinite(trades.loc[active, ['actual_cost','gross_return']].to_numpy()).all():
        raise ValueError('交易成本或收益缺失，不能零成本交易')
    trades.loc[active, 'position'] = np.sign(trades.loc[active,'prediction'])
    trades['gross'] = trades.position * trades.gross_return
    trades['cost'] = trades.position.abs() * trades.actual_cost.fillna(0.)
    trades['net'] = trades.gross-trades.cost
    trades['turnover'] = 2*trades.position.abs()
    daily = trades.groupby('trading_date')[['gross','cost','net','turnover']].sum().reindex(dates, fill_value=0.)
    denominators = pd.Series([len(universe.get(d.year,[])) for d in dates], index=dates)
    daily = daily.div(denominators.replace(0,np.nan), axis=0).fillna(0.)
    daily.index.name = 'trading_date'
    return trades, daily
