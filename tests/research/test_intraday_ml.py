import numpy as np
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.research.intraday import features, ml
from tfcta.research.backtest.costs import intraday_costs


def minutes():
    parts = []
    for day in pd.bdate_range('2016-01-04', periods=4):
        idx = pd.date_range(day + pd.Timedelta(541, unit='min'), periods=8, freq='min')
        p = np.array([100, 101, 101, 102, 103, 102, 101, 104.], dtype=float)
        parts.append(pd.DataFrame({'close':p, 'open':p-.5, 'closew':p+500,
            'highw':p+501, 'loww':p+499, 'volume':np.arange(1,9),
            'trading_date':day}, index=idx))
    return pd.concat(parts)


def panel():
    dates = pd.to_datetime(['2016-06-01', '2016-07-01', '2017-01-03',
                            '2017-06-01', '2018-01-02', '2018-06-01'])
    decisions = dates + pd.Timedelta(570, unit='min')
    data = pd.DataFrame({'symbol':'A', 'trading_date':dates,
        'decision_time':decisions, 'entry_time':decisions+pd.Timedelta(1, unit='min'),
        'exit_time':decisions+pd.Timedelta(30, unit='min'),
        'gross_return':[.01, -.01, .02, -.02, .03, -.03]})
    for i, name in enumerate(features.FEATURES):
        data[name] = np.arange(len(data), dtype=float) + i
    return data


def test_features_ignore_every_price_and_volume_after_decision():
    source = minutes()
    first = features.samples_from_minutes('A', source, 3, 2, lookback=2)
    altered = source.copy()
    after = (altered.index.normalize() == altered.index[-1].normalize()) & (altered.index.minute >= 4)
    altered.loc[after, ['close','open','closew','highw','loww','volume']] += 123
    second = features.samples_from_minutes('A', altered, 3, 2, lookback=2)
    pd.testing.assert_frame_equal(first[list(features.FEATURES)], second[list(features.FEATURES)])
    assert not first.gross_return.equals(second.gross_return)
    assert first.entry_price.iloc[0] == 101.5
    assert first.exit_price.iloc[0] == 103
    assert first.reference_price.iloc[0] == 101
    assert first.duration_last.iloc[2] == 2  # nearest qualifying price is the first bar
    assert first.prefix_dfp_top3.iloc[:2].isna().all()


def test_sample_cannot_silently_trade_across_a_missing_entry_minute():
    source = minutes()
    source = source[source.index.minute != 4]
    with pytest.raises(ValueError, match='缺失'):
        features.samples_from_minutes('A', source, 3, 2, lookback=2)


def test_research_feature_builder_rejects_holdout():
    source = minutes()
    source['trading_date'] = pd.Timestamp('2023-01-03')
    with pytest.raises(C.HoldoutViolation):
        features.samples_from_minutes('A', source, 3, 2, lookback=2)


@pytest.mark.parametrize('model', ['ridge', 'hgb'])
def test_walk_forward_does_not_fit_transformations_or_labels_on_test(model):
    source = panel()
    predicted, folds = ml.walk_forward(source, model=model, min_train=2)
    changed = source.copy()
    later = changed.trading_date.dt.year >= 2017
    changed.loc[later, 'gross_return'] = 1000.
    changed.loc[changed.trading_date.dt.year == 2018, list(features.FEATURES)] = 10000.
    alternate, _ = ml.walk_forward(changed, model=model, min_train=2)
    pd.testing.assert_series_equal(predicted.loc[predicted.trading_date.dt.year == 2017, 'prediction'],
                                  alternate.loc[alternate.trading_date.dt.year == 2017, 'prediction'])
    assert predicted.loc[predicted.trading_date.dt.year == 2016, 'prediction'].isna().all()
    assert (folds.train_last_exit < folds.test_first_decision).all()


def test_walk_forward_purges_labels_unavailable_at_first_test_decision():
    source = panel()
    source.loc[1, 'exit_time'] = source.loc[2, 'decision_time']
    _, folds = ml.walk_forward(source, min_train=1)
    assert folds.loc[folds.test_year.eq(2017), 'train_rows'].item() == 1


def test_ml_whitelist_cannot_include_future_entry_or_target():
    with pytest.raises(ValueError, match='特征'):
        ml.walk_forward(panel(), feature_names=['gross_return'], min_train=1)


def test_intraday_fee_uses_close_today_and_entry_notional_for_both_sides():
    samples = pd.DataFrame({'symbol':['A','A'], 'trading_date':pd.to_datetime(['2020-01-02','2021-01-04']),
        'reference_price':[100.,100.], 'entry_price':[200.,200.], 'exit_price':[300.,300.]})
    fees = pd.DataFrame({'symbol':['A','A'], 'trading_date':samples.trading_date,
        'commission_type':['by_volume','by_money'], 'open_commission':[1.,.001],
        'close_commission':[2.,.002], 'close_commission_today':[10.,.01]})
    ticks = pd.DataFrame({'symbol':['A','A','A'], 'year':[2019,2020,2021], 'tick':[1.,2.,50.]})
    result = intraday_costs(samples, fees, ticks)
    assert result.estimated_cost.tolist() == pytest.approx([11.02/1000 + 2/100, .01111+4/100])
    assert result.actual_cost.tolist() == pytest.approx([11.02/2000 + 2/200, .00101+.01515+4/200])
    with pytest.raises(ValueError, match='平今'):
        intraday_costs(samples, fees.drop(columns='close_commission_today'), ticks)


def test_portfolio_keeps_cash_weight_and_charges_one_roundtrip():
    source = panel().iloc[2:3].copy()
    source['prediction'] = .03
    source['estimated_cost'] = .01
    source['actual_cost'] = .012
    trades, daily = ml.evaluate(source, {2017:['A','B']})
    assert trades.position.item() == 1
    assert daily.gross.item() == .01
    assert daily.cost.item() == .006
    assert daily.net.item() == pytest.approx(.004)
    source['prediction'] = -.03
    trades, daily = ml.evaluate(source, {2017:['A','B']})
    assert trades.position.item() == -1
    assert daily.net.item() == pytest.approx(-.016)
    source['prediction'] = .005
    assert ml.evaluate(source, {2017:['A','B']})[1].net.item() == 0


def test_portfolio_refuses_duplicate_same_day_positions():
    source = panel().iloc[2:3].assign(prediction=.03, estimated_cost=.01, actual_cost=.012)
    with pytest.raises(ValueError, match='重复'):
        ml.evaluate(pd.concat([source,source]), {2017:['A']})


def test_rolling_training_excludes_old_years():
    source = panel()
    latest = source.iloc[-1:].copy()
    latest['trading_date'] = pd.Timestamp('2019-01-02')
    latest['decision_time'] = pd.Timestamp('2019-01-02 09:30')
    latest['entry_time'] = pd.Timestamp('2019-01-02 09:31')
    latest['exit_time'] = pd.Timestamp('2019-01-02 10:00')
    source = pd.concat([source,latest], ignore_index=True)
    _, folds = ml.walk_forward(source, min_train=1, training_years=2)
    assert folds.loc[folds.test_year.eq(2019),'train_rows'].item() == 4
    assert folds.loc[folds.test_year.eq(2019),'train_first_date'].item().year == 2017


def test_quarterly_refit_only_uses_labels_available_before_that_quarter():
    source = panel()
    predicted, folds = ml.walk_forward(source, min_train=1, refit='quarterly')
    fold = folds[folds.test_period.eq('2017Q2')].iloc[0]
    assert fold.train_rows == 3  # 2016 and the first 2017 observation
    assert fold.train_last_exit < fold.test_first_decision
    changed = source.copy()
    changed.loc[3,'gross_return'] = 123.
    other,_ = ml.walk_forward(changed, min_train=1, refit='quarterly')
    assert predicted.loc[3,'prediction'] == other.loc[3,'prediction']


def test_risk_scaled_training_restores_predicted_raw_return_units():
    source = panel().assign(risk_scale=.01)
    raw,_ = ml.walk_forward(source, min_train=1)
    normalized,_ = ml.walk_forward(source, min_train=1, target='risk_scaled')
    np.testing.assert_allclose(raw.prediction, normalized.prediction, equal_nan=True)
    source.loc[2,'risk_scale'] = 0
    with pytest.raises(ValueError, match='风险'):
        ml.walk_forward(source, min_train=1, target='risk_scaled')


def test_symbol_models_and_balanced_pool_use_only_training_membership_counts():
    days=pd.bdate_range('2016-01-04',periods=8)
    test_days=pd.to_datetime(['2017-01-03','2017-01-04'])
    dates=days.append(test_days)
    decisions=dates+pd.Timedelta(570,unit='min')
    source=pd.DataFrame(dict(symbol=['A']*6+['B']*2+['A','B'],trading_date=dates,
        decision_time=decisions,entry_time=decisions+pd.Timedelta(1,unit='min'),
        exit_time=decisions+pd.Timedelta(15,unit='min'),gross_return=[.01]*6+[.03]*2+[.01,.01]))
    for name in features.FEATURES:
        source[name]=0.
    pooled,_=ml.walk_forward(source,min_train=1)
    balanced,_=ml.walk_forward(source,min_train=1,scope='balanced')
    symbol,_=ml.walk_forward(source,min_train=1,scope='symbol')
    assert pooled.prediction.iloc[-2:].tolist() == pytest.approx([.015,.015])
    assert balanced.prediction.iloc[-2:].tolist() == pytest.approx([.02,.02])
    assert symbol.prediction.iloc[-2:].tolist() == pytest.approx([.01,.03])


def test_a_future_trading_date_never_enters_an_earlier_fold_even_if_timestamp_is_bad():
    source=panel()
    for col in ['decision_time','entry_time','exit_time']:
        source.loc[5,col]=source.loc[0,col]
    _,folds=ml.walk_forward(source,min_train=1)
    assert folds.loc[folds.test_year.eq(2017),'train_rows'].item() == 2
