from __future__ import annotations
# 合并弱收盘特征、价格极值和路径量能诊断；子研究分别保留协议与输出目录。
# 收盘特征区分力。
import json as close__json
from pathlib import Path as close__Path
import numpy as close__np
import pandas as close__pd
from scipy.optimize import minimize as close__minimize
from scipy.special import expit as close__expit
from scipy.stats import rankdata as close__rankdata, t as close__student_t
from research.time_factor_screen import ols_hac as close__ols_hac
from tfcta import config as close__C
from tfcta.data import load_shard as close__load_shard
close__ROOT = close__Path(__file__).resolve().parents[2]
close__CONFIG_PATH = close__ROOT / 'config' / 'research' / 'index_close_discriminators.json'
close__OUT = close__C.study_dir('index', 'index_close_discriminators')
close__SYMBOLS = ('IF', 'IH', 'IC')
close__FEATURES = ('ret_5d', 'ret_20d', 'overnight_gap', 'intraday_return', 'range_surprise', 'volume_surprise', 'signed_volume_surprise', 'amihud_surprise')
close__WEAK_CLOSE = 0.4
close__SCORE_START = close__pd.Timestamp('2015-04-16')
close__SCORE_END = close__pd.Timestamp('2021-12-31')
close__EVAL_YEARS = tuple(range(2016, 2022))
close__HAC_LAGS = 5
close__ROLLING_WINDOW = 60
close__ROLLING_MINIMUM = 40
close__LOGISTIC_C = 1.0
close__RANGE_GROUPS = ('低', '中', '高')
close__REQUIRED_FIELDS = ('trading_date', 'open', 'openw', 'close', 'closew', 'highw', 'loww', 'volume', 'total_turnover')

def close__load_protocol(path: close__Path=close__CONFIG_PATH) -> dict:
    protocol = close__json.loads(path.read_text(encoding='utf-8'))
    sample = protocol.get('sample', '')
    parameters = protocol.get('parameters', {})
    grouped_follow_up = protocol.get('follow_up_grouped_range_surprise', {})
    if protocol.get('status') != 'declared_test_not_a_strategy' or '不读 2022' not in sample or '不读 2026' not in sample or (protocol.get('candidate_features') is None) or (tuple(protocol['candidate_features']) != close__FEATURES) or (parameters.get('instruments') != list(close__SYMBOLS)) or (parameters.get('weak_close_threshold') != close__WEAK_CLOSE) or (parameters.get('rolling_window') != close__ROLLING_WINDOW) or (parameters.get('rolling_minimum') != close__ROLLING_MINIMUM) or (parameters.get('hac_lags') != close__HAC_LAGS) or (parameters.get('logistic_C') != close__LOGISTIC_C) or (parameters.get('evaluation_years') != list(close__EVAL_YEARS)) or (grouped_follow_up.get('status') != 'post_hoc_exploratory_follow_up') or (grouped_follow_up.get('feature') != '使用已登记的 range_surprise：日内振幅 / 此前20日振幅中位数，再按此前60日均值和标准差标准化；标准化不含当天。'):
        raise ValueError('弱收盘区分力检验登记与实现或样本边界不一致')
    return protocol

def close___daily_from_minutes(minute: close__pd.DataFrame, symbol: str) -> close__pd.DataFrame:
    missing = sorted(set(close__REQUIRED_FIELDS) - set(minute.columns))
    if missing:
        raise KeyError(f'{symbol} 分片缺少区分力检验所需字段：{missing}')
    frame = minute.copy()
    frame['trading_date'] = close__pd.to_datetime(frame['trading_date']).dt.normalize()
    close__C.assert_no_holdout_dates(frame['trading_date'], what=f'{symbol} 区分力分钟数据')
    grouped = frame.groupby('trading_date', sort=True)
    daily = close__pd.DataFrame({'open': grouped['open'].first(), 'openw': grouped['openw'].first(), 'close': grouped['close'].last(), 'closew': grouped['closew'].last(), 'highw': grouped['highw'].max(), 'loww': grouped['loww'].min(), 'volume': grouped['volume'].sum(min_count=1), 'turnover': grouped['total_turnover'].sum(min_count=1)})
    daily.index = close__pd.DatetimeIndex(daily.index)
    return daily

def close__symbol_panel(daily: close__pd.DataFrame) -> close__pd.DataFrame:
    """单品种日线转为收盘已知特征及下一开盘段结果。"""
    required = {'open', 'openw', 'close', 'closew', 'highw', 'loww', 'volume', 'turnover'}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise KeyError(f'日线缺少字段：{missing}')
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError('日线交易日必须唯一且升序')
    previous_close = daily['close'].shift(1).where(daily['close'].shift(1) > 0)
    close_to_close = (daily['closew'] - daily['closew'].shift(1)) / previous_close
    overnight_gap = (daily['openw'] - daily['closew'].shift(1)) / previous_close
    intraday_return = (daily['closew'] - daily['openw']) / daily['open'].where(daily['open'] > 0)
    daily_range = (daily['highw'] - daily['loww']) / previous_close
    prior_range = daily_range.shift(1).rolling(20, min_periods=20).median()
    prior_volume = daily['volume'].shift(1).rolling(20, min_periods=20).median()
    amihud = close__np.log(close_to_close.abs().where(close_to_close.abs() > 0) / daily['turnover'].where(daily['turnover'] > 0))
    prior_amihud = amihud.shift(1).rolling(20, min_periods=20).median()
    raw_features = close__pd.DataFrame(index=daily.index)
    raw_features['ret_5d'] = (daily['closew'] - daily['closew'].shift(5)) / daily['close'].shift(5).where(daily['close'].shift(5) > 0)
    raw_features['ret_20d'] = (daily['closew'] - daily['closew'].shift(20)) / daily['close'].shift(20).where(daily['close'].shift(20) > 0)
    raw_features['overnight_gap'] = overnight_gap
    raw_features['intraday_return'] = intraday_return
    raw_features['range_surprise'] = daily_range / prior_range.where(prior_range > 0)
    raw_features['volume_surprise'] = close__np.log(daily['volume'].where(daily['volume'] > 0) / prior_volume.where(prior_volume > 0))
    raw_features['signed_volume_surprise'] = close__np.sign(close_to_close) * raw_features['volume_surprise']
    raw_features['amihud_surprise'] = amihud - prior_amihud
    standardized = {}
    for name in close__FEATURES:
        past = raw_features[name].shift(1)
        mean = past.rolling(close__ROLLING_WINDOW, min_periods=close__ROLLING_MINIMUM).mean()
        scale = past.rolling(close__ROLLING_WINDOW, min_periods=close__ROLLING_MINIMUM).std()
        standardized[name] = (raw_features[name] - mean) / scale.where(scale > 0)
    high_low = daily['highw'] - daily['loww']
    clv = (daily['closew'] - daily['loww']) / high_low.where(high_low > 0)
    future_return = (daily['openw'].shift(-2) - daily['openw'].shift(-1)) / daily['open'].shift(-1).where(daily['open'].shift(-1) > 0)
    out = close__pd.DataFrame({'clv': clv, 'outcome_return': future_return}, index=daily.index)
    for name, values in standardized.items():
        out[name] = values
    return out.replace([close__np.inf, -close__np.inf], close__np.nan)

def close__build_market_panel(daily_by_symbol: dict[str, close__pd.DataFrame]) -> close__pd.DataFrame:
    if tuple(daily_by_symbol) != close__SYMBOLS:
        raise ValueError(f'必须按 {close__SYMBOLS} 顺序提供全部品种')
    per_symbol = {symbol: close__symbol_panel(daily_by_symbol[symbol]) for symbol in close__SYMBOLS}
    dates = per_symbol[close__SYMBOLS[0]].index
    for symbol in close__SYMBOLS[1:]:
        dates = dates.intersection(per_symbol[symbol].index)
    dates = dates[(dates >= close__SCORE_START) & (dates <= close__SCORE_END)]
    if dates.empty:
        raise ValueError('共同上市研究期没有共同交易日')
    close__C.assert_no_holdout_dates(dates, what='股指弱收盘区分力共同日期')
    panel = close__pd.DataFrame(index=dates)
    for name in ('clv', 'outcome_return', *close__FEATURES):
        aligned = close__pd.concat({symbol: per_symbol[symbol][name].reindex(dates) for symbol in close__SYMBOLS}, axis=1)
        panel[name] = aligned.mean(axis=1, skipna=False)
    return panel

def close___standardize(train: close__np.ndarray, test: close__np.ndarray) -> tuple[close__np.ndarray, close__np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return ((train - mean) / scale, (test - mean) / scale)

def close___fit_logistic(train_x: close__np.ndarray, train_y: close__np.ndarray, test_x: close__np.ndarray) -> close__np.ndarray:
    if len(train_y) < 30 or close__np.unique(train_y).size < 2:
        raise ValueError('扩展窗口训练集不足或只有一个结果类别')
    train_z, test_z = close___standardize(train_x, test_x)
    design = close__np.column_stack([close__np.ones(len(train_z)), train_z])
    test_design = close__np.column_stack([close__np.ones(len(test_z)), test_z])

    def objective(beta):
        linear = design @ beta
        loss = close__np.logaddexp(0.0, linear).sum() - close__np.dot(train_y, linear)
        penalty = 0.5 * close__np.dot(beta[1:], beta[1:]) / close__LOGISTIC_C
        gradient = design.T @ (close__expit(linear) - train_y)
        gradient[1:] += beta[1:] / close__LOGISTIC_C
        return (loss + penalty, gradient)
    fitted = close__minimize(objective, close__np.zeros(design.shape[1]), method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-12})
    if not fitted.success:
        raise RuntimeError(f'logistic 回归未收敛：{fitted.message}')
    return close__expit(test_design @ fitted.x)

def close___classification_metrics(actual: close__np.ndarray, probability: close__np.ndarray) -> dict[str, float]:
    actual = close__np.asarray(actual, dtype=int)
    probability = close__np.asarray(probability, dtype=float)
    positives = int(actual.sum())
    negatives = len(actual) - positives
    if not positives or not negatives:
        auc = close__np.nan
        balanced_accuracy = close__np.nan
    else:
        ranks = close__rankdata(probability, method='average')
        auc = float((ranks[actual == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
        predicted = probability >= 0.5
        sensitivity = float(predicted[actual == 1].mean())
        specificity = float((~predicted[actual == 0]).mean())
        balanced_accuracy = (sensitivity + specificity) / 2
    return {'AUC': auc, 'Brier': float(close__np.mean((probability - actual) ** 2)), '平衡准确率': balanced_accuracy, '反转占比': float(actual.mean()), '样本数': int(len(actual))}

def close___walk_forward(panel: close__pd.DataFrame, weak_only: bool) -> tuple[close__pd.DataFrame, close__pd.DataFrame]:
    rows = []
    predictions = []
    for feature in close__FEATURES:
        columns = ['clv', 'outcome_return', feature]
        usable = panel.dropna(subset=columns).copy()
        usable['label'] = (usable['outcome_return'] > 0).astype(int)
        factor_name = feature
        for year in close__EVAL_YEARS:
            train = usable[usable.index.year < year]
            test = usable[usable.index.year == year]
            if weak_only:
                train = train[train['clv'] < close__WEAK_CLOSE]
                test = test[test['clv'] < close__WEAK_CLOSE]
            if test.empty:
                raise ValueError(f'{year} 年 {factor_name} 预测集为空')
            for model_label, additional_feature in (('CLV基线', None), (feature, feature)):
                predictor_columns = ['clv'] + ([additional_feature] if additional_feature else [])
                probability = close___fit_logistic(train[predictor_columns].to_numpy(dtype=float), train['label'].to_numpy(dtype=int), test[predictor_columns].to_numpy(dtype=float))
                actual = test['label'].to_numpy(dtype=int)
                metrics = close___classification_metrics(actual, probability)
                rows.append({'样本': '弱收盘' if weak_only else '全样本', '年份': year, '因子': factor_name, '模型': model_label, **metrics})
                predictions.extend(({'样本': '弱收盘' if weak_only else '全样本', '年份': year, '因子': factor_name, '模型': model_label, '日期': date, '标签': int(y), '概率': float(p)} for date, y, p in zip(test.index, actual, probability)))
    fold_table = close__pd.DataFrame(rows)
    prediction_table = close__pd.DataFrame(predictions)
    pooled = []
    for (factor_name, model), group in prediction_table.groupby(['因子', '模型'], sort=False):
        metrics = close___classification_metrics(group['标签'].to_numpy(), group['概率'].to_numpy())
        pooled.append({'样本': '弱收盘' if weak_only else '全样本', '年份': '合并', '因子': factor_name, '模型': model, **metrics})
    return (close__pd.concat([fold_table, close__pd.DataFrame(pooled)], ignore_index=True), prediction_table)

def close___range_surprise_group_predictions(panel: close__pd.DataFrame, evaluation_years: tuple[int, ...]=close__EVAL_YEARS, score_start: close__pd.Timestamp | None=None, score_end: close__pd.Timestamp | None=None) -> close__pd.DataFrame:
    """用扩展训练样本的相对振幅三分位切点分组弱收盘日。"""
    feature = 'range_surprise'
    usable = panel.loc[panel['clv'] < close__WEAK_CLOSE, [feature, 'outcome_return']].dropna()
    rows = []
    for year in evaluation_years:
        train = usable[usable.index.year < year]
        test = usable[usable.index.year == year]
        if score_start is not None:
            test = test.loc[test.index >= score_start]
        if score_end is not None:
            test = test.loc[test.index <= score_end]
        if len(train) < 30 or test.empty:
            raise ValueError(f'{year} 年相对振幅分组训练集或预测集为空')
        lower, upper = train[feature].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
        if not close__np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f'{year} 年相对振幅训练三分位切点无效')
        codes = close__np.searchsorted([lower, upper], test[feature].to_numpy(), side='right')
        rows.extend(({'年份': year, '日期': date, '相对振幅_z': float(value), '分组': close__RANGE_GROUPS[int(code)], '训练低切点': float(lower), '训练高切点': float(upper), '下一开盘段收益_bp': float(outcome * 10000.0), '反转代理': bool(outcome > 0)} for date, value, code, outcome in zip(test.index, test[feature], codes, test['outcome_return'])))
    return close__pd.DataFrame(rows)

def close___early_if_range_panel(path: close__Path) -> close__pd.DataFrame:
    required = ('trading_date', 'open', 'openw', 'close', 'closew', 'highw', 'loww')
    minute = close__pd.read_parquet(path, columns=list(required))
    missing = sorted(set(required) - set(minute.columns))
    if missing:
        raise KeyError(f'早期 IF 分片缺少字段：{missing}')
    minute['trading_date'] = close__pd.to_datetime(minute['trading_date']).dt.normalize()
    close__C.assert_no_holdout_dates(minute['trading_date'], what='IF 早期相对振幅分钟数据')
    if minute['trading_date'].min() != close__pd.Timestamp('2010-04-16') or minute['trading_date'].max() != close__pd.Timestamp('2015-04-16'):
        raise ValueError('早期 IF 相对振幅分片日期范围与登记不符')
    grouped = minute.groupby('trading_date', sort=True)
    daily = close__pd.DataFrame({'open': grouped['open'].first(), 'openw': grouped['openw'].first(), 'close': grouped['close'].last(), 'closew': grouped['closew'].last(), 'highw': grouped['highw'].max(), 'loww': grouped['loww'].min()})
    previous_close = daily['close'].shift(1).where(daily['close'].shift(1) > 0)
    daily_range = (daily['highw'] - daily['loww']) / previous_close
    prior_range = daily_range.shift(1).rolling(20, min_periods=20).median()
    range_surprise = daily_range / prior_range.where(prior_range > 0)
    past = range_surprise.shift(1)
    mean = past.rolling(close__ROLLING_WINDOW, min_periods=close__ROLLING_MINIMUM).mean()
    scale = past.rolling(close__ROLLING_WINDOW, min_periods=close__ROLLING_MINIMUM).std()
    clv = (daily['closew'] - daily['loww']) / (daily['highw'] - daily['loww'])
    outcome_return = (daily['openw'].shift(-2) - daily['openw'].shift(-1)) / daily['open'].shift(-1).where(daily['open'].shift(-1) > 0)
    panel = close__pd.DataFrame({'clv': clv, 'range_surprise': (range_surprise - mean) / scale.where(scale > 0), 'outcome_return': outcome_return}, index=daily.index).replace([close__np.inf, -close__np.inf], close__np.nan)
    if close__pd.isna(panel.loc['2015-04-14', 'outcome_return']):
        raise ValueError('缺少 2015-04-14 的完整下一开盘段收益')
    if close__pd.notna(panel.loc['2015-04-15', 'outcome_return']):
        raise ValueError('2015-04-15 不应有 IF 单独样本的下一开盘段收益')
    return panel

def close__run_early_if_range_groups() -> None:
    spec = close__json.loads((close__C.PROJECT_ROOT / 'config' / 'research' / 'if_before_ih_ic.json').read_text(encoding='utf-8'))
    follow_up = spec.get('follow_up_grouped_range_surprise', {})
    if follow_up.get('status') != 'post_hoc_exploratory_follow_up' or '不读 2022' not in spec.get('sample', '') or '不读取任何 OOS 分区' not in spec.get('sample', '') or (follow_up.get('validation', '').find('2011-01-01 至 2015-04-14') < 0):
        raise ValueError('早期 IF 相对振幅分组未登记或样本边界不匹配')
    source = close__C.INDEX_DATA_ROOT / 'if_before_ih_ic' / 'minute_shards' / 'IF.parquet'
    panel = close___early_if_range_panel(source)
    predictions = close___range_surprise_group_predictions(panel, evaluation_years=(2011, 2012, 2013, 2014, 2015), score_start=close__pd.Timestamp('2011-01-01'), score_end=close__pd.Timestamp('2015-04-14'))
    summary = close___summarize_range_groups(predictions)
    output = close__C.study_dir('index', 'if_before_ih_ic')
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'range_surprise_group_predictions.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(output / 'range_surprise_group_summary.csv', index=False, encoding='utf-8-sig')
    (output / 'range_surprise_protocol.json').write_text(close__json.dumps(follow_up, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"IF 单独信号日: {predictions['日期'].min().date()}..{predictions['日期'].max().date()}")
    print(f'有效弱收盘观测: {len(predictions)}')
    print(summary.to_string(index=False))
    print(f'结果目录: {output}')

def close___summarize_range_groups(predictions: close__pd.DataFrame) -> close__pd.DataFrame:
    rows = []
    for year, year_data in [*[(int(year), group) for year, group in predictions.groupby('年份', sort=True)], ('合并', predictions)]:
        for label in close__RANGE_GROUPS:
            group = year_data[year_data['分组'] == label]
            returns = group['下一开盘段收益_bp']
            rows.append({'年份': year, '相对振幅组': label, '观测数': len(group), '反转率_%': group['反转代理'].mean() * 100, '平均收益_bp': returns.mean(), '中位收益_bp': returns.median()})
    return close__pd.DataFrame(rows)

def close___bh_adjust(p_values: close__np.ndarray) -> close__np.ndarray:
    values = close__np.asarray(p_values, dtype=float)
    order = close__np.argsort(values)
    adjusted = close__np.empty(len(values), dtype=float)
    running = 1.0
    for position in range(len(values) - 1, -1, -1):
        rank = position + 1
        original_index = order[position]
        running = min(running, values[original_index] * len(values) / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted

def close___interaction_regression(panel: close__pd.DataFrame, start, end, label: str) -> close__pd.DataFrame:
    sample = panel.loc[(panel.index >= start) & (panel.index <= end)].copy()
    weak = (sample['clv'] < close__WEAK_CLOSE).astype(float)
    rows = []
    for name in close__FEATURES:
        predictors = [('interaction', sample[name] * weak), ('clv', sample['clv']), ('weak_close', weak), (name, sample[name])]
        coefficient, t_stat = close__ols_hac(sample['outcome_return'].to_numpy(), [series.to_numpy() for _, series in predictors], close__HAC_LAGS)
        n = int(sample[['outcome_return', 'clv', name]].dropna().shape[0])
        p_value = float(2 * close__student_t.sf(abs(t_stat), df=max(n - len(predictors) - 1, 1))) if close__np.isfinite(t_stat) else close__np.nan
        rows.append({'样本': label, '因子': name, '交互系数_bp': coefficient * 10000.0, 'NW_t': t_stat, 'p_双侧': p_value, '样本数': n})
    result = close__pd.DataFrame(rows)
    result['BH_q'] = close___bh_adjust(result['p_双侧'].to_numpy())
    return result

def close___yearly_interactions(panel: close__pd.DataFrame, start, end, label: str) -> close__pd.DataFrame:
    parts = []
    sample = panel.loc[(panel.index >= start) & (panel.index <= end)]
    for year, group in sample.groupby(sample.index.year):
        part = close___interaction_regression(group, group.index.min(), group.index.max(), label)
        parts.append(part.assign(年份=int(year)))
    return close__pd.concat(parts, ignore_index=True) if parts else close__pd.DataFrame()

def close__run() -> None:
    protocol = close__load_protocol()
    close__C.assert_research_only(close__C.INDEX_RESEARCH_DIR)
    daily_by_symbol = {}
    for symbol in close__SYMBOLS:
        minute = close__load_shard(symbol, directory=close__C.INDEX_RESEARCH_DIR, columns=list(close__REQUIRED_FIELDS))
        daily_by_symbol[symbol] = close___daily_from_minutes(minute, symbol)
    panel = close__build_market_panel(daily_by_symbol)
    main = close___interaction_regression(panel, close__SCORE_START, close__SCORE_END, 'IF/IH/IC共同')
    main_years = close___yearly_interactions(panel, close__SCORE_START, close__SCORE_END, 'IF/IH/IC共同')
    weak_scores, weak_predictions = close___walk_forward(panel, weak_only=True)
    all_scores, all_predictions = close___walk_forward(panel, weak_only=False)
    range_group_predictions = close___range_surprise_group_predictions(panel)
    range_group_summary = close___summarize_range_groups(range_group_predictions)
    close__OUT.mkdir(parents=True, exist_ok=True)
    (close__OUT / 'protocol.json').write_text(close__json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    main.to_csv(close__OUT / 'common_interactions.csv', index=False, encoding='utf-8-sig')
    main_years.to_csv(close__OUT / 'common_interactions_by_year.csv', index=False, encoding='utf-8-sig')
    weak_scores.to_csv(close__OUT / 'weak_close_walk_forward.csv', index=False, encoding='utf-8-sig')
    all_scores.to_csv(close__OUT / 'all_days_walk_forward.csv', index=False, encoding='utf-8-sig')
    weak_predictions.to_csv(close__OUT / 'weak_close_predictions.csv', index=False, encoding='utf-8-sig')
    all_predictions.to_csv(close__OUT / 'all_days_predictions.csv', index=False, encoding='utf-8-sig')
    range_group_predictions.to_csv(close__OUT / 'range_surprise_groups.csv', index=False, encoding='utf-8-sig')
    range_group_summary.to_csv(close__OUT / 'range_surprise_group_summary.csv', index=False, encoding='utf-8-sig')
    print(f'共同面板: {panel.index.min().date()}..{panel.index.max().date()}, {len(panel)} 天')
    print(f"弱收盘有效样本: {(panel['clv'] < close__WEAK_CLOSE).sum()} 天")
    print('\n共同市场弱收盘交互回归:')
    print(main[['因子', '交互系数_bp', 'NW_t', 'BH_q', '样本数']].to_string(index=False))
    print('\n弱收盘逐年扩展窗口预测差（相对同因子样本的 CLV 基线）:')
    weak_pooled = weak_scores[weak_scores['年份'] == '合并'].set_index(['因子', '模型'])
    for name in close__FEATURES:
        base = weak_pooled.loc[name, 'CLV基线']
        row = weak_pooled.loc[name, name]
        factor_folds = weak_scores[(weak_scores['因子'] == name) & (weak_scores['模型'] == name) & (weak_scores['年份'] != '合并')].set_index('年份')
        baseline_folds = weak_scores[(weak_scores['因子'] == name) & (weak_scores['模型'] == 'CLV基线') & (weak_scores['年份'] != '合并')].set_index('年份')
        better_years = int((factor_folds['Brier'] < baseline_folds['Brier']).sum())
        print(f"{name}: n={int(row['样本数'])}, ΔAUC={row['AUC'] - base['AUC']:+.3f}, ΔBrier={row['Brier'] - base['Brier']:+.4f}, Brier改善年份={better_years}/6")
    print('\n早期 IF 2011–2014 不在当前股指研究分片中；该阶段只引用既有研究复核记录。')
    print('\n弱收盘日相对振幅分组（逐年仅用此前样本的三分位切点）:')
    print(range_group_summary.to_string(index=False))
    print(f'\n结果目录: {close__OUT}')
# 价格极值与日内形态诊断。
import json as extremes__json
from pathlib import Path as extremes__Path
import numpy as extremes__np
import pandas as extremes__pd
from scipy.stats import t as extremes__student_t
from research.time_factor_screen import ols_hac as extremes__ols_hac
from tfcta import config as extremes__C
extremes__ROOT = extremes__Path(__file__).resolve().parents[2]
extremes__PROTOCOL_PATH = extremes__ROOT / 'config' / 'research' / 'index_price_extremes.json'
extremes__SYMBOLS = ('IF', 'IH', 'IC')
extremes__FEATURES = ('high_open', 'low_open', 'range_open', 'high_twap', 'low_twap', 'range_twap')
extremes__REQUIRED_FIELDS = ('trading_date', 'open', 'close', 'openw', 'closew', 'highw', 'loww')
extremes__IF_START = extremes__pd.Timestamp('2011-01-01')
extremes__COMMON_START = extremes__pd.Timestamp('2015-04-16')
extremes__SAMPLE_END = extremes__pd.Timestamp('2021-12-31')
extremes__ROLLING_WINDOW = 60
extremes__ROLLING_MINIMUM = 40
extremes__HAC_LAGS = 5
extremes__WEAK_CLOSE = 0.4

def extremes__load_protocol() -> dict:
    protocol = extremes__json.loads(extremes__PROTOCOL_PATH.read_text(encoding='utf-8'))
    parameters = protocol.get('parameters', {})
    if protocol.get('status') != 'declared_exploratory_research' or '不读2022' not in protocol.get('sample_boundary', '') or parameters.get('instruments') != list(extremes__SYMBOLS) or (parameters.get('feature_names') != list(extremes__FEATURES)) or (parameters.get('rolling_window') != extremes__ROLLING_WINDOW) or (parameters.get('rolling_minimum') != extremes__ROLLING_MINIMUM) or (parameters.get('hac_lags') != extremes__HAC_LAGS) or (parameters.get('weak_close_threshold') != extremes__WEAK_CLOSE):
        raise ValueError('高低点偏离检验协议与实现或样本边界不匹配')
    return protocol

def extremes__aggregate_daily(minute: extremes__pd.DataFrame, symbol: str) -> extremes__pd.DataFrame:
    missing = sorted(set(extremes__REQUIRED_FIELDS) - set(minute.columns))
    if missing:
        raise KeyError(f'{symbol} 分钟行情缺少字段：{missing}')
    frame = minute.copy()
    frame['trading_date'] = extremes__pd.to_datetime(frame['trading_date']).dt.normalize()
    extremes__C.assert_no_holdout_dates(frame['trading_date'], what=f'{symbol}高低点偏离分钟数据')
    if 'datetime' in frame.columns:
        frame = frame.sort_values(['trading_date', 'datetime'], kind='mergesort')
    elif isinstance(frame.index, extremes__pd.DatetimeIndex):
        frame = frame.sort_index(kind='mergesort')
    else:
        raise KeyError(f'{symbol}分钟数据既没有datetime列，也没有DatetimeIndex')
    if frame[['open', 'close', 'openw', 'closew', 'highw', 'loww']].isna().any().any():
        raise ValueError(f'{symbol}分钟价格存在缺失')
    frame['typical_price'] = (frame['highw'] + frame['loww'] + frame['closew']) / 3
    grouped = frame.groupby('trading_date', sort=True)
    daily = grouped.agg(open=('open', 'first'), openw=('openw', 'first'), close=('close', 'last'), closew=('closew', 'last'), highw=('highw', 'max'), loww=('loww', 'min'), twapw=('typical_price', 'mean'))
    if daily['open'].le(0).any() or daily['twapw'].le(0).any():
        raise ValueError(f'{symbol}开盘价或分钟典型价TWAP代理非正')
    if daily['twapw'].lt(daily['loww']).any() or daily['twapw'].gt(daily['highw']).any():
        raise ValueError(f'{symbol}分钟典型价TWAP代理落在当日高低区间之外')
    return daily

def extremes__features_from_daily(daily: extremes__pd.DataFrame) -> extremes__pd.DataFrame:
    symbol_fields = ('open', 'openw', 'close', 'closew', 'highw', 'loww', 'twapw')
    missing = sorted(set(symbol_fields) - set(daily.columns))
    if missing:
        raise KeyError(f'日线缺少特征计算字段：{missing}')
    high_low = daily['highw'] - daily['loww']
    if high_low.le(0).any() or daily['open'].le(0).any() or daily['twapw'].le(0).any():
        raise ValueError('日线高低区间、开盘价或TWAP代理无效')
    raw_features = extremes__pd.DataFrame(index=daily.index)
    raw_features['high_open'] = (daily['highw'] - daily['openw']) / daily['open'] * 10000.0
    raw_features['low_open'] = (daily['loww'] - daily['openw']) / daily['open'] * 10000.0
    raw_features['range_open'] = high_low / daily['open'] * 10000.0
    raw_features['high_twap'] = (daily['highw'] - daily['twapw']) / daily['twapw'] * 10000.0
    raw_features['low_twap'] = (daily['loww'] - daily['twapw']) / daily['twapw'] * 10000.0
    raw_features['range_twap'] = high_low / daily['twapw'] * 10000.0
    standardized = {}
    for feature in extremes__FEATURES:
        prior = raw_features[feature].shift(1)
        mean = prior.rolling(extremes__ROLLING_WINDOW, min_periods=extremes__ROLLING_MINIMUM).mean()
        scale = prior.rolling(extremes__ROLLING_WINDOW, min_periods=extremes__ROLLING_MINIMUM).std()
        standardized[feature] = (raw_features[feature] - mean) / scale.where(scale > 0)
    clv = (daily['closew'] - daily['loww']) / high_low
    outcome = (daily['openw'].shift(-2) - daily['openw'].shift(-1)) / daily['open'].shift(-1).where(daily['open'].shift(-1) > 0)
    result = extremes__pd.DataFrame({'clv': clv, 'outcome': outcome}, index=daily.index)
    for feature in extremes__FEATURES:
        result[feature] = standardized[feature]
    return result.replace([extremes__np.inf, -extremes__np.inf], extremes__np.nan)

def extremes__daily_features(minute: extremes__pd.DataFrame, symbol: str) -> extremes__pd.DataFrame:
    return extremes__features_from_daily(extremes__aggregate_daily(minute, symbol))

def extremes___read_minutes(path: extremes__Path, symbol: str) -> extremes__pd.DataFrame:
    frame = extremes__pd.read_parquet(path, columns=list(extremes__REQUIRED_FIELDS), filters=[('trading_date', '<=', extremes__SAMPLE_END.to_pydatetime())])
    if frame.empty:
        raise ValueError(f'{symbol}分钟分片为空：{path}')
    dates = extremes__pd.to_datetime(frame['trading_date']).dt.normalize()
    extremes__C.assert_no_holdout_dates(dates, what=f'{symbol}价格极值输入')
    if dates.max() > extremes__SAMPLE_END:
        raise ValueError(f'{symbol}价格极值输入超过登记结束日 {extremes__SAMPLE_END.date()}')
    return frame

def extremes__load_if_history() -> extremes__pd.DataFrame:
    early_path = extremes__C.INDEX_DATA_ROOT / 'if_before_ih_ic' / 'minute_shards' / 'IF.parquet'
    research_path = extremes__C.INDEX_RESEARCH_DIR / 'IF.parquet'
    early = extremes___read_minutes(early_path, 'IF')
    research = extremes___read_minutes(research_path, 'IF')
    early_daily = extremes__aggregate_daily(early, 'IF')
    research_daily = extremes__aggregate_daily(research, 'IF')
    overlap = early_daily.index.intersection(research_daily.index)
    if overlap.empty:
        raise ValueError('早期IF与研究期IF分片没有重叠日期，无法核验拼接')
    compare_fields = ('open', 'openw', 'close', 'closew', 'highw', 'loww', 'twapw')
    left = early_daily.loc[overlap, list(compare_fields)]
    right = research_daily.loc[overlap, list(compare_fields)]
    extremes__pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-08, atol=1e-10, check_names=False, obj='早期与研究期IF重叠日行情')
    early_only = early_daily.loc[early_daily.index < overlap.max()]
    stitched_daily = extremes__pd.concat([early_only, research_daily.loc[research_daily.index >= overlap.max()]])
    if stitched_daily.index.has_duplicates or not stitched_daily.index.is_monotonic_increasing:
        raise ValueError('IF专用历史与研究分片拼接后日期重复或乱序')
    return extremes__features_from_daily(stitched_daily)

def extremes__load_common_panel() -> extremes__pd.DataFrame:
    per_symbol = {}
    for symbol in extremes__SYMBOLS:
        path = extremes__C.INDEX_RESEARCH_DIR / f'{symbol}.parquet'
        per_symbol[symbol] = extremes__daily_features(extremes___read_minutes(path, symbol), symbol)
    common = per_symbol[extremes__SYMBOLS[0]].index
    for symbol in extremes__SYMBOLS[1:]:
        common = common.intersection(per_symbol[symbol].index)
    if common.empty:
        raise ValueError('IF/IH/IC没有共同交易日期')
    extremes__C.assert_no_holdout_dates(common, what='IF/IH/IC共同日期')
    panel = extremes__pd.DataFrame(index=common)
    for column in ('clv', 'outcome', *extremes__FEATURES):
        aligned = extremes__pd.concat({symbol: per_symbol[symbol][column].reindex(common) for symbol in extremes__SYMBOLS}, axis=1)
        panel[column] = aligned.mean(axis=1, skipna=False)
    return panel

def extremes___p_value(t_stat: float, n: int, columns: int) -> float:
    if not extremes__np.isfinite(t_stat):
        return extremes__np.nan
    return float(2 * extremes__student_t.sf(abs(t_stat), df=max(n - columns - 1, 1)))

def extremes___fit_effects(panel: extremes__pd.DataFrame, scope: str, period: str) -> tuple[extremes__pd.DataFrame, extremes__pd.DataFrame]:
    sample = panel.dropna(subset=['outcome', 'clv', *extremes__FEATURES]).copy()
    if period == '弱收盘':
        sample = sample.loc[sample['clv'] < extremes__WEAK_CLOSE]
    primary_rows = []
    interaction_rows = []
    weak = (sample['clv'] < extremes__WEAK_CLOSE).astype(float)
    y = sample['outcome'].to_numpy(dtype=float)
    clv = sample['clv'].to_numpy(dtype=float)
    for feature in extremes__FEATURES:
        x = sample[feature].to_numpy(dtype=float)
        coefficient, t_stat = extremes__ols_hac(y, [x, clv], extremes__HAC_LAGS)
        primary_rows.append({'样本': scope, '区间': period, '特征': feature, '观测数': len(sample), '每1标准差收益变化_bp': coefficient * 10000.0, 'HAC5_t': t_stat, 'p值': extremes___p_value(t_stat, len(sample), 2)})
        if period == '全样本':
            interaction = x * weak.to_numpy(dtype=float)
            int_coef, int_t = extremes__ols_hac(y, [interaction, x, clv, weak.to_numpy(dtype=float)], extremes__HAC_LAGS)
            interaction_rows.append({'样本': scope, '特征': feature, '观测数': len(sample), '弱收盘增量_bp每标准差': int_coef * 10000.0, 'HAC5_t': int_t, 'p值': extremes___p_value(int_t, len(sample), 4)})
    primary = extremes__pd.DataFrame(primary_rows)
    primary['BH_q'] = close___bh_adjust(primary['p值'].to_numpy())
    interactions = extremes__pd.DataFrame(interaction_rows)
    if not interactions.empty:
        interactions['BH_q'] = close___bh_adjust(interactions['p值'].to_numpy())
    return (primary, interactions)

def extremes___annual_effects(panel: extremes__pd.DataFrame, scope: str) -> extremes__pd.DataFrame:
    rows = []
    eligible = panel.dropna(subset=['outcome', 'clv', *extremes__FEATURES])
    for year, sample in eligible.groupby(eligible.index.year, sort=True):
        for feature in extremes__FEATURES:
            coef, t_stat = extremes__ols_hac(sample['outcome'].to_numpy(dtype=float), [sample[feature].to_numpy(dtype=float), sample['clv'].to_numpy(dtype=float)], extremes__HAC_LAGS)
            rows.append({'样本': scope, '年份': int(year), '特征': feature, '观测数': len(sample), '每1标准差收益变化_bp': coef * 10000.0, 'HAC5_t': t_stat})
    return extremes__pd.DataFrame(rows)

def extremes__run() -> None:
    protocol = extremes__load_protocol()
    extremes__C.assert_research_only(extremes__C.INDEX_RESEARCH_DIR)
    if_panel = extremes__load_if_history()
    common_panel = extremes__load_common_panel()
    if_panel = if_panel.loc[(if_panel.index >= extremes__IF_START) & (if_panel.index <= extremes__SAMPLE_END)]
    common_panel = common_panel.loc[(common_panel.index >= extremes__COMMON_START) & (common_panel.index <= extremes__SAMPLE_END)]
    if if_panel.empty or common_panel.empty:
        raise ValueError('登记样本内没有可用观测')
    extremes__C.assert_no_holdout_dates(if_panel.index, what='IF 2011–2021计分日')
    extremes__C.assert_no_holdout_dates(common_panel.index, what='共同样本计分日')
    outputs = []
    interactions = []
    yearly = []
    for name, panel in (('IF 2011-2021', if_panel), ('IF/IH/IC 2015-2021', common_panel)):
        valid = panel.dropna(subset=['outcome', 'clv', *extremes__FEATURES])
        full, weak_interaction = extremes___fit_effects(valid, name, '全样本')
        weak, _ = extremes___fit_effects(valid, name, '弱收盘')
        outputs.extend([full, weak])
        interactions.append(weak_interaction)
        yearly.append(extremes___annual_effects(valid, name))
    result_dir = extremes__C.study_dir('index', 'index_price_extremes')
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / 'protocol.json').write_text(extremes__json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    primary = extremes__pd.concat(outputs, ignore_index=True)
    int_result = extremes__pd.concat(interactions, ignore_index=True)
    annual = extremes__pd.concat(yearly, ignore_index=True)
    primary.to_csv(result_dir / 'primary_effects.csv', index=False, encoding='utf-8-sig')
    int_result.to_csv(result_dir / 'weak_close_interactions.csv', index=False, encoding='utf-8-sig')
    annual.to_csv(result_dir / 'annual_effects.csv', index=False, encoding='utf-8-sig')
    print('主效应（每1标准差特征变化对应的收益bp，已控制CLV）:')
    print(primary.to_string(index=False))
    print('\n弱收盘交互（weak close < 0.4）:')
    print(int_result.to_string(index=False))
    print(f"\nIF有效样本: {len(if_panel.dropna(subset=['outcome', 'clv', *extremes__FEATURES]))} 日")
    print(f"IF/IH/IC有效共同样本: {len(common_panel.dropna(subset=['outcome', 'clv', *extremes__FEATURES]))} 日")
    print(f'结果目录: {result_dir}')
extremes__SHAPE_PATH = extremes__ROOT / 'config' / 'research' / 'index_weak_close_shape.json'
extremes__SHAPE_OUT = extremes__C.study_dir('index', 'index_weak_close_shape')
extremes__SHAPE_FEATURES = ('open_location', 'high_time', 'dfp_top3', 'low_dwell', 'info_discreteness')
extremes__SHAPE_EXTRA = ('high_open', 'range_open', 'intraday_return')
extremes__SHAPE_FIELDS = ('trading_date', 'datetime', 'open', 'close', 'openw', 'closew', 'highw', 'loww')
extremes__IF_REPLICATION = (extremes__pd.Timestamp('2011-01-01'), extremes__pd.Timestamp('2015-04-14'))
extremes__LOW_DWELL_BAND = 0.2
extremes__DFP_LOOKBACK, extremes__DFP_PCT = (250, 55.0)

def extremes__load_shape_protocol() -> dict:
    protocol = extremes__json.loads(extremes__SHAPE_PATH.read_text(encoding='utf-8'))
    parameters = protocol.get('parameters', {})
    if protocol.get('status') != 'declared_exploratory_research' or '不读2022' not in protocol.get('sample_boundary', '') or parameters.get('features') != list(extremes__SHAPE_FEATURES) or (parameters.get('weak_close_threshold') != extremes__WEAK_CLOSE) or (parameters.get('rolling_window') != extremes__ROLLING_WINDOW) or (parameters.get('low_dwell_band') != extremes__LOW_DWELL_BAND) or (parameters.get('dfp_lookback') != extremes__DFP_LOOKBACK) or (parameters.get('dfp_percentile') != extremes__DFP_PCT):
        raise ValueError('弱收盘形态检验协议与实现不一致')
    return protocol

def extremes__shape_from_minutes(minute: extremes__pd.DataFrame, symbol: str) -> extremes__pd.DataFrame:
    """单品种分钟表 → 每日原始形态特征（收盘时已知）和下一开盘段收益。"""
    frame = minute.copy()
    if 'datetime' not in frame.columns:
        frame = frame.reset_index()
    frame['trading_date'] = extremes__pd.to_datetime(frame['trading_date']).dt.normalize()
    extremes__C.assert_no_holdout_dates(frame['trading_date'], what=f'{symbol} 弱收盘形态分钟')
    frame = frame.set_index(extremes__pd.to_datetime(frame['datetime'])).sort_index(kind='mergesort')
    if frame[['open', 'close', 'openw', 'closew', 'highw', 'loww']].isna().any().any():
        raise ValueError(f'{symbol} 分钟价格缺失')
    day = frame['trading_date']
    g = frame.groupby(day, sort=True)
    daily = g.agg(open=('open', 'first'), openw=('openw', 'first'), closew=('closew', 'last'), highw=('highw', 'max'), loww=('loww', 'min'))
    width = (daily['highw'] - daily['loww']).where(lambda w: w > 0)
    pos = g.cumcount()
    n = g['closew'].transform('size')
    at_max = frame['highw'].eq(g['highw'].transform('max'))
    first_max = at_max & at_max.astype(int).groupby(day).cumsum().eq(1)
    high_pos = (pos / (n - 1).where(n > 1))[first_max].groupby(day[first_max]).first()
    low_cut = daily['loww'] + extremes__LOW_DWELL_BAND * width
    dwell = (frame['closew'] <= low_cut.reindex(day).to_numpy()).groupby(day).mean()
    previous = g['closew'].shift(1)
    step = frame['closew'] - previous.fillna(frame['openw'])
    up, down = ((step > 0).groupby(day).mean(), (step < 0).groupby(day).mean())
    intraday = (daily['closew'] - daily['openw']) / daily['open']
    out = extremes__pd.DataFrame({'clv': (daily['closew'] - daily['loww']) / width, 'open_location': (daily['openw'] - daily['loww']) / width, 'high_time': high_pos, 'low_dwell': dwell, 'info_discreteness': extremes__np.sign(intraday) * (down - up), 'high_open': (daily['highw'] - daily['openw']) / daily['open'], 'range_open': width / daily['open'], 'intraday_return': intraday, 'outcome': (daily['openw'].shift(-2) - daily['openw'].shift(-1)) / daily['open'].shift(-1)}, index=daily.index)
    out['dfp_top3'] = extremes___dfp_top3(frame).reindex(out.index)
    return out.replace([extremes__np.inf, -extremes__np.inf], extremes__np.nan)

def extremes___dfp_top3(frame: extremes__pd.DataFrame) -> extremes__pd.Series:
    """冻结策略口径的 dfp_top3：250 日、55% 分位持续期阈值（不含当天）。"""
    from tfcta.data import add_intraday_coords
    from tfcta.factors import duration_factors
    coords = add_intraday_coords(frame.drop(columns=['datetime'], errors='ignore'))
    return duration_factors(coords, extremes__DFP_LOOKBACK, extremes__DFP_PCT)['dfp_top3']

def extremes___rolling_z(series: extremes__pd.Series) -> extremes__pd.Series:
    prior = series.shift(1)
    mean = prior.rolling(extremes__ROLLING_WINDOW, min_periods=extremes__ROLLING_MINIMUM).mean()
    scale = prior.rolling(extremes__ROLLING_WINDOW, min_periods=extremes__ROLLING_MINIMUM).std()
    return (series - mean) / scale.where(scale > 0)

def extremes__shape_panel(per_symbol: dict[str, extremes__pd.DataFrame]) -> extremes__pd.DataFrame:
    """等权市场面板：原始特征取均值，标准化特征逐品种 60 日 z 后取均值。"""
    common = None
    for frame in per_symbol.values():
        common = frame.index if common is None else common.intersection(frame.index)
    panel = extremes__pd.DataFrame(index=common)
    for column in ('clv', 'outcome', *extremes__SHAPE_FEATURES, *extremes__SHAPE_EXTRA):
        stacked = extremes__pd.concat({s: f[column].reindex(common) for s, f in per_symbol.items()}, axis=1)
        panel[column] = stacked.mean(axis=1, skipna=False)
        if column not in ('clv', 'outcome'):
            z = extremes__pd.concat({s: extremes___rolling_z(f[column]).reindex(common) for s, f in per_symbol.items()}, axis=1)
            panel[f'{column}_z'] = z.mean(axis=1, skipna=False)
    return panel

def extremes___fit(y, xs) -> tuple[float, float, int]:
    frame = extremes__pd.concat([y, *xs], axis=1).dropna()
    if len(frame) < 40:
        return (extremes__np.nan, extremes__np.nan, len(frame))
    coef, t_stat = extremes__ols_hac(frame.iloc[:, 0].to_numpy(dtype=float), [frame.iloc[:, k].to_numpy(dtype=float) for k in range(1, frame.shape[1])], extremes__HAC_LAGS)
    return (coef, t_stat, len(frame))

def extremes__shape_tests(panel: extremes__pd.DataFrame, scope: str, signs: dict) -> extremes__pd.DataFrame:
    rows = []
    weak = panel['clv'] < extremes__WEAK_CLOSE
    w = panel[weak]
    z = lambda name: w[f'{name}_z']
    for label, xs in (('high_open_z + CLV（复述）', [z('high_open'), w['clv']]), ('OL + CLV', [w['open_location'], w['clv']]), ('high_open_z | + OL + range_z + CLV', [z('high_open'), w['open_location'], z('range_open'), w['clv']]), ('OL | + high_open_z + range_z + CLV', [w['open_location'], z('high_open'), z('range_open'), w['clv']]), ('OL | + intraday_z + range_z + CLV', [w['open_location'], z('intraday_return'), z('range_open'), w['clv']])):
        coef, t_stat, nobs = extremes___fit(w['outcome'], xs)
        rows.append({'样本': scope, '检验': 'T0', '模型': label, '第一个变量系数_bp': coef * 10000.0, 'HAC5_t': t_stat, '天数': nobs})
    p_vals = []
    for feature in extremes__SHAPE_FEATURES:
        coef, t_stat, nobs = extremes___fit(w['outcome'], [z(feature), w['clv']])
        centered = panel[f'{feature}_z']
        weak_f = weak.astype(float)
        icoef, it, inobs = extremes___fit(panel['outcome'], [centered * weak_f, centered, panel['clv'], weak_f])
        right = extremes__np.sign(coef) == signs[feature]
        rows.append({'样本': scope, '检验': 'T1', '模型': feature, '预测方向': '+' if signs[feature] > 0 else '−', '弱收盘斜率_bp每标准差': coef * 10000.0, 'HAC5_t': t_stat, '天数': nobs, '方向一致': bool(right), '全日交互_bp': icoef * 10000.0, '交互_t': it})
        p_vals.append(extremes___p_value(t_stat, nobs, 2))
    table = extremes__pd.DataFrame(rows)
    t1 = table['检验'] == 'T1'
    table.loc[t1, 'BH_q'] = close___bh_adjust(extremes__np.array(p_vals))
    return table

def extremes__shape_terciles(panel: extremes__pd.DataFrame, years: range, scope: str) -> extremes__pd.DataFrame:
    """扩展窗口三分位：切点只用此前年份的弱收盘日。"""
    weak = panel[panel['clv'] < extremes__WEAK_CLOSE].dropna(subset=['outcome'])
    rows = []
    for feature in extremes__SHAPE_FEATURES:
        for year in years:
            train = weak.loc[weak.index.year < year, feature].dropna()
            test = weak.loc[weak.index.year == year].dropna(subset=[feature])
            if len(train) < 60 or test.empty:
                continue
            low, high = train.quantile([1 / 3, 2 / 3])
            if not low < high:
                raise ValueError(f'{scope} {feature} {year} 三分位切点相等')
            group = extremes__np.where(test[feature] <= low, '低', extremes__np.where(test[feature] <= high, '中', '高'))
            for name in ('低', '中', '高'):
                part = test.loc[group == name, 'outcome']
                rows.append({'样本': scope, '特征': feature, '年份': int(year), '组': name, '天数': len(part), '反弹率_%': float((part > 0).mean() * 100) if len(part) else extremes__np.nan, '均值_bp': float(part.mean() * 10000.0) if len(part) else extremes__np.nan, '中位_bp': float(part.median() * 10000.0) if len(part) else extremes__np.nan})
    return extremes__pd.DataFrame(rows)

def extremes___tercile_summary(groups: extremes__pd.DataFrame) -> extremes__pd.DataFrame:
    rows = []
    for (scope, feature), part in groups.groupby(['样本', '特征'], sort=False):
        pooled = {}
        for name in ('低', '中', '高'):
            g = part[part['组'] == name]
            days = g['天数'].sum()
            pooled[name] = (days, (g['均值_bp'] * g['天数']).sum() / days, (g['反弹率_%'] * g['天数']).sum() / days)
        spread = part.pivot(index='年份', columns='组', values='均值_bp')
        diff = spread['高'] - spread['低']
        rows.append({'样本': scope, '特征': feature, **{f'{k}组_天数': v[0] for k, v in pooled.items()}, **{f'{k}组_均值_bp': v[1] for k, v in pooled.items()}, **{f'{k}组_反弹率_%': v[2] for k, v in pooled.items()}, '高减低为正年份': f'{int((diff > 0).sum())}/{int(diff.notna().sum())}'})
    return extremes__pd.DataFrame(rows)

def extremes___liquidity_link(panel: extremes__pd.DataFrame) -> extremes__pd.DataFrame:
    from download.index.getQvix import load_research_qvix
    l1 = extremes__pd.read_csv(extremes__C.study_dir('index', 'index_close_location_mechanism') / 'daily.csv', parse_dates=['trading_date']).set_index('trading_date')
    extremes__C.assert_no_holdout_dates(l1.index, what='L1 日汇总')
    qvix = load_research_qvix().set_index('date')['close']
    q = qvix.groupby(qvix.index.year).transform(lambda s: (s - s.mean()) / s.std())
    flows = extremes__pd.read_csv(extremes__C.INDEX_MARKET_FLOWS_PATH, parse_dates=['date']).set_index('date')
    extremes__C.assert_no_holdout_dates(flows.index, what='市场资金流')
    margin_net = (flows['buy_on_margin_value'] - flows['margin_repayment']) / flows['margin_balance'].shift(1)
    proxies = extremes__pd.DataFrame({'尾盘价差比': l1['spread_ratio'], '尾盘深度比': l1['depth_ratio'], 'QVIX年内z': q, '融资净买入': margin_net}).reindex(panel.index)
    weak = panel[panel['clv'] < extremes__WEAK_CLOSE].join(proxies)
    rows = []
    for feature in extremes__SHAPE_FEATURES:
        row = {'特征': feature}
        for name in proxies.columns:
            pair = weak[[feature, name]].dropna()
            row[f'与{name}相关'] = float(pair[feature].rank().corr(pair[name].rank())) if len(pair) > 30 else extremes__np.nan
        sub = weak.dropna(subset=['尾盘价差比', 'QVIX年内z', f'{feature}_z', 'outcome'])
        coef, t_stat, nobs = extremes___fit(sub['outcome'], [sub[f'{feature}_z'], sub['clv'], sub['尾盘价差比'], sub['QVIX年内z']])
        base, base_t, _ = extremes___fit(sub['outcome'], [sub[f'{feature}_z'], sub['clv']])
        row.update({'同样本斜率_bp': base * 10000.0, '同样本_t': base_t, '控制价差与QVIX后斜率_bp': coef * 10000.0, '控制后_t': t_stat, '天数': nobs})
        rows.append(row)
    return extremes__pd.DataFrame(rows)

def extremes___early_if_minutes() -> extremes__pd.DataFrame:
    early = extremes__pd.read_parquet(extremes__C.INDEX_DATA_ROOT / 'if_before_ih_ic' / 'minute_shards' / 'IF.parquet', columns=list(extremes__SHAPE_FIELDS), filters=[('trading_date', '<=', extremes__pd.Timestamp('2015-04-16').to_pydatetime())])
    research = extremes__pd.read_parquet(extremes__C.INDEX_RESEARCH_DIR / 'IF.parquet', columns=list(extremes__SHAPE_FIELDS), filters=[('trading_date', '<=', extremes__pd.Timestamp('2015-04-16').to_pydatetime())])
    early = early.reset_index() if 'datetime' not in early.columns else early
    research = research.reset_index() if 'datetime' not in research.columns else research
    for frame, label in ((early, '早期 IF'), (research, '研究期 IF')):
        extremes__C.assert_no_holdout_dates(extremes__pd.to_datetime(frame['trading_date']), what=label)
    overlap = sorted(set(extremes__pd.to_datetime(early['trading_date'])) & set(extremes__pd.to_datetime(research['trading_date'])))
    if not overlap:
        raise ValueError('早期 IF 与研究期 IF 没有重叠日期')
    key = ['datetime', 'open', 'close', 'openw', 'closew', 'highw', 'loww']
    a = early[extremes__pd.to_datetime(early['trading_date']).isin(overlap)].sort_values('datetime')[key].reset_index(drop=True)
    b = research[extremes__pd.to_datetime(research['trading_date']).isin(overlap)].sort_values('datetime')[key].reset_index(drop=True)
    extremes__pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-08, atol=1e-08, check_dtype=False, obj='早期与研究期 IF 重叠日分钟')
    return early

def extremes__run_shape() -> None:
    protocol = extremes__load_shape_protocol()
    signs = protocol['parameters']['predicted_sign']
    extremes__C.assert_research_only(extremes__C.INDEX_RESEARCH_DIR)
    per_symbol = {}
    for symbol in extremes__SYMBOLS:
        minute = extremes__pd.read_parquet(extremes__C.INDEX_RESEARCH_DIR / f'{symbol}.parquet', columns=list(extremes__SHAPE_FIELDS), filters=[('trading_date', '<=', extremes__SAMPLE_END.to_pydatetime())])
        per_symbol[symbol] = extremes__shape_from_minutes(minute, symbol)
    common = extremes__shape_panel(per_symbol)
    common = common[(common.index >= extremes__COMMON_START) & (common.index <= extremes__SAMPLE_END)]
    early = extremes__shape_panel({'IF': extremes__shape_from_minutes(extremes___early_if_minutes(), 'IF')})
    early = early[(early.index >= extremes__IF_REPLICATION[0]) & (early.index <= extremes__IF_REPLICATION[1])]
    extremes__C.assert_no_holdout_dates(common.index, what='共同市场形态计分日')
    extremes__C.assert_no_holdout_dates(early.index, what='早期 IF 形态计分日')
    tests = extremes__pd.concat([extremes__shape_tests(common, 'IF/IH/IC 2015-2021', signs), extremes__shape_tests(early, 'IF 2011-2015.04', signs)], ignore_index=True)
    groups = extremes__pd.concat([extremes__shape_terciles(common, range(2016, 2022), 'IF/IH/IC 2015-2021'), extremes__shape_terciles(early, range(2012, 2016), 'IF 2011-2015.04')], ignore_index=True)
    summary = extremes___tercile_summary(groups)
    weak_common = common[common['clv'] < extremes__WEAK_CLOSE]
    corr = weak_common[[*extremes__SHAPE_FEATURES, 'high_open_z', 'intraday_return_z', 'range_open_z']].corr(method='spearman')
    liquidity = extremes___liquidity_link(common)
    extremes__SHAPE_OUT.mkdir(parents=True, exist_ok=True)
    tests.to_csv(extremes__SHAPE_OUT / 'tests.csv', index=False, encoding='utf-8-sig')
    groups.to_csv(extremes__SHAPE_OUT / 'terciles_by_year.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(extremes__SHAPE_OUT / 'terciles.csv', index=False, encoding='utf-8-sig')
    corr.to_csv(extremes__SHAPE_OUT / 'weak_close_correlations.csv', encoding='utf-8-sig')
    liquidity.to_csv(extremes__SHAPE_OUT / 'liquidity_link.csv', index=False, encoding='utf-8-sig')
    extremes__pd.concat({'common': common, 'if_early': early}).to_csv(extremes__SHAPE_OUT / 'daily_panel.csv', encoding='utf-8-sig')
    coverage = {'共同市场': [str(common.index.min().date()), str(common.index.max().date()), int((common['clv'] < extremes__WEAK_CLOSE).sum())], '早期IF': [str(early.index.min().date()), str(early.index.max().date()), int((early['clv'] < extremes__WEAK_CLOSE).sum())], 'dfp首个有效日_共同': str(common['dfp_top3'].dropna().index.min().date())}
    (extremes__SHAPE_OUT / 'protocol.json').write_text(extremes__json.dumps({'protocol': protocol, 'coverage': coverage}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    extremes__pd.set_option('display.width', 260)
    extremes__pd.set_option('display.max_columns', 30)
    print(tests.round(3).to_string(index=False))
    print(summary.round(1).to_string(index=False))
    print(corr.round(2).to_string())
    print(liquidity.round(3).to_string(index=False))
    print(coverage)
    print(f'结果：{extremes__SHAPE_OUT}')
# 价格路径与成交量持续期诊断。
import json as path_volume__json
from pathlib import Path as path_volume__Path
import numpy as path_volume__np
import pandas as path_volume__pd
from scipy.stats import t as path_volume__student_t
from research.time_factor_screen import ols_hac as path_volume__ols_hac
from tfcta import config as path_volume__C
from tfcta.data import load_shard as path_volume__load_shard
from tfcta.factors import gtja_index_factors as path_volume__gtja_index_factors
path_volume__ROOT = path_volume__Path(__file__).resolve().parents[2]
path_volume__CONFIG_PATH = path_volume__ROOT / 'config' / 'research' / 'index_path_volume_discriminators.json'
path_volume__EARLY_IF_PATH = path_volume__C.INDEX_DATA_ROOT / 'if_before_ih_ic' / 'minute_shards' / 'IF.parquet'
path_volume__OUT = path_volume__C.study_dir('index', 'index_path_volume_discriminators')
path_volume__SYMBOLS = ('IF', 'IH', 'IC')
path_volume__FEATURES = ('high_open', 'Volume_duration_return')
path_volume__MODELS = {'CLV基线': ('clv', 'weak'), 'CLV+高点−开盘': ('clv', 'weak', 'high_open', 'weak_x_high_open'), 'CLV+成交量持续期收益': ('clv', 'weak', 'Volume_duration_return', 'weak_x_Volume_duration_return'), 'CLV+两候选': ('clv', 'weak', 'high_open', 'weak_x_high_open', 'Volume_duration_return', 'weak_x_Volume_duration_return')}
path_volume__REQUIRED_FIELDS = ('trading_date', 'open', 'openw', 'close', 'closew', 'highw', 'loww', 'volume', 'total_turnover')
path_volume__WEAK_CLOSE = 0.4
path_volume__HIGH_OPEN_WINDOW = 60
path_volume__HIGH_OPEN_MINIMUM = 40
path_volume__VOLUME_RETURN_WINDOW = 250
path_volume__VOLUME_RETURN_MINIMUM = 125
path_volume__HAC_LAGS = 5
path_volume__LOGISTIC_C = 1.0
path_volume__EARLY_START = path_volume__pd.Timestamp('2011-01-04')
path_volume__EARLY_END = path_volume__pd.Timestamp('2015-04-14')
path_volume__EARLY_EVALUATION_YEARS = (2012, 2013, 2014, 2015)
path_volume__COMMON_START = path_volume__pd.Timestamp('2015-04-16')
path_volume__COMMON_EVALUATION_YEARS = (2016, 2017, 2018, 2019, 2020, 2021)
path_volume__COMMON_RESULT_END = path_volume__pd.Timestamp('2021-12-31')

def path_volume__load_protocol(path: path_volume__Path=path_volume__CONFIG_PATH) -> dict:
    protocol = path_volume__json.loads(path.read_text(encoding='utf-8'))
    sample = protocol.get('sample', '')
    parameters = protocol.get('parameters', {})
    definitions = protocol.get('definitions', {})
    if protocol.get('status') != 'declared_test_not_a_strategy' or '不读 2022' not in sample or '不读 2026' not in sample or ('不读取任何 OOS 分区' not in sample) or (parameters.get('instruments') != list(path_volume__SYMBOLS)) or (parameters.get('weak_close_threshold') != path_volume__WEAK_CLOSE) or (parameters.get('high_open_rolling_window') != path_volume__HIGH_OPEN_WINDOW) or (parameters.get('high_open_rolling_minimum') != path_volume__HIGH_OPEN_MINIMUM) or (parameters.get('volume_duration_price_threshold') != 3.0) or (parameters.get('volume_duration_volume_threshold') != 50.0) or (parameters.get('volume_return_rolling_window') != path_volume__VOLUME_RETURN_WINDOW) or (parameters.get('volume_return_rolling_minimum') != path_volume__VOLUME_RETURN_MINIMUM) or (parameters.get('hac_lags') != path_volume__HAC_LAGS) or (parameters.get('logistic_C') != path_volume__LOGISTIC_C) or (parameters.get('early_regression_start') != path_volume__EARLY_START.date().isoformat()) or (parameters.get('early_regression_end') != path_volume__EARLY_END.date().isoformat()) or (parameters.get('early_evaluation_years') != list(path_volume__EARLY_EVALUATION_YEARS)) or (parameters.get('common_training_start') != path_volume__COMMON_START.date().isoformat()) or (parameters.get('common_evaluation_years') != list(path_volume__COMMON_EVALUATION_YEARS)) or (parameters.get('common_result_end') != path_volume__COMMON_RESULT_END.date().isoformat()) or ('与弱收盘指示变量的交互' not in definitions.get('models', '')) or ('此前所有有效交易日' not in definitions.get('validation', '')) or ('只在该折弱收盘日评估' not in definitions.get('validation', '')):
        raise ValueError('路径与成交量判别检验协议和实现或样本边界不匹配')
    return protocol

def path_volume___standardize_prior(values: path_volume__pd.Series, window: int, minimum: int) -> path_volume__pd.Series:
    prior = values.shift(1)
    mean = prior.rolling(window, min_periods=minimum).mean()
    scale = prior.rolling(window, min_periods=minimum).std()
    return (values - mean) / scale.where(scale > 0)

def path_volume__symbol_panel(minute: path_volume__pd.DataFrame, symbol: str) -> path_volume__pd.DataFrame:
    missing = sorted(set(path_volume__REQUIRED_FIELDS) - set(minute.columns))
    if missing:
        raise KeyError(f'{symbol} 分钟分片缺少字段：{missing}')
    frame = minute.sort_index(kind='mergesort').copy()
    frame['trading_date'] = path_volume__pd.to_datetime(frame['trading_date']).dt.normalize()
    path_volume__C.assert_no_holdout_dates(frame['trading_date'], what=f'{symbol}路径量能分钟数据')
    if frame[['open', 'openw', 'close', 'closew', 'highw', 'loww', 'volume']].isna().any().any():
        raise ValueError(f'{symbol}路径量能输入包含缺失价格或成交量')
    grouped = frame.groupby('trading_date', sort=True)
    daily = grouped.agg(open=('open', 'first'), openw=('openw', 'first'), close=('close', 'last'), closew=('closew', 'last'), highw=('highw', 'max'), loww=('loww', 'min'))
    if daily['open'].le(0).any() or daily['highw'].le(daily['loww']).any():
        raise ValueError(f'{symbol}日线开盘价或高低区间无效')
    report_factors = path_volume__gtja_index_factors(frame, price_threshold=3.0, volume_threshold=50.0)
    raw_volume_return = report_factors['Volume_duration_return'].reindex(daily.index)
    high_open = (daily['highw'] - daily['openw']) / daily['open'] * 10000.0
    high_open_z = path_volume___standardize_prior(high_open, path_volume__HIGH_OPEN_WINDOW, path_volume__HIGH_OPEN_MINIMUM)
    volume_return_z = path_volume___standardize_prior(raw_volume_return, path_volume__VOLUME_RETURN_WINDOW, path_volume__VOLUME_RETURN_MINIMUM)
    outcome = (daily['openw'].shift(-2) - daily['openw'].shift(-1)) / daily['open'].shift(-1).where(daily['open'].shift(-1) > 0)
    clv = (daily['closew'] - daily['loww']) / (daily['highw'] - daily['loww'])
    return path_volume__pd.DataFrame({'clv': clv, 'outcome_return': outcome, 'high_open': high_open_z, 'Volume_duration_return': volume_return_z}, index=daily.index).replace([path_volume__np.inf, -path_volume__np.inf], path_volume__np.nan)

def path_volume__market_panel(panels: dict[str, path_volume__pd.DataFrame], start: path_volume__pd.Timestamp, end: path_volume__pd.Timestamp) -> path_volume__pd.DataFrame:
    symbols = tuple(panels)
    if symbols not in (('IF',), path_volume__SYMBOLS):
        raise ValueError(f'市场面板必须是单独 IF 或完整 {path_volume__SYMBOLS}，收到 {symbols}')
    dates = panels[symbols[0]].index
    for symbol in symbols[1:]:
        dates = dates.intersection(panels[symbol].index)
    dates = dates[(dates >= start) & (dates <= end)]
    if dates.empty:
        raise ValueError('指定的路径与成交量研究区间没有交易日')
    path_volume__C.assert_no_holdout_dates(dates, what='路径与成交量市场面板')
    columns = ('clv', 'outcome_return', *path_volume__FEATURES)
    out = path_volume__pd.DataFrame(index=dates)
    for column in columns:
        aligned = path_volume__pd.concat({symbol: panels[symbol][column].reindex(dates) for symbol in symbols}, axis=1)
        out[column] = aligned.mean(axis=1, skipna=False)
    return out.replace([path_volume__np.inf, -path_volume__np.inf], path_volume__np.nan)

def path_volume__walk_forward(panel: path_volume__pd.DataFrame, evaluation_years: tuple[int, ...], sample_name: str) -> tuple[path_volume__pd.DataFrame, path_volume__pd.DataFrame]:
    required = ['clv', 'outcome_return', *path_volume__FEATURES]
    usable = panel.dropna(subset=required).copy()
    usable['weak'] = (usable['clv'] < path_volume__WEAK_CLOSE).astype(float)
    for feature in path_volume__FEATURES:
        usable[f'weak_x_{feature}'] = usable['weak'] * usable[feature]
    usable['label'] = (usable['outcome_return'] > 0).astype(int)
    rows = []
    predictions = []
    for year in evaluation_years:
        train = usable[usable.index.year < year]
        test = usable[(usable.index.year == year) & (usable['weak'] == 1.0)]
        if len(train) < 30 or test.empty or train['label'].nunique() < 2:
            raise ValueError(f'{sample_name} {year} 年训练/预测集不足或训练仅有单一类别')
        for model_name, features in path_volume__MODELS.items():
            probability = close___fit_logistic(train[list(features)].to_numpy(dtype=float), train['label'].to_numpy(dtype=int), test[list(features)].to_numpy(dtype=float))
            actual = test['label'].to_numpy(dtype=int)
            metrics = close___classification_metrics(actual, probability)
            rows.append({'样本': sample_name, '年份': year, '模型': model_name, **metrics})
            predictions.extend(({'样本': sample_name, '年份': year, '模型': model_name, '日期': date, '标签': int(label), '概率': float(prob), '结果收益': float(outcome)} for date, label, prob, outcome in zip(test.index, actual, probability, test['outcome_return'])))
    prediction_table = path_volume__pd.DataFrame(predictions)
    pooled = []
    for model_name, group in prediction_table.groupby('模型', sort=False):
        metrics = close___classification_metrics(group['标签'].to_numpy(), group['概率'].to_numpy())
        pooled.append({'样本': sample_name, '年份': '合并', '模型': model_name, **metrics})
    scores = path_volume__pd.concat([path_volume__pd.DataFrame(rows), path_volume__pd.DataFrame(pooled)], ignore_index=True)
    baseline = scores[scores['模型'] == 'CLV基线'].set_index(['样本', '年份'])
    for metric, delta in (('AUC', 'ΔAUC_vs_CLV'), ('Brier', 'ΔBrier_vs_CLV')):
        scores[delta] = [row[metric] - baseline.loc[(row['样本'], row['年份']), metric] for _, row in scores.iterrows()]
    return (scores, prediction_table)

def path_volume__weak_close_interactions(panel: path_volume__pd.DataFrame, sample_name: str, start: path_volume__pd.Timestamp, end: path_volume__pd.Timestamp) -> path_volume__pd.DataFrame:
    sample = panel.loc[(panel.index >= start) & (panel.index <= end)].copy()
    sample = sample.dropna(subset=['clv', 'outcome_return', *path_volume__FEATURES])
    weak = (sample['clv'] < path_volume__WEAK_CLOSE).astype(float)
    sample['weak'] = weak
    sample['outcome_bp'] = sample['outcome_return'] * 10000.0
    for feature in path_volume__FEATURES:
        sample[f'weak_x_{feature}'] = sample['weak'] * sample[feature]
    other = {'high_open': 'Volume_duration_return', 'Volume_duration_return': 'high_open'}
    rows = []
    for feature in path_volume__FEATURES:
        coefficient, t_stat = path_volume__ols_hac(sample['outcome_bp'].to_numpy(), [sample[f'weak_x_{feature}'].to_numpy(), sample['clv'].to_numpy(), sample['weak'].to_numpy(), sample[feature].to_numpy(), sample[other[feature]].to_numpy(), sample[f'weak_x_{other[feature]}'].to_numpy()], path_volume__HAC_LAGS)
        degrees = len(sample) - 7
        p_value = float(2 * path_volume__student_t.sf(abs(t_stat), degrees)) if degrees > 0 and path_volume__np.isfinite(t_stat) else path_volume__np.nan
        rows.append({'样本': sample_name, '特征': feature, '弱收盘交互_bp每标准差': coefficient, 'NW_t': t_stat, 'p_双侧': p_value, '样本数': len(sample)})
    result = path_volume__pd.DataFrame(rows)
    result['BH_q'] = close___bh_adjust(result['p_双侧'].to_numpy())
    return result

def path_volume___read_early_if() -> path_volume__pd.DataFrame:
    if not path_volume__EARLY_IF_PATH.is_file():
        raise FileNotFoundError(f'缺少已经登记的早期 IF 分片：{path_volume__EARLY_IF_PATH}')
    minute = path_volume__pd.read_parquet(path_volume__EARLY_IF_PATH, columns=list(path_volume__REQUIRED_FIELDS))
    dates = path_volume__pd.to_datetime(minute['trading_date']).dt.normalize()
    path_volume__C.assert_no_holdout_dates(dates, what='早期 IF 路径量能分片')
    if dates.min() < path_volume__pd.Timestamp('2010-04-16') or dates.max() > path_volume__pd.Timestamp('2015-04-16'):
        raise ValueError('早期 IF 路径量能分片超出登记的 2010–2015 边界')
    return minute

def path_volume__run() -> None:
    protocol = path_volume__load_protocol()
    path_volume__C.assert_research_only(path_volume__C.INDEX_RESEARCH_DIR)
    early_minute = path_volume___read_early_if()
    early_by_symbol = {'IF': path_volume__symbol_panel(early_minute, 'IF')}
    early_panel = path_volume__market_panel(early_by_symbol, path_volume__EARLY_START, path_volume__EARLY_END)
    common_by_symbol = {}
    for symbol in path_volume__SYMBOLS:
        minute = path_volume__load_shard(symbol, directory=path_volume__C.INDEX_RESEARCH_DIR, columns=list(path_volume__REQUIRED_FIELDS))
        common_by_symbol[symbol] = path_volume__symbol_panel(minute, symbol)
    common_panel = path_volume__market_panel(common_by_symbol, path_volume__COMMON_START, path_volume__COMMON_RESULT_END)
    early_scores, early_predictions = path_volume__walk_forward(early_panel, path_volume__EARLY_EVALUATION_YEARS, 'IF 2011–2015')
    common_scores, common_predictions = path_volume__walk_forward(common_panel, path_volume__COMMON_EVALUATION_YEARS, 'IF/IH/IC 2016–2021')
    scores = path_volume__pd.concat([early_scores, common_scores], ignore_index=True)
    predictions = path_volume__pd.concat([early_predictions, common_predictions], ignore_index=True)
    interactions = path_volume__pd.concat([path_volume__weak_close_interactions(early_panel, 'IF 2011–2015', path_volume__EARLY_START, path_volume__EARLY_END), path_volume__weak_close_interactions(common_panel, 'IF/IH/IC 2016–2021', path_volume__pd.Timestamp('2016-01-04'), path_volume__pd.Timestamp('2021-12-29'))], ignore_index=True)
    interactions['BH_q'] = close___bh_adjust(interactions['p_双侧'].to_numpy())
    path_volume__OUT.mkdir(parents=True, exist_ok=True)
    (path_volume__OUT / 'protocol.json').write_text(path_volume__json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    scores.to_csv(path_volume__OUT / 'walk_forward.csv', index=False, encoding='utf-8-sig')
    predictions.to_csv(path_volume__OUT / 'predictions.csv', index=False, encoding='utf-8-sig')
    interactions.to_csv(path_volume__OUT / 'weak_close_interactions.csv', index=False, encoding='utf-8-sig')
    print(scores.round(6).to_string(index=False))
    print(interactions.round(6).to_string(index=False))
    print(f'结果目录：{path_volume__OUT}')

def main() -> None:
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Index weak-close and price-path diagnostics")
    parser.add_argument("study", choices=("close", "price-extremes", "path-volume"))
    parser.add_argument("--shape", action="store_true", help="Run price-path shape diagnostics")
    args, remainder = parser.parse_known_args()
    sys.argv = [f"research.index.weak_close_diagnostics.{args.study}", *remainder]
    if args.study == "close": close__run()
    elif args.study == "path-volume": path_volume__run()
    elif args.shape: extremes__run_shape()
    else: extremes__run()


_bh_adjust = close___bh_adjust
close_features = close__FEATURES
close_symbol_panel = close__symbol_panel
range_surprise_group_predictions = close___range_surprise_group_predictions
FEATURES = path_volume__FEATURES
load_protocol = path_volume__load_protocol
symbol_panel = path_volume__symbol_panel
aggregate_daily = extremes__aggregate_daily
daily_features = extremes__daily_features
shape_from_minutes = extremes__shape_from_minutes
ROLLING_MINIMUM = extremes__ROLLING_MINIMUM
ROLLING_WINDOW = extremes__ROLLING_WINDOW
REQUIRED_FIELDS = extremes__REQUIRED_FIELDS

if __name__ == "__main__":
    main()
