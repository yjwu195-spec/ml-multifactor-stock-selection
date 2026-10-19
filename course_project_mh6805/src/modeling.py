"""Purged chronological selection, shared-sample evaluation and factor ablations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from .data import FAMILIES, FEATURES


class TrainWinsorizer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        frame = pd.DataFrame(x).replace([np.inf, -np.inf], np.nan)
        self.lower_ = frame.quantile(.01).to_numpy()
        self.upper_ = frame.quantile(.99).to_numpy()
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=float).copy()
        values[~np.isfinite(values)] = np.nan
        return np.clip(values, self.lower_, self.upper_)


def make_model(name, params, cfg):
    common = {'random_state': cfg['seed']}
    if name == 'Logistic Regression':
        model = LogisticRegression(**common, max_iter=1500, **params)
    elif name == 'Random Forest':
        model = RandomForestClassifier(**common, n_jobs=cfg['n_jobs'], max_features='sqrt', **params)
    elif name == 'XGBoost':
        model = XGBClassifier(**common, n_jobs=cfg['n_jobs'], objective='binary:logistic', eval_metric='logloss', tree_method='hist', **params)
    else:
        model = LGBMClassifier(**common, n_jobs=cfg['n_jobs'], verbosity=-1, subsample_freq=1, **params)
    return Pipeline([('winsor', TrainWinsorizer()), ('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler()), ('model', model)])


def candidate_grid():
    return {
        'Logistic Regression': [{'C': x} for x in [.01, .1, 1]],
        'Random Forest': [dict(n_estimators=160, max_depth=d, min_samples_leaf=l) for d,l in [(4,100), (6,50), (8,100)]],
        'XGBoost': [dict(n_estimators=n, max_depth=d, learning_rate=.03, min_child_weight=50, subsample=.75, colsample_bytree=.75, reg_alpha=1., reg_lambda=10.) for n,d in [(100,2),(160,2),(120,3)]],
        'LightGBM': [dict(n_estimators=n, max_depth=d, num_leaves=2**d-1, learning_rate=.03, min_child_samples=200, subsample=.75, colsample_bytree=.75, reg_alpha=1., reg_lambda=10.) for n,d in [(100,2),(160,2),(120,3)]]
    }


def split_panel(panel, cfg):
    dates = pd.DatetimeIndex(panel.signal_date.unique()).sort_values()
    n = len(dates)
    a, b, c = [int(n * sum(cfg['split_ratios'][:i])) for i in [1,2,3]]
    groups = dict(zip(['train', 'validation', 'test', 'backtest'], [dates[:a],dates[a:b],dates[b:c],dates[c:]]))
    out = panel.copy()
    out['split'] = ''
    meta = {}
    for name, group in groups.items():
        out.loc[out.signal_date.isin(group), 'split'] = name
        meta[name] = {'signal_start': str(group[0].date()), 'signal_end': str(group[-1].date()), 'dates': len(group)}
    return out, meta


def classification(y, score):
    pred = score >= .5
    return {'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred), 'f1': f1_score(y, pred, zero_division=0), 'auc': roc_auc_score(y, score), 'log_loss': log_loss(y, score, labels=[0,1])}


def ranking(frame):
    rows = []
    for date, group in frame.groupby('signal_date'):
        usable = group.dropna(subset=['forward_return', 'score'])
        if len(usable) < 20:
            continue
        rows.append({'signal_date': date, 'rank_ic': usable.score.corr(usable.excess_return, method='spearman'), 'coverage': len(usable) / len(group)})
    return pd.DataFrame(rows)


def run_models(root: Path, cfg):
    panel = pd.read_parquet(root / 'data/processed/panel.parquet')
    panel, metadata = split_panel(panel, cfg)
    train_end = pd.Timestamp(metadata['validation']['signal_start'])
    test_start = pd.Timestamp(metadata['test']['signal_start'])
    train = panel[(panel.split == 'train') & panel.label.notna() & (panel.exit_date < train_end)].copy()
    validation = panel[(panel.split == 'validation') & panel.label.notna() & (panel.exit_date < test_start)].copy()
    final_train = pd.concat([train, validation], ignore_index=True)
    test = panel[(panel.split == 'test') & panel.label.notna()].copy()
    for name in metadata:
        selected = panel[panel.split == name]
        metadata[name].update({'inference_rows': len(selected), 'labeled_rows': int(selected.label.notna().sum())})
    metadata['final_training_rows_after_purge'] = len(final_train)
    metadata['training_only_rows_after_purge'] = len(train)
    (root / 'results/splits.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    panel[['signal_date','ts_code','split','execution_date','exit_date','label']].to_parquet(root / 'results/sample_assignments.parquet', index=False)
    dates = pd.DatetimeIndex(train.signal_date.unique()).sort_values()
    folds = []
    for ti, vi in TimeSeriesSplit(n_splits=cfg['cv_folds']).split(dates):
        vdates = dates[vi]
        tr = train[train.signal_date.isin(dates[ti]) & (train.exit_date < vdates[0])]
        va = train[train.signal_date.isin(vdates)]
        folds.append((tr, va))
    search = []
    chosen = {}
    fitted = {}
    stats = []
    prediction_parts = []
    importance_rows = []
    for name, candidates in candidate_grid().items():
        print(f'[MODEL] tuning {name}', flush=True)
        local = []
        for i, params in enumerate(candidates):
            fold_losses = []
            for fold, (tr, va) in enumerate(folds):
                model = make_model(name, params, cfg)
                model.fit(tr[FEATURES], tr.label.astype(int))
                prob = model.predict_proba(va[FEATURES])[:,1]
                loss = log_loss(va.label, prob, labels=[0,1])
                fold_losses.append(loss)
                search.append({'model': name, 'candidate': i+1, 'phase': 'training_time_cv', 'fold': fold+1, 'log_loss': loss, 'train_rows': len(tr), 'validation_rows': len(va), 'params': json.dumps(params)})
            model = make_model(name, params, cfg)
            model.fit(train[FEATURES], train.label.astype(int))
            val = classification(validation.label, model.predict_proba(validation[FEATURES])[:,1])
            local.append({'candidate': i+1, 'params': params, 'cv_log_loss': float(np.mean(fold_losses)), 'validation_log_loss': float(val['log_loss']), 'validation_accuracy': float(val['accuracy'])})
        # Freeze selection before test/backtest; primary objective is training-only time-CV loss.
        best = min(local, key=lambda x: (x['cv_log_loss'], x['validation_log_loss'], x['candidate']))
        chosen[name] = {'selection': 'minimum mean purged training time-CV log loss; validation loss tie-break', 'selected': best, 'candidates': local}
        model = make_model(name, best['params'], cfg)
        model.fit(final_train[FEATURES], final_train.label.astype(int))
        fitted[name] = model
        dump(model, root / 'results' / (name.lower().replace(' ', '_') + '.joblib'))
        for split in ['test', 'backtest']:
            subset = panel[panel.split == split].copy()
            subset['score'] = model.predict_proba(subset[FEATURES])[:,1]
            subset['model'] = name
            prediction_parts.append(subset)
            labeled = subset[subset.label.notna()]
            measure = classification(labeled.label, labeled.score)
            ranks = ranking(subset)
            measure.update({'rank_ic_mean': ranks.rank_ic.mean(), 'rank_ic_std': ranks.rank_ic.std(), 'rank_ic_ir': ranks.rank_ic.mean()/ranks.rank_ic.std(), 'forward_return_coverage': ranks.coverage.mean()})
            stats.append({'model': name, 'split': split, **measure})
            pd.DataFrame(confusion_matrix(labeled.label, labeled.score >= .5), index=['actual_low','actual_high'], columns=['pred_low','pred_high']).to_csv(root / 'results' / (name.lower().replace(' ','_') + '_' + split + '_confusion.csv'))
        sample = test.sample(min(len(test), cfg['permutation_sample_size']), random_state=cfg['seed'])
        result = permutation_importance(model, sample[FEATURES], sample.label.astype(int), scoring='neg_log_loss', n_repeats=cfg['permutation_repeats'], random_state=cfg['seed'], n_jobs=1)
        for j, feature in enumerate(FEATURES):
            importance_rows.append({'model': name, 'feature': feature, 'importance': result.importances_mean[j], 'std': result.importances_std[j], 'interpretation_sample': len(sample)})
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_parquet(root / 'results/predictions.parquet', index=False)
    pd.DataFrame(stats).to_csv(root / 'results/prediction_metrics.csv', index=False)
    pd.DataFrame(search).to_csv(root / 'results/cv_search.csv', index=False)
    pd.DataFrame(importance_rows).to_csv(root / 'results/permutation_importance.csv', index=False)
    (root / 'results/selected_parameters.json').write_text(json.dumps(chosen, indent=2), encoding='utf-8')
    ablations = []
    variants = {'all_factors': FEATURES, 'fundamentals_only': sum([FAMILIES[x] for x in ['valuation','growth','quality','leverage_liquidity']], []), 'market_only': sum([FAMILIES[x] for x in ['momentum','trading_risk','size']], [])}
    variants.update({'without_' + family: [x for x in FEATURES if x not in members] for family,members in FAMILIES.items()})
    test_full = panel[panel.split == 'test'].copy()
    ablation_predictions = []
    for name in fitted:
        print(f'[MODEL] factor ablations {name}', flush=True)
        for variant, cols in variants.items():
            if variant == 'all_factors':
                model = fitted[name]
            else:
                model = make_model(name, chosen[name]['selected']['params'], cfg)
                model.fit(final_train[cols], final_train.label.astype(int))
            subset = test_full.copy()
            subset['score'] = model.predict_proba(subset[cols])[:,1]
            labeled = subset[subset.label.notna()]
            rank = ranking(subset)
            ablations.append({'model': name, 'variant': variant, 'features': len(cols), **classification(labeled.label, labeled.score), 'rank_ic': rank.rank_ic.mean()})
            subset['model'] = name
            subset['variant'] = variant
            ablation_predictions.append(subset[['signal_date','execution_date','ts_code','model','variant','score']])
    pd.DataFrame(ablations).to_csv(root / 'results/factor_ablations.csv', index=False)
    pd.concat(ablation_predictions, ignore_index=True).to_parquet(root / 'results/ablation_predictions.parquet', index=False)
    training_diagnostics(root, cfg, panel)
    print('[MODEL] completed all shared-sample model and factor comparisons', flush=True)


def training_diagnostics(root, cfg, panel=None):
    if panel is None:
        panel, _ = split_panel(pd.read_parquet(root / 'data/processed/panel.parquet'), cfg)
    chosen = json.loads((root / 'results/selected_parameters.json').read_text())
    valstart = panel.loc[panel.split == 'validation', 'signal_date'].min()
    teststart = panel.loc[panel.split == 'test', 'signal_date'].min()
    tr = panel[(panel.split == 'train') & panel.label.notna() & (panel.exit_date < valstart)]
    va = panel[(panel.split == 'validation') & panel.label.notna() & (panel.exit_date < teststart)]
    rows = []
    for name, selection in chosen.items():
        model = make_model(name, selection['selected']['params'], cfg)
        model.fit(tr[FEATURES], tr.label.astype(int))
        for label, data in [('train_before_refit',tr),('validation_before_refit',va)]:
            rows.append({'model': name, 'sample': label, **classification(data.label, model.predict_proba(data[FEATURES])[:,1])})
    pd.DataFrame(rows).to_csv(root / 'results/training_diagnostics.csv', index=False)
