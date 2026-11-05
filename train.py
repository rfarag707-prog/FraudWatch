"""Reproducible PaySim experiment. Run: python train.py --data data/paysim.csv"""
import argparse
import json
import platform
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, PrecisionRecallDisplay, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from core import RAW, NUM, validate, feature_step

SEED = 42

def load_sample(path, max_rows=300000):
    """Uniform random-priority reservoir over ALL chunks, independent of labels."""
    if max_rows < 1000:
        raise ValueError('Use at least 1,000 rows, normally 300,000 or more.')
    rng = np.random.default_rng(SEED)
    pool = None
    seen = 0
    for chunk in pd.read_csv(path, chunksize=100000, usecols=lambda c: c in RAW + ['isFraud']):
        chunk = validate(chunk, labeled=True)
        seen += len(chunk)
        chunk['_priority'] = rng.random(len(chunk))
        pool = chunk if pool is None else pd.concat([pool, chunk], ignore_index=True)
        if len(pool) > max_rows:
            pool = pool.nsmallest(max_rows, '_priority')
    if pool is None:
        raise ValueError('Empty dataset.')
    return pool.drop(columns='_priority').sort_values('step', kind='stable').reset_index(drop=True), seen

def check_classes(d, name, minimum=5):
    counts = d.isFraud.value_counts()
    if len(counts) != 2 or counts.min() < minimum:
        raise ValueError(f'{name}: fewer than {minimum} examples of a class. Increase --max-rows; do not balance the test set.')

def split_time(d):
    """Split entire time steps, so the same simulated hour never crosses a boundary."""
    steps = np.sort(d.step.unique())
    if len(steps) < 20:
        raise ValueError('Need at least 20 distinct time steps for temporal validation.')
    first, second = steps[int(.60 * len(steps))], steps[int(.80 * len(steps))]
    parts = [d[d.step < first].copy(), d[(d.step >= first) & (d.step < second)].copy(), d[d.step >= second].copy()]
    for name, part in zip(['train', 'validation', 'test'], parts):
        check_classes(part, name)
    return parts

def time_folds(train):
    steps = np.sort(train.step.unique())
    folds = []
    for a, b in TimeSeriesSplit(n_splits=3).split(steps):
        ia = np.flatnonzero(train.step.isin(steps[a]).to_numpy())
        ib = np.flatnonzero(train.step.isin(steps[b]).to_numpy())
        check_classes(train.iloc[ia], 'CV training fold', minimum=2)
        check_classes(train.iloc[ib], 'CV validation fold', minimum=2)
        folds.append((ia, ib))
    return folds

def pipeline(model):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler())])
    prep = ColumnTransformer([('numeric', numeric, NUM),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['type'])])
    return Pipeline([('features', feature_step()), ('preprocess', prep), ('model', model)])

def metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(average_precision=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)), precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(threshold))

def choose_threshold(y, p, fn_cost, fp_cost):
    """Evaluate all unique scores using sorted cumulative counts, not a coarse grid."""
    order = np.argsort(-np.asarray(p), kind='stable')
    prob = np.asarray(p)[order]
    labels = np.asarray(y)[order]
    ends = np.r_[np.flatnonzero(np.diff(prob) != 0), len(prob)-1]
    tp = np.cumsum(labels)[ends]
    fp = ends + 1 - tp
    fn = labels.sum() - tp
    table = pd.DataFrame({'threshold': prob[ends], 'tp': tp, 'fp': fp, 'fn': fn})
    # Explicit no-review option, including models that emit exactly 1.0.
    table = pd.concat([pd.DataFrame([dict(threshold=np.nextafter(1., 2.), tp=0, fp=0, fn=labels.sum())]), table], ignore_index=True)
    table['assumed_cost'] = table.fn * fn_cost + table.fp * fp_cost
    table['precision'] = table.tp / (table.tp + table.fp).replace(0, np.nan)
    table['recall'] = table.tp / labels.sum()
    # A tie chooses the higher threshold and fewer reviews.
    best = table.sort_values(['assumed_cost', 'threshold'], ascending=[True, False]).iloc[0]
    return float(best.threshold), table

def eda(train, out):
    summary = train.describe(include='all').astype(str)
    summary.to_csv(out / 'training_summary.csv')
    train.isna().sum().rename('missing').to_csv(out / 'missing_values.csv')
    bytype = train.groupby('type').isFraud.agg(['count', 'sum', 'mean'])
    bytype.to_csv(out / 'fraud_by_type.csv')
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    train.isFraud.value_counts().sort_index().plot.bar(ax=axs[0], color=['#203B5C', '#E69442'])
    axs[0].set(title='Training class balance', xlabel='isFraud', ylabel='Transactions', yscale='log')
    bytype['mean'].plot.bar(ax=axs[1], color='#0F9E91')
    axs[1].set(title='Fraud rate by transaction type', ylabel='Fraction')
    for label, color in [(0, '#203B5C'), (1, '#E69442')]:
        axs[2].hist(np.log1p(train.loc[train.isFraud == label, 'amount']), bins=40, density=True, alpha=.55, label=str(label), color=color)
    axs[2].set(title='Amount distribution by label', xlabel='log(1 + amount)', ylabel='Density')
    axs[2].legend(title='isFraud')
    fig.tight_layout(); fig.savefig(out / 'eda.png', dpi=160); plt.close(fig)
    corr = train[['amount', 'oldbalanceOrg', 'oldbalanceDest', 'step', 'isFraud']].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha='right')
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(im, ax=ax); ax.set_title('Training correlations: association, not causation')
    fig.tight_layout(); fig.savefig(out / 'correlations.png', dpi=160); plt.close(fig)
    common = bytype['count'].idxmax()
    risky = bytype['mean'].idxmax()
    insights = [f'Training fraud prevalence: {train.isFraud.mean():.4%}. Accuracy alone can conceal missed fraud.',
        f'{common} is the most common transaction type in the training sample.',
        f'{risky} has the highest observed training fraud rate ({bytype.loc[risky, "mean"]:.4%}); this is an association.',
        f'Training missing balance cells: {int(train[["oldbalanceOrg", "oldbalanceDest"]].isna().sum().sum())}. Imputation fits inside each fold.',
        'Large transfers may be legitimate. Amount alone does not establish fraud. Inspect the class-conditional chart.']
    (out / 'insights.md').write_text('\n'.join('- ' + s for s in insights), encoding='utf-8')
    return insights

def run(data_path, output='outputs', max_rows=300000, jobs=2, fn_cost=25., fp_cost=1.):
    if fn_cost <= 0 or fp_cost <= 0:
        raise ValueError('Both hypothetical costs must be positive.')
    out = Path(output)
    if (out / 'model.joblib').exists() or (out / 'report.json').exists():
        raise ValueError('Output already contains a run. Choose a NEW --output directory to preserve the test result.')
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print('Reading all chunks and selecting a uniform sample...', flush=True)
    d, seen = load_sample(data_path, max_rows)
    train, val, test = split_time(d)
    folds = time_folds(train)
    insights = eda(train, out)
    x, y = train[RAW], train.isFraud
    specs = {
        'Logistic regression': (LogisticRegression(max_iter=1500, class_weight='balanced', random_state=SEED), {'model__C': [.1, 1.]}),
        'Random forest': (RandomForestClassifier(n_estimators=120, class_weight='balanced_subsample', n_jobs=1, random_state=SEED), {'model__max_depth': [10, 18], 'model__min_samples_leaf': [5]}),
        'Histogram gradient boosting': (HistGradientBoostingClassifier(max_iter=150, early_stopping=False, random_state=SEED), {'model__max_leaf_nodes': [15, 31], 'model__l2_regularization': [1.]})}
    trained = {'Baseline': pipeline(DummyClassifier(strategy='prior')).fit(x, y)}
    rows, params = [], {}
    for name, (est, grid) in specs.items():
        print('Training and tuning:', name, flush=True)
        search = GridSearchCV(pipeline(est), grid, scoring='average_precision', cv=folds,
            n_jobs=jobs, error_score='raise', refit=True)
        search.fit(x, y)
        trained[name] = search.best_estimator_
        params[name] = search.best_params_
        rows.append({'model': name, 'cv_average_precision': float(search.best_score_),
                     'cv_std': float(search.cv_results_['std_test_score'][search.best_index_])})
        pd.DataFrame(search.cv_results_).to_csv(out / (name.replace(' ', '_') + '_cv.csv'), index=False)
    cv_table = pd.DataFrame(rows).sort_values('cv_average_precision', ascending=False)
    best_name = cv_table.iloc[0]['model']
    best = trained[best_name]
    vp = best.predict_proba(val[RAW])[:, 1]
    threshold, threshold_table = choose_threshold(val.isFraud, vp, fn_cost, fp_cost)
    threshold_table.to_csv(out / 'validation_thresholds.csv', index=False)
    # Freeze model and threshold before accessing test outcomes. No refit afterwards.
    report = {'status': 'Measured on supplied dataset', 'source_file': Path(data_path).name,
        'source_rows': seen, 'sample_rows': len(d), 'seed': SEED, 'selected_model': best_name,
        'selection_rule': 'Highest mean average precision on forward training CV',
        'best_params': params, 'threshold': threshold,
        'cost_assumptions': {'missed_fraud': fn_cost, 'false_alarm': fp_cost, 'unit': 'hypothetical units, not currency'},
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
        'splits': {n: {'rows': len(p), 'fraud': int(p.isFraud.sum()), 'fraud_rate': float(p.isFraud.mean()), 'start_step': float(p.step.min()), 'end_step': float(p.step.max())} for n, p in zip(['train', 'validation', 'test'], [train, val, test])},
        'validation': metrics(val.isFraud, vp, threshold), 'insights': insights}
    test_rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in trained.items():
        p = model.predict_proba(test[RAW])[:, 1]
        test_rows.append({'model': name, **metrics(test.isFraud, p, .5)})
        PrecisionRecallDisplay.from_predictions(test.isFraud, p, name=name, ax=ax)
    ax.set_title('Held-out test: precision versus recall')
    fig.tight_layout(); fig.savefig(out / 'precision_recall.png', dpi=160); plt.close(fig)
    pd.DataFrame(test_rows).to_csv(out / 'model_comparison.csv', index=False)
    cv_table.to_csv(out / 'cv_comparison.csv', index=False)
    tp = best.predict_proba(test[RAW])[:, 1]
    report['test_selected'] = metrics(test.isFraud, tp, threshold)
    report['test_selected']['assumed_cost'] = report['test_selected']['fn'] * fn_cost + report['test_selected']['fp'] * fp_cost
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(test.isFraud, tp >= threshold, labels=[0, 1], ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Selected threshold: held-out test')
    fig.tight_layout(); fig.savefig(out / 'confusion_matrix.png', dpi=160); plt.close(fig)
    review = test[RAW + ['isFraud']].copy()
    review['risk_score'] = tp
    review['review_required'] = (tp >= threshold).astype(int)
    review['error'] = np.where(review.isFraud == review.review_required, 'correct', np.where(review.isFraud == 1, 'missed_fraud', 'false_alarm'))
    review[review.error != 'correct'].head(500).to_csv(out / 'error_examples.csv', index=False)
    review.groupby(['type', 'error']).size().rename('count').to_csv(out / 'errors_by_type.csv')
    # Validation scores only: app threshold exploration must not tune on test labels.
    pd.DataFrame({'isFraud': val.isFraud.to_numpy(), 'risk_score': vp}).to_csv(out / 'validation_scores.csv', index=False)
    test[RAW].head(25).to_csv(out / 'example_transactions.csv', index=False)
    report['duration_seconds'] = round(time.time()-start, 2)
    joblib.dump({'pipeline': best, 'threshold': threshold, 'model_name': best_name,
        'sklearn_version': sklearn.__version__, 'raw_columns': RAW, 'report': report}, out / 'model.joblib', compress=3)
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.Series(report['versions']).to_csv(out / 'environment.csv', header=['version'])
    from deliverables import make_deliverables
    make_deliverables(out, report)
    print(f'Completed. Selected {best_name}. Results: {out.resolve()}', flush=True)
    return report

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='data/paysim.csv')
    ap.add_argument('--output', default='outputs')
    ap.add_argument('--max-rows', type=int, default=300000)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--fn-cost', type=float, default=25.)
    ap.add_argument('--fp-cost', type=float, default=1.)
    args = ap.parse_args()
    try:
        run(args.data, args.output, args.max_rows, args.jobs, args.fn_cost, args.fp_cost)
    except (ValueError, FileNotFoundError) as e:
        ap.exit(1, f'\nCannot complete training: {e}\n')
