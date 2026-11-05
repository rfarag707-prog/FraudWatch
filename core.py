"""Shared schema and inference. Only features available before execution are used."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

RAW = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']
NUM = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'hour',
       'amount_to_balance', 'origin_shortfall']
TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

def validate(frame, labeled=False):
    required = RAW + (['isFraud'] if labeled else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('The CSV has no transactions.')
    out = frame[required].copy()
    for col in ['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest']:
        converted = pd.to_numeric(out[col], errors='coerce')
        bad = out[col].notna() & converted.isna()
        if bad.any() or np.isinf(converted).any() or (converted.dropna() < 0).any():
            raise ValueError(f'{col}: use finite, nonnegative numeric values (or empty balances).')
        out[col] = converted
    if out[['step', 'amount']].isna().any().any():
        raise ValueError('step and amount cannot be empty.')
    if (out.step % 1 != 0).any():
        raise ValueError('step must contain whole hours.')
    out['type'] = out['type'].astype('string').str.strip().str.upper()
    if (~out['type'].isin(TYPES)).any():
        raise ValueError('type must be one of: ' + ', '.join(TYPES))
    if labeled:
        out['isFraud'] = pd.to_numeric(out['isFraud'], errors='coerce')
        if not out.isFraud.isin([0, 1]).all():
            raise ValueError('isFraud must be 0 or 1 on every row.')
        out['isFraud'] = out.isFraud.astype(int)
    return out

def features(frame):
    """Pure row-wise transformation; no fitting or future balances."""
    x = frame[RAW].copy()
    x['hour'] = x['step'] % 24
    x['amount_to_balance'] = (x.amount / (x.oldbalanceOrg + 1)).clip(0, 1e6)
    x['origin_shortfall'] = (x.amount - x.oldbalanceOrg).clip(lower=0)
    return x[NUM + ['type']]

def feature_step():
    return FunctionTransformer(features, validate=False)

def score(frame, artifact, threshold=None):
    x = validate(frame)
    p = artifact['pipeline'].predict_proba(x)[:, 1]
    cutoff = artifact['threshold'] if threshold is None else float(threshold)
    result = x.copy()
    result['risk_score'] = p
    result['review_required'] = (p >= cutoff).astype(int)
    return result
