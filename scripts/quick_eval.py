"""Quick evaluation: baseline vs augmented (engineered) models.

Trains two LightGBM models on sampled rows to compare validation RMSE.
Usage example:
  PYTHONPATH=. ./.venv/bin/python scripts/quick_eval.py

Defaults sample 200k rows from each split to keep runtime reasonable.
"""
import argparse
import pandas as pd
import lightgbm as lgb
import json
import math
from pathlib import Path
from sklearn.metrics import mean_squared_error


def select_original_features(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith('feature') and not c.startswith('feature_target')]


def safe_load_parquet(path, sample_rows=None, random_state=42):
    df = pd.read_parquet(path)
    if sample_rows is not None and sample_rows > 0 and len(df) > sample_rows:
        return df.sample(n=sample_rows, random_state=random_state)
    return df


def train_and_eval(train_df, val_df, features, target_col, lgb_params, num_boost_round=400, early_stop=50):
    dtrain = lgb.Dataset(train_df[features], label=train_df[target_col])
    dval = lgb.Dataset(val_df[features], label=val_df[target_col])
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stop)],
    )
    preds = model.predict(val_df[features])
    rmse = math.sqrt(mean_squared_error(val_df[target_col], preds))
    return model, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-base', default='v5.0/train_with_adaptive.parquet')
    ap.add_argument('--train-aug', default='pipeline_runs/pf45_tail200/fe_train/engineered.parquet')
    ap.add_argument('--val-base', default='v5.0/validation.parquet')
    ap.add_argument('--val-aug', default='pipeline_runs/pf45_tail200/fe_validation/engineered.parquet')
    ap.add_argument('--new-features-file', default='pipeline_runs/pf45_tail200/fe_train/new_feature_names.json')
    ap.add_argument('--target-col', default='adaptive_target')
    ap.add_argument('--sample-rows', type=int, default=200000, help='Number of rows to sample per split for faster runs (0 = use all)')
    ap.add_argument('--num-leaves', type=int, default=64)
    ap.add_argument('--num-boost', type=int, default=400)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    args = ap.parse_args()

    print('Loading data (may take a while)...')
    train_base = safe_load_parquet(args.train_base, sample_rows=(args.sample_rows or None))
    val_base = safe_load_parquet(args.val_base, sample_rows=(args.sample_rows or None))
    train_aug = safe_load_parquet(args.train_aug, sample_rows=(args.sample_rows or None))
    val_aug = safe_load_parquet(args.val_aug, sample_rows=(args.sample_rows or None))

    print('Samples loaded:')
    print(' train_base', len(train_base), 'val_base', len(val_base))
    print(' train_aug', len(train_aug), 'val_aug', len(val_aug))

    feats_orig = select_original_features(train_base)
    if not feats_orig:
        raise SystemExit('No original features found in training base; aborting')

    # Load engineered feature names
    new_feats = []
    try:
        new_feats = json.load(open(args.new_features_file))
    except Exception as e:
        print('Warning: failed to read new features file:', e)

    # Filter to columns actually present
    new_feats = [f for f in new_feats if f in train_aug.columns]

    print('Original feature count:', len(feats_orig))
    print('New engineered feature count:', len(new_feats))

    target = args.target_col
    for df in [train_base, val_base, train_aug, val_aug]:
        if target not in df.columns:
            raise SystemExit(f'Target column {target} not found in one of the inputs')

    lgb_params = {
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }

    print('\nTraining baseline (original features)')
    _, rmse_base = train_and_eval(train_base, val_base, feats_orig, target, lgb_params, num_boost_round=args.num_boost)
    print('Baseline RMSE:', rmse_base)

    print('\nTraining augmented (orig + engineered)')
    feats_aug = feats_orig + new_feats
    _, rmse_aug = train_and_eval(train_aug, val_aug, feats_aug, target, lgb_params, num_boost_round=args.num_boost)
    print('Augmented RMSE:', rmse_aug)

    print('\nDelta (augmented - baseline):', rmse_aug - rmse_base)

    # Quick feature importance for augmented
    print('\nDone')


if __name__ == '__main__':
    main()
