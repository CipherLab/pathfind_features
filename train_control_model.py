"""Train a baseline (control) LightGBM model on original targets."""
import argparse
import lightgbm as lgb
import pandas as pd
import pickle
from pathlib import Path


def select_features(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith('feature') and not c.startswith('feature_target')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-data', required=True)
    ap.add_argument('--validation-data', required=True)
    ap.add_argument('--target-col', default='target')
    ap.add_argument('--output-model', required=True)
    ap.add_argument('--num-leaves', type=int, default=64)
    ap.add_argument('--estimators', type=int, default=600)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    args = ap.parse_args()

    train_df = pd.read_parquet(args.train_data)
    val_df = pd.read_parquet(args.validation_data)
    feats = select_features(train_df)
    if args.target_col not in train_df.columns:
        raise ValueError(f"Target column {args.target_col} not in training data")

    train_set = lgb.Dataset(train_df[feats], label=train_df[args.target_col])
    valid_set = lgb.Dataset(val_df[feats], label=val_df[args.target_col])
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }
    model = lgb.train(params, train_set, num_boost_round=args.estimators, valid_sets=[valid_set], callbacks=[lgb.early_stopping(50)])
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_model, 'wb') as f:
        pickle.dump({'model': model, 'features': feats}, f)
    print(f"Control model saved to {args.output_model}")


if __name__ == '__main__':
    main()
