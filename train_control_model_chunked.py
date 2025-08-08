"""Chunked LightGBM training for control model on massive Parquet.

Trains in mini-batches, warm-starting the booster across chunks.
Validation is a fixed small sample to control memory.
"""
import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def select_features(pf: pq.ParquetFile, features_json: str | None) -> list[str]:
    cols = pf.schema.names
    if features_json and Path(features_json).exists():
        try:
            fr = pd.read_json(features_json)
            # Accept either a list or an object with a key like 'features'
            if isinstance(fr, pd.DataFrame):
                feats = fr.iloc[:, 0].tolist()
            else:
                feats = list(fr)
        except Exception:
            try:
                import json as _json
                with open(features_json, 'r') as f:
                    data = _json.load(f)
                feats = data.get('features', data if isinstance(data, list) else [])
            except Exception:
                feats = []
        feats = [c for c in feats if c in cols]
        if feats:
            return feats
    # Fallback heuristic
    # Prefer only modeled feature columns, avoid metadata like era, data_type, id, etc.
    return [c for c in cols if c.startswith('feature')]


def read_val_sample(parquet_path: str, features: list[str], target_col: str, rows: int) -> tuple[pd.DataFrame, pd.Series]:
    pf = pq.ParquetFile(parquet_path)
    needed = features + [target_col]
    acc = []
    remaining = rows
    for batch in pf.iter_batches(columns=needed, batch_size=min(100_000, rows)):
        # Avoid pandas metadata turning columns (e.g., 'id') into index
        try:
            df = batch.to_pandas(use_pandas_metadata=False)
        except TypeError:
            df = batch.to_pandas()
            # If an index is set from parquet metadata, bring it back as columns
            if any(n is not None for n in (df.index.names or [])):
                df = df.reset_index()
        acc.append(df)
        remaining -= len(df)
        if remaining <= 0:
            break
    val = pd.concat(acc, ignore_index=True)
    Xv = val[features].astype('float32')
    yv = val[target_col].astype('float32')
    return Xv, yv


def train_chunked(train_path: str, valid_path: str, target_col: str, out_path: str,
                  features_json: str | None = None,
                  chunk_rows: int = 250_000, val_rows: int = 200_000,
                  total_rounds: int = 1000, rounds_per_chunk: int = 200,
                  num_leaves: int = 64, learning_rate: float = 0.05, seed: int = 42):
    pf_train = pq.ParquetFile(train_path)
    pf_valid = pq.ParquetFile(valid_path)
    # Initial desired feature list (from JSON or heuristic) based on train columns
    desired_features = select_features(pf_train, features_json)

    # Ensure target is not part of features (defensive)
    if target_col in desired_features:
        desired_features = [c for c in desired_features if c != target_col]

    # Validate target exists in both datasets early
    train_cols = set(pf_train.schema.names)
    valid_cols = set(pf_valid.schema.names)
    if target_col not in train_cols:
        raise KeyError(f"Target column '{target_col}' not found in train dataset: {train_path}")
    if target_col not in valid_cols:
        raise KeyError(f"Target column '{target_col}' not found in validation dataset: {valid_path}")

    # Keep feature order but intersect with columns present in BOTH train and validation
    features = [c for c in desired_features if c in train_cols and c in valid_cols and c != target_col]
    if not features:
        raise ValueError(
            "No common feature columns found between train and validation after filtering. "
            "Check your data and features_json."
        )
    dropped = [c for c in desired_features if c not in features]
    if dropped:
        preview = dropped[:10]
        more = '...' if len(dropped) > 10 else ''
        print(
            f"Warning: Dropping {len(dropped)} feature(s) not present in both datasets: {preview}{more}"
        )
    print(f"Using {len(features)} aligned feature(s) for training/validation.")

    # Validation sample
    Xv, yv = read_val_sample(valid_path, features, target_col, val_rows)
    # Keep raw data for the validation set to avoid LightGBM reference/predictor errors across chunks
    valid_set = lgb.Dataset(Xv, label=yv, free_raw_data=False)

    params = {
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': seed,
    }

    booster = None
    built_rounds = 0
    needed_cols = features + [target_col]
    for batch in pf_train.iter_batches(columns=needed_cols, batch_size=chunk_rows):
        try:
            df = batch.to_pandas(use_pandas_metadata=False)
        except TypeError:
            df = batch.to_pandas()
            if any(n is not None for n in (df.index.names or [])):
                df = df.reset_index()
        X = df[features].astype('float32')
        y = df[target_col].astype('float32')
        train_set = lgb.Dataset(X, label=y, free_raw_data=True)

        rounds = min(rounds_per_chunk, max(0, total_rounds - built_rounds))
        if rounds == 0:
            break
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=rounds,
            valid_sets=[valid_set],
            init_model=booster,
            keep_training_booster=True,
        )
        built_rounds += rounds

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({'model': booster, 'features': features}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-data', required=True)
    ap.add_argument('--validation-data', required=True)
    ap.add_argument('--target-col', default='target')
    ap.add_argument('--features-json', default=None)
    ap.add_argument('--output-model', required=True)
    ap.add_argument('--chunk-rows', type=int, default=250_000)
    ap.add_argument('--val-rows', type=int, default=200_000)
    ap.add_argument('--total-rounds', type=int, default=1000)
    ap.add_argument('--rounds-per-chunk', type=int, default=200)
    ap.add_argument('--num-leaves', type=int, default=64)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    train_chunked(
        args.train_data,
        args.validation_data,
    args.target_col,
        args.output_model,
    features_json=args.features_json,
        chunk_rows=args.chunk_rows,
        val_rows=args.val_rows,
        total_rounds=args.total_rounds,
        rounds_per_chunk=args.rounds_per_chunk,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(f"Control (chunked) model saved to {args.output_model}")


if __name__ == '__main__':
    main()
