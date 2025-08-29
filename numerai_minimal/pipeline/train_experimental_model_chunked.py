"""Chunked LightGBM training for experimental model using engineered features.

- Uses adaptive_target as the target (from Stage 1)
- Uses both original feature_* and path_* engineered features (from Stage 3)
- Trains in chunks with warm-start booster
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class CallableModel:
    def __init__(self, booster):
        self.booster = booster
    
    def __call__(self, *args, **kwargs):
        return self.booster.predict(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.booster.predict(*args, **kwargs)
    
    def __getattr__(self, name):
        # Avoid recursion during unpickling
        if name == 'booster' or not hasattr(self, 'booster'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.booster, name)


class EnsembleModel:
    """Ensemble of LightGBM models with averaged predictions."""
    
    def __init__(self, models):
        self.models = models
    
    def predict(self, X, **kwargs):
        """Average predictions from all models in the ensemble."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X, **kwargs)
            predictions.append(pred)
        
        # Average predictions
        return np.mean(predictions, axis=0)
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


def select_features_from_schema(pf: pq.ParquetFile, new_feature_names_path: str | None, features_json: str | None) -> list[str]:
    cols = pf.schema.names
    base: list[str]
    if features_json and Path(features_json).exists():
        try:
            fr = pd.read_json(features_json)
            if isinstance(fr, pd.DataFrame):
                base = [c for c in fr.iloc[:, 0].tolist() if c in cols]
            else:
                base = [c for c in list(fr) if c in cols]
        except Exception:
            try:
                import json as _json
                with open(features_json, 'r') as f:
                    data = _json.load(f)
                feats = data.get('features', data if isinstance(data, list) else [])
                base = [c for c in feats if c in cols]
            except Exception:
                base = [c for c in cols if c.startswith('feature')]
    else:
        base = [c for c in cols if c.startswith('feature')]
    extra: list[str] = []
    if new_feature_names_path and Path(new_feature_names_path).exists():
        with open(new_feature_names_path, 'r') as f:
            try:
                extra = json.load(f)
            except Exception:
                extra = []
    engineered = [c for c in extra if c in cols]
    return base + engineered


def read_val_sample(parquet_path: str, features: list[str], target_col: str, rows: int) -> tuple[pd.DataFrame, pd.Series]:
    pf = pq.ParquetFile(parquet_path)
    needed = [c for c in (features + [target_col]) if c in pf.schema.names]
    acc = []
    remaining = rows
    for batch in pf.iter_batches(columns=needed, batch_size=min(100_000, rows)):
        df = batch.to_pandas()
        acc.append(df)
        remaining -= len(df)
        if remaining <= 0:
            break
    val = pd.concat(acc, ignore_index=True)
    Xv = val[[c for c in features if c in val.columns]].astype('float32')
    yv = val[target_col].astype('float32')
    return Xv, yv


def train_chunked(train_path: str, valid_path: str, target_col: str, out_path: str,
                  new_feature_names_path: str | None = None,
                  features_json: str | None = None,
                  chunk_rows: int = 250_000, val_rows: int = 200_000,
                  total_rounds: int = 1000, rounds_per_chunk: int = 200,
                  num_leaves: int = 64, learning_rate: float = 0.05, seed: int = 42):
    pf_train = pq.ParquetFile(train_path)
    features = select_features_from_schema(pf_train, new_feature_names_path, features_json)

    # Validation sample
    Xv, yv = read_val_sample(valid_path, features, target_col, val_rows)
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
    needed_cols = [c for c in (features + [target_col]) if c in pf_train.schema.names]
    for batch in pf_train.iter_batches(columns=needed_cols, batch_size=chunk_rows):
        df = batch.to_pandas()
        X = df[[c for c in features if c in df.columns]].astype('float32')
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
    
    # Create callable wrapper for Numerai compatibility
    callable_model = CallableModel(booster)
    
    with open(out_path, 'wb') as f:
        pickle.dump(callable_model, f)
    with open(Path(out_path).with_suffix('.json'), 'w') as f:
        json.dump(features, f)


def train_ensemble_chunked(train_path: str, valid_path: str, target_col: str, out_path: str,
                          new_feature_names_path: str | None = None,
                          curated_features_file: str | None = None,
                          features_json: str | None = None,
                          chunk_rows: int = 250_000, val_rows: int = 200_000,
                          total_rounds: int = 1000, rounds_per_chunk: int = 200,
                          num_leaves: int = 64, learning_rate: float = 0.05,
                          n_models: int = 5, seeds: List[int] | None = None):
    """Train an ensemble of experimental models using engineered features."""
    
    # Set up seeds for ensemble
    if seeds is None:
        seeds = [42 + i for i in range(n_models)]
    elif len(seeds) != n_models:
        seeds = seeds[:n_models] if len(seeds) > n_models else seeds + [42 + i for i in range(n_models - len(seeds))]
    
    print(f"Training ensemble of {n_models} experimental models with seeds: {seeds}")
    
    pf_train = pq.ParquetFile(train_path)
    pf_valid = pq.ParquetFile(valid_path)
    
    # Select features - prioritize curated features, then engineered + base features
    if curated_features_file and Path(curated_features_file).exists():
        with open(curated_features_file, 'r') as f:
            curated_features = json.load(f)
        print(f"Loaded {len(curated_features)} curated features from {curated_features_file}")
        features = [f for f in curated_features if f in pf_train.schema.names]
    else:
        features = select_features_from_schema(pf_train, new_feature_names_path, features_json)
    
    # Ensure target is not part of features
    if target_col in features:
        features = [c for c in features if c != target_col]
    
    print(f"Using {len(features)} features for ensemble training")
    
    # Train validation sample
    Xv, yv = read_val_sample(valid_path, features, target_col, val_rows)
    valid_set = lgb.Dataset(Xv, label=yv, free_raw_data=False)
    
    # Train ensemble
    models = []
    for i, seed in enumerate(seeds):
        print(f"\nTraining experimental model {i+1}/{n_models} (seed={seed})")
        
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
        needed_cols = [c for c in (features + [target_col]) if c in pf_train.schema.names]
        
        for batch in pf_train.iter_batches(columns=needed_cols, batch_size=chunk_rows):
            df = batch.to_pandas()
            X = df[[c for c in features if c in df.columns]].astype('float32')
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
        
        models.append(booster)
        print(f"  Experimental model {i+1} trained with {built_rounds} rounds")
    
    # Create ensemble model
    ensemble_model = EnsembleModel(models)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    # Save feature list
    with open(Path(out_path).with_suffix('.json'), 'w') as f:
        json.dump(features, f)
    
    print(f"\nEnsemble experimental model saved to {out_path}")
    print(f"Features saved to {Path(out_path).with_suffix('.json')}")
    
    return ensemble_model, features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-data', required=True, help='Enhanced train parquet with engineered features')
    ap.add_argument('--validation-data', required=True, help='Enhanced validation parquet with engineered features')
    ap.add_argument('--target-col', default='adaptive_target')
    ap.add_argument('--new-feature-names', default=None, help='Path to JSON list of engineered feature names')
    ap.add_argument('--features-json', '--curated-features', dest='curated_features', default=None,
                    help='Path to JSON file with curated features')
    ap.add_argument('--base-features-json', default=None, help='Optional features.json to pin baseline features')
    ap.add_argument('--output-model', required=True)
    ap.add_argument('--chunk-rows', type=int, default=250_000)
    ap.add_argument('--val-rows', type=int, default=200_000)
    ap.add_argument('--total-rounds', type=int, default=1000)
    ap.add_argument('--rounds-per-chunk', type=int, default=200)
    ap.add_argument('--num-leaves', type=int, default=64)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--n-models', type=int, default=5, help='Number of models in ensemble')
    ap.add_argument('--seeds', nargs='+', type=int, help='Random seeds for ensemble models')
    ap.add_argument('--seed', type=int, default=42, help='Base seed (for backward compatibility)')
    args = ap.parse_args()

    # Set up seeds
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [args.seed + i for i in range(args.n_models)]

    ensemble_model, features = train_ensemble_chunked(
        args.train_data,
        args.validation_data,
        args.target_col,
        args.output_model,
        new_feature_names_path=args.new_feature_names,
        curated_features_file=args.curated_features,
        features_json=args.base_features_json,
        chunk_rows=args.chunk_rows,
        val_rows=args.val_rows,
        total_rounds=args.total_rounds,
        rounds_per_chunk=args.rounds_per_chunk,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_models=args.n_models,
        seeds=seeds,
    )
    print(f"Ensemble experimental model saved to {args.output_model}")
    print(f"Trained {args.n_models} models with {len(features)} features each")


if __name__ == '__main__':
    main()
