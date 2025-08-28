#!/usr/bin/env python3
"""
Hyperparameter optimization for the adaptive-only model.
Tests different combinations of LightGBM parameters to find optimal settings.
"""
import argparse
import json
import pickle
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional

class CallableModel:
    def __init__(self, booster):
        self.booster = booster

    def __call__(self, *args, **kwargs):
        return self.booster.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.booster.predict(*args, **kwargs)

def filter_by_era_range(df: pd.DataFrame, era_range: str) -> pd.DataFrame:
    """Filter dataframe by era range (e.g., 'MAX-200:MAX-75')."""
    if 'era' not in df.columns:
        print("Warning: 'era' column not found, skipping era filtering")
        return df

    # Parse era range (e.g., "MAX-200:MAX-75")
    try:
        start_era, end_era = era_range.split(':')

        # Get max era
        max_era = df['era'].max()

        # Parse start era
        if start_era.startswith('MAX-'):
            start_offset = int(start_era.split('-')[1])
            start_era_val = max_era - start_offset
        else:
            start_era_val = int(start_era)

        # Parse end era
        if end_era.startswith('MAX-'):
            end_offset = int(end_era.split('-')[1])
            end_era_val = max_era - end_offset
        else:
            end_era_val = int(end_era)

        # Filter dataframe
        filtered_df = df[(df['era'] >= start_era_val) & (df['era'] <= end_era_val)]
        print(f"Filtered from {len(df)} to {len(filtered_df)} rows using era range {start_era_val}:{end_era_val}")

        return filtered_df

    except Exception as e:
        print(f"Error parsing era range '{era_range}': {e}")
        print("Skipping era filtering")
        return df

def load_data(train_path: str, val_path: str, features: List[str], target_col: str, era_range: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and prepare training and validation data."""
    print(f"Loading data from {train_path} and {val_path}")

    # Load training data
    pf_train = pq.ParquetFile(train_path)
    train_df = pf_train.read().to_pandas()

    # Load validation data
    pf_val = pq.ParquetFile(val_path)
    val_df = pf_val.read().to_pandas()

    # Apply era filtering if specified
    if era_range:
        print(f"Applying era range filter: {era_range}")
        train_df = filter_by_era_range(train_df, era_range)
        val_df = filter_by_era_range(val_df, era_range)
        print(f"After era filtering - Training data shape: {train_df.shape}, Validation data shape: {val_df.shape}")

    X_train = train_df[features].astype('float32')
    y_train = train_df[target_col].astype('float32')

    X_val = val_df[features].astype('float32')
    y_val = val_df[target_col].astype('float32')

    print(f"Final data shapes - Training: {X_train.shape}, Validation: {X_val.shape}")
    return X_train, y_train, X_val, y_val

def get_feature_list(features_json: str) -> List[str]:
    """Extract feature list from features.json file."""
    if not Path(features_json).exists():
        raise FileNotFoundError(f"Features file not found: {features_json}")

    with open(features_json, 'r') as f:
        data = json.load(f)

    # Try different possible structures
    if 'feature_sets' in data and 'medium' in data['feature_sets']:
        return data['feature_sets']['medium']
    elif isinstance(data, list):
        return data
    elif 'features' in data:
        return data['features']
    else:
        # Fallback: extract from data keys
        features = [k for k in data.keys() if k.startswith('feature_')]
        return features

def train_and_evaluate(params: Dict, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series, seed: int = 42) -> Dict:
    """Train a model with given parameters and evaluate on validation set."""
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    # Set up callbacks for early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)  # Disable logging
    ]

    # Update params with seed
    params['seed'] = seed

    # Train model
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=callbacks
    )

    # Make predictions
    val_preds = model.predict(X_val)

    # Calculate metrics - for ranking objectives, we still use correlation as our primary metric
    from scipy.stats import spearmanr

    # For lambdarank/ndcg, we still use correlation as our primary metric
    corr_result = spearmanr(y_val, val_preds)
    corr = abs(corr_result[0])  # type: ignore

    # Calculate Sharpe ratio (assuming we have era information)
    val_with_era = X_val.copy()
    val_with_era['target'] = y_val
    val_with_era['prediction'] = val_preds

    # Group by era and calculate Sharpe
    era_groups = val_with_era.groupby(val_with_era.index // 1000)  # Assuming eras are grouped
    era_corrs = []
    for _, group in era_groups:
        if len(group) > 1:
            era_corr, _ = spearmanr(group['target'], group['prediction'])
            era_corrs.append(era_corr)

    sharpe = np.mean(era_corrs) / np.std(era_corrs) if era_corrs else 0

    return {
        'correlation': corr,
        'sharpe_ratio': sharpe,
        'best_iteration': model.best_iteration,
        'params': params
    }

def train_and_evaluate_chunked(params: Dict, train_path: str, val_path: str, features: List[str], target_col: str,
                                chunk_rows: int = 250_000, val_rows: int = 200_000, total_rounds: int = 1000, rounds_per_chunk: int = 200, seed: int = 42) -> Dict:
    """Chunked training and evaluation for hyperparameter search."""
    import pyarrow.parquet as pq
    import lightgbm as lgb
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from scipy.stats import spearmanr

    pf_train = pq.ParquetFile(train_path)
    features = [c for c in features if c in pf_train.schema.names]

    # Validation sample
    def read_val_sample(parquet_path: str, features: list[str], target_col: str, rows: int):
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

    Xv, yv = read_val_sample(val_path, features, target_col, val_rows)
    valid_set = lgb.Dataset(Xv, label=yv, free_raw_data=False)

    params = params.copy()
    params['seed'] = seed

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

    if booster is None:
        # If no training was done, return NaNs
        return {
            'correlation': float('nan'),
            'sharpe_ratio': float('nan'),
            'params': params
        }

    # Make predictions
    val_preds = booster.predict(Xv)

    # Calculate metrics
    corr_result = spearmanr(yv, val_preds)
    corr = abs(corr_result[0])  # type: ignore

    # Sharpe ratio
    val_with_era = Xv.copy()
    val_with_era['target'] = yv
    val_with_era['prediction'] = val_preds
    era_groups = val_with_era.groupby(val_with_era.index // 1000)
    era_corrs = []
    for _, group in era_groups:
        if len(group) > 1:
            era_corr, _ = spearmanr(group['target'], group['prediction'])
            era_corrs.append(era_corr)
    sharpe = np.mean(era_corrs) / np.std(era_corrs) if era_corrs else 0

    return {
        'correlation': corr,
        'sharpe_ratio': sharpe,
        'params': params
    }

def hyperparameter_search(train_path: str, val_path: str, features_json: str,
                        target_col: str, output_dir: str, search_type: str = 'grid',
                        chunk_rows: int = 250_000, val_rows: int = 200_000, total_rounds: int = 1000, rounds_per_chunk: int = 200, seed: int = 42):
    """Perform hyperparameter search with chunked training."""

    # Define parameter grids and base params (move to top of function)
    param_grids = {
        'num_leaves': [31, 64, 128, 256],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0],
        'bagging_freq': [1, 5],
        'max_depth': [-1, 8, 10],
        'min_child_samples': [20, 50],
        'min_child_weight': [0.001, 0.01],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
    base_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'verbosity': -1,
        'seed': seed,
        'n_jobs': -1,
        'early_stopping_rounds': 50,
        'n_estimators': 200
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load features
    features = get_feature_list(features_json)
    print(f"Using {len(features)} features")

    results = []

    if search_type == 'grid':
        # Grid search - try all combinations (will be large!)
        param_combinations = list(product(*param_grids.values()))
        param_names = list(param_grids.keys())

        print(f"Starting grid search with {len(param_combinations)} combinations")

        for i, combo in enumerate(param_combinations):
            if i % 10 == 0:
                print(f"Testing combination {i+1}/{len(param_combinations)}")

            params = base_params.copy()
            for name, value in zip(param_names, combo):
                params[name] = value

            try:
                result = train_and_evaluate_chunked(params, train_path, val_path, features, target_col,
                                                   chunk_rows=chunk_rows, val_rows=val_rows, total_rounds=total_rounds, rounds_per_chunk=rounds_per_chunk, seed=seed)
                result['combination_id'] = i
                results.append(result)
            except Exception as e:
                print(f"Error with combination {i}: {e}")
                continue

    elif search_type == 'random':
        # Random search - sample random combinations
        n_samples = 50
        print(f"Starting random search with {n_samples} samples")

        np.random.seed(42)
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"Testing sample {i+1}/{n_samples}")

            params = base_params.copy()
            for param_name, values in param_grids.items():
                params[param_name] = np.random.choice(values)

            try:
                result = train_and_evaluate_chunked(params, train_path, val_path, features, target_col,
                                                   chunk_rows=chunk_rows, val_rows=val_rows, total_rounds=total_rounds, rounds_per_chunk=rounds_per_chunk, seed=seed)
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                print(f"Error with sample {i}: {e}")
                continue

    elif search_type == 'focused':
        # Focused search around best known parameters
        print("Starting focused search around best known parameters")

        # Base parameter sets from successful experiments
        base_param_sets = [
            {
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'max_depth': -1,
                'min_child_samples': 20,
                'objective': 'lambdarank',
                'metric': 'ndcg'
            },
            {
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'max_depth': -1,
                'min_child_samples': 20,
                'objective': 'lambdarank',
                'metric': 'ndcg'
            }
        ]

        n_samples = 30  # Default, can be made configurable

        for i in range(n_samples):
            if i % 5 == 0:
                print(f"Testing focused sample {i+1}/{n_samples}")

            # Start with a base parameter set
            base_set = base_param_sets[i % len(base_param_sets)]
            params = base_params.copy()
            params.update(base_set)

            # Add some variation around the base parameters
            np.random.seed(42 + i)

            # Vary learning rate
            lr_variation = np.random.choice([0.8, 1.0, 1.2])
            params['learning_rate'] = base_set['learning_rate'] * lr_variation

            # Vary num_leaves
            leaves_variation = np.random.choice([0.5, 1.0, 2.0])
            params['num_leaves'] = int(base_set['num_leaves'] * leaves_variation)

            # Vary feature_fraction
            params['feature_fraction'] = np.random.choice([0.7, 0.8, 0.9, 1.0])

            # Vary bagging_fraction
            params['bagging_fraction'] = np.random.choice([0.7, 0.8, 0.9, 1.0])

            # Sometimes vary max_depth
            if np.random.random() < 0.3:
                params['max_depth'] = np.random.choice([-1, 8, 10])

            # Add regularization sometimes
            if np.random.random() < 0.4:
                params['reg_alpha'] = np.random.choice([0, 0.1, 0.5])
                params['reg_lambda'] = np.random.choice([0, 0.1, 0.5])

            try:
                result = train_and_evaluate_chunked(params, train_path, val_path, features, target_col,
                                                   chunk_rows=chunk_rows, val_rows=val_rows, total_rounds=total_rounds, rounds_per_chunk=rounds_per_chunk, seed=seed)
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                print(f"Error with focused sample {i}: {e}")
                continue

    # Save results
    results_file = output_path / 'hyperparameter_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Find best results
    best_corr = max(results, key=lambda x: x['correlation'])
    best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])

    print("\n=== HYPERPARAMETER SEARCH RESULTS ===")
    print(f"Total combinations tested: {len(results)}")
    print(f"Best correlation: {best_corr['correlation']:.4f}")
    print(f"Best Sharpe ratio: {best_sharpe['sharpe_ratio']:.4f}")

    # Save best parameters
    with open(output_path / 'best_params_corr.json', 'w') as f:
        json.dump(best_corr['params'], f, indent=2)

    with open(output_path / 'best_params_sharpe.json', 'w') as f:
        json.dump(best_sharpe['params'], f, indent=2)

    return results

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for adaptive model")
    parser.add_argument('--train-data', required=True, help='Path to training parquet')
    parser.add_argument('--validation-data', required=True, help='Path to validation parquet')
    parser.add_argument('--features-json', required=True, help='Path to features.json')
    parser.add_argument('--target-col', default='target', help='Target column name')
    parser.add_argument('--output-dir', default='hyperparameter_tuning_results', help='Output directory')
    parser.add_argument('--search-type', choices=['grid', 'random', 'focused'], default='focused',
                       help='Search type: grid=all combinations, random=sample, focused=around best known params')
    parser.add_argument('--n-iterations', type=int, default=30, help='Number of iterations for random/focused search')
    parser.add_argument('--era-range', help='Era range for training data (e.g., "MAX-200:MAX-75")')

    args = parser.parse_args()

    hyperparameter_search(
        args.train_data,
        args.validation_data,
        args.features_json,
        args.target_col,
        args.output_dir,
        args.search_type,
        args.era_range
    )

if __name__ == '__main__':
    main()
