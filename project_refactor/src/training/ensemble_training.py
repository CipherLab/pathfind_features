#!/usr/bin/env python3
"""
Ensemble methods for the adaptive-only model.
Combines multiple models to improve performance and robustness.
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from tests import setup_script_output, get_output_path, initialize_script_output, add_output_dir_arguments

class CallableModel:
    def __init__(self, booster):
        self.booster = booster

    def __call__(self, *args, **kwargs):
        return self.booster.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.booster.predict(*args, **kwargs)

class EnsembleModel:
    def __init__(self, models: List[CallableModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += pred * weight

        return ensemble_pred

def load_data(parquet_path: str, features: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data from parquet file."""
    pf = pq.ParquetFile(parquet_path)
    df = pf.read().to_pandas()
    X = df[features].astype('float32')
    y = df[target_col].astype('float32')
    return X, y

def train_base_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame,
                    y_val: pd.Series, params: Dict, seed: int) -> CallableModel:
    """Train a single base model."""
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    params['seed'] = seed

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    return CallableModel(model)

def create_diverse_models(train_path: str, val_path: str, features: List[str],
                        target_col: str, n_models: int = 5) -> List[CallableModel]:
    """Create diverse models with different hyperparameters."""
    X_train, y_train = load_data(train_path, features, target_col)
    X_val, y_val = load_data(val_path, features, target_col)

    models = []

    # Different hyperparameter sets for diversity
    param_sets = [
        {'num_leaves': 64, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8},
        {'num_leaves': 128, 'learning_rate': 0.03, 'feature_fraction': 0.9, 'bagging_fraction': 0.7},
        {'num_leaves': 32, 'learning_rate': 0.1, 'feature_fraction': 0.7, 'bagging_fraction': 0.9},
        {'num_leaves': 256, 'learning_rate': 0.02, 'feature_fraction': 0.6, 'bagging_fraction': 0.6},
        {'num_leaves': 96, 'learning_rate': 0.04, 'feature_fraction': 0.85, 'bagging_fraction': 0.75}
    ]

    base_params = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'n_jobs': -1
    }

    for i in range(min(n_models, len(param_sets))):
        params = base_params.copy()
        params.update(param_sets[i])

        print(f"Training model {i+1}/{n_models} with params: {param_sets[i]}")
        model = train_base_model(X_train, y_train, X_val, y_val, params, seed=42 + i)
        models.append(model)

    return models

def create_bootstrap_models(train_path: str, val_path: str, features: List[str],
                          target_col: str, n_models: int = 5) -> List[CallableModel]:
    """Create models trained on different bootstrap samples."""
    X_train, y_train = load_data(train_path, features, target_col)
    X_val, y_val = load_data(val_path, features, target_col)

    models = []

    base_params = {
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbosity': -1,
        'n_jobs': -1
    }

    for i in range(n_models):
        # Bootstrap sample
        np.random.seed(42 + i)
        sample_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_bootstrap = X_train.iloc[sample_indices]
        y_bootstrap = y_train.iloc[sample_indices]

        print(f"Training bootstrap model {i+1}/{n_models}")
        model = train_base_model(X_bootstrap, y_bootstrap, X_val, y_val, base_params, seed=42 + i)
        models.append(model)

    return models

def create_feature_subset_models(train_path: str, val_path: str, features: List[str],
                               target_col: str, n_models: int = 5) -> List[CallableModel]:
    """Create models trained on different feature subsets."""
    X_train, y_train = load_data(train_path, features, target_col)
    X_val, y_val = load_data(val_path, features, target_col)

    models = []

    base_params = {
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbosity': -1,
        'n_jobs': -1
    }

    for i in range(n_models):
        # Random feature subset (80-100% of features)
        np.random.seed(42 + i)
        subset_size = np.random.randint(int(0.8 * len(features)), len(features) + 1)
        subset_features = np.random.choice(features, size=subset_size, replace=False)

        X_train_subset = X_train[subset_features]
        X_val_subset = X_val[subset_features]

        print(f"Training feature subset model {i+1}/{n_models} with {len(subset_features)} features")
        model = train_base_model(X_train_subset, y_train, X_val_subset, y_val, base_params, seed=42 + i)
        models.append(model)

    return models

def evaluate_ensemble(models: List[CallableModel], X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """Evaluate ensemble performance."""
    from scipy.stats import spearmanr

    # Individual model predictions
    individual_preds = []
    for model in models:
        pred = model.predict(X_val)
        individual_preds.append(pred)

    # Ensemble prediction (simple average)
    ensemble_pred = np.mean(individual_preds, axis=0)

    # Calculate metrics
    corr_result = spearmanr(y_val, ensemble_pred)
    corr = abs(corr_result[0])  # type: ignore

    # Sharpe ratio
    val_df = X_val.copy()
    val_df['target'] = y_val
    val_df['prediction'] = ensemble_pred

    # Group by era (assuming eras are sequential)
    era_groups = val_df.groupby(val_df.index // 1000)
    era_corrs = []
    for _, group in era_groups:
        if len(group) > 1:
            era_corr, _ = spearmanr(group['target'], group['prediction'])
            era_corrs.append(era_corr[0])  # type: ignore

    sharpe = float(np.mean(era_corrs) / np.std(era_corrs) if era_corrs else 0)

    return {
        'correlation': float(corr),
        'sharpe_ratio': sharpe,
        'n_models': len(models)
    }

def main():
    parser = argparse.ArgumentParser(description="Ensemble methods for adaptive model")
    parser.add_argument('--train-data', required=True, help='Path to training parquet')
    parser.add_argument('--validation-data', required=True, help='Path to validation parquet')
    parser.add_argument('--features-json', required=True, help='Path to features.json')
    parser.add_argument('--target-col', default='adaptive_target', help='Target column name')
    parser.add_argument('--output-model', required=True, help='Output model path')
    parser.add_argument('--ensemble-type', choices=['diverse', 'bootstrap', 'feature_subset', 'mixed'],
                       default='mixed', help='Type of ensemble to create')
    parser.add_argument('--n-models', type=int, default=5, help='Number of base models')
    add_output_dir_arguments(parser)

    args = parser.parse_args()

    # Set up output directory
    script_name = "ensemble_training"
    output_dir = initialize_script_output(script_name, args)
    print(f"Logs and results will be saved to: {output_dir}")

    # Load features
    with open(args.features_json, 'r') as f:
        features_data = json.load(f)

    if 'feature_sets' in features_data and 'medium' in features_data['feature_sets']:
        features = features_data['feature_sets']['medium']
    elif isinstance(features_data, list):
        features = features_data
    else:
        features = [k for k in features_data.keys() if k.startswith('feature_')]

    print(f"Loaded {len(features)} features")

    models = []

    if args.ensemble_type == 'diverse':
        models = create_diverse_models(args.train_data, args.validation_data, features,
                                    args.target_col, args.n_models)
    elif args.ensemble_type == 'bootstrap':
        models = create_bootstrap_models(args.train_data, args.validation_data, features,
                                       args.target_col, args.n_models)
    elif args.ensemble_type == 'feature_subset':
        models = create_feature_subset_models(args.train_data, args.validation_data, features,
                                            args.target_col, args.n_models)
    elif args.ensemble_type == 'mixed':
        # Create mixed ensemble
        diverse_models = create_diverse_models(args.train_data, args.validation_data, features,
                                             args.target_col, max(2, args.n_models // 3))
        bootstrap_models = create_bootstrap_models(args.train_data, args.validation_data, features,
                                                 args.target_col, max(2, args.n_models // 3))
        subset_models = create_feature_subset_models(args.train_data, args.validation_data, features,
                                                   args.target_col, max(2, args.n_models // 3))
        models = diverse_models + bootstrap_models + subset_models

    # Create ensemble
    ensemble = EnsembleModel(models)

    # Evaluate ensemble
    X_val, y_val = load_data(args.validation_data, features, args.target_col)
    metrics = evaluate_ensemble(models, X_val, y_val)

    print("\n=== ENSEMBLE RESULTS ===")
    print(f"Correlation: {metrics['correlation']:.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Number of models: {len(models)}")

    # Save ensemble to tests directory
    original_model_name = Path(args.output_model).name
    model_output_path = get_output_path(output_dir, original_model_name)
    info_output_path = get_output_path(output_dir, Path(args.output_model).stem + '_info.json')

    with open(model_output_path, 'wb') as f:
        pickle.dump(ensemble, f)

    # Save ensemble info
    ensemble_info = {
        'ensemble_type': args.ensemble_type,
        'n_models': len(models),
        'metrics': metrics,
        'features': features
    }

    with open(info_output_path, 'w') as f:
        json.dump(ensemble_info, f, indent=2, default=str)

    print(f"Ensemble saved to {model_output_path}")
    print(f"Ensemble info saved to {info_output_path}")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()
