#!/usr/bin/env python3
"""
Hyperparameter search utilities and parameter space definitions.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from data_utils import get_feature_list
from model_utils import train_and_evaluate_chunked


def get_param_grids() -> Dict:
    """Get parameter grids for hyperparameter search."""
    return {
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


def get_base_params(seed: int = 42) -> Dict:
    """Get base parameters for LightGBM regression."""
    return {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'seed': seed,
        'n_jobs': -1,
        'early_stopping_rounds': 20,  # More aggressive early stopping
        'n_estimators': 100,  # Reduced for faster tuning
        # GPU settings
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'gpu_use_dp': True,  # Use double precision for GPU
        'num_threads': 8,  # Limit CPU threads when using GPU
    }


def get_focused_param_sets() -> List[Dict]:
    """Get focused parameter sets based on successful experiments."""
    return [
        {
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'max_depth': -1,
            'min_child_samples': 20,
            'objective': 'regression',
            'metric': 'l2'
        },
        {
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'max_depth': -1,
            'min_child_samples': 20,
            'objective': 'regression',
            'metric': 'l2'
        }
    ]


def generate_focused_params(base_set: Dict, sample_idx: int) -> Dict:
    """Generate parameter variations around a base parameter set."""
    params = base_set.copy()

    np.random.seed(42 + sample_idx)

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

    return params


def save_results(results: List[Dict], output_path: Path) -> None:
    """Save hyperparameter search results to files."""
    # Save all results
    results_file = output_path / 'hyperparameter_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Find best results
    valid_results = [r for r in results if not np.isnan(r['correlation'])]
    if not valid_results:
        print("No valid results found!")
        return

    best_corr = max(valid_results, key=lambda x: x['correlation'])
    best_sharpe = max(valid_results, key=lambda x: x['sharpe_ratio'])

    print("\n=== HYPERPARAMETER SEARCH RESULTS ===")
    print(f"Total combinations tested: {len(results)}")
    print(f"Valid results: {len(valid_results)}")
    print(f"Best correlation: {best_corr['correlation']:.4f}")
    print(f"Best Sharpe ratio: {best_sharpe['sharpe_ratio']:.4f}")

    # Save best parameters
    with open(output_path / 'best_params_corr.json', 'w') as f:
        json.dump(best_corr['params'], f, indent=2, default=str)

    with open(output_path / 'best_params_sharpe.json', 'w') as f:
        json.dump(best_sharpe['params'], f, indent=2, default=str)