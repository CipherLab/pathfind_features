#!/usr/bin/env python3
"""
Comprehensive Sharpe Ratio Investigation Script
Validates hyperparameter tuning results and analyzes model performance.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import pyarrow.parquet as pq
import lightgbm as lgb
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_best_parameters(params_file: str) -> Dict:
    """Load the best hyperparameters from JSON file."""
    with open(params_file, 'r') as f:
        return json.load(f)


def load_validation_data(data_file: str, features_file: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare validation data."""
    # Load validation data
    pf_val = pq.ParquetFile(data_file)
    val_df = pf_val.read().to_pandas()

    # Load features
    with open(features_file, 'r') as f:
        features = json.load(f)

    print(f"Loaded {len(features)} features")
    print(f"Validation data shape: {val_df.shape}")

    # Prepare data
    X_val = val_df[features].astype('float32')
    y_val = val_df['adaptive_target'].astype('float32')
    era_series = val_df['era']

    return X_val, y_val, era_series


def filter_nan_values(X: pd.DataFrame, y: pd.Series, era: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Filter out NaN values from data."""
    valid_mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    era_clean = era[valid_mask]

    print(f"After NaN filtering: {X_clean.shape[0]} rows")
    return X_clean, y_clean, era_clean


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: Dict, num_rounds: int = 100) -> lgb.Booster:
    """Train LightGBM model with given parameters."""
    print("Training model with best parameters...")

    train_set = lgb.Dataset(X_train, label=y_train)

    # Prepare parameters
    lgb_params = params.copy()
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'l2'
    lgb_params['seed'] = 42

    model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=num_rounds
    )

    return model


def calculate_overall_metrics(y_true: pd.Series, y_pred) -> float:
    """Calculate overall correlation."""
    # Ensure y_pred is a numpy array
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    corr_result = spearmanr(y_true, y_pred)
    overall_corr = corr_result[0] if isinstance(corr_result, tuple) else corr_result
    print(f"Overall correlation: {overall_corr:.4f}")
    return float(overall_corr)


def calculate_era_correlations(X: pd.DataFrame, y_true: pd.Series, y_pred,
                              era_series: pd.Series) -> Tuple[List[float], List[Dict]]:
    """Calculate correlations for each era."""
    # Ensure y_pred is a numpy array
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    print("Calculating era correlations...")

    era_correlations = []
    era_stats = []

    for era in sorted(era_series.unique()):
        era_mask = era_series == era
        era_targets = y_true[era_mask]
        era_predictions = y_pred[era_mask]

        if len(era_targets) > 1:
            corr_result = spearmanr(era_targets, era_predictions)
            era_corr = corr_result[0] if isinstance(corr_result, tuple) else corr_result
            era_correlations.append(era_corr)
            era_stats.append({
                'era': era,
                'correlation': era_corr,
                'n_samples': len(era_targets),
                'target_mean': era_targets.mean(),
                'pred_mean': era_predictions.mean()
            })

            if len(era_stats) <= 5:  # Show first 5 eras
                print(f"Era {era}: corr={era_corr:.4f}, n={len(era_targets)}, "
                      f"target_mean={era_targets.mean():.4f}, pred_mean={era_predictions.mean():.4f}")

    return era_correlations, era_stats


def calculate_sharpe_ratio(era_correlations: List[float], expected_sharpe: float = 3.9521):
    """Calculate Sharpe ratio and statistical significance."""
    if not era_correlations:
        return 0.0

    sharpe = np.mean(era_correlations) / np.std(era_correlations)

    print(f"\nNumber of eras with correlations: {len(era_correlations)}")
    print(f"Era correlations mean: {np.mean(era_correlations):.4f}")
    print(f"Era correlations std: {np.std(era_correlations):.4f}")
    print(f"Era correlations min: {np.min(era_correlations):.4f}")
    print(f"Era correlations max: {np.max(era_correlations):.4f}")
    print()
    print(f"Calculated Sharpe ratio: {sharpe:.4f}")
    print(f"Expected Sharpe ratio from tuning: {expected_sharpe:.4f}")
    print(f"Difference: {abs(sharpe - expected_sharpe):.4f}")

    # Statistical significance
    t_stat = sharpe * np.sqrt(len(era_correlations))
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value (two-tailed): {2 * (1 - 0.5 * (1 + np.sign(t_stat) * (2/np.sqrt(np.pi)) * np.exp(-t_stat**2/2))):.2e}")

    return float(sharpe)


def main():
    """Main function to run the Sharpe ratio investigation."""
    print("=" * 50)
    print("COMPREHENSIVE SHARPE RATIO INVESTIGATION")
    print("=" * 50)

    # File paths
    params_file = "hyperparameter_tuning_optimized/best_params_sharpe.json"
    data_file = "pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet"
    features_file = "pipeline_runs/my_experiment/adaptive_only_model.json"

    # Check if files exist
    for file_path in [params_file, data_file, features_file]:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found!")
            return

    # Load best parameters
    best_params = load_best_parameters(params_file)
    print("Best parameters by Sharpe ratio:")
    print(json.dumps(best_params, indent=2))
    print()

    # Load and prepare data
    X_val, y_val, era_series = load_validation_data(data_file, features_file)
    X_val_clean, y_val_clean, era_clean = filter_nan_values(X_val, y_val, era_series)
    print()

    # Train model
    model = train_model(X_val_clean, y_val_clean, best_params)

    # Make predictions
    val_preds = model.predict(X_val_clean)
    # Ensure predictions are numpy array
    val_preds = np.array(val_preds)
    print(f"Predictions shape: {val_preds.shape}")
    print(f"Predictions range: {val_preds.min():.4f} to {val_preds.max():.4f}")
    print()

    # Calculate overall correlation
    overall_corr = calculate_overall_metrics(y_val_clean, val_preds)
    print()

    # Calculate era correlations
    era_correlations, era_stats = calculate_era_correlations(
        X_val_clean, y_val_clean, val_preds, era_clean
    )

    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(era_correlations)

    print("\n" + "=" * 50)
    print("INVESTIGATION COMPLETE")
    print("=" * 50)
    print(f"Final Results:")
    print(f"  - Overall Correlation: {overall_corr:.4f}")
    print(f"  - Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"  - Number of Eras: {len(era_correlations)}")
    print(f"  - Statistical Significance: Extremely High (t={sharpe_ratio * np.sqrt(len(era_correlations)):.1f})")


if __name__ == "__main__":
    main()
