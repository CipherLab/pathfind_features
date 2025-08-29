#!/usr/bin/env python3
"""
Robust Cross-Validation Framework for Adaptive Model
Implements proper era-aware validation to address methodological concerns.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import pyarrow.parquet as pq
import lightgbm as lgb
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class EraAwareCrossValidator:
    """Implements proper era-aware cross-validation for financial time series."""

    def __init__(self, n_splits: int = 5, era_column: str = 'era'):
        self.n_splits = n_splits
        self.era_column = era_column

    def split(self, X: pd.DataFrame, y: pd.Series, era_series: pd.Series) -> List[Tuple]:
        """Generate era-aware train/validation splits."""
        splits = []

        # Sort by era to ensure temporal order
        sorted_indices = era_series.sort_values().index
        X_sorted = X.loc[sorted_indices]
        y_sorted = y.loc[sorted_indices]
        era_sorted = era_series.loc[sorted_indices]

        # Get unique eras in order
        unique_eras = era_sorted.unique()
        n_eras = len(unique_eras)

        # Create splits ensuring temporal order
        for i in range(self.n_splits):
            # Use first 60% of eras for training, next 20% for validation
            train_end_idx = int(n_eras * 0.6 * (i + 1) / self.n_splits)
            val_start_idx = train_end_idx
            val_end_idx = int(n_eras * 0.8 * (i + 1) / self.n_splits)

            if val_end_idx > n_eras:
                val_end_idx = n_eras

            train_eras = unique_eras[:train_end_idx]
            val_eras = unique_eras[val_start_idx:val_end_idx]

            train_mask = era_sorted.isin(train_eras)
            val_mask = era_sorted.isin(val_eras)

            train_indices = X_sorted[train_mask].index
            val_indices = X_sorted[val_mask].index

            splits.append((train_indices, val_indices))

        return splits


def load_adaptive_data(data_file: str, features_file: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load adaptive target data with proper validation."""
    print("Loading adaptive target data...")

    # Load validation data
    pf_val = pq.ParquetFile(data_file)
    val_df = pf_val.read().to_pandas()

    # Load features
    with open(features_file, 'r') as f:
        features = json.load(f)

    print(f"Dataset: {len(val_df):,} rows, {len(features)} features")

    # Prepare data
    X = val_df[features].astype('float32')
    y = val_df['adaptive_target'].astype('float32')
    era_series = val_df['era']

    # Filter out NaN values
    valid_mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
    if era_series is not None:
        valid_mask &= ~pd.isna(era_series)

    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    era_clean = era_series[valid_mask]

    print(f"After cleaning: {len(X_clean):,} rows ({len(X_clean)/len(val_df)*100:.1f}%)")
    print(f"Era range: {era_clean.min()} to {era_clean.max()}")

    return X_clean, y_clean, era_clean


def calculate_realistic_sharpe(era_correlations: List[float], transaction_cost_bps: float = 10) -> Dict:
    """Calculate Sharpe ratio with transaction cost adjustments."""
    if not era_correlations:
        return {"sharpe_ratio": 0, "sharpe_with_tc": 0, "tc_impact": 0}

    # Basic Sharpe calculation
    mean_corr = np.mean(era_correlations)
    std_corr = np.std(era_correlations, ddof=1)
    sharpe = mean_corr / std_corr if std_corr > 0 else 0

    # Estimate transaction cost impact
    # Assume correlation roughly translates to returns, and TC reduces Sharpe
    assumed_volatility = 0.15  # 15% annual volatility
    tc_impact = transaction_cost_bps / 10000 / assumed_volatility  # Convert bps to Sharpe reduction
    sharpe_with_tc = max(0, sharpe - tc_impact)  # Can't go below 0

    return {
        "sharpe_ratio": sharpe,
        "sharpe_with_tc": sharpe_with_tc,
        "tc_impact": tc_impact,
        "mean_correlation": mean_corr,
        "std_correlation": std_corr
    }


def train_and_evaluate_fold(X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           params: Dict, era_val: pd.Series) -> Dict:
    """Train model and evaluate on a single fold."""
    # Train model
    train_set = lgb.Dataset(X_train, label=y_train)
    lgb_params = params.copy()
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'l2'
    lgb_params['seed'] = 42

    model = lgb.train(lgb_params, train_set, num_boost_round=200, verbose_eval=False)

    # Make predictions
    val_preds = model.predict(X_val)

    # Calculate overall correlation
    overall_corr, _ = spearmanr(y_val, val_preds)

    # Calculate era correlations
    era_correlations = []
    for era in sorted(era_val.unique()):
        era_mask = era_val == era
        era_targets = y_val[era_mask]
        era_predictions = val_preds[era_mask]

        if len(era_targets) > 1:
            era_corr, _ = spearmanr(era_targets, era_predictions)
            era_correlations.append(era_corr)

    # Calculate Sharpe ratios
    sharpe_results = calculate_realistic_sharpe(era_correlations)

    return {
        "overall_correlation": overall_corr,
        "era_correlations": era_correlations,
        **sharpe_results,
        "n_eras": len(era_correlations)
    }


def run_robust_cross_validation(data_file: str, features_file: str, params_file: str,
                               n_splits: int = 5) -> Dict:
    """Run comprehensive cross-validation with proper methodology."""
    print("=" * 80)
    print("ROBUST ERA-AWARE CROSS-VALIDATION")
    print("=" * 80)

    # Load data
    X, y, era_series = load_adaptive_data(data_file, features_file)

    # Load parameters
    with open(params_file, 'r') as f:
        best_params = json.load(f)

    # Initialize cross-validator
    cv = EraAwareCrossValidator(n_splits=n_splits)

    # Run cross-validation
    fold_results = []
    print(f"\nRunning {n_splits}-fold cross-validation...")

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, era_series)):
        print(f"\nFold {fold_idx + 1}/{n_splits}:")

        # Split data
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]
        era_val = era_series.loc[val_idx]

        print(f"  Train: {len(X_train):,} samples, {len(era_val.unique())} eras")
        print(f"  Val: {len(X_val):,} samples, {len(era_val.unique())} eras")

        # Train and evaluate
        fold_result = train_and_evaluate_fold(X_train, y_train, X_val, y_val, best_params, era_val)
        fold_results.append(fold_result)

        print(f"  Overall correlation: {fold_result['overall_correlation']:.4f}")
        print(f"  Sharpe ratio: {fold_result['sharpe_ratio']:.2f}")
        print(f"  Sharpe with TC: {fold_result['sharpe_with_tc']:.2f}")

    # Aggregate results
    overall_correlations = [r['overall_correlation'] for r in fold_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in fold_results]
    sharpe_with_tc = [r['sharpe_with_tc'] for r in fold_results]

    results = {
        "fold_results": fold_results,
        "mean_overall_correlation": np.mean(overall_correlations),
        "std_overall_correlation": np.std(overall_correlations),
        "mean_sharpe_ratio": np.mean(sharpe_ratios),
        "std_sharpe_ratio": np.std(sharpe_ratios),
        "mean_sharpe_with_tc": np.mean(sharpe_with_tc),
        "std_sharpe_with_tc": np.std(sharpe_with_tc),
        "n_folds": n_splits
    }

    return results


def print_validation_summary(results: Dict):
    """Print comprehensive validation summary."""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    print("
📊 PERFORMANCE METRICS:")
    print(".4f")
    print(".4f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")

    print("
🔍 INTERPRETATION:")

    mean_sharpe = results['mean_sharpe_with_tc']
    if mean_sharpe > 2.0:
        print(f"  ✅ EXCELLENT: Sharpe {mean_sharpe:.1f} (top-tier performance)")
    elif mean_sharpe > 1.0:
        print(f"  ✅ GOOD: Sharpe {mean_sharpe:.1f} (solid performance)")
    elif mean_sharpe > 0.5:
        print(f"  ⚠️  MODERATE: Sharpe {mean_sharpe:.1f} (needs improvement)")
    else:
        print(f"  ❌ POOR: Sharpe {mean_sharpe:.1f} (significant issues)")

    print("
📈 FOLD-BY-FOLD RESULTS:")
    for i, fold in enumerate(results['fold_results']):
        print(f"  Fold {i+1}: Corr={fold['overall_correlation']:.3f}, "
              f"Sharpe={fold['sharpe_with_tc']:.2f}")

    print("
💡 RECOMMENDATIONS:")

    if results['std_sharpe_with_tc'] > 0.5:
        print("  ⚠️  HIGH VARIABILITY: Consider model regularization")
    else:
        print("  ✅ CONSISTENT: Good stability across folds")

    if results['mean_sharpe_with_tc'] < 1.0:
        print("  📊 IMPROVEMENT NEEDED: Consider feature engineering or model architecture")
    else:
        print("  🚀 READY FOR PRODUCTION: Strong, consistent performance")


def main():
    """Main function to run robust cross-validation."""
    # File paths
    data_file = "pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet"
    features_file = "pipeline_runs/my_experiment/adaptive_only_model.json"
    params_file = "hyperparameter_tuning_optimized/best_params_sharpe.json"

    # Check if files exist
    for file_path in [data_file, features_file, params_file]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            return

    # Run cross-validation
    results = run_robust_cross_validation(data_file, features_file, params_file, n_splits=5)

    # Print summary
    print_validation_summary(results)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. ✅ Era-aware cross-validation implemented")
    print("2. ✅ Transaction cost analysis included")
    print("3. 🔄 Test on completely different time periods")
    print("4. 🔄 Implement walk-forward analysis")
    print("5. 🔄 Add more baseline comparisons")
    print("6. 🔄 Consider ensemble methods for stability")


if __name__ == "__main__":
    main()
