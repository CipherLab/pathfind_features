#!/usr/bin/env python3
"""
Validation Framework for Financial Time Series Models

This script provides a robust, era-aware cross-validation framework with features
specifically designed for financial machine learning, including:

- Era-aware splitting with configurable gaps between train and validation sets.
- Regime-based analysis using VIX thresholds to evaluate performance in different
  market conditions (e.g., crisis, low-volatility).
- Transaction cost (TC) adjusted Sharpe ratio calculation.

This framework is designed to provide a more realistic assessment of model
performance than standard cross-validation.
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
    """Implements era-aware cross-validation with a gap between train and validation sets."""

    def __init__(self, n_splits: int = 5, era_column: str = 'era', gap_eras: int = 100):
        self.n_splits = n_splits
        self.era_column = era_column
        self.gap_eras = gap_eras

    def split(self, X: pd.DataFrame) -> List[Tuple]:
        """Generate era-aware train/validation splits."""
        unique_eras = sorted(X[self.era_column].unique())
        n_eras = len(unique_eras)
        
        # Ensure we have enough eras for the requested splits
        min_eras_needed = self.n_splits * 4 + self.gap_eras  # Conservative estimate
        if n_eras < min_eras_needed:
            raise ValueError(f"Not enough eras ({n_eras}) for {self.n_splits} splits with gap {self.gap_eras}. Need at least {min_eras_needed}.")
        
        # Calculate split size to fit all splits
        total_space_needed = self.n_splits * 2 + (self.n_splits - 1)  # train + val for each + gaps between
        available_space = n_eras - self.gap_eras
        split_size = max(1, available_space // total_space_needed)
        
        n_train_eras = split_size
        n_val_eras = split_size
        
        splits = []
        current_era_idx = 0
        
        for i in range(self.n_splits):
            # Calculate train era range
            train_start_idx = current_era_idx
            train_end_idx = min(train_start_idx + n_train_eras - 1, n_eras - 1)
            
            # Calculate validation era range (after gap)
            val_start_idx = train_end_idx + 1 + self.gap_eras
            val_end_idx = min(val_start_idx + n_val_eras - 1, n_eras - 1)
            
            # Ensure we have valid validation data
            if val_start_idx >= n_eras or val_end_idx < val_start_idx:
                break
                
            train_start_era = unique_eras[train_start_idx]
            train_end_era = unique_eras[train_end_idx]
            val_start_era = unique_eras[val_start_idx]
            val_end_era = unique_eras[val_end_idx]

            train_indices = X[X[self.era_column].between(train_start_era, train_end_era)].index
            val_indices = X[X[self.era_column].between(val_start_era, val_end_era)].index
            
            splits.append((train_indices, val_indices))
            
            # Move to next era for next split
            current_era_idx = val_end_idx + 1

        return splits


def load_vix_data(eras: list, vix_file: str | None = None) -> pd.DataFrame:
    """
    Load VIX data and align it with the given eras.
    If vix_file is provided, loads from that file. Otherwise, simulates data.
    """
    if vix_file and os.path.exists(vix_file):
        try:
            vix_df = pd.read_csv(vix_file)
            # Assume CSV has 'era' and 'vix' columns
            vix_df['era'] = vix_df['era'].astype(str)
            return vix_df[['era', 'vix']]
        except Exception as e:
            print(f"Warning: Failed to load VIX data from {vix_file}: {e}")
            print("Falling back to simulated data...")

    print("Simulating VIX data loading...")
    # Create more realistic VIX data with some temporal patterns
    np.random.seed(42)  # For reproducibility
    n_eras = len(eras)

    # Generate VIX values with some autocorrelation and regime-like behavior
    vix_values = []
    current_vix = 20  # Starting VIX level

    for i in range(n_eras):
        # Add some random walk with mean reversion
        change = np.random.normal(0, 2)
        current_vix = max(10, min(50, current_vix + change))
        vix_values.append(current_vix)

    return pd.DataFrame({
        'era': eras,
        'vix': vix_values
    })


def categorize_eras_by_vix(era_series: pd.Series, vix_data: pd.DataFrame) -> pd.Series:
    """Categorize eras into volatility regimes based on VIX levels."""
    era_to_vix = vix_data.set_index('era')['vix']
    
    def get_regime(era):
        vix = era_to_vix.get(era, np.nan)
        if pd.isna(vix):
            return "unknown"
        if vix > 25:
            return "high_vol_crisis"
        elif vix < 15:
            return "low_vol_grind"
        else:
            return "transition"
            
    return era_series.apply(get_regime)


def calculate_realistic_sharpe(era_correlations: List[float], transaction_cost_bps: float = 25) -> Dict:
    """Calculate Sharpe ratio with transaction cost adjustments."""
    if not era_correlations:
        return {"sharpe_ratio": 0, "sharpe_with_tc": 0, "tc_impact": 0}

    mean_corr = np.mean(era_correlations)
    std_corr = np.std(era_correlations, ddof=1)
    sharpe = mean_corr / std_corr if std_corr > 0 else 0

    # Estimate transaction cost impact
    assumed_volatility = 0.15  # 15% annual volatility
    tc_impact = transaction_cost_bps / 10000 / assumed_volatility
    sharpe_with_tc = max(0.0, float(sharpe - tc_impact))

    return {
        "sharpe_ratio": sharpe,
        "sharpe_with_tc": sharpe_with_tc,
        "tc_impact": tc_impact,
        "mean_correlation": mean_corr,
        "std_correlation": std_corr
    }


def train_and_evaluate_fold(X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           params: Dict, era_val: pd.Series, vix_regimes: pd.Series) -> Dict:
    """Train model and evaluate on a single fold, including regime analysis."""
    train_set = lgb.Dataset(X_train, label=y_train)
    lgb_params = params.copy()
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'l2'
    lgb_params['seed'] = 42

    model = lgb.train(lgb_params, train_set, num_boost_round=200)
    val_preds = model.predict(X_val)

    # Overall correlation
    overall_corr, _ = spearmanr(y_val, val_preds)

    # Era-level correlations
    era_correlations = []
    for era in sorted(era_val.unique()):
        era_mask = era_val == era
        era_targets = y_val[era_mask]
        era_predictions = val_preds[era_mask]
        if len(era_targets) > 1:
            era_corr, _ = spearmanr(era_targets, era_predictions)
            era_correlations.append(era_corr)

    # Regime-level correlations
    regime_metrics = {}
    for regime in vix_regimes.unique():
        regime_mask = vix_regimes == regime
        regime_targets = y_val[regime_mask]
        regime_preds = val_preds[regime_mask]
        if len(regime_targets) > 1:
            regime_corr, _ = spearmanr(regime_targets, regime_preds)
            regime_metrics[f'corr_{regime}'] = regime_corr

    sharpe_results = calculate_realistic_sharpe(era_correlations)

    return {
        "overall_correlation": overall_corr,
        "era_correlations": era_correlations,
        **sharpe_results,
        **regime_metrics,
        "n_eras": len(era_correlations)
    }


def aggregate_validation_results(fold_results: List[Dict], vix_data: pd.DataFrame, era_series: pd.Series) -> Dict:
    """Aggregate results across folds and analyze regime performance."""
    # Aggregate basic metrics
    aggregated = {}
    metric_keys = ['overall_correlation', 'sharpe_ratio', 'sharpe_with_tc', 'mean_correlation', 'std_correlation']

    for key in metric_keys:
        values = [result.get(key, 0) for result in fold_results if key in result]
        if values:
            aggregated[f'mean_{key}'] = float(np.mean(values))
            aggregated[f'std_{key}'] = float(np.std(values))

    # Aggregate regime-specific metrics
    regime_keys = [k for k in fold_results[0].keys() if k.startswith('corr_')]
    for key in regime_keys:
        values = [result.get(key, np.nan) for result in fold_results]
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            aggregated[f'mean_{key}'] = float(np.mean(valid_values))
            aggregated[f'std_{key}'] = float(np.std(valid_values))

    # Analyze VIX regime distribution
    vix_regimes = categorize_eras_by_vix(era_series, vix_data)
    regime_counts = vix_regimes.value_counts()
    aggregated['regime_distribution'] = regime_counts.to_dict()

    # Calculate regime stability metrics
    vix_stats = vix_data['vix'].describe()
    aggregated['vix_stats'] = {
        'mean': float(vix_stats['mean']),
        'std': float(vix_stats['std']),
        'min': float(vix_stats['min']),
        'max': float(vix_stats['max'])
    }

    return aggregated


def print_validation_summary(aggregated_results: Dict):
    """Print a comprehensive summary of validation results."""
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print("\nOverall Performance:")
    if 'mean_overall_correlation' in aggregated_results:
        print(f"  Overall Correlation: {aggregated_results['mean_overall_correlation']:.4f}")
    if 'mean_sharpe_with_tc' in aggregated_results:
        print(f"  Sharpe with TC: {aggregated_results['mean_sharpe_with_tc']:.2f}")

    print("\nRegime Distribution:")
    regime_dist = aggregated_results.get('regime_distribution', {})
    for regime, count in regime_dist.items():
        print(f"  {regime}: {count} eras")

    print("\nRegime-Specific Performance:")
    for key, value in aggregated_results.items():
        if key.startswith('mean_corr_'):
            regime = key.replace('mean_corr_', '')
            std_key = key.replace('mean_', 'std_')
            std_val = aggregated_results.get(std_key, 0)
            print(f"  {regime}: {value:.4f} (std: {std_val:.4f})")

    print("\nVIX Statistics:")
    vix_stats = aggregated_results.get('vix_stats', {})
    print(f"  Mean: {vix_stats.get('mean', 0):.1f}")
    print(f"  Std: {vix_stats.get('std', 0):.1f}")
    print(f"  Min: {vix_stats.get('min', 0):.1f}")
    print(f"  Max: {vix_stats.get('max', 0):.1f}")


def run_honest_validation(data_file: str, features_file: str,
                         params_file: str, vix_file: str | None = None,
                         n_splits: int = 5, gap_eras: int = 100) -> Dict:
    """Run an honest validation framework that predicts live performance."""
    print("=" * 80)
    print("RUNNING HONEST VALIDATION FRAMEWORK")
    print("=" * 80)

    # Load data
    pf = pq.ParquetFile(data_file)
    df = pf.read().to_pandas()
    with open(features_file, 'r') as f:
        features_dict = json.load(f)
    
    # Extract feature list from dict structure
    if isinstance(features_dict, dict) and 'feature_sets' in features_dict:
        features = features_dict['feature_sets'].get('medium', [])
    elif isinstance(features_dict, list):
        features = features_dict
    else:
        raise ValueError(f"Invalid features file format: {features_file}")

    X = df[features].astype('float32')
    X['era'] = df['era']  # Add era column for cross-validation
    y = df['adaptive_target'].astype('float32')
    era_series = df['era']

    # Load VIX data and categorize eras
    vix_data = load_vix_data(era_series.unique().tolist(), vix_file)
    vix_regimes = categorize_eras_by_vix(era_series, vix_data)

    # Initialize cross-validator with larger gap for honesty
    cv = EraAwareCrossValidator(n_splits=n_splits, gap_eras=max(gap_eras, 200))

    # Load parameters
    with open(params_file, 'r') as f:
        best_params = json.load(f)

    fold_results = []
    time_machine_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        print(f"\nFold {fold_idx + 1}/{n_splits}:")

        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]
        era_val = era_series.loc[val_idx]
        regimes_val = vix_regimes.loc[val_idx]

        print(f"  Train: {len(X_train):,} samples, {len(era_val.unique())} eras")
        print(f"  Val:   {len(X_val):,} samples, {len(era_val.unique())} eras")

        # Remove era column from X before training (LightGBM doesn't need it)
        X_train_features = X_train.drop(columns=['era'])
        X_val_features = X_val.drop(columns=['era'])
        
        fold_result = train_and_evaluate_fold(X_train_features, y_train, X_val_features, y_val, best_params, era_val, regimes_val)
        fold_results.append(fold_result)

        print(f"  Overall correlation: {fold_result['overall_correlation']:.4f}")
        print(f"  Sharpe with TC: {fold_result['sharpe_with_tc']:.2f}")

        # Run time machine test for this fold
        train_data = pd.DataFrame({'era': era_series.loc[train_idx], 'adaptive_target': y_train})
        val_data = pd.DataFrame({'era': era_val, 'adaptive_target': y_val})

        # Add VIX data for regime analysis
        if vix_data is not None:
            train_data = train_data.merge(vix_data, on='era', how='left')
            val_data = val_data.merge(vix_data, on='era', how='left')

        time_machine_result = run_time_machine_test(train_data, val_data, features,
                                                   X_train_features.values, y_train.values,
                                                   X_val_features.values, y_val.values)
        time_machine_results.append(time_machine_result)

    # Aggregate and print results
    aggregated_results = aggregate_validation_results(fold_results, vix_data, era_series)
    print_validation_summary(aggregated_results)

    # Analyze time machine results
    time_machine_summary = analyze_time_machine_results(time_machine_results)
    print("\n" + "=" * 80)
    print("TIME MACHINE TEST RESULTS")
    print("=" * 80)
    print(f"Average train correlation: {time_machine_summary['avg_train_corr']:.4f}")
    print(f"Average test correlation: {time_machine_summary['avg_test_corr']:.4f}")
    print(f"Average drop: {time_machine_summary['avg_drop']:.2f}")
    print(f"Honesty score: {time_machine_summary['honesty_score']:.2f}")
    print(f"Validation is {'HONEST' if time_machine_summary['honesty_score'] > 0.8 else 'QUESTIONABLE'}")

    return {
        "fold_results": fold_results,
        "aggregated": aggregated_results,
        "time_machine": time_machine_summary,
        "honest_assessment": time_machine_summary['honesty_score'] > 0.8
    }


def run_time_machine_test(train_data: pd.DataFrame, test_data: pd.DataFrame,
                         features: list, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Run a time machine test to check if validation predicts live performance."""
    from honest_frameworks import HonestValidationFramework

    honest_validator = HonestValidationFramework(min_era_gap=200)

    # Create dataframes with features
    train_df = pd.DataFrame(X_train, columns=features)
    train_df['adaptive_target'] = y_train
    train_df['era'] = train_data['era'].values

    test_df = pd.DataFrame(X_test, columns=features)
    test_df['adaptive_target'] = y_test
    test_df['era'] = test_data['era'].values

    # Add VIX if available
    if 'vix' in train_data.columns:
        train_df['vix'] = train_data['vix'].values
        test_df['vix'] = test_data['vix'].values

    return honest_validator.time_machine_test(train_df, test_df, features, 'adaptive_target')


def analyze_time_machine_results(time_machine_results: list) -> dict:
    """Analyze results from time machine tests."""
    if not time_machine_results:
        return {"error": "No time machine results"}

    train_corrs = [r['train_correlation'] for r in time_machine_results if not np.isnan(r['train_correlation'])]
    test_corrs = [r['test_correlation'] for r in time_machine_results if not np.isnan(r['test_correlation'])]
    drops = [r['drop'] for r in time_machine_results if not np.isnan(r['drop'])]
    honesty_scores = [r['brutal_honesty_score'] for r in time_machine_results if not np.isnan(r['brutal_honesty_score'])]

    return {
        "avg_train_corr": np.mean(train_corrs) if train_corrs else 0,
        "avg_test_corr": np.mean(test_corrs) if test_corrs else 0,
        "avg_drop": np.mean(drops) if drops else 0,
        "honesty_score": np.mean(honesty_scores) if honesty_scores else 0,
        "n_tests": len(time_machine_results)
    }


def run_validation(data_file: str, features_file: str,
                   params_file: str, vix_file: str | None = None,
                   n_splits: int = 5, gap_eras: int = 100) -> Dict:
    """Run the full validation framework - now defaults to honest validation."""
    return run_honest_validation(data_file, features_file, params_file, vix_file, n_splits, gap_eras)


def main():
    """Main function to run the validation framework."""
    # Note: The paths are relative to the new project_refactor directory
    data_file = "dump/pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet"
    features_file = "dump/pipeline_runs/my_experiment/adaptive_only_model.json"
    params_file = "dump/hyperparameter_tuning_optimized/best_params_sharpe.json"
    vix_file = None  # Optional: path to VIX data CSV

    for file_path in [data_file, features_file, params_file]:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found!")
            return

    results = run_validation(data_file, features_file, params_file, vix_file)
    # Save results to file
    output_file = "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nValidation results saved to {output_file}")

if __name__ == "__main__":
    main()
