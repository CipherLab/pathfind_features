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
        
        # Calculate the number of eras for each part of the split
        n_train_eras = (n_eras - self.gap_eras) // self.n_splits
        
        splits = []
        for i in range(self.n_splits):
            train_start_era = unique_eras[i * n_train_eras]
            train_end_era = unique_eras[(i + 1) * n_train_eras - 1]
            
            val_start_era = unique_eras[(i + 1) * n_train_eras + self.gap_eras]
            
            # For the last split, validation goes to the end
            if i == self.n_splits - 1:
                val_end_era = unique_eras[-1]
            else:
                val_end_era = unique_eras[(i + 1) * n_train_eras + self.gap_eras + n_train_eras -1]


            train_indices = X[X[self.era_column].between(train_start_era, train_end_era)].index
            val_indices = X[X[self.era_column].between(val_start_era, val_end_era)].index
            
            splits.append((train_indices, val_indices))

        return splits


def load_vix_data(eras: list) -> pd.DataFrame:
    """
    Placeholder function to simulate loading VIX data.
    In a real scenario, this function would load VIX data from a file or API
    and align it with the given eras.
    """
    print("Simulating VIX data loading...")
    # Create a dummy VIX dataframe with random values
    vix_data = pd.DataFrame({
        'era': eras,
        'vix': np.random.uniform(10, 40, size=len(eras))
    })
    return vix_data


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
    sharpe_with_tc = max(0, sharpe - tc_impact)

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

    model = lgb.train(lgb_params, train_set, num_boost_round=200, verbose_eval=False)
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


def run_validation(data_file: str, features_file: str,
                   params_file: str,
                   n_splits: int = 5, gap_eras: int = 100) -> Dict:
    """Run the full validation framework."""
    print("=" * 80)
    print("RUNNING VALIDATION FRAMEWORK")
    print("=" * 80)

    # Load data
    pf = pq.ParquetFile(data_file)
    df = pf.read().to_pandas()
    with open(features_file, 'r') as f:
        features = json.load(f)
    
    X = df[features].astype('float32')
    y = df['adaptive_target'].astype('float32')
    era_series = df['era']

    # Load VIX data and categorize eras
    vix_data = load_vix_data(era_series.unique().tolist())
    vix_regimes = categorize_eras_by_vix(era_series, vix_data)

    # Initialize cross-validator
    cv = EraAwareCrossValidator(n_splits=n_splits, gap_eras=gap_eras)

    # Load parameters
    with open(params_file, 'r') as f:
        best_params = json.load(f)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        print(f"\nFold {fold_idx + 1}/{n_splits}:")

        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]
        era_val = era_series.loc[val_idx]
        regimes_val = vix_regimes.loc[val_idx]

        print(f"  Train: {len(X_train):,} samples, {len(era_val.unique())} eras")
        print(f"  Val:   {len(X_val):,} samples, {len(era_val.unique())} eras")

        fold_result = train_and_evaluate_fold(X_train, y_train, X_val, y_val, best_params, era_val, regimes_val)
        fold_results.append(fold_result)

        print(f"  Overall correlation: {fold_result['overall_correlation']:.4f}")
        print(f"  Sharpe with TC: {fold_result['sharpe_with_tc']:.2f}")

    # Aggregate and print results
    # ... (summary printing logic to be added)

    return {"fold_results": fold_results}


def main():
    """Main function to run the validation framework."""
    # Note: The paths are relative to the new project_refactor directory
    data_file = "dump/pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet"
    features_file = "dump/pipeline_runs/my_experiment/adaptive_only_model.json"
    params_file = "dump/hyperparameter_tuning_optimized/best_params_sharpe.json"

    for file_path in [data_file, features_file, params_file]:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found!")
            return

    results = run_validation(data_file, features_file, params_file)
    # print_summary(results) # Implement this next

if __name__ == "__main__":
    main()
