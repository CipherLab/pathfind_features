#!/usr/bin/env python3
"""
Out-of-Sample Testing Framework
Tests model performance on completely different time periods to check generalization.
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
import warnings
warnings.filterwarnings('ignore')


def load_out_of_sample_data(data_file: str, features_file: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load out-of-sample data for generalization testing."""
    print(f"Loading out-of-sample data from {data_file}...")

    # Load data
    pf = pq.ParquetFile(data_file)
    df = pf.read().to_pandas()

    # Load features
    with open(features_file, 'r') as f:
        features = json.load(f)

    print(f"Dataset: {len(df):,} rows, {len(features)} features")

    # Prepare data
    X = df[features].astype('float32')
    y = df['adaptive_target'].astype('float32')
    era_series = df['era']

    # Filter out NaN values
    valid_mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
    if era_series is not None:
        valid_mask &= ~pd.isna(era_series)

    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    era_clean = era_series[valid_mask]

    print(f"After cleaning: {len(X_clean):,} rows ({len(X_clean)/len(df)*100:.1f}%)")
    print(f"Era range: {era_clean.min()} to {era_clean.max()}")

    return X_clean, y_clean, era_clean


def train_on_historical_test_on_future(train_file: str, test_file: str, features_file: str,
                                       params: Dict) -> Dict:
    """Train on historical data, test on future data."""
    print("\n" + "=" * 60)
    print("TRAIN ON HISTORICAL, TEST ON FUTURE")
    print("=" * 60)

    # Load training data (historical)
    print("Loading historical training data...")
    X_train, y_train, era_train = load_out_of_sample_data(train_file, features_file)

    # Load test data (future)
    print("Loading future test data...")
    X_test, y_test, era_test = load_out_of_sample_data(test_file, features_file)

    # Check era separation
    train_eras = set(era_train.unique())
    test_eras = set(era_test.unique())
    overlap = train_eras.intersection(test_eras)

    print(f"\nEra Analysis:")
    print(f"  Training eras: {len(train_eras)} (min: {min(train_eras)}, max: {max(train_eras)})")
    print(f"  Test eras: {len(test_eras)} (min: {min(test_eras)}, max: {max(test_eras)})")
    print(f"  Era overlap: {len(overlap)} eras")

    if overlap:
        print(f"  ⚠️  WARNING: {len(overlap)} overlapping eras detected!")
        return {"error": "Era overlap detected", "overlap_count": len(overlap)}

    # Train model
    print("\nTraining model on historical data...")
    train_set = lgb.Dataset(X_train, label=y_train)
    lgb_params = params.copy()
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'l2'
    lgb_params['seed'] = 42

    model = lgb.train(lgb_params, train_set, num_boost_round=200)

    # Test on future data
    print("Testing on future data...")
    test_preds = model.predict(X_test)

    # Ensure predictions are numpy array
    test_preds = np.asarray(test_preds).flatten()

    # Calculate performance metrics
    overall_corr, _ = spearmanr(y_test, test_preds)

    # Era-by-era correlations
    era_correlations = []
    for era in sorted(era_test.unique()):
        era_mask = era_test == era
        era_targets = y_test[era_mask]
        era_predictions = test_preds[era_mask]

        if len(era_targets) > 1:
            era_corr, _ = spearmanr(era_targets, era_predictions)
            era_correlations.append(era_corr)

    # Calculate Sharpe ratios
    if era_correlations:
        mean_corr = np.mean(era_correlations)
        std_corr = np.std(era_correlations, ddof=1)
        sharpe = mean_corr / std_corr if std_corr > 0 else 0

        # Transaction cost adjustment
        assumed_volatility = 0.15
        tc_impact = 10 / 10000 / assumed_volatility  # 10bps
        sharpe_with_tc = np.maximum(0.0, sharpe - tc_impact)
    else:
        sharpe = sharpe_with_tc = 0.0

    results = {
        "overall_correlation": overall_corr,
        "era_correlations": era_correlations,
        "sharpe_ratio": sharpe,
        "sharpe_with_tc": sharpe_with_tc,
        "n_test_eras": len(era_correlations),
        "train_eras": len(train_eras),
        "test_eras": len(test_eras),
        "era_overlap": len(overlap)
    }

    return results


def run_comprehensive_oos_test(params_file: str, features_file: str) -> Dict:
    """Run comprehensive out-of-sample testing across different time periods."""
    print("=" * 80)
    print("COMPREHENSIVE OUT-OF-SAMPLE TESTING")
    print("=" * 80)

    # Load parameters
    with open(params_file, 'r') as f:
        params = json.load(f)

    # Define test scenarios
    test_scenarios = [
        {
            "name": "Historical vs Recent",
            "train": "v5.0/train.parquet",
            "test": "v5.0/validation.parquet"
        },
        {
            "name": "Train vs Live",
            "train": "v5.0/train.parquet",
            "test": "v5.0/live.parquet"
        }
    ]

    all_results = {}

    for scenario in test_scenarios:
        print(f"\n🔍 Testing Scenario: {scenario['name']}")

        train_file = scenario['train']
        test_file = scenario['test']

        # Check if files exist
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"  ❌ Skipping: Missing files")
            continue

        # Run test
        results = train_on_historical_test_on_future(
            train_file, test_file, features_file, params
        )

        if "error" in results:
            print(f"  ❌ Error: {results['error']}")
            continue

        all_results[scenario['name']] = results

        # Print results
        print("\n📊 Results:")
        print(f"  Overall correlation: {results['overall_correlation']:.4f}")
        print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Sharpe with TC: {results['sharpe_with_tc']:.2f}")
        print(f"  Era overlap: {results['era_overlap']}")

    return all_results


def print_oos_summary(all_results: Dict):
    """Print comprehensive out-of-sample testing summary."""
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE TESTING SUMMARY")
    print("=" * 80)

    if not all_results:
        print("❌ No valid test scenarios completed")
        return

    print("\n📊 SCENARIO RESULTS:")
    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name}:")
        print(f"  Overall correlation: {results['overall_correlation']:.4f}")
        print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Sharpe with TC: {results['sharpe_with_tc']:.2f}")
        print(f"  Test eras: {results['n_test_eras']}")

    # Calculate averages
    correlations = [r['overall_correlation'] for r in all_results.values()]
    sharpes = [r['sharpe_with_tc'] for r in all_results.values()]

    print("\n📈 AVERAGE PERFORMANCE:")
    print(f"  Mean correlation: {np.mean(correlations):.4f}")
    print(f"  Mean Sharpe with TC: {np.mean(sharpes):.2f}")

    print("\n🔍 GENERALIZATION ASSESSMENT:")

    mean_corr = np.mean(correlations)
    mean_sharpe = np.mean(sharpes)

    if mean_sharpe > 1.5:
        print("  ✅ EXCELLENT GENERALIZATION: Strong out-of-sample performance")
    elif mean_sharpe > 1.0:
        print("  ✅ GOOD GENERALIZATION: Solid performance on new data")
    elif mean_sharpe > 0.5:
        print("  ⚠️  MODERATE GENERALIZATION: Some overfitting detected")
    else:
        print("  ❌ POOR GENERALIZATION: Significant overfitting issues")

    print("\n💡 RECOMMENDATIONS:")

    if mean_sharpe < 1.0:
        print("  📊 CONSIDER: More regularization or feature selection")
        print("  🔄 TRY: Different model architectures")
        print("  📈 ADD: More diverse training data")

    print("  ✅ NEXT: Implement walk-forward analysis")
    print("  ✅ NEXT: Add ensemble methods for stability")


def main():
    """Main function to run out-of-sample testing."""
    params_file = "hyperparameter_tuning_optimized/best_params_sharpe.json"
    features_file = "pipeline_runs/my_experiment/adaptive_only_model.json"

    # Check if files exist
    if not os.path.exists(params_file):
        print(f"❌ Error: {params_file} not found!")
        return

    if not os.path.exists(features_file):
        print(f"❌ Error: {features_file} not found!")
        return

    # Run comprehensive testing
    results = run_comprehensive_oos_test(params_file, features_file)

    # Print summary
    print_oos_summary(results)


if __name__ == "__main__":
    main()
