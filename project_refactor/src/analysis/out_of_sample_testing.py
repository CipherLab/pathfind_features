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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_out_of_sample_data_chunked(data_file: str, features_file: str, weights_file: Optional[str] = None, 
                                   chunk_size: int = 100000) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load out-of-sample data using chunked processing for memory efficiency."""
    print(f"Loading out-of-sample data from {data_file} using chunks...")

    # Load features
    with open(features_file, 'r') as f:
        features = json.load(f)

    pf = pq.ParquetFile(data_file)
    
    # Check if adaptive_target exists in schema
    schema_cols = pf.schema.names
    has_adaptive = 'adaptive_target' in schema_cols
    
    if not has_adaptive and not weights_file:
        raise ValueError(f"adaptive_target not found in {data_file} and no weights file provided")

    # Get target columns for adaptive target creation
    target_cols = [col for col in schema_cols if col.startswith('target')]
    
    # Load weights if needed
    weights_data = None
    if not has_adaptive and weights_file and os.path.exists(weights_file):
        with open(weights_file, 'r') as f:
            weights_data = json.load(f)
        print(f"Loaded weights for {len(weights_data)} eras")

    # Process in chunks
    all_X = []
    all_y = []
    all_era = []
    
    total_rows = 0
    needed_cols = features + ['era']
    if has_adaptive:
        needed_cols.append('adaptive_target')
    else:
        needed_cols.extend(target_cols)
    
    for batch in pf.iter_batches(columns=needed_cols, batch_size=chunk_size):
        df = batch.to_pandas()
        
        # Create adaptive targets if needed
        if not has_adaptive and weights_data:
            adaptive_targets = []
            for _, row in df.iterrows():
                era = row['era']
                if str(era) in weights_data:
                    weights = weights_data[str(era)]
                elif era in weights_data:
                    weights = weights_data[era]
                else:
                    # Use equal weights if era not found
                    weights = [1.0/len(target_cols)] * len(target_cols)

                # Calculate weighted target
                target_values = [row[col] for col in target_cols]
                adaptive_target = sum(w * t for w, t in zip(weights, target_values) if pd.notna(t))
                adaptive_targets.append(adaptive_target)
            
            df['adaptive_target'] = adaptive_targets
        
        # Prepare data
        X_chunk = df[features].astype('float32')
        y_chunk = df['adaptive_target'].astype('float32')
        era_chunk = df['era']
        
        # Filter out NaN values
        valid_mask = ~(pd.isna(y_chunk) | pd.isna(X_chunk).any(axis=1))
        if era_chunk is not None:
            valid_mask &= ~pd.isna(era_chunk)
        
        all_X.append(X_chunk[valid_mask])
        all_y.append(y_chunk[valid_mask])
        all_era.append(era_chunk[valid_mask])
        
        total_rows += len(df)
        if total_rows % 500000 == 0:
            print(f"Processed {total_rows:,} rows...")

    # Concatenate all chunks
    X_clean = pd.concat(all_X, ignore_index=True)
    
    # Concatenate Series properly
    if all_y:
        y_clean = pd.concat(all_y, ignore_index=True)
        if isinstance(y_clean, pd.DataFrame):
            y_clean = y_clean.iloc[:, 0]  # Take first column if DataFrame
    else:
        y_clean = pd.Series([], dtype='float32', name='adaptive_target')
    
    if all_era:
        era_clean = pd.concat(all_era, ignore_index=True)
        if isinstance(era_clean, pd.DataFrame):
            era_clean = era_clean.iloc[:, 0]  # Take first column if DataFrame
    else:
        era_clean = pd.Series([], dtype='object', name='era')
    
    print(f"Dataset: {len(X_clean):,} rows, {len(features)} features")
    if len(era_clean) > 0:
        print(f"Era range: {era_clean.min()} to {era_clean.max()}")

    return X_clean, y_clean, era_clean


def train_on_historical_test_on_future(train_file: str, test_file: str, features_file: str,
                                       params: Dict, weights_file: Optional[str] = None) -> Dict:
    """Train on historical data, test on future data."""
    print("\n" + "=" * 60)
    print("TRAIN ON HISTORICAL, TEST ON FUTURE")
    print("=" * 60)

    # Load training data (historical)
    print("Loading historical training data...")
    X_train, y_train, era_train = load_out_of_sample_data_chunked(train_file, features_file, weights_file)

    # Load test data (future)
    print("Loading future test data...")
    X_test, y_test, era_test = load_out_of_sample_data_chunked(test_file, features_file, weights_file)

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
    
    # Make predictions in chunks to handle large datasets
    predictions = []
    chunk_size = 100000
    
    for i in range(0, len(X_test), chunk_size):
        X_chunk = X_test.iloc[i:i+chunk_size]
        chunk_preds = model.predict(X_chunk)
        
        # Convert to numpy array and then to list
        chunk_preds = np.asarray(chunk_preds).flatten()
        predictions.extend(chunk_preds.tolist())
    
    test_preds = np.array(predictions)
    
    # Calculate overall correlation
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


def run_comprehensive_oos_test(params_file: str, features_file: str, weights_file: Optional[str] = None) -> Dict:
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
            "name": "Adaptive Train vs Validation",
            "train": "v5.0/train_with_adaptive.parquet",
            "test": "pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet"
        },
        {
            "name": "Adaptive Train vs Live",
            "train": "v5.0/train_with_adaptive.parquet",
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
            train_file, test_file, features_file, params, weights_file
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
    weights_file = "v5.0/weights_by_era_full.json"

    # Check if files exist
    if not os.path.exists(params_file):
        print(f"❌ Error: {params_file} not found!")
        return

    if not os.path.exists(features_file):
        print(f"❌ Error: {features_file} not found!")
        return

    if not os.path.exists(weights_file):
        print(f"⚠️  Warning: {weights_file} not found - will skip adaptive target creation")
        weights_file = None

    # Run comprehensive testing
    results = run_comprehensive_oos_test(params_file, features_file, weights_file)

    # Print summary
    print_oos_summary(results)


if __name__ == "__main__":
    main()
