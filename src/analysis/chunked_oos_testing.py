#!/usr/bin/env python3
"""
Chunked Out-of-Sample Testing Framework
Tests model performance on different time periods using memory-efficient chunked processing.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import pyarrow.parquet as pq
import lightgbm as lgb
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def select_features(pf: pq.ParquetFile, features_json: str) -> list[str]:
    """Select features from parquet file based on features_json or heuristic."""
    cols = pf.schema.names
    if features_json and Path(features_json).exists():
        with open(features_json, 'r') as f:
            features_data = json.load(f)
        if isinstance(features_data, dict) and 'feature_sets' in features_data:
            if 'medium' in features_data['feature_sets']:
                desired = features_data['feature_sets']['medium']
            else:
                desired = list(features_data['feature_sets'].values())[0] if features_data['feature_sets'] else []
        elif isinstance(features_data, list):
            desired = features_data
        else:
            desired = []
    else:
        # Fallback heuristic
        desired = [c for c in cols if c.startswith('feature')]

    # Keep only features present in the dataset
    features = [c for c in desired if c in cols]
    if len(features) != len(desired):
        dropped = len(desired) - len(features)
        print(f"Warning: Dropped {dropped} features not present in dataset")

    return features


def create_adaptive_targets_chunked(df: pd.DataFrame, weights_file: str) -> pd.Series:
    """Create adaptive targets for a data chunk."""
    if 'adaptive_target' in df.columns:
        return df['adaptive_target']

    # Load weights
    with open(weights_file, 'r') as f:
        weights_data = json.load(f)

    # Get target columns
    target_cols = [col for col in df.columns if col.startswith('target')]

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

    return pd.Series(adaptive_targets, index=df.index)


def process_chunk_predictions(model, chunk_df: pd.DataFrame, features: List[str],
                            target_col: str, weights_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a chunk and return predictions, targets, and eras."""
    # Create adaptive targets if needed
    if target_col == 'adaptive_target' and target_col not in chunk_df.columns:
        if weights_file:
            chunk_df = chunk_df.copy()
            chunk_df[target_col] = create_adaptive_targets_chunked(chunk_df, weights_file)
        else:
            raise ValueError("weights_file required for adaptive_target creation")

    # Filter out NaN values
    valid_mask = ~(pd.isna(chunk_df[target_col]) | pd.isna(chunk_df[features]).any(axis=1))
    if 'era' in chunk_df.columns:
        valid_mask &= ~pd.isna(chunk_df['era'])

    if valid_mask.sum() == 0:
        return np.array([]), np.array([]), np.array([])

    chunk_valid = chunk_df[valid_mask]
    X_chunk = chunk_valid[features].astype('float32')
    y_chunk = chunk_valid[target_col].astype('float32')
    era_chunk = chunk_valid['era'].values if 'era' in chunk_valid.columns else np.array([])

    # Make predictions
    preds = model.predict(X_chunk)

    return preds, y_chunk.values, era_chunk


def train_on_chunked_data(train_path: str, features: List[str], target_col: str,
                         params: Dict, weights_file: Optional[str] = None,
                         chunk_size: int = 250_000) -> lgb.Booster:
    """Train model on chunked data."""
    print(f"Training on chunked data from {train_path}...")

    pf_train = pq.ParquetFile(train_path)
    needed_cols = features + [target_col, 'era'] if 'era' in pf_train.schema.names else features + [target_col]

    booster = None
    total_processed = 0

    for batch in pf_train.iter_batches(columns=needed_cols, batch_size=chunk_size):
        try:
            df = batch.to_pandas(use_pandas_metadata=False)
        except TypeError:
            df = batch.to_pandas()
            if any(n is not None for n in (df.index.names or [])):
                df = df.reset_index()

        # Create adaptive targets if needed
        if target_col == 'adaptive_target' and target_col not in df.columns:
            if weights_file:
                df = df.copy()
                df[target_col] = create_adaptive_targets_chunked(df, weights_file)
            else:
                continue  # Skip chunks without adaptive targets

        # Filter valid data
        valid_mask = ~(pd.isna(df[target_col]) | pd.isna(df[features]).any(axis=1))
        if valid_mask.sum() == 0:
            continue

        df_valid = df[valid_mask]
        X_chunk = df_valid[features].astype('float32')
        y_chunk = df_valid[target_col].astype('float32')

        # Create dataset
        train_set = lgb.Dataset(X_chunk, label=y_chunk, free_raw_data=True)

        # Train or continue training
        if booster is None:
            # First chunk
            booster = lgb.train(params, train_set, num_boost_round=50)
        else:
            # Continue training
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=50,
                init_model=booster,
                keep_training_booster=True
            )

        total_processed += len(df_valid)
        print(f"Processed {total_processed:,} training samples...")

    return booster


def test_on_chunked_data(model, test_path: str, features: List[str], target_col: str,
                        weights_file: Optional[str] = None, chunk_size: int = 250_000) -> Dict:
    """Test model on chunked data."""
    print(f"Testing on chunked data from {test_path}...")

    pf_test = pq.ParquetFile(test_path)
    needed_cols = features + [target_col, 'era'] if 'era' in pf_test.schema.names else features + [target_col]

    all_predictions = []
    all_targets = []
    all_eras = []
    total_processed = 0

    for batch in pf_test.iter_batches(columns=needed_cols, batch_size=chunk_size):
        try:
            df = batch.to_pandas(use_pandas_metadata=False)
        except TypeError:
            df = batch.to_pandas()
            if any(n is not None for n in (df.index.names or [])):
                df = df.reset_index()

        preds, targets, eras = process_chunk_predictions(model, df, features, target_col, weights_file)

        if len(preds) > 0:
            all_predictions.extend(preds)
            all_targets.extend(targets)
            all_eras.extend(eras)

        total_processed += len(df)
        print(f"Processed {total_processed:,} test samples...")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_eras = np.array(all_eras)

    if len(all_predictions) == 0:
        return {"error": "No valid data found for testing"}

    # Calculate overall correlation
    overall_corr, _ = spearmanr(all_targets, all_predictions)

    # Calculate era correlations
    era_correlations = []
    unique_eras = np.unique(all_eras)
    for era in unique_eras:
        era_mask = all_eras == era
        era_targets = all_targets[era_mask]
        era_predictions = all_predictions[era_mask]

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

    return {
        "overall_correlation": overall_corr,
        "era_correlations": era_correlations,
        "sharpe_ratio": sharpe,
        "sharpe_with_tc": sharpe_with_tc,
        "n_test_eras": len(era_correlations),
        "n_test_samples": len(all_predictions)
    }


def run_chunked_oos_test(train_file: str, test_file: str, features_file: str,
                        params_file: str, weights_file: Optional[str] = None,
                        target_col: str = 'adaptive_target', chunk_size: int = 250_000) -> Dict:
    """Run chunked out-of-sample test."""
    print("=" * 80)
    print("CHUNKED OUT-OF-SAMPLE TESTING")
    print("=" * 80)

    # Load parameters
    with open(params_file, 'r') as f:
        params = json.load(f)

    # Setup LightGBM parameters
    lgb_params = params.copy()
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'l2'
    lgb_params['seed'] = 42

    # Get features
    pf_train = pq.ParquetFile(train_file)
    features = select_features(pf_train, features_file)
    print(f"Using {len(features)} features")

    # Train model on chunked data
    model = train_on_chunked_data(train_file, features, target_col, lgb_params, weights_file, chunk_size)

    # Test on chunked data
    results = test_on_chunked_data(model, test_file, features, target_col, weights_file, chunk_size)

    return results


def main():
    """Main function to run chunked out-of-sample testing."""
    # File paths
    train_file = "v5.0/train_with_adaptive.parquet"
    test_file = "pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet"
    features_file = "pipeline_runs/my_experiment/adaptive_only_model.json"
    params_file = "hyperparameter_tuning_optimized/best_params_sharpe.json"
    weights_file = "v5.0/weights_by_era_full.json"

    # Check if files exist
    for file_path in [train_file, test_file, features_file, params_file]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            return

    if not os.path.exists(weights_file):
        print(f"⚠️  Warning: {weights_file} not found - adaptive targets may not be created properly")
        weights_file = None

    # Run chunked out-of-sample test
    results = run_chunked_oos_test(
        train_file, test_file, features_file, params_file, weights_file,
        target_col='adaptive_target', chunk_size=100_000
    )

    # Print results
    print("\n" + "=" * 80)
    print("CHUNKED OUT-OF-SAMPLE TEST RESULTS")
    print("=" * 80)

    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return

    print("📊 PERFORMANCE METRICS:")
    print(".4f")
    print(".2f")
    print(".2f")
    print(f"  Test samples: {results['n_test_samples']:,}")
    print(f"  Test eras: {results['n_test_eras']}")

    print("
🔍 INTERPRETATION:")

    sharpe = results['sharpe_with_tc']
    if sharpe > 2.0:
        print(f"  ✅ EXCELLENT: Sharpe {sharpe:.1f} (top-tier performance)")
    elif sharpe > 1.0:
        print(f"  ✅ GOOD: Sharpe {sharpe:.1f} (solid performance)")
    elif sharpe > 0.5:
        print(f"  ⚠️  MODERATE: Sharpe {sharpe:.1f} (needs improvement)")
    else:
        print(f"  ❌ POOR: Sharpe {sharpe:.1f} (significant issues)")

    print("
💡 SUMMARY:")
    print("  ✅ Memory-efficient chunked processing implemented")
    print("  ✅ Era-aware out-of-sample testing completed")
    print("  ✅ Transaction cost analysis included")
    print(f"  📈 Model shows {'strong' if sharpe > 1.0 else 'moderate' if sharpe > 0.5 else 'weak'} generalization")


if __name__ == "__main__":
    main()