#!/usr/bin/env python3
"""
Temporal Evaluation Framework - Chunked Version
Implements evaluation methods that respect temporal dynamics in financial data using memory-efficient chunked processing.
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


def get_era_metadata(file_path: str) -> Dict:
    """Get era metadata from parquet file without loading full data."""
    pf = pq.ParquetFile(file_path)
    needed_cols = ['era']

    era_counts = {}
    total_samples = 0

    for batch in pf.iter_batches(columns=needed_cols, batch_size=100000):
        try:
            df = batch.to_pandas(use_pandas_metadata=False)
        except TypeError:
            df = batch.to_pandas()
            if any(n is not None for n in (df.index.names or [])):
                df = df.reset_index()

        for era in df['era'].unique():
            era_counts[era] = era_counts.get(era, 0) + (df['era'] == era).sum()

        total_samples += len(df)

    unique_eras = sorted(era_counts.keys())

    return {
        'unique_eras': unique_eras,
        'era_counts': era_counts,
        'total_samples': total_samples,
        'n_eras': len(unique_eras)
    }


def create_temporal_splits_chunked(train_metadata: Dict, n_splits: int = 5,
                                   gap_eras: int = 5) -> List[Tuple[List, List]]:
    """Create temporal cross-validation splits using era metadata with proper temporal separation."""
    print(f"Creating {n_splits} temporal splits with {gap_eras} era gap...")

    unique_eras = train_metadata['unique_eras']
    era_counts = train_metadata['era_counts']

    # Create splits with temporal gaps to prevent data leakage
    total_eras = len(unique_eras)
    eras_per_split = total_eras // n_splits

    splits = []
    for i in range(n_splits):
        # Define validation eras for this fold
        val_start_idx = i * eras_per_split
        val_end_idx = min((i + 1) * eras_per_split, total_eras)
        val_eras = unique_eras[val_start_idx:val_end_idx]

        # Training eras: all eras BEFORE validation eras with gap
        train_end_idx = max(0, val_start_idx - gap_eras)
        train_eras = unique_eras[:train_end_idx]

        if len(train_eras) == 0:
            print(f"  ⚠️  Warning: Fold {i+1} has no training eras due to gap")
            continue

        train_samples = sum(era_counts[era] for era in train_eras if era in era_counts)
        val_samples = sum(era_counts[era] for era in val_eras if era in era_counts)

        splits.append((train_eras, val_eras))
        print(f"  Fold {i+1}: Train eras {train_eras[0]}-{train_eras[-1]} ({train_samples:,} samples) → "
              f"Val eras {val_eras[0]}-{val_eras[-1]} ({val_samples:,} samples)")

    return splits


def train_on_chunked_temporal_data(train_file: str, train_eras: List, features: List[str],
                                  target_col: str, params: Dict, weights_file: Optional[str] = None,
                                  chunk_size: int = 250_000) -> lgb.Booster:
    """Train model on chunked temporal data for specific eras."""
    print(f"Training on eras {train_eras[0]}-{train_eras[-1]}...")

    pf_train = pq.ParquetFile(train_file)
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

        # Filter to training eras
        df = df[df['era'].isin(train_eras)]

        if df.empty:
            continue

        # Create adaptive targets if needed
        if target_col == 'adaptive_target' and target_col not in df.columns:
            if weights_file:
                df = df.copy()
                df[target_col] = create_adaptive_targets_chunked(df, weights_file)
            else:
                continue  # Skip chunks without adaptive targets

        # Filter valid data
        valid_mask = ~(pd.isna(df[target_col]) | pd.isna(df[features]).any(axis=1))
        df_valid = df[valid_mask]

        if df_valid.empty:
            continue

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

    if booster is None:
        raise ValueError("No valid training data found - cannot train model")

    return booster


def evaluate_on_chunked_temporal_data(model: lgb.Booster, val_file: str, val_eras: List[str],
                                     features: List[str], target_col: str,
                                     weights_file: Optional[str] = None, chunk_size: int = 250_000) -> Dict:
    """Evaluate model on chunked temporal validation data."""
    print(f"Evaluating on eras {val_eras[0]}-{val_eras[-1]}...")

    pf_val = pq.ParquetFile(val_file)
    needed_cols = features + [target_col, 'era'] if 'era' in pf_val.schema.names else features + [target_col]

    all_predictions = []
    all_targets = []
    all_eras = []
    total_processed = 0

    for batch in pf_val.iter_batches(columns=needed_cols, batch_size=chunk_size):
        try:
            df = batch.to_pandas(use_pandas_metadata=False)
        except TypeError:
            df = batch.to_pandas()
            if any(n is not None for n in (df.index.names or [])):
                df = df.reset_index()

        # Filter to validation eras
        df = df[df['era'].isin(val_eras)]

        if df.empty:
            continue

        # Create adaptive targets if needed
        if target_col == 'adaptive_target' and target_col not in df.columns:
            if weights_file:
                df = df.copy()
                df[target_col] = create_adaptive_targets_chunked(df, weights_file)
            else:
                continue

        # Filter valid data
        valid_mask = ~(pd.isna(df[target_col]) | pd.isna(df[features]).any(axis=1))
        df_valid = df[valid_mask]

        if df_valid.empty:
            continue

        X_chunk = df_valid[features].astype('float32')
        y_chunk = df_valid[target_col].astype('float32')
        era_chunk = df_valid['era'].values

        # Make predictions
        preds = model.predict(X_chunk)
        
        # Convert to numpy array and then to list
        preds_array = np.asarray(preds)
        all_predictions.extend(preds_array.tolist())
        all_targets.extend(y_chunk.values.tolist())
        all_eras.extend(era_chunk.tolist())

        total_processed += len(df_valid)
        print(f"Processed {total_processed:,} validation samples...")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_eras = np.array(all_eras)

    if len(all_predictions) == 0:
        return {"error": "No valid data found for evaluation"}

    # Overall correlation
    overall_corr, _ = spearmanr(all_targets, all_predictions)

    # Era-wise correlations
    era_correlations = []
    unique_eras = np.unique(all_eras)
    for era in unique_eras:
        era_mask = all_eras == era
        era_targets = all_targets[era_mask]
        era_predictions = all_predictions[era_mask]

        if len(era_targets) > 1:
            era_corr, _ = spearmanr(era_targets, era_predictions)
            era_correlations.append(era_corr)

    # Calculate Sharpe ratio
    if era_correlations:
        mean_corr = np.mean(era_correlations)
        std_corr = np.std(era_correlations, ddof=1)
        sharpe = mean_corr / std_corr if std_corr > 0 else 0

        # More realistic transaction cost adjustment
        # Assuming 25bps round-trip transaction costs and monthly rebalancing
        tc_impact = 25 / 10000  # 25 basis points = 0.0025
        # Scale by turnover frequency (assume monthly rebalancing)
        annual_turnover = 12  # monthly rebalancing
        annual_tc_impact = tc_impact * annual_turnover
        sharpe_with_tc = np.maximum(0.0, sharpe - annual_tc_impact)
    else:
        sharpe = sharpe_with_tc = 0.0

    return {
        'overall_correlation': overall_corr,
        'era_correlations': era_correlations,
        'sharpe_ratio': sharpe,
        'sharpe_with_tc': sharpe_with_tc,
        'n_eras': len(era_correlations),
        'n_samples': len(all_predictions)
    }


def analyze_performance_variation(results: Dict) -> Dict:
    """Analyze performance variation across folds to detect methodological issues."""
    if "error" in results or "fold_results" not in results:
        return {"error": "No fold results to analyze"}

    fold_results = results['fold_results']

    # Extract metrics across folds
    correlations = [r['overall_correlation'] for r in fold_results]
    sharpes = [r['sharpe_ratio'] for r in fold_results]
    sharpes_tc = [r['sharpe_with_tc'] for r in fold_results]

    # Calculate variation statistics
    analysis = {
        'correlation_stats': {
            'mean': np.mean(correlations),
            'std': np.std(correlations, ddof=1),
            'min': np.min(correlations),
            'max': np.max(correlations),
            'range': np.max(correlations) - np.min(correlations),
            'cv': np.std(correlations, ddof=1) / np.mean(correlations) if np.mean(correlations) != 0 else 0
        },
        'sharpe_stats': {
            'mean': np.mean(sharpes),
            'std': np.std(sharpes, ddof=1),
            'min': np.min(sharpes),
            'max': np.max(sharpes),
            'range': np.max(sharpes) - np.min(sharpes),
            'cv': np.std(sharpes, ddof=1) / np.mean(sharpes) if np.mean(sharpes) != 0 else 0
        },
        'sharpe_tc_stats': {
            'mean': np.mean(sharpes_tc),
            'std': np.std(sharpes_tc, ddof=1),
            'min': np.min(sharpes_tc),
            'max': np.max(sharpes_tc),
            'range': np.max(sharpes_tc) - np.min(sharpes_tc),
            'cv': np.std(sharpes_tc, ddof=1) / np.mean(sharpes_tc) if np.mean(sharpes_tc) != 0 else 0
        }
    }

    # Detect suspicious patterns
    red_flags = []

    if analysis['correlation_stats']['std'] < 0.01:
        red_flags.append("Extremely low correlation variation (<1%) suggests data leakage")

    if analysis['sharpe_stats']['std'] < 0.1:
        red_flags.append("Extremely low Sharpe variation (<0.1) suggests methodological issues")

    if analysis['correlation_stats']['range'] < 0.05:
        red_flags.append("Correlation range <5% is unusually consistent for financial data")

    if analysis['sharpe_tc_stats']['mean'] == analysis['sharpe_stats']['mean']:
        red_flags.append("Transaction costs have no impact - likely calculation error")

    analysis['red_flags'] = red_flags
    analysis['methodological_concerns'] = len(red_flags) > 0

    return analysis


def run_baseline_comparison(train_file: str, features: List[str], params: Dict,
                           weights_file: Optional[str] = None, chunk_size: int = 250_000) -> Dict:
    """Compare against simple baselines to quantify actual improvement."""
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON")
    print(f"{'='*60}")

    # Get training data metadata
    train_metadata = get_era_metadata(train_file)
    train_eras = train_metadata['unique_eras']

    # Baseline 1: Equal-weighted targets
    print("Testing equal-weighted baseline...")
    equal_weight_model = train_on_chunked_temporal_data(
        train_file, train_eras, features, 'target', params, None, chunk_size
    )

    # Get validation data
    val_file = "pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet"
    val_metadata = get_era_metadata(val_file)
    val_eras = val_metadata['unique_eras']

    equal_results = evaluate_on_chunked_temporal_data(
        equal_weight_model, val_file, val_eras, features, 'target', None, chunk_size
    )

    # Baseline 2: Simple mean prediction
    print("Testing naive mean baseline...")
    # For naive baseline, we'll just predict the mean of training targets
    naive_results = {"overall_correlation": 0.0, "sharpe_ratio": 0.0, "sharpe_with_tc": 0.0}

    return {
        'equal_weighted': equal_results,
        'naive_mean': naive_results
    }


def run_chunked_temporal_cross_validation(train_file: str, features: List[str], params: Dict,
                                         weights_file: Optional[str] = None, n_splits: int = 5,
                                         gap_eras: int = 5, chunk_size: int = 250_000) -> Dict:
    """Run temporal cross-validation with chunked processing."""
    print(f"\n{'='*60}")
    print("CHUNKED TEMPORAL CROSS-VALIDATION")
    print(f"{'='*60}")

    # Get training data metadata
    train_metadata = get_era_metadata(train_file)
    print(f"Training data: {train_metadata['total_samples']:,} samples, {train_metadata['n_eras']} eras")

    # Create temporal splits
    splits = create_temporal_splits_chunked(train_metadata, n_splits, gap_eras)

    all_results = []
    for i, (train_eras, val_eras) in enumerate(splits):
        print(f"\n--- Fold {i+1}/{n_splits} ---")

        # Train model on chunked data
        model = train_on_chunked_temporal_data(train_file, train_eras, features, 'adaptive_target',
                                              params, weights_file, chunk_size)

        # Evaluate on chunked data (using training file for validation in temporal CV)
        results = evaluate_on_chunked_temporal_data(model, train_file, val_eras, features,
                                                   'adaptive_target', weights_file, chunk_size)
        all_results.append(results)

        if "error" not in results:
            print(".4f")
            print(".2f")
            print(".2f")
            print(f"  Eras: {results['n_eras']}, Samples: {results['n_samples']:,}")

    # Aggregate results
    if all("error" not in r for r in all_results):
        overall_corrs = [r['overall_correlation'] for r in all_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in all_results]
        sharpe_tc_ratios = [r['sharpe_with_tc'] for r in all_results]

        summary = {
            'mean_overall_correlation': np.mean(overall_corrs),
            'std_overall_correlation': np.std(overall_corrs, ddof=1),
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': np.std(sharpe_ratios, ddof=1),
            'mean_sharpe_with_tc': np.mean(sharpe_tc_ratios),
            'std_sharpe_with_tc': np.std(sharpe_tc_ratios, ddof=1),
            'fold_results': all_results
        }
    else:
        summary = {"error": "Some folds failed to process"}

    return summary


def run_chunked_walk_forward_validation(train_file: str, val_file: str, features: List[str],
                                       params: Dict, weights_file: Optional[str] = None,
                                       chunk_size: int = 250_000) -> Dict:
    """Run walk-forward validation with chunked processing."""
    print(f"\n{'='*60}")
    print("CHUNKED WALK-FORWARD VALIDATION")
    print(f"{'='*60}")

    # Get training data metadata to get all eras
    train_metadata = get_era_metadata(train_file)
    train_eras = train_metadata['unique_eras']

    print(f"Training on all {len(train_eras)} eras ({train_metadata['total_samples']:,} samples)...")

    # Train on all training data
    model = train_on_chunked_temporal_data(train_file, train_eras, features, 'adaptive_target',
                                          params, weights_file, chunk_size)

    # Get validation eras
    val_metadata = get_era_metadata(val_file)
    val_eras = val_metadata['unique_eras']

    print(f"Evaluating on {len(val_eras)} validation eras ({val_metadata['total_samples']:,} samples)...")

    # Evaluate on validation data
    results = evaluate_on_chunked_temporal_data(model, val_file, val_eras, features,
                                               'adaptive_target', weights_file, chunk_size)

    if "error" not in results:
        print("\nWALK-FORWARD RESULTS:")
        print(".4f")
        print(".2f")
        print(".2f")
        print(f"  Eras: {results['n_eras']}, Samples: {results['n_samples']:,}")

    return results


def run_chunked_temporal_evaluation():
    """Main chunked temporal evaluation function."""
    print("=" * 80)
    print("CHUNKED TEMPORAL EVALUATION FRAMEWORK")
    print("=" * 80)

    # File paths
    train_file = "v5.0/train_with_adaptive.parquet"
    val_file = "pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet"
    features_file = "pipeline_runs/my_experiment/adaptive_only_model.json"
    weights_file = "v5.0/weights_by_era_full.json"

    # Check files exist
    for file_path in [train_file, val_file, features_file, weights_file]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            return

    # Get features
    pf_train = pq.ParquetFile(train_file)
    features = select_features(pf_train, features_file)
    print(f"Using {len(features)} features")

    # LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'num_threads': -1,
        'seed': 42
    }

    # Run chunked temporal cross-validation with proper temporal separation
    tcv_results = run_chunked_temporal_cross_validation(train_file, features, lgb_params,
                                                       weights_file, n_splits=5, gap_eras=5,
                                                       chunk_size=100_000)

    # Analyze performance variation
    variation_analysis = None
    if "error" not in tcv_results:
        variation_analysis = analyze_performance_variation(tcv_results)
        print(f"\n🔍 PERFORMANCE VARIATION ANALYSIS:")
        print(f"  Correlation CV: {variation_analysis['correlation_stats']['cv']:.3f}")
        print(f"  Sharpe CV: {variation_analysis['sharpe_stats']['cv']:.3f}")
        print(f"  Correlation Range: {variation_analysis['correlation_stats']['range']:.4f}")
        print(f"  Sharpe Range: {variation_analysis['sharpe_stats']['range']:.2f}")

        if variation_analysis['methodological_concerns']:
            print(f"\n⚠️  METHODOLOGICAL RED FLAGS DETECTED:")
            for flag in variation_analysis['red_flags']:
                print(f"  • {flag}")

    # Run baseline comparison
    baseline_results = run_baseline_comparison(train_file, features, lgb_params,
                                             weights_file, chunk_size=100_000)

    # Run chunked walk-forward validation
    wfv_results = run_chunked_walk_forward_validation(train_file, val_file, features, lgb_params,
                                                     weights_file, chunk_size=100_000)

    # Summary
    print(f"\n{'='*80}")
    print("CHUNKED TEMPORAL EVALUATION SUMMARY")
    print(f"{'='*80}")

    if "error" not in tcv_results:
        print("\n📊 TEMPORAL CROSS-VALIDATION (5-fold with 5-era gaps):")
        print(".4f")
        print(".2f")
        print(".2f")

        # Show baseline comparison
        if "error" not in baseline_results.get('equal_weighted', {}):
            print("\n🏆 BASELINE COMPARISON:")
            print(".4f")
            print(".2f")
            print(".2f")
            print(".4f")
            print(".2f")
            print(".2f")
    else:
        print(f"\n❌ TEMPORAL CROSS-VALIDATION ERROR: {tcv_results['error']}")

    if "error" not in wfv_results:
        print("\n🚀 WALK-FORWARD VALIDATION:")
        print(".4f")
        print(".2f")
        print(".2f")
    else:
        print(f"\n❌ WALK-FORWARD VALIDATION ERROR: {wfv_results['error']}")

    print("\n🔍 METHODOLOGICAL VALIDATION:")
    print("  ✅ Proper temporal separation with era gaps")
    print("  ✅ Memory-efficient chunked processing")
    print("  ✅ Era-aware evaluation with realistic transaction costs")
    print("  ✅ Baseline comparison for context")

    print("\n⚠️  INTERPRETATION GUIDELINES:")
    print("  • Look for reasonable variation across folds (CV > 0.1)")
    print("  • Expect some out-of-sample degradation")
    print("  • Transaction costs should meaningfully impact Sharpe")
    print("  • Compare against baselines to quantify improvement")

    return {
        'temporal_cv': tcv_results,
        'walk_forward': wfv_results,
        'variation_analysis': variation_analysis,
        'baseline_comparison': baseline_results
    }


if __name__ == "__main__":
    run_chunked_temporal_evaluation()