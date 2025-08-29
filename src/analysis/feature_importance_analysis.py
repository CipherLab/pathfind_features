#!/usr/bin/env python3
"""
Feature Importance Comparison Analysis
Compares feature importance between train and validation periods to investigate overfitting.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import json
import os
from pathlib import Path
from scipy.stats import spearmanr

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    HAS_PLOTTING = False
    print("⚠️  Warning: matplotlib/seaborn not available - skipping plots")

from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_data_with_adaptive_targets(file_path: str, weights_file: str, features_file: str,
                                  target_col: str = 'adaptive_target', chunk_size: int = 100_000) -> pd.DataFrame:
    """Load data and compute adaptive targets if needed."""
    print(f"Loading data from {file_path}...")

    # Load features
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    if 'feature_sets' in features_data and 'medium' in features_data['feature_sets']:
        features = features_data['feature_sets']['medium']
    else:
        features = [col for col in pq.ParquetFile(file_path).schema.names if col.startswith('feature')]

    # Load weights for adaptive target computation
    with open(weights_file, 'r') as f:
        weights_data = json.load(f)

    # Load data in chunks and compute adaptive targets
    all_data = []
    pf = pq.ParquetFile(file_path)

    needed_cols = features + ['era', 'data_type'] + [col for col in pf.schema.names if col.startswith('target')]

    for batch in pf.iter_batches(columns=needed_cols, batch_size=chunk_size):
        df = batch.to_pandas()

        if target_col not in df.columns:
            # Compute adaptive targets
            adaptive_targets = []
            target_cols = [col for col in df.columns if col.startswith('target')]

            for _, row in df.iterrows():
                era = row['era']
                if str(era) in weights_data:
                    era_weights = weights_data[str(era)]
                elif era in weights_data:
                    era_weights = weights_data[era]
                else:
                    era_weights = None
                
                # Extract weights from the era data
                if era_weights is not None:
                    # Handle nested weights format
                    if isinstance(era_weights, dict) and 'weights' in era_weights:
                        weights = era_weights['weights']
                    elif isinstance(era_weights, list):
                        weights = era_weights
                    else:
                        weights = [1.0/len(target_cols)] * len(target_cols)
                    
                    # Ensure weights are floats
                    weights = [float(w) for w in weights]
                else:
                    weights = [1.0/len(target_cols)] * len(target_cols)
                
                target_values = [row[col] for col in target_cols]
                adaptive_target = sum(w * t for w, t in zip(weights, target_values) if pd.notna(t))
                adaptive_targets.append(adaptive_target)

            df[target_col] = adaptive_targets

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def train_model_on_period(data: pd.DataFrame, features: List[str], target_col: str,
                         params: Dict) -> Tuple[lgb.Booster, np.ndarray, np.ndarray]:
    """Train model on a specific period and return model with predictions."""
    print(f"Training on {len(data)} samples...")

    # Prepare data
    valid_mask = ~(pd.isna(data[target_col]) | pd.isna(data[features]).any(axis=1))
    data_valid = data[valid_mask]

    X = data_valid[features].astype('float32')
    y = data_valid[target_col].astype('float32')

    # Train model
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=100)

    # Get predictions
    preds = model.predict(X)

    return model, preds, y.values


def analyze_feature_importance(model: lgb.Booster, features: List[str],
                              period_name: str) -> Dict[str, float]:
    """Extract and analyze feature importance."""
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    importance_dict = dict(zip(features, importance))

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    print(f"\n=== {period_name.upper()} FEATURE IMPORTANCE ===")
    print("Top 20 features:")
    for i, (feature, imp) in enumerate(sorted_features[:20]):
        print("2d")

    print(f"\nFeature importance statistics for {period_name}:")
    importance_values = list(importance_dict.values())
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(f"  Number of zero-importance features: {sum(1 for v in importance_values if v == 0)}")

    return importance_dict


def compare_feature_stability(train_importance: Dict[str, float],
                            val_importance: Dict[str, float]) -> Dict:
    """Compare feature importance stability between periods."""
    common_features = set(train_importance.keys()) & set(val_importance.keys())

    print("\n=== FEATURE STABILITY ANALYSIS ===")
    print(f"Common features: {len(common_features)}")

    # Rank correlation of importance
    train_ranks = []
    val_ranks = []
    for feature in common_features:
        train_ranks.append(train_importance[feature])
        val_ranks.append(val_importance[feature])

    rank_corr = None
    if len(train_ranks) > 10:
        rank_corr, _ = spearmanr(train_ranks, val_ranks)
        print(".4f")

    # Top features comparison
    train_top_50 = set(dict(sorted(train_importance.items(), key=lambda x: x[1], reverse=True)[:50]).keys())
    val_top_50 = set(dict(sorted(val_importance.items(), key=lambda x: x[1], reverse=True)[:50]).keys())

    overlap = len(train_top_50 & val_top_50)
    print(f"Overlap in top 50 features: {overlap}/50 ({overlap/50*100:.1f}%)")

    # Importance distribution comparison
    train_values = [train_importance[f] for f in common_features]
    val_values = [val_importance[f] for f in common_features]

    return {
        'rank_correlation': rank_corr,
        'top_50_overlap': overlap,
        'train_importance_stats': {
            'mean': np.mean(train_values),
            'std': np.std(train_values),
            'max': np.max(train_values),
            'zeros': sum(1 for v in train_values if v == 0)
        },
        'val_importance_stats': {
            'mean': np.mean(val_values),
            'std': np.std(val_values),
            'max': np.max(val_values),
            'zeros': sum(1 for v in val_values if v == 0)
        }
    }


def plot_feature_importance_comparison(train_importance: Dict[str, float],
                                     val_importance: Dict[str, float],
                                     output_dir: str = "feature_analysis_plots"):
    """Create comparison plots."""
    if not HAS_PLOTTING:
        print("⚠️  Skipping plots - matplotlib not available")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Get common features
    common_features = list(set(train_importance.keys()) & set(val_importance.keys()))

    # Sort by train importance for consistent ordering
    sorted_features = sorted(common_features, key=lambda x: train_importance[x], reverse=True)[:30]

    train_vals = [train_importance[f] for f in sorted_features]
    val_vals = [val_importance[f] for f in sorted_features]

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Bar plot comparison
    x = np.arange(len(sorted_features))
    width = 0.35

    ax1.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
    ax1.bar(x + width/2, val_vals, width, label='Validation', alpha=0.8)
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Importance (Gain)')
    ax1.set_title('Feature Importance: Train vs Validation')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.split('_')[-1] for f in sorted_features], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2.scatter(train_vals, val_vals, alpha=0.6)
    ax2.set_xlabel('Train Importance')
    ax2.set_ylabel('Validation Importance')
    ax2.set_title('Feature Importance Correlation')
    ax2.grid(True, alpha=0.3)

    # Add diagonal line
    max_val = max(max(train_vals), max(val_vals))
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect correlation')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n📊 Plots saved to {output_dir}/feature_importance_comparison.png")


def run_feature_importance_analysis():
    """Main analysis function."""
    print("=" * 80)
    print("FEATURE IMPORTANCE OVERFITTING ANALYSIS")
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

    # Load features
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    if 'feature_sets' in features_data and 'medium' in features_data['feature_sets']:
        features = features_data['feature_sets']['medium']
    else:
        # Fallback
        pf = pq.ParquetFile(train_file)
        features = [col for col in pf.schema.names if col.startswith('feature')]

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

    # Load training data (sample for speed)
    print("\n🔄 Loading training data...")
    train_data = load_data_with_adaptive_targets(train_file, weights_file, features_file, chunk_size=50_000)
    if len(train_data) > 200_000:  # Limit for speed
        train_data = train_data.sample(n=200_000, random_state=42)

    # Load validation data
    print("\n🔄 Loading validation data...")
    val_data = load_data_with_adaptive_targets(val_file, weights_file, features_file, chunk_size=50_000)

    print(f"\n📊 Data Summary:")
    print(f"  Train samples: {len(train_data):,}")
    print(f"  Validation samples: {len(val_data):,}")
    print(f"  Train eras: {train_data['era'].nunique()}")
    print(f"  Validation eras: {val_data['era'].nunique()}")

    # Train models on each period
    print("\n🏃 Training models...")

    train_model, train_preds, train_targets = train_model_on_period(
        train_data, features, 'adaptive_target', lgb_params
    )

    val_model, val_preds, val_targets = train_model_on_period(
        val_data, features, 'adaptive_target', lgb_params
    )

    # Analyze feature importance
    train_importance = analyze_feature_importance(train_model, features, "Training Period")
    val_importance = analyze_feature_importance(val_model, features, "Validation Period")

    # Compare stability
    stability_results = compare_feature_stability(train_importance, val_importance)

    # Create plots
    try:
        plot_feature_importance_comparison(train_importance, val_importance)
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

    # Performance comparison
    print("\n=== PERFORMANCE COMPARISON ===")

    # Handle different scipy versions
    train_result = spearmanr(train_targets, train_preds)
    val_result = spearmanr(val_targets, val_preds)

    # Extract correlation coefficient
    if hasattr(train_result, 'correlation'):
        train_corr = float(train_result.correlation)
        val_corr = float(val_result.correlation)
    else:
        # Older scipy versions return tuple
        train_corr = float(train_result[0])
        val_corr = float(val_result[0])

    print("Train performance:")
    print(".4f")
    print(".4f")

    print("Validation performance:")
    print(".4f")
    print(".4f")

    print("\n🔍 OVERFITTING ANALYSIS:")

    perf_drop = train_corr - val_corr
    if perf_drop > 0.2:
        print("  ❌ SEVERE OVERFITTING: Performance dropped significantly")
    elif perf_drop > 0.1:
        print("  ⚠️  MODERATE OVERFITTING: Some performance degradation")
    else:
        print("  ✅ GOOD GENERALIZATION: Minimal performance drop")

    rank_corr = stability_results.get('rank_correlation')
    if rank_corr is not None:
        if rank_corr < 0.3:
            print("  ❌ FEATURE INSTABILITY: Low correlation in feature importance")
        elif rank_corr < 0.6:
            print("  ⚠️  MODERATE INSTABILITY: Some feature importance differences")
        else:
            print("  ✅ FEATURE STABILITY: Consistent feature importance")

    top_overlap = stability_results['top_50_overlap']
    if top_overlap < 20:
        print("  ❌ TOP FEATURES MISMATCH: Very different important features")
    elif top_overlap < 35:
        print("  ⚠️  SOME MISMATCH: Moderate differences in top features")
    else:
        print("  ✅ CONSISTENT TOP FEATURES: Similar important features")

    print("\n💡 RECOMMENDATIONS:")

    if perf_drop > 0.1:
        print("  • Use more conservative regularization")
        print("  • Implement feature selection stability checks")
        print("  • Consider ensemble methods to reduce variance")

    if rank_corr is not None and rank_corr < 0.5:
        print("  • Investigate feature engineering differences between periods")
        print("  • Check for data quality issues in validation period")
        print("  • Consider time-based feature importance weighting")

    print("  • Implement walk-forward validation instead of random splits")
    print("  • Add feature importance monitoring in production")


if __name__ == "__main__":
    run_feature_importance_analysis()