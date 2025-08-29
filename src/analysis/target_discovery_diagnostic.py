#!/usr/bin/env python3
"""
Target Discovery Diagnostic
Investigates why adaptive target discovery is failing to find meaningful combinations.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import pyarrow.parquet as pq
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def analyze_target_weights(weights_file: str) -> dict:
    """Analyze the distribution of weights across eras."""
    print("🔍 Analyzing target weights distribution...")

    with open(weights_file, 'r') as f:
        weights_data = json.load(f)

    # Analyze weight patterns
    single_target_eras = 0
    multi_target_eras = 0
    zero_weight_targets = []
    non_zero_weight_targets = []

    target_weights_sum = np.zeros(37)  # Assuming 37 targets based on previous analysis

    for era, data in weights_data.items():
        weights = np.array(data['weights'])

        # Count non-zero weights
        non_zero_count = np.sum(weights > 0.001)  # Small threshold for numerical precision

        if non_zero_count == 1:
            single_target_eras += 1
        elif non_zero_count > 1:
            multi_target_eras += 1

        # Accumulate weights for each target
        target_weights_sum += weights

        # Track which targets get weights
        for i, w in enumerate(weights):
            if w > 0.001:
                if i not in non_zero_weight_targets:
                    non_zero_weight_targets.append(i)
            else:
                if i not in zero_weight_targets:
                    zero_weight_targets.append(i)

    analysis = {
        'total_eras': len(weights_data),
        'single_target_eras': single_target_eras,
        'multi_target_eras': multi_target_eras,
        'single_target_percentage': single_target_eras / len(weights_data) * 100,
        'multi_target_percentage': multi_target_eras / len(weights_data) * 100,
        'targets_with_weights': sorted(non_zero_weight_targets),
        'targets_without_weights': sorted(zero_weight_targets),
        'target_weights_distribution': target_weights_sum.tolist(),
        'most_used_targets': np.argsort(target_weights_sum)[::-1][:5].tolist(),
        'least_used_targets': np.argsort(target_weights_sum)[:5].tolist()
    }

    print(f"  Total eras: {analysis['total_eras']}")
    print(f"  Single target eras: {analysis['single_target_eras']} ({analysis['single_target_percentage']:.1f}%)")
    print(f"  Multi-target eras: {analysis['multi_target_eras']} ({analysis['multi_target_percentage']:.1f}%)")
    print(f"  Targets with weights: {len(analysis['targets_with_weights'])}")
    print(f"  Targets without weights: {len(analysis['targets_without_weights'])}")
    print(f"  Most used targets: {analysis['most_used_targets']}")
    print(f"  Least used targets: {analysis['least_used_targets']}")

    return analysis


def analyze_target_correlations(train_file: str, sample_size: int = 10000) -> dict:
    """Analyze correlations between individual targets."""
    print(f"🔍 Analyzing target correlations (sample size: {sample_size})...")

    pf = pq.ParquetFile(train_file)
    df = pf.read().to_pandas()
    df = df.sample(n=min(sample_size, len(df)), random_state=42)

    target_cols = [col for col in df.columns if col.startswith('target')]

    # Calculate correlation matrix
    corr_matrix = df[target_cols].corr()

    # Find highly correlated target pairs
    high_corr_pairs = []
    for i in range(len(target_cols)):
        for j in range(i+1, len(target_cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.8:
                high_corr_pairs.append((target_cols[i], target_cols[j], corr))

    # Calculate average correlations
    avg_corr = corr_matrix.mean().mean()
    avg_abs_corr = corr_matrix.abs().mean().mean()

    analysis = {
        'n_targets': len(target_cols),
        'target_columns': target_cols,
        'correlation_matrix_shape': corr_matrix.shape,
        'average_correlation': avg_corr,
        'average_absolute_correlation': avg_abs_corr,
        'highly_correlated_pairs': high_corr_pairs[:10],  # Top 10
        'correlation_matrix': corr_matrix
    }

    print(f"  Number of targets: {analysis['n_targets']}")
    print(".4f")
    print(".4f")
    print(f"  Highly correlated pairs (>0.8): {len(analysis['highly_correlated_pairs'])}")

    return analysis


def analyze_adaptive_vs_individual_performance(train_file: str, weights_file: str,
                                             features_file: str, sample_size: int = 5000) -> dict:
    """Compare adaptive target performance vs individual targets."""
    print(f"🔍 Comparing adaptive vs individual target performance (sample size: {sample_size})...")

    # Load sample data
    pf = pq.ParquetFile(train_file)
    df = pf.read().to_pandas()
    df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Get features
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    if 'feature_sets' in features_data and 'medium' in features_data['feature_sets']:
        features = features_data['feature_sets']['medium']
    else:
        features = [col for col in df.columns if col.startswith('feature')]

    # Load weights
    with open(weights_file, 'r') as f:
        weights_data = json.load(f)

    # Create adaptive targets
    adaptive_targets = []
    target_cols = [col for col in df.columns if col.startswith('target')]

    for _, row in df.iterrows():
        era = str(row['era'])
        if era in weights_data:
            weights = weights_data[era]['weights']
        else:
            weights = [1.0/len(target_cols)] * len(target_cols)

        target_values = [row[col] for col in target_cols]
        adaptive_target = sum(w * t for w, t in zip(weights, target_values) if pd.notna(t))
        adaptive_targets.append(adaptive_target)

    df = df.copy()
    df['adaptive_target'] = adaptive_targets

    # Train simple models on individual targets vs adaptive target
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    individual_scores = {}
    for target_col in target_cols[:5]:  # Test first 5 targets
        # Filter valid data
        valid_mask = ~(pd.isna(df[target_col]) | pd.isna(df[features]).any(axis=1))
        X = df[valid_mask][features]
        y = df[valid_mask][target_col]

        if len(X) > 100:  # Need minimum samples
            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict(X)
            mse = mean_squared_error(y, pred)
            individual_scores[target_col] = mse

    # Test adaptive target
    valid_mask = ~(pd.isna(df['adaptive_target']) | pd.isna(df[features]).any(axis=1))
    X = df[valid_mask][features]
    y = df[valid_mask]['adaptive_target']

    adaptive_model = LinearRegression()
    adaptive_model.fit(X, y)
    adaptive_pred = adaptive_model.predict(X)
    adaptive_mse = mean_squared_error(y, adaptive_pred)

    analysis = {
        'individual_target_mse': individual_scores,
        'adaptive_target_mse': adaptive_mse,
        'best_individual_mse': min(individual_scores.values()) if individual_scores else None,
        'adaptive_improvement': None
    }

    if individual_scores:
        best_individual = min(individual_scores.values())
        analysis['adaptive_improvement'] = (best_individual - adaptive_mse) / best_individual

    print(".6f")
    print(".6f")
    if analysis['adaptive_improvement'] is not None:
        print(".2%")

    return analysis


def run_target_discovery_diagnostic():
    """Run comprehensive target discovery diagnostic."""
    print("=" * 80)
    print("TARGET DISCOVERY DIAGNOSTIC")
    print("=" * 80)

    # File paths
    train_file = "v5.0/train_with_adaptive.parquet"
    weights_file = "v5.0/weights_by_era_full.json"
    features_file = "pipeline_runs/my_experiment/adaptive_only_model.json"

    # Check files exist
    for file_path in [train_file, weights_file, features_file]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            return

    # 1. Analyze weight distribution
    print("\n" + "="*60)
    print("1. WEIGHT DISTRIBUTION ANALYSIS")
    print("="*60)
    weights_analysis = analyze_target_weights(weights_file)

    # 2. Analyze target correlations
    print("\n" + "="*60)
    print("2. TARGET CORRELATION ANALYSIS")
    print("="*60)
    correlation_analysis = analyze_target_correlations(train_file)

    # 3. Compare adaptive vs individual performance
    print("\n" + "="*60)
    print("3. ADAPTIVE VS INDIVIDUAL PERFORMANCE")
    print("="*60)
    performance_analysis = analyze_adaptive_vs_individual_performance(
        train_file, weights_file, features_file
    )

    # 4. Diagnostic summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    print("\n🔍 KEY FINDINGS:")

    if weights_analysis['single_target_percentage'] > 80:
        print("  ❌ CRITICAL: >80% of eras use single targets only")
        print("     This suggests target discovery is failing to find combinations")

    if weights_analysis['multi_target_percentage'] < 20:
        print("  ❌ CRITICAL: <20% of eras use multiple targets")
        print("     Adaptive approach is essentially just target selection")

    if len(weights_analysis['targets_without_weights']) > 0:
        print(f"  ⚠️  WARNING: {len(weights_analysis['targets_without_weights'])} targets never used")
        print("     Some targets may be irrelevant or problematic")

    if correlation_analysis['average_absolute_correlation'] > 0.7:
        print(".4f")
        print("     High correlations may limit diversification benefits")

    if performance_analysis['adaptive_improvement'] is not None:
        if performance_analysis['adaptive_improvement'] < 0.05:
            print(".2%")
            print("     Adaptive approach adds minimal value")

    print("\n💡 RECOMMENDATIONS:")

    if weights_analysis['single_target_percentage'] > 80:
        print("  • Investigate target discovery algorithm - may be overfitting to single targets")
        print("  • Check if regularization is too strong, preventing combination discovery")
        print("  • Consider ensemble methods or alternative combination approaches")

    if correlation_analysis['average_absolute_correlation'] > 0.7:
        print("  • High target correlations suggest limited diversification potential")
        print("  • Consider expanding target universe or using orthogonal targets")

    if len(weights_analysis['targets_without_weights']) > 0:
        print("  • Remove or investigate unused targets")
        print("  • Check for data quality issues in unused targets")

    print("\n🎯 CONCLUSION:")
    if (weights_analysis['single_target_percentage'] > 80 and
        performance_analysis.get('adaptive_improvement', 0) < 0.05):
        print("  The adaptive target approach is essentially equivalent to equal-weighting")
        print("  because target discovery is failing to find meaningful combinations.")
        print("  This explains why adaptive and equal-weighted models perform identically.")
    else:
        print("  Target discovery appears to be working, but further investigation needed.")

    return {
        'weights_analysis': weights_analysis,
        'correlation_analysis': correlation_analysis,
        'performance_analysis': performance_analysis
    }


if __name__ == "__main__":
    run_target_discovery_diagnostic()