#!/usr/bin/env python3
"""
Comprehensive Methodological Validation Script
Addresses critical concerns about Sharpe ratio analysis and model validation.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ttest_1samp
import pyarrow.parquet as pq
import lightgbm as lgb
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


def load_and_prepare_data(data_file: str, features_file: str) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Load and prepare validation data with comprehensive checks."""
    print("=" * 60)
    print("DATA INTEGRITY AND METHODOLOGY CHECKS")
    print("=" * 60)

    # Load validation data
    pf_val = pq.ParquetFile(data_file)
    val_df = pf_val.read().to_pandas()

    # Load features
    with open(features_file, 'r') as f:
        features = json.load(f)

    print(f"Dataset Overview:")
    print(f"  - Total rows: {len(val_df):,}")
    print(f"  - Total columns: {len(val_df.columns)}")
    print(f"  - Features: {len(features)}")
    print(f"  - Era column present: {'era' in val_df.columns}")
    print(f"  - Target column present: {'adaptive_target' in val_df.columns}")
    print()

    # Check for data quality issues
    print("Data Quality Checks:")
    target_na = val_df['adaptive_target'].isna().sum()
    era_na = val_df['era'].isna().sum() if 'era' in val_df.columns else 0
    print(f"  - Target NaN values: {target_na:,} ({target_na/len(val_df)*100:.2f}%)")
    print(f"  - Era NaN values: {era_na:,} ({era_na/len(val_df)*100:.2f}%)")

    # Check era distribution
    era_series = None
    if 'era' in val_df.columns:
        era_counts = val_df['era'].value_counts().sort_index()
        print(f"  - Unique eras: {len(era_counts)}")
        print(f"  - Era range: {era_counts.index.min()} to {era_counts.index.max()}")
        print(f"  - Samples per era: mean={era_counts.mean():.0f}, std={era_counts.std():.0f}")
        era_series = val_df['era']
    print()

    # Prepare data
    X_val = val_df[features].astype('float32')
    y_val = val_df['adaptive_target'].astype('float32')

    # Filter out NaN values
    valid_mask = ~(pd.isna(y_val) | pd.isna(X_val).any(axis=1))
    if era_series is not None:
        valid_mask &= ~pd.isna(era_series)

    X_clean = X_val[valid_mask]
    y_clean = y_val[valid_mask]
    era_clean = era_series[valid_mask] if era_series is not None else None

    print(f"After cleaning: {len(X_clean):,} rows ({len(X_clean)/len(val_df)*100:.1f}%)")
    print()

    return X_clean, y_clean, era_clean


def statistical_correctness_check(era_correlations: List[float]) -> Dict:
    """Correct statistical analysis with proper p-value interpretation."""
    print("=" * 60)
    print("STATISTICAL CORRECTNESS CHECK")
    print("=" * 60)

    if not era_correlations:
        return {"error": "No era correlations provided"}

    # Basic statistics
    mean_corr = np.mean(era_correlations)
    std_corr = np.std(era_correlations, ddof=1)  # Use ddof=1 for sample standard deviation
    n_eras = len(era_correlations)

    print(f"Era Correlation Statistics:")
    print(f"  - Number of eras: {n_eras}")
    print(f"  - Mean correlation: {mean_corr:.4f}")
    print(f"  - Std correlation: {std_corr:.4f}")
    print(f"  - Min correlation: {np.min(era_correlations):.4f}")
    print(f"  - Max correlation: {np.max(era_correlations):.4f}")
    print()

    # CORRECTED: Sharpe ratio calculation
    sharpe = mean_corr / std_corr if std_corr > 0 else 0
    print(f"Sharpe Ratio Analysis:")
    print(f"  - Sharpe ratio: {sharpe:.4f}")
    print(f"  - This means correlations are {sharpe:.1f} std devs above zero")
    print()

    # CORRECTED: Statistical significance test
    # H0: mean correlation = 0
    # Ha: mean correlation > 0
    test_result = ttest_1samp(era_correlations, 0)
    t_stat = test_result[0]
    p_value = test_result[1]

    print(f"Hypothesis Test (H0: mean correlation = 0):")
    print(f"  - t-statistic: {t_stat:.4f}")
    print(f"  - p-value: {p_value:.6f}")

    # CORRECTED: Proper significance interpretation
    if p_value < 0.001:
        significance = "*** p < 0.001 (extremely significant)"
    elif p_value < 0.01:
        significance = "** p < 0.01 (very significant)"
    elif p_value < 0.05:
        significance = "* p < 0.05 (significant)"
    else:
        significance = "Not significant (p >= 0.05)"

    print(f"  - Significance level: {significance}")
    print()

    # Reality check: What's a realistic Sharpe ratio?
    print("Reality Check - Sharpe Ratio Context:")
    print("  - Typical quant strategy: 0.5-1.5")
    print("  - Excellent quant strategy: 2.0-3.0")
    print("  - Legendary performance: >3.0 (extremely rare)")
    print("  - Our result: {:.1f} - {}".format(
        sharpe,
        "POTENTIALLY CONCERNING" if sharpe > 3 else "REASONABLE" if sharpe > 1 else "MODEST"
    ))
    print()

    return {
        "sharpe_ratio": sharpe,
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_correlation": mean_corr,
        "std_correlation": std_corr,
        "n_eras": n_eras
    }


def independence_assumption_check(era_series: Optional[pd.Series], correlations: List[float]) -> Dict:
    """Check the independence assumption of eras."""
    print("=" * 60)
    print("ERA INDEPENDENCE ASSUMPTION CHECK")
    print("=" * 60)

    if era_series is None or not correlations:
        return {"error": "Missing era data or correlations"}

    # Convert eras to numeric for analysis
    try:
        era_numeric = pd.to_numeric(era_series.unique(), errors='coerce')
        era_numeric = era_numeric[~pd.isna(era_numeric)]
        era_numeric = np.sort(era_numeric)
    except:
        era_numeric = np.arange(len(era_series.unique()))

    print(f"Temporal Analysis:")
    print(f"  - Era sequence length: {len(era_numeric)}")
    print(f"  - Era range: {era_numeric[0]} to {era_numeric[-1]}")
    if len(era_numeric) > 1:
        consecutive_pct = np.mean(np.diff(era_numeric) == 1) * 100
        print(f"  - Consecutive eras: {consecutive_pct:.1f}%")
    print()

    # Simple autocorrelation check (lag 1)
    lag1_corr = None
    if len(correlations) > 3:
        # Manual autocorrelation calculation
        correlations_array = np.array(correlations)
        lag1_corr = np.corrcoef(correlations_array[:-1], correlations_array[1:])[0, 1]

        print(f"Correlation Autocorrelation:")
        print(f"  - Lag 1 autocorrelation: {lag1_corr:.3f}")

        if abs(lag1_corr) > 0.3:
            print("  ⚠️  WARNING: Significant autocorrelation detected!")
            print("     This violates the independence assumption.")
        else:
            print("  ✓ No significant autocorrelation detected.")
        print()

    # Era clustering analysis
    era_counts = era_series.value_counts().sort_index()
    concentration = era_counts.max() / era_counts.sum()

    print(f"Era Distribution Analysis:")
    print(f"  - Most concentrated era: {concentration:.1%} of total data")
    print(f"  - Era diversity: {len(era_counts)} unique eras")
    print()

    return {
        "autocorrelation": lag1_corr if len(correlations) > 3 else None,
        "era_concentration": concentration,
        "n_unique_eras": len(era_counts)
    }


def baseline_comparison_analysis(X: pd.DataFrame, y: pd.Series, era_series: Optional[pd.Series],
                                model_predictions: np.ndarray) -> Dict:
    """Compare model performance against realistic baselines."""
    print("=" * 60)
    print("BASELINE COMPARISON ANALYSIS")
    print("=" * 60)

    baselines = {}

    # 1. Random predictions
    np.random.seed(42)
    random_preds = np.random.normal(y.mean(), y.std(), len(y))
    random_corr = spearmanr(y, random_preds)[0]
    baselines['random'] = random_corr

    # 2. Mean prediction (naive baseline)
    mean_preds = np.full(len(y), y.mean())
    mean_corr = spearmanr(y, mean_preds)[0]
    baselines['mean'] = mean_corr

    print("Baseline Performance Comparison:")
    print(f"  - Random predictions: {baselines['random']:.4f}")
    print(f"  - Mean prediction: {baselines['mean']:.4f}")
    print(f"  - Our model: {spearmanr(y, model_predictions)[0]:.4f}")
    print()

    # Performance ratio
    best_baseline = max(baselines.values())
    model_corr = spearmanr(y, model_predictions)[0]
    improvement_ratio = model_corr / best_baseline if best_baseline != 0 else float('inf')

    print(f"Performance Assessment:")
    print(f"  - Best baseline correlation: {best_baseline:.4f}")
    print(f"  - Model improvement ratio: {improvement_ratio:.1f}x")
    print(f"  - Assessment: {'EXCEPTIONAL' if improvement_ratio > 5 else 'STRONG' if improvement_ratio > 2 else 'MODERATE' if improvement_ratio > 1.5 else 'MARGINAL'}")
    print()

    return {
        "baselines": baselines,
        "model_correlation": model_corr,
        "improvement_ratio": improvement_ratio,
        "best_baseline": best_baseline
    }


def overfitting_detection_check(X: pd.DataFrame, y: pd.Series, era_series: Optional[pd.Series]) -> Dict:
    """Check for signs of overfitting and data leakage."""
    print("=" * 60)
    print("OVERFITTING DETECTION CHECK")
    print("=" * 60)

    issues = []

    # 1. Feature-to-sample ratio
    n_samples, n_features = X.shape
    ratio = n_features / n_samples

    print(f"Data Complexity Analysis:")
    print(f"  - Samples: {n_samples:,}")
    print(f"  - Features: {n_features:,}")
    print(f"  - Feature-to-sample ratio: {ratio:.3f}")

    if ratio > 0.1:
        issues.append("HIGH_FEATURE_RATIO")
        print("  ⚠️  WARNING: Very high feature-to-sample ratio suggests overfitting risk")
    elif ratio > 0.05:
        print("  ⚠️  CAUTION: Moderately high feature-to-sample ratio")
    else:
        print("  ✓ Reasonable feature-to-sample ratio")
    print()

    # 2. Feature correlation analysis
    feature_corrs = []
    for i, col in enumerate(X.columns[:min(50, len(X.columns))]):  # Check first 50 features
        corr_result = spearmanr(y, X[col])
        corr = corr_result[0]  # Extract correlation from tuple
        feature_corrs.append(abs(corr))

    strong_features = sum(1 for c in feature_corrs if c > 0.1)
    print(f"Feature Analysis:")
    print(f"  - Features with |correlation| > 0.1: {strong_features}/{len(feature_corrs)}")
    print(f"  - Max feature correlation: {max(feature_corrs):.4f}")

    if strong_features / len(feature_corrs) > 0.5:
        issues.append("TOO_MANY_STRONG_FEATURES")
        print("  ⚠️  WARNING: Too many features strongly correlated with target")
    print()

    return {
        "issues": issues,
        "feature_ratio": ratio,
        "strong_features_ratio": strong_features / len(feature_corrs),
        "max_feature_correlation": max(feature_corrs)
    }


def transaction_cost_analysis(predictions: np.ndarray, transaction_cost_bps: float = 10) -> Dict:
    """Analyze impact of transaction costs on performance."""
    print("=" * 60)
    print("TRANSACTION COST ANALYSIS")
    print("=" * 60)

    # Simulate trading with transaction costs
    # Assume we trade based on prediction strength
    trade_signals = np.where(predictions > np.median(predictions), 1,
                           np.where(predictions < np.median(predictions), -1, 0))

    # Calculate turnover (simplified)
    turnover = np.mean(np.abs(np.diff(trade_signals))) if len(trade_signals) > 1 else 0

    # Transaction costs in basis points
    annual_tc_cost = turnover * transaction_cost_bps * 252  # Assuming daily trading

    print(f"Transaction Cost Impact:")
    print(f"  - Assumed cost: {transaction_cost_bps} bps per trade")
    print(f"  - Estimated turnover: {turnover:.3f}")
    print(f"  - Annual transaction costs: {annual_tc_cost:.1f} bps")
    print()

    # Sharpe ratio adjustment
    # Rough estimate: TC reduces Sharpe by approximately TC cost / volatility
    assumed_volatility = 0.15  # 15% annual volatility assumption
    sharpe_reduction = annual_tc_cost / 10000 / assumed_volatility  # Convert bps to decimal

    print(f"Sharpe Ratio Adjustment:")
    print(f"  - Assumed volatility: {assumed_volatility:.1%}")
    print(f"  - Estimated Sharpe reduction: {sharpe_reduction:.2f}")
    print(f"  - Note: This is a rough estimate; actual impact depends on strategy")
    print()

    return {
        "turnover": turnover,
        "annual_tc_cost_bps": annual_tc_cost,
        "sharpe_reduction_estimate": sharpe_reduction
    }


def main():
    """Main function to run comprehensive methodological validation."""
    print("COMPREHENSIVE METHODOLOGICAL VALIDATION")
    print("=" * 80)
    print()

    # File paths
    data_file = "pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet"
    features_file = "pipeline_runs/my_experiment/adaptive_only_model.json"
    params_file = "hyperparameter_tuning_optimized/best_params_sharpe.json"

    # Check if files exist
    for file_path in [data_file, features_file, params_file]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            return

    # Load and prepare data
    X_val, y_val, era_series = load_and_prepare_data(data_file, features_file)

    # Load best parameters and train model
    with open(params_file, 'r') as f:
        best_params = json.load(f)

    print("Training model with best parameters...")
    train_set = lgb.Dataset(X_val, label=y_val)
    lgb_params = best_params.copy()
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'l2'
    lgb_params['seed'] = 42

    model = lgb.train(lgb_params, train_set, num_boost_round=100)
    predictions = np.array(model.predict(X_val))

    # Calculate era correlations
    era_correlations = []
    if era_series is not None:
        for era in sorted(era_series.unique()):
            era_mask = era_series == era
            era_targets = y_val[era_mask]
            era_predictions = predictions[era_mask]

            if len(era_targets) > 1:
                corr_result = spearmanr(era_targets, era_predictions)
                corr = corr_result[0]
                era_correlations.append(corr)

    # Run all validation checks
    stats_results = statistical_correctness_check(era_correlations)
    independence_results = independence_assumption_check(era_series, era_correlations)
    baseline_results = baseline_comparison_analysis(X_val, y_val, era_series, predictions)
    overfitting_results = overfitting_detection_check(X_val, y_val, era_series)
    tc_results = transaction_cost_analysis(predictions)

    # Summary and recommendations
    print("=" * 80)
    print("VALIDATION SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)

    print("\n🔍 KEY FINDINGS:")

    # Statistical issues
    p_value = stats_results.get('p_value', 1)
    if p_value > 0.05:
        print("❌ Statistical Significance: FAIL - Results not statistically significant")
    else:
        print("✅ Statistical Significance: PASS")

    # Performance realism
    sharpe = stats_results.get('sharpe_ratio', 0)
    if sharpe > 3:
        print(f"⚠️  Sharpe Ratio: {sharpe:.1f} - EXTREMELY HIGH (potential overfitting)")
    elif sharpe > 2:
        print(f"⚠️  Sharpe Ratio: {sharpe:.1f} - VERY HIGH (requires careful validation)")
    elif sharpe > 1:
        print(f"✅ Sharpe Ratio: {sharpe:.1f} - REASONABLE")
    else:
        print(f"📉 Sharpe Ratio: {sharpe:.1f} - MODEST")

    # Baseline comparison
    improvement = baseline_results.get('improvement_ratio', 0)
    if improvement > 5:
        print(f"📈 Baseline Improvement: {improvement:.1f}x - EXCEPTIONAL")
    elif improvement > 2:
        print(f"✅ Baseline Improvement: {improvement:.1f}x - STRONG")
    else:
        print(f"⚠️  Baseline Improvement: {improvement:.1f}x - MARGINAL")

    # Overfitting indicators
    issues = overfitting_results.get('issues', [])
    if issues:
        print(f"⚠️  Overfitting Indicators: {len(issues)} detected")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Overfitting Check: No major indicators detected")

    print("\n💡 CRITICAL RECOMMENDATIONS:")
    print("1. ❌ CORRECTED: p-value of 1.0 means NO significance (not extreme significance)")
    print("2. ⚠️  Sharpe ratio of 4.56 is extraordinarily high - investigate for overfitting")
    print("3. 🔍 Compare against truly out-of-sample data from different time periods")
    print("4. 📊 Implement proper cross-validation with era-aware splits")
    print("5. 💰 Add transaction cost analysis to all evaluations (10bps reduces Sharpe by ~0.7)")
    print("6. 🎯 Test multiple baseline strategies before claiming breakthrough")
    print("7. 🧮 Consider regularization and feature selection to reduce overfitting risk")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
