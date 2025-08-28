#!/usr/bin/env python3
"""
Feature engineering for the adaptive-only model.
Creates statistical features, ratios, and interactions from existing features.
"""
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any, Optional
from scipy.stats import percentileofscore

def load_data(parquet_path: str, features: List[str], target_col: Optional[str] = None) -> pd.DataFrame:
    """Load data from parquet file."""
    pf = pq.ParquetFile(parquet_path)
    columns = features.copy()
    if target_col:
        columns.append(target_col)

    df = pf.read(columns=columns).to_pandas()
    return df

def create_statistical_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Create statistical features from existing features."""
    print("Creating statistical features...")

    # Convert to float32 for efficiency
    feature_data = df[feature_cols].astype('float32')

    new_features = {}

    # Basic statistics
    new_features['feature_mean'] = feature_data.mean(axis=1)
    new_features['feature_std'] = feature_data.std(axis=1)
    new_features['feature_skew'] = feature_data.skew(axis=1)
    new_features['feature_kurtosis'] = feature_data.kurtosis(axis=1)

    # Percentiles
    for percentile in [25, 50, 75, 90, 95]:
        new_features[f'feature_p{percentile}'] = np.percentile(feature_data.values, percentile, axis=1)

    # Range and IQR
    new_features['feature_range'] = feature_data.max(axis=1) - feature_data.min(axis=1)
    new_features['feature_iqr'] = np.percentile(feature_data.values, 75, axis=1) - np.percentile(feature_data.values, 25, axis=1)

    # Coefficient of variation
    new_features['feature_cv'] = feature_data.std(axis=1) / (feature_data.mean(axis=1) + 1e-8)

    return pd.DataFrame(new_features)

def create_ratio_features(df: pd.DataFrame, feature_cols: List[str], n_samples: int = 1000) -> pd.DataFrame:
    """Create ratio features between randomly selected feature pairs."""
    print(f"Creating ratio features from {n_samples} random pairs...")

    np.random.seed(42)
    feature_data = df[feature_cols].astype('float32')

    new_features = {}

    # Sample random pairs
    n_features = len(feature_cols)
    pairs = np.random.choice(n_features, size=(n_samples, 2), replace=True)

    for i, (idx1, idx2) in enumerate(pairs):
        col1, col2 = feature_cols[idx1], feature_cols[idx2]

        # Ratio (with small epsilon to avoid division by zero)
        ratio = feature_data[col1] / (feature_data[col2] + 1e-8)
        new_features[f'ratio_{i}'] = ratio

        # Difference
        diff = feature_data[col1] - feature_data[col2]
        new_features[f'diff_{i}'] = diff

        # Product
        prod = feature_data[col1] * feature_data[col2]
        new_features[f'prod_{i}'] = prod

    return pd.DataFrame(new_features)

def create_correlation_features(df: pd.DataFrame, feature_cols: List[str], window_size: int = 100) -> pd.DataFrame:
    """Create rolling correlation features."""
    print(f"Creating rolling correlation features with window size {window_size}...")

    feature_data = df[feature_cols].astype('float32')

    new_features = {}

    # Rolling correlations between first few features and target (if available)
    if 'adaptive_target' in df.columns:
        target = df['adaptive_target'].astype('float32')

        # Correlation with target for first 10 features
        for i, col in enumerate(feature_cols[:10]):
            corr_series = []
            for j in range(len(df)):
                start_idx = max(0, j - window_size + 1)
                window_features = feature_data.iloc[start_idx:j+1][col]
                window_target = target.iloc[start_idx:j+1]

                if len(window_features) > 1:
                    corr = np.corrcoef(window_features, window_target)[0, 1]
                    corr_series.append(corr)
                else:
                    corr_series.append(0)

            new_features[f'rolling_corr_{i}'] = corr_series

    return pd.DataFrame(new_features)

def create_rank_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Create rank-based features."""
    print("Creating rank features...")

    feature_data = df[feature_cols].astype('float32')

    new_features = {}

    # Rank within each row
    ranks = feature_data.rank(axis=1, method='average')
    for i, col in enumerate(feature_cols[:20]):  # First 20 features
        new_features[f'rank_{i}'] = ranks[col]

    # Percentile ranks
    for i, col in enumerate(feature_cols[:20]):
        percentiles = feature_data[col].rank(pct=True)
        new_features[f'percentile_{i}'] = percentiles

    return pd.DataFrame(new_features)

def create_interaction_features(df: pd.DataFrame, feature_cols: List[str], n_features: int = 50) -> pd.DataFrame:
    """Create polynomial interaction features."""
    print(f"Creating polynomial interaction features for {n_features} features...")

    # Select a subset of features for interactions
    selected_features = feature_cols[:n_features]
    feature_data = df[selected_features].astype('float32')

    new_features = {}

    # Pairwise products for first 10 features
    for i in range(min(10, len(selected_features))):
        for j in range(i+1, min(15, len(selected_features))):
            col1, col2 = selected_features[i], selected_features[j]
            new_features[f'interact_{i}_{j}'] = feature_data[col1] * feature_data[col2]

    # Squared features
    for i, col in enumerate(selected_features[:20]):
        new_features[f'squared_{i}'] = feature_data[col] ** 2

    # Cube root features (to handle outliers)
    for i, col in enumerate(selected_features[:20]):
        new_features[f'cuberoot_{i}'] = np.sign(feature_data[col]) * np.abs(feature_data[col]) ** (1/3)

    return pd.DataFrame(new_features)

def create_era_based_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Create era-based statistical features."""
    print("Creating era-based features...")

    if 'era' not in df.columns:
        print("No era column found, skipping era-based features")
        return pd.DataFrame()

    feature_data = df[feature_cols].astype('float32')

    new_features = {}

    # Group by era and create statistical features
    era_stats = df.groupby('era')[feature_cols].agg(['mean', 'std', 'min', 'max'])

    # Flatten column names
    era_stats.columns = ['_'.join(col).strip() for col in era_stats.columns]

    # Merge back to original dataframe
    era_stats = era_stats.reset_index()
    df_with_stats = df.merge(era_stats, on='era', how='left')

    # Create deviation from era mean
    for col in feature_cols[:50]:  # First 50 features
        new_features[f'era_deviation_{col}'] = feature_data[col] - df_with_stats[f'{col}_mean']

        # Z-score within era
        new_features[f'era_zscore_{col}'] = (feature_data[col] - df_with_stats[f'{col}_mean']) / (df_with_stats[f'{col}_std'] + 1e-8)

    return pd.DataFrame(new_features)

def main():
    parser = argparse.ArgumentParser(description="Feature engineering for adaptive model")
    parser.add_argument('--input-data', required=True, help='Path to input parquet file')
    parser.add_argument('--features-json', required=True, help='Path to features.json')
    parser.add_argument('--target-col', default='adaptive_target', help='Target column name')
    parser.add_argument('--output-file', required=True, help='Output parquet file')
    parser.add_argument('--max-new-features', type=int, default=500, help='Maximum number of new features to create')

    args = parser.parse_args()

    # Load features
    with open(args.features_json, 'r') as f:
        features_data = json.load(f)

    if 'feature_sets' in features_data and 'medium' in features_data['feature_sets']:
        feature_cols = features_data['feature_sets']['medium']
    elif isinstance(features_data, list):
        feature_cols = features_data
    else:
        feature_cols = [k for k in features_data.keys() if k.startswith('feature_')]

    print(f"Loaded {len(feature_cols)} base features")

    # Load data
    df = load_data(args.input_data, feature_cols, args.target_col)
    print(f"Loaded data with shape: {df.shape}")

    # Create new features
    all_new_features = []

    # 1. Statistical features
    stat_features = create_statistical_features(df, feature_cols)
    all_new_features.append(stat_features)
    print(f"Created {stat_features.shape[1]} statistical features")

    # 2. Ratio and interaction features
    ratio_features = create_ratio_features(df, feature_cols, n_samples=200)
    all_new_features.append(ratio_features)
    print(f"Created {ratio_features.shape[1]} ratio/interaction features")

    # 3. Rank features
    rank_features = create_rank_features(df, feature_cols)
    all_new_features.append(rank_features)
    print(f"Created {rank_features.shape[1]} rank features")

    # 4. Polynomial features
    poly_features = create_interaction_features(df, feature_cols, n_features=30)
    all_new_features.append(poly_features)
    print(f"Created {poly_features.shape[1]} polynomial features")

    # 5. Era-based features (if era column exists)
    if 'era' in df.columns:
        era_features = create_era_based_features(df, feature_cols)
        if not era_features.empty:
            all_new_features.append(era_features)
            print(f"Created {era_features.shape[1]} era-based features")

    # Combine all new features
    new_features_df = pd.concat(all_new_features, axis=1)

    # Limit number of features if needed
    if new_features_df.shape[1] > args.max_new_features:
        # Select most variable features
        variances = new_features_df.var()
        top_features = variances.nlargest(args.max_new_features).index
        new_features_df = new_features_df[top_features]
        print(f"Limited to top {args.max_new_features} features by variance")

    # Combine original features with new features
    result_df = pd.concat([df[feature_cols + [args.target_col]], new_features_df], axis=1)

    print(f"Final dataset shape: {result_df.shape}")
    print(f"New features created: {new_features_df.shape[1]}")

    # Save to parquet
    result_df.to_parquet(args.output_file, index=False)
    print(f"Saved enhanced dataset to {args.output_file}")

    # Save feature names
    all_feature_names = feature_cols + list(new_features_df.columns)
    with open(Path(args.output_file).with_suffix('.json'), 'w') as f:
        json.dump(all_feature_names, f, indent=2)
    print(f"Saved feature names to {Path(args.output_file).with_suffix('.json')}")

if __name__ == '__main__':
    main()
