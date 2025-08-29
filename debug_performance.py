#!/usr/bin/env python3
"""
Debug the target performance calculation issue.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import json
from pathlib import Path

def debug_target_performance():
    """Debug why target performance calculation returns all zeros."""
    print("=" * 80)
    print("DEBUGGING TARGET PERFORMANCE CALCULATION")
    print("=" * 80)

    # Load features
    with open('v5.0/features.json', 'r') as f:
        features_data = json.load(f)
    features = features_data['feature_sets']['medium']

    # Get target columns
    pf = pq.ParquetFile('v5.0/train.parquet')
    all_columns = [field.name for field in pf.schema]
    target_columns = [col for col in all_columns if col.startswith('target')]

    print(f"Features: {len(features)}")
    print(f"Target columns: {len(target_columns)}")
    print(f"Sample targets: {target_columns[:5]}")

    # Sample a few batches to check data
    print("\n" + "="*60)
    print("DATA SAMPLE ANALYSIS")
    print("="*60)

    pf_train = pq.ParquetFile('v5.0/train.parquet')
    sample_count = 0

    for batch in pf_train.iter_batches(columns=features[:5] + target_columns[:3] + ['era'], batch_size=10_000):
        df = batch.to_pandas()
        if df.empty:
            continue

        print(f"\nBatch {sample_count + 1}:")
        print(f"Shape: {df.shape}")
        print(f"Eras: {sorted(df['era'].unique())[:5]}")
        print(f"Target value ranges:")
        for target in target_columns[:3]:
            vals = df[target].dropna()
            if len(vals) > 0:
                print(f"  {target}: {vals.min():.6f} to {vals.max():.6f} (std: {vals.std():.6f})")
            else:
                print(f"  {target}: ALL NaN")

        print(f"Feature value ranges:")
        for feat in features[:3]:
            vals = df[feat].dropna()
            if len(vals) > 0:
                print(f"  {feat}: {vals.min():.6f} to {vals.max():.6f} (std: {vals.std():.6f})")
            else:
                print(f"  {feat}: ALL NaN")

        # Check correlation manually
        print(f"\nManual correlation check:")
        available_features = [f for f in features[:10] if f in df.columns]
        if available_features:
            for target in target_columns[:2]:
                if target in df.columns:
                    target_vals = df[target].dropna()
                    feature_mean = df[available_features].mean(axis=1).dropna()

                    # Align indices
                    common_idx = target_vals.index.intersection(feature_mean.index)
                    if len(common_idx) > 10:
                        t_vals = target_vals.loc[common_idx]
                        f_vals = feature_mean.loc[common_idx]

                        # Manual correlation
                        t_mean = np.mean(t_vals)
                        f_mean = np.mean(f_vals)
                        numerator = np.sum((t_vals - t_mean) * (f_vals - f_mean))
                        denominator = np.sqrt(np.sum((t_vals - t_mean)**2) * np.sum((f_vals - f_mean)**2))

                        if denominator > 0:
                            corr = numerator / denominator
                            print(f"  {target} vs feature_mean: {corr:.6f}")
                        else:
                            print(f"  {target} vs feature_mean: denominator is zero")
                    else:
                        print(f"  {target}: insufficient data ({len(common_idx)} samples)")
        else:
            print("  No features available in this batch")

        sample_count += 1
        if sample_count >= 3:
            break

    # Check if features are constant
    print("\n" + "="*60)
    print("FEATURE VARIANCE ANALYSIS")
    print("="*60)

    pf_train = pq.ParquetFile('v5.0/train.parquet')
    feature_stats = {}

    for batch in pf_train.iter_batches(columns=features[:20] + ['era'], batch_size=50_000):
        df = batch.to_pandas()
        if df.empty:
            continue

        for feat in features[:20]:
            if feat not in feature_stats:
                feature_stats[feat] = []
            vals = df[feat].dropna()
            if len(vals) > 0:
                feature_stats[feat].append(vals.std())

        if len(feature_stats) >= 20:
            break

    print("Feature standard deviations across batches:")
    for feat, stds in list(feature_stats.items())[:10]:
        avg_std = np.mean(stds) if stds else 0
        print(f"  {feat}: {avg_std:.6f}")

if __name__ == "__main__":
    debug_target_performance()