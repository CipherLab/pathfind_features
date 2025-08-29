#!/usr/bin/env python3
"""
Simple test script to debug meta-learning prediction issue
"""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from typing import Dict, List
import os

def test_meta_learning():
    """Test meta-learning with minimal setup."""
    print("=" * 60)
    print("TESTING META-LEARNING PREDICTION")
    print("=" * 60)

    # Configuration
    train_file = "v5.0/train.parquet"
    features_file = "v5.0/features.json"

    # Load features
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    features = features_data['feature_sets']['medium'][:50]  # Limit for speed

    # Get target columns
    pf = pq.ParquetFile(train_file)
    all_columns = [field.name for field in pf.schema]
    target_columns = [col for col in all_columns if col.startswith('target')][:5]  # Limit targets

    print(f"Using {len(features)} features and {len(target_columns)} targets")

    # Create simple meta-features for testing
    print("\nCreating test meta-features...")

    meta_features = []
    meta_targets = []

    for batch in pf.iter_batches(columns=features + ['era'] + target_columns, batch_size=10_000):
        df = batch.to_pandas()

        for era, era_df in df.groupby('era'):
            if len(era_df) < 10:
                continue

            # Simple meta-features: just feature means
            feat_means = []
            for feat in features:
                if feat in era_df.columns:
                    feat_means.append(era_df[feat].mean())
                else:
                    feat_means.append(0.0)

            # Simple target: pick the one with highest mean
            target_means = {}
            for target in target_columns:
                if target in era_df.columns:
                    target_means[target] = era_df[target].mean()
                else:
                    target_means[target] = 0.0

            best_target = max(target_means.items(), key=lambda x: x[1])[0]
            best_target_idx = target_columns.index(best_target)

            meta_features.append(feat_means)
            meta_targets.append(best_target_idx)

            if len(meta_features) >= 100:  # Limit for testing
                break

        if len(meta_features) >= 100:
            break

    print(f"Created {len(meta_features)} meta-training samples")

    if len(meta_features) < 10:
        print("❌ Not enough training data!")
        return

    # Convert to numpy arrays
    X_meta = np.array(meta_features)
    y_meta = np.array(meta_targets)

    print(f"X_meta shape: {X_meta.shape}, y_meta shape: {y_meta.shape}")
    print(f"Target distribution: {np.bincount(y_meta)}")

    # Train simple model
    print("\nTraining meta-model...")
    train_set = lgb.Dataset(X_meta, label=y_meta)
    lgb_params = {
        'objective': 'multiclass',
        'num_class': len(target_columns),
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbosity': -1,
        'num_threads': -1,
        'seed': 42
    }

    model = lgb.train(lgb_params, train_set, num_boost_round=50)

    # Test prediction
    print("\nTesting prediction...")
    test_sample = X_meta[:5]  # Use first 5 samples
    print(f"Test sample shape: {test_sample.shape}")

    try:
        pred_result = model.predict(test_sample)
        print(f"Prediction result type: {type(pred_result)}")
        print(f"Prediction result shape: {pred_result.shape}")
        print(f"First prediction: {pred_result[0]}")

        # Test different ways to access prediction
        if isinstance(pred_result, (list, tuple)):
            print("Prediction is list/tuple")
            if len(pred_result) > 0:
                pred_probs = pred_result[0]
                print(f"Accessed pred_result[0]: {pred_probs}")
        else:
            print("Prediction is not list/tuple")
            pred_probs = pred_result
            print(f"Using pred_result directly: {pred_probs}")

        # Convert to numpy array and get argmax
        pred_probs = np.asarray(pred_probs).flatten()
        print(f"Flattened pred_probs: {pred_probs}")
        pred_target_idx = int(np.argmax(pred_probs))
        print(f"Predicted target index: {pred_target_idx}")

        print("✅ Prediction test successful!")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_meta_learning()
