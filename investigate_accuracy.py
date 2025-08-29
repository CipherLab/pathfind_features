#!/usr/bin/env python3
"""
Investigate the 100% accuracy issue in meta-learning evaluation.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import pyarrow.parquet as pq
import lightgbm as lgb
import json
import pickle
from pathlib import Path

def investigate_meta_accuracy():
    """Investigate why meta-learning achieves 100% accuracy."""
    print("=" * 80)
    print("INVESTIGATING META-LEARNING 100% ACCURACY ISSUE")
    print("=" * 80)

    # Load the best meta config
    with open('v5.0/best_meta_config.json', 'r') as f:
        config = json.load(f)
    print(f"Best config: {config}")

    # Load cached meta-model and features
    cache_dir = Path('cache/meta_opt')
    meta_model_path = cache_dir / '6eb11b64811007c790828a0c2fd0a2a7_meta_model.pkl'
    meta_features_path = cache_dir / '31df75526df72dc919fc78384eb8aefc_meta_data.pkl'

    with open(meta_model_path, 'rb') as f:
        meta_model = pickle.load(f)

    with open(meta_features_path, 'rb') as f:
        X_meta, y_meta = pickle.load(f)

    print(f"Meta-model loaded: {X_meta.shape[0]} training samples, {X_meta.shape[1]} features")
    print(f"Target distribution: {np.bincount(y_meta.astype(int))}")

    # Load validation target performance
    val_perf_path = cache_dir / 'a12649912d129b15deb353f7a3c273f3_target_perf.pkl'
    with open(val_perf_path, 'rb') as f:
        val_target_perf = pickle.load(f)

    print(f"Validation eras: {len(val_target_perf)}")

    # Load features
    with open('v5.0/features.json', 'r') as f:
        features_data = json.load(f)
    features = features_data['feature_sets']['medium']

    # Get target columns
    pf = pq.ParquetFile('v5.0/train.parquet')
    all_columns = [field.name for field in pf.schema]
    target_columns = [col for col in all_columns if col.startswith('target')]

    print(f"Target columns: {len(target_columns)}")

    # Sample a few validation eras to check predictions
    val_file = 'v5.0/validation.parquet'
    pf_val = pq.ParquetFile(val_file)

    correct_predictions = 0
    total_predictions = 0
    sample_count = 0

    print("\n" + "="*60)
    print("SAMPLING VALIDATION PREDICTIONS")
    print("="*60)

    for batch in pf_val.iter_batches(columns=features + ['era'], batch_size=100_000):
        df = batch.to_pandas()
        if df.empty:
            continue

        # Group by era
        for era, era_df in df.groupby('era'):
            if era not in val_target_perf:
                continue

            if sample_count >= 10:  # Only check first 10 eras
                break

            # Get actual best target
            era_perfs = val_target_perf[era]
            actual_best = max(era_perfs.items(), key=lambda x: x[1])[0]
            actual_idx = target_columns.index(actual_best)

            # Create meta-features
            meta_feat = []
            for feat in features[:50]:  # Same as in optimization
                feat_vals = era_df[feat].dropna()
                if len(feat_vals) > 0:
                    meta_feat.extend([
                        float(feat_vals.mean()),
                        float(feat_vals.std()),
                        float(feat_vals.min()),
                        float(feat_vals.max()),
                        float(feat_vals.median())
                    ])
                else:
                    meta_feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # Add era info
            try:
                era_val = era_df['era'].iloc[0]
                era_str = str(era_val)
                if era_str.startswith('era'):
                    era_num = int(era_str.replace('era_', '').replace('era', ''))
                else:
                    era_num = int(float(era_str))
                meta_feat.append(float(era_num))
            except:
                meta_feat.append(0.0)

            # Make prediction
            X_pred = np.array([meta_feat])
            pred_result = meta_model.predict(X_pred)
            pred_probs = pred_result[0] if isinstance(pred_result, (list, tuple)) else pred_result
            pred_probs = np.asarray(pred_probs).flatten()
            pred_idx = int(np.argmax(pred_probs))
            pred_target = target_columns[pred_idx]

            # Check if correct
            is_correct = (pred_target == actual_best)
            correct_predictions += int(is_correct)
            total_predictions += 1

            print(f"Era {era}:")
            print(f"  Actual best: {actual_best} (idx {actual_idx})")
            print(f"  Predicted:   {pred_target} (idx {pred_idx})")
            print(f"  Correct:     {is_correct}")
            print(f"  Top 3 probs: {pred_probs[:3]}")
            print()

            sample_count += 1

        if sample_count >= 10:
            break

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Sample accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")

    # Check if training and validation eras overlap
    print("\n" + "="*60)
    print("ERA OVERLAP ANALYSIS")
    print("="*60)

    train_eras = set()
    pf_train = pq.ParquetFile('v5.0/train.parquet')
    for batch in pf_train.iter_batches(columns=['era'], batch_size=100_000):
        df = batch.to_pandas()
        train_eras.update(df['era'].unique())

    val_eras = set(val_target_perf.keys())

    overlap = train_eras & val_eras
    print(f"Training eras: {len(train_eras)}")
    print(f"Validation eras: {len(val_eras)}")
    print(f"Overlapping eras: {len(overlap)}")

    if overlap:
        print(f"Sample overlapping eras: {sorted(list(overlap))[:10]}...")
        print("⚠️  WARNING: Training and validation eras overlap!")

    # Check target performance distributions
    print("\n" + "="*60)
    print("TARGET PERFORMANCE ANALYSIS")
    print("="*60)

    all_perfs = []
    for era, perfs in val_target_perf.items():
        all_perfs.extend(perfs.values())

    print(f"Performance values range: {min(all_perfs):.4f} to {max(all_perfs):.4f}")
    print(f"Performance std: {np.std(all_perfs):.4f}")
    print(f"Very close performances: {sum(1 for p in all_perfs if abs(p) < 0.001):,}")

if __name__ == "__main__":
    investigate_meta_accuracy()