#!/usr/bin/env python3
"""
Target Preference Meta-Learning Framework
Trains a model to predict which target will perform best based on features and era context.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import pyarrow.parquet as pq
import pyarrow as pa
import lightgbm as lgb
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TargetPreferencePredictor:
    """Meta-learning approach to predict which target performs best for given context."""

    def __init__(self, target_columns: List[str], random_state: int = 42):
        self.target_columns = target_columns
        self.n_targets = len(target_columns)
        self.random_state = random_state
        self.target_models = {}  # Individual models for each target
        self.meta_model = None  # Model that predicts which target is best
        self.target_performance = {}  # Historical performance data

    def train_individual_target_models(self, train_file: str, features: List[str],
                                     chunk_size: int = 100_000) -> Dict:
        """Train individual LightGBM models for each target."""
        print(f"Training {self.n_targets} individual target models...")

        pf = pq.ParquetFile(train_file)
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
            'seed': self.random_state
        }

        for i, target_col in enumerate(self.target_columns):
            print(f"  Training model for {target_col} ({i+1}/{self.n_targets})...")

            model = None
            total_samples = 0

            for batch in pf.iter_batches(columns=features + [target_col, 'era'], batch_size=chunk_size):
                df = batch.to_pandas()

                # Filter valid data
                valid_mask = ~(pd.isna(df[target_col]) | pd.isna(df[features]).any(axis=1))
                df_valid = df[valid_mask]

                if df_valid.empty:
                    continue

                X = df_valid[features].astype('float32')
                y = df_valid[target_col].astype('float32')

                train_set = lgb.Dataset(X, label=y, free_raw_data=True)

                if model is None:
                    model = lgb.train(lgb_params, train_set, num_boost_round=100)
                else:
                    model = lgb.train(lgb_params, train_set, num_boost_round=50,
                                    init_model=model, keep_training_booster=True)

                total_samples += len(df_valid)

            self.target_models[target_col] = model
            print(f"    Trained on {total_samples:,} samples")

        return self.target_models

    def evaluate_target_performance(self, data_file: str, features: List[str],
                                  chunk_size: int = 100_000) -> Dict:
        """Evaluate performance of each target model across eras."""
        print("Evaluating target model performance across eras...")

        pf = pq.ParquetFile(data_file)
        era_performance = {}

        for batch in pf.iter_batches(columns=features + self.target_columns + ['era'],
                                   batch_size=chunk_size):
            df = batch.to_pandas()

            # Group by era
            for era, era_df in df.groupby('era'):
                if era not in era_performance:
                    era_performance[era] = {}

                # Filter valid data for this era
                valid_mask = ~(pd.isna(era_df[self.target_columns]).any(axis=1) |
                             pd.isna(era_df[features]).any(axis=1))
                era_df_valid = era_df[valid_mask]

                if era_df_valid.empty:
                    continue

                X = era_df_valid[features].astype('float32')

                # Evaluate each target model
                for target_col in self.target_columns:
                    if target_col not in self.target_models:
                        continue

                    model = self.target_models[target_col]
                    try:
                        preds = model.predict(X)
                        true_vals = era_df_valid[target_col].values

                        if len(true_vals) > 1 and np.std(preds) > 0 and np.std(true_vals) > 0:
                            try:
                                corr_result = spearmanr(preds, true_vals)
                                # Handle different scipy versions
                                if hasattr(corr_result, 'statistic'):  # type: ignore
                                    corr = corr_result.statistic  # type: ignore
                                elif isinstance(corr_result, tuple):
                                    corr = corr_result[0]
                                else:
                                    corr = corr_result
                                
                                if isinstance(corr, (int, float)) and np.isfinite(corr):
                                    if target_col not in era_performance[era]:
                                        era_performance[era][target_col] = []
                                    era_performance[era][target_col].append(float(corr))
                            except Exception:
                                continue  # Skip this evaluation if correlation fails
                    except Exception as e:
                        print(f"Warning: Failed to evaluate {target_col} for era {era}: {e}")

        # Aggregate performance by era
        self.target_performance = {}
        for era, target_perfs in era_performance.items():
            self.target_performance[era] = {}
            for target_col, corrs in target_perfs.items():
                if corrs:
                    self.target_performance[era][target_col] = np.mean(corrs)

        print(f"Evaluated performance across {len(self.target_performance)} eras")
        return self.target_performance

    def create_meta_training_data(self, data_file: str, features: List[str],
                                chunk_size: int = 100_000):
        """Create training data for meta-model that predicts best target."""
        print("Creating meta-model training data...")

        pf = pq.ParquetFile(data_file)
        meta_features = []
        meta_targets = []

        for batch in pf.iter_batches(columns=features + ['era'], batch_size=chunk_size):
            df = batch.to_pandas()

            # Group by era
            for era, era_df in df.groupby('era'):
                if era not in self.target_performance:
                    continue

                # Get best performing target for this era
                era_perfs = self.target_performance[era]
                if not era_perfs:
                    continue

                best_target = max(era_perfs.items(), key=lambda x: x[1])[0]
                best_target_idx = self.target_columns.index(best_target)

                # Use era-level feature statistics as meta-features
                era_features = era_df[features]
                meta_feat = []

                # Basic statistics for each feature in this era
                for feat in features:
                    feat_vals = era_features[feat].dropna()
                    if len(feat_vals) > 0:
                        meta_feat.extend([
                            feat_vals.mean(),
                            feat_vals.std(),
                            feat_vals.min(),
                            feat_vals.max(),
                            feat_vals.median(),
                            feat_vals.skew() if len(feat_vals) > 2 else 0,
                            feat_vals.kurtosis() if len(feat_vals) > 2 else 0
                        ])
                    else:
                        meta_feat.extend([0, 0, 0, 0, 0, 0, 0])

                # Add era information
                try:
                    era_num = int(str(era).replace('era_', '').replace('era', ''))
                    meta_feat.append(era_num)
                except:
                    meta_feat.append(0)

                meta_features.append(meta_feat)
                meta_targets.append(best_target_idx)

        X_meta = np.array(meta_features)
        y_meta = np.array(meta_targets)

        print(f"Created meta-training data: {X_meta.shape[0]} samples, {X_meta.shape[1]} features")
        return X_meta, y_meta

    def train_meta_model(self, X_meta: np.ndarray, y_meta: np.ndarray) -> lgb.Booster:
        """Train the meta-model to predict best target."""
        print("Training meta-model to predict best target...")

        lgb_params = {
            'objective': 'multiclass',
            'num_class': self.n_targets,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'num_threads': -1,
            'seed': self.random_state
        }

        train_set = lgb.Dataset(X_meta, label=y_meta)
        self.meta_model = lgb.train(lgb_params, train_set, num_boost_round=200)

        print("Meta-model trained successfully")
        return self.meta_model

    def predict_best_target(self, era_features: pd.DataFrame) -> str:
        """Predict which target will perform best for given era features."""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained yet")

        # Create meta-features from era data
        meta_feat = []
        for feat in era_features.columns:
            feat_vals = era_features[feat].dropna()
            if len(feat_vals) > 0:
                meta_feat.extend([
                    feat_vals.mean(),
                    feat_vals.std(),
                    feat_vals.min(),
                    feat_vals.max(),
                    feat_vals.median(),
                    feat_vals.skew() if len(feat_vals) > 2 else 0,
                    feat_vals.kurtosis() if len(feat_vals) > 2 else 0
                ])
            else:
                meta_feat.extend([0, 0, 0, 0, 0, 0, 0])

        # Add era information (use a default if not available)
        meta_feat.append(0)  # Default era number

        X_meta = np.array([meta_feat])

        # Predict best target
        pred_result = self.meta_model.predict(X_meta)
        # Handle different possible return types from LightGBM
        if isinstance(pred_result, (list, tuple)) and len(pred_result) > 0:
            pred_probs = pred_result[0]
        elif hasattr(pred_result, 'shape') and len(pred_result.shape) > 1:  # type: ignore
            pred_probs = pred_result[0]  # type: ignore
        else:
            pred_probs = pred_result
        
        best_target_idx = int(np.argmax(pred_probs))  # type: ignore
        best_target = self.target_columns[best_target_idx]

        return best_target

    def create_adaptive_targets_meta(self, data_file: str, features: List[str],
                                   output_file: str, chunk_size: int = 100_000) -> None:
        """Create adaptive targets using meta-model predictions."""
        print("Creating adaptive targets using meta-model...")

        pf = pq.ParquetFile(data_file)
        writer = None

        for batch in pf.iter_batches(columns=features + self.target_columns + ['era'],
                                   batch_size=chunk_size):
            df = batch.to_pandas()

            # Group by era for prediction
            adaptive_targets = []
            for era, era_df in df.groupby('era'):
                # Predict best target for this era
                best_target = self.predict_best_target(era_df[features])

                # Use the predicted best target values
                era_targets = era_df[best_target].values
                adaptive_targets.extend(era_targets)

            df['adaptive_target'] = adaptive_targets

            # Write to parquet
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema)
            writer.write_table(table)

        if writer:
            writer.close()

        print(f"Adaptive targets created and saved to {output_file}")


def run_target_preference_meta_learning():
    """Main function to run the target preference meta-learning pipeline."""
    print("=" * 80)
    print("TARGET PREFERENCE META-LEARNING FRAMEWORK")
    print("=" * 80)

    # Configuration
    train_file = "v5.0/train.parquet"
    features_file = "v5.0/features.json"
    output_file = "v5.0/train_adaptive_meta.parquet"

    # Load features
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    features = features_data['feature_sets']['medium']

    # Get target columns
    pf = pq.ParquetFile(train_file)
    all_columns = [field.name for field in pf.schema]
    target_columns = [col for col in all_columns if col.startswith('target')]

    print(f"Found {len(target_columns)} targets and {len(features)} features")

    # Initialize predictor
    predictor = TargetPreferencePredictor(target_columns)

    # Step 1: Train individual target models
    print("\n" + "="*50)
    print("STEP 1: Training Individual Target Models")
    print("="*50)
    predictor.train_individual_target_models(train_file, features)

    # Step 2: Evaluate target performance
    print("\n" + "="*50)
    print("STEP 2: Evaluating Target Performance")
    print("="*50)
    predictor.evaluate_target_performance(train_file, features)

    # Step 3: Create meta-training data
    print("\n" + "="*50)
    print("STEP 3: Creating Meta-Training Data")
    print("="*50)
    X_meta, y_meta = predictor.create_meta_training_data(train_file, features)

    # Step 4: Train meta-model
    print("\n" + "="*50)
    print("STEP 4: Training Meta-Model")
    print("="*50)
    predictor.train_meta_model(X_meta, y_meta)

    # Step 5: Create adaptive targets
    print("\n" + "="*50)
    print("STEP 5: Creating Adaptive Targets")
    print("="*50)
    predictor.create_adaptive_targets_meta(train_file, features, output_file)
    
    # Step 6: Create adaptive targets for validation
    print("\n" + "="*50)
    print("STEP 6: Creating Adaptive Targets for Validation")
    print("="*50)
    val_output_file = "v5.0/validation_adaptive_meta.parquet"
    predictor.create_adaptive_targets_meta("v5.0/validation.parquet", features, val_output_file)
    print(f"Validation adaptive targets saved to: {val_output_file}")

    print("\n" + "="*80)
    print("META-LEARNING PIPELINE COMPLETED")
    print("="*80)
    print(f"Adaptive targets saved to: {output_file}")
    print("\nNext steps:")
    print("1. Evaluate the meta-learning approach vs. combination approach")
    print("2. Compare performance on validation data")
    print("3. Consider ensemble approaches combining both methods")

    return predictor


if __name__ == "__main__":
    predictor = run_target_preference_meta_learning()
