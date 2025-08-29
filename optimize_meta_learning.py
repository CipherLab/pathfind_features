#!/usr/bin/env python3
"""
Meta-Learning Parameter Sweep
Optimizes meta-learning hyperparameters for better target prediction.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import pyarrow.parquet as pq
import lightgbm as lgb
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import hashlib
import pickle

warnings.filterwarnings('ignore')


class MetaLearningOptimizer:
    """Optimizes meta-learning approach with parameter sweeps."""

    def __init__(self, target_columns: List[str], random_state: int = 42, cache_dir: str = "cache/meta_opt"):
        self.target_columns = target_columns
        self.n_targets = len(target_columns)
        self.random_state = random_state
        self.target_models = {}
        self.target_performance = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of a file."""
        h = hashlib.md5()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _get_cache_key(self, data_file: str, features: List[str], operation: str) -> str:
        """Generate cache key for a specific operation."""
        file_hash = self._get_file_hash(data_file)
        features_hash = hashlib.md5(json.dumps(sorted(features), sort_keys=True).encode()).hexdigest()
        targets_hash = hashlib.md5(json.dumps(sorted(self.target_columns), sort_keys=True).encode()).hexdigest()
        
        key_data = {
            'operation': operation,
            'data_file': file_hash,
            'features': features_hash,
            'targets': targets_hash,
            'random_state': self.random_state
        }
        
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def _cache_path(self, cache_key: str, suffix: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}_{suffix}"

    def _save_pickle(self, data: Any, path: Path) -> None:
        """Save data to pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _load_pickle(self, path: Path) -> Any:
        """Load data from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def compute_target_performance_cached(self, data_file: str, features: List[str],
                                        chunk_size: int = 100_000) -> Dict:
        """Compute target performance with caching."""
        cache_key = self._get_cache_key(data_file, features, 'target_performance')
        cache_path = self._cache_path(cache_key, 'target_perf.pkl')
        
        if cache_path.exists():
            print(f"Loading cached target performance from {cache_path}")
            self.target_performance = self._load_pickle(cache_path)
            return self.target_performance
        
        print("Computing target performance (no cache found)...")
        result = self._compute_target_performance(data_file, features, chunk_size)
        
        # Cache the result
        self._save_pickle(result, cache_path)
        print(f"Cached target performance to {cache_path}")
        
        return result

    def create_meta_features_cached(self, data_file: str, features: List[str],
                                  config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Create meta-features with caching."""
        # Include config in cache key
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
        cache_key = self._get_cache_key(data_file, features, f'meta_features_{config_hash}')
        cache_path = self._cache_path(cache_key, 'meta_data.pkl')
        
        if cache_path.exists():
            print(f"Loading cached meta-features from {cache_path}")
            X_meta, y_meta = self._load_pickle(cache_path)
            return X_meta, y_meta
        
        print("Creating meta-features (no cache found)...")
        X_meta, y_meta = self.create_meta_features(data_file, features, config)
        
        # Cache the result
        self._save_pickle((X_meta, y_meta), cache_path)
        print(f"Cached meta-features to {cache_path}")
        
        return X_meta, y_meta

    def train_meta_model_cached(self, X_meta: np.ndarray, y_meta: np.ndarray,
                              config: Dict) -> lgb.Booster:
        """Train meta-model with caching."""
        # Create hash of training data and config
        data_hash = hashlib.md5()
        data_hash.update(X_meta.tobytes())
        data_hash.update(y_meta.tobytes())
        data_hash.update(json.dumps(config, sort_keys=True).encode())
        
        cache_key = data_hash.hexdigest()
        cache_path = self._cache_path(cache_key, 'meta_model.pkl')
        
        if cache_path.exists():
            print(f"Loading cached meta-model from {cache_path}")
            return self._load_pickle(cache_path)
        
        print("Training meta-model (no cache found)...")
        model = self.train_meta_model(X_meta, y_meta, config)
        
        # Cache the model
        self._save_pickle(model, cache_path)
        print(f"Cached meta-model to {cache_path}")
        
        return model

    def create_meta_features(self, data_file: str, features: List[str],
                           config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Create meta-features and targets for training meta-model."""
        print(f"Creating meta-features with config: {config}")
        
        pf = pq.ParquetFile(data_file)
        meta_features = []
        meta_targets = []

        for batch in pf.iter_batches(columns=features + ['era'], batch_size=50_000):
            try:
                df = batch.to_pandas()
                if df.empty:
                    continue
                    
                # Group by era for meta-feature creation
                for era, era_df in df.groupby('era'):
                    if len(era_df) < 10:  # Skip eras with too few samples
                        continue

                    # Get best performing target for this era
                    era_perfs = self.target_performance[era]
                    if not era_perfs:
                        continue

                    best_target = max(era_perfs.items(), key=lambda x: x[1])[0]
                    best_target_idx = self.target_columns.index(best_target)

                    # Create meta-features based on config
                    meta_feat = self._create_era_meta_features(era_df, features, config)
                    
                    if not meta_feat:  # Skip if no features could be created
                        continue

                    meta_features.append(meta_feat)
                    meta_targets.append(best_target_idx)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        X_meta = np.array(meta_features)
        y_meta = np.array(meta_targets)

        print(f"Created meta-training data: {X_meta.shape[0]} samples, {X_meta.shape[1]} features")
        return X_meta, y_meta

    def train_meta_model(self, X_meta: np.ndarray, y_meta: np.ndarray,
                        config: Dict) -> lgb.Booster:
        """Train meta-model with configurable parameters."""
        print(f"Training meta-model with config: {config}")

        lgb_params = {
            'objective': 'multiclass',
            'num_class': self.n_targets,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.05),
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'verbosity': -1,
            'num_threads': -1,
            'seed': self.random_state,
            'num_boost_round': config.get('num_boost_round', 200)
        }

        train_set = lgb.Dataset(X_meta, label=y_meta)
        model = lgb.train(lgb_params, train_set)

        print("Meta-model trained successfully")
        return model

    def evaluate_config(self, config: Dict, train_file: str, val_file: str,
                       features: List[str]) -> Dict:
        """Evaluate a meta-learning configuration."""
        print(f"\n{'='*50}")
        print(f"Evaluating config: {config}")
        print(f"{'='*50}")

        # Create meta-features and train meta-model on training data (with caching)
        X_meta, y_meta = self.create_meta_features_cached(train_file, features, config)

        if X_meta.shape[0] < 10:
            return {'error': 'Insufficient meta-training data'}

        # Train meta-model (with caching)
        meta_model = self.train_meta_model_cached(X_meta, y_meta, config)

        # For evaluation, we need target performance on validation data
        val_optimizer = MetaLearningOptimizer(self.target_columns, self.random_state, str(self.cache_dir))
        val_optimizer.compute_target_performance_cached(val_file, features)

        # Evaluate on validation data
        pf = pq.ParquetFile(val_file)
        correct_predictions = 0
        total_predictions = 0
        meta_features = []
        meta_targets = []

        for batch in pf.iter_batches(columns=features + ['era'], batch_size=50_000):
            try:
                df = batch.to_pandas()
                if df.empty:
                    continue
                    
                # Group by era for meta-feature creation
                for era, era_df in df.groupby('era'):
                    if len(era_df) < 10:  # Skip eras with too few samples
                        continue

                    # Get best performing target for this era
                    if era not in val_optimizer.target_performance:
                        continue
                    era_perfs = val_optimizer.target_performance[era]
                    if not era_perfs:
                        continue

                    best_target = max(era_perfs.items(), key=lambda x: x[1])[0]
                    best_target_idx = self.target_columns.index(best_target)

                    # Create meta-features based on config
                    meta_feat = self._create_era_meta_features(era_df, features, config)
                    
                    if not meta_feat:  # Skip if no features could be created
                        continue

                    # Predict best target
                    X_pred = np.array([meta_feat])
                    pred_result = meta_model.predict(X_pred)
                    if isinstance(pred_result, (list, tuple)) and len(pred_result) > 0:
                        pred_probs = pred_result[0]
                    else:
                        pred_probs = pred_result
                    
                    # Ensure pred_probs is a numpy array
                    pred_probs = np.asarray(pred_probs).flatten()
                    pred_target_idx = int(np.argmax(pred_probs))
                    pred_target = self.target_columns[pred_target_idx]

                    # Check if prediction is correct
                    if pred_target == best_target:
                        correct_predictions += 1
                    total_predictions += 1
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        print(f"Meta-model accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")

        return {
            'accuracy': accuracy,
            'n_predictions': total_predictions,
            'config': config
        }

    def _create_era_meta_features(self, era_df: pd.DataFrame, features: List[str], config: Dict) -> List[float]:
        """Create meta-features for a single era (simplified version)."""
        if len(era_df) == 0:
            return []
            
        meta_feat = []

        if config.get('use_basic_stats', True):
            for feat in features[:50]:  # Limit features for speed
                feat_vals = era_df[feat].dropna()
                if len(feat_vals) > 0:
                    try:
                        meta_feat.extend([
                            float(feat_vals.mean()),
                            float(feat_vals.std()),
                            float(feat_vals.min()),
                            float(feat_vals.max()),
                            float(feat_vals.median())
                        ])
                    except:
                        meta_feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    meta_feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Add era information
        try:
            # Get era value - it might be a string or already processed
            era_val = era_df['era'].iloc[0] if len(era_df) > 0 else '0001'
            era_str = str(era_val)
            
            # Handle different era formats
            if era_str.startswith('era'):
                era_num = int(era_str.replace('era_', '').replace('era', ''))
            else:
                # Try to parse as direct number
                era_num = int(float(era_str))
            
            meta_feat.append(float(era_num))
        except Exception as e:
            # If all else fails, use a default value
            meta_feat.append(0.0)

        return meta_feat

    def _compute_target_performance(self, data_file: str, features: List[str],
                                  chunk_size: int = 100_000) -> Dict:
        """Compute target performance (simplified version)."""
        pf = pq.ParquetFile(data_file)
        target_performance = {}

        for batch in pf.iter_batches(columns=features + ['era'], batch_size=chunk_size):
            df = batch.to_pandas()

            # Group by era
            for era, era_df in df.groupby('era'):
                # Calculate simple mean correlation for each target
                era_performance = {}
                for target in self.target_columns:
                    if target in era_df:
                        try:
                            # Simple correlation calculation without scipy
                            target_vals = era_df[target].values
                            feature_mean = era_df[features].mean(axis=1).values
                            
                            # Remove NaN values
                            valid_mask = ~(np.isnan(target_vals) | np.isnan(feature_mean))
                            if np.sum(valid_mask) > 1:
                                t_clean = target_vals[valid_mask]
                                f_clean = feature_mean[valid_mask]
                                
                                # Simple correlation coefficient
                                t_mean = np.mean(t_clean)
                                f_mean = np.mean(f_clean)
                                numerator = np.sum((t_clean - t_mean) * (f_clean - f_mean))
                                denominator = np.sqrt(np.sum((t_clean - t_mean)**2) * np.sum((f_clean - f_mean)**2))
                                
                                if denominator > 0:
                                    corr = numerator / denominator
                                else:
                                    corr = 0.0
                            else:
                                corr = 0.0
                                
                            era_performance[target] = float(corr)
                        except Exception as e:
                            print(f"Error computing correlation for {target} in era {era}: {e}")
                            era_performance[target] = 0.0
                    else:
                        era_performance[target] = 0.0

                target_performance[era] = era_performance

        print(f"Computed target performance for {len(target_performance)} eras")
        return target_performance


def run_meta_optimization():
    """Run parameter sweep for meta-learning optimization."""
    print("=" * 80)
    print("META-LEARNING PARAMETER OPTIMIZATION")
    print("=" * 80)

    # Configuration
    train_file = "v5.0/train.parquet"
    val_file = "v5.0/validation.parquet"
    features_file = "v5.0/features.json"

    # Load features
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    features = features_data['feature_sets']['medium']

    # Get target columns
    pf = pq.ParquetFile(train_file)
    all_columns = [field.name for field in pf.schema]
    target_columns = [col for col in all_columns if col.startswith('target')]

    print(f"Found {len(target_columns)} targets and {len(features)} features")

    # Initialize optimizer
    optimizer = MetaLearningOptimizer(target_columns)

    # Compute target performance (required for meta-learning)
    print("\nComputing target performance across eras...")
    optimizer.compute_target_performance_cached(train_file, features)

    # Define parameter sweep configurations
    configs = [
        # Baseline
        {
            'name': 'baseline',
            'use_basic_stats': True,
            'use_percentiles': False,
            'use_correlation_stats': False,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'num_boost_round': 200
        },
        # More features
        {
            'name': 'more_features',
            'use_basic_stats': True,
            'use_percentiles': True,
            'use_correlation_stats': False,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'num_boost_round': 200
        },
        # Deeper trees
        {
            'name': 'deeper_trees',
            'use_basic_stats': True,
            'use_percentiles': False,
            'use_correlation_stats': False,
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'num_boost_round': 300
        },
        # Faster learning
        {
            'name': 'faster_learning',
            'use_basic_stats': True,
            'use_percentiles': False,
            'use_correlation_stats': False,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'num_boost_round': 100
        }
    ]

    results = []
    for config in configs:
        try:
            result = optimizer.evaluate_config(config, train_file, val_file, features)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating config {config['name']}: {e}")
            results.append({'error': str(e), 'config': config})

    # Sort results by accuracy
    valid_results = [r for r in results if 'accuracy' in r]
    valid_results.sort(key=lambda x: x['accuracy'], reverse=True)

    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)

    for i, result in enumerate(valid_results[:5]):
        config = result['config']
        print(f"{i+1}. {config['name']}: Accuracy {result['accuracy']:.3f}")

    if valid_results:
        best_config = valid_results[0]['config']
        print(f"\n🏆 BEST CONFIG: {best_config['name']} (Accuracy: {valid_results[0]['accuracy']:.3f})")

        # Save best config
        with open('v5.0/best_meta_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)

    return valid_results


def run_simple_optimization():
    """Run a simple optimization with just the baseline config."""
    print("=" * 80)
    print("SIMPLE META-LEARNING OPTIMIZATION")
    print("=" * 80)

    # Configuration
    train_file = "v5.0/train.parquet"
    val_file = "v5.0/validation.parquet"
    features_file = "v5.0/features.json"

    # Load features
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    features = features_data['feature_sets']['medium']

    # Get target columns
    pf = pq.ParquetFile(train_file)
    all_columns = [field.name for field in pf.schema]
    target_columns = [col for col in all_columns if col.startswith('target')]

    print(f"Found {len(target_columns)} targets and {len(features)} features")

    # Initialize optimizer
    optimizer = MetaLearningOptimizer(target_columns)

    # Compute target performance (required for meta-learning)
    print("\nComputing target performance across eras...")
    optimizer.compute_target_performance_cached(train_file, features)

    # Just use baseline config
    config = {
        'name': 'baseline',
        'use_basic_stats': True,
        'use_percentiles': False,
        'use_correlation_stats': False,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'num_boost_round': 200
    }

    try:
        result = optimizer.evaluate_config(config, train_file, val_file, features)
        print(f"\n✅ SUCCESS! Baseline config accuracy: {result.get('accuracy', 'N/A')}")
        
        # Save best config
        with open('v5.0/best_meta_config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        return [result]
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return [{'error': str(e), 'config': config}]


if __name__ == "__main__":
    results = run_meta_optimization()
