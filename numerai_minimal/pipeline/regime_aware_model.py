#!/usr/bin/env python3
"""
Regime-Aware Model Training for Numerai

This module implements specialized models for different market regimes:
- High volatility/crisis periods
- Low volatility/grind periods
- Transition periods

The goal is to create models that work well in their specific regimes
rather than one model that fails everywhere equally.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class RegimeAwareModelTrainer:
    """Train specialized models for different market regimes."""

    def __init__(self, vix_thresholds: Tuple[float, float] = (15, 25)):
        self.vix_thresholds = vix_thresholds
        self.low_vol_threshold, self.high_vol_threshold = vix_thresholds
        self.models = {}
        self.logger = logging.getLogger(__name__)

    def classify_regime(self, vix_value: float) -> str:
        """Classify market regime based on VIX level."""
        if vix_value > self.high_vol_threshold:
            return "crisis"
        elif vix_value < self.low_vol_threshold:
            return "grind"
        else:
            return "transition"

    def prepare_regime_data(self, df: pd.DataFrame, target_col: str = 'adaptive_target') -> Dict[str, pd.DataFrame]:
        """Split data into regime-specific datasets."""
        if 'vix' not in df.columns:
            raise ValueError("Data must contain 'vix' column for regime classification")

        regime_data = {}
        for regime in ['crisis', 'grind', 'transition']:
            if regime == 'crisis':
                mask = df['vix'] > self.high_vol_threshold
            elif regime == 'grind':
                mask = df['vix'] < self.low_vol_threshold
            else:  # transition
                mask = (df['vix'] >= self.low_vol_threshold) & (df['vix'] <= self.high_vol_threshold)

            regime_df = df[mask].copy()
            if len(regime_df) > 0:
                regime_data[regime] = regime_df
                self.logger.info(f"Regime {regime}: {len(regime_df)} samples")
            else:
                self.logger.warning(f"No data for regime {regime}")

        return regime_data

    def train_regime_model(self, df: pd.DataFrame, features: List[str],
                          target_col: str = 'adaptive_target',
                          params: Optional[Dict] = None) -> lgb.Booster:
        """Train a model for a specific regime."""
        if len(df) < 100:
            self.logger.warning(f"Very small dataset: {len(df)} samples")
            return None

        X = df[features]
        y = df[target_col]

        # Default parameters optimized for regression
        default_params = {
            'objective': 'regression',
            'metric': 'l2',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }

        if params:
            default_params.update(params)

        # Group by era for ranking
        if 'era' in df.columns:
            groups = df.groupby('era').size().values
            train_set = lgb.Dataset(X, label=y, group=groups)
        else:
            train_set = lgb.Dataset(X, label=y)

        model = lgb.train(default_params, train_set, num_boost_round=200)
        return model

    def train_all_regimes(self, df: pd.DataFrame, features: List[str],
                         target_col: str = 'adaptive_target',
                         params: Optional[Dict] = None) -> Dict[str, lgb.Booster]:
        """Train specialized models for each regime."""
        self.logger.info("Training regime-aware models...")

        regime_data = self.prepare_regime_data(df, target_col)

        for regime, regime_df in regime_data.items():
            self.logger.info(f"Training {regime} model with {len(regime_df)} samples")
            model = self.train_regime_model(regime_df, features, target_col, params)
            if model is not None:
                self.models[regime] = model
                self.logger.info(f"Successfully trained {regime} model")
            else:
                self.logger.warning(f"Failed to train {regime} model")

        return self.models

    def predict_regime_aware(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """Make predictions using the appropriate regime model."""
        if 'vix' not in df.columns:
            raise ValueError("Data must contain 'vix' column for regime classification")

        predictions = np.zeros(len(df))

        for regime in ['crisis', 'grind', 'transition']:
            if regime not in self.models:
                self.logger.warning(f"No model available for regime {regime}")
                continue

            if regime == 'crisis':
                mask = df['vix'] > self.high_vol_threshold
            elif regime == 'grind':
                mask = df['vix'] < self.low_vol_threshold
            else:  # transition
                mask = (df['vix'] >= self.low_vol_threshold) & (df['vix'] <= self.high_vol_threshold)

            if mask.any():
                regime_X = df.loc[mask, features]
                regime_preds = self.models[regime].predict(regime_X)
                predictions[mask.values] = regime_preds

        return predictions

    def evaluate_regime_performance(self, df: pd.DataFrame, features: List[str],
                                   target_col: str = 'adaptive_target') -> Dict[str, float]:
        """Evaluate performance of regime-aware predictions."""
        predictions = self.predict_regime_aware(df, features)
        targets = df[target_col]

        # Overall correlation
        overall_corr, _ = spearmanr(predictions, targets)

        # Era-level correlations for Sharpe calculation
        era_correlations = []
        if 'era' in df.columns:
            for era in sorted(df['era'].unique()):
                era_mask = df['era'] == era
                era_targets = targets[era_mask]
                era_preds = predictions[era_mask]
                if len(era_targets) > 1:
                    era_corr, _ = spearmanr(era_preds, era_targets)
                    era_correlations.append(era_corr)

        # Calculate Sharpe ratio
        if era_correlations:
            mean_corr = np.mean(era_correlations)
            std_corr = np.std(era_correlations, ddof=1)
            sharpe = mean_corr / std_corr if std_corr > 0 else 0

            # Transaction cost adjustment
            tc_impact = 25 / 10000 / 0.15  # 25bps TC, 15% vol
            sharpe_with_tc = max(0.0, sharpe - tc_impact)
        else:
            sharpe = sharpe_with_tc = 0

        # Regime-specific performance
        regime_performance = {}
        for regime in ['crisis', 'grind', 'transition']:
            if regime == 'crisis':
                mask = df['vix'] > self.high_vol_threshold
            elif regime == 'grind':
                mask = df['vix'] < self.low_vol_threshold
            else:  # transition
                mask = (df['vix'] >= self.low_vol_threshold) & (df['vix'] <= self.high_vol_threshold)

            if mask.any():
                regime_targets = targets[mask]
                regime_preds = predictions[mask.values]
                regime_corr, _ = spearmanr(regime_preds, regime_targets)
                regime_performance[f'corr_{regime}'] = regime_corr

        return {
            'overall_correlation': overall_corr,
            'sharpe_ratio': sharpe,
            'sharpe_with_tc': sharpe_with_tc,
            'n_eras': len(era_correlations),
            **regime_performance
        }

    def save_models(self, output_dir: str):
        """Save trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)

        for regime, model in self.models.items():
            model_path = os.path.join(output_dir, f'{regime}_model.txt')
            model.save_model(model_path)
            self.logger.info(f"Saved {regime} model to {model_path}")

        # Save metadata
        metadata = {
            'vix_thresholds': self.vix_thresholds,
            'regimes': list(self.models.keys()),
            'model_paths': {regime: f'{regime}_model.txt' for regime in self.models.keys()}
        }

        metadata_path = os.path.join(output_dir, 'regime_models_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved metadata to {metadata_path}")

    def load_models(self, input_dir: str):
        """Load trained models from disk."""
        metadata_path = os.path.join(input_dir, 'regime_models_metadata.json')

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.vix_thresholds = tuple(metadata['vix_thresholds'])
        self.low_vol_threshold, self.high_vol_threshold = self.vix_thresholds

        for regime in metadata['regimes']:
            model_path = os.path.join(input_dir, metadata['model_paths'][regime])
            if os.path.exists(model_path):
                self.models[regime] = lgb.Booster(model_file=model_path)
                self.logger.info(f"Loaded {regime} model from {model_path}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")


def run_regime_aware_training(data_file: str, features_file: str,
                             output_dir: str, vix_file: str | None = None,
                             target_col: str = 'adaptive_target') -> Dict:
    """Run the complete regime-aware model training pipeline."""

    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'regime_training.log')),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting regime-aware model training")

    # Load data
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(data_file)
    df = pf.read().to_pandas()

    # Load VIX data if not in main dataset
    if 'vix' not in df.columns:
        if vix_file and os.path.exists(vix_file):
            vix_df = pd.read_csv(vix_file)
            df = df.merge(vix_df[['era', 'vix']], on='era', how='left')
        else:
            logger.warning("No VIX data available, using simulated regime classification")
            # Simulate VIX based on era for demo purposes
            np.random.seed(42)
            df['vix'] = np.random.normal(20, 5, len(df))

    # Load features
    with open(features_file, 'r') as f:
        features_data = json.load(f)

    if isinstance(features_data, dict) and 'features' in features_data:
        features = features_data['features']
    elif isinstance(features_data, list):
        features = features_data
    else:
        raise ValueError("Features file must contain a list or dict with 'features' key")

    logger.info(f"Loaded {len(features)} features")

    # Initialize trainer
    trainer = RegimeAwareModelTrainer()

    # Train models
    models = trainer.train_all_regimes(df, features, target_col)

    # Evaluate performance
    performance = trainer.evaluate_regime_performance(df, features, target_col)

    # Save models
    trainer.save_models(output_dir)

    # Save performance results
    results = {
        'performance': performance,
        'regimes_trained': list(models.keys()),
        'n_features': len(features),
        'data_shape': df.shape,
        'vix_stats': {
            'mean': float(df['vix'].mean()),
            'std': float(df['vix'].std()),
            'min': float(df['vix'].min()),
            'max': float(df['vix'].max())
        }
    }

    results_path = os.path.join(output_dir, 'regime_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Regime-aware training complete")
    logger.info(f"Overall correlation: {performance['overall_correlation']:.4f}")
    logger.info(f"Sharpe with TC: {performance['sharpe_with_tc']:.2f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regime-Aware Model Training")
    parser.add_argument('--data-file', required=True, help='Path to parquet data file')
    parser.add_argument('--features-file', required=True, help='Path to JSON file with features')
    parser.add_argument('--output-dir', required=True, help='Output directory for models')
    parser.add_argument('--vix-file', help='Optional path to VIX data CSV')
    parser.add_argument('--target-col', default='adaptive_target', help='Target column name')

    args = parser.parse_args()

    run_regime_aware_training(
        args.data_file,
        args.features_file,
        args.output_dir,
        args.vix_file,
        args.target_col
    )
