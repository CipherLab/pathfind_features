#!/usr/bin/env python3
"""
Feature Stability Engine for Financial Time Series Models

This module evaluates feature stability across different market regimes and
creates a curated set of robust features for model training.

Key functionality:
- Load candidate features from feature engineering output
- Evaluate feature correlations across VIX-based market regimes
- Remove unstable features (sign flips, magnitude collapse >50%)
- Build ratio features from stable feature pairs
- Output curated feature list for model training
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import pyarrow.parquet as pq
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from utils import reduce_mem_usage, RegimeDetector


class FeatureStabilityEngine:
    """Evaluates feature stability across market regimes and curates robust features."""

    def __init__(self, vix_file: str | None = None, min_regime_samples: int = 50):
        self.vix_file = vix_file
        self.min_regime_samples = min_regime_samples
        self.regime_data = None
        self.logger = logging.getLogger(__name__)

    def load_and_prepare_data(self, data_file: str, target_col: str = 'adaptive_target') -> pd.DataFrame:
        """Load data and prepare for stability analysis."""
        self.logger.info(f"Loading data from {data_file}")
        pf = pq.ParquetFile(data_file)
        df = pf.read().to_pandas()
        df = reduce_mem_usage(df, _verbose=False)

        # Load VIX data and categorize regimes using shared detector
        detector = RegimeDetector()
        vix_data = detector.load_vix_data(df['era'].unique().tolist(), self.vix_file)
        df['vix_regime'] = detector.classify_eras(df['era'], vix_data, use_percentiles=True)

        self.regime_data = vix_data
        self.logger.info(f"Data loaded: {len(df):,} rows, {len(df.columns)} columns")
        self.logger.info(f"Regime distribution: {df['vix_regime'].value_counts().to_dict()}")

        return df

    def evaluate_feature_stability(self, df: pd.DataFrame, features: List[str],
                                 target_col: str = 'adaptive_target') -> Dict[str, Dict]:
        """Evaluate feature stability across regimes."""
        self.logger.info(f"Evaluating stability for {len(features)} features")

        stability_results = {}

        for feature in features:
            if feature not in df.columns:
                self.logger.warning(f"Feature {feature} not found in data, skipping")
                continue

            feature_stability = self._analyze_single_feature(df, feature, target_col)
            stability_results[feature] = feature_stability

        return stability_results

    def _analyze_single_feature(self, df: pd.DataFrame, feature: str,
                              target_col: str) -> Dict:
        """Analyze stability of a single feature across regimes."""
        regime_correlations = {}
        regime_samples = {}

        for regime in df['vix_regime'].unique():
            regime_mask = df['vix_regime'] == regime
            regime_data = df[regime_mask]

            if len(regime_data) < self.min_regime_samples:
                self.logger.warning(f"Regime {regime} has only {len(regime_data)} samples, skipping")
                continue

            try:
                corr, p_value = spearmanr(regime_data[feature], regime_data[target_col])
                if not np.isnan(corr):
                    regime_correlations[regime] = float(corr)
                    regime_samples[regime] = len(regime_data)
            except Exception as e:
                self.logger.warning(f"Failed to compute correlation for {feature} in {regime}: {e}")

        if not regime_correlations:
            return {
                'stable': False,
                'reason': 'no_valid_regime_correlations',
                'regime_correlations': {},
                'stability_score': 0.0
            }

        # Check for sign flips
        correlations = list(regime_correlations.values())
        signs = [np.sign(corr) for corr in correlations]
        sign_consistent = len(set(signs)) == 1

        # Check for magnitude collapse (>50% drop from best to worst)
        if len(correlations) > 1:
            max_corr = max(abs(c) for c in correlations)
            min_corr = min(abs(c) for c in correlations)
            magnitude_drop = (max_corr - min_corr) / max_corr if max_corr > 0 else 0
            magnitude_stable = magnitude_drop <= 0.5
        else:
            magnitude_stable = True
            magnitude_drop = 0.0

        # Calculate stability score (0-1, higher is better)
        mean_corr = np.mean([abs(c) for c in correlations])
        corr_std = np.std(correlations)
        stability_score = mean_corr * (1 - min(float(corr_std), 1.0))  # Penalize high variance

        stable = sign_consistent and magnitude_stable and mean_corr > 0.01

        return {
            'stable': stable,
            'reason': 'stable' if stable else f'sign_flip:{not sign_consistent}, magnitude_drop:{magnitude_drop:.2f}',
            'regime_correlations': regime_correlations,
            'regime_samples': regime_samples,
            'stability_score': float(stability_score),
            'mean_correlation': float(mean_corr),
            'correlation_std': float(corr_std),
            'magnitude_drop': float(magnitude_drop)
        }

    def build_ratio_features(self, df: pd.DataFrame, stable_features: List[str],
                           max_ratios: int = 10) -> List[str]:
        """Build ratio features from pairs of stable features."""
        self.logger.info(f"Building up to {max_ratios} ratio features from {len(stable_features)} stable features")

        ratio_features = []
        ratio_scores = []

        # Sample a subset for efficiency if too many features
        if len(stable_features) > 50:
            np.random.seed(42)
            candidate_features = np.random.choice(stable_features, 50, replace=False)
        else:
            candidate_features = stable_features

        # Generate ratio feature candidates
        for i, feat1 in enumerate(candidate_features):
            for feat2 in candidate_features[i+1:]:
                try:
                    # Calculate ratio
                    ratio_values = df[feat1] / (df[feat2].abs() + 1e-8)

                    # Check if ratio is well-behaved (not too many infinities/NaNs)
                    valid_ratio = ratio_values.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(valid_ratio) / len(df) < 0.8:  # Less than 80% valid values
                        continue

                    # Evaluate ratio stability by computing correlation with target
                    try:
                        ratio_corr, _ = spearmanr(valid_ratio, df.loc[valid_ratio.index, 'adaptive_target'])
                        ratio_stability_score = abs(ratio_corr) if not np.isnan(ratio_corr) else 0.0
                    except:
                        ratio_stability_score = 0.0

                    if ratio_stability_score > 0.01:  # Minimum correlation threshold
                        ratio_name = f"ratio_{str(feat1).split('_')[-1]}_{str(feat2).split('_')[-1]}"
                        ratio_features.append(ratio_name)
                        ratio_scores.append((ratio_name, ratio_stability_score))

                        # Add to dataframe for later use
                        df[ratio_name] = ratio_values.astype('float32')

                except Exception as e:
                    self.logger.debug(f"Failed to create ratio {feat1}/{feat2}: {e}")
                    continue

        # Select top ratio features by stability score
        ratio_features_sorted = sorted(ratio_scores, key=lambda x: x[1], reverse=True)
        selected_ratios = [name for name, score in ratio_features_sorted[:max_ratios]]

        self.logger.info(f"Created {len(selected_ratios)} stable ratio features")
        return selected_ratios

    def curate_features(self, stability_results: Dict[str, Dict], original_features: List[str],
                       max_ratios: int = 10) -> Dict[str, List[str]]:
        """Curate the final feature set based on stability analysis."""
        self.logger.info("Curating final feature set")

        # Get stable features
        stable_features = [feat for feat, result in stability_results.items()
                          if result.get('stable', False)]

        self.logger.info(f"Found {len(stable_features)} stable features out of {len(stability_results)} evaluated")

        # Sort by stability score
        stable_features_sorted = sorted(
            [(feat, stability_results[feat]['stability_score']) for feat in stable_features],
            key=lambda x: x[1],
            reverse=True
        )

        # Take top stable features (limit to reasonable number)
        max_stable_features = min(len(stable_features), 200)  # Reasonable limit
        top_stable_features = [feat for feat, score in stable_features_sorted[:max_stable_features]]

        # Build ratio features from top stable features
        # Note: This would require the dataframe, so we'll prepare the list
        ratio_features = []  # Will be populated during processing

        curated_features = {
            'stable_features': top_stable_features,
            'ratio_features': ratio_features,
            'all_curated': top_stable_features + ratio_features
        }

        return curated_features


def run_feature_stability_analysis(data_file: str, features_file: str,
                                 output_dir: str, vix_file: str | None = None,
                                 target_col: str = 'adaptive_target') -> Dict:
    """Run the complete feature stability analysis pipeline."""

    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'feature_stability.log')),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting feature stability analysis")

    # Initialize engine
    engine = FeatureStabilityEngine(vix_file=vix_file)

    # Load data
    df = engine.load_and_prepare_data(data_file, target_col)

    # Load candidate features
    with open(features_file, 'r') as f:
        candidate_features_data = json.load(f)

    if isinstance(candidate_features_data, dict) and 'features' in candidate_features_data:
        candidate_features = candidate_features_data['features']
    elif isinstance(candidate_features_data, list):
        candidate_features = candidate_features_data
    else:
        raise ValueError("Features file must contain a list or dict with 'features' key")

    logger.info(f"Loaded {len(candidate_features)} candidate features")

    # Evaluate stability
    stability_results = engine.evaluate_feature_stability(df, candidate_features, target_col)

    # Curate features
    curated_features = engine.curate_features(stability_results, candidate_features)

    # Build ratio features (requires dataframe)
    stable_features = curated_features['stable_features']
    if len(stable_features) >= 2:
        ratio_features = engine.build_ratio_features(df, stable_features[:50])  # Use top 50 for ratios
        curated_features['ratio_features'] = ratio_features
        curated_features['all_curated'] = stable_features + ratio_features

    # Save results
    results = {
        'stability_analysis': stability_results,
        'curated_features': curated_features,
        'regime_distribution': df['vix_regime'].value_counts().to_dict(),
        'summary': {
            'total_candidates': len(candidate_features),
            'stable_features': len(curated_features['stable_features']),
            'ratio_features': len(curated_features['ratio_features']),
            'total_curated': len(curated_features['all_curated'])
        }
    }

    # Save curated features list
    curated_file = os.path.join(output_dir, 'curated_features.json')
    with open(curated_file, 'w') as f:
        json.dump(curated_features['all_curated'], f, indent=2)

    # Save full results
    results_file = os.path.join(output_dir, 'feature_stability_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Feature stability analysis complete. Results saved to {output_dir}")
    logger.info(f"Curated {len(curated_features['all_curated'])} features from {len(candidate_features)} candidates")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Stability Analysis")
    parser.add_argument('--data-file', required=True, help='Path to parquet data file')
    parser.add_argument('--features-file', required=True, help='Path to JSON file with candidate features')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--vix-file', help='Optional path to VIX data CSV')
    parser.add_argument('--target-col', default='adaptive_target', help='Target column name')

    args = parser.parse_args()

    run_feature_stability_analysis(
        args.data_file,
        args.features_file,
        args.output_dir,
        args.vix_file,
        args.target_col
    )
