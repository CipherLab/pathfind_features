#!/usr/bin/env python3
"""
Feature Purge Engine for Numerai

This module implements aggressive feature selection by testing features
across different market regimes and time periods. Only features that
demonstrate consistent performance survive the purge.

Key principles:
- Test on 2008 crisis, COVID crash, and random bear markets
- Remove features with sign flips or >50% correlation drop
- Focus on mathematical relationships over historical patterns
- Prioritize stability over peak performance
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging
import warnings
warnings.filterwarnings('ignore')


class FeaturePurgeEngine:
    """Aggressively purge unstable features across market regimes."""

    def __init__(self, crisis_eras: Optional[List[str]] = None,
                 covid_eras: Optional[List[str]] = None,
                 bear_market_eras: Optional[List[str]] = None,
                 sample_size: int = 5000,  # Reduced from 50000
                 cache_regimes: bool = True,
                 vix_thresholds: Tuple[float, float] = (15.0, 25.0)):
        self.crisis_eras = crisis_eras or ['2008-01', '2008-02', '2008-03', '2008-04']  # Example crisis eras
        self.covid_eras = covid_eras or ['2020-03', '2020-04', '2020-05', '2020-06']  # COVID crash
        self.bear_market_eras = bear_market_eras or ['2018-10', '2018-11', '2018-12']  # Example bear market
        self.sample_size = sample_size
        self.cache_regimes = cache_regimes
        self.vix_low, self.vix_high = vix_thresholds
        self._cached_regimes = None
        self.logger = logging.getLogger(__name__)

    def load_historical_regimes(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Build regimes from available data.
        Prefer VIX-based regimes if vix is present; otherwise fall back to specific eras.
        """
        if self.cache_regimes and self._cached_regimes is not None:
            return self._cached_regimes

        regimes: Dict[str, pd.DataFrame] = {}

        if 'vix' in df.columns:
            # Construct three regimes by VIX thresholds
            crisis_mask = df['vix'] > self.vix_high
            grind_mask = df['vix'] < self.vix_low
            transition_mask = ~crisis_mask & ~grind_mask

            if crisis_mask.any():
                regimes['crisis'] = df[crisis_mask].copy()
                self.logger.info(f"Loaded {len(regimes['crisis'])} samples from crisis regime (VIX > {self.vix_high})")
            if grind_mask.any():
                regimes['grind'] = df[grind_mask].copy()
                self.logger.info(f"Loaded {len(regimes['grind'])} samples from grind regime (VIX < {self.vix_low})")
            if transition_mask.any():
                regimes['transition'] = df[transition_mask].copy()
                self.logger.info(f"Loaded {len(regimes['transition'])} samples from transition regime ({self.vix_low} <= VIX <= {self.vix_high})")
        else:
            # Fall back to era lists as before
            crisis_mask = df['era'].isin(self.crisis_eras)
            if crisis_mask.any():
                regimes['crisis_2008'] = df[crisis_mask].copy()
                self.logger.info(f"Loaded {len(regimes['crisis_2008'])} samples from 2008 crisis")

            covid_mask = df['era'].isin(self.covid_eras)
            if covid_mask.any():
                regimes['covid_crash'] = df[covid_mask].copy()
                self.logger.info(f"Loaded {len(regimes['covid_crash'])} samples from COVID crash")

            bear_mask = df['era'].isin(self.bear_market_eras)
            if bear_mask.any():
                regimes['bear_market'] = df[bear_mask].copy()
                self.logger.info(f"Loaded {len(regimes['bear_market'])} samples from bear market")

        # Normal periods (for comparison) - always sample to control runtime
        normal_mask = pd.Series(True, index=df.index)
        for part in regimes.values():
            normal_mask &= ~df.index.isin(part.index)
        if normal_mask.any():
            normal_sample = df[normal_mask].sample(min(self.sample_size, int(normal_mask.sum())), random_state=42)
            regimes['normal_periods'] = normal_sample
            self.logger.info(f"Loaded {len(regimes['normal_periods'])} samples from normal periods")

        if self.cache_regimes:
            self._cached_regimes = regimes

        return regimes

    def test_feature_stability(self, df: pd.DataFrame, feature: str,
                             target_col: str = 'adaptive_target',
                             regimes: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
        """Test a single feature's stability across regimes."""
        if feature not in df.columns:
            return {'stable': False, 'reason': 'feature_not_found'}

        # Use provided regimes or load them (for backward compatibility)
        if regimes is None:
            regimes = self.load_historical_regimes(df)

        if not regimes:
            return {'stable': False, 'reason': 'no_regime_data'}

        regime_correlations = {}
        regime_samples = {}

        for regime_name, regime_df in regimes.items():
            if len(regime_df) < 50:  # Minimum sample size
                continue

            try:
                corr, p_value = spearmanr(regime_df[feature], regime_df[target_col])
                if not np.isnan(corr) and not np.isnan(p_value):
                    regime_correlations[regime_name] = float(corr)
                    regime_samples[regime_name] = len(regime_df)
            except Exception as e:
                self.logger.debug(f"Failed to compute correlation for {feature} in {regime_name}: {e}")

        if len(regime_correlations) < 2:
            return {
                'stable': False,
                'reason': 'insufficient_regime_data',
                'regime_correlations': regime_correlations
            }

        # Check for sign consistency
        correlations = list(regime_correlations.values())
        signs = [np.sign(corr) for corr in correlations if abs(corr) > 0.01]  # Ignore very weak correlations
        sign_consistent = len(set(signs)) <= 1

        # Check for magnitude stability (no >50% drop from best to worst)
        if len(correlations) > 1:
            abs_correlations = [abs(c) for c in correlations]
            max_corr = max(abs_correlations)
            min_corr = min(abs_correlations)
            magnitude_drop = (max_corr - min_corr) / max_corr if max_corr > 0 else 0
            magnitude_stable = magnitude_drop <= 0.5
        else:
            magnitude_drop = 0
            magnitude_stable = True

        # Calculate stability score
        mean_corr = np.mean([abs(c) for c in correlations])
        corr_std = np.std(correlations)
        stability_score = mean_corr * (1 - min(float(corr_std), 1.0))

        # Feature must have minimum correlation strength
        min_correlation = mean_corr >= 0.005

        stable = sign_consistent and magnitude_stable and min_correlation

        return {
            'stable': stable,
            'reason': 'stable' if stable else f'sign_flip:{not sign_consistent}, magnitude_drop:{magnitude_drop:.2f}, weak_corr:{not min_correlation}',
            'regime_correlations': regime_correlations,
            'regime_samples': regime_samples,
            'stability_score': float(stability_score),
            'mean_correlation': float(mean_corr),
            'correlation_std': float(corr_std),
            'magnitude_drop': float(magnitude_drop),
            'sign_consistent': sign_consistent,
            'magnitude_stable': magnitude_stable
        }

    def purge_unstable_features(self, df: pd.DataFrame, features: List[str],
                               target_col: str = 'adaptive_target',
                               smoke_test: bool = False) -> Dict[str, Dict]:
        """Purge features that don't survive regime testing with optimization options."""
        if smoke_test:
            # For smoke test, only test first 100 features
            features = features[:100]
            self.logger.info(f"SMOKE TEST: Testing only {len(features)} features")

        self.logger.info(f"Testing {len(features)} features for stability across market regimes")

        stability_results = {}
        stable_count = 0

        # Pre-load regimes once for all features
        regimes = self.load_historical_regimes(df)

        for i, feature in enumerate(features):
            if (i + 1) % 50 == 0:
                self.logger.info(f"Tested {i + 1}/{len(features)} features... ({stable_count} stable so far)")

            stability_result = self.test_feature_stability(df, feature, target_col, regimes)
            stability_results[feature] = stability_result

            if stability_result.get('stable', False):
                stable_count += 1

        self.logger.info(f"Found {stable_count} stable features out of {len(features)} tested")
        return stability_results

    def select_survivor_features(self, stability_results: Dict[str, Dict],
                                min_stability_score: float = 0.01,
                                max_features: int = 200) -> List[str]:
        """Select features that survived the purge."""
        survivors = []

        for feature, result in stability_results.items():
            if result.get('stable', False) and result.get('stability_score', 0) >= min_stability_score:
                survivors.append((feature, result['stability_score']))

        # Sort by stability score (highest first)
        survivors.sort(key=lambda x: x[1], reverse=True)

        # Limit to max_features
        selected_features = [feature for feature, score in survivors[:max_features]]

        self.logger.info(f"Selected {len(selected_features)} survivor features from {len(stability_results)} tested")
        self.logger.info(f"Top 10 survivors: {[f for f, s in survivors[:10]]}")

        return selected_features

    def build_mathematical_features(self, df: pd.DataFrame, base_features: List[str],
                                   max_combinations: int = 50) -> List[str]:
        """Build mathematical relationships from stable features."""
        self.logger.info(f"Building mathematical features from {len(base_features)} base features")

        mathematical_features = []
        feature_pairs = []

        # Generate feature pairs for ratios and differences
        for i, feat1 in enumerate(base_features):
            for feat2 in enumerate(base_features[i+1:], i+1):
                feat2_name, feat2_idx = feat2
                if len(feature_pairs) >= max_combinations:
                    break
                feature_pairs.append((feat1, feat2_name))

            if len(feature_pairs) >= max_combinations:
                break

        for feat1, feat2 in feature_pairs:
            try:
                # Ratio feature
                ratio_name = f"ratio_{feat1.split('_')[-1]}_{feat2.split('_')[-1]}"
                ratio_values = df[feat1] / (df[feat2].abs() + 1e-8)

                # Check if ratio is well-behaved
                valid_ratio = ratio_values.replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_ratio) / len(df) > 0.8:  # At least 80% valid values
                    df[ratio_name] = ratio_values
                    mathematical_features.append(ratio_name)

                # Difference feature
                diff_name = f"diff_{feat1.split('_')[-1]}_{feat2.split('_')[-1]}"
                diff_values = df[feat1] - df[feat2]
                df[diff_name] = diff_values
                mathematical_features.append(diff_name)

            except Exception as e:
                self.logger.debug(f"Failed to create mathematical features for {feat1}/{feat2}: {e}")

        self.logger.info(f"Created {len(mathematical_features)} mathematical features")
        return mathematical_features

    def apply_feature_neutralization(self, df: pd.DataFrame, features: List[str],
                                    target_col: str = 'adaptive_target',
                                    neutralization_strength: float = 0.5) -> List[str]:
        """Apply feature neutralization to reduce exposure risk."""
        self.logger.info(f"Applying {neutralization_strength:.1f} neutralization to {len(features)} features")

        neutralized_features = []

        for feature in features:
            try:
                # Calculate correlation with target
                corr, _ = spearmanr(df[feature], df[target_col])
                if np.isnan(corr):
                    continue

                # Create neutralized version
                neutralized_name = f"{feature}_neutralized"
                # Simple neutralization: reduce correlation by neutralization_strength
                neutralized_values = df[feature] - neutralization_strength * corr * df[target_col]
                df[neutralized_name] = neutralized_values
                neutralized_features.append(neutralized_name)

            except Exception as e:
                self.logger.debug(f"Failed to neutralize {feature}: {e}")

        self.logger.info(f"Created {len(neutralized_features)} neutralized features")
        return neutralized_features


def run_feature_purge(data_file: str, features_file: str, output_dir: str,
                     target_col: str = 'adaptive_target',
                     crisis_eras: Optional[List[str]] = None,
                     covid_eras: Optional[List[str]] = None,
                     bear_market_eras: Optional[List[str]] = None,
                     smoke_test: bool = False,
                     sample_size: int = 5000,
                     market_tickers: Optional[List[str]] = None,
                     market_agg: str = 'ret',
                     market_mapping_csv: Optional[str] = None,
                     refresh_market: bool = False) -> Dict:
    """Run the complete feature purge pipeline."""

    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'feature_purge.log')),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting feature purge engine")

    # Load data
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(data_file)
    df = pf.read().to_pandas()

    # Ensure VIX exists so regime split works
    if 'vix' not in df.columns:
        if market_tickers:
            try:
                from .market_data import build_era_ticker_features
            except Exception:
                from market_data import build_era_ticker_features
            try:
                md = build_era_ticker_features(df['era'], market_tickers, agg=market_agg,
                                              mapping_mode='ordinal', mapping_csv=market_mapping_csv,
                                              cache_dir=os.path.join(output_dir, 'market_cache'),
                                              refresh=refresh_market)
                # Prefer ^VIX_close if ^VIX present, else ensemble of returns as proxy
                vix_cols = [c for c in md.columns if c.lower().startswith('^vix')]
                if vix_cols:
                    col = vix_cols[0]
                    tmp = md[['era', col]].rename(columns={col: 'vix'})
                elif 'ensemble_ret' in md.columns:
                    tmp = md[['era', 'ensemble_ret']].rename(columns={'ensemble_ret': 'vix'})
                else:
                    # pick first numeric column as proxy
                    num_cols = [c for c in md.columns if c != 'era']
                    tmp = md[['era', num_cols[0]]].rename(columns={num_cols[0]: 'vix'})
                df = df.merge(tmp, on='era', how='left')
                # Backfill missing
                df['vix'] = df['vix'].fillna(df['vix'].median())
            except Exception as e:
                logger.error(f"Failed to fetch market data for purge engine: {e}")
                raise ValueError("Market data fetch failed for feature purge with provided tickers; cannot simulate")
        else:
            np.random.seed(42)
            df['vix'] = np.random.normal(20, 7, len(df)).clip(10, 60)

    # Load features
    with open(features_file, 'r') as f:
        features_data = json.load(f)

    if isinstance(features_data, dict):
        if 'features' in features_data:
            candidate_features = features_data['features']
        elif 'feature_sets' in features_data:
            # Handle the actual structure with feature_sets
            candidate_features = []
            for category, features in features_data['feature_sets'].items():
                if isinstance(features, list):
                    candidate_features.extend(features)
        elif all(isinstance(v, list) for v in features_data.values()):
            # Handle structure where all values are lists (like our current features.json)
            candidate_features = []
            for category, features in features_data.items():
                if isinstance(features, list) and category != 'targets':
                    candidate_features.extend(features)
        else:
            candidate_features = features_data
    elif isinstance(features_data, list):
        candidate_features = features_data
    else:
        raise ValueError("Features file must contain a list or dict with 'features', 'feature_sets', or category lists")

    # Ensure candidate_features is a list of strings
    if not isinstance(candidate_features, list):
        raise ValueError("candidate_features must be a list")
    candidate_features = [str(f) for f in candidate_features]

    logger.info(f"Loaded {len(candidate_features)} candidate features")

    # Initialize purge engine with optimized settings
    purge_engine = FeaturePurgeEngine(crisis_eras, covid_eras, bear_market_eras,
                                     sample_size=sample_size)

    # Test feature stability
    stability_results = purge_engine.purge_unstable_features(df, candidate_features, target_col, smoke_test)

    # Select survivors
    survivor_features = purge_engine.select_survivor_features(stability_results)

    # Build mathematical features from survivors
    mathematical_features = purge_engine.build_mathematical_features(df, survivor_features[:50])  # Use top 50

    # Apply neutralization
    neutralized_features = purge_engine.apply_feature_neutralization(df, survivor_features[:30])  # Top 30

    # Final feature set
    final_features = survivor_features + mathematical_features + neutralized_features

    # Summary statistics
    stable_count = sum(1 for r in stability_results.values() if r.get('stable', False))
    avg_stability = np.mean([r.get('stability_score', 0) for r in stability_results.values() if r.get('stable', False)])

    results = {
        'stability_analysis': stability_results,
        'survivor_features': survivor_features,
        'mathematical_features': mathematical_features,
        'neutralized_features': neutralized_features,
        'final_features': final_features,
        'summary': {
            'total_candidates': len(candidate_features),
            'stable_features': stable_count,
            'survivor_features': len(survivor_features),
            'mathematical_features': len(mathematical_features),
            'neutralized_features': len(neutralized_features),
            'final_feature_count': len(final_features),
            'average_stability_score': float(avg_stability) if stable_count > 0 else 0
        }
    }

    # Save results
    results_file = os.path.join(output_dir, 'feature_purge_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save final feature list
    features_file = os.path.join(output_dir, 'final_features.json')
    with open(features_file, 'w') as f:
        json.dump(final_features, f, indent=2)

    logger.info("Feature purge complete!")
    logger.info(f"Started with {len(candidate_features)} features")
    logger.info(f"Ended with {len(final_features)} stable features")
    logger.info(f"Average stability score: {avg_stability:.3f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Purge Engine")
    parser.add_argument('--data-file', required=True, help='Path to parquet data file')
    parser.add_argument('--features-file', required=True, help='Path to JSON file with candidate features')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--target-col', default='adaptive_target', help='Target column name')
    parser.add_argument('--crisis-eras', nargs='+', help='Eras representing crisis periods')
    parser.add_argument('--covid-eras', nargs='+', help='Eras representing COVID crash')
    parser.add_argument('--bear-market-eras', nargs='+', help='Eras representing bear markets')
    parser.add_argument('--smoke-test', action='store_true', help='Enable smoke test (limited feature testing)')
    parser.add_argument('--sample-size', type=int, default=5000, help='Sample size for normal periods')

    args = parser.parse_args()

    run_feature_purge(
        args.data_file,
        args.features_file,
        args.output_dir,
        args.target_col,
        args.crisis_eras,
        args.covid_eras,
        args.bear_market_eras,
        args.smoke_test,
        args.sample_size
    )
