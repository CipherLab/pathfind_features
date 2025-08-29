"""
Integration Tests for Numerai Pipeline

These tests validate the integration between different pipeline components
and ensure they work together correctly with synthetic data.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path

# Import pipeline modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from validation_framework import EraAwareCrossValidator, categorize_eras_by_vix, calculate_realistic_sharpe
from feature_stability_engine import FeatureStabilityEngine


class TestPipelineIntegration:
    """Test integration between pipeline components."""

    @pytest.fixture
    def synthetic_dataset(self, tmp_path):
        """Create a synthetic dataset for integration testing."""
        np.random.seed(42)
        n_rows = 10000
        n_features = 50

        # Create eras (simulate time series)
        eras = np.repeat(range(1, 101), n_rows // 100)

        # Create features with some patterns
        data = {'era': eras}
        for i in range(n_features):
            if i < 10:
                # Some features correlated with era (trend)
                data[f'feature_{i}'] = np.random.randn(n_rows) + eras * 0.01
            elif i < 20:
                # Some features with regime-dependent behavior
                regime_factor = (eras > 50).astype(int)
                data[f'feature_{i}'] = np.random.randn(n_rows) + regime_factor * 2
            else:
                # Random features
                data[f'feature_{i}'] = np.random.randn(n_rows)

        # Create targets
        data['target_0'] = np.random.randn(n_rows) * 0.1 + eras * 0.001
        data['adaptive_target'] = data['target_0'] + np.random.randn(n_rows) * 0.05

        df = pd.DataFrame(data)

        # Create mock VIX data
        vix_data = pd.DataFrame({
            'era': range(1, 101),
            'vix': 20 + np.sin(np.arange(100) * 0.1) * 10 + np.random.randn(100) * 5
        })

        return {
            'dataframe': df,
            'vix_data': vix_data,
            'features_file': tmp_path / 'features.json',
            'data_file': tmp_path / 'data.parquet'
        }

    def test_validation_and_stability_integration(self, synthetic_dataset):
        """Test integration between validation framework and feature stability."""
        df = synthetic_dataset['dataframe']
        vix_data = synthetic_dataset['vix_data']

        # Step 1: Era-aware cross validation
        cv = EraAwareCrossValidator(n_splits=3, gap_eras=5)
        splits = cv.split(df)

        assert len(splits) == 3

        # Step 2: For each split, test feature stability
        for i, (train_idx, val_idx) in enumerate(splits):
            train_data = df.loc[train_idx].copy()

            # Add regime information
            train_data['vix_regime'] = categorize_eras_by_vix(train_data['era'], vix_data)

            # Select features for stability analysis
            feature_cols = [col for col in df.columns if col.startswith('feature_')][:20]

            # Test feature stability
            engine = FeatureStabilityEngine()
            stability_results = engine.evaluate_feature_stability(
                train_data, feature_cols, target_col='adaptive_target'
            )

            # Should have results for all features
            assert len(stability_results) == len(feature_cols)

            # Should have some stable features
            stable_count = sum(1 for result in stability_results.values()
                             if result.get('stable', False))
            assert stable_count > 0, f"No stable features found in split {i}"

    def test_regime_aware_feature_selection(self, synthetic_dataset):
        """Test that feature stability respects regime boundaries."""
        df = synthetic_dataset['dataframe']
        vix_data = synthetic_dataset['vix_data']

        # Add regime information
        df['vix_regime'] = categorize_eras_by_vix(df['era'], vix_data)

        # Get regime distribution
        regime_counts = df['vix_regime'].value_counts()

        # Should have multiple regimes
        assert len(regime_counts) >= 2

        # Test feature stability across regimes
        feature_cols = [col for col in df.columns if col.startswith('feature_')][:10]
        engine = FeatureStabilityEngine()

        stability_results = engine.evaluate_feature_stability(
            df, feature_cols, target_col='adaptive_target'
        )

        # Check that stability analysis considers regime information
        for feature, result in stability_results.items():
            assert 'regime_correlations' in result
            assert len(result['regime_correlations']) > 0

    def test_performance_regression_detection(self, synthetic_dataset):
        """Test that we can detect performance regressions across splits."""
        df = synthetic_dataset['dataframe']
        vix_data = synthetic_dataset['vix_data']

        cv = EraAwareCrossValidator(n_splits=5, gap_eras=3)
        splits = cv.split(df)

        performance_scores = []

        for train_idx, val_idx in splits:
            train_data = df.loc[train_idx]
            val_data = df.loc[val_idx]

            # Simple performance metric (correlation with target)
            feature_cols = [col for col in df.columns if col.startswith('feature_')][:5]

            split_scores = []
            for feature in feature_cols:
                corr = train_data[feature].corr(train_data['adaptive_target'])
                if not pd.isna(corr):
                    split_scores.append(abs(corr))

            if split_scores:
                avg_score = np.mean(split_scores)
                performance_scores.append(avg_score)

        # Should have consistent performance across splits
        if len(performance_scores) > 1:
            score_std = np.std(performance_scores)
            score_mean = np.mean(performance_scores)

            # Performance shouldn't vary too much (coefficient of variation < 50%)
            cv_score = score_std / score_mean if score_mean > 0 else 0
            assert cv_score < 0.5, f"High performance variation: CV = {cv_score:.2f}"

    def test_memory_efficiency_integration(self, synthetic_dataset):
        """Test memory efficiency of integrated pipeline components."""
        df = synthetic_dataset['dataframe']
        vix_data = synthetic_dataset['vix_data']

        # Test memory usage during integrated workflow
        initial_memory = df.memory_usage(deep=True).sum()

        # Add regime information
        df_with_regime = df.copy()
        df_with_regime['vix_regime'] = categorize_eras_by_vix(df['era'], vix_data)

        # Test cross-validation
        cv = EraAwareCrossValidator(n_splits=3, gap_eras=5)
        splits = cv.split(df_with_regime)

        # Process splits
        for train_idx, val_idx in splits:
            train_data = df_with_regime.loc[train_idx]

            # Feature stability analysis
            feature_cols = [col for col in df.columns if col.startswith('feature_')][:10]
            engine = FeatureStabilityEngine()
            stability_results = engine.evaluate_feature_stability(
                train_data, feature_cols, target_col='adaptive_target'
            )

        # Check memory hasn't grown excessively
        final_memory = df_with_regime.memory_usage(deep=True).sum()
        memory_growth = (final_memory - initial_memory) / initial_memory

        assert memory_growth < 2.0, f"Memory growth too high: {memory_growth:.2f}x"


class TestDataQualityIntegration:
    """Test data quality checks integrated with pipeline components."""

    def test_era_completeness_check(self, synthetic_dataset):
        """Test that era data is complete and properly ordered."""
        df = synthetic_dataset['dataframe']

        # Check era column exists and is properly formatted
        assert 'era' in df.columns
        assert df['era'].dtype in [np.int64, np.int32, int]

        # Check eras are sequential and start from 1
        unique_eras = sorted(df['era'].unique())
        expected_eras = list(range(1, len(unique_eras) + 1))
        assert unique_eras == expected_eras

        # Check sufficient data per era
        era_counts = df['era'].value_counts()
        min_era_size = era_counts.min()
        assert min_era_size > 10, f"Era {era_counts.idxmin()} has only {min_era_size} samples"

    def test_target_distribution_check(self, synthetic_dataset):
        """Test target variable distributions are reasonable."""
        df = synthetic_dataset['dataframe']

        target_cols = [col for col in df.columns if col.startswith('target_')]

        for target_col in target_cols:
            target_values = df[target_col]

            # Check for reasonable value range
            assert target_values.min() > -10, f"{target_col} has suspiciously low values"
            assert target_values.max() < 10, f"{target_col} has suspiciously high values"

            # Check for reasonable standard deviation
            target_std = target_values.std()
            assert 0.01 < target_std < 1.0, f"{target_col} has unusual std: {target_std}"

            # Check for NaN values
            nan_count = target_values.isna().sum()
            assert nan_count == 0, f"{target_col} has {nan_count} NaN values"

    def test_feature_correlation_check(self, synthetic_dataset):
        """Test feature correlations are within reasonable bounds."""
        df = synthetic_dataset['dataframe']

        feature_cols = [col for col in df.columns if col.startswith('feature_')][:20]
        target_col = 'adaptive_target'

        correlations = {}
        for feature in feature_cols:
            corr = df[feature].corr(df[target_col])
            if not pd.isna(corr):
                correlations[feature] = abs(corr)

        if correlations:
            # Should have some correlation with target
            max_corr = max(correlations.values())
            assert max_corr > 0.01, f"No features correlate with target (max corr: {max_corr})"

            # Should not have perfect correlation (data leakage)
            assert max_corr < 0.99, f"Suspiciously high correlation: {max_corr}"
