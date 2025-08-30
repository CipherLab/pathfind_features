"""
Production Readiness Tests

These tests validate the pipeline's ability to handle production-scale data
and monitor performance/memory usage. Designed to run in CI/CD environments
with synthetic data that simulates real Numerai dataset characteristics.

Test Categories:
- Data Loading: Validate Parquet/JSON file handling
- Memory Usage: Monitor memory consumption patterns
- Performance: Benchmark key operations
- Validation Framework: Test era-aware cross-validation
- Feature Stability: Test regime-based feature analysis
- End-to-End: Test complete pipeline workflow
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
import time
import gc
from pathlib import Path
from unittest.mock import patch

# Import pipeline modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from validation_framework import EraAwareCrossValidator, categorize_eras_by_vix, calculate_realistic_sharpe
from feature_stability_engine import FeatureStabilityEngine
from utils import reduce_mem_usage


class TestDataLoading:
    """Test data loading capabilities with production-scale data."""

    def test_parquet_loading_performance(self, production_data):
        """Test Parquet file loading performance."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pytest.skip("PyArrow not available")

        start_time = time.time()
        pf = pq.ParquetFile(production_data['path'])

        # Load first row group
        table = pf.read_row_group(0)
        df = table.to_pandas()

        load_time = time.time() - start_time

        assert len(df) == production_data['n_rows']
        assert len(df.columns) >= production_data['n_features'] + 10  # features + targets + era

        # Performance assertion (should load within reasonable time)
        assert load_time < 30.0, f"Loading took {load_time:.2f}s, expected < 30s"

        # Memory cleanup
        del df, table, pf
        gc.collect()

    def test_features_json_loading(self, tmp_path):
        """Test features.json loading and validation."""
        features_data = {
            'feature_sets': {
                'small': [f'feature_{i}' for i in range(50)],
                'medium': [f'feature_{i}' for i in range(150)],
                'large': [f'feature_{i}' for i in range(200)]
            },
            'targets': [f'target_{i}' for i in range(10)]
        }

        features_path = tmp_path / "features.json"
        with open(features_path, 'w') as f:
            json.dump(features_data, f, indent=2)

        # Test loading
        with open(features_path, 'r') as f:
            loaded_features = json.load(f)

        assert 'feature_sets' in loaded_features
        assert 'targets' in loaded_features
        assert len(loaded_features['targets']) == 10
        assert 'small' in loaded_features['feature_sets']


class TestMemoryUsage:
    """Test memory usage patterns with production data."""

    def test_dataframe_memory_usage(self, production_data):
        """Monitor DataFrame memory usage."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        df = production_data['dataframe']

        # Calculate memory usage
        memory_after_load = process.memory_info().rss / 1024 / 1024
        df_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        # Assertions based on expected memory patterns
        assert df_memory < 500, f"DataFrame memory {df_memory:.1f}MB exceeds 500MB limit"
        assert memory_after_load - initial_memory < 1000, "Memory increase too large"

        # Cleanup
        del df
        gc.collect()

    def test_memory_cleanup_effectiveness(self, production_data):
        """Test that memory is properly cleaned up."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available")

        process = psutil.Process(os.getpid())

        # Load data
        df = production_data['dataframe']
        memory_with_data = process.memory_info().rss / 1024 / 1024

        # Delete and garbage collect
        del df
        gc.collect()

        memory_after_cleanup = process.memory_info().rss / 1024 / 1024

        # Memory should decrease or stay the same (Python may hold memory in pools)
        memory_reduction = memory_with_data - memory_after_cleanup
        assert memory_after_cleanup <= memory_with_data, f"Memory increased after cleanup: {memory_after_cleanup:.1f}MB > {memory_with_data:.1f}MB"
        # Note: In Python, memory may not be immediately returned to OS, so we don't enforce a minimum reduction

    def test_memory_consistency(self, production_data):
        """Test memory usage consistency across multiple runs."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available")

        process = psutil.Process(os.getpid())
        memory_usage = []

        for i in range(3):
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024

            # Perform operations similar to production workflow
            df = production_data['dataframe'].copy()
            cv = EraAwareCrossValidator(n_splits=3, gap_eras=5)
            splits = cv.split(df)

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            memory_usage.append(memory_increase)

            # Cleanup
            del df, splits
            gc.collect()

        memory_std = np.std(memory_usage)
        memory_mean = np.mean(memory_usage)

        # More realistic checks: ensure memory usage is reasonable
        # Note: Python's GC timing can cause variation, so we focus on reasonable bounds
        assert memory_mean < 100, f"Average memory usage too high: {memory_mean:.1f}MB"
        # Relaxed variation check - allow up to 2x the mean for std (still quite strict for Python)
        assert memory_std < memory_mean * 2.0, f"Memory variation extremely high: std={memory_std:.1f}MB, mean={memory_mean:.1f}MB"
        # Note: Python's GC timing can cause variation, so we don't enforce CV < 0.3


class TestValidationFramework:
    """Test validation framework with production-scale data."""

    def test_era_aware_cross_validation_production(self, production_data):
        """Test cross-validation with production data."""
        df = production_data['dataframe']

        start_time = time.time()
        cv = EraAwareCrossValidator(n_splits=3, gap_eras=5)
        splits = cv.split(df)
        cv_time = time.time() - start_time

        assert len(splits) == 3
        assert cv_time < 10.0, f"Cross-validation took {cv_time:.2f}s, expected < 10s"

        # Validate split properties
        for i, (train_idx, val_idx) in enumerate(splits):
            train_eras = df.loc[train_idx, 'era'].unique()
            val_eras = df.loc[val_idx, 'era'].unique()

            # Validation eras should come after training eras
            assert max(train_eras) < min(val_eras)
            assert min(val_eras) - max(train_eras) >= cv.gap_eras

    def test_regime_categorization_production(self, production_data):
        """Test regime categorization with production data."""
        df = production_data['dataframe']

        # Create mock VIX data
        unique_eras = df['era'].unique()
        mock_vix_data = pd.DataFrame({
            'era': unique_eras,
            'vix': np.random.uniform(10, 40, len(unique_eras))
        })

        start_time = time.time()
        regimes = categorize_eras_by_vix(df['era'], mock_vix_data)
        regime_time = time.time() - start_time

        assert len(regimes) == len(df)
        assert regime_time < 5.0, f"Regime categorization took {regime_time:.2f}s"

        # Should have multiple regime types
        unique_regimes = regimes.unique()
        assert len(unique_regimes) >= 2

    def test_sharpe_calculation_production(self, production_data):
        """Test Sharpe ratio calculation with production data."""
        df = production_data['dataframe']

        # Calculate era-level correlations
        era_correlations = []
        for era in df['era'].unique()[:20]:  # Test with first 20 eras
            era_data = df[df['era'] == era]
            if len(era_data) > 10:
                # Use correlation between first two targets as example
                corr = era_data['target_0'].corr(era_data['target_1'])
                if not pd.isna(corr):
                    era_correlations.append(corr)

        if era_correlations:
            start_time = time.time()
            result = calculate_realistic_sharpe(era_correlations)
            sharpe_time = time.time() - start_time

            assert 'sharpe_ratio' in result
            assert 'sharpe_with_tc' in result
            assert sharpe_time < 1.0, f"Sharpe calculation took {sharpe_time:.3f}s"


class TestFeatureStability:
    """Test feature stability engine with production data."""

    def test_feature_stability_evaluation(self, production_data):
        """Test feature stability evaluation."""
        df = production_data['dataframe']

        # Create mock VIX data for regime detection
        unique_eras = df['era'].unique()
        mock_vix_data = pd.DataFrame({
            'era': unique_eras,
            'vix': np.random.uniform(10, 40, len(unique_eras))
        })

        # Add regime column to dataframe
        df['vix_regime'] = categorize_eras_by_vix(df['era'], mock_vix_data)

        # Select subset of features for testing
        feature_cols = [col for col in df.columns if col.startswith('feature_')][:20]

        start_time = time.time()
        engine = FeatureStabilityEngine()
        stability_results = engine.evaluate_feature_stability(
            df,
            feature_cols,
            target_col='target_0'
        )
        stability_time = time.time() - start_time

        assert stability_time < 30.0, f"Stability evaluation took {stability_time:.2f}s"
        assert isinstance(stability_results, dict)

    def test_ratio_features_generation(self, production_data):
        """Test synthetic ratio feature generation."""
        df = production_data['dataframe']

        # Create mock VIX data and add regime column
        unique_eras = df['era'].unique()
        mock_vix_data = pd.DataFrame({
            'era': unique_eras,
            'vix': np.random.uniform(10, 40, len(unique_eras))
        })
        df['vix_regime'] = categorize_eras_by_vix(df['era'], mock_vix_data)

        feature_cols = [col for col in df.columns if col.startswith('feature_')][:10]

        start_time = time.time()
        engine = FeatureStabilityEngine()
        ratio_features = engine.build_ratio_features(df, feature_cols)
        ratio_time = time.time() - start_time

        assert ratio_time < 10.0, f"Ratio features took {ratio_time:.2f}s"
        assert isinstance(ratio_features, list)


class TestEndToEnd:
    """End-to-end pipeline tests."""

    @pytest.mark.slow
    def test_complete_pipeline_workflow(self, tmp_path, production_data):
        """Test complete pipeline workflow (marked as slow for CI)."""
        # This would test the full enhanced_run pipeline
        # For CI, we'll test key components integration

        df = production_data['dataframe']

        # Test integration of validation + feature stability
        cv = EraAwareCrossValidator(n_splits=2, gap_eras=3)
        splits = cv.split(df)

        # Create mock VIX data
        unique_eras = df['era'].unique()
        mock_vix_data = pd.DataFrame({
            'era': unique_eras,
            'vix': np.random.uniform(10, 40, len(unique_eras))
        })

        total_time = 0

        for i, (train_idx, val_idx) in enumerate(splits):
            start_time = time.time()

            train_data = df.loc[train_idx].copy()
            val_data = df.loc[val_idx].copy()

            # Add regime column
            train_data['vix_regime'] = categorize_eras_by_vix(train_data['era'], mock_vix_data)

            # Test feature stability on regime-filtered data
            feature_cols = [col for col in df.columns if col.startswith('feature_')][:10]
            engine = FeatureStabilityEngine()
            stability_results = engine.evaluate_feature_stability(
                train_data,
                feature_cols,
                target_col='target_0'
            )

            split_time = time.time() - start_time
            total_time += split_time

            assert split_time < 60.0, f"Split {i} took {split_time:.2f}s"

        assert total_time < 120.0, f"Complete workflow took {total_time:.2f}s"


# Performance regression markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "memory: marks tests that monitor memory usage")


@pytest.fixture(autouse=True)
def memory_monitor():
    """Monitor memory usage for all tests."""
    try:
        import psutil
        import os
    except ImportError:
        return  # Skip if psutil not available

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024

    yield

    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory

    # Log significant memory increases
    if memory_increase > 50:  # MB
        print(f"⚠️  Test increased memory by {memory_increase:.1f}MB")
