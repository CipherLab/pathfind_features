"""
Performance Tests for Numerai Pipeline

These tests monitor performance regressions and benchmark key operations.
They use synthetic data scaled to production sizes to detect performance issues.
"""

import pytest
import pandas as pd
import numpy as np
import time
import gc
from pathlib import Path

# Import pipeline modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from validation_framework import EraAwareCrossValidator, categorize_eras_by_vix, calculate_realistic_sharpe
from feature_stability_engine import FeatureStabilityEngine


class TestPerformanceBenchmarks:
    """Performance benchmarks for key pipeline operations."""

    @pytest.fixture
    def benchmark_dataset(self):
        """Create dataset sized for performance benchmarking."""
        np.random.seed(42)
        n_rows = 25000  # Medium size for performance testing
        n_features = 100

        # Create eras
        eras = np.repeat(range(1, 126), n_rows // 125)  # 125 eras

        data = {'era': eras}

        # Create features
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.randn(n_rows)

        # Create targets
        for i in range(5):
            data[f'target_{i}'] = np.random.randn(n_rows) * 0.05

        df = pd.DataFrame(data)

        # Create VIX data
        vix_data = pd.DataFrame({
            'era': range(1, 126),
            'vix': np.random.uniform(10, 40, 125)
        })

        return {'dataframe': df, 'vix_data': vix_data}

    def test_cross_validation_performance(self, benchmark_dataset, benchmark):
        """Benchmark era-aware cross-validation performance."""
        df = benchmark_dataset['dataframe']

        def cv_operation():
            cv = EraAwareCrossValidator(n_splits=5, gap_eras=5)
            splits = cv.split(df)
            return len(splits)

        result = benchmark(cv_operation)
        assert result == 5

        # Performance assertion (should complete within time limit)
        assert benchmark.last_duration < 5.0, f"CV too slow: {benchmark.last_duration:.2f}s"

    def test_regime_categorization_performance(self, benchmark_dataset, benchmark):
        """Benchmark regime categorization performance."""
        df = benchmark_dataset['dataframe']
        vix_data = benchmark_dataset['vix_data']

        def regime_operation():
            regimes = categorize_eras_by_vix(df['era'], vix_data)
            return len(regimes.unique())

        result = benchmark(regime_operation)
        assert result >= 2  # Should have multiple regimes

        assert benchmark.last_duration < 2.0, f"Regime categorization too slow: {benchmark.last_duration:.2f}s"

    def test_feature_stability_performance(self, benchmark_dataset, benchmark):
        """Benchmark feature stability evaluation performance."""
        df = benchmark_dataset['dataframe']
        vix_data = benchmark_dataset['vix_data']

        # Add regime information
        df = df.copy()
        df['vix_regime'] = categorize_eras_by_vix(df['era'], vix_data)

        feature_cols = [col for col in df.columns if col.startswith('feature_')][:20]

        def stability_operation():
            engine = FeatureStabilityEngine()
            results = engine.evaluate_feature_stability(df, feature_cols, target_col='target_0')
            return len(results)

        result = benchmark(stability_operation)
        assert result == len(feature_cols)

        assert benchmark.last_duration < 15.0, f"Feature stability too slow: {benchmark.last_duration:.2f}s"

    def test_memory_efficiency_during_operations(self, benchmark_dataset):
        """Test memory efficiency during benchmark operations."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        df = benchmark_dataset['dataframe']
        vix_data = benchmark_dataset['vix_data']

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform operations
        df = df.copy()
        df['vix_regime'] = categorize_eras_by_vix(df['era'], vix_data)

        cv = EraAwareCrossValidator(n_splits=3, gap_eras=5)
        splits = cv.split(df)

        feature_cols = [col for col in df.columns if col.startswith('feature_')][:10]
        engine = FeatureStabilityEngine()

        for train_idx, val_idx in splits:
            train_data = df.loc[train_idx]
            stability_results = engine.evaluate_feature_stability(
                train_data, feature_cols, target_col='target_0'
            )

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 500, f"Memory increase too high: {memory_increase:.1f}MB"

        # Cleanup
        del df, splits
        gc.collect()


class TestScalabilityTests:
    """Test scalability with different data sizes."""

    @pytest.mark.parametrize("n_rows,n_features", [
        (5000, 50),    # Small
        (15000, 100),  # Medium
        (30000, 150),  # Large
    ])
    def test_scalability_cross_validation(self, n_rows, n_features):
        """Test cross-validation scales reasonably with data size."""
        np.random.seed(42)

        # Create test data
        eras = np.repeat(range(1, n_rows//100 + 2), 100)[:n_rows]
        data = {'era': eras}

        for i in range(n_features):
            data[f'feature_{i}'] = np.random.randn(n_rows)

        df = pd.DataFrame(data)

        # Time the operation
        start_time = time.time()
        cv = EraAwareCrossValidator(n_splits=3, gap_eras=5)
        splits = cv.split(df)
        duration = time.time() - start_time

        # Should complete and scale reasonably
        assert len(splits) == 3

        # Performance should scale roughly linearly with data size
        # Base case: 5000 rows should take < 1s
        expected_max_time = (n_rows / 5000) * 1.0
        assert duration < expected_max_time * 2, f"Too slow for {n_rows} rows: {duration:.2f}s"

    @pytest.mark.parametrize("n_features", [25, 50, 100])
    def test_scalability_feature_stability(self, n_features):
        """Test feature stability scales with number of features."""
        np.random.seed(42)
        n_rows = 10000

        # Create test data
        eras = np.repeat(range(1, 51), n_rows // 50)
        data = {'era': eras}

        for i in range(n_features):
            data[f'feature_{i}'] = np.random.randn(n_rows)

        data['target_0'] = np.random.randn(n_rows) * 0.1
        df = pd.DataFrame(data)

        # Add regime info
        vix_data = pd.DataFrame({
            'era': range(1, 51),
            'vix': np.random.uniform(10, 40, 50)
        })
        df['vix_regime'] = categorize_eras_by_vix(df['era'], vix_data)

        feature_cols = [f'feature_{i}' for i in range(n_features)]

        # Time the operation
        start_time = time.time()
        engine = FeatureStabilityEngine()
        results = engine.evaluate_feature_stability(df, feature_cols, target_col='target_0')
        duration = time.time() - start_time

        # Should complete and return results for all features
        assert len(results) == n_features

        # Performance should scale reasonably with feature count
        expected_max_time = (n_features / 25) * 2.0  # Base: 25 features in ~2s
        assert duration < expected_max_time * 2, f"Too slow for {n_features} features: {duration:.2f}s"


class TestPerformanceRegressionDetection:
    """Tests to detect performance regressions over time."""

    def test_operation_consistency(self, benchmark_dataset):
        """Test that operations produce consistent timing patterns."""
        df = benchmark_dataset['dataframe']
        vix_data = benchmark_dataset['vix_data']

        # Run the same operation multiple times
        times = []

        for _ in range(3):
            start_time = time.time()

            # Same operation each time
            df_copy = df.copy()
            df_copy['vix_regime'] = categorize_eras_by_vix(df['era'], vix_data)

            cv = EraAwareCrossValidator(n_splits=3, gap_eras=5)
            splits = cv.split(df_copy)

            duration = time.time() - start_time
            times.append(duration)

        # Calculate coefficient of variation
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv_time = std_time / mean_time if mean_time > 0 else 0

        # Should have consistent timing (CV < 20%)
        assert cv_time < 0.2, f"Inconsistent timing: CV = {cv_time:.2f}"

    def test_memory_consistency(self, benchmark_dataset):
        """Test memory usage consistency across runs."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available")

        df = benchmark_dataset['dataframe']
        vix_data = benchmark_dataset['vix_data']

        process = psutil.Process(os.getpid())

        memory_usage = []

        for _ in range(3):
            # Clean up before each run
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024

            # Perform operations
            df_copy = df.copy()
            df_copy['vix_regime'] = categorize_eras_by_vix(df['era'], vix_data)

            cv = EraAwareCrossValidator(n_splits=3, gap_eras=5)
            splits = cv.split(df_copy)

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            memory_usage.append(memory_increase)

            # Clean up
            del df_copy, splits
            gc.collect()

        # Memory usage should be consistent
        memory_std = np.std(memory_usage)
        memory_mean = np.mean(memory_usage)
        memory_cv = memory_std / memory_mean if memory_mean > 0 else 0

        assert memory_cv < 0.3, f"Inconsistent memory usage: CV = {memory_cv:.2f}"


# Custom benchmark fixture for pytest-benchmark
@pytest.fixture
def benchmark():
    """Simple benchmark fixture when pytest-benchmark is not available."""
    class SimpleBenchmark:
        def __init__(self):
            self.last_duration = 0.0

        def __call__(self, func):
            start_time = time.time()
            result = func()
            self.last_duration = time.time() - start_time
            return result

    return SimpleBenchmark()
