import os
import sys
import time
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import json

# Ensure project root and original/ are on sys.path for imports in tests
ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "original"
for p in [ROOT, ORIG]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Optionally, set environment variables used by some modules
os.environ.setdefault("PYTHONHASHSEED", "0")


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a synthetic dataset for integration testing."""
    np.random.seed(42)
    n_rows = 10000
    n_features = 50

    # Create eras (simulate time series)
    eras = np.repeat(range(1, 101), n_rows // 100)

    data = {'era': eras}

    # Create features with some patterns
    for i in range(n_features):
        if i < 10:
            # Some features correlated with era (trend) - these should be stable
            base_corr = 0.1 + (i * 0.05)  # Increasing correlation strength
            data[f'feature_{i}'] = np.random.randn(n_rows) + eras * base_corr
        elif i < 20:
            # Some features with regime-dependent behavior but consistent sign
            regime_factor = (eras > 50).astype(int) * 0.5  # Consistent positive effect
            data[f'feature_{i}'] = np.random.randn(n_rows) + regime_factor
        else:
            # Random features - may or may not be stable
            data[f'feature_{i}'] = np.random.randn(n_rows)

    # Create targets with some correlation to stable features
    data['target_0'] = np.random.randn(n_rows) * 0.1 + eras * 0.02
    data['adaptive_target'] = data['target_0'] + np.random.randn(n_rows) * 0.05

    df = pd.DataFrame(data)

    # Create mock VIX data with realistic regime patterns
    vix_data = pd.DataFrame({
        'era': range(1, 101),
        'vix': 20 + np.sin(np.arange(100) * 0.2) * 8 + np.random.randn(100) * 3
    })

    return {
        'dataframe': df,
        'vix_data': vix_data,
        'features_file': tmp_path / 'features.json',
        'data_file': tmp_path / 'data.parquet'
    }


@pytest.fixture
def benchmark_dataset():
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


@pytest.fixture
def production_data(tmp_path):
    """Generate production-scale synthetic data."""
    np.random.seed(42)
    n_rows = 50000  # Production subset
    n_features = 200

    data = {
        'era': np.repeat(range(1, 251), n_rows // 250),  # 250 eras
    }

    # Simulate different feature types
    for i in range(n_features):
        if i % 4 == 0:
            data[f'feature_{i}'] = np.random.randn(n_rows)
        elif i % 4 == 1:
            data[f'feature_{i}'] = np.random.exponential(1, n_rows)
        elif i % 4 == 2:
            data[f'feature_{i}'] = np.random.uniform(-1, 1, n_rows)
        else:
            data[f'feature_{i}'] = np.random.beta(2, 2, n_rows)

    # Add targets
    for i in range(10):
        data[f'target_{i}'] = np.random.randn(n_rows) * 0.05

    df = pd.DataFrame(data)

    # Save to parquet
    parquet_path = tmp_path / "train_production.parquet"
    df.to_parquet(parquet_path, index=False)

    return {
        'dataframe': df,
        'path': parquet_path,
        'n_rows': n_rows,
        'n_features': n_features
    }


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


@pytest.fixture(autouse=True)
def memory_monitor():
    """Monitor memory usage for all tests."""
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        yield

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # Log significant memory increases
        if memory_increase > 50:  # MB
            print(f"⚠️  Test increased memory by {memory_increase:.1f}MB")
    except ImportError:
        # If psutil is unavailable, still yield control so tests proceed
        yield
