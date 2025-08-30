# Numerai Pipeline CI/CD Strategy

This document outlines the comprehensive CI/CD strategy for the Numerai pipeline, including production readiness testing and regression detection.

## Test Categories

### 1. Unit Tests (`test_*.py`)
- **Purpose**: Test individual functions and classes in isolation
- **Data**: Mock/synthetic data
- **Frequency**: Every commit
- **Duration**: < 30 seconds
- **Coverage**: Core functionality

### 2. Integration Tests (`test_integration.py`)
- **Purpose**: Test component interactions and data flow
- **Data**: Synthetic data simulating production characteristics
- **Frequency**: Every commit
- **Duration**: < 2 minutes
- **Coverage**: Pipeline workflow integration

### 3. Performance Tests (`test_performance.py`)
- **Purpose**: Detect performance regressions and scalability issues
- **Data**: Scaled synthetic data (10k-30k rows)
- **Frequency**: Every commit
- **Duration**: < 5 minutes
- **Coverage**: Operation timing and memory usage

### 4. Production Readiness Tests (`test_production_readiness.py`)
- **Purpose**: Validate production-scale data handling
- **Data**: Synthetic data scaled to production size (50k+ rows, 200+ features)
- **Frequency**: Manual trigger or weekly schedule
- **Duration**: < 15 minutes
- **Coverage**: Memory usage, data loading, end-to-end workflow

## CI/CD Pipeline Structure

### GitHub Actions Workflow (`.github/workflows/ci-cd.yml`)

```yaml
jobs:
  unit-tests:           # Fast feedback on every commit
  integration-tests:    # Component integration validation
  performance-tests:    # Performance regression detection
  production-readiness: # Manual/production validation
  scheduled-production: # Weekly comprehensive testing
```

### Test Execution Strategy

#### Development (Every Commit)
```bash
# Run all fast tests
pytest pipeline/tests/ -m "not slow" --durations=10

# Run with coverage
pytest pipeline/tests/ --cov=pipeline --cov-report=xml
```

#### Pre-Release
```bash
# Run all tests including slow ones
pytest pipeline/tests/ --durations=0

# Run production readiness tests
pytest pipeline/tests/test_production_readiness.py -v
```

#### Production Deployment
```bash
# Full pipeline validation with real data characteristics
pytest pipeline/tests/test_production_readiness.py -v -s
```

## Data Strategy for CI/CD

### Handling Large Data Files (Git LFS)

The Numerai dataset includes large Parquet files:
- `train.parquet`: ~2.3GB
- `validation.parquet`: ~3.3GB
- `features.json`: ~284KB

#### Git LFS Setup
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.parquet"
git lfs track "*.pq"

# Add .gitattributes
echo "*.parquet filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
echo "*.pq filter=lfs diff=lfs merge=lfs -text" >> .gitattributes

# Commit changes
git add .gitattributes
git commit -m "Add Git LFS tracking for large data files"
```

#### Repository Structure for Large Files
```
numerai_minimal/
├── data/                    # Git LFS tracked (not in repo)
│   ├── train.parquet       # 2.3GB - LFS
│   ├── validation.parquet  # 3.3GB - LFS
│   └── features.json       # 284KB - regular Git
├── test_data/              # Synthetic data for CI/CD
│   ├── train_small.parquet
│   ├── train_medium.parquet
│   └── features.json
└── pipeline/
    └── tests/
```

#### CI/CD Data Strategy
1. **Development**: Use synthetic `test_data/` for all automated tests
2. **Local Testing**: Use local `data/` directory with real files
3. **Production**: Download real data or use mounted volumes
4. **Git LFS**: Track only essential large files, use synthetic data for most tests

### Synthetic Data Generation
- **Purpose**: Avoid dependency on real Numerai data files in CI
- **Characteristics**: Match real data structure and statistical properties
- **Scalability**: Generate different sizes for different test levels

### Data Size Guidelines
- **Unit Tests**: 1k-5k rows, 10-50 features
- **Integration Tests**: 5k-15k rows, 50-100 features
- **Performance Tests**: 10k-30k rows, 100-200 features
- **Production Tests**: 25k-100k rows, 200-500 features

## Performance Monitoring

### Key Metrics
- **Memory Usage**: Track DataFrame memory and process memory
- **Operation Timing**: Benchmark key pipeline operations
- **Scalability**: Test performance scaling with data size
- **Regression Detection**: Alert on performance degradation

### Memory Thresholds
- **DataFrame Memory**: < 500MB for medium datasets
- **Process Memory Increase**: < 1GB during operations
- **Memory Cleanup**: > 50MB reduction after cleanup

### Performance Baselines
- **Cross-Validation (5 splits)**: < 5 seconds
- **Regime Categorization**: < 2 seconds
- **Feature Stability (20 features)**: < 15 seconds
- **Data Loading (50k rows)**: < 30 seconds

## Running Tests Locally

### Prerequisites
```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock psutil memory-profiler
```

### Initial Setup
```bash
# Run the setup script to configure Git LFS and generate test data
./setup.sh

# Or manually:
git lfs install
git lfs pull  # If large files exist in repo
```

### Run Test Suites
```bash
# Unit tests only
pytest pipeline/tests/test_*.py -v

# Integration tests
pytest pipeline/tests/test_integration.py -v

# Performance tests
pytest pipeline/tests/test_performance.py -v

# Production readiness tests
pytest pipeline/tests/test_production_readiness.py -v -s

# All tests with timing
pytest pipeline/tests/ --durations=0
```

### Run with Coverage
```bash
pytest pipeline/tests/ --cov=pipeline --cov-report=html
open htmlcov/index.html
```

## Test Data Management

### Generating Test Data
```python
# Generate synthetic data for testing
python -c "
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Generate test dataset
np.random.seed(42)
n_rows, n_features = 50000, 200
data = {'era': np.repeat(range(1, 251), n_rows//250)}

for i in range(n_features):
    data[f'feature_{i}'] = np.random.randn(n_rows)

for i in range(10):
    data[f'target_{i}'] = np.random.randn(n_rows) * 0.05

df = pd.DataFrame(data)
Path('test_data').mkdir(exist_ok=True)
df.to_parquet('test_data/train_production.parquet')
"
```

### Data Validation
- **Era Completeness**: Sequential eras starting from 1
- **Target Distribution**: Reasonable value ranges (-10 to 10)
- **Feature Correlation**: No perfect correlations (>0.99)
- **Memory Efficiency**: Data loads within memory limits

## Monitoring and Alerts

### Performance Regression Alerts
- **Timing Increase**: > 50% slowdown from baseline
- **Memory Increase**: > 100MB unexpected growth
- **Failure Rate**: Any test failures in CI

### Weekly Production Tests
- **Schedule**: Every Monday at 2 AM UTC
- **Scope**: Full production readiness test suite
- **Data**: Production-scale synthetic data
- **Reporting**: Generate detailed performance report

## Troubleshooting

### Common Issues

#### Memory Issues
```python
# Check memory usage
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f}MB')

# Force garbage collection
import gc
gc.collect()
```

#### Performance Issues
```python
# Profile code execution
import time
start = time.time()
# ... code to profile ...
print(f'Duration: {time.time() - start:.2f}s')
```

#### Data Loading Issues
```python
# Check data structure
print(f'Rows: {len(df)}, Columns: {len(df.columns)}')
print(f'Era range: {df[\"era\"].min()} to {df[\"era\"].max()}')
print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB')
```

## Future Enhancements

### Planned Improvements
- **Real Data Testing**: Secure access to anonymized real data for CI
- **Performance Baselines**: Historical performance tracking and alerting
- **Load Testing**: Test pipeline under concurrent load
- **Integration Testing**: Test with actual Numerai API endpoints
- **Chaos Engineering**: Test pipeline resilience to failures

### Advanced Monitoring
- **Custom Metrics**: Pipeline-specific performance indicators
- **Historical Tracking**: Performance trends over time
- **Anomaly Detection**: Automatic detection of performance anomalies
- **Resource Monitoring**: CPU, disk, and network usage tracking

## Contributing

### Adding New Tests
1. **Unit Tests**: Add to existing `test_*.py` files or create new ones
2. **Integration Tests**: Add to `test_integration.py`
3. **Performance Tests**: Add to `test_performance.py`
4. **Production Tests**: Add to `test_production_readiness.py`

### Test Naming Convention
- `test_function_name`: Unit tests for specific functions
- `test_integration_*`: Integration between components
- `test_performance_*`: Performance benchmarking
- `test_production_*`: Production readiness validation

### Test Documentation
- Include docstrings explaining test purpose
- Document any special setup or teardown requirements
- Note performance expectations and memory requirements
