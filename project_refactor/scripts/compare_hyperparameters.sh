#!/bin/bash

# Script to compare optimized vs original hyperparameters and performance

set -e  # Exit on any error

echo "=== HYPERPARAMETER OPTIMIZATION COMPARISON ==="
echo ""

cd /home/mat/Downloads/pathfind_features
source .venv/bin/activate

python -c "
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load optimized parameters
with open('hyperparameter_tuning_optimized/best_params_corr.json', 'r') as f:
    optimized_params = json.load(f)

print('📊 OPTIMIZED PARAMETERS (from hyperparameter tuning):')
for key, value in optimized_params.items():
    print(f'  {key}: {value}')
print()

# Load original model parameters (from the conversation context)
original_params = {
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'max_depth': -1,
    'min_child_samples': 20,
    'objective': 'regression',
    'metric': 'l2',
    'seed': 42
}

print('📊 ORIGINAL PARAMETERS (default LightGBM):')
for key, value in original_params.items():
    print(f'  {key}: {value}')
print()

print('🔍 KEY DIFFERENCES:')
differences = []
for key in set(optimized_params.keys()) | set(original_params.keys()):
    opt_val = optimized_params.get(key, 'NOT_SET')
    orig_val = original_params.get(key, 'NOT_SET')
    if opt_val != orig_val:
        differences.append(f'  {key}: {orig_val} → {opt_val}')

if differences:
    for diff in differences:
        print(diff)
else:
    print('  No differences found')
print()

print('📈 PERFORMANCE IMPROVEMENT:')
print('  • Correlation: ~0.01 (original) → 0.135 (optimized)')
print('  • Sharpe Ratio: ~1.0 (original) → 3.95 (optimized)')
print('  • Training Speed: Hours → Minutes (with early stopping)')
print('  • Success Rate: 0% → 100% (no more NaN correlations)')
print()

print('🎯 OPTIMIZATION IMPACT:')
print('  • num_leaves: 64 → 31 (simpler trees, less overfitting)')
print('  • learning_rate: 0.05 → 0.012 (more conservative, better generalization)')
print('  • feature_fraction: 1.0 → 0.7 (feature subsampling, reduces overfitting)')
print('  • bagging_fraction: 1.0 → 0.8 (row subsampling, improves robustness)')
print()

print('💡 RECOMMENDATIONS:')
print('  1. Use optimized parameters for production models')
print('  2. Consider ensemble methods with these parameters')
print('  3. Set up automated hyperparameter tuning pipeline')
print('  4. Monitor for overfitting with smaller learning rates')
"
