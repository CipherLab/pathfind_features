#!/bin/bash

# Script to compare our models with Numerai benchmark models
# Generates predictions and calculates performance metrics

set -e  # Exit on any error

echo "Activating virtual environment..."
source /home/mat/Downloads/pathfind_features/.venv/bin/activate

echo "Generating predictions from our models..."

# Generate predictions on validation data
/home/mat/Downloads/pathfind_features/.venv/bin/python generate_predictions.py \
    --model pipeline_runs/my_experiment/control_model.pkl \
    --data v5.0/validation.parquet \
    --output pipeline_runs/my_experiment/control_validation_predictions.csv

/home/mat/Downloads/pathfind_features/.venv/bin/python generate_predictions.py \
    --model pipeline_runs/my_experiment/adaptive_only_model.pkl \
    --data pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet \
    --output pipeline_runs/my_experiment/adaptive_only_validation_predictions.csv

/home/mat/Downloads/pathfind_features/.venv/bin/python generate_predictions.py \
    --model pipeline_runs/my_experiment/experimental_model.pkl \
    --data pipeline_runs/my_experiment/03_features_validation.parquet \
    --output pipeline_runs/my_experiment/experimental_validation_predictions.csv

echo "Running benchmark comparison analysis..."

/home/mat/Downloads/pathfind_features/.venv/bin/python -c "
import pandas as pd
import numpy as np
from pathlib import Path
import json

def calculate_metrics(predictions_df, targets_df, model_name):
    \"\"\"Calculate correlation and Sharpe ratio using positional alignment\"\"\"
    # Use positional alignment instead of merge to avoid index issues
    pred = predictions_df['prediction'].values
    targ = targets_df['target'].values
    
    # Ensure same length
    min_len = min(len(pred), len(targ))
    pred = pred[:min_len]
    targ = targ[:min_len]
    
    # Remove NaN values
    valid_idx = ~(np.isnan(pred) | np.isnan(targ))
    pred = pred[valid_idx]
    targ = targ[valid_idx]
    
    if len(pred) == 0:
        return {'correlation': np.nan, 'sharpe': np.nan, 'n_predictions': 0}
    
    # Calculate correlation
    correlation = np.corrcoef(pred, targ)[0, 1]
    
    # Calculate Sharpe ratio (correlation / std of predictions)
    sharpe = correlation / np.std(pred) if np.std(pred) > 0 else np.nan
    
    return {
        'correlation': correlation,
        'sharpe': sharpe,
        'n_predictions': len(pred)
    }

# Load benchmark data
print('Loading benchmark models...')
benchmark_val = pd.read_parquet('v5.0/validation_benchmark_models.parquet')
benchmark_live = pd.read_parquet('v5.0/live_benchmark_models.parquet')

print(f'Validation benchmark shape: {benchmark_val.shape}')
print(f'Live benchmark shape: {benchmark_live.shape}')

# Load our predictions
print('Loading our model predictions...')
control_pred = pd.read_csv('pipeline_runs/my_experiment/control_validation_predictions.csv', index_col=None)
adaptive_pred = pd.read_csv('pipeline_runs/my_experiment/adaptive_only_validation_predictions.csv', index_col=None)
experimental_pred = pd.read_csv('pipeline_runs/my_experiment/experimental_validation_predictions.csv', index_col=None)

# Load validation targets
validation_targets = pd.read_parquet('v5.0/validation.parquet')[['era', 'target']].reset_index(drop=True)
validation_targets['id'] = validation_targets.index

print('\n=== MODEL PERFORMANCE COMPARISON ===')
print('Validation Set Performance (Correlation with target)')
print('-' * 60)

# Calculate metrics for our models
control_metrics = calculate_metrics(control_pred, validation_targets, 'Control')
adaptive_metrics = calculate_metrics(adaptive_pred, validation_targets, 'Adaptive-Only')
experimental_metrics = calculate_metrics(experimental_pred, validation_targets, 'Experimental')

print(f'Control Model      | Corr: {control_metrics[\"correlation\"]:.6f} | Sharpe: {control_metrics[\"sharpe\"]:.4f} | N: {control_metrics[\"n_predictions\"]}')
print(f'Adaptive-Only     | Corr: {adaptive_metrics[\"correlation\"]:.6f} | Sharpe: {adaptive_metrics[\"sharpe\"]:.4f} | N: {adaptive_metrics[\"n_predictions\"]}')
print(f'Experimental      | Corr: {experimental_metrics[\"correlation\"]:.6f} | Sharpe: {experimental_metrics[\"sharpe\"]:.4f} | N: {experimental_metrics[\"n_predictions\"]}')

# Compare with benchmark models if they have predictions
if 'prediction' in benchmark_val.columns:
    benchmark_metrics = calculate_metrics(benchmark_val[['prediction']], validation_targets, 'Benchmark')
    print(f'Benchmark Model   | Corr: {benchmark_metrics["correlation"]:.6f} | Sharpe: {benchmark_metrics["sharpe"]:.4f} | N: {benchmark_metrics["n_predictions"]}')

print('\n=== FEATURE ANALYSIS ===')
print('Feature counts:')
control_features = json.load(open('pipeline_runs/my_experiment/control_model.json'))
adaptive_features = json.load(open('pipeline_runs/my_experiment/adaptive_only_model.json'))
experimental_features = json.load(open('pipeline_runs/my_experiment/experimental_model.json'))

print(f'Control Model: {len(control_features)} features')
print(f'Adaptive-Only: {len(adaptive_features)} features') 
print(f'Experimental: {len(experimental_features)} features')
print(f'Experimental features: {experimental_features}')

print('\n=== IMPROVEMENT ANALYSIS ===')
if not np.isnan(control_metrics['correlation']) and not np.isnan(adaptive_metrics['correlation']):
    adaptive_improvement = adaptive_metrics['correlation'] - control_metrics['correlation']
    print(f'Adaptive vs Control improvement: {adaptive_improvement:.6f} ({adaptive_improvement/control_metrics[\"correlation\"]*100:.1f}%)')

if not np.isnan(adaptive_metrics['correlation']) and not np.isnan(experimental_metrics['correlation']):
    path_improvement = experimental_metrics['correlation'] - adaptive_metrics['correlation']
    print(f'Path features vs Adaptive improvement: {path_improvement:.6f} ({path_improvement/adaptive_metrics[\"correlation\"]*100:.1f}%)')

print('\nComparison complete! Check the CSV files for detailed predictions.')
"

echo "Benchmark comparison completed!"
echo "Results saved to:"
echo "- pipeline_runs/my_experiment/control_validation_predictions.csv"
echo "- pipeline_runs/my_experiment/adaptive_only_validation_predictions.csv"
echo "- pipeline_runs/my_experiment/experimental_validation_predictions.csv"
