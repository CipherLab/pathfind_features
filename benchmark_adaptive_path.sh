#!/bin/bash

# Script to benchmark adaptive+path model performance

set -e  # Exit on any error

# Default values
EXPERIMENT_NAME=${1:-"adaptive_path_experiment"}
MODEL_TYPE=${2:-"experimental"}  # experimental, adaptive_only, or control
COMPARE_MODELS=${3:-true}  # Whether to compare with other models

echo "=== BENCHMARKING $MODEL_TYPE MODEL ==="
echo "Experiment: $EXPERIMENT_NAME"
echo "Model type: $MODEL_TYPE"
echo "Compare with other models: $COMPARE_MODELS"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

EXPERIMENT_DIR="pipeline_runs/$EXPERIMENT_NAME"

# Check if experiment directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Experiment directory $EXPERIMENT_DIR does not exist!"
    exit 1
fi

# Check if model exists
if [ ! -f "$EXPERIMENT_DIR/${MODEL_TYPE}_model.pkl" ]; then
    echo "Error: Model $EXPERIMENT_DIR/${MODEL_TYPE}_model.pkl does not exist!"
    echo "Run './train_adaptive_path_model.sh $EXPERIMENT_NAME $MODEL_TYPE' first."
    exit 1
fi

echo "Generating predictions..."

if [ "$MODEL_TYPE" = "experimental" ]; then
    # Generate predictions for experimental model
    python generate_predictions.py \
        --model "$EXPERIMENT_DIR/${MODEL_TYPE}_model.pkl" \
        --data "$EXPERIMENT_DIR/03_features_validation.parquet" \
        --output "$EXPERIMENT_DIR/${MODEL_TYPE}_validation_predictions.csv"

    VALIDATION_DATA="$EXPERIMENT_DIR/03_features_validation.parquet"
    TARGET_COL="adaptive_target"

elif [ "$MODEL_TYPE" = "adaptive_only" ]; then
    # Generate predictions for adaptive-only model
    python generate_predictions.py \
        --model "$EXPERIMENT_DIR/${MODEL_TYPE}_model.pkl" \
        --data "$EXPERIMENT_DIR/01_adaptive_targets_validation.parquet" \
        --output "$EXPERIMENT_DIR/${MODEL_TYPE}_validation_predictions.csv"

    VALIDATION_DATA="$EXPERIMENT_DIR/01_adaptive_targets_validation.parquet"
    TARGET_COL="adaptive_target"

elif [ "$MODEL_TYPE" = "control" ]; then
    # Generate predictions for control model
    python generate_predictions.py \
        --model "$EXPERIMENT_DIR/${MODEL_TYPE}_model.pkl" \
        --data v5.0/validation.parquet \
        --output "$EXPERIMENT_DIR/${MODEL_TYPE}_validation_predictions.csv"

    VALIDATION_DATA="v5.0/validation.parquet"
    TARGET_COL="target"

else
    echo "Error: Unknown model type '$MODEL_TYPE'"
    exit 1
fi

echo "Running performance analysis..."

# Run performance analysis
python -c "
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr

def calculate_metrics(predictions_df, targets_df, model_name, pred_col='prediction', target_col='target'):
    \"\"\"Calculate correlation and Sharpe ratio\"\"\"
    # Merge on index to align predictions with targets
    merged = pd.merge(predictions_df, targets_df, left_index=True, right_index=True)
    
    pred = merged[pred_col].values
    targ = merged[target_col].values
    
    # Remove NaN values
    valid_idx = ~(np.isnan(pred) | np.isnan(targ))
    pred = pred[valid_idx]
    targ = targ[valid_idx]
    
    if len(pred) == 0:
        return {'correlation': np.nan, 'sharpe': np.nan, 'n_predictions': 0}
    
    # Calculate Spearman correlation (more robust for financial data)
    correlation, _ = spearmanr(pred, targ)
    
    # Calculate Sharpe ratio (correlation / std of predictions)
    sharpe = correlation / np.std(pred) if np.std(pred) > 0 else np.nan
    
    return {
        'correlation': correlation,
        'sharpe': sharpe,
        'n_predictions': len(pred)
    }

# Load predictions and validation data
print('Loading predictions and validation data...')
predictions = pd.read_csv('$EXPERIMENT_DIR/${MODEL_TYPE}_validation_predictions.csv')
validation_df = pd.read_parquet('$VALIDATION_DATA')

print(f'Predictions shape: {predictions.shape}')
print(f'Validation shape: {validation_df.shape}')

# Calculate metrics
metrics = calculate_metrics(predictions, validation_df[['$TARGET_COL']], '$MODEL_TYPE', target_col='$TARGET_COL')

print('\n=== $MODEL_TYPE MODEL PERFORMANCE ===')
print(f'Correlation (Spearman): {metrics[\"correlation\"]:.6f}')
print(f'Sharpe Ratio: {metrics[\"sharpe\"]:.4f}')
print(f'Number of predictions: {metrics[\"n_predictions\"]}')

# Load feature information
try:
    with open('$EXPERIMENT_DIR/${MODEL_TYPE}_model.json', 'r') as f:
        features = json.load(f)
    print(f'Number of features used: {len(features)}')
    
    # Show feature types
    original_features = [f for f in features if f.startswith('feature')]
    path_features = [f for f in features if f.startswith('path')]
    print(f'Original features: {len(original_features)}')
    print(f'Path features: {len(path_features)}')
    
except Exception as e:
    print(f'Could not load feature information: {e}')

print('\n=== DETAILED ANALYSIS ===')

# Era-by-era performance
if 'era' in validation_df.columns:
    print('\nEra-by-era performance:')
    eras = validation_df['era'].unique()
    era_correlations = []
    
    for era in sorted(eras)[:10]:  # Show first 10 eras
        era_data = validation_df[validation_df['era'] == era]
        era_pred = predictions.iloc[era_data.index]
        
        if len(era_data) > 10:  # Only calculate if enough data
            era_metrics = calculate_metrics(era_pred, era_data[['$TARGET_COL']], f'era_{era}', target_col='$TARGET_COL')
            if not np.isnan(era_metrics['correlation']):
                era_correlations.append(era_metrics['correlation'])
                print(f'Era {era}: {era_metrics[\"correlation\"]:.6f}')
    
    if era_correlations:
        print(f'\nEra correlation stats:')
        print(f'Mean: {np.mean(era_correlations):.6f}')
        print(f'Std: {np.std(era_correlations):.6f}')
        print(f'Min: {np.min(era_correlations):.6f}')
        print(f'Max: {np.max(era_correlations):.6f}')

# Save results
results = {
    'model_type': '$MODEL_TYPE',
    'experiment': '$EXPERIMENT_NAME',
    'metrics': metrics,
    'features': {
        'total': len(features) if 'features' in locals() else None,
        'original': len(original_features) if 'original_features' in locals() else None,
        'path': len(path_features) if 'path_features' in locals() else None
    }
}

with open('$EXPERIMENT_DIR/${MODEL_TYPE}_benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f'\nResults saved to: $EXPERIMENT_DIR/${MODEL_TYPE}_benchmark_results.json')
"

echo ""
echo "=== BENCHMARK COMPLETE ==="
echo "Generated files:"
echo "- $EXPERIMENT_DIR/${MODEL_TYPE}_validation_predictions.csv"
echo "- $EXPERIMENT_DIR/${MODEL_TYPE}_benchmark_results.json"

if [ "$COMPARE_MODELS" = "true" ]; then
    echo ""
    echo "To compare with other models, run:"
    echo "  ./benchmark_adaptive_path.sh $EXPERIMENT_NAME control"
    echo "  ./benchmark_adaptive_path.sh $EXPERIMENT_NAME adaptive_only"
    echo "Then use compare_model_performance.py to compare all results"
fi
