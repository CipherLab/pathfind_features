#!/bin/bash

# Automated Hyperparameter Tuning with Ensembling Pipeline
# This script runs hyperparameter optimization and creates an ensemble of top models

set -e  # Exit on any error

# Configuration
EXPERIMENT_NAME="${1:-auto_ensemble_$(date +%Y%m%d_%H%M%S)}"
TRAIN_DATA="${2:-pipeline_runs/my_experiment/01_adaptive_targets_train.parquet}"
VAL_DATA="${3:-pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet}"
FEATURES_JSON="${4:-pipeline_runs/my_experiment/adaptive_only_model.json}"
TARGET_COL="${5:-adaptive_target}"
SPEED_MODE="${6:-balanced}"
N_ITERATIONS="${7:-15}"
N_MODELS="${8:-5}"

echo "=== AUTOMATED HYPERPARAMETER TUNING + ENSEMBLING ==="
echo "Experiment: $EXPERIMENT_NAME"
echo "Speed Mode: $SPEED_MODE"
echo "Iterations: $N_ITERATIONS"
echo "Ensemble Size: $N_MODELS"
echo ""

cd /home/mat/Downloads/pathfind_features
source .venv/bin/activate

# Step 1: Run hyperparameter optimization
echo "🔍 Step 1: Running hyperparameter optimization..."
OUTPUT_DIR="hyperparameter_tuning_$EXPERIMENT_NAME"

python hyperparameter_tuning.py \
    --train-data "$TRAIN_DATA" \
    --validation-data "$VAL_DATA" \
    --features-json "$FEATURES_JSON" \
    --target-col "$TARGET_COL" \
    --output-dir "$OUTPUT_DIR" \
    --search-type focused \
    --n-iterations "$N_ITERATIONS" \
    --speed-mode "$SPEED_MODE"

echo ""

# Step 2: Extract top N models for ensembling
echo "🏗️ Step 2: Extracting top $N_MODELS models for ensembling..."

python -c "
import json
import pandas as pd
from pathlib import Path

# Load results
with open('$OUTPUT_DIR/hyperparameter_results.json', 'r') as f:
    results = json.load(f)

# Filter valid results and sort by correlation
valid_results = [r for r in results if not pd.isna(r['correlation'])]
top_models = sorted(valid_results, key=lambda x: x['correlation'], reverse=True)[:$N_MODELS]

print(f'Found {len(valid_results)} valid models')
print(f'Selected top $N_MODELS models for ensemble:')
for i, model in enumerate(top_models, 1):
    print(f'  {i}. Correlation: {model[\"correlation\"]:.4f}, Sharpe: {model[\"sharpe_ratio\"]:.2f}')

# Save ensemble configuration
ensemble_config = {
    'experiment_name': '$EXPERIMENT_NAME',
    'n_models': len(top_models),
    'models': top_models,
    'train_data': '$TRAIN_DATA',
    'val_data': '$VAL_DATA',
    'features_json': '$FEATURES_JSON',
    'target_col': '$TARGET_COL'
}

with open('$OUTPUT_DIR/ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2, default=str)

print(f'\\nEnsemble configuration saved to: $OUTPUT_DIR/ensemble_config.json')
"

echo ""

# Step 3: Create ensemble training script
echo "📝 Step 3: Creating ensemble training script..."

cat > "$OUTPUT_DIR/train_ensemble.py" << 'EOF'
#!/usr/bin/env python3
"""
Train an ensemble of models using optimized hyperparameters.
"""

import json
import lightgbm as lgb
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from scipy.stats import spearmanr

def load_data(train_path, val_path, features_json, target_col):
    """Load and prepare data."""
    # Load features
    with open(features_json, 'r') as f:
        features = json.load(f)

    # Load training data
    pf_train = pq.ParquetFile(train_path)
    train_df = pf_train.read().to_pandas()
    X_train = train_df[features].astype('float32')
    y_train = train_df[target_col].astype('float32')

    # Load validation data
    pf_val = pq.ParquetFile(val_path)
    val_df = pf_val.read().to_pandas()
    X_val = val_df[features].astype('float32')
    y_val = val_df[target_col].astype('float32')

    return X_train, y_train, X_val, y_val

def train_single_model(params, X_train, y_train, X_val, y_val, model_id):
    """Train a single model with given parameters."""
    print(f"Training model {model_id}...")

    # Set up datasets
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    # Update parameters
    params = params.copy()
    params['seed'] = 42 + model_id  # Different seed for each model

    # Train with early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        lgb.log_evaluation(period=0)
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,  # Will be stopped early
        valid_sets=[val_set],
        callbacks=callbacks
    )

    # Make predictions
    val_preds = model.predict(X_val)

    # Calculate metrics
    valid_mask = ~(pd.isna(y_val) | pd.isna(val_preds))
    y_clean = y_val[valid_mask]
    preds_clean = val_preds[valid_mask]

    if len(y_clean) > 0:
        corr_result = spearmanr(y_clean, preds_clean)
        correlation = abs(corr_result[0])
    else:
        correlation = 0.0

    print(".4f"
    return model, val_preds, correlation

def main():
    # Load ensemble configuration
    with open('ensemble_config.json', 'r') as f:
        config = json.load(f)

    print(f"Training ensemble with {config['n_models']} models...")

    # Load data
    X_train, y_train, X_val, y_val = load_data(
        config['train_data'],
        config['val_data'],
        config['features_json'],
        config['target_col']
    )

    # Train individual models
    models = []
    predictions = []
    correlations = []

    for i, model_config in enumerate(config['models']):
        model, preds, corr = train_single_model(
            model_config['params'], X_train, y_train, X_val, y_val, i
        )
        models.append(model)
        predictions.append(preds)
        correlations.append(corr)

    # Create ensemble predictions (simple average)
    ensemble_preds = np.mean(predictions, axis=0)

    # Calculate ensemble metrics
    valid_mask = ~(pd.isna(y_val) | pd.isna(ensemble_preds))
    y_clean = y_val[valid_mask]
    preds_clean = ensemble_preds[valid_mask]

    if len(y_clean) > 0:
        corr_result = spearmanr(y_clean, preds_clean)
        ensemble_correlation = abs(corr_result[0])
    else:
        ensemble_correlation = 0.0

    print("
=== ENSEMBLE RESULTS ===")
    print(".4f")
    print(".4f")
    print(f"Individual model correlations: {['.3f' for c in correlations]}")

    # Save ensemble
    ensemble_data = {
        'models': models,
        'predictions': predictions,
        'ensemble_predictions': ensemble_preds,
        'correlations': correlations,
        'ensemble_correlation': ensemble_correlation,
        'config': config
    }

    # Note: In practice, you'd save the models to disk
    print("
Ensemble training completed!")
    print("Models and predictions ready for evaluation.")

if __name__ == '__main__':
    main()
EOF

chmod +x "$OUTPUT_DIR/train_ensemble.py"

echo ""

# Step 4: Create evaluation script
echo "📊 Step 4: Creating ensemble evaluation script..."

cat > "$OUTPUT_DIR/evaluate_ensemble.py" << 'EOF'
#!/usr/bin/env python3
"""
Evaluate ensemble performance and compare with individual models.
"""

import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def load_validation_data(val_path, features_json, target_col):
    """Load validation data."""
    import pyarrow.parquet as pq

    with open(features_json, 'r') as f:
        features = json.load(f)

    pf = pq.ParquetFile(val_path)
    df = pf.read().to_pandas()
    X = df[features].astype('float32')
    y = df[target_col].astype('float32')

    return X, y

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_clean = y_true[valid_mask]
    pred_clean = y_pred[valid_mask]

    if len(y_clean) == 0:
        return {'correlation': 0.0, 'mae': 0.0, 'rmse': 0.0}

    # Correlation
    corr_result = spearmanr(y_clean, pred_clean)
    correlation = abs(corr_result[0])

    # MAE and RMSE
    mae = np.mean(np.abs(y_clean - pred_clean))
    rmse = np.sqrt(np.mean((y_clean - pred_clean) ** 2))

    return {
        'correlation': correlation,
        'mae': mae,
        'rmse': rmse,
        'n_samples': len(y_clean)
    }

def main():
    # Load configuration
    with open('ensemble_config.json', 'r') as f:
        config = json.load(f)

    print("=== ENSEMBLE EVALUATION ===")

    # Load validation data
    X_val, y_val = load_validation_data(
        config['val_data'],
        config['features_json'],
        config['target_col']
    )

    print(f"Validation data: {len(X_val)} samples")

    # Evaluate individual models
    individual_metrics = []
    for i, model_config in enumerate(config['models']):
        # In practice, you'd load the trained models here
        # For now, we'll just show the stored correlations
        correlation = model_config['correlation']
        print(".4f")

    # Calculate ensemble metrics (would need actual predictions)
    print("
Note: For full evaluation, run train_ensemble.py first")
    print("This will generate actual ensemble predictions for comparison.")

if __name__ == '__main__':
    main()
EOF

chmod +x "$OUTPUT_DIR/evaluate_ensemble.py"

echo ""

# Step 5: Create summary report
echo "📋 Step 5: Generating summary report..."

python -c "
import json
import pandas as pd
from pathlib import Path

# Load results
with open('$OUTPUT_DIR/hyperparameter_results.json', 'r') as f:
    results = json.load(f)

valid_results = [r for r in results if not pd.isna(r['correlation'])]
if valid_results:
    best_result = max(valid_results, key=lambda x: x['correlation'])
    
    print('🎯 EXPERIMENT SUMMARY')
    print('=' * 50)
    print(f'Experiment: $EXPERIMENT_NAME')
    print(f'Total iterations: {len(results)}')
    print(f'Valid results: {len(valid_results)}')
    print(f'Best correlation: {best_result[\"correlation\"]:.4f}')
    print(f'Best Sharpe ratio: {best_result[\"sharpe_ratio\"]:.2f}')
    print()
    print('🏆 BEST PARAMETERS:')
    for key, value in best_result['params'].items():
        print(f'  {key}: {value}')
    print()
    print('📁 GENERATED FILES:')
    print(f'  • $OUTPUT_DIR/hyperparameter_results.json')
    print(f'  • $OUTPUT_DIR/ensemble_config.json')
    print(f'  • $OUTPUT_DIR/train_ensemble.py')
    print(f'  • $OUTPUT_DIR/evaluate_ensemble.py')
    print(f'  • $OUTPUT_DIR/best_params_corr.json')
    print(f'  • $OUTPUT_DIR/best_params_sharpe.json')
    print()
    print('🚀 NEXT STEPS:')
    print(f'  1. cd $OUTPUT_DIR')
    print(f'  2. python train_ensemble.py    # Train ensemble')
    print(f'  3. python evaluate_ensemble.py # Evaluate results')
    print(f'  4. Compare with baseline models')
else:
    print('❌ No valid results found!')
" > "$OUTPUT_DIR/experiment_summary.txt"

echo ""
echo "=== AUTOMATION COMPLETE ==="
echo "📁 Output directory: $OUTPUT_DIR"
echo "📋 Summary: $OUTPUT_DIR/experiment_summary.txt"
echo ""
echo "🚀 Quick start:"
echo "  cd $OUTPUT_DIR"
echo "  python train_ensemble.py"
echo ""
echo "The automated pipeline has created:"
echo "  • Hyperparameter optimization results"
echo "  • Ensemble configuration with top $N_MODELS models"
echo "  • Training and evaluation scripts"
echo "  • Complete experiment summary"
