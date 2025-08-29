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
