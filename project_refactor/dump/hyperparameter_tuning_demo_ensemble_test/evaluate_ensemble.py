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
