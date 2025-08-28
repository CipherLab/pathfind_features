#!/usr/bin/env python3
"""
Model training and evaluation utilities for hyperparameter optimization.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from scipy.stats import spearmanr
from typing import Dict, List, Optional
from data_utils import filter_by_era_range, read_val_sample


class CallableModel:
    def __init__(self, booster):
        self.booster = booster

    def __call__(self, *args, **kwargs):
        return self.booster.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.booster.predict(*args, **kwargs)


def train_and_evaluate(params: Dict, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series, seed: int = 42) -> Dict:
    """Train a model with given parameters and evaluate on validation set."""
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    # Set up callbacks for early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)  # Disable logging
    ]

    # Update params with seed
    params['seed'] = seed

    # Train model
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=callbacks
    )

    # Make predictions
    val_preds = model.predict(X_val)

    # Calculate metrics
    corr_result = spearmanr(y_val, val_preds)
    corr = abs(corr_result[0])  # type: ignore

    # Calculate Sharpe ratio (assuming we have era information)
    val_with_era = X_val.copy()
    val_with_era['target'] = y_val
    val_with_era['prediction'] = val_preds

    # Group by era and calculate Sharpe
    era_groups = val_with_era.groupby(val_with_era.index // 1000)  # Assuming eras are grouped
    era_corrs = []
    for _, group in era_groups:
        if len(group) > 1:
            era_corr, _ = spearmanr(group['target'], group['prediction'])
            era_corrs.append(era_corr)

    sharpe = np.mean(era_corrs) / np.std(era_corrs) if era_corrs else 0

    return {
        'correlation': corr,
        'sharpe_ratio': sharpe,
        'best_iteration': model.best_iteration,
        'params': params
    }


def train_and_evaluate_chunked(params: Dict, train_path: str, val_path: str, features: List[str], target_col: str,
                                chunk_rows: int = 250_000, val_rows: int = 200_000, total_rounds: int = 1000, rounds_per_chunk: int = 200, seed: int = 42, era_range: Optional[str] = None) -> Dict:
    """Chunked training and evaluation for hyperparameter search."""
    pf_train = pq.ParquetFile(train_path)
    features = [c for c in features if c in pf_train.schema.names]

    # Validation sample
    Xv, yv, era_series = read_val_sample(val_path, features, target_col, val_rows)
    valid_set = lgb.Dataset(Xv, label=yv, free_raw_data=False)

    params = params.copy()
    params['seed'] = seed

    booster = None
    built_rounds = 0
    needed_cols = [c for c in (features + [target_col, 'era']) if c in pf_train.schema.names]  # Add era column
    for batch in pf_train.iter_batches(columns=needed_cols, batch_size=chunk_rows):
        df = batch.to_pandas()

        # Apply era filtering if specified
        if era_range:
            df = filter_by_era_range(df, era_range)
            if len(df) == 0:
                continue  # Skip empty batches after filtering

        X = df[[c for c in features if c in df.columns]].astype('float32')
        y = df[target_col].astype('float32')
        train_set = lgb.Dataset(X, label=y, free_raw_data=True)

        rounds = min(rounds_per_chunk, max(0, total_rounds - built_rounds))
        if rounds == 0:
            break
            
        # Set up callbacks for early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0)  # Disable logging
        ]
        
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=rounds,
            valid_sets=[valid_set],
            callbacks=callbacks,
            init_model=booster,
            keep_training_booster=True,
        )
        built_rounds += rounds

    if booster is None:
        # If no training was done, return NaNs
        return {
            'correlation': float('nan'),
            'sharpe_ratio': float('nan'),
            'params': params
        }

    # Make predictions
    val_preds = booster.predict(Xv)
    val_preds = np.array(val_preds)  # Ensure it's a numpy array

    # Debug: Check for NaN values
    print(f"Validation predictions shape: {val_preds.shape}")
    print(f"Validation predictions NaN count: {pd.isna(val_preds).sum()}")
    print(f"Validation targets NaN count: {pd.isna(yv).sum()}")
    print(f"Validation predictions unique values: {len(pd.unique(val_preds))}")

    # Filter out NaN values for correlation calculation
    valid_mask = ~(pd.isna(yv) | pd.isna(val_preds))
    yv_clean = yv[valid_mask]
    val_preds_clean = val_preds[valid_mask]
    
    print(f"After filtering NaNs - Clean targets shape: {yv_clean.shape}, Clean predictions shape: {val_preds_clean.shape}")

    # Calculate metrics
    if len(yv_clean) > 0:
        corr_result = spearmanr(yv_clean, val_preds_clean)
        corr = abs(corr_result[0])  # type: ignore
    else:
        corr_result = None
        corr = float('nan')
    
    print(f"Correlation result: {corr_result}")
    print(f"Correlation value: {corr}")
    print(f"Is correlation NaN: {pd.isna(corr)}")

    # Sharpe ratio
    if era_series is not None and len(yv_clean) > 0:
        # Filter era series to match the clean data
        era_series_clean = era_series[valid_mask]
        
        val_with_era = Xv.loc[valid_mask].copy()
        val_with_era['target'] = yv_clean
        val_with_era['prediction'] = val_preds_clean
        val_with_era['era'] = era_series_clean
        
        era_groups = val_with_era.groupby('era')
        era_corrs = []
        for _, group in era_groups:
            if len(group) > 1:
                era_corr_result = spearmanr(group['target'], group['prediction'])
                era_corr = era_corr_result[0]
                if isinstance(era_corr, (int, float)) and not np.isnan(era_corr):
                    era_corrs.append(era_corr)
        sharpe = np.mean(era_corrs) / np.std(era_corrs) if era_corrs else 0
    else:
        sharpe = 0

    return {
        'correlation': corr,
        'sharpe_ratio': sharpe,
        'params': params
    }