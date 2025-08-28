#!/usr/bin/env python3
"""
Data loading and preprocessing utilities for hyperparameter optimization.
"""
import json
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Tuple, Optional, Union


def filter_by_era_range(df: pd.DataFrame, era_range: str) -> pd.DataFrame:
    """Filter dataframe by era range (e.g., 'MAX-200:MAX-75')."""
    if 'era' not in df.columns:
        print("Warning: 'era' column not found, skipping era filtering")
        return df

    try:
        # Convert era column to integers for easier handling
        df = df.copy()
        
        # Convert to numeric, handling any conversion errors
        df['era_num'] = pd.to_numeric(df['era'], errors='coerce')
        
        # Check for NaN values after conversion
        nan_count = df['era_num'].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} era values could not be converted to numeric")
            print(f"Non-numeric era values: {df[df['era_num'].isna()]['era'].unique()[:10]}")
            df = df.dropna(subset=['era_num'])
        
        df['era_num'] = df['era_num'].astype(int)
        
        max_era = df['era_num'].max()
        print(f"Max era: {max_era}, Era range: {era_range}")

        start_era, end_era = era_range.split(':')

        # Parse start era
        if start_era.startswith('MAX-'):
            start_offset = int(start_era.split('-')[1])
            start_era_val = max(1, max_era - start_offset)  # Don't go below 1
        else:
            start_era_val = int(start_era)

        # Parse end era
        if end_era.startswith('MAX-'):
            end_offset = int(end_era.split('-')[1])
            end_era_val = max(1, max_era - end_offset)  # Don't go below 1
        else:
            end_era_val = int(end_era)

        # Filter using integer comparison
        mask = (df['era_num'] >= start_era_val) & (df['era_num'] <= end_era_val)
        filtered_df = df[mask].drop('era_num', axis=1)
        
        print(f"Filtered from {len(df)} to {len(filtered_df)} rows")
        return filtered_df

    except Exception as e:
        print(f"Error in era filtering: {e}")
        import traceback
        traceback.print_exc()
        return df


def load_data(train_path: str, val_path: str, features: List[str], target_col: str, era_range: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and prepare training and validation data."""
    print(f"Loading data from {train_path} and {val_path}")

    # Load training data
    pf_train = pq.ParquetFile(train_path)
    train_df = pf_train.read().to_pandas()

    # Load validation data
    pf_val = pq.ParquetFile(val_path)
    val_df = pf_val.read().to_pandas()

    # Apply era filtering if specified
    if era_range:
        print(f"Applying era range filter: {era_range}")
        train_df = filter_by_era_range(train_df, era_range)
        val_df = filter_by_era_range(val_df, era_range)
        print(f"After era filtering - Training data shape: {train_df.shape}, Validation data shape: {val_df.shape}")

    X_train = train_df[features].astype('float32')
    y_train = train_df[target_col].astype('float32')

    X_val = val_df[features].astype('float32')
    y_val = val_df[target_col].astype('float32')

    print(f"Final data shapes - Training: {X_train.shape}, Validation: {X_val.shape}")
    return X_train, y_train, X_val, y_val


def get_feature_list(features_json: str) -> List[str]:
    """Extract feature list from features.json file."""
    if not Path(features_json).exists():
        raise FileNotFoundError(f"Features file not found: {features_json}")

    with open(features_json, 'r') as f:
        data = json.load(f)

    # Try different possible structures
    if 'feature_sets' in data and 'medium' in data['feature_sets']:
        return data['feature_sets']['medium']
    elif isinstance(data, list):
        return data
    elif 'features' in data:
        return data['features']
    else:
        # Fallback: extract from data keys
        features = [k for k in data.keys() if k.startswith('feature_')]
        return features


def read_val_sample(parquet_path: str, features: List[str], target_col: str, rows: int) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Read a sample from validation data."""
    pf = pq.ParquetFile(parquet_path)
    needed = [c for c in (features + [target_col, 'era']) if c in pf.schema.names]
    acc = []
    remaining = rows
    for batch in pf.iter_batches(columns=needed, batch_size=min(100_000, rows)):
        df = batch.to_pandas()
        acc.append(df)
        remaining -= len(df)
        if remaining <= 0:
            break
    val = pd.concat(acc, ignore_index=True)
    Xv = val[[c for c in features if c in val.columns]].astype('float32')
    yv = val[target_col].astype('float32')
    era_series = val['era'] if 'era' in val.columns else None
    return Xv, yv, era_series