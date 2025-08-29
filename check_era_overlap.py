#!/usr/bin/env python3
"""
Check era overlap between training and validation data
"""
import pyarrow.parquet as pq
import pandas as pd

def check_era_overlap():
    print("Checking era overlap between train and validation data...")

    # Load training data
    train_pf = pq.ParquetFile("v5.0/train.parquet")
    train_eras = set()
    for batch in train_pf.iter_batches(columns=['era'], batch_size=100_000):
        df = batch.to_pandas()
        train_eras.update(df['era'].unique())

    # Load validation data
    val_pf = pq.ParquetFile("v5.0/validation.parquet")
    val_eras = set()
    for batch in val_pf.iter_batches(columns=['era'], batch_size=100_000):
        df = batch.to_pandas()
        val_eras.update(df['era'].unique())

    print(f"Training eras: {len(train_eras)} total")
    print(f"Validation eras: {len(val_eras)} total")
    print(f"Overlapping eras: {len(train_eras & val_eras)}")
    print(f"Training-only eras: {len(train_eras - val_eras)}")
    print(f"Validation-only eras: {len(val_eras - train_eras)}")

    # Show some examples
    print(f"\nSample training eras: {sorted(list(train_eras))[:10]}")
    print(f"Sample validation eras: {sorted(list(val_eras))[:10]}")
    print(f"Sample overlapping eras: {sorted(list(train_eras & val_eras))[:10]}")

if __name__ == "__main__":
    check_era_overlap()
