"""Generate predictions using a saved LightGBM model."""
import argparse
import csv
import pickle
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument(
        '--batch-size',
        type=int,
        default=100_000,
        help='number of rows to process per batch',
    )
    args = ap.parse_args()

    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    features_path = args.model.replace('.pkl', '_features.json')
    with open(features_path, 'r') as f:
        feats = json.load(f)

    parquet_file = pq.ParquetFile(args.data)
    available_cols = set(parquet_file.schema.names)
    missing = [f for f in feats if f not in available_cols]
    if missing:
        raise ValueError(f"Missing features in data: {missing[:10]}")

    output_path = Path(args.output)
    with output_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prediction'])
        for batch in parquet_file.iter_batches(columns=feats, batch_size=args.batch_size):
            df = batch.to_pandas()
            preds = model.predict(df[feats])
            # Avoid creating an intermediate list of lists
            try:
                writer.writerows(preds.reshape(-1, 1))
            except AttributeError:
                for p in preds:
                    writer.writerow([p])
    print(f"Predictions written to {args.output}")


if __name__ == '__main__':
    main()
