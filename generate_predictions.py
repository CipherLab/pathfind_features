"""Generate predictions using a saved LightGBM model."""
import argparse
import csv
import pickle
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
        bundle = pickle.load(f)
    model = bundle['model']
    feats = bundle['features']

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
            writer.writerows([[p] for p in preds])
    print(f"Predictions written to {args.output}")


if __name__ == '__main__':
    main()
