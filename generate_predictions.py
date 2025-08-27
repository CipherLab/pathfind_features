"""Generate predictions using a saved LightGBM model."""
import argparse
import csv
import pickle
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


class CallableModel:
    def __init__(self, booster):
        self.booster = booster
    
    def __call__(self, *args, **kwargs):
        return self.booster.predict(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.booster.predict(*args, **kwargs)
    
    def __getattr__(self, name):
        # Avoid recursion during unpickling
        if name == 'booster' or not hasattr(self, 'booster'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.booster, name)


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
        writer.writerow(['id', 'prediction'])
        row_id = 0
        for batch in parquet_file.iter_batches(columns=feats, batch_size=args.batch_size):
            df = batch.to_pandas()
            preds = model.predict(df[feats])
            # Write predictions with sequential IDs
            try:
                for p in preds:
                    writer.writerow([row_id, p])
                    row_id += 1
            except TypeError:
                writer.writerow([row_id, preds])
                row_id += 1
    print(f"Predictions written to {args.output}")


if __name__ == '__main__':
    main()
