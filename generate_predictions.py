"""Generate predictions using a saved LightGBM model."""
import argparse
import pandas as pd
import pickle
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    with open(args.model, 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    feats = bundle['features']
    df = pd.read_parquet(args.data)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing[:10]}")
    preds = model.predict(df[feats])
    out_df = pd.DataFrame({'prediction': preds})
    out_df.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")


if __name__ == '__main__':
    main()
