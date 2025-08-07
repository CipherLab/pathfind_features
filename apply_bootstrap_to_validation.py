"""Apply discovered era weights and relationships to validation data to build enhanced validation dataset.

Steps:
1. Load validation parquet.
2. Create adaptive_target using discovery weights (fallback to equal if era missing).
3. (Optional) Apply discovered relationships to create interaction features (path_*).
"""
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path


def load_weights(path: str) -> dict[str, np.ndarray]:
    with open(path, 'r') as f:
        data = json.load(f)
    return {str(k): np.array(v, dtype=float) for k, v in data.items()}


def load_relationships(path: str | None):
    if not path:
        return []
    with open(path, 'r') as f:
        return json.load(f)


def compute_adaptive_target(df: pd.DataFrame, weights_map: dict[str, np.ndarray]):
    target_cols = [c for c in df.columns if c.startswith('target')]
    if not target_cols:
        raise ValueError('No target* columns in validation data')
    tvals = df[target_cols].values.astype(float)
    adaptive = []
    default = np.ones(len(target_cols)) / len(target_cols)
    for era, row in zip(df['era'].values, tvals):
        w = weights_map.get(str(era), None)
        if w is None or len(w) != len(target_cols):
            w = default
        adaptive.append(float(np.dot(row, w)))
    df['adaptive_target'] = adaptive


def apply_relationships(df: pd.DataFrame, relationships: list, max_features: int):
    relationships = relationships[:max_features]
    new_cols = []
    for i, rel in enumerate(relationships):
        feat1, feat2 = rel.get('feature1'), rel.get('feature2')
        strength = float(rel.get('strength', 1.0))
        if feat1 not in df.columns or feat2 not in df.columns:
            continue
        name = f"path_val_{i:02d}_{feat1[-4:]}x{feat2[-4:]}"
        df[name] = (df[feat1].astype('float32') * df[feat2].astype('float32') * strength).astype('float32')
        new_cols.append(name)
    return new_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-data', required=True)
    ap.add_argument('--era-weights', required=True, help='01_target_discovery.json from training run')
    ap.add_argument('--relationships-file', required=False, help='02_discovered_relationships.json from training run')
    ap.add_argument('--output-data', required=True)
    ap.add_argument('--max-new-features', type=int, default=40)
    ap.add_argument('--row-limit', type=int, default=None)
    args = ap.parse_args()

    weights = load_weights(args.era_weights)
    relationships = load_relationships(args.relationships_file) if args.relationships_file else []

    pf = pq.ParquetFile(args.input_data)
    writer = None
    processed = 0
    new_features_all = []
    for batch in pf.iter_batches(batch_size=50_000):
        if args.row_limit is not None and processed >= args.row_limit:
            break
        df = batch.to_pandas()
        if args.row_limit is not None:
            remain = args.row_limit - processed
            if len(df) > remain:
                df = df.iloc[:remain]
        compute_adaptive_target(df, weights)
        if relationships:
            new_cols = apply_relationships(df, relationships, args.max_new_features)
            new_features_all.extend(new_cols)
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(args.output_data, table.schema)
        writer.write_table(table)
        processed += len(df)
    if writer:
        writer.close()
    # meta file with list of new validation features
    meta_path = Path(args.output_data).with_suffix('.new_features.json')
    meta_path.write_text(json.dumps(sorted(set(new_features_all))), encoding='utf-8')
    print(f"Wrote enhanced validation data to {args.output_data}")


if __name__ == '__main__':
    main()
