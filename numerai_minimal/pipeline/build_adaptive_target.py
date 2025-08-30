#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tests import setup_script_output, get_output_path, initialize_script_output, add_output_dir_arguments


def load_weights(weights_path: str) -> Dict[str, List[float]]:
    with open(weights_path) as f:
        d = json.load(f)
    # Support both {era: [..]} and {era: {'weights':[..]}}
    out: Dict[str, List[float]] = {}
    for era, v in d.items():
        if isinstance(v, dict) and 'weights' in v:
            out[era] = list(map(float, v['weights']))
        elif isinstance(v, list):
            out[era] = list(map(float, v))
        else:
            raise ValueError(f"Unsupported weights format for era {era}: {type(v)}")
    return out


def average_last_k(weights_by_era: Dict[str, List[float]], k: int) -> np.ndarray:
    eras_sorted = sorted(weights_by_era.keys())
    last = eras_sorted[-k:] if k > 0 else eras_sorted
    arr = np.array([weights_by_era[e] for e in last], dtype=float)
    return arr.mean(axis=0)


def build_adaptive_target(
    df: pd.DataFrame,
    weights_by_era: Dict[str, List[float]],
    targets_prefix: str,
    era_col: str,
    mode: str,
    k: int,
    era_fallback: str,
) -> pd.Series:
    target_cols = [c for c in df.columns if c.startswith(targets_prefix)]
    if not target_cols:
        raise SystemExit(f"No columns starting with '{targets_prefix}' found")

    # Establish a canonical target order
    target_cols = sorted(target_cols)

    n_targets = len(next(iter(weights_by_era.values())))
    if n_targets != len(target_cols):
        raise SystemExit(
            f"Weights length {n_targets} != number of target columns {len(target_cols)}."
        )

    if mode == 'ensemble':
        w = average_last_k(weights_by_era, k)
        return np.dot(df[target_cols].to_numpy(dtype=float), w)

    # per-era mode
    # Precompute fallback
    if era_fallback == 'ensemble':
        fallback_w = average_last_k(weights_by_era, k)
    elif era_fallback == 'last':
        fallback_w = np.array(weights_by_era[sorted(weights_by_era.keys())[-1]], dtype=float)
    elif era_fallback == 'equal':
        fallback_w = np.ones(len(target_cols), dtype=float) / len(target_cols)
    else:
        raise ValueError("era_fallback must be one of: ensemble, last, equal")

    # Compute per-era
    out = np.empty(len(df), dtype=float)
    if era_col not in df.columns:
        raise SystemExit(f"Era column '{era_col}' not found in input data")

    for era, g in df.groupby(era_col):
        w_list: Optional[List[float]] = weights_by_era.get(str(era))
        w = np.array(w_list, dtype=float) if w_list is not None else fallback_w
        # Fix indexing issue by using positional indices
        mask = df[era_col] == era
        out[mask] = np.dot(g[target_cols].to_numpy(dtype=float), w)
    return pd.Series(out, index=df.index)


def main():
    ap = argparse.ArgumentParser(description="Build adaptive_target for a parquet using learned weights")
    ap.add_argument('--input', required=True, help='Input parquet path')
    ap.add_argument('--output', required=True, help='Output parquet path to write with adaptive_target')
    ap.add_argument('--weights-file', default='cache/weights_by_era_full.json', help='JSON file with per-era weights')
    ap.add_argument('--targets-prefix', default='target', help='Prefix to identify target columns')
    ap.add_argument('--era-col', default='era', help='Era column name')
    ap.add_argument('--mode', choices=['ensemble','per-era'], default='ensemble', help='How to apply weights')
    ap.add_argument('--k', type=int, default=10, help='For ensemble mode or fallback, average last K eras')
    ap.add_argument('--era-fallback', choices=['ensemble','last','equal'], default='ensemble', help='Fallback when era weight missing (per-era mode)')
    ap.add_argument('--row-limit', type=int, default=None, help='Optional row limit for quick tests')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    weights_by_era = load_weights(args.weights_file)
    df = pd.read_parquet(args.input)
    if args.row_limit is not None:
        df = df.head(args.row_limit)

    adaptive = build_adaptive_target(
        df,
        weights_by_era,
        targets_prefix=args.targets_prefix,
        era_col=args.era_col,
        mode=args.mode,
        k=args.k,
        era_fallback=args.era_fallback,
    )
    df = df.copy()
    df['adaptive_target'] = adaptive
    df.to_parquet(args.output, index=False)
    print(f"Wrote {args.output} rows={len(df)} mode={args.mode} k={args.k}")


if __name__ == '__main__':
    main()
