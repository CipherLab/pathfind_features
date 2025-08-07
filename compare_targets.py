"""Compare adaptive targets vs naive average of original targets.

Generates an HTML report summarizing feature correlations.
"""
import argparse
import json
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import html
import statistics


def load_features(features_json_path: str) -> list[str]:
    with open(features_json_path, 'r') as f:
        cfg = json.load(f)
    # Prefer medium set; fallback to any list containing "feature"
    if 'feature_sets' in cfg and 'medium' in cfg['feature_sets']:
        return cfg['feature_sets']['medium']
    for k, v in cfg.get('feature_sets', {}).items():
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
    # fallback: auto-detect later
    return []


def compute_correlations(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.Series:
    corrs = {}
    t = df[target_col]
    for c in feature_cols:
        if c not in df.columns:
            continue
        x = df[c]
        if x.nunique() < 3:
            continue
        corrs[c] = x.corr(t)
    return pd.Series(corrs).sort_values(key=lambda s: s.abs(), ascending=False)


def sample_df(parquet_path: str, row_limit: int = 250_000, columns: list[str] | None = None) -> pd.DataFrame:
    pf = pq.ParquetFile(parquet_path)
    batches = []
    remaining = row_limit
    for batch in pf.iter_batches(batch_size=50_000, columns=columns):
        pdf = batch.to_pandas()
        batches.append(pdf)
        remaining -= len(pdf)
        if remaining <= 0:
            break
    return pd.concat(batches, ignore_index=True)


def build_html(adaptive_corrs: pd.Series, naive_corrs: pd.Series, summary: dict) -> str:
    rows = []
    shared = list({*adaptive_corrs.index, *naive_corrs.index})
    for f in shared:
        a = adaptive_corrs.get(f, np.nan)
        n = naive_corrs.get(f, np.nan)
        delta = a - n if (not np.isnan(a) and not np.isnan(n)) else np.nan
        rows.append((f, a, n, delta))
    rows.sort(key=lambda r: (abs(r[1]) if not np.isnan(r[1]) else -1), reverse=True)

    def fmt(v):
        return "" if pd.isna(v) else f"{v:.4f}"

    table_rows = "\n".join(
        f"<tr><td>{html.escape(f)}</td><td>{fmt(a)}</td><td>{fmt(n)}</td><td>{fmt(d)}</td></tr>" for f, a, n, d in rows
    )
    summary_items = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(str(v))}</li>" for k, v in summary.items())
    return f"""
    <html><head><title>Adaptive Target vs Naive Comparison</title>
    <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 6px; font-size: 12px; }}
    th {{ background: #f0f0f0; position: sticky; top:0; }}
    tbody tr:nth-child(even) {{ background: #fafafa; }}
    </style></head><body>
    <h1>Adaptive Target vs Naive Average</h1>
    <h2>Summary</h2>
    <ul>{summary_items}</ul>
    <h2>Per-Feature Pearson Correlations</h2>
    <table>
      <thead><tr><th>Feature</th><th>Adaptive Corr</th><th>Naive Corr</th><th>Delta</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    </body></html>
    """


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original-data', required=True)
    ap.add_argument('--adaptive-data', required=True)
    ap.add_argument('--features-json', required=False, help='Path to features.json for feature subset')
    ap.add_argument('--output-comparison', required=True)
    ap.add_argument('--row-limit', type=int, default=250_000)
    args = ap.parse_args()

    # Sample original for targets + features
    orig_pf = pq.ParquetFile(args.original_data)
    target_cols = [f.name for f in orig_pf.schema if f.name.startswith('target')]
    if not target_cols:
        raise ValueError('No target* columns found in original data')

    features = load_features(args.features_json) if args.features_json else []
    if not features:
        features = [f.name for f in orig_pf.schema if f.name.startswith('feature')]

    needed_cols = list({*target_cols, *features})
    orig_df = sample_df(args.original_data, args.row_limit, needed_cols)

    # Adaptive file already contains adaptive_target + (likely) the same feature cols
    adaptive_df = sample_df(args.adaptive_data, args.row_limit, ['adaptive_target', *features])
    if 'adaptive_target' not in adaptive_df.columns:
        raise ValueError('adaptive_target not found in adaptive data parquet')

    # Build naive average target
    orig_df['naive_avg_target'] = orig_df[target_cols].mean(axis=1)
    # Align rows (simple approach: truncate to smallest)
    n = min(len(orig_df), len(adaptive_df))
    orig_df = orig_df.iloc[:n].reset_index(drop=True)
    adaptive_df = adaptive_df.iloc[:n].reset_index(drop=True)
    merged = pd.concat([orig_df[features + ['naive_avg_target']], adaptive_df[['adaptive_target']]], axis=1)

    adaptive_corrs = compute_correlations(merged, features, 'adaptive_target')
    naive_corrs = compute_correlations(merged, features, 'naive_avg_target')

    # Summary metrics
    deltas = []
    improved = 0
    comparable = 0
    for f in features:
        a = adaptive_corrs.get(f, np.nan)
        n0 = naive_corrs.get(f, np.nan)
        if pd.isna(a) or pd.isna(n0):
            continue
        d = abs(a) - abs(n0)
        deltas.append(d)
        if d > 0:
            improved += 1
        elif abs(d) < 0.001:
            comparable += 1

    summary = {
        'num_features_evaluated': len(adaptive_corrs),
        'features_better_adaptive': improved,
        'features_similar(|delta|<0.001)': comparable,
        'median_abs_corr_adaptive': round(statistics.median(map(abs, adaptive_corrs.values)), 5) if len(adaptive_corrs) else None,
        'median_abs_corr_naive': round(statistics.median(map(abs, naive_corrs.values)), 5) if len(naive_corrs) else None,
        'median_delta_abs_corr': round(statistics.median(deltas), 5) if deltas else None,
    }

    html_report = build_html(adaptive_corrs, naive_corrs, summary)
    Path(args.output_comparison).write_text(html_report, encoding='utf-8')
    print(f"Wrote comparison report to {args.output_comparison}")


if __name__ == '__main__':
    main()
