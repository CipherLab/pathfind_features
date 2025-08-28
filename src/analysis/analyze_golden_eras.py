"""Analyze specific 'golden eras' from target discovery weights.

Produces basic stats and an HTML summary for the focus eras provided.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pandas as pd


def load_weights(discovery_file: str) -> dict[str, list[float]]:
    with open(discovery_file, 'r') as f:
        data = json.load(f)
    return {str(k): list(v) for k, v in data.items()}


def entropy(weights: np.ndarray, eps: float = 1e-12) -> float:
    w = weights / (weights.sum() + eps)
    return float(-(w * np.log(w + eps)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--discovery-file', required=True)
    ap.add_argument('--data-file', required=True, help='Adaptive target parquet (for era ordering)')
    ap.add_argument('--focus-eras', required=True, help='Comma separated era ids e.g. "0122,0538"')
    ap.add_argument('--output-html', default='golden_eras_analysis.html')
    args = ap.parse_args()

    weights_map = load_weights(args.discovery_file)
    focus = [e.strip() for e in args.focus_eras.split(',') if e.strip()]

    # Determine target labels count by inspecting one entry
    any_weights = next(iter(weights_map.values()))
    num_targets = len(any_weights)
    target_labels = [f'target_{i:02d}' for i in range(num_targets)]

    rows = []
    for era in focus:
        w = np.array(weights_map.get(era, []), dtype=float)
        if len(w) == 0:
            rows.append({'era': era, 'note': 'missing in discovery weights'})
            continue
        top_idx = int(np.argmax(w))
        rows.append({
            'era': era,
            'num_targets': len(w),
            'sum_weights': float(w.sum()),
            'max_weight': float(w.max()),
            'min_weight': float(w.min()),
            'avg_weight': float(w.mean()),
            'std_weight': float(w.std()),
            'entropy': entropy(w),
            'top_target_index': top_idx,
            'top_target_label': target_labels[top_idx] if top_idx < len(target_labels) else str(top_idx)
        })

    df = pd.DataFrame(rows)

    # HTML summary
    table_html = df.to_html(index=False, float_format=lambda x: f"{x:.5f}")
    html_doc = f"""
    <html><head><title>Golden Era Analysis</title>
    <style>body{{font-family:Arial;margin:1.5rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 6px}}</style>
    </head><body>
    <h1>Golden Era Weight Distribution Analysis</h1>
    <p>Discovery File: {args.discovery_file}</p>
    <p>Focus Eras: {', '.join(focus)}</p>
    {table_html}
    <h2>Interpretation Guide</h2>
    <ul>
      <li><b>Entropy</b>: Lower entropy indicates more concentrated (decisive) weighting across targets.</li>
      <li><b>max_weight</b>: High max relative to avg suggests a standout target for that era.</li>
    </ul>
    </body></html>
    """
    Path(args.output_html).write_text(html_doc, encoding='utf-8')
    print(f"Golden era analysis written to {args.output_html}")


if __name__ == '__main__':
    main()
