#!/usr/bin/env python3
"""
Blend ensemble of adaptive target approaches.
Trains a simple linear blend between:
  - combo-adaptive (Stage 1 discovery weights)
  - meta-adaptive (target_preference_meta_learning)
Selects weights via temporal CV on training eras and applies to validation/live.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr


def load_series_from_parquet(path: str, col: str) -> pd.Series:
	pf = pq.ParquetFile(path)
	vals: list[float] = []
	for batch in pf.iter_batches(columns=[col], batch_size=200_000):
		df = batch.to_pandas()
		if col not in df.columns:
			raise ValueError(f"Column {col} not in {path}")
		vals.extend(df[col].astype(float).tolist())
	return pd.Series(vals, name=col)


def era_iterator(path: str, cols: List[str]) -> tuple[np.ndarray, pd.DataFrame]:
	pf = pq.ParquetFile(path)
	eras: list = []
	frames: list[pd.DataFrame] = []
	for batch in pf.iter_batches(columns=['era'] + cols, batch_size=200_000):
		df = batch.to_pandas()
		frames.append(df[cols])
		eras.extend(df['era'].tolist())
	return np.asarray(eras), pd.concat(frames, ignore_index=True)


def temporal_blend(train_file: str,
				   combo_file: str,
				   meta_file: str,
				   out_file: str,
				   n_splits: int = 5,
				   gap_eras: int = 5) -> dict:
	# Load eras and both adaptive target series aligned to train_file order
	eras, _ = era_iterator(train_file, cols=['era'])
	combo = load_series_from_parquet(combo_file, 'adaptive_target').to_numpy()
	meta = load_series_from_parquet(meta_file, 'adaptive_target').to_numpy()

	if combo.shape[0] != meta.shape[0]:
		n = min(combo.shape[0], meta.shape[0])
		combo = combo[:n]
		meta = meta[:n]
		eras = eras[:n]

	unique_eras = np.unique(eras)
	total_eras = len(unique_eras)
	eras_per_split = max(1, total_eras // n_splits)

	weights = []
	fold_stats = []
	for i in range(n_splits):
		val_start = i * eras_per_split
		val_end = min((i + 1) * eras_per_split, total_eras)
		val_eras = set(unique_eras[val_start:val_end])
		train_end = max(0, val_start - gap_eras)
		train_eras = set(unique_eras[:train_end])
		if not train_eras:
			continue
		tr_mask = np.isin(eras, list(train_eras))
		Xtr = np.stack([combo[tr_mask], meta[tr_mask]], axis=1)
		# Fit simple ridge-free closed-form least squares to maximize Spearman proxy
		# We can't maximize rank corr directly; use z-scored OLS toward z-score of combo (proxy)
		# Here, we use variance scaling so weights sum to 1 non-negatively.
		# Constrained weights: w >=0, sum w =1 — use simple grid search over w in [0,1].
		best_w = 0.5
		best_corr = -1.0
		# Build a weak proxy target = average of inputs (acts as stability anchor)
		y_proxy = (Xtr[:, 0] + Xtr[:, 1]) / 2.0
		y_rank = pd.Series(y_proxy).rank(pct=True).to_numpy()
		for w in np.linspace(0.0, 1.0, 21):
			blend = w * Xtr[:, 0] + (1 - w) * Xtr[:, 1]
			sr = spearmanr(y_rank, blend)
			if hasattr(sr, 'statistic'):
				r = sr.statistic  # type: ignore
			elif isinstance(sr, tuple):
				r = sr[0]
			else:
				r = sr
			r = 0.0 if (r is None or not isinstance(r, (int, float)) or not np.isfinite(r)) else float(r)
			if r > best_corr:
				best_corr = r
				best_w = float(w)
		weights.append(best_w)
		fold_stats.append({'fold': i+1, 'w_combo': best_w, 'w_meta': 1.0 - best_w, 'proxy_corr': best_corr})

	# Final weight = median of folds for robustness
	if not weights:
		w_final = 0.5
	else:
		w_final = float(np.median(weights))

	# Apply to full sequence and write out
	blended = w_final * combo + (1.0 - w_final) * meta
	# Write alongside a minimal schema
	pf = pq.ParquetFile(train_file)
	writer = None
	idx = 0
	total = blended.size
	for batch in pf.iter_batches(columns=['era'], batch_size=200_000):
		if idx >= total:
			break
		df_era = batch.to_pandas()
		n = len(df_era)
		take = min(n, total - idx)
		if take <= 0:
			break
		# Truncate DataFrame to 'take' rows to align with available blended values
		if take != n:
			df_era = df_era.iloc[:take].copy()
		else:
			df_era = df_era.copy()
		df_era['adaptive_target_ensemble'] = blended[idx: idx + take]
		idx += take
		import pyarrow as pa
		table = pa.Table.from_pandas(df_era)
		if writer is None:
			writer = pq.ParquetWriter(out_file, table.schema)
		writer.write_table(table)
	if writer:
		writer.close()

	result = {
		'weights': {'combo': w_final, 'meta': 1.0 - w_final},
		'folds': fold_stats,
		'output': out_file,
	}
	(Path(out_file).with_suffix('.json')).write_text(json.dumps(result, indent=2))
	return result


def main():
	import argparse
	ap = argparse.ArgumentParser(description='Blend combo and meta adaptive targets')
	ap.add_argument('--train-file', default='v5.0/train.parquet')
	ap.add_argument('--combo-file', default='pipeline_runs/test_improved_td/01_adaptive_targets.parquet')
	ap.add_argument('--meta-file', default='v5.0/train_adaptive_meta.parquet')
	ap.add_argument('--out-file', default='v5.0/train_adaptive_ensemble.parquet')
	ap.add_argument('--splits', type=int, default=5)
	ap.add_argument('--gap', type=int, default=5)
	args = ap.parse_args()

	res = temporal_blend(args.train_file, args.combo_file, args.meta_file, args.out_file, args.splits, args.gap)
	print('Ensemble weights:', res['weights'])
	print('Saved to:', res['output'])


if __name__ == '__main__':
	main()

