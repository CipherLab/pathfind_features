"""Full discovery run using the fast settings.
Processes all eras starting after a warmup and saves per-era weights to cache/weights_by_era_full.json
Run in background if desired.
"""
import os, json
import pandas as pd, numpy as np
import importlib.util
HERE='python_scripts/bootstrap'
TD_PATH=os.path.join(HERE,'target_discovery.py')
spec=importlib.util.spec_from_file_location('td',TD_PATH)
if spec is None or spec.loader is None:
    raise SystemExit('Failed to load target_discovery module spec')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
WalkForwardTargetDiscovery=mod.WalkForwardTargetDiscovery

PQ='v5.0/train.parquet'
rows_per_era=2000
history_window=20
max_features=800
warmup_eras=120

# eras
era_df=pd.read_parquet(PQ, columns=['era'])
unique=sorted(era_df['era'].unique())
start_idx=warmup_eras if warmup_eras < len(unique) else 0
process_eras=unique[start_idx:]

# columns
import pyarrow.parquet as pq
pf2 = pq.ParquetFile(PQ)
all_cols=[f.name for f in pf2.schema]
target_columns=[c for c in all_cols if c.startswith('target')]
feature_columns=[c for c in all_cols if c.startswith('feature')]
feat_use = feature_columns[:max_features]

wfd=WalkForwardTargetDiscovery(target_columns, min_history_eras=5)

# Load existing results to support resume
results = {}
cache_path = 'cache/weights_by_era_full.json'
if os.path.exists(cache_path):
    try:
        with open(cache_path) as f:
            results = json.load(f)
        if not isinstance(results, dict):
            results = {}
    except Exception:
        # Corrupt/partial file, ignore and start fresh for this session
        results = {}

# Determine eras still to process (skip already completed)
already = set(results.keys())
to_process = [e for e in process_eras if e not in already]
skipped = len(process_eras) - len(to_process)
print('Processing eras count', len(to_process), f'(skipping {skipped} already done)')

total = len(to_process)
for i, era in enumerate(to_process, start=1):
    idx = unique.index(era)
    history_eras = unique[max(0, idx-history_window):idx]
    frames=[]
    for e in history_eras:
        try:
            df=pd.read_parquet(PQ, filters=[('era','in',[e])], columns=target_columns+feat_use+['era','id'])
        except Exception:
            df=pd.read_parquet(PQ, columns=target_columns+feat_use+['era','id'])
            df=df[df['era']==e]
        if df.empty: continue
        n=min(rows_per_era,len(df))
        frames.append(df.sample(n=n, random_state=42))
    if not frames:
        print('no history for', era); continue
    history_df=pd.concat(frames, ignore_index=True)
    w = wfd.discover_weights_for_era(era, history_df, feat_use)
    results[era] = {'weights': w.tolist()}
    print(f'Era {era} done ({i}/{total})')
    # flush intermediate results every era for better visibility
    os.makedirs('cache', exist_ok=True)
    with open(cache_path,'w') as f:
        json.dump(results, f)

os.makedirs('cache', exist_ok=True)
with open(cache_path,'w') as f:
    json.dump(results, f, indent=2)
print('Saved cache/weights_by_era_full.json')
