# bootstrap_pipeline/steps/step_01_target_discovery.py

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import json
import logging
import os
import time
import sys
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
from bootstrap_pipeline.bootstrap.target_discovery import WalkForwardTargetDiscovery
from bootstrap_pipeline.utils.utils import reduce_mem_usage

# Bump this when changing discovery/caching behavior to invalidate old cache entries
ALGO_VERSION = "td-v1.4"

def setup_logging(log_file):
    """Initializes logging to both file and console for a specific run."""
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # If an outer process (API/pipeline runner) is already capturing stdout to the same log file,
    # avoid double-logging by only emitting to stdout here. That outer process will write to file.
    stdout_only = bool(os.environ.get("PIPELINE_LOG_TO_STDOUT_ONLY", "").strip())
    handlers = [logging.StreamHandler(sys.stdout)]
    if not stdout_only:
        handlers.insert(0, logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )

def _file_md5(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute MD5 of a file by streaming chunks (memory-safe for large parquet)."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run(
    input_file: str,
    features_json_file: str,
    output_file: str,
    discovery_file: str,
    skip_walk_forward: bool = False,
    max_eras: int | None = None,
    row_limit: int | None = None,
    row_limit_per_era: int | None = None,
    target_limit: int | None = None,
    cache_dir: str | None = None,
    force_recache: bool = False,
    no_cache: bool = True,
    td_warmup_eras: int | None = None,
    **kwargs,
):
    """
    Performs the Target Bootstrap Discovery stage.
    This function contains the core logic from the original fast_target_bootstrap.py.
    """
    run_dir = Path(output_file).parent
    log_file = run_dir / "logs.log"
    setup_logging(log_file)

    logging.info(f"Running Target Bootstrap Discovery (algo={ALGO_VERSION})...")
    logging.info(f"Input: {input_file}")
    logging.info(f"Output: {output_file}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input parquet file not found: {input_file}")
    if not os.path.exists(features_json_file):
        raise FileNotFoundError(f"Features JSON file not found: {features_json_file}")

    pf = pq.ParquetFile(input_file)
    all_columns = [field.name for field in pf.schema]

    target_columns = [col for col in all_columns if col.startswith('target')]
    if not target_columns:
        raise ValueError("No columns starting with 'target' were found in the input file.")
    if target_limit is not None:
        if target_limit > len(target_columns):
            logging.warning(
                "target_limit %s exceeds available targets %s; using all targets",
                target_limit,
                len(target_columns),
            )
        target_columns = target_columns[:target_limit]
    with open(features_json_file, 'r') as f:
        features_json = json.load(f)
    try:
        feature_columns = features_json['feature_sets']['medium']
    except KeyError as e:
        raise ValueError("features_json must contain feature_sets['medium']") from e
    if not feature_columns:
        raise ValueError("feature_sets['medium'] is empty")

    logging.info(f"Using {len(target_columns)} targets and {len(feature_columns)} features")
    logging.info("Tip: For long runs, enable persistent pre-cache (--td-persist-pre-cache) and set a cache dir (--td-pre-cache-dir) to accelerate future passes.")

    # -------- Cache key & early return --------
    try:
        input_hash = _file_md5(input_file)
    except Exception:
        input_hash = f"sz{os.path.getsize(input_file)}_mt{int(os.path.getmtime(input_file))}"
    try:
        features_hash = _file_md5(features_json_file)
    except Exception:
        try:
            features_hash = hashlib.md5(Path(features_json_file).read_bytes()).hexdigest()
        except Exception:
            features_hash = f"sz{os.path.getsize(features_json_file)}_mt{int(os.path.getmtime(features_json_file))}"

    # Resolve warmup configuration (allows overriding default 20 for short runs)
    configured_warmup = int(td_warmup_eras) if td_warmup_eras is not None else 20

    cache_payload = {
        "algo": ALGO_VERSION,
        "input_hash": input_hash,
        "features_hash": features_hash,
        "skip_walk_forward": bool(skip_walk_forward),
        "max_eras": int(max_eras) if max_eras is not None else None,
        "row_limit": int(row_limit) if row_limit is not None else None,
    "row_limit_per_era": int(row_limit_per_era) if row_limit_per_era is not None else None,
        "target_limit": int(target_limit) if target_limit is not None else None,
        "warmup_eras": configured_warmup,
        "targets": target_columns[: (target_limit or len(target_columns))],
    }
    cache_sig = hashlib.md5(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()
    # Resolve cache dir
    default_cache_dir = os.environ.get("TARGET_DISCOVERY_CACHE_DIR", "cache/target_discovery_cache")
    use_cache_dir = Path(cache_dir or default_cache_dir)
    use_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_parquet = use_cache_dir / f"td_{cache_sig}.parquet"
    cache_weights = use_cache_dir / f"td_{cache_sig}.weights.json"

    if force_recache and no_cache:
        logging.warning("Both force_recache and no_cache were set; proceeding with no-cache behavior (no reuse, no save).")

    use_cache = (not no_cache)
    if (not force_recache) and use_cache and (not skip_walk_forward) and cache_parquet.exists() and cache_weights.exists():
        logging.info(
            "CACHE HIT: Reusing target discovery outputs for signature %s\n  -> %s\n  -> %s",
            cache_sig,
            str(cache_parquet),
            str(cache_weights),
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(discovery_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_parquet, output_file)
        shutil.copyfile(cache_weights, discovery_file)
        logging.info("Target Bootstrap Discovery complete (from cache).")
        return

    era_df = pd.read_parquet(input_file, columns=['era'])
    unique_eras = sorted(era_df['era'].unique())
    if max_eras is not None:
        unique_eras = unique_eras[:max_eras]

    # Row-limit aware restriction: only discover weights for eras actually needed to write out.
    # If a per-era limit is provided, DO NOT prune eras using the first N rows of the file.
    if row_limit_per_era is None and row_limit is not None:
        needed = set()
        seen = 0
        for batch in pf.iter_batches(batch_size=50000, columns=['era']):
            batch_series = batch.to_pandas()['era']
            needed.update(batch_series.tolist())
            seen += len(batch_series)
            if seen >= row_limit:
                break
        needed = sorted(needed)
        before = len(unique_eras)
        unique_eras = [e for e in unique_eras if e in needed]
        logging.info(
            "Row-limit aware: restricting discovery eras from %s to %s based on first %s rows.",
            before,
            len(unique_eras),
            row_limit,
        )

    total_rows = pf.metadata.num_rows
    if row_limit_per_era is not None:
        logging.info(
            "Processing %s eras and up to %s rows per era",
            len(unique_eras),
            row_limit_per_era,
        )
    else:
        logging.info(
            f"Processing {len(unique_eras)} eras and up to {row_limit or total_rows} rows",
        )

    # Initialize discovery with optional tuning kwargs forwarded from CLI / env
    td_kwargs = {}
    # Pull from kwargs if provided (CLI wiring added below)
    for k in [
        'td_eval_mode','td_top_full_models','td_ridge_lambda','td_sample_per_era',
        'td_max_combinations','td_feature_fraction','td_num_boost_round',
        'td_max_era_cache','td_clear_cache_every','td_pre_cache_dir','td_persist_pre_cache',
        'td_use_tversky','td_tversky_k','td_tversky_alpha','td_tversky_beta','td_robust_stats',
        'td_skip_degenerate_eras','td_mad_tol'
    ]:
        if k in kwargs and kwargs[k] is not None:
            td_kwargs[k] = kwargs[k]
    # Map CLI-style names to constructor parameter names
    mapping = {
        'td_eval_mode':'eval_mode',
        'td_top_full_models':'top_full_models',
        'td_ridge_lambda':'ridge_lambda',
        'td_sample_per_era':'sample_per_era',
        'td_max_combinations':'max_combinations',
        'td_feature_fraction':'feature_fraction',
        'td_num_boost_round':'num_boost_round',
        'td_max_era_cache':'max_era_cache',
        'td_clear_cache_every':'clear_cache_every',
        'td_pre_cache_dir':'pre_cache_dir',
        'td_persist_pre_cache':'persist_pre_cache',
        'td_use_tversky':'use_tversky',
        'td_tversky_k':'tversky_k',
        'td_tversky_alpha':'tversky_alpha',
        'td_tversky_beta':'tversky_beta',
        'td_robust_stats':'robust_stats',
        'td_skip_degenerate_eras':'skip_degenerate_eras',
        'td_mad_tol':'mad_tol',
    }
    ctor_kwargs = {mapping[k]:v for k,v in td_kwargs.items()}
    # History window for selecting prior eras in discovery (default 50)
    history_window = int(kwargs.get('td_history_window', 50))
    target_discovery = WalkForwardTargetDiscovery(target_columns, 20, **ctor_kwargs)

    # --- Refactor for Resumability: Load existing weights ---
    if os.path.exists(discovery_file):
        try:
            logging.info(f"Found existing discovery file, loading weights from {discovery_file}")
            with open(discovery_file, 'r') as f:
                # Load weights and convert string keys back to original era format if needed
                # Assuming eras are strings like '0001', simple load is fine.
                loaded_weights = json.load(f)
                target_discovery.era_weights = {k: np.array(v) for k, v in loaded_weights.items()}
                logging.info(f"Successfully loaded {len(target_discovery.era_weights)} existing era weights.")
        except Exception as e:
            logging.warning(f"Could not load or parse existing discovery file, starting from scratch. Error: {e}")
    # --- End Refactor ---

    if skip_walk_forward:
        logging.warning("---\n⏩ SKIPPING WALK-FORWARD DISCOVERY ⏩\nUsing equal weights for all eras. This is for parameter tuning ONLY.\n---")
        for current_era in unique_eras:
            target_discovery.era_weights[current_era] = np.ones(len(target_columns)) / len(target_columns)
    else:
        logging.info("Starting walk-forward discovery loop...")
        start = time.time()
        # Warmup: skip discovery for the first N eras. Ensure we don't skip ALL eras on short runs.
        warmup = min(configured_warmup, max(0, len(unique_eras) - 1))
        if warmup < configured_warmup:
            logging.info(
                "Adjusted warmup from %s to %s to allow discovery with only %s eras",
                configured_warmup,
                warmup,
                len(unique_eras),
            )
        if len(unique_eras) - warmup <= 1:
            logging.warning(
                "Warmup of %s eras leaves only %s era for discovery (total eras=%s). Most weights will be equal-weight baselines.",
                warmup,
                len(unique_eras) - warmup,
                len(unique_eras),
            )
        for i, current_era in enumerate(unique_eras):
            logging.info(f"Processing era {i+1}/{len(unique_eras)}: {current_era}")
            if i < warmup:
                logging.info("Skipping weight discovery for initial eras.")
                target_discovery.era_weights[current_era] = np.ones(len(target_columns)) / len(target_columns)
                continue

            # --- Refactor for Resumability: Check if weights already exist ---
            # Eras from file may be strings, so check both original and string-cast types
            if current_era in target_discovery.era_weights or str(current_era) in target_discovery.era_weights:
                logging.info(f"Weights for era {current_era} already discovered. Skipping.")
                continue
            # --- End Refactor ---

            # Use configurable history window length
            history_eras = unique_eras[max(0, i-history_window):i]
            logging.info(f"Reading history for {len(history_eras)} eras...")
            history_df = pd.read_parquet(
                input_file,
                filters=[('era', 'in', history_eras)],
                columns=target_columns + feature_columns + ['era']
            )
            # Optional memory reduction (skip if ENV set for speed, since we sample anyway)
            if not os.environ.get("TD_SKIP_REDUCE_MEM"):
                logging.info("Reducing memory usage of history dataframe...")
                history_df = reduce_mem_usage(history_df, _verbose=False)
            logging.info("Finished reading history (%d rows).", len(history_df))

            logging.info("Discovering optimal weights...")
            optimal_weights = target_discovery.discover_weights_for_era(
                current_era, history_df, feature_columns
            )
            target_discovery.era_weights[current_era] = optimal_weights
            logging.info("Finished discovering optimal weights.")

            # --- Refactor for Resumability: Save weights incrementally ---
            try:
                with open(discovery_file, 'w') as f:
                    json.dump({str(k): v.tolist() for k, v in target_discovery.era_weights.items()}, f, indent=2)
            except Exception as e:
                logging.warning(f"Failed to save incremental weights for era {current_era}. Error: {e}")
            # --- End Refactor ---

            # Periodic cache maintenance
            target_discovery.maybe_periodic_cache_clear(i)

            del history_df
            import gc
            gc.collect()

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (len(unique_eras) - (i + 1))
                logging.info(
                    f"Processed {i+1}/{len(unique_eras)} eras for target discovery (ETA {eta/60:.1f} min)"
                )

    # Create adaptive targets and save
    writer = None
    processed_rows = 0  # global processed rows (used when row_limit is set)
    era_counts: dict = defaultdict(int) if row_limit_per_era is not None else {}
    stats_count = 0
    stats_mean = 0.0
    stats_M2 = 0.0
    stats_min = float('inf')
    stats_max = float('-inf')
    target_total_rows = row_limit or total_rows
    per_era_limit = row_limit_per_era
    for batch in pf.iter_batches(batch_size=50000, columns=all_columns):
        batch_df = batch.to_pandas()
        batch_df = reduce_mem_usage(batch_df, _verbose=False)

        # Filter to eras subset (if max_eras applied)
        if max_eras is not None:
            batch_df = batch_df[batch_df['era'].isin(set(unique_eras))]
            if batch_df.empty:
                continue

        # If per-era limit is set, cap rows per era within this batch
        if per_era_limit is not None:
            if not era_counts:
                era_counts = defaultdict(int)
            # compute remaining per era and filter accordingly
            if not batch_df.empty:
                groups = []
                for era, g in batch_df.groupby('era', sort=False):
                    if era not in unique_eras:
                        continue
                    remaining = per_era_limit - era_counts[era]
                    if remaining > 0:
                        groups.append(g.iloc[:remaining])
                if groups:
                    batch_df = pd.concat(groups, ignore_index=True)
                else:
                    batch_df = batch_df.iloc[0:0]
            if batch_df.empty:
                # Check if all eras reached their per-era limits; if so, we can stop early
                if all((era_counts.get(e, 0) >= per_era_limit) for e in unique_eras):
                    logging.info(
                        "Processed per-era limits for all %s eras (each up to %s rows)",
                        len(unique_eras),
                        per_era_limit,
                    )
                    break
                else:
                    continue

        if row_limit is not None and processed_rows >= row_limit:
            break
        if row_limit is not None:
            remaining = row_limit - processed_rows
            if len(batch_df) > remaining:
                batch_df = batch_df.iloc[:remaining]
        
        adaptive_targets = []
        for _, row in batch_df.iterrows():
            era = row['era']
            if era in target_discovery.era_weights:
                weights = target_discovery.era_weights[era]
            else:
                weights = np.ones(len(target_columns)) / len(target_columns)

            # Robust weighted aggregation: handle NaNs by renormalizing weights over available targets
            target_values = np.array([row[col] for col in target_columns], dtype=float)
            weights = np.asarray(weights, dtype=float)
            mask = np.isfinite(target_values) & np.isfinite(weights)
            if not np.any(mask):
                adaptive_target = 0.0  # fallback when row has no valid target values
            else:
                w = weights[mask]
                w_sum = w.sum()
                if w_sum <= 0 or not np.isfinite(w_sum):
                    w = np.ones(mask.sum(), dtype=float) / mask.sum()
                else:
                    w = w / w_sum
                adaptive_target = float(np.dot(target_values[mask], w))
            adaptive_targets.append(adaptive_target)
        
        batch_df['adaptive_target'] = adaptive_targets
        
        table = pa.Table.from_pandas(batch_df)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
        processed_rows += len(batch_df)
        # update per-era counters
        if per_era_limit is not None and not batch_df.empty:
            counts = batch_df['era'].value_counts()
            for era, cnt in counts.items():
                era_counts[era] += int(cnt)

        arr = np.asarray(adaptive_targets, dtype=float)
        arr_finite = arr[np.isfinite(arr)]
        batch_count = arr_finite.size
        if batch_count > 0:
            batch_mean = float(arr_finite.mean())
            # Sum of squares of differences from the batch mean
            batch_M2 = float(((arr_finite - batch_mean) ** 2).sum())
            delta = batch_mean - stats_mean
            combined = stats_count + batch_count
            stats_mean = (stats_mean * stats_count + batch_mean * batch_count) / combined
            stats_M2 += batch_M2 + (delta ** 2) * stats_count * batch_count / combined
            stats_count = combined
            stats_min = min(stats_min, float(arr_finite.min()))
            stats_max = max(stats_max, float(arr_finite.max()))

        # Break conditions
        if row_limit is not None and processed_rows >= target_total_rows:
            logging.info(f"Processed {processed_rows}/{target_total_rows} rows (100.0%)")
            break
        if per_era_limit is not None and all((era_counts.get(e, 0) >= per_era_limit) for e in unique_eras):
            total_written = sum(min(era_counts.get(e, 0), per_era_limit) for e in unique_eras)
            logging.info(
                "Processed per-era limits for all eras: %s total rows across %s eras",
                total_written,
                len(unique_eras),
            )
            break
        if processed_rows % 500000 == 0:
            logging.info(
                f"Processed {processed_rows}/{target_total_rows} rows ({processed_rows/target_total_rows*100:.1f}%)"
            )

    if writer:
        writer.close()

    # --- Refactor for Resumability: Final weight save is now redundant ---
    # The weights file is now saved incrementally within the loop.
    # --- End Refactor ---

    # Save outputs to cache for reuse (best-effort) unless caching is disabled
    if use_cache and not force_recache:
        try:
            tmp_parquet = cache_parquet.with_suffix(".parquet.tmp")
            tmp_weights = cache_weights.with_suffix(".json.tmp")
            shutil.copyfile(output_file, tmp_parquet)
            shutil.copyfile(discovery_file, tmp_weights)
            tmp_parquet.replace(cache_parquet)
            tmp_weights.replace(cache_weights)
            logging.info("CACHE SAVE: Stored outputs for signature %s in %s", cache_sig, str(use_cache_dir))
        except Exception:
            logging.warning("Failed to save cache artifacts (non-fatal).", exc_info=True)
    elif not use_cache:
        logging.info("Cache disabled (--no-cache): skipping cache save.")

    if stats_count:
        variance = stats_M2 / stats_count
        std = np.sqrt(variance)
        logging.info(
            f"Adaptive target stats: mean={stats_mean:.6f}, std={std:.6f}, min={stats_min:.6f}, max={stats_max:.6f}, n={stats_count}"
        )
    # Report average weights, including and excluding warmup eras, to avoid confusion from equal-weight warmup bleed
    try:
        # All eras (includes warmup equal-weights)
        all_w = [v for k, v in target_discovery.era_weights.items()]
        if all_w:
            avg_weights = np.mean(np.stack(all_w), axis=0)
            logging.info(
                "Average target weights (all eras): %s",
                {target_columns[i]: float(w) for i, w in enumerate(avg_weights)}
            )
        # Discovery-only eras (exclude first `warmup` eras by position in unique_eras)
        # Note: `warmup` and `unique_eras` are defined above in this function
        try:
            # Use resolved warmup count if not in local scope
            warmup_idx = int(locals().get('warmup', configured_warmup or 0))
            warmup_idx = max(0, min(len(unique_eras), warmup_idx))
            disc_eras = unique_eras[warmup_idx:]
            disc_w = [target_discovery.era_weights[e] for e in disc_eras if e in target_discovery.era_weights]
            if disc_w:
                disc_avg = np.mean(np.stack(disc_w), axis=0)
                logging.info(
                    "Average target weights (discovery-only): %s",
                    {target_columns[i]: float(w) for i, w in enumerate(disc_avg)}
                )
                # Activation frequency and conditional average (when weight > 0)
                W = np.stack(disc_w)
                act_mask = W > 1e-9
                act_freq = act_mask.mean(axis=0)
                # avoid divide by zero
                sums = W.sum(axis=0)
                counts = act_mask.sum(axis=0)
                cond_avg = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
                logging.info(
                    "Activation frequency (discovery-only): %s",
                    {target_columns[i]: float(f) for i, f in enumerate(act_freq)}
                )
                logging.info(
                    "Avg weight when active (discovery-only): %s",
                    {target_columns[i]: float(w) for i, w in enumerate(cond_avg)}
                )
        except Exception:
            pass
    except Exception:
        pass

    logging.info("Target Bootstrap Discovery complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Target Bootstrap Discovery")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--features-json-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--discovery-file", required=True)
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--max-eras", type=int)
    parser.add_argument("--row-limit", type=int, help="Global cap of rows written across all eras (may restrict eras scanned)")
    parser.add_argument("--rows-per-era", type=int, help="Cap rows written per era; does not prune eras")
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory to store/reuse cached outputs")
    parser.add_argument("--force-recache", action="store_true", help="Ignore cache and recompute (still saves to cache)")
    # Cache control: default to no-cache, allow opt-in with --cache
    parser.add_argument("--no-cache", dest="no_cache", action="store_true", default=True, help="Disable cache reuse and saving (default)")
    parser.add_argument("--cache", dest="no_cache", action="store_false", help="Enable cache reuse and saving")
    # Tuning knobs for discovery speed/quality
    parser.add_argument("--td-eval-mode", choices=["gbm_full","linear_fast","hybrid"], default="gbm_full", help="Evaluation mode for target discovery")
    parser.add_argument("--td-top-full-models", type=int, default=3, help="Top K combos to refine with full models in hybrid mode")
    parser.add_argument("--td-ridge-lambda", type=float, default=1.0, help="Ridge regularization for linear screening")
    parser.add_argument("--td-sample-per-era", type=int, default=2000, help="Rows sampled per era for discovery")
    parser.add_argument("--td-max-combinations", type=int, default=20, help="Max weight combinations to test")
    parser.add_argument("--td-feature-fraction", type=float, default=0.5, help="Feature fraction for model training")
    parser.add_argument("--td-num-boost-round", type=int, default=12, help="LightGBM boosting rounds for full models")
    parser.add_argument("--td-max-era-cache", type=int, help="Max eras to keep in preprocessing cache (0 = unlimited)")
    parser.add_argument("--td-clear-cache-every", type=int, help="Clear entire preprocessing cache every N discovery eras (0 = never)")
    parser.add_argument("--td-pre-cache-dir", type=str, help="Directory for persistent preprocessed era cache")
    parser.add_argument("--td-persist-pre-cache", action="store_true", help="Persist per-era preprocessing cache to disk")
    parser.add_argument("--td-warmup-eras", type=int, help="Number of initial eras to skip for discovery (default 20)")
    # New optional feature-projection and robustness knobs
    parser.add_argument("--td-use-tversky", action="store_true", help="Append Tversky projection features (boolean z>0 binarization)")
    parser.add_argument("--td-tversky-k", type=int, default=8, help="Number of Tversky prototypes per era")
    parser.add_argument("--td-tversky-alpha", type=float, default=0.7, help="Tversky alpha parameter")
    parser.add_argument("--td-tversky-beta", type=float, default=0.3, help="Tversky beta parameter")
    parser.add_argument("--td-robust-stats", action="store_true", help="Use robust Sharpe/IR and degenerate era triage")
    parser.add_argument("--td-history-window", type=int, default=50, help="Number of prior eras to use for discovery")
    # Era quality controls
    parser.add_argument("--td-skip-degenerate-eras", action="store_true", help="Skip eras where all targets have near-zero MAD")
    parser.add_argument("--td-mad-tol", type=float, default=1e-12, help="MAD tolerance threshold for degeneracy")

    args = parser.parse_args()

    run(
        input_file=args.input_file,
        features_json_file=args.features_json_file,
        output_file=args.output_file,
        discovery_file=args.discovery_file,
        skip_walk_forward=args.skip_walk_forward,
        max_eras=args.max_eras,
        row_limit=args.row_limit,
        row_limit_per_era=args.rows_per_era,
        target_limit=args.target_limit,
        cache_dir=args.cache_dir,
        force_recache=args.force_recache,
        no_cache=args.no_cache,
        td_eval_mode=args.td_eval_mode,
        td_top_full_models=args.td_top_full_models,
        td_ridge_lambda=args.td_ridge_lambda,
        td_sample_per_era=args.td_sample_per_era,
        td_max_combinations=args.td_max_combinations,
        td_feature_fraction=args.td_feature_fraction,
        td_num_boost_round=args.td_num_boost_round,
        td_max_era_cache=args.td_max_era_cache,
        td_clear_cache_every=args.td_clear_cache_every,
        td_pre_cache_dir=args.td_pre_cache_dir,
        td_persist_pre_cache=args.td_persist_pre_cache,
        td_warmup_eras=args.td_warmup_eras,
        td_use_tversky=args.td_use_tversky,
        td_tversky_k=args.td_tversky_k,
        td_tversky_alpha=args.td_tversky_alpha,
        td_tversky_beta=args.td_tversky_beta,
        td_robust_stats=args.td_robust_stats,
    td_history_window=args.td_history_window,
        td_skip_degenerate_eras=args.td_skip_degenerate_eras,
        td_mad_tol=args.td_mad_tol,
    )
