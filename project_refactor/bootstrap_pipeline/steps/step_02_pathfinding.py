# bootstrap_pipeline/steps/step_02_pathfinding.py

import json
import logging
import os
import time
import hashlib
import shutil
import numpy as np
import pyarrow.parquet as pq
import sys
from pathlib import Path
from bootstrap_pipeline.bootstrap.feature_discovery import CreativePathfindingDiscovery
from bootstrap_pipeline.utils.utils import reduce_mem_usage
from tests import setup_script_output, get_output_path, initialize_script_output, add_output_dir_arguments

def setup_logging(log_file):
    """Initializes logging to both file and console for a specific run."""
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Allow outer runner to capture stdout to file; avoid duplicate writes here
    stdout_only = bool(os.environ.get("PIPELINE_LOG_TO_STDOUT_ONLY", "").strip())
    handlers = [logging.StreamHandler(sys.stdout)]
    if not stdout_only:
        handlers.insert(0, logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )

def run(
    input_file: str,
    output_relationships_file: str,
    yolo_mode: bool,
    feature_limit: int | None = None,
    row_limit: int | None = None,
    debug: bool = False,
    debug_every_rows: int = 10000,
    cache_dir: str | None = None,
    run_sanity_check: bool = False,
    pf_feature_cap: int | None = None,
    n_paths: int | None = None,
    max_path_length: int | None = None,
    min_strength: float | None = None,
    top_k: int | None = None,
    batch_size: int = 25000,
    last_n_eras: int | None = None,
    era_col: str = "era",
    target_col: str = "adaptive_target",
    feature_tail: bool = False,
    **kwargs,
):
    run_dir = Path(output_relationships_file).parent
    log_file = run_dir / "logs.log"
    setup_logging(log_file)

    logging.info("Running Creative Pathfinding Discovery...")

    # Ensure the input parquet has at least some rows before starting (guards race with step_01)
    def _parquet_row_count(path: str) -> int:
        try:
            pf_local = pq.ParquetFile(path)
            # Sum row groups for speed
            return sum(pf_local.metadata.row_group(i).num_rows for i in range(pf_local.metadata.num_row_groups))
        except Exception:
            return 0

    max_wait_s = int(os.environ.get("PF_WAIT_FOR_INPUT_SECS", "300"))
    min_start_rows = int(os.environ.get("PF_MIN_START_ROWS", "0"))
    start_wait = time.time()
    # Wait until there is at least some data (or configured minimum rows) before starting
    while True:
        rows_now = _parquet_row_count(input_file)
        if rows_now > 0 and rows_now >= min_start_rows:
            break
        if (time.time() - start_wait) >= max_wait_s:
            break
        logging.info(
            "Waiting for input parquet to be populated... rows=%d, min=%d (%ds/%ds)",
            rows_now,
            min_start_rows,
            int(time.time() - start_wait),
            max_wait_s,
        )
        time.sleep(2)

    # Caching: split cache identity into parameter hash (folder) and input identity (file)
    def _make_param_key() -> str:
        h = hashlib.sha256()
        parts = {
            'target_col': target_col,
            'yolo': bool(yolo_mode),
            'feature_limit': feature_limit,
            'feature_tail': bool(feature_tail),
            'row_limit': row_limit,
            'pf_feature_cap': pf_feature_cap,
            'n_paths': n_paths,
            'max_path_length': max_path_length,
            'min_strength': min_strength,
            'top_k': top_k,
            'batch_size': batch_size,
            'last_n_eras': last_n_eras,
            'era_col': era_col,
        }
        h.update(json.dumps(parts, sort_keys=True).encode())
        return h.hexdigest()[:12]

    def _make_input_key() -> str:
        h = hashlib.sha256()
        try:
            st = os.stat(input_file)
            h.update(str(st.st_size).encode())
            h.update(str(int(st.st_mtime)).encode())
        except FileNotFoundError:
            pass
        return h.hexdigest()[:8]

    cache_used = False
    cache_path = None
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        # Use parameter-hashed subfolder to share cache across jobs safely
        param_folder = cache_dir_path / f"pf_{_make_param_key()}"
        param_folder.mkdir(parents=True, exist_ok=True)
        cache_path = param_folder / f"rel_{_make_input_key()}.json"
        if cache_path.exists():
            try:
                shutil.copyfile(cache_path, output_relationships_file)
                logging.info("Cache hit: %s -> %s", cache_path, output_relationships_file)
                cache_used = True
                if run_sanity_check:
                    logging.info("Running sanity check on cached relationships...")
                    # Basic existence/JSON sanity
                    with open(output_relationships_file, 'r') as cf:
                        _ = json.load(cf)
                return
            except Exception as e:
                logging.warning("Failed to use cache (%s), proceeding to compute.", e)

    pf = pq.ParquetFile(input_file)
    all_columns = [field.name for field in pf.schema]
    if last_n_eras is not None and era_col not in all_columns:
        raise ValueError(f"Requested last_n_eras but era column '{era_col}' not found in input. Columns: {all_columns[:10]}...")
    feature_columns = [col for col in all_columns if col.startswith('feature')]
    if feature_limit is not None:
        if feature_tail:
            feature_columns = feature_columns[-feature_limit:]
        else:
            feature_columns = feature_columns[:feature_limit]
    logging.info(
        f"Pathfinding setup: features={len(feature_columns)} (limit={feature_limit}), "
        f"row_limit={row_limit}, yolo_mode={yolo_mode}, debug={debug}"
    )

    # If requested, compute the last N eras present in the dataset using a light pass on the era column
    selected_eras_set = None
    if isinstance(last_n_eras, int) and last_n_eras > 0:
        from collections import deque
        logging.info("Computing last %d eras using a fast prepass on '%s'...", last_n_eras, era_col)
        dq = deque(maxlen=last_n_eras)
        prev_last = None
        for era_batch in pf.iter_batches(batch_size=max(100_000, batch_size), columns=[era_col]):
            ser = era_batch.to_pandas()[era_col]
            # Identify new era runs accounting for previous batch tail
            starts = ser != ser.shift(fill_value=prev_last)
            transitions = ser[starts]
            for v in transitions:
                dq.append(v)
            if not ser.empty:
                prev_last = ser.iloc[-1]
        selected_eras_set = set(dq)
        logging.info("Selected %d unique eras (last %d). Example tail: %s", len(selected_eras_set), last_n_eras, list(dq)[-5:] if len(dq) >= 5 else list(dq))

    # Params parity and sane defaults
    # Default internal cap: 60 (non-yolo) or 120 (yolo), unless pf_feature_cap overrides
    internal_cap = (120 if yolo_mode else 60)
    if isinstance(pf_feature_cap, int) and pf_feature_cap > 0:
        internal_cap = pf_feature_cap
    # Default path search params
    if n_paths is None:
        n_paths = 16 if yolo_mode else 8
    if max_path_length is None:
        max_path_length = 6 if yolo_mode else 4
    # Default output filtering
    if min_strength is None:
        min_strength = 0.2 if yolo_mode else 0.25
    if top_k is None:
        top_k = 80 if yolo_mode else 20

    pathfinding_discovery = CreativePathfindingDiscovery(feature_columns, max_features=internal_cap)

    processed_rows = 0
    next_debug_at = debug_every_rows if debug else None
    progress_file = f"{output_relationships_file}.progress.jsonl" if debug else None
    for batch in pf.iter_batches(batch_size=batch_size, columns=all_columns):
        batch_df = batch.to_pandas()
        if selected_eras_set is not None:
            # Filter to selected eras only
            before = len(batch_df)
            batch_df = batch_df[batch_df[era_col].isin(selected_eras_set)]
            if before and len(batch_df) == 0:
                # Skip entirely empty batches after filtering to avoid extra overhead
                continue
        batch_df = reduce_mem_usage(batch_df, _verbose=False)

        if batch_df.empty:
            continue

        if row_limit is not None and processed_rows >= row_limit:
            break
        if row_limit is not None:
            remaining = row_limit - processed_rows
            if len(batch_df) > remaining:
                batch_df = batch_df.iloc[:remaining]

        for _, row in batch_df.iterrows():
            feature_values = row[feature_columns].values.astype(float)
            target_value = float(row[target_col])

            # Search paths with configured params
            paths = pathfinding_discovery.find_creative_paths(
                feature_values,
                target_value,
                max_path_length=max_path_length,
                n_paths=n_paths,
            )
            pathfinding_discovery.update_relationships_from_paths(paths, feature_values, target_value)
        processed_rows += len(batch_df)
        del batch_df
        import gc
        gc.collect()

        # Periodic debug metrics
        if debug and next_debug_at is not None and processed_rows >= next_debug_at:
            try:
                mat = pathfinding_discovery.relationship_matrix
                # Compute off-diagonal counts > 0.1
                offdiag = mat.copy()
                np.fill_diagonal(offdiag, 0.0)
                max_strength = float(np.max(mat)) if mat.size else 0.0
                mean_strength = float(np.mean(mat)) if mat.size else 0.0
                n_over_0p1 = int(np.sum(offdiag > 0.1))
                fi = pathfinding_discovery.feature_importance
                fi_min = float(np.min(fi)) if fi.size else 0.0
                fi_max = float(np.max(fi)) if fi.size else 0.0
                n_paths = len(pathfinding_discovery.successful_paths)
                logging.info(
                    "Pathfinding debug (rows %d): matrix max=%.4f mean=%.4f offdiag>0.1=%d, "
                    "successful_paths=%d, feature_importance=[%.4f, %.4f]",
                    processed_rows,
                    max_strength,
                    mean_strength,
                    n_over_0p1,
                    n_paths,
                    fi_min,
                    fi_max,
                )
                # Write a tiny progress snapshot for UI (JSONL)
                if progress_file:
                    # Extract top edges cheaply
                    top_k = 15
                    edges = []
                    # Use actual matrix size in case internal engine caps features
                    n_mat = mat.shape[0]
                    feat_names = feature_columns[:n_mat]
                    # scan upper triangle of the actual matrix
                    for i in range(n_mat):
                        for j in range(i + 1, n_mat):
                            s = float(mat[i, j])
                            if s >= 0.15:  # avoid noise
                                edges.append((s, i, j))
                    edges.sort(reverse=True)
                    top_edges = [
                        {
                            'f1': feat_names[i],
                            'f2': feat_names[j],
                            's': float(s)
                        }
                        for s, i, j in edges[:top_k]
                    ]
                    snap = {
                        'rows': int(processed_rows),
                        'matrix_max': max_strength,
                        'matrix_mean': mean_strength,
                        'offdiag_gt_0p1': n_over_0p1,
                        'successful_paths': int(n_paths),
                        'fi_min': fi_min,
                        'fi_max': fi_max,
                        'top_edges': top_edges,
                    }
                    with open(progress_file, 'a') as pfout:
                        pfout.write(json.dumps(snap) + "\n")
            except Exception as e:
                logging.warning(f"Debug stats computation failed at rows {processed_rows}: {e}")
            finally:
                next_debug_at += debug_every_rows

    pathfinding_discovery.decay_unused_relationships()

    relationships = pathfinding_discovery.get_discovered_relationships(
        min_strength=float(min_strength), top_k=int(top_k)
    )

    # Optional sanity check
    if run_sanity_check:
        try:
            mat = pathfinding_discovery.relationship_matrix
            max_strength_val = float(np.max(mat)) if mat.size else 0.0
            sanity = {
                'rows_processed': int(processed_rows),
                'relationships_found': int(len(relationships)),
                'matrix_max': max_strength_val,
                'passed': bool(len(relationships) > 0 and max_strength_val >= (min_strength or 0.2) * 0.8),
                'threshold_used': float(min_strength or (0.2 if yolo_mode else 0.25)),
            }
            with open(f"{output_relationships_file}.sanity.json", 'w') as sf:
                json.dump(sanity, sf, indent=2)
            if not sanity['passed']:
                logging.warning("Sanity check did not pass: %s", sanity)
        except Exception as e:
            logging.warning("Sanity check failed to run: %s", e)

    # Final debug snapshot
    if debug:
        try:
            mat = pathfinding_discovery.relationship_matrix
            offdiag = mat.copy()
            np.fill_diagonal(offdiag, 0.0)
            max_strength = float(np.max(mat)) if mat.size else 0.0
            mean_strength = float(np.mean(mat)) if mat.size else 0.0
            n_over_0p1 = int(np.sum(offdiag > 0.1))
            fi = pathfinding_discovery.feature_importance
            fi_min = float(np.min(fi)) if fi.size else 0.0
            fi_max = float(np.max(fi)) if fi.size else 0.0
            n_paths = len(pathfinding_discovery.successful_paths)
            logging.info(
                "Final pathfinding stats: matrix max=%.4f mean=%.4f offdiag>0.1=%d, "
                "successful_paths=%d, feature_importance=[%.4f, %.4f], relationships=%d",
                max_strength,
                mean_strength,
                n_over_0p1,
                n_paths,
                fi_min,
                fi_max,
                len(relationships),
            )
            if progress_file:
                # Final snapshot as well
                edges = []
                n_mat = mat.shape[0]
                feat_names = feature_columns[:n_mat]
                for i in range(n_mat):
                    for j in range(i + 1, n_mat):
                        s = float(mat[i, j])
                        if s >= 0.15:
                            edges.append((s, i, j))
                edges.sort(reverse=True)
                top_edges = [
                    {'f1': feat_names[i], 'f2': feat_names[j], 's': float(s)}
                    for s, i, j in edges[:15]
                ]
                snap = {
                    'rows': int(processed_rows),
                    'matrix_max': max_strength,
                    'matrix_mean': mean_strength,
                    'offdiag_gt_0p1': n_over_0p1,
                    'successful_paths': int(n_paths),
                    'fi_min': fi_min,
                    'fi_max': fi_max,
                    'top_edges': top_edges,
                    'relationships': int(len(relationships)),
                }
                with open(progress_file, 'a') as pfout:
                    pfout.write(json.dumps(snap) + "\n")
        except Exception as e:
            logging.warning(f"Final debug stats computation failed: {e}")

    with open(output_relationships_file, 'w') as f:
        json.dump(relationships, f, indent=2)
    if cache_path and not cache_used:
        try:
            shutil.copyfile(output_relationships_file, cache_path)
            logging.info("Wrote cache: %s", cache_path)
        except Exception as e:
            logging.warning("Failed to write cache copy: %s", e)

    # Save a minimal debug dump if requested or if nothing was found
    if debug or len(relationships) == 0:
        try:
            mat = pathfinding_discovery.relationship_matrix
            offdiag = mat.copy()
            np.fill_diagonal(offdiag, 0.0)
            debug_summary = {
                'n_features': len(feature_columns),
                'rows_processed': processed_rows,
                'yolo_mode': bool(yolo_mode),
                'matrix_max': float(np.max(mat)) if mat.size else 0.0,
                'matrix_mean': float(np.mean(mat)) if mat.size else 0.0,
                'offdiag_gt_0p1': int(np.sum(offdiag > 0.1)),
                'successful_paths': int(len(pathfinding_discovery.successful_paths)),
                'feature_importance_min': float(np.min(pathfinding_discovery.feature_importance)) if pathfinding_discovery.feature_importance.size else 0.0,
                'feature_importance_max': float(np.max(pathfinding_discovery.feature_importance)) if pathfinding_discovery.feature_importance.size else 0.0,
                'relationships_found': int(len(relationships)),
            }
            debug_file = f"{output_relationships_file}.debug.json"
            with open(debug_file, 'w') as df:
                json.dump(debug_summary, df, indent=2)
            if len(relationships) == 0:
                logging.error(
                    "ZERO RELATIONSHIPS FOUND - wrote debug summary to %s",
                    os.path.abspath(debug_file),
                )
        except Exception as e:
            logging.warning(f"Failed to write debug summary: {e}")

    logging.info("Creative Pathfinding Discovery complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Creative Pathfinding Discovery")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-relationships-file", required=True)
    parser.add_argument("--yolo-mode", action="store_true")
    parser.add_argument("--feature-limit", type=int)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-every-rows", type=int, default=10000)
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--run-sanity-check", action="store_true")
    parser.add_argument("--pf-feature-cap", type=int, help="Internal feature cap used by the pathfinding engine")
    parser.add_argument("--n-paths", type=int, help="Paths to explore per row")
    parser.add_argument("--max-path-length", type=int, help="Maximum path length to explore")
    parser.add_argument("--min-strength", type=float, help="Minimum relationship strength to keep")
    parser.add_argument("--top-k", type=int, help="Top-K relationships to keep")
    parser.add_argument("--batch-size", type=int, default=25000)
    parser.add_argument("--last-n-eras", type=int, help="Restrict processing to the last N eras present in the input")
    parser.add_argument("--era-col", type=str, default="era")
    parser.add_argument("--target-col", type=str, default="adaptive_target")
    parser.add_argument("--feature-tail", action="store_true", help="When set, select the last N features instead of the first N (works with --feature-limit)")

    args = parser.parse_args()

    run(
        input_file=args.input_file,
        output_relationships_file=args.output_relationships_file,
        yolo_mode=args.yolo_mode,
        feature_limit=args.feature_limit,
        row_limit=args.row_limit,
        debug=args.debug,
        debug_every_rows=args.debug_every_rows,
    cache_dir=args.cache_dir,
    run_sanity_check=args.run_sanity_check,
    pf_feature_cap=args.pf_feature_cap,
    n_paths=args.n_paths,
    max_path_length=args.max_path_length,
    min_strength=args.min_strength,
    top_k=args.top_k,
    batch_size=args.batch_size,
    last_n_eras=args.last_n_eras,
    era_col=args.era_col,
    target_col=args.target_col
    )
