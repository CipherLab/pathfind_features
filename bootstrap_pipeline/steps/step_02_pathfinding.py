# bootstrap_pipeline/steps/step_02_pathfinding.py

import json
import logging
import os
import numpy as np
import pyarrow.parquet as pq
import sys
from pathlib import Path
from bootstrap_pipeline.bootstrap.feature_discovery import CreativePathfindingDiscovery
from bootstrap_pipeline.utils.utils import reduce_mem_usage

def setup_logging(log_file):
    """Initializes logging to both file and console for a specific run."""
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run(
    input_file: str,
    target_col: str,
    output_relationships_file: str,
    yolo_mode: bool,
    feature_limit: int | None = None,
    row_limit: int | None = None,
    debug: bool = False,
    debug_every_rows: int = 10000,
    **kwargs,
):
    run_dir = Path(output_relationships_file).parent
    log_file = run_dir / "logs.log"
    setup_logging(log_file)

    logging.info("Running Creative Pathfinding Discovery...")

    pf = pq.ParquetFile(input_file)
    all_columns = [field.name for field in pf.schema]
    feature_columns = [col for col in all_columns if col.startswith('feature')]
    if feature_limit is not None:
        feature_columns = feature_columns[:feature_limit]
    logging.info(
        f"Pathfinding setup: features={len(feature_columns)} (limit={feature_limit}), "
        f"row_limit={row_limit}, yolo_mode={yolo_mode}, debug={debug}"
    )

    # Allow more features in yolo mode to broaden the search space
    pathfinding_discovery = CreativePathfindingDiscovery(feature_columns, max_features=120 if yolo_mode else 60)

    processed_rows = 0
    next_debug_at = debug_every_rows if debug else None
    progress_file = f"{output_relationships_file}.progress.jsonl" if debug else None
    for batch in pf.iter_batches(batch_size=25000, columns=all_columns):
        batch_df = batch.to_pandas()
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

            # In yolo mode, search deeper and try more paths per row
            if yolo_mode:
                paths = pathfinding_discovery.find_creative_paths(feature_values, target_value, max_path_length=6, n_paths=16)
            else:
                paths = pathfinding_discovery.find_creative_paths(feature_values, target_value)
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
                    # mat is at most 120x120; scanning upper triangle is fine
                    for i in range(len(feature_columns)):
                        for j in range(i + 1, len(feature_columns)):
                            s = float(mat[i, j])
                            if s >= 0.15:  # avoid noise
                                edges.append((s, i, j))
                    edges.sort(reverse=True)
                    top_edges = [
                        {
                            'f1': feature_columns[i],
                            'f2': feature_columns[j],
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
        min_strength=0.2 if yolo_mode else 0.25, top_k=80 if yolo_mode else 20
    )

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
                for i in range(len(feature_columns)):
                    for j in range(i + 1, len(feature_columns)):
                        s = float(mat[i, j])
                        if s >= 0.15:
                            edges.append((s, i, j))
                edges.sort(reverse=True)
                top_edges = [
                    {'f1': feature_columns[i], 'f2': feature_columns[j], 's': float(s)}
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
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--output-relationships-file", required=True)
    parser.add_argument("--yolo-mode", action="store_true")
    parser.add_argument("--feature-limit", type=int)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-every-rows", type=int, default=10000)

    args = parser.parse_args()

    run(
        input_file=args.input_file,
        target_col=args.target_col,
        output_relationships_file=args.output_relationships_file,
        yolo_mode=args.yolo_mode,
        feature_limit=args.feature_limit,
        row_limit=args.row_limit,
        debug=args.debug,
        debug_every_rows=args.debug_every_rows,
    )
