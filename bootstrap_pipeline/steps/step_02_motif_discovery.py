# bootstrap_pipeline/steps/step_02_motif_discovery.py
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
from collections import defaultdict
from bootstrap_pipeline.utils.utils import reduce_mem_usage

def setup_logging(log_file):
    """Initializes logging to both file and console for a specific run."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    stdout_only = bool(os.environ.get("PIPELINE_LOG_TO_STDOUT_ONLY", "").strip())
    handlers = [logging.StreamHandler(sys.stdout)]
    if not stdout_only:
        handlers.insert(0, logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )

class MotifDiscovery:
    """
    Discovers predictive feature motifs using a sliding window approach,
    inspired by sequence alignment tools like BLAST.
    """
    def __init__(self, motif_lengths: list[int], feature_names: list[str]):
        if not isinstance(motif_lengths, list) or not motif_lengths:
            raise ValueError("motif_lengths must be a non-empty list of integers.")
        self.motif_lengths = sorted(list(set(motif_lengths)))
        self.feature_names = feature_names
        # The key will be a tuple: (start_feature_index, (value1, value2, ...))
        self.motif_stats = defaultdict(lambda: {'sum_target': 0.0, 'count': 0})

    def update_with_row(self, feature_sequence: np.ndarray, target_value: float):
        """
        Processes a single row of features, extracting all possible motifs
        of the configured lengths and updating their statistics.
        """
        if not np.isfinite(target_value):
            return # Do not learn from rows with non-finite targets

        for length in self.motif_lengths:
            if len(feature_sequence) < length:
                continue
            # Create a sliding window over the feature sequence
            for i in range(len(feature_sequence) - length + 1):
                motif_values = tuple(feature_sequence[i : i + length])
                # The key now includes the starting feature index to resolve names later
                motif_key = (i, motif_values)
                
                stats = self.motif_stats[motif_key]
                stats['sum_target'] += target_value
                stats['count'] += 1

    def get_discovered_motifs(self, top_k: int, min_count: int):
        """
        Analyzes the collected motif statistics and returns the top_k most
        significant motifs.
        """
        if not self.motif_stats:
            return []

        significant_motifs = []
        for motif_key, stats in self.motif_stats.items():
            count = stats['count']
            if count < min_count:
                continue
            
            start_index, motif_values = motif_key
            motif_feature_names = self.feature_names[start_index : start_index + len(motif_values)]
            
            avg_target = stats['sum_target'] / count
            # Score could be more sophisticated, but this is a good start.
            # It rewards high-average targets and penalizes low-count motifs slightly.
            score = avg_target * np.log1p(count)

            significant_motifs.append({
                'motif_features': motif_feature_names,
                'motif_values': [float(x) for x in motif_values],
                'score': float(score),
                'avg_target': float(avg_target),
                'count': int(count)
            })
        
        # Sort by score in descending order and take the top k
        significant_motifs.sort(key=lambda x: x['score'], reverse=True)
        
        return significant_motifs[:top_k]

def run(
    input_file: str,
    target_col: str,
    output_motifs_file: str,
    feature_limit: int | None = None,
    row_limit: int | None = None,
    cache_dir: str | None = None,
    motif_lengths: str = "3,4", # Comma-separated string
    min_motif_count: int = 10,
    top_k_motifs: int = 100,
    batch_size: int = 25000,
    last_n_eras: int | None = None,
    era_col: str = "era",
    **kwargs,
):
    run_dir = Path(output_motifs_file).parent
    log_file = run_dir / "logs.log"
    setup_logging(log_file)

    logging.info("Running BLAST-inspired Motif Discovery...")

    # --- Caching Logic (adapted for motif discovery params) ---
    def _make_param_key() -> str:
        h = hashlib.sha256()
        parts = {
            'target_col': target_col,
            'feature_limit': feature_limit,
            'row_limit': row_limit,
            'motif_lengths': motif_lengths,
            'min_motif_count': min_motif_count,
            'top_k_motifs': top_k_motifs,
            'last_n_eras': last_n_eras,
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

    if cache_dir:
        cache_dir_path = Path(cache_dir)
        param_folder = cache_dir_path / f"md_{_make_param_key()}"
        param_folder.mkdir(parents=True, exist_ok=True)
        cache_path = param_folder / f"motifs_{_make_input_key()}.json"
        if cache_path.exists():
            try:
                shutil.copyfile(cache_path, output_motifs_file)
                logging.info("Cache hit: %s -> %s", cache_path, output_motifs_file)
                return
            except Exception as e:
                logging.warning("Failed to use cache (%s), proceeding to compute.", e)

    # --- Data Loading and Processing ---
    pf = pq.ParquetFile(input_file)
    all_columns = [field.name for field in pf.schema]
    feature_columns = [col for col in all_columns if col.startswith('feature')]
    if feature_limit is not None:
        feature_columns = feature_columns[:feature_limit]
    
    logging.info(
        f"Motif Discovery setup: features={len(feature_columns)} (limit={feature_limit}), "
        f"row_limit={row_limit}, motif_lengths={motif_lengths}"
    )

    # Parse motif_lengths from string
    try:
        parsed_motif_lengths = [int(x.strip()) for x in motif_lengths.split(',')]
    except ValueError:
        raise ValueError("motif_lengths must be a comma-separated list of integers.")

    # Pass feature names to the discovery class
    motif_discovery = MotifDiscovery(
        motif_lengths=parsed_motif_lengths, feature_names=feature_columns
    )

    processed_rows = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=all_columns):
        batch_df = batch.to_pandas()
        # Optional era filtering logic can be added here if needed
        batch_df = reduce_mem_usage(batch_df, _verbose=False)

        if batch_df.empty:
            continue

        if row_limit is not None and processed_rows >= row_limit:
            break
        if row_limit is not None:
            remaining = row_limit - processed_rows
            if len(batch_df) > remaining:
                batch_df = batch_df.iloc[:remaining]

        # Process rows
        feature_matrix = batch_df[feature_columns].values
        target_vector = batch_df[target_col].values
        
        for i in range(len(batch_df)):
            motif_discovery.update_with_row(feature_matrix[i], target_vector[i])
            
        processed_rows += len(batch_df)
        logging.info(f"Processed {processed_rows} rows...")

    logging.info("Finished processing rows. Analyzing discovered motifs...")
    
    motifs = motif_discovery.get_discovered_motifs(
        top_k=top_k_motifs, min_count=min_motif_count
    )

    logging.info(f"Found {len(motifs)} significant motifs.")

    with open(output_motifs_file, 'w') as f:
        json.dump(motifs, f, indent=2)

    # --- Caching Output ---
    if cache_dir and 'cache_path' in locals() and not cache_path.exists():
        try:
            shutil.copyfile(output_motifs_file, cache_path)
            logging.info("Wrote cache: %s", cache_path)
        except Exception as e:
            logging.warning("Failed to write cache copy: %s", e)

    logging.info("Motif Discovery complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BLAST-inspired Motif Discovery")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--output-motifs-file", required=True)
    parser.add_argument("--feature-limit", type=int)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--motif-lengths", type=str, default="3,4", help="Comma-separated list of motif lengths to search for.")
    parser.add_argument("--min-motif-count", type=int, default=10, help="Minimum times a motif must appear to be considered.")
    parser.add_argument("--top-k-motifs", type=int, default=100, help="Number of top motifs to save.")
    parser.add_argument("--batch-size", type=int, default=25000)
    # Keeping these for compatibility, though they are not used in the core logic
    parser.add_argument("--last-n-eras", type=int)
    parser.add_argument("--era-col", type=str, default="era")

    args = parser.parse_args()

    run(
        input_file=args.input_file,
        target_col=args.target_col,
        output_motifs_file=args.output_motifs_file,
        feature_limit=args.feature_limit,
        row_limit=args.row_limit,
        cache_dir=args.cache_dir,
        motif_lengths=args.motif_lengths,
        min_motif_count=args.min_motif_count,
        top_k_motifs=args.top_k_motifs,
        batch_size=args.batch_size,
        last_n_eras=args.last_n_eras,
        era_col=args.era_col,
    )
