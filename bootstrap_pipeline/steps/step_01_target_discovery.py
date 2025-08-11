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
from pathlib import Path
from bootstrap_pipeline.bootstrap.target_discovery import WalkForwardTargetDiscovery
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

def run(input_file: str, features_json_file: str, output_file: str, discovery_file: str, skip_walk_forward: bool = False,
    max_eras: int | None = None, row_limit: int | None = None, target_limit: int | None = None, **kwargs):
    """
    Performs the Target Bootstrap Discovery stage.
    This function contains the core logic from the original fast_target_bootstrap.py.
    """
    run_dir = Path(output_file).parent
    log_file = run_dir / "logs.log"
    setup_logging(log_file)

    logging.info(f"Running Target Bootstrap Discovery...")
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

    era_df = pd.read_parquet(input_file, columns=['era'])
    unique_eras = sorted(era_df['era'].unique())
    if max_eras is not None:
        unique_eras = unique_eras[:max_eras]

    total_rows = pf.metadata.num_rows
    logging.info(
        f"Processing {len(unique_eras)} eras and up to {row_limit or total_rows} rows",
    )

    target_discovery = WalkForwardTargetDiscovery(target_columns, 20)

    if skip_walk_forward:
        logging.warning("---\n⏩ SKIPPING WALK-FORWARD DISCOVERY ⏩\nUsing equal weights for all eras. This is for parameter tuning ONLY.\n---")
        for current_era in unique_eras:
            target_discovery.era_weights[current_era] = np.ones(len(target_columns)) / len(target_columns)
    else:
        logging.info("Starting walk-forward discovery loop...")
        start = time.time()
        for i, current_era in enumerate(unique_eras):
            logging.info(f"Processing era {i+1}/{len(unique_eras)}: {current_era}")
            if i < 20:
                logging.info("Skipping weight discovery for initial eras.")
                target_discovery.era_weights[current_era] = np.ones(len(target_columns)) / len(target_columns)
                continue

            history_eras = unique_eras[max(0, i-50):i]
            logging.info(f"Reading history for {len(history_eras)} eras...")
            history_df = pd.read_parquet(
                input_file,
                filters=[('era', 'in', history_eras)],
                columns=target_columns + feature_columns + ['era']
            )
            logging.info("Reducing memory usage of history dataframe...")
            history_df = reduce_mem_usage(history_df, _verbose=False)
            logging.info("Finished reading history.")

            logging.info("Discovering optimal weights...")
            optimal_weights = target_discovery.discover_weights_for_era(
                current_era, history_df, feature_columns
            )
            target_discovery.era_weights[current_era] = optimal_weights
            logging.info("Finished discovering optimal weights.")

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
    processed_rows = 0
    stats_count = 0
    stats_mean = 0.0
    stats_M2 = 0.0
    stats_min = float('inf')
    stats_max = float('-inf')
    target_total_rows = row_limit or total_rows
    for batch in pf.iter_batches(batch_size=50000, columns=all_columns):
        batch_df = batch.to_pandas()
        batch_df = reduce_mem_usage(batch_df, _verbose=False)

        # Filter to eras subset (if max_eras applied)
        if max_eras is not None:
            batch_df = batch_df[batch_df['era'].isin(set(unique_eras))]
            if batch_df.empty:
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
            
            target_values = [row[col] for col in target_columns]
            adaptive_target = np.dot(target_values, weights)
            adaptive_targets.append(adaptive_target)
        
        batch_df['adaptive_target'] = adaptive_targets
        
        table = pa.Table.from_pandas(batch_df)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
        processed_rows += len(batch_df)

        arr = np.array(adaptive_targets)
        batch_count = arr.size
        batch_mean = arr.mean()
        batch_M2 = ((arr - batch_mean) ** 2).sum()
        delta = batch_mean - stats_mean
        combined = stats_count + batch_count
        if combined > 0:
            stats_mean = (stats_mean * stats_count + batch_mean * batch_count) / combined
            stats_M2 += batch_M2 + delta ** 2 * stats_count * batch_count / combined
        stats_count = combined
        stats_min = min(stats_min, arr.min())
        stats_max = max(stats_max, arr.max())

        if processed_rows >= target_total_rows:
            logging.info(f"Processed {processed_rows}/{target_total_rows} rows (100.0%)")
            break
        if processed_rows % 500000 == 0:
            logging.info(
                f"Processed {processed_rows}/{target_total_rows} rows ({processed_rows/target_total_rows*100:.1f}%)"
            )

    if writer:
        writer.close()

    with open(discovery_file, 'w') as f:
        json.dump({str(k): v.tolist() for k, v in target_discovery.era_weights.items()}, f, indent=2)

    if stats_count:
        variance = stats_M2 / stats_count
        std = np.sqrt(variance)
        logging.info(
            f"Adaptive target stats: mean={stats_mean:.6f}, std={std:.6f}, min={stats_min:.6f}, max={stats_max:.6f}, n={stats_count}"
        )
    try:
        avg_weights = np.mean(np.stack(list(target_discovery.era_weights.values())), axis=0)
        logging.info(
            "Average target weights: %s",
            {target_columns[i]: float(w) for i, w in enumerate(avg_weights)}
        )
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
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--target-limit", type=int)

    args = parser.parse_args()

    run(
        input_file=args.input_file,
        features_json_file=args.features_json_file,
        output_file=args.output_file,
        discovery_file=args.discovery_file,
        skip_walk_forward=args.skip_walk_forward,
        max_eras=args.max_eras,
        row_limit=args.row_limit,
        target_limit=args.target_limit,
    )