# bootstrap_pipeline/steps/step_03_motif_engineering.py

import logging
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import json
import sys
import numpy as np
from pathlib import Path
from ..utils.utils import reduce_mem_usage

def setup_logging(log_file: str | os.PathLike):
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

def run(
    input_file: str,
    motifs_file: str,
    output_file: str,
    max_motifs: int,
    row_limit: int | None = None,
    **kwargs,
):
    run_dir = Path(output_file).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "logs.log"
    setup_logging(log_file)

    logging.info("Running Motif Feature Engineering...")
    logging.info("Input: %s", input_file)
    logging.info("Motifs: %s", motifs_file)
    logging.info("Output: %s", output_file)
    logging.info("Params: max_motifs=%s, row_limit=%s", max_motifs, row_limit)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input parquet file not found: {input_file}")
    if not os.path.exists(motifs_file):
        raise FileNotFoundError(f"Motifs JSON file not found: {motifs_file}")

    with open(motifs_file, 'r') as f:
        motifs = json.load(f)

    if not isinstance(motifs, list):
        raise ValueError("motifs_file must contain a JSON list of motifs")
    if max_motifs is not None and max_motifs > 0:
        motifs = motifs[:max_motifs]

    logging.info("Loaded %d motifs (capped)", len(motifs))

    pf = pq.ParquetFile(input_file)
    total_rows = pf.metadata.num_rows if pf.metadata is not None else None
    writer = None
    new_feature_names: list[str] = []

    processed_rows = 0
    for batch in pf.iter_batches(batch_size=50000):
        if row_limit is not None and processed_rows >= row_limit:
            break

        batch_df = batch.to_pandas()
        batch_df = reduce_mem_usage(batch_df, _verbose=False)

        if row_limit is not None:
            remaining = row_limit - processed_rows
            if len(batch_df) > remaining:
                batch_df = batch_df.iloc[:remaining]

        # Materialize new features for this batch
        for i, motif_obj in enumerate(motifs):
            try:
                motif_features = motif_obj['motif_features']
                motif_values = motif_obj['motif_values']
                score = motif_obj.get('score', 0.0)
            except KeyError as e:
                raise ValueError(
                    f"Motif at index {i} is malformed. Missing key: {e}"
                )

            # Ensure all features for the motif exist in the dataframe
            if all(f in batch_df.columns for f in motif_features):
                # Create a unique and descriptive name for the new feature
                # e.g., motif_01_f1234_f5678_f9012
                feature_name_parts = [f[-4:] for f in motif_features]
                new_feature_name = f"motif_{i:02d}_{'_'.join(feature_name_parts)}"
                
                # Create the boolean mask for the motif match
                mask = np.full(len(batch_df), True)
                for feature, value in zip(motif_features, motif_values):
                    # Using np.isclose for safe floating point comparison
                    mask &= np.isclose(batch_df[feature], value)
                
                # Create the new binary feature
                batch_df[new_feature_name] = mask.astype(np.int8)

                if new_feature_name not in new_feature_names:
                    new_feature_names.append(new_feature_name)

        table = pa.Table.from_pandas(batch_df)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
        processed_rows += len(batch_df)

        if total_rows is not None and processed_rows % 500000 == 0:
            pct = processed_rows / (row_limit or total_rows) * 100
            logging.info(
                "Processed %d/%d rows (%.1f%%)",
                processed_rows,
                (row_limit or total_rows),
                pct,
            )

    if writer:
        writer.close()

    features_list_path = run_dir / "new_feature_names.json"
    with open(features_list_path, "w") as f:
        json.dump(new_feature_names, f, indent=2)

    logging.info(
        "Motif Feature Engineering complete. Rows processed=%s, new features created=%s",
        processed_rows,
        len(new_feature_names),
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Motif Feature Engineering")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--motifs-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-motifs", type=int, required=True)
    parser.add_argument("--row-limit", type=int)

    args = parser.parse_args()

    run(
        input_file=args.input_file,
        motifs_file=args.motifs_file,
        output_file=args.output_file,
        max_motifs=args.max_motifs,
        row_limit=args.row_limit,
    )
