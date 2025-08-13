# bootstrap_pipeline/steps/step_03_feature_engineering.py

import logging
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import json
import sys
from pathlib import Path
from ..utils.utils import reduce_mem_usage


def setup_logging(log_file: str | os.PathLike):
    """Initializes logging to both file and console for a specific run."""
    # Remove all handlers associated with the root logger object.
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
    relationships_file: str,
    output_file: str,
    max_features: int,
    row_limit: int | None = None,
    yolo_mode: bool = False,
    **kwargs,
):
    # Ensure run dir and logging like other steps
    run_dir = Path(output_file).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "logs.log"
    setup_logging(log_file)

    logging.info("Running Feature Engineering...")
    logging.info("Input: %s", input_file)
    logging.info("Relationships: %s", relationships_file)
    logging.info("Output: %s", output_file)
    logging.info(
        "Params: max_features=%s, row_limit=%s, yolo_mode=%s",
        max_features,
        row_limit,
        yolo_mode,
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input parquet file not found: {input_file}")
    if not os.path.exists(relationships_file):
        raise FileNotFoundError(f"Relationships JSON file not found: {relationships_file}")

    with open(relationships_file, 'r') as f:
        relationships = json.load(f)

    if not isinstance(relationships, list):
        raise ValueError("relationships_file must contain a JSON list of relationships")
    if max_features is not None and max_features > 0:
        relationships = relationships[:max_features]

    logging.info("Loaded %d relationships (capped)", len(relationships))

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
        for i, rel in enumerate(relationships):
            try:
                feat1, feat2 = rel['feature1'], rel['feature2']
                strength = rel.get('strength', 1.0)
            except Exception as e:
                raise ValueError(
                    f"Relationship at index {i} must have keys 'feature1','feature2','strength'"
                ) from e

            if feat1 in batch_df.columns and feat2 in batch_df.columns:
                interaction_name = f"path_{i:02d}_{feat1[-4:]}x{feat2[-4:]}"
                batch_df[interaction_name] = (
                    batch_df[feat1] * batch_df[feat2] * float(strength)
                ).astype('float32')
                if interaction_name not in new_feature_names:
                    new_feature_names.append(interaction_name)

                # YOLO mode: add ratio and diff features for early relationships to prevent explosion
                if yolo_mode:
                    if i < 30:  # cap ratio features to first 30 relationships
                        ratio_name = f"path_{i:02d}_ratio_{feat1[-4:]}_{feat2[-4:]}"
                        batch_df[ratio_name] = (
                            batch_df[feat1] / (batch_df[feat2].abs() + 1e-6) * float(strength)
                        ).astype('float32')
                        if ratio_name not in new_feature_names:
                            new_feature_names.append(ratio_name)
                    if i < 15:  # cap diff features to first 15 relationships
                        diff_name = f"path_{i:02d}_diff_{feat1[-4:]}_{feat2[-4:]}"
                        batch_df[diff_name] = (
                            (batch_df[feat1] - batch_df[feat2]) * float(strength)
                        ).astype('float32')
                        if diff_name not in new_feature_names:
                            new_feature_names.append(diff_name)

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

    # Save the new feature names to a file in the run directory
    features_list_path = run_dir / "new_feature_names.json"
    with open(features_list_path, "w") as f:
        json.dump(new_feature_names, f, indent=2)

    logging.info(
        "Feature Engineering complete. Rows processed=%s, new features created=%s",
        processed_rows,
        len(new_feature_names),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--relationships-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-features", type=int, required=True)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--yolo-mode", action="store_true")

    args = parser.parse_args()

    run(
        input_file=args.input_file,
        relationships_file=args.relationships_file,
        output_file=args.output_file,
        max_features=args.max_features,
        row_limit=args.row_limit,
        yolo_mode=bool(args.yolo_mode),
    )