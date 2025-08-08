# bootstrap_pipeline/steps/step_03_feature_engineering.py

import logging
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import json
from ..utils.utils import reduce_mem_usage

def run(input_file: str, relationships_file: str, output_file: str, max_features: int, row_limit: int | None = None, yolo_mode: bool = False, **kwargs):
    logging.info("Running Feature Engineering...")

    with open(relationships_file, 'r') as f:
        relationships = json.load(f)

    relationships = relationships[:max_features]

    pf = pq.ParquetFile(input_file)
    writer = None
    new_feature_names = []

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

        for i, rel in enumerate(relationships):
            feat1, feat2 = rel['feature1'], rel['feature2']
            strength = rel['strength']

            if feat1 in batch_df.columns and feat2 in batch_df.columns:
                interaction_name = f"path_{i:02d}_{feat1[-4:]}x{feat2[-4:]}"
                batch_df[interaction_name] = (batch_df[feat1] * batch_df[feat2] * strength).astype('float32')
                if interaction_name not in new_feature_names:
                    new_feature_names.append(interaction_name)

                # YOLO mode: add ratio and diff features for early relationships to prevent explosion
                if yolo_mode:
                    if i < 30:  # cap ratio features to first 30 relationships
                        ratio_name = f"path_{i:02d}_ratio_{feat1[-4:]}_{feat2[-4:]}"
                        batch_df[ratio_name] = (batch_df[feat1] / (batch_df[feat2].abs() + 1e-6) * strength).astype('float32')
                        if ratio_name not in new_feature_names:
                            new_feature_names.append(ratio_name)
                    if i < 15:  # cap diff features to first 15 relationships
                        diff_name = f"path_{i:02d}_diff_{feat1[-4:]}_{feat2[-4:]}"
                        batch_df[diff_name] = ((batch_df[feat1] - batch_df[feat2]) * strength).astype('float32')
                        if diff_name not in new_feature_names:
                            new_feature_names.append(diff_name)

        table = pa.Table.from_pandas(batch_df)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
        processed_rows += len(batch_df)

    if writer:
        writer.close()

    # Save the new feature names to a file in the run directory
    run_dir = os.path.dirname(output_file)
    features_list_path = os.path.join(run_dir, "new_feature_names.json")
    with open(features_list_path, "w") as f:
        json.dump(new_feature_names, f)

    logging.info(f"Feature Engineering complete. Created {len(new_feature_names)} new features.")