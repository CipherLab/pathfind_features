# bootstrap_pipeline/steps/step_02_pathfinding.py

import json
import logging
import pyarrow.parquet as pq
from bootstrap_pipeline.bootstrap.feature_discovery import CreativePathfindingDiscovery
from bootstrap_pipeline.utils.utils import reduce_mem_usage

def run(input_file: str, target_col: str, output_relationships_file: str, yolo_mode: bool,
    feature_limit: int | None = None, row_limit: int | None = None, **kwargs):
    logging.info("Running Creative Pathfinding Discovery...")

    pf = pq.ParquetFile(input_file)
    all_columns = [field.name for field in pf.schema]
    feature_columns = [col for col in all_columns if col.startswith('feature')]
    if feature_limit is not None:
        feature_columns = feature_columns[:feature_limit]

    pathfinding_discovery = CreativePathfindingDiscovery(feature_columns, max_features=80 if yolo_mode else 40)

    processed_rows = 0
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

            paths = pathfinding_discovery.find_creative_paths(feature_values, target_value)
            pathfinding_discovery.update_relationships_from_paths(paths, feature_values, target_value)
        processed_rows += len(batch_df)
        del batch_df
        import gc
        gc.collect()

    pathfinding_discovery.decay_unused_relationships()

    relationships = pathfinding_discovery.get_discovered_relationships(
        min_strength=0.25, top_k=50 if yolo_mode else 20
    )

    with open(output_relationships_file, 'w') as f:
        json.dump(relationships, f, indent=2)

    logging.info("Creative Pathfinding Discovery complete.")