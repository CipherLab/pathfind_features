# bootstrap_pipeline/steps/step_01_target_discovery.py

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import json
import logging
from ..utils.utils import reduce_mem_usage
from bootstrap_pipeline.bootstrap.target_discovery import WalkForwardTargetDiscovery
from bootstrap_pipeline.utils.utils import reduce_mem_usage

def run(input_file: str, features_json_file: str, output_file: str, discovery_file: str, skip_walk_forward: bool = False,
    max_eras: int | None = None, row_limit: int | None = None, target_limit: int | None = None, **kwargs):
    """
    Performs the Target Bootstrap Discovery stage.
    This function contains the core logic from the original fast_target_bootstrap.py.
    """
    logging.info(f"Running Target Bootstrap Discovery...")
    logging.info(f"Input: {input_file}")
    logging.info(f"Output: {output_file}")
    
    pf = pq.ParquetFile(input_file)
    all_columns = [field.name for field in pf.schema]
    
    target_columns = [col for col in all_columns if col.startswith('target')]
    if target_limit is not None:
        target_columns = target_columns[:target_limit]
    with open(features_json_file, 'r') as f:
        features_json = json.load(f)
    feature_columns = features_json['feature_sets']['medium']
    
    logging.info(f"Using {len(target_columns)} targets and {len(feature_columns)} features")

    era_df = pd.read_parquet(input_file, columns=['era'])
    unique_eras = sorted(era_df['era'].unique())
    if max_eras is not None:
        unique_eras = unique_eras[:max_eras]

    target_discovery = WalkForwardTargetDiscovery(target_columns, 20)

    if skip_walk_forward:
        logging.warning("---\n⏩ SKIPPING WALK-FORWARD DISCOVERY ⏩\nUsing equal weights for all eras. This is for parameter tuning ONLY.\n---")
        for current_era in unique_eras:
            target_discovery.era_weights[current_era] = np.ones(len(target_columns)) / len(target_columns)
    else:
        for i, current_era in enumerate(unique_eras):
            if i < 20:
                target_discovery.era_weights[current_era] = np.ones(len(target_columns)) / len(target_columns)
                continue
            
            history_eras = unique_eras[max(0, i-50):i]
            history_df = pd.read_parquet(
                input_file, 
                filters=[('era', 'in', history_eras)],
                columns=target_columns + feature_columns + ['era']
            )
            history_df = reduce_mem_usage(history_df, _verbose=False)
            
            optimal_weights = target_discovery.discover_weights_for_era(
                current_era, history_df, feature_columns
            )
            target_discovery.era_weights[current_era] = optimal_weights
            
            del history_df
            import gc
            gc.collect()
            
            if i % 50 == 0:
                logging.info(f"Processed {i+1}/{len(unique_eras)} eras for target discovery")

    # Create adaptive targets and save
    writer = None
    processed_rows = 0
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
        
        import pyarrow as pa
        table = pa.Table.from_pandas(batch_df)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
        processed_rows += len(batch_df)

    if writer:
        writer.close()

    with open(discovery_file, 'w') as f:
        json.dump({str(k): v.tolist() for k, v in target_discovery.era_weights.items()}, f, indent=2)

    logging.info("Target Bootstrap Discovery complete.")
