#!/usr/bin/env python3
"""
FAST Target Bootstrap - Get results quickly
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import json
import logging
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def fast_target_bootstrap(input_file, output_file, min_era=120, quick_tune=False, skip_walk_forward=False, max_tuning_eras=30):
    """
    Fast target bootstrap discovery - skip old eras, use simple evaluation
    """
    
    # Load data schema
    pf = pq.ParquetFile(input_file)
    all_columns = [field.name for field in pf.schema]
    
    # Get columns
    target_columns = [col for col in all_columns if col.startswith('target')]
    # Load feature sets and select the medium set
    with open('v5.0/features.json', 'r') as f:
        features_json = json.load(f)
    feature_columns = features_json['feature_sets']['medium']
    
    logging.info(f"Using {len(target_columns)} targets and {len(feature_columns)} features")
    
    # Get unique eras directly from the data
    unique_eras = sorted(pf.read(columns=['era'])['era'].unique().to_pylist())
    if quick_tune:
        unique_eras = unique_eras[-max_tuning_eras:]
        logging.info(f"Quick tune mode: Using last {max_tuning_eras} eras: {unique_eras}")

    # Era-specific weights
    era_weights = {}
    era_scores = {}

    if skip_walk_forward:
        logging.warning("\n---\n⏩ SKIPPING WALK-FORWARD DISCOVERY ⏩\nUsing equal weights for all eras. This is for parameter tuning ONLY.\n---")
        # Use equal weights for all eras and skip discovery
        equal_weights = (np.ones(len(target_columns)) / len(target_columns)).tolist()
        for era in unique_eras:
            era_weights[era] = equal_weights
            era_scores[era] = 0.0 # No score as we are not evaluating
    else:
        # Process data
        columns_needed = target_columns + feature_columns + ['era', 'id']
        columns_needed = [col for col in columns_needed if col in all_columns]
        
        batch_count = 0
        for batch in pf.iter_batches(batch_size=50000, columns=columns_needed):
            batch_count += 1
            df = batch.to_pandas()
            
            for era in df['era'].unique():
                try:
                    era_num = int(str(era).zfill(4))
                except (ValueError, TypeError):
                    continue #skip eras that are not numbers
                
                # Skip old eras or eras not in the tuning set
                if (quick_tune and era not in unique_eras):
                    continue
                    
                era_data = df[df['era'] == era]
                
                if len(era_data) < 100:
                    continue
                
                # Sample for speed
                if len(era_data) > 2000:
                    era_data = era_data.sample(n=2000, random_state=42)
                
                # Test simple and random combinations
                combinations_to_test = [
                    np.array([1.0] + [0.0] * (len(target_columns)-1)),  # First target only
                    np.ones(len(target_columns)) / len(target_columns),   # Equal weights
                    np.array([0.7, 0.3] + [0.0] * (len(target_columns)-2)),  # Top 2 weighted
                ]
                # Add a few random combinations to explore the space
                for _ in range(20):
                    random_weights = np.random.rand(len(target_columns))
                    random_weights /= random_weights.sum() # Normalize to sum to 1
                    combinations_to_test.append(random_weights)
                
                best_score = 0
                best_weights = np.ones(len(target_columns)) / len(target_columns)
                
                for weights in combinations_to_test:
                    if len(weights) != len(target_columns):
                        weights = np.ones(len(target_columns)) / len(target_columns)
                    
                    # Create combined target
                    combined_target = np.dot(era_data[target_columns].values, weights)

                    # NUCLEAR BUT FUNCTIONAL: Multi-feature sampling
                    best_correlation = 0
                    for _ in range(10):  # Try 10 different subsets
                        sample_features = random.sample(feature_columns, min(5, len(feature_columns)))
                        feature_combination = era_data[sample_features].mean(axis=1)
                        
                        if np.std(combined_target) > 1e-8 and np.std(feature_combination) > 1e-8:
                            corr = abs(np.corrcoef(combined_target, feature_combination)[0, 1])
                            if not np.isnan(corr):
                                best_correlation = max(best_correlation, corr)
                    
                    score = best_correlation

                    if score > best_score:
                        best_score = score
                        best_weights = weights.copy()
                
                era_weights[era] = best_weights.tolist()
                era_scores[era] = float(best_score)
                
                if best_score > 0.01:
                    logging.info(f"Era {era}: Score = {best_score:.4f}")
            
            logging.info(f"Processed batch {batch_count}, found {len(era_weights)} good eras")
    
    # Create adaptive targets with a memory-efficient streaming approach
    logging.info("Creating adaptive target column efficiently...")

    # Add the new column to the schema
    output_schema = pf.schema.to_arrow_schema().append(pa.field('adaptive_target', pa.float64()))

    with pq.ParquetWriter(output_file, output_schema) as writer:
        for batch in pf.iter_batches(batch_size=50000):
            df = batch.to_pandas()
            
            adaptive_targets = []
            for _, row in df.iterrows():
                era = row['era']
                if era in era_weights:
                    weights = np.array(era_weights[era])
                else:
                    # Default to equal weights if era not in discovery
                    weights = np.ones(len(target_columns)) / len(target_columns)
                
                target_vals = row[target_columns].values
                adaptive_target = np.dot(target_vals, weights)
                adaptive_targets.append(adaptive_target)
            
            df['adaptive_target'] = adaptive_targets
            
            # Convert back to Arrow Table to write
            table = pa.Table.from_pandas(df, schema=output_schema)
            writer.write_table(table)
    
    # Save discovery results
    discovery_file = output_file.replace('.parquet', '_discovery.json')
    results = {
        'era_weights': era_weights,
        'era_scores': era_scores,
        'target_columns': target_columns,
        'summary': {
            'eras_processed': len(era_weights),
            'mean_score': float(np.mean(list(era_scores.values()))) if era_scores else 0,
            'max_score': float(max(era_scores.values())) if era_scores else 0
        }
    }
    
    with open(discovery_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Fast bootstrap complete!")
    logging.info(f"Processed {len(era_weights)} eras")
    logging.info(f"Mean score: {results['summary']['mean_score']:.4f}")
    logging.info(f"Max score: {results['summary']['max_score']:.4f}")
    logging.info(f"Output: {output_file}")
    logging.info(f"Discovery: {discovery_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast Target Bootstrap - Get results quickly")
    parser.add_argument("input_file", nargs='?', default="v5.0/features.parquet", help="Input parquet file")
    parser.add_argument("output_file", nargs='?', default="artifacts/adaptive_target_data.parquet", help="Output parquet file")
    parser.add_argument("--quick-tune", action="store_true", help="Use a smaller subset of eras for quick tuning")
    parser.add_argument("--skip-walk-forward", action="store_true", help="Skip walk-forward discovery and use equal weights")
    parser.add_argument("--max-tuning-eras", type=int, default=30, help="Number of recent eras to use for quick tuning")
    args = parser.parse_args()

    # Your existing logic, adapted to use args
    # For example, you'll need to pass these arguments to your main function
    # and modify the function to handle them.
    
    # This is a placeholder for where you'd call your main function
    # fast_target_bootstrap(args.input_file, args.output_file, quick_tune=args.quick_tune, ...)
    print(f"Running with: {args}")
    fast_target_bootstrap(args.input_file, args.output_file, quick_tune=args.quick_tune, skip_walk_forward=args.skip_walk_forward, max_tuning_eras=args.max_tuning_eras)
