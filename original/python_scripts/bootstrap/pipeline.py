"""
Main pipeline for the fixed unified bootstrap discovery.
"""

import argparse
import json
import logging
import os
import gc
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import hashlib

from .utils import setup_logging, reduce_mem_usage
from .target_discovery import WalkForwardTargetDiscovery
from .feature_discovery import CreativePathfindingDiscovery
from .validation import run_null_hypothesis_test

def main():
    """
    The fixed unified bootstrap pipeline that keeps the good stuff
    """
    parser = argparse.ArgumentParser(description="Fixed Unified Bootstrap Discovery")
    parser.add_argument("--input-data", required=True, help="Input Parquet file")
    parser.add_argument("--output-data", required=True, help="Output file with adaptive target + relationship features")
    parser.add_argument("--feature-map-file", help="JSON file with feature columns (optional)")
    parser.add_argument("--log-file", default="logs/fixed_unified_bootstrap.log", help="Log file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--skip-early-eras", type=int, default=200, help="Skip early eras (different data regime)")
    parser.add_argument("--min-history-eras", type=int, default=30, help="Minimum eras for target discovery")
    parser.add_argument("--max-features", type=int, default=80, help="Max features for relationship discovery")
    parser.add_argument("--max-new-features", type=int, default=35, help="Max new relationship features to create - go big or go home!")
    parser.add_argument("--conservative-mode", action="store_true", help="Enable if you're scared of your own success (reduces features to 8)")
    parser.add_argument("--yolo-mode", action="store_true", help="Maximum feature creation - for when you trust your original results")
    parser.add_argument("--run-sanity-check", action="store_true", help="Run null hypothesis test")
    parser.add_argument("--quick-tune", action="store_true", help="Run on a small subset of recent eras for quick parameter tuning")
    parser.add_argument("--max-tuning-eras", type=int, default=30, help="Number of recent eras to use in quick-tune mode")
    parser.add_argument("--skip-walk-forward", action="store_true", help="Use equal weights for all eras during tuning, skipping walk-forward discovery")
    parser.add_argument("--cache-dir", default="cache/bootstrap_cache", help="Directory to store cache files")
    parser.add_argument("--force-recache", action="store_true", help="Force re-running and overwriting the cache for Stage 1")
    
    args = parser.parse_args()
    
    # Apply feature count adjustments based on mode
    if args.conservative_mode:
        args.max_new_features = min(8, args.max_new_features)
        logging.info("🙈 Conservative mode enabled: Limiting to 8 features (playing it safe)")
    elif args.yolo_mode:
        args.max_new_features = max(50, args.max_new_features) 
        logging.info("🚀 YOLO mode enabled: Maximum feature creation (trusting your original results)")
    else:
        logging.info(f"🎯 Standard mode: Creating up to {args.max_new_features} features")
    
    logging.info("🚀 Starting Fixed Unified Bootstrap Discovery")
    logging.info("📊 Remember: You already proved +0.51 correlation improvement with 40+ features!")
    
    # Load data schema
    pf = pq.ParquetFile(args.input_data)
    all_columns = [f.name for f in pf.schema]
    
    target_columns = [col for col in all_columns if col.startswith('target')]
    if args.feature_map_file and os.path.exists(args.feature_map_file):
        with open(args.feature_map_file, 'r') as f:
            feature_data = json.load(f)
            # Handle different JSON structures
            if isinstance(feature_data, dict):
                if 'feature_sets' in feature_data:
                    # Use 'all' features if available, otherwise use 'medium', otherwise 'small'
                    if 'all' in feature_data['feature_sets']:
                        feature_columns = feature_data['feature_sets']['all']
                    elif 'medium' in feature_data['feature_sets']:
                        feature_columns = feature_data['feature_sets']['medium']
                    elif 'small' in feature_data['feature_sets']:
                        feature_columns = feature_data['feature_sets']['small']
                    else:
                        # Fallback to all features from the dict
                        feature_columns = feature_data.get('features', [])
                elif 'features' in feature_data:
                    feature_columns = feature_data['features']
                else:
                    # Assume the dict itself contains feature names as keys
                    feature_columns = list(feature_data.keys())
            else:
                # Assume it's already a list of feature names
                feature_columns = feature_data
    else:
        feature_columns = [col for col in all_columns if col.startswith('feature')]
    
    logging.info(f"Found {len(target_columns)} targets and {len(feature_columns)} features")
    
    # Load era information
    era_df = pd.read_parquet(args.input_data, columns=['era'])
    unique_eras = sorted(era_df['era'].unique())
    
    if args.skip_early_eras > 0:
        unique_eras = [era for era in unique_eras if int(str(era).zfill(4)) >= args.skip_early_eras]
        logging.info(f"Processing {len(unique_eras)} eras (skipped early ones)")

    if args.quick_tune:
        unique_eras = unique_eras[-args.max_tuning_eras:]
        logging.warning(f"\n---\n⚡ QUICK TUNE MODE ⚡\nRunning on the last {len(unique_eras)} eras only.\n---")
    
    # Initialize discovery systems
    target_discovery = WalkForwardTargetDiscovery(target_columns, args.min_history_eras)
    pathfinding_discovery = CreativePathfindingDiscovery(feature_columns, max_features=args.max_features)
    
    # --- Stage 1: Walk-Forward Target Discovery ---
    logging.info("=== Stage 1: Walk-Forward Target Discovery ===")

    columns_needed = target_columns + feature_columns[:args.max_features] + ['era', 'id']
    columns_needed = [col for col in columns_needed if col in all_columns]

    # Caching logic for Stage 1
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_params = {
        "input_data": args.input_data,
        "skip_early_eras": args.skip_early_eras,
        "min_history_eras": args.min_history_eras,
        "targets": sorted(target_columns)
    }
    cache_signature = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
    cache_file = os.path.join(args.cache_dir, f"target_weights_{cache_signature}.json")

    if os.path.exists(cache_file) and not args.force_recache and not args.skip_walk_forward:
        logging.info(f"CACHE HIT: Loading era weights from {cache_file}")
        with open(cache_file, 'r') as f:
            era_weights_str_keys = json.load(f)
            target_discovery.era_weights = {int(k): np.array(v) for k, v in era_weights_str_keys.items()}
    else:
        if args.skip_walk_forward:
            logging.warning("\n---\n⏩ SKIPPING WALK-FORWARD DISCOVERY ⏩\nUsing equal weights for all eras. This is for parameter tuning ONLY.\n---")
            for current_era in unique_eras:
                target_discovery.era_weights[current_era] = np.ones(len(target_columns)) / len(target_columns)
        else:
            logging.info(f"CACHE MISS: Running walk-forward discovery. Cache file: {cache_file}")
            
            for i, current_era in enumerate(unique_eras):
                if i < args.min_history_eras:
                    target_discovery.era_weights[current_era] = np.ones(len(target_columns)) / len(target_columns)
                    continue
                
                history_eras = unique_eras[max(0, i-50):i]
                history_df = pd.read_parquet(
                    args.input_data, 
                    filters=[('era', 'in', history_eras)],
                    columns=columns_needed
                )
                history_df = reduce_mem_usage(history_df, _verbose=False)
                
                optimal_weights = target_discovery.discover_weights_for_era(
                    current_era, history_df, feature_columns[:args.max_features]
                )
                target_discovery.era_weights[current_era] = optimal_weights
                
                del history_df
                gc.collect()
                
                if i % 50 == 0:
                    logging.info(f"Processed {i+1}/{len(unique_eras)} eras for target discovery")

            # Save to cache
            logging.info(f"CACHE SAVE: Saving era weights to {cache_file}")
            era_weights_to_save = {str(k): v.tolist() for k, v in target_discovery.era_weights.items()}
            with open(cache_file, 'w') as f:
                json.dump(era_weights_to_save, f)
    
    # --- Stage 2: Creative Pathfinding Feature Discovery ---
    logging.info("=== Stage 2: Creative Pathfinding Feature Discovery ===")
    
    # Load data in batches for pathfinding discovery
    for batch in pf.iter_batches(batch_size=25000, columns=columns_needed):
        batch_df = batch.to_pandas()
        batch_df = reduce_mem_usage(batch_df, _verbose=False)
        
        if batch_df.empty:
            continue
        
        # Create adaptive targets for this batch
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
        
        # Run pathfinding discovery on adaptive targets
        for _, row in batch_df.iterrows():
            feature_values = row[feature_columns[:args.max_features]].values.astype(float)
            target_value = float(row['adaptive_target'])
            
            paths = pathfinding_discovery.find_creative_paths(feature_values, target_value)
            pathfinding_discovery.update_relationships_from_paths(paths, feature_values, target_value)
        
        del batch_df
        gc.collect()
    
    # Decay relationships after discovery
    pathfinding_discovery.decay_unused_relationships()
    
    # --- Stage 3: Sanity Check (Optional) ---
    if args.run_sanity_check:
        logging.info("=== Stage 3: Sanity Check ===")
        
        # Load a small sample for null hypothesis test
        sample_df = pd.read_parquet(args.input_data, columns=columns_needed)
        sample_df = sample_df.sample(n=min(2000, len(sample_df)), random_state=42)
        
        # Create adaptive targets for sample
        adaptive_targets = []
        for _, row in sample_df.iterrows():
            era = row['era']
            if era in target_discovery.era_weights:
                weights = target_discovery.era_weights[era]
            else:
                weights = np.ones(len(target_columns)) / len(target_columns)
            
            target_values = [row[col] for col in target_columns]
            adaptive_target = np.dot(target_values, weights)
            adaptive_targets.append(adaptive_target)
        
        sample_df['adaptive_target'] = adaptive_targets
        
        # Run null hypothesis test
        sanity_passed = run_null_hypothesis_test(
            pathfinding_discovery, sample_df, 'adaptive_target', feature_columns[:args.max_features]
        )
        
        if not sanity_passed:
            logging.error("❌ Failed sanity check! Consider reducing complexity or changing parameters.")
        
        del sample_df
        gc.collect()
    
    # --- Stage 4: Create Final Dataset ---
    logging.info("=== Stage 4: Creating Final Dataset with Features ===")
    
    # Get discovered relationships
    relationships = pathfinding_discovery.get_discovered_relationships(
        min_strength=0.25, top_k=args.max_new_features
    )
    
    logging.info(f"Creating {len(relationships)} new relationship features")
    for i, rel in enumerate(relationships):
        logging.info(f"  {i+1}. {rel['feature1']} <-> {rel['feature2']} (strength: {rel['strength']:.3f})")
    
    # Process final data
    final_data = []
    
    for batch in pf.iter_batches(batch_size=50000, columns=all_columns):
        batch_df = batch.to_pandas()
        batch_df = reduce_mem_usage(batch_df, _verbose=False)
        
        # Create adaptive targets
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
        
        # Create relationship features (with appropriate feature engineering aggression)
        for i, rel in enumerate(relationships):
            feat1, feat2 = rel['feature1'], rel['feature2']
            strength = rel['strength']
            
            if feat1 in batch_df.columns and feat2 in batch_df.columns:
                # Interaction feature (the bread and butter)
                interaction_name = f"path_{i:02d}_{feat1[-4:]}x{feat2[-4:]}"
                batch_df[interaction_name] = (
                    batch_df[feat1] * batch_df[feat2] * strength
                ).astype(np.float32)
                
                # If we're being aggressive, add ratio features too
                if not args.conservative_mode:
                    ratio_name = f"path_{i:02d}_ratio_{feat1[-4:]}_{feat2[-4:]}"
                    batch_df[ratio_name] = (
                        batch_df[feat1] / (batch_df[feat2].abs() + 1e-6) * strength
                    ).astype(np.float32)
                    
                # YOLO mode: Add even more creative combinations
                if args.yolo_mode and i < 15:  # Limit to prevent explosion
                    diff_name = f"path_{i:02d}_diff_{feat1[-4:]}_{feat2[-4:]}"
                    batch_df[diff_name] = (
                        (batch_df[feat1] - batch_df[feat2]) * strength
                    ).astype(np.float32)
        
        final_data.append(batch_df)
    
    # Combine and save
    final_df = pd.concat(final_data, ignore_index=True)
    final_df = reduce_mem_usage(final_df)
    
    output_dir = os.path.dirname(args.output_data)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    final_df.to_parquet(args.output_data, index=False)
    
    # Save discovery results
    results_file = args.output_data.replace('.parquet', '_discovery_results.json')
    results = {
        'target_discovery': {
            'era_weights': {str(k): v.tolist() for k, v in target_discovery.era_weights.items()},
            'eras_processed': len(target_discovery.era_weights)
        },
        'relationship_discovery': {
            'relationships_found': relationships,
            'feature_importance': {
                pathfinding_discovery.feature_columns[i]: float(pathfinding_discovery.feature_importance[i])
                for i in range(len(pathfinding_discovery.feature_columns))
            }
        },
        'summary': {
            'adaptive_target_created': True,
            'relationship_features_created': len(relationships),
            'total_columns': len(final_df.columns)
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"✅ Fixed unified bootstrap complete!")
    logging.info(f"📊 Output: {args.output_data}")
    logging.info(f"📋 Results: {results_file}")

if __name__ == "__main__":
    main()
