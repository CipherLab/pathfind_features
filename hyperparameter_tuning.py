#!/usr/bin/env python3
"""
Hyperparameter optimization for the adaptive-only model.
Tests different combinations of LightGBM parameters to find optimal settings.
"""
import argparse
from pathlib import Path
from itertools import product
import numpy as np
from typing import Optional

from data_utils import get_feature_list
from model_utils import train_and_evaluate_chunked
from search_utils import (
    get_param_grids, get_base_params, get_focused_param_sets,
    generate_focused_params, save_results
)


def hyperparameter_search(train_path: str, val_path: str, features_json: str,
                        target_col: str, output_dir: str, search_type: str = 'focused',
                        chunk_rows: int = 100_000, val_rows: int = 50_000, total_rounds: int = 200, rounds_per_chunk: int = 50, seed: int = 42, era_range: Optional[str] = None, speed_mode: str = 'fast', n_iterations: int = 15):
    """Perform hyperparameter search with chunked training."""
    
    # Adjust parameters based on speed mode
    if speed_mode == 'fast':
        chunk_rows = 50_000
        val_rows = 25_000
        total_rounds = 100
        rounds_per_chunk = 25
    elif speed_mode == 'balanced':
        chunk_rows = 100_000
        val_rows = 50_000
        total_rounds = 200
        rounds_per_chunk = 50
    elif speed_mode == 'thorough':
        chunk_rows = 250_000
        val_rows = 100_000
        total_rounds = 500
        rounds_per_chunk = 100
    
    print(f"Speed mode: {speed_mode}")
    print(f"Using chunk_rows={chunk_rows}, val_rows={val_rows}, total_rounds={total_rounds}")

    # Load features
    features = get_feature_list(features_json)
    print(f"Using {len(features)} features")

    # Get parameter spaces
    param_grids = get_param_grids()
    base_params = get_base_params(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    if search_type == 'grid':
        # Grid search - try all combinations (will be large!)
        param_combinations = list(product(*param_grids.values()))
        param_names = list(param_grids.keys())

        print(f"Starting grid search with {len(param_combinations)} combinations")

        for i, combo in enumerate(param_combinations):
            if i % 10 == 0:
                print(f"Testing combination {i+1}/{len(param_combinations)}")

            params = base_params.copy()
            for name, value in zip(param_names, combo):
                params[name] = value

            try:
                result = train_and_evaluate_chunked(params, train_path, val_path, features, target_col,
                                                   chunk_rows=chunk_rows, val_rows=val_rows, total_rounds=total_rounds, rounds_per_chunk=rounds_per_chunk, seed=seed, era_range=era_range)
                result['combination_id'] = i
                results.append(result)
            except Exception as e:
                print(f"Error with combination {i}: {e}")
                continue

    elif search_type == 'random':
        # Random search - sample random combinations
        n_samples = n_iterations  # Use configurable iterations
        print(f"Starting random search with {n_samples} samples")

        np.random.seed(42)
        for i in range(n_samples):
            if i % 3 == 0:  # More frequent progress updates
                print(f"Testing sample {i+1}/{n_samples}")

            params = base_params.copy()
            for param_name, values in param_grids.items():
                params[param_name] = np.random.choice(values)

            try:
                result = train_and_evaluate_chunked(params, train_path, val_path, features, target_col,
                                                   chunk_rows=chunk_rows, val_rows=val_rows, total_rounds=total_rounds, rounds_per_chunk=rounds_per_chunk, seed=seed, era_range=era_range)
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                print(f"Error with sample {i}: {e}")
                continue

    elif search_type == 'focused':
        # Focused search around best known parameters
        print("Starting focused search around best known parameters")

        base_param_sets = get_focused_param_sets()
        n_samples = n_iterations  # Use configurable iterations

        for i in range(n_samples):
            if i % 3 == 0:  # More frequent progress updates
                print(f"Testing focused sample {i+1}/{n_samples}")

            # Start with a base parameter set
            base_set = base_param_sets[i % len(base_param_sets)]
            params = generate_focused_params(base_set, i)

            try:
                result = train_and_evaluate_chunked(params, train_path, val_path, features, target_col,
                                                   chunk_rows=chunk_rows, val_rows=val_rows, total_rounds=total_rounds, rounds_per_chunk=rounds_per_chunk, seed=seed, era_range=era_range)
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                print(f"Error with focused sample {i}: {e}")
                continue

    # Save results
    save_results(results, output_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for adaptive model")
    parser.add_argument('--train-data', required=True, help='Path to training parquet')
    parser.add_argument('--validation-data', required=True, help='Path to validation parquet')
    parser.add_argument('--features-json', required=True, help='Path to features.json')
    parser.add_argument('--target-col', default='target', help='Target column name')
    parser.add_argument('--output-dir', default='hyperparameter_tuning_results', help='Output directory')
    parser.add_argument('--search-type', choices=['grid', 'random', 'focused'], default='focused',
                       help='Search type: grid=all combinations, random=sample, focused=around best known params')
    parser.add_argument('--n-iterations', type=int, default=15, help='Number of iterations for random/focused search')
    parser.add_argument('--era-range', help='Era range for training data (e.g., "MAX-200:MAX-75")')
    parser.add_argument('--speed-mode', choices=['fast', 'balanced', 'thorough'], default='fast',
                       help='Speed mode: fast=quick tuning, balanced=moderate, thorough=detailed')

    args = parser.parse_args()

    hyperparameter_search(
        args.train_data,
        args.validation_data,
        args.features_json,
        args.target_col,
        args.output_dir,
        args.search_type,
        era_range=args.era_range,
        speed_mode=args.speed_mode,
        n_iterations=args.n_iterations
    )


if __name__ == '__main__':
    main()
