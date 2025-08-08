#!/usr/bin/env python3
"""
FIXED Bootstrap Analysis - Handles data alignment issues that cause NaN correlations
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import logging
import argparse
import os

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_predictions_and_targets_fixed(predictions_file, validation_data_file, target_col='target', era_col='era'):
    """Load predictions and merge with validation targets - FIXED VERSION"""
    
    print("=== DEBUGGING DATA LOADING ===")
    
    # Load predictions
    print(f"Loading predictions from: {predictions_file}")
    pred_df = pd.read_parquet(predictions_file)
    print(f"Predictions shape: {pred_df.shape}")
    print(f"Predictions columns: {pred_df.columns.tolist()}")
    print(f"Predictions sample:\n{pred_df.head()}")
    
    # Load validation data
    print(f"\nLoading validation data from: {validation_data_file}")
    val_df = pd.read_parquet(validation_data_file)
    val_df.reset_index(inplace=True)  # ARCH-9000: The ID is the index. Let's make it a column.
    val_df = val_df[['id', target_col, era_col]]
    
    print(f"Validation shape: {val_df.shape}")
    print(f"Validation columns: {val_df.columns.tolist()}")
    print(f"Validation sample:\n{val_df.head()}")
    
    # Check ID column types and convert if needed
    print(f"\nPrediction ID type: {pred_df['id'].dtype}")
    print(f"Validation ID type: {val_df['id'].dtype}")
    
    # Ensure ID columns are the same type
    if pred_df['id'].dtype != val_df['id'].dtype:
        print("Converting ID types to match...")
        # Convert both to string to be safe
        pred_df['id'] = pred_df['id'].astype(str)
        val_df['id'] = val_df['id'].astype(str)
    
    # Check for overlapping IDs
    pred_ids = set(pred_df['id'])
    val_ids = set(val_df['id'])
    common_ids = pred_ids.intersection(val_ids)
    print(f"\nPrediction unique IDs: {len(pred_ids)}")
    print(f"Validation unique IDs: {len(val_ids)}")
    print(f"Common IDs: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("❌ ERROR: No common IDs found between predictions and validation!")
        print("Sample prediction IDs:", list(pred_ids)[:5])
        print("Sample validation IDs:", list(val_ids)[:5])
        raise ValueError("No common IDs found!")
    
    # Merge predictions with targets
    print(f"\nMerging dataframes...")
    merged_df = pd.merge(pred_df, val_df, on='id', how='inner')
    print(f"Merged shape: {merged_df.shape}")
    
    # Check for NaN values
    print(f"\nChecking for NaN values...")
    print(f"NaN predictions: {merged_df['prediction'].isna().sum()}")
    print(f"NaN targets: {merged_df[target_col].isna().sum()}")
    print(f"NaN eras: {merged_df[era_col].isna().sum()}")
    
    # Remove any rows with NaN predictions or targets
    before_dropna = len(merged_df)
    merged_df = merged_df.dropna(subset=['prediction', target_col])
    after_dropna = len(merged_df)
    print(f"Dropped {before_dropna - after_dropna} rows with NaN values")
    
    # Final data validation
    print(f"\nFinal merged data:")
    print(f"Shape: {merged_df.shape}")
    print(f"Prediction range: [{merged_df['prediction'].min():.6f}, {merged_df['prediction'].max():.6f}]")
    print(f"Target range: [{merged_df[target_col].min():.6f}, {merged_df[target_col].max():.6f}]")
    print(f"Unique eras: {merged_df[era_col].nunique()}")
    
    if len(merged_df) == 0:
        raise ValueError("No valid data remaining after merge and NaN removal!")
    
    return merged_df

def calculate_performance_metrics_fixed(df, prediction_col='prediction', target_col='target', era_col='era'):
    """Calculate performance metrics with better error handling"""
    
    metrics = {}
    
    print(f"\n=== CALCULATING METRICS ===")
    print(f"Data shape: {df.shape}")
    print(f"Prediction column: {prediction_col}")
    print(f"Target column: {target_col}")
    
    # Check data validity
    if df[prediction_col].isna().any() or df[target_col].isna().any():
        print("❌ ERROR: NaN values found in prediction or target columns!")
        return None
    
    if len(df) < 10:
        print("❌ ERROR: Insufficient data for correlation calculation!")
        return None
    
    # Overall correlation with error handling
    try:
        pearson_r, pearson_p = pearsonr(df[prediction_col].astype(float), df[target_col].astype(float))
        spearman_r, spearman_p = spearmanr(df[prediction_col].astype(float), df[target_col].astype(float))
        
        print(f"Overall Pearson correlation: {pearson_r:.6f}")
        print(f"Overall Spearman correlation: {spearman_r:.6f}")
        
    except Exception as e:
        print(f"❌ ERROR calculating overall correlation: {e}")
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan
    
    metrics['overall'] = {
        'pearson_correlation': pearson_r,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_r,
        'spearman_p_value': spearman_p,
        'mse': np.mean((df[prediction_col] - df[target_col]) ** 2),
        'mae': np.mean(np.abs(df[prediction_col] - df[target_col])),
        'samples': len(df)
    }
    
    # Per-era correlation
    era_correlations = []
    era_metrics = {}
    
    print(f"\nCalculating per-era correlations...")
    for era in df[era_col].unique():
        era_data = df[df[era_col] == era]
        if len(era_data) > 10:  # Ensure sufficient samples
            try:
                era_pearson, _ = pearsonr(era_data[prediction_col].astype(float), era_data[target_col].astype(float))
                era_spearman, _ = spearmanr(era_data[prediction_col].astype(float), era_data[target_col].astype(float))
                
                if not np.isnan(era_pearson):
                    era_correlations.append(era_pearson)
                    era_metrics[era] = {
                        'pearson': era_pearson,
                        'spearman': era_spearman,
                        'samples': len(era_data)
                    }
            except Exception as e:
                print(f"Warning: Could not calculate correlation for era {era}: {e}")
    
    print(f"Valid era correlations: {len(era_correlations)}")
    
    if len(era_correlations) > 0:
        metrics['per_era'] = {
            'mean_correlation': np.mean(era_correlations),
            'std_correlation': np.std(era_correlations),
            'median_correlation': np.median(era_correlations),
            'sharpe_ratio': np.mean(era_correlations) / np.std(era_correlations) if np.std(era_correlations) > 0 else 0,
            'era_details': era_metrics
        }
    else:
        print("❌ Warning: No valid era correlations calculated!")
        metrics['per_era'] = {
            'mean_correlation': np.nan,
            'std_correlation': np.nan,
            'median_correlation': np.nan,
            'sharpe_ratio': 0,
            'era_details': {}
        }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="FIXED Bootstrap experiment analysis")
    parser.add_argument("--control-predictions", required=True, help="Control model predictions")
    parser.add_argument("--experimental-predictions", required=True, help="Experimental model predictions")
    parser.add_argument("--validation-data", required=True, help="Validation data with targets")
    parser.add_argument("--relationships-file", required=True, help="Discovered relationships JSON")
    parser.add_argument("--output-dir", default="fixed_analysis_results", help="Output directory")
    parser.add_argument("--target-col", default="target", help="Target column name")
    parser.add_argument("--era-col", default="era", help="Era column name")
    
    args = parser.parse_args()
    
    setup_logging()
    logging.info("Starting FIXED bootstrap experiment analysis")
    
    try:
        # Load data with better error handling
        logging.info("Loading control predictions and targets...")
        control_df = load_predictions_and_targets_fixed(
            args.control_predictions, args.validation_data, args.target_col, args.era_col
        )
        
        logging.info("Loading experimental predictions and targets...")
        experimental_df = load_predictions_and_targets_fixed(
            args.experimental_predictions, args.validation_data, args.target_col, args.era_col
        )
        
        logging.info(f"Control model: {len(control_df)} predictions")
        logging.info(f"Experimental model: {len(experimental_df)} predictions")
        
        # Calculate performance metrics
        logging.info("Calculating performance metrics...")
        control_metrics = calculate_performance_metrics_fixed(control_df, 'prediction', args.target_col, args.era_col)
        experimental_metrics = calculate_performance_metrics_fixed(experimental_df, 'prediction', args.target_col, args.era_col)
        
        if control_metrics is None or experimental_metrics is None:
            print("❌ ERROR: Could not calculate metrics!")
            return
        
        # Print results
        print("\n" + "="*60)
        print("FIXED BOOTSTRAP RELATIONSHIP DISCOVERY - EXPERIMENT RESULTS")
        print("="*60)
        
        print(f"\n PERFORMANCE COMPARISON:")
        control_corr = control_metrics['overall']['pearson_correlation']
        exp_corr = experimental_metrics['overall']['pearson_correlation']
        improvement = exp_corr - control_corr if not (np.isnan(control_corr) or np.isnan(exp_corr)) else np.nan
        
        print(f"Control Model Correlation:      {control_corr:.6f}")
        print(f"Experimental Model Correlation: {exp_corr:.6f}")
        print(f"Improvement:                    {improvement:.6f}")
        
        # Verdict
        if not np.isnan(improvement):
            if improvement > 0.001:
                print(f"\n✅ SUCCESS: Bootstrap discovery shows meaningful improvement!")
            elif improvement > 0:
                print(f"\n MARGINAL: Small improvement detected")
            else:
                print(f"\n❌ NO IMPROVEMENT: Bootstrap discovery did not help")
        else:
            print(f"\n❓ INCONCLUSIVE: Could not calculate improvement")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
