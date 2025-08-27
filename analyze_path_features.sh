#!/bin/bash

# Script to analyze why path features performed poorly

set -e  # Exit on any error

echo "Activating virtual environment..."
source /home/mat/Downloads/pathfind_features/.venv/bin/activate

echo "Running path feature analysis..."

/home/mat/Downloads/pathfind_features/.venv/bin/python -c "
import pandas as pd
import numpy as np
import json
from pathlib import Path
import pickle

def analyze_path_features():
    print('=== PATH FEATURE ANALYSIS ===')
    
    # Load data
    print('Loading data...')
    validation_data = pd.read_parquet('v5.0/validation.parquet')
    engineered_data = pd.read_parquet('pipeline_runs/my_experiment/03_features_validation.parquet')
    
    # Load feature lists
    with open('pipeline_runs/my_experiment/new_feature_names.json', 'r') as f:
        path_features = json.load(f)
    
    with open('v5.0/features.json', 'r') as f:
        original_features_data = json.load(f)
        original_features = original_features_data.get('feature_sets', {}).get('medium', [])
        if not original_features:
            original_features = [c for c in validation_data.columns if c.startswith('feature')]
    
    print(f'Original features: {len(original_features)}')
    print(f'Path features: {len(path_features)}')
    print(f'Path features: {path_features}')
    
    # Calculate correlations with target
    print('\n=== CORRELATION WITH TARGET ===')
    
    target_correlations = {}
    
    # Original features correlation with target
    original_corr_sum = 0
    original_corr_count = 0
    for feat in original_features[:100]:  # Check first 100 to avoid too much output
        if feat in validation_data.columns:
            corr = validation_data[feat].corr(validation_data['target'])
            if not np.isnan(corr):
                original_corr_sum += abs(corr)
                original_corr_count += 1
    
    avg_original_corr = original_corr_sum / max(original_corr_count, 1)
    print(f'Average |correlation| of original features with target: {avg_original_corr:.6f}')
    
    # Path features correlation with target
    path_corr_sum = 0
    path_corr_count = 0
    for feat in path_features:
        if feat in engineered_data.columns:
            corr = engineered_data[feat].corr(engineered_data['adaptive_target'])
            target_correlations[feat] = corr
            if not np.isnan(corr):
                path_corr_sum += abs(corr)
                path_corr_count += 1
                print(f'{feat}: correlation with adaptive_target = {corr:.6f}')
    
    avg_path_corr = path_corr_sum / max(path_corr_count, 1)
    print(f'Average |correlation| of path features with adaptive_target: {avg_path_corr:.6f}')
    
    # Load experimental model and check feature importance
    print('\n=== EXPERIMENTAL MODEL FEATURE IMPORTANCE ===')
    try:
        with open('pipeline_runs/my_experiment/experimental_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Get feature importance if available
        if hasattr(model, 'feature_importance'):
            feature_importance = model.feature_importance
            print(f'Model has feature_importance array of length: {len(feature_importance)}')
            
            # Get feature names
            with open('pipeline_runs/my_experiment/experimental_model_features.json', 'r') as f:
                model_features = json.load(f)
            
            # Show top 10 most important features
            if len(feature_importance) == len(model_features):
                feature_importance_pairs = list(zip(model_features, feature_importance))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                print('Top 10 most important features in experimental model:')
                for i, (feat, imp) in enumerate(feature_importance_pairs[:10]):
                    print(f'{i+1:2d}. {feat}: {imp:.6f}')
                    
                # Check how many path features are in top 10
                path_in_top_10 = sum(1 for feat, _ in feature_importance_pairs[:10] if feat in path_features)
                print(f'Path features in top 10: {path_in_top_10}')
        
        elif hasattr(model, 'booster'):
            # LightGBM booster
            feature_importance = model.booster.feature_importance(importance_type='gain')
            print(f'LightGBM feature importance (gain) - length: {len(feature_importance)}')
            
            with open('pipeline_runs/my_experiment/experimental_model_features.json', 'r') as f:
                model_features = json.load(f)
            
            if len(feature_importance) == len(model_features):
                feature_importance_pairs = list(zip(model_features, feature_importance))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                print('Top 10 most important features (gain):')
                for i, (feat, imp) in enumerate(feature_importance_pairs[:10]):
                    print(f'{i+1:2d}. {feat}: {imp:.2f}')
                    
                path_in_top_10 = sum(1 for feat, _ in feature_importance_pairs[:10] if feat in path_features)
                print(f'Path features in top 10: {path_in_top_10}')
    
    except Exception as e:
        print(f'Could not load model feature importance: {e}')
    
    # Analyze correlation between path features and original features
    print('\n=== PATH FEATURE REDUNDANCY ANALYSIS ===')
    
    # Sample a subset of data for efficiency
    sample_size = min(50000, len(validation_data))
    validation_sample = validation_data.sample(sample_size, random_state=42)
    engineered_sample = engineered_data.sample(sample_size, random_state=42)
    
    # Check correlation between path features and their constituent original features
    print('Analyzing path feature constituents...')
    
    relationships = [
        {'feature1': 'feature_itinerant_hexahedral_photoengraver', 'feature2': 'feature_prudent_pileate_oven', 'path': 'path_00_averxoven'},
        {'feature1': 'feature_itinerant_hexahedral_photoengraver', 'feature2': 'feature_aseptic_eely_hemiplegia', 'path': 'path_01_averxegia'},
        {'feature1': 'feature_prudent_pileate_oven', 'feature2': 'feature_subalpine_apothegmatical_ajax', 'path': 'path_02_ovenxajax'},
        {'feature1': 'feature_subalpine_apothegmatical_ajax', 'feature2': 'feature_symmetrical_spongy_tricentenary', 'path': 'path_03_ajaxxnary'},
        {'feature1': 'feature_pistachio_atypical_malison', 'feature2': 'feature_symmetrical_spongy_tricentenary', 'path': 'path_04_isonxnary'},
        {'feature1': 'feature_ungrounded_transpontine_winder', 'feature2': 'feature_aseptic_eely_hemiplegia', 'path': 'path_05_nderxegia'}
    ]
    
    for rel in relationships:
        feat1, feat2, path_feat = rel['feature1'], rel['feature2'], rel['path']
        
        if feat1 in validation_sample.columns and feat2 in validation_sample.columns and path_feat in engineered_sample.columns:
            # Correlation between path feature and target
            path_target_corr = engineered_sample[path_feat].corr(engineered_sample['adaptive_target'])
            
            # Correlation between constituent features and target
            feat1_target_corr = validation_sample[feat1].corr(validation_sample['target'])
            feat2_target_corr = validation_sample[feat2].corr(validation_sample['target'])
            
            # Correlation between constituent features
            feat1_feat2_corr = validation_sample[feat1].corr(validation_sample[feat2])
            
            print(f'{path_feat}:')
            print(f'  Path vs target correlation: {path_target_corr:.6f}')
            print(f'  {feat1} vs target: {feat1_target_corr:.6f}')
            print(f'  {feat2} vs target: {feat2_target_corr:.6f}')
            print(f'  {feat1} vs {feat2}: {feat1_feat2_corr:.6f}')
            print()
    
    # Check for multicollinearity issues
    print('=== MULTICOLLINEARITY CHECK ===')
    if len(path_features) > 1:
        path_corr_matrix = engineered_sample[path_features].corr()
        high_corr_pairs = []
        for i in range(len(path_features)):
            for j in range(i+1, len(path_features)):
                corr = path_corr_matrix.iloc[i, j]
                if abs(corr) > 0.8:
                    high_corr_pairs.append((path_features[i], path_features[j], corr))
        
        if high_corr_pairs:
            print('Highly correlated path feature pairs (|corr| > 0.8):')
            for feat1, feat2, corr in high_corr_pairs:
                print(f'  {feat1} vs {feat2}: {corr:.6f}')
        else:
            print('No highly correlated path feature pairs found.')
    
    print('\n=== SUMMARY ===')
    print(f'Path features showed lower correlation with target ({avg_path_corr:.6f})')
    print(f'vs original features ({avg_original_corr:.6f})')
    print('This suggests the path features may be:')
    print('1. Over-engineered (multiplying features that are already correlated)')
    print('2. Including noise from weak relationships')
    print('3. Not capturing meaningful interactions')
    print('4. Suffering from multicollinearity')

analyze_path_features()
"

echo "Path feature analysis completed!"
echo "Check the output above for detailed analysis."