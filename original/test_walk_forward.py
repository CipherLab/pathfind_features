#!/usr/bin/env python3
"""
Test script to validate the walk-forward architecture implementation
"""

import numpy as np
import pandas as pd
from original.python_scripts.bootstrap.target_discovery import WalkForwardTargetDiscovery

def test_walk_forward_architecture():
    """
    Test the walk-forward architecture with synthetic data
    """
    # Create synthetic data with clear era patterns
    n_samples = 1000
    n_features = 10
    n_targets = 3
    
    # Create features
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    
    # Create targets with different relationships per era
    targets = np.zeros((n_samples, n_targets))
    
    # Era 100: target_0 is most predictive
    targets[:300, 0] = features[:300, 0] + np.random.randn(300) * 0.1
    targets[:300, 1] = np.random.randn(300) * 0.1
    targets[:300, 2] = np.random.randn(300) * 0.1
    
    # Era 101: target_1 is most predictive  
    targets[300:600, 1] = features[300:600, 1] + np.random.randn(300) * 0.1
    targets[300:600, 0] = np.random.randn(300) * 0.1
    targets[300:600, 2] = np.random.randn(300) * 0.1
    
    # Era 102: target_2 is most predictive
    targets[600:, 2] = features[600:, 2] + np.random.randn(400) * 0.1
    targets[600:, 0] = np.random.randn(400) * 0.1
    targets[600:, 1] = np.random.randn(400) * 0.1
    
    # Create DataFrame
    data = pd.DataFrame()
    
    # Add features
    feature_cols = []
    for i in range(n_features):
        col_name = f'feature_{i:04d}'
        data[col_name] = features[:, i]
        feature_cols.append(col_name)
    
    # Add targets
    target_cols = []
    for i in range(n_targets):
        col_name = f'target_{i}'
        data[col_name] = targets[:, i]
        target_cols.append(col_name)
    
    # Add eras
    data['era'] = ['100'] * 300 + ['101'] * 300 + ['102'] * 400
    
    # Test the discovery
    discovery = WalkForwardTargetDiscovery(target_cols)
    
    # Test walk-forward discovery for era 101 (should use data from era 100)
    era_100_data = data[data['era'] == '100']
    
    # Discover robust weights from era 100 history
    robust_weights = discovery.discover_weights_for_era('101', era_100_data, feature_cols)
    
    # For era 100, since there's no history, it should use default weights
    era_100_weights = np.ones(n_targets) / n_targets
    
    # Test that era 100 weights are close to equal (since no history)
    assert np.allclose(robust_weights, era_100_weights, atol=0.1)