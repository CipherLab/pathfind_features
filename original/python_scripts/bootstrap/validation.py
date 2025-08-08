"""
Validation module for the bootstrap discovery pipeline.
"""

import logging
import numpy as np

def run_null_hypothesis_test(discovery, data_sample, target_col, feature_cols):
    """
    Gemini's brilliant suggestion: shuffle targets and see if we still find "relationships"
    If we do, we know we're just finding patterns in noise
    """
    logging.info("Running null hypothesis test (shuffled targets)...")
    
    # Shuffle the target column
    shuffled_sample = data_sample.copy()
    shuffled_sample[target_col] = np.random.permutation(shuffled_sample[target_col].values)
    
    # Run discovery on shuffled data
    shuffled_relationships = []
    
    # Sample a few rows for testing
    test_rows = shuffled_sample.sample(n=min(500, len(shuffled_sample)), random_state=42)
    
    for _, row in test_rows.iterrows():
        feature_values = row[feature_cols].values.astype(float)
        target_value = float(row[target_col])
        
        paths = discovery.find_creative_paths(feature_values, target_value)
        discovery.update_relationships_from_paths(paths, feature_values, target_value)
    
    shuffled_relationships = discovery.get_discovered_relationships(min_strength=0.15)
    
    if len(shuffled_relationships) > 5:
        logging.warning(f"🚨 OVERFITTING ALERT: Found {len(shuffled_relationships)} 'relationships' in SHUFFLED data!")
        logging.warning("This suggests the discovery process is finding patterns in pure noise.")
        return False
    else:
        logging.info(f"✅ Sanity check passed: Only {len(shuffled_relationships)} weak relationships in shuffled data")
        return True
