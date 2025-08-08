import logging
import numpy as np
import copy


def run_null_hypothesis_test(discovery, data_sample, target_col, feature_cols):
    """Run null hypothesis test with shuffled targets to detect overfitting.

    To avoid contaminating the live discovery state, operate on a deep copy
    of the discovery object.
    """
    logging.info("Running null hypothesis test (shuffled targets)...")

    # work on a deep copy so we don't mutate the live discovery state
    test_disc = copy.deepcopy(discovery)

    shuffled = data_sample.copy()
    rng = np.random.default_rng(42)
    shuffled[target_col] = rng.permutation(shuffled[target_col].values)

    test_rows = shuffled.sample(n=min(NULL_HYPOTHESIS_SAMPLE_SIZE, len(shuffled)), random_state=42)
    for _, row in test_rows.iterrows():
        fv = row[feature_cols].values.astype(float)
        tv = float(row[target_col])
        paths = test_disc.find_creative_paths(fv, tv)
        test_disc.update_relationships_from_paths(paths, fv, tv)

    shuffled_relationships = test_disc.get_discovered_relationships(min_strength=0.15)
    if len(shuffled_relationships) > NULL_HYPOTHESIS_RELATIONSHIP_THRESHOLD:
        logging.warning("🚨 OVERFITTING ALERT: meaningful relationships found on shuffled targets")
        return False
    logging.info("✅ Sanity check passed: no significant relationships on shuffled targets")
    return True
