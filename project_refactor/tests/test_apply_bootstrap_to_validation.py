import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from apply_bootstrap_to_validation import compute_adaptive_target
def test_compute_adaptive_target_defaults_to_equal_weights_for_missing_era():
    df = pd.DataFrame({
        'era': [1, 2, 3],
        'target_a': [0.1, 0.2, 0.3],
        'target_b': [0.9, 0.8, 0.7],
    })
    weights_map = {1: np.array([0.7, 0.3]), 2: np.array([0.6, 0.4])}

    compute_adaptive_target(df, weights_map)

    missing = df.loc[df['era'] == 3]
    expected = missing[['target_a', 'target_b']].mean(axis=1).iloc[0]

    assert missing['adaptive_target'].iloc[0] == pytest.approx(expected)
