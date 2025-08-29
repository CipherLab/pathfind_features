import numpy as np
import pandas as pd
from honest_frameworks import (
    HonestValidationFramework,
    FeatureStabilityEngine,
    AdaptiveEnsemble,
    RiskManager,
)


def create_synthetic_data():
    rng = np.random.default_rng(42)
    n = 200
    eras_train = np.arange(201900, 201900 + n)
    eras_test = np.arange(202200, 202200 + n)
    x_train = rng.normal(size=n)
    y_train = x_train * 0.5 + rng.normal(scale=0.1, size=n)
    x_test = rng.normal(size=n)
    y_test = x_test * 0.5 + rng.normal(scale=0.1, size=n)
    train = pd.DataFrame({'era': eras_train, 'f1': x_train, 'target': y_train})
    test = pd.DataFrame({'era': eras_test, 'f1': x_test, 'target': y_test})
    return train, test


def test_time_machine_and_tc():
    train, test = create_synthetic_data()
    hvf = HonestValidationFramework(min_era_gap=1)
    res = hvf.time_machine_test(train, test, ['f1'], 'target')
    assert res['brutal_honesty_score'] > 0
    data = test.copy()
    data['vix'] = np.linspace(10, 30, len(data))
    regimes = hvf.regime_aware_splits(data)
    assert set(regimes.keys()) == {'grind', 'transition', 'crisis'}
    tc = hvf.transaction_cost_reality_check(test['target'])
    assert tc['sharpe_after_cost'] <= tc['sharpe']


def test_feature_engine_and_ratios():
    rng = np.random.default_rng(0)
    n = 100
    base = pd.DataFrame({
        'f1': rng.normal(size=n),
        'f2': rng.normal(size=n),
        'target': rng.normal(size=n),
        'vix': np.linspace(10, 30, n)
    })
    engine = FeatureStabilityEngine(target_col='target')
    regimes = {
        'grind': base[base['vix'] < 15],
        'crisis': base[base['vix'] > 25],
        'transition': base[(base['vix'] >= 15) & (base['vix'] <= 25)],
    }
    engine.test_across_regimes('f1', base, regimes)
    curated = engine.curate_final_list(['f1', 'f2'])
    assert isinstance(curated, list)
    ratios = engine.build_ratio_features(['f1', 'f2'])
    assert ratios and ratios[0] == 'f1_over_f2'


def test_ensemble_and_risk_manager():
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        'x1': rng.normal(size=n),
        'target': rng.normal(size=n),
        'regime': ['crisis'] * 40 + ['grind'] * 40 + ['transition'] * 40,
    })
    df['x2'] = df['x1'] * 0.3 + rng.normal(scale=0.1, size=n)
    ensemble = AdaptiveEnsemble()
    ensemble.train_specialized_models(df, regimes_col='regime', target='target')
    weights = ensemble.dynamic_weighting(30, {'crisis': 0.02, 'grind': 0.01})
    assert abs(sum(weights.values()) - 1.0) < 1e-6

    rm = RiskManager()
    pos = rm.calculate_position_size(0.02, 0.8, 'crisis')
    assert pos < 0.02
    change = rm.regime_change_detection(pd.DataFrame({'prediction': [0, 1, 2], 'target': [2, 1, 0]}))
    assert change is True
