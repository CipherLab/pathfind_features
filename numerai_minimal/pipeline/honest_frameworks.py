import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from itertools import combinations


def _fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a simple linear regression using least squares."""
    X_ = np.c_[np.ones(len(X)), X]
    coef, _, _, _ = np.linalg.lstsq(X_, y, rcond=None)
    return coef


def _predict_linear(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_ = np.c_[np.ones(len(X)), X]
    return X_ @ coef


class HonestValidationFramework:
    """Validation helpers that attempt to mirror live performance."""

    def __init__(self, min_era_gap: int = 100, transaction_cost_bps: int = 25):
        self.min_era_gap = min_era_gap
        self.transaction_cost_bps = transaction_cost_bps

    def _check_era_gap(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> bool:
        if 'era' not in train_data.columns or 'era' not in test_data.columns:
            return True
        try:
            max_train = train_data['era'].astype(int).max()
            min_test = test_data['era'].astype(int).min()
            gap = min_test - max_train
            return gap >= self.min_era_gap
        except Exception:
            return True

    def time_machine_test(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                           features: list[str], target: str) -> dict:
        """Train on old data and test on future data with an enforced gap."""
        if not self._check_era_gap(train_data, test_data):
            raise ValueError("Train and test eras are too close. Time travel is disallowed.")

        coef = _fit_linear(train_data[features].values, train_data[target].values)
        train_pred = _predict_linear(coef, train_data[features].values)
        test_pred = _predict_linear(coef, test_data[features].values)

        train_corr = spearmanr(train_pred, train_data[target]).correlation
        test_corr = spearmanr(test_pred, test_data[target]).correlation
        if np.isnan(train_corr) or np.isnan(test_corr):
            honesty = 0.0
            drop = np.inf
        else:
            drop = 1 - (test_corr / train_corr) if train_corr else np.inf
            honesty = max(0.0, min(1.0, test_corr / train_corr)) if train_corr else 0.0
            if drop > 0.5:
                honesty = 0.0
        return {
            "train_correlation": float(train_corr),
            "test_correlation": float(test_corr),
            "drop": float(drop),
            "brutal_honesty_score": float(honesty),
        }

    def regime_aware_splits(self, data: pd.DataFrame, vix_thresholds: tuple[int, int] = (15, 25)) -> dict[str, pd.DataFrame]:
        """Split data into VIX regimes: crisis (>high), grind (<low) and transition.
        
        Args:
            data (pd.DataFrame): DataFrame containing a 'vix' column.
            vix_thresholds (tuple[int, int]): (low, high) VIX thresholds for regime splitting.
        """
        low, high = vix_thresholds
        if 'vix' not in data.columns:
            raise ValueError("Data must contain 'vix' column for regime splitting")
        return {
            'grind': data[data['vix'] < low],
            'transition': data[(data['vix'] >= low) & (data['vix'] <= high)],
            'crisis': data[data['vix'] > high],
        }

    def transaction_cost_reality_check(self, predictions, tc_bps: int | float = 25) -> dict:
        """Reduce Sharpe ratio estimate by transaction cost impact."""
        preds = np.asarray(predictions)
        mean = preds.mean()
        std = preds.std(ddof=1)
        sharpe = mean / std if std > 0 else 0.0
        tc_impact = (tc_bps / 10000) / 0.15  # assume 15% vol
        sharpe_after = sharpe - tc_impact
        return {
            "sharpe": float(sharpe),
            "sharpe_after_cost": float(sharpe_after),
            "tc_impact": float(tc_impact),
        }


class FeatureStabilityEngine:
    """Simple feature stability diagnostics across volatility regimes."""

    def __init__(self, target_col: str = 'target'):
        self.target_col = target_col
        self.feature_results: dict[str, dict] = {}

    def test_across_regimes(self, feature: str, data: pd.DataFrame, regimes: dict[str, pd.DataFrame]) -> dict:
        correlations = {}
        for name, df in regimes.items():
            if feature not in df.columns or self.target_col not in df.columns:
                continue
            corr, _ = spearmanr(df[feature], df[self.target_col])
            if not np.isnan(corr):
                correlations[name] = float(corr)
        if not correlations:
            result = {"stable": False, "correlations": correlations}
            self.feature_results[feature] = result
            return result
        signs = {np.sign(v) for v in correlations.values()}
        sign_flip = len(signs) > 1
        max_corr = max(abs(v) for v in correlations.values())
        min_corr = min(abs(v) for v in correlations.values())
        drop = 1 - (min_corr / max_corr) if max_corr else 1.0
        stable = (not sign_flip) and drop <= 0.5
        result = {
            "stable": stable,
            "correlations": correlations,
            "sign_flip": sign_flip,
            "drop": drop,
        }
        self.feature_results[feature] = result
        return result

    def build_ratio_features(self, stable_features: list[str], max_count: int = 10, epsilon: float = 1e-8) -> list[str]:
        ratios = []
        for a, b in combinations(stable_features, 2):
            ratios.append(f"{a}_over_{b}")
            if len(ratios) >= max_count:
                break
        return ratios

    def curate_final_list(self, original_features: list[str], max_final: int = 50) -> list[str]:
        stable = [f for f in original_features if self.feature_results.get(f, {}).get('stable')]
        return stable[:max_final]


class AdaptiveEnsemble:
    """Train different models for different regimes and weight dynamically."""

    def __init__(self, n_models: int = 5, regime_detector=None):
        self.n_models = n_models
        self.regime_detector = regime_detector
        self.crisis_model = None
        self.grind_model = None
        self.transition_model = None
        self.features: list[str] = []

    def train_specialized_models(self, data: pd.DataFrame, regimes_col: str = 'regime', target: str = 'target'):
        self.features = [c for c in data.columns if c not in {regimes_col, target}]
        for regime, attr in [('crisis', 'crisis_model'), ('grind', 'grind_model'), ('transition', 'transition_model')]:
            subset = data[data[regimes_col] == regime]
            if len(subset) > 0:
                coef = _fit_linear(subset[self.features].values, subset[target].values)
            else:
                coef = None
            setattr(self, attr, coef)

    def dynamic_weighting(self, current_vix: float, recent_performance: dict[str, float]) -> dict[str, float]:
        if current_vix > 25:
            base = {'crisis': 0.7, 'transition': 0.2, 'grind': 0.1}
        elif current_vix < 15:
            base = {'grind': 0.7, 'transition': 0.2, 'crisis': 0.1}
        else:
            base = {'transition': 0.7, 'grind': 0.15, 'crisis': 0.15}
        weighted = {}
        for regime, weight in base.items():
            perf = recent_performance.get(regime, 0)
            weighted[regime] = max(0.0, weight * (1 + perf))
        total = sum(weighted.values())
        if total == 0:
            return {regime: 0.0 for regime in base}
        return {k: v / total for k, v in weighted.items()}


class RiskManager:
    """Basic risk management utilities."""

    def __init__(self, max_drawdown: float = 0.05):
        self.max_drawdown = max_drawdown

    def calculate_position_size(self, prediction: float, confidence: float, current_regime: str) -> float:
        regime_factor = {'crisis': 0.5, 'transition': 0.75, 'grind': 1.0}.get(current_regime, 0.5)
        size = prediction * confidence * regime_factor
        return float(max(-1.0, min(1.0, size)))

    def regime_change_detection(self, recent_data: pd.DataFrame) -> bool:
        if {'prediction', 'target'} <= set(recent_data.columns):
            corr, _ = spearmanr(recent_data['prediction'], recent_data['target'])
            return bool(corr is not None and corr < 0)
        return False
