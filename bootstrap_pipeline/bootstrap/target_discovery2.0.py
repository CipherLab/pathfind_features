"""
Walk-forward target discovery module.
"""

import logging
import random
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from typing import Any, cast
import warnings

class WalkForwardTargetDiscovery:
    """
    Target discovery that doesn't peek into the future
    Like a time traveler with ethics
    """
    
    def __init__(self, target_columns, min_history_eras=20):
        self.target_columns = target_columns
        self.n_targets = len(target_columns)
        self.min_history_eras = min_history_eras
        self.era_weights = {}
        
        logging.info(f"Initialized walk-forward target discovery for {self.n_targets} targets")

    @staticmethod
    def _pearson_stat(x, y) -> float:
        """Robustly compute Pearson correlation coefficient and return the statistic as float."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res: Any = pearsonr(x, y)
            # SciPy >= 1.10 returns PearsonRResult with attribute 'statistic'; otherwise a tuple
            if hasattr(res, 'statistic'):
                r_val = float(getattr(res, 'statistic'))
            else:
                r_tuple = cast(tuple[float, float], res)
                r_val = float(r_tuple[0])
            r = r_val
            if not np.isfinite(r):
                return 0.0
            return r
        except Exception:
            return 0.0
    
    def generate_smart_combinations(self, n_combinations=15):
        """Generate target combinations to test - fewer but smarter"""
        combinations = []
        
        # Pure targets (top performers only)
        for i in range(min(5, self.n_targets)):
            weights = np.zeros(self.n_targets)
            weights[i] = 1.0
            combinations.append(weights)
        
        # Equal weight (the baseline everyone uses)
        combinations.append(np.ones(self.n_targets) / self.n_targets)
        
        # Top-heavy (first few targets get most weight)
        for focus in [2, 3]:
            if focus <= self.n_targets:
                weights = np.zeros(self.n_targets)
                weights[:focus] = np.random.dirichlet(np.ones(focus))
                combinations.append(weights)
        
        # Random sparse combinations
        remaining = max(0, n_combinations - len(combinations))
        for _ in range(remaining):
            weights = np.zeros(self.n_targets)
            n_active = random.randint(2, min(4, self.n_targets))
            active_idx = random.sample(range(self.n_targets), n_active)
            active_weights = np.random.dirichlet(np.ones(n_active))
            for i, idx in enumerate(active_idx):
                weights[idx] = active_weights[i]
            combinations.append(weights)
        
        return np.array(combinations)
    
    def evaluate_combination_robustly(self, weights, history_df, feature_cols, top_k_features: int = 80):
        """
        Evaluate a target combination across multiple historical eras
        Uses cross-validation instead of cherry-picking the best era
        """
        try:
            eras = history_df['era'].unique()
            if len(eras) < 3:
                return 0.0, 0.0  # Not enough data for robust evaluation
            
            era_scores = []
            
            for era in eras:
                era_data = history_df[history_df['era'] == era]
                if len(era_data) < 100:
                    continue
                
                # Sample for efficiency
                if len(era_data) > 1000:
                    era_data = era_data.sample(n=1000, random_state=42)
                
                # Create combined target
                combined_target = np.dot(era_data[self.target_columns].values, weights)
                
                # Quick LightGBM evaluation with aggressive feature selection
                X_df = era_data[feature_cols]
                X_df = X_df.replace([np.inf, -np.inf], np.nan)
                X_df = X_df.fillna(0)
                combined_target = np.nan_to_num(combined_target, nan=0.0, posinf=0.0, neginf=0.0)

                # Per-era fast Pearson feature selection
                # Guard against degenerate y
                if np.std(combined_target) < 1e-8:
                    continue
                corrs = []
                for col in X_df.columns:
                    r = self._pearson_stat(X_df[col].values, combined_target)
                    corrs.append((col, abs(r)))
                corrs.sort(key=lambda x: x[1], reverse=True)
                selected_cols = [c for c, _ in corrs[:min(top_k_features, len(corrs))]]
                if not selected_cols:
                    continue
                feature_matrix = X_df[selected_cols].values
                
                if len(feature_matrix) < 100 or np.std(combined_target) < 1e-8:
                    continue
                
                # Train-test split for this era
                X_train, X_test, y_train, y_test = train_test_split(
                    feature_matrix, combined_target, test_size=0.3, random_state=42
                )
                
                # Quick model
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',  # enable early stopping, still fast
                    'num_leaves': 8,
                    'learning_rate': 0.2,
                    'feature_fraction': 0.5,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'verbosity': -1,
                    'num_threads': -1,
                    'seed': 42
                }

                model = lgb.train(
                    params, train_data, num_boost_round=20,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(5), lgb.log_evaluation(0)]
                )
                
                y_pred = np.asarray(model.predict(X_test))
                
                if np.std(y_pred) > 1e-8 and np.std(y_test) > 1e-8:
                    correlation = self._pearson_stat(y_pred, np.asarray(y_test))
                    if np.isfinite(correlation):
                        era_scores.append(abs(correlation))
            
            if len(era_scores) < 2:
                return 0.0, 0.0
            
            # Return mean and std (Sharpe-like measure)
            mean_score = np.mean(era_scores)
            std_score = np.std(era_scores)
            
            return mean_score, std_score
            
        except Exception as e:
            logging.warning(f"Evaluation error: {e}")
            return 0.0, 0.0

    # --------- Fast Ridge pre-screening to cut LGBM calls ---------
    def evaluate_with_ridge(self, weights, history_df, feature_cols, top_k_features: int = 80):
        try:
            eras = history_df['era'].unique()
            if len(eras) < 3:
                return 0.0
            scores = []
            for era in eras:
                era_data = history_df[history_df['era'] == era]
                if len(era_data) < 100:
                    continue
                if len(era_data) > 1000:
                    era_data = era_data.sample(n=1000, random_state=42)
                # Build target with NaNs handled
                y = np.dot(era_data[self.target_columns].fillna(0).values, weights)
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                if np.std(y) < 1e-8:
                    continue
                X_df = era_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                corrs = []
                for col in X_df.columns:
                    r = self._pearson_stat(X_df[col].values, y)
                    corrs.append((col, abs(r)))
                corrs.sort(key=lambda x: x[1], reverse=True)
                selected = [c for c,_ in corrs[:min(top_k_features, len(corrs))]]
                if not selected:
                    continue
                X = X_df[selected].values
                if len(X) < 100:
                    continue
                # simple train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                pred = np.asarray(model.predict(X_test))
                if np.std(pred) < 1e-8 or np.std(y_test) < 1e-8:
                    continue
                corr = self._pearson_stat(pred, np.asarray(y_test))
                if np.isfinite(corr):
                    scores.append(abs(corr))
            if not scores:
                return 0.0
            return float(np.mean(scores))
        except Exception as e:
            logging.warning(f"Ridge pre-screen error: {e}")
            return 0.0

    def fast_combination_screen(self, combinations, history_df, feature_cols, keep_top: int = 3):
        scored = []
        for combo in combinations:
            score = self.evaluate_with_ridge(combo, history_df, feature_cols)
            scored.append((combo, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c,_ in scored[:keep_top]]
    
    def discover_weights_for_era(self, current_era, history_df, feature_cols):
        """
        Find the best target weights using ONLY historical data
        No peeking at the current era's future!
        """
        if len(history_df) < 1000:
            # Not enough history, use equal weights
            return np.ones(self.n_targets) / self.n_targets
        
        combinations = self.generate_smart_combinations()
        # Pre-screen combinations cheaply with Ridge to reduce LGBM calls
        try:
            combinations = self.fast_combination_screen(combinations, history_df, feature_cols)
        except Exception as e:
            logging.warning(f"fast_combination_screen failed, falling back to all combos: {e}")
        combination_results = []
        
        for i, weights in enumerate(combinations):
            mean_score, std_score = self.evaluate_combination_robustly(
                weights, history_df, feature_cols
            )
            
            # Sharpe-like ratio (prefer consistent performance)
            sharpe = mean_score / (std_score + 1e-6)
            
            combination_results.append({
                'weights': weights,
                'mean_score': mean_score,
                'std_score': std_score,
                'sharpe': sharpe
            })
        
        # Sort by Sharpe ratio (consistency over lucky peaks)
        combination_results.sort(key=lambda x: x['sharpe'], reverse=True)
        
        if combination_results:
            best = combination_results[0]
            logging.info(f"Era {current_era}: Best combination - Mean: {best['mean_score']:.4f}, Sharpe: {best['sharpe']:.4f}")
            return best['weights']
        else:
            return np.ones(self.n_targets) / self.n_targets
