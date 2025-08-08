# bootstrap_pipeline/bootstrap/target_discovery.py

import logging
import random
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

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
        self.last_weights = None

        logging.info(f"Initialized walk-forward target discovery for {self.n_targets} targets")
    
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
    
    def evaluate_combination_robustly(self, weights, history_df, feature_cols):
        """
        Evaluate a target combination across multiple historical eras
        Uses cross-validation instead of cherry-picking the best era
        """
        try:
            eras = history_df['era'].unique()
            if len(eras) < 3:
                return 0.0, 0.0, 0.0, 0.0  # Not enough data for robust evaluation

            era_scores_raw = []
            
            for era in eras:
                era_data = history_df[history_df['era'] == era]
                if len(era_data) < 100:
                    continue
                
                # Sample for efficiency
                if len(era_data) > 2000:
                    era_data = era_data.sample(n=2000, random_state=42)
                
                # Create combined target
                combined_target = np.dot(era_data[self.target_columns].values, weights)

                # Quick LightGBM evaluation
                feature_matrix = era_data[feature_cols].values

                if len(feature_matrix) < 100 or np.std(combined_target) < 1e-8:
                    continue

                # Train-test split for this era
                X_train, X_test, y_train, y_test = train_test_split(
                    feature_matrix, combined_target, test_size=0.3, random_state=42
                )

                # Median imputation based on training slice
                feature_medians = np.nanmedian(X_train, axis=0)
                feature_medians = np.where(np.isnan(feature_medians), 0, feature_medians)
                X_train = np.where(np.isnan(X_train), feature_medians, X_train)
                X_test = np.where(np.isnan(X_test), feature_medians, X_test)

                target_median = np.nanmedian(y_train)
                target_median = 0.0 if np.isnan(target_median) else target_median
                y_train = np.where(np.isnan(y_train), target_median, y_train)
                y_test = np.where(np.isnan(y_test), target_median, y_test)
                
                # Quick model
                train_data = lgb.Dataset(X_train, label=y_train)
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 15,  # Smaller to prevent overfitting
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'verbosity': -1,
                    'seed': 42
                }
                
                model = lgb.train(
                    params, train_data, num_boost_round=30,
                    callbacks=[lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_test)
                
                if np.std(y_pred) > 1e-8 and np.std(y_test) > 1e-8:
                    correlation = np.corrcoef(y_pred, y_test)[0, 1]
                    if not np.isnan(correlation):
                        era_scores_raw.append(correlation)

            if len(era_scores_raw) < 2:
                return 0.0, 0.0, 0.0, 0.0

            signs = [np.sign(c) for c in era_scores_raw if not np.isnan(c)]
            sign_consistency = np.mean([s > 0 for s in signs]) if signs else 0.0
            era_scores = [abs(c) for c in era_scores_raw if not np.isnan(c)]

            mean_score = np.mean(era_scores)
            std_score = np.std(era_scores)
            # Scale Sharpe by sign consistency, offset by SIGN_CONSISTENCY_OFFSET for interpretability
            sharpe = (mean_score / (std_score + 1e-6)) * (2*sign_consistency - self.SIGN_CONSISTENCY_OFFSET)

            return mean_score, std_score, sign_consistency, sharpe

        except Exception as e:
            logging.warning(f"Evaluation error: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def discover_weights_for_era(self, current_era, history_df, feature_cols):
        """
        Find the best target weights using ONLY historical data
        No peeking at the current era's future!
        """
        if len(history_df) < 1000:
            # Not enough history, use equal weights
            return np.ones(self.n_targets) / self.n_targets
        
        combinations = self.generate_smart_combinations()
        combination_results = []
        
        for i, weights in enumerate(combinations):
            mean_score, std_score, sign_consistency, sharpe = self.evaluate_combination_robustly(
                weights, history_df, feature_cols
            )

            combination_results.append({
                'weights': weights,
                'mean_score': mean_score,
                'std_score': std_score,
                'sign_consistency': sign_consistency,
                'sharpe': sharpe
            })
        
        # Sort by Sharpe ratio (consistency over lucky peaks)
        combination_results.sort(key=lambda x: x['sharpe'], reverse=True)
        
        if combination_results:
            best = combination_results[0]
            drift = 1.0
            if self.last_weights is not None:
                norm_best = np.linalg.norm(best['weights'])
                norm_last = np.linalg.norm(self.last_weights)
                denom = norm_best * norm_last
                if denom != 0:
                    drift = np.dot(best['weights'], self.last_weights) / denom
                elif np.allclose(best['weights'], 0) and np.allclose(self.last_weights, 0):
                    drift = 1.0  # Both zero vectors: perfect similarity
                else:
                    drift = 0.0  # One zero, one nonzero: no similarity
            logging.info(
                f"Era {current_era}: Mean={best['mean_score']:.4f}, "
                f"Sharpe={best['sharpe']:.3f}, Sign+={best['sign_consistency']:.2%}, "
                f"EffTargets={1/np.sum(best['weights']**2):.2f}, Drift={drift:.3f}"
            )
            self.last_weights = best['weights']
            return best['weights']
        else:
            return np.ones(self.n_targets) / self.n_targets
