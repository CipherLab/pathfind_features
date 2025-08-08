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
        
        logging.info(f"Initialized walk-forward target discovery for {self.n_targets} targets")
    
    def generate_smart_combinations(self, n_combinations=15):
        """Generate target combinations to test - fewer but smarter"""
        combinations = []
        
        # All pure single-target combinations
        for i in range(self.n_targets):
            weights = np.zeros(self.n_targets)
            weights[i] = 1.0
            combinations.append(weights)
        
        # Equal weight (the baseline everyone uses)
        combinations.append(np.ones(self.n_targets) / self.n_targets)
        
        # Top-heavy (first few indices for now; later: rank by history)
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
                return 0.0, 0.0  # Not enough data for robust evaluation
            
            era_scores = []
            
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
                feature_matrix = np.nan_to_num(feature_matrix, 0)
                combined_target = np.nan_to_num(combined_target, 0)
                
                if len(feature_matrix) < 100 or np.std(combined_target) < 1e-8:
                    continue
                
                # Train-test split for this era
                X_train, X_test, y_train, y_test = train_test_split(
                    feature_matrix, combined_target, test_size=0.3, random_state=42
                )
                
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
