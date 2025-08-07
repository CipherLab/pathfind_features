# bootstrap_pipeline/bootstrap/feature_discovery.py

import logging
import random
import numpy as np
from collections import defaultdict

class CreativePathfindingDiscovery:
    """
    The original pathfinding algorithm that Gemini wanted to lobotomize
    We keep the creativity but add walk-forward discipline
    """
    
    def __init__(self, feature_columns, learning_rate=0.05, decay_rate=0.95, max_features=100):
        self.feature_columns = feature_columns[:max_features]
        self.n_features = len(self.feature_columns)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # The relationship matrix (the heart of the algorithm)
        self.relationship_matrix = np.random.uniform(0.01, 0.1, (self.n_features, self.n_features))
        np.fill_diagonal(self.relationship_matrix, 1.0)
        
        # Path memory
        self.successful_paths = defaultdict(float)
        self.feature_importance = np.zeros(self.n_features)
        
        logging.info(f"Initialized creative pathfinding for {self.n_features} features")
    
    def find_creative_paths(self, feature_values, target_value, max_path_length=4, n_paths=8):
        """
        The creative pathfinding algorithm - this is what makes it special
        Like A* search but for feature relationships
        """
        paths = []
        
        start_candidates = np.arange(self.n_features)
        
        for _ in range(n_paths):
            start_idx = random.choice(start_candidates)
            path = self._explore_path(feature_values, target_value, start_idx, max_path_length)
            if len(path) > 1:
                paths.append(path)
        
        return paths
    
    def _explore_path(self, feature_values, target_value, start_idx, max_length):
        """
        Explore a path through feature space using relationship strengths
        """
        path = [start_idx]
        
        for step in range(max_length - 1):
            current_idx = path[-1]
            
            # Get relationship strengths to other features
            strengths = self.relationship_matrix[current_idx].copy()
            
            # Don't revisit features in current path
            for visited in path:
                strengths[visited] = 0
            
            # Add exploration noise (prevents getting stuck)
            strengths += np.random.uniform(0, 0.01, len(strengths))
            
            # Choose next feature
            next_idx = np.argmax(strengths)
            
            if strengths[next_idx] < 0.1:  # No good paths left
                break
            
            path.append(next_idx)
            
            # Early stopping if path prediction is good
            path_prediction = self._evaluate_path_prediction(path, feature_values)
            if abs(path_prediction - target_value) < 0.1:
                break
        
        return path
    
    def _evaluate_path_prediction(self, path, feature_values):
        """
        Evaluate what this path predicts using relationship weights
        """
        if not path:
            return 0.0
        
        prediction = 0.0
        total_weight = 0.0
        
        for i, feature_idx in enumerate(path):
            if i == 0:
                weight = 1.0  # Starting feature gets full weight
            else:
                prev_idx = path[i-1]
                weight = self.relationship_matrix[prev_idx, feature_idx]
            
            prediction += feature_values[feature_idx] * weight
            total_weight += weight
        
        return prediction / max(total_weight, 1e-6)
    
    def update_relationships_from_paths(self, paths, feature_values, target_value):
        """
        Update the relationship matrix based on path performance
        This is where learning happens
        """
        for path in paths:
            if len(path) < 2:
                continue
            
            # How well did this path predict?
            path_prediction = self._evaluate_path_prediction(path, feature_values)
            prediction_error = abs(path_prediction - target_value)
            success_score = max(0, 1.0 - prediction_error * 2)  # Scale error
            
            # Update relationships in this path
            for i in range(len(path) - 1):
                from_idx = path[i]
                to_idx = path[i + 1]
                
                # Strengthen successful relationships
                reinforcement = self.learning_rate * success_score * 0.02
                self.relationship_matrix[from_idx, to_idx] += reinforcement
                self.relationship_matrix[to_idx, from_idx] += reinforcement  # Symmetric
                
                # Cap at reasonable maximum
                self.relationship_matrix[from_idx, to_idx] = min(3.0, self.relationship_matrix[from_idx, to_idx])
                self.relationship_matrix[to_idx, from_idx] = min(3.0, self.relationship_matrix[to_idx, from_idx])
            
            # Update feature importance
            for feature_idx in path:
                self.feature_importance[feature_idx] += success_score * self.learning_rate
            
            # Remember successful patterns
            path_key = tuple(sorted(path))
            self.successful_paths[path_key] += success_score
    
    def decay_unused_relationships(self):
        """
        Decay relationships that aren't being used (forget bad patterns)
        """
        self.relationship_matrix *= self.decay_rate
        np.fill_diagonal(self.relationship_matrix, 1.0)  # Keep self-relationships strong
        
        # Decay path memories
        for path_key in list(self.successful_paths.keys()):
            self.successful_paths[path_key] *= self.decay_rate
            if self.successful_paths[path_key] < 0.01:
                del self.successful_paths[path_key]
        
        self.feature_importance *= self.decay_rate
    
    def get_discovered_relationships(self, min_strength=0.2, top_k=20):
        """
        Extract the strongest discovered relationships
        """
        relationships = []
        
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                strength = self.relationship_matrix[i, j]
                if strength >= min_strength:
                    relationships.append({
                        'feature1': self.feature_columns[i],
                        'feature2': self.feature_columns[j],
                        'strength': float(strength),
                        'feature1_idx': i,
                        'feature2_idx': j
                    })
        
        relationships.sort(key=lambda x: x['strength'], reverse=True)
        return relationships[:top_k]
