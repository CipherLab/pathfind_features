import pathlib
import sys
import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

# Add the pipeline directory to the path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from validation_framework import EraAwareCrossValidator, categorize_eras_by_vix, calculate_realistic_sharpe


class TestEraAwareCrossValidator:
    """Test cases for the EraAwareCrossValidator class."""

    def test_split_basic(self):
        """Test basic era-aware splitting."""
        # Create mock data with eras - use more eras to make 5 splits feasible
        eras = list(range(1, 151))  # 150 eras (same as integration test)
        mock_data = pd.DataFrame({'era': eras})
        
        cv = EraAwareCrossValidator(n_splits=5, gap_eras=10)
        splits = cv.split(mock_data)
        
        assert len(splits) == 5
        
        for train_idx, val_idx in splits:
            train_eras = mock_data.loc[train_idx, 'era'].unique()
            val_eras = mock_data.loc[val_idx, 'era'].unique()
            
            # Check that validation eras come after training eras with gap
            assert max(train_eras) + cv.gap_eras < min(val_eras)
            
            # Check that there's no overlap
            assert len(set(train_eras) & set(val_eras)) == 0

    def test_split_with_insufficient_data(self):
        """Test splitting with insufficient data."""
        eras = list(range(1, 21))  # Only 20 eras
        mock_data = pd.DataFrame({'era': eras})
        
        cv = EraAwareCrossValidator(n_splits=5, gap_eras=10)
        
        # Should raise ValueError when there's insufficient data
        with pytest.raises(ValueError, match="Not enough eras"):
            cv.split(mock_data)


def test_categorize_eras_by_vix():
    """Test VIX-based era categorization."""
    # Create mock VIX data
    vix_data = pd.DataFrame({
        'era': [1, 2, 3, 4, 5],
        'vix': [15, 28, 22, 12, 35]  # Mix of low, high, and transition VIX levels
    })
    
    era_series = pd.Series([1, 2, 3, 4, 5])
    regimes = categorize_eras_by_vix(era_series, vix_data)
    
    assert len(regimes) == 5
    assert regimes.iloc[0] == 'transition'  # VIX = 15 (between 15-25)
    assert regimes.iloc[1] == 'high_vol_crisis'  # VIX = 28
    assert regimes.iloc[2] == 'transition'  # VIX = 22
    assert regimes.iloc[3] == 'low_vol_grind'  # VIX = 12
    assert regimes.iloc[4] == 'high_vol_crisis'  # VIX = 35


def test_calculate_realistic_sharpe():
    """Test Sharpe ratio calculation with transaction costs."""
    # Test with valid correlations
    era_correlations = [0.05, 0.03, 0.07, -0.02, 0.04]
    result = calculate_realistic_sharpe(era_correlations, transaction_cost_bps=25)
    
    assert 'sharpe_ratio' in result
    assert 'sharpe_with_tc' in result
    assert 'tc_impact' in result
    assert 'mean_correlation' in result
    assert 'std_correlation' in result
    
    # Sharpe with TC should be lower than raw Sharpe
    assert result['sharpe_with_tc'] <= result['sharpe_ratio']
    
    # Test with empty correlations
    empty_result = calculate_realistic_sharpe([])
    assert empty_result['sharpe_ratio'] == 0
    assert empty_result['sharpe_with_tc'] == 0


def test_calculate_realistic_sharpe_edge_cases():
    """Test Sharpe calculation edge cases."""
    # Test with constant correlations (zero variance)
    constant_correlations = [0.05] * 5
    result = calculate_realistic_sharpe(constant_correlations)
    
    # Should handle zero variance case
    assert isinstance(result['sharpe_ratio'], (int, float))
    assert isinstance(result['sharpe_with_tc'], (int, float))


class TestValidationFrameworkIntegration:
    """Integration tests for validation framework components."""

    def test_regime_analysis_workflow(self):
        """Test the complete regime analysis workflow."""
        # Create mock data
        np.random.seed(42)
        n_samples = 500
        eras = np.repeat(range(1, 11), n_samples // 10)
        
        mock_data = pd.DataFrame({
            'era': eras,
            'adaptive_target': np.random.randn(n_samples),
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples)
        })
        
        # Create mock VIX data
        vix_data = pd.DataFrame({
            'era': range(1, 11),
            'vix': np.random.uniform(10, 40, 10)
        })
        
        # Test regime categorization
        regimes = categorize_eras_by_vix(mock_data['era'], vix_data)
        assert len(regimes) == len(mock_data)
        
        # Test that we get all expected regime types
        unique_regimes = regimes.unique()
        expected_regimes = {'high_vol_crisis', 'low_vol_grind', 'transition', 'unknown'}
        assert len(set(unique_regimes) & expected_regimes) > 0

    def test_cross_validator_with_regimes(self):
        """Test cross-validator working with regime data."""
        # Create data with sufficient eras for splitting
        eras = list(range(1, 151))  # 150 eras
        mock_data = pd.DataFrame({'era': eras})
        
        cv = EraAwareCrossValidator(n_splits=3, gap_eras=10)
        splits = cv.split(mock_data)
        
        assert len(splits) == 3
        
        # Verify temporal ordering and gaps
        for i, (train_idx, val_idx) in enumerate(splits):
            train_eras = sorted(mock_data.loc[train_idx, 'era'].unique())
            val_eras = sorted(mock_data.loc[val_idx, 'era'].unique())
            
            assert max(train_eras) < min(val_eras)
            assert min(val_eras) - max(train_eras) >= cv.gap_eras
