# bootstrap_pipeline/utils/utils.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

def reduce_mem_usage(df, _verbose=True):
    """Memory optimization - because RAM is finite and dreams are not"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if _verbose: 
        logging.info(f'Memory usage: {start_mem:.1f}MB -> {end_mem:.1f}MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df


class RegimeDetector:
    """Detect market regimes from a local VIX time series."""

    def __init__(self,
                 vix_window: int = 60,
                 corr_window: int = 30,
                 vix_high_threshold: float = 75,
                 vix_low_threshold: float = 25,
                 vix_high_level: float = 25.0,
                 vix_low_level: float = 15.0):
        """Initialize detector with percentile and absolute thresholds.

        Args:
            vix_window: Rolling window for VIX percentile calculation.
            corr_window: Rolling window for correlation analysis (unused).
            vix_high_threshold: Percentile threshold for high volatility regime.
            vix_low_threshold: Percentile threshold for low volatility regime.
            vix_high_level: Absolute VIX level considered "crisis" when
                ``use_percentiles`` is False.
            vix_low_level: Absolute VIX level considered "grind" when
                ``use_percentiles`` is False.
        """

        self.vix_window = vix_window
        self.corr_window = corr_window
        self.vix_high_threshold = vix_high_threshold
        self.vix_low_threshold = vix_low_threshold
        self.vix_high_level = vix_high_level
        self.vix_low_level = vix_low_level

    # ------------------------------------------------------------------
    # Loading and labelling helpers
    # ------------------------------------------------------------------
    def load_vix_data(self, eras: list, vix_file: str | None = None) -> pd.DataFrame:
        """Load or simulate VIX data for the provided eras.

        The function first attempts to read ``vix_file`` which is expected to
        contain ``era`` and ``vix`` columns.  If the file cannot be read, a
        pseudo VIX series is generated locally to keep the pipeline
        self‑contained.
        """

        if vix_file and Path(vix_file).exists():
            try:
                df = pd.read_csv(vix_file)
                df['era'] = df['era'].astype(str)
                return df[['era', 'vix']]
            except Exception as exc:  # pragma: no cover - fallback path
                logging.warning(f"Failed to load VIX data from {vix_file}: {exc}")

        # Fallback: simulate VIX values with a bounded random walk
        np.random.seed(42)
        current = 20.0
        values = []
        for _ in eras:
            change = np.random.normal(0, 2)
            current = float(np.clip(current + change, 10, 50))
            values.append(current)
        return pd.DataFrame({'era': [str(e) for e in eras], 'vix': values})

    def _classify_by_level(self, vix_data: pd.DataFrame) -> pd.DataFrame:
        df = vix_data.copy()
        df['era'] = df['era'].astype(str)
        cond_high = df['vix'] > self.vix_high_level
        cond_low = df['vix'] < self.vix_low_level
        df['vix_regime'] = np.select(
            [cond_high, cond_low],
            ['high_vol_crisis', 'low_vol_grind'],
            default='transition'
        )
        return df

    def classify_eras(self, era_series: pd.Series, vix_data: pd.DataFrame,
                      use_percentiles: bool = False) -> pd.Series:
        """Assign regimes to each era in ``era_series``.

        Args:
            era_series: Series of eras to label.
            vix_data: DataFrame with ``era`` and ``vix`` columns.
            use_percentiles: If ``True`` use rolling percentile thresholds,
                otherwise rely on absolute ``vix_high_level``/``vix_low_level``.
        """

        if use_percentiles:
            labelled = self.detect_regimes(vix_data)
        else:
            labelled = self._classify_by_level(vix_data)

        mapping = labelled.set_index('era')['vix_regime']
        return era_series.astype(str).map(mapping).fillna('unknown')

    def detect_regimes(self, vix_data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None,
                      era_column: str = 'era') -> pd.DataFrame:
        """
        Detect market regimes based on VIX levels and market correlations.

        Args:
            vix_data: DataFrame with 'era' and 'vix' columns
            market_data: Optional DataFrame with market features for correlation analysis
            era_column: Name of the era column

        Returns:
            DataFrame with regime classifications
        """
        # Sort by era to ensure proper ordering
        vix_data = vix_data.sort_values(era_column).copy()

        # Calculate rolling VIX percentiles
        vix_data['vix_rolling_percentile'] = (
            vix_data['vix'].rolling(window=self.vix_window, min_periods=1)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
        )

        # Classify regimes based on VIX percentiles
        conditions = [
            (vix_data['vix_rolling_percentile'] >= self.vix_high_threshold),
            (vix_data['vix_rolling_percentile'] <= self.vix_low_threshold)
        ]
        choices = ['high_vol_crisis', 'low_vol_grind']
        vix_data['vix_regime'] = np.select(conditions, choices, default='transition')

        # If market data is provided, enhance regime detection with correlations
        if market_data is not None:
            vix_data = self._add_market_correlation_regimes(vix_data, market_data, era_column)

        return vix_data

    def _add_market_correlation_regimes(self, vix_data: pd.DataFrame,
                                       market_data: pd.DataFrame,
                                       era_column: str) -> pd.DataFrame:
        """Enhance regime detection with market correlation analysis."""
        # This is a simplified version - in practice you'd want more sophisticated
        # correlation analysis with multiple market indicators

        # For now, just use VIX-based classification
        # Could be extended to include:
        # - Rolling correlations with major indices
        # - Bond yields, credit spreads, etc.
        # - Cross-sectional market correlations

        return vix_data

    def get_regime_stats(self, regime_data: pd.DataFrame) -> Dict:
        """Calculate statistics for each regime."""
        stats = {}

        for regime in regime_data['vix_regime'].unique():
            regime_mask = regime_data['vix_regime'] == regime
            regime_vix = regime_data.loc[regime_mask, 'vix']

            stats[regime] = {
                'count': int(regime_mask.sum()),
                'percentage': float(regime_mask.mean() * 100),
                'vix_mean': float(regime_vix.mean()),
                'vix_std': float(regime_vix.std()),
                'vix_min': float(regime_vix.min()),
                'vix_max': float(regime_vix.max())
            }

        return stats


def load_market_data(market_file: str | None = None) -> Optional[pd.DataFrame]:
    """
    Load market data for regime detection enhancement.
    This is a placeholder for loading additional market indicators.
    """
    if not market_file or not Path(market_file).exists():
        return None

    try:
        # Assume CSV format with era column and market indicators
        market_df = pd.read_csv(market_file)
        return market_df
    except Exception as e:
        logging.warning(f"Failed to load market data from {market_file}: {e}")
        return None


def classify_market_regime(vix_value: float, vix_percentile: float,
                          market_corr: Optional[float] = None) -> str:
    """
    Classify market regime based on VIX level and optional market correlation.

    Args:
        vix_value: Current VIX value
        vix_percentile: Rolling percentile of VIX
        market_corr: Optional market correlation measure

    Returns:
        Regime classification string
    """
    # Primary classification based on VIX
    if vix_value > 25:
        return "high_vol_crisis"
    elif vix_value < 15:
        return "low_vol_grind"
    else:
        return "transition"

    # Could be enhanced with market correlation:
    # if market_corr is not None:
    #     if market_corr > 0.7 and vix_percentile > 75:
    #         return "crisis_high_corr"
    #     elif market_corr < 0.3 and vix_percentile < 25:
    #         return "grind_low_corr"


def calculate_regime_transitions(regime_series: pd.Series) -> Dict:
    """Calculate regime transition statistics."""
    transitions = {}
    prev_regime = None

    for current_regime in regime_series:
        if prev_regime is not None and current_regime != prev_regime:
            transition_key = f"{prev_regime}_to_{current_regime}"
            transitions[transition_key] = transitions.get(transition_key, 0) + 1
        prev_regime = current_regime

    return transitions


def get_regime_features_for_model(vix_data: pd.DataFrame,
                                 window_sizes: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """
    Create regime-based features for model input.

    Args:
        vix_data: DataFrame with VIX and regime data
        window_sizes: Rolling window sizes for feature calculation

    Returns:
        DataFrame with regime features
    """
    features_df = vix_data.copy()

    # Rolling VIX statistics
    for window in window_sizes:
        features_df[f'vix_mean_{window}d'] = features_df['vix'].rolling(window=window, min_periods=1).mean()
        features_df[f'vix_std_{window}d'] = features_df['vix'].rolling(window=window, min_periods=1).std()
        features_df[f'vix_percentile_{window}d'] = (
            features_df['vix'].rolling(window=window, min_periods=1)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
        )

    # Regime dummy variables
    regime_dummies = pd.get_dummies(features_df['vix_regime'], prefix='regime')
    features_df = pd.concat([features_df, regime_dummies], axis=1)

    # Regime change indicators
    features_df['regime_changed'] = (features_df['vix_regime'] != features_df['vix_regime'].shift(1)).astype(int)

    return features_df
