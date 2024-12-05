import pandas as pd
import numpy as np

class IntermarketStrategy:
    def __init__(self, related_markets=None, correlation_threshold=0.7, lookback_period=20):
        self.related_markets = related_markets or {}  # Dict of related market DataFrames
        self.correlation_threshold = correlation_threshold
        self.lookback_period = lookback_period

    def generate_signals(self, df):
        """Generate trading signals based on intermarket relationships.
        
        Args:
            df (pd.DataFrame): Primary market data with OHLCV columns
        Returns:
            pd.Series: Trading signals (1: Buy, -1: Sell, 0: Hold)
        """
        signals = pd.Series(0, index=df.index)
        
        # Calculate correlations with related markets
        for market_name, market_data in self.related_markets.items():
            correlation = self._calculate_rolling_correlation(
                df['Close'], 
                market_data['Close'],
                self.lookback_period
            )
            
            # Generate signals based on correlation strength and price movements
            signals += np.where(
                (correlation.abs() > self.correlation_threshold),
                df['Close'].pct_change().apply(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                ),
                0
            )
        
        # Normalize signals
        return signals.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    def _calculate_rolling_correlation(self, series1, series2, window):
        """Calculate rolling correlation between two price series."""
        return series1.rolling(window).corr(series2)

    def get_optimal_parameters(self, market_regime):
        """Adjust strategy parameters based on market regime.
        
        Args:
            market_regime (str): Current market regime ('trending', 'ranging', etc.)
        Returns:
            dict: Optimized parameters
        """
        params = {
            'trending': {
                'correlation_threshold': 0.8,
                'lookback_period': 30
            },
            'ranging': {
                'correlation_threshold': 0.6,
                'lookback_period': 15
            }
        }
        return params.get(market_regime, {
            'correlation_threshold': self.correlation_threshold,
            'lookback_period': self.lookback_period
        })