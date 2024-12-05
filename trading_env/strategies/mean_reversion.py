from .base_strategy import BaseStrategy
import pandas as pd
from ta.momentum import RSIIndicator
import numpy as np

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback_period=20, zscore_threshold=2, rsi_period=14):
        super().__init__()
        self.lookback_period = lookback_period
        self.zscore_threshold = zscore_threshold
        self.rsi_period = rsi_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Z-score calculation
        df['price_zscore'] = (df['Close'] - df['Close'].rolling(self.lookback_period).mean()) / \
                            df['Close'].rolling(self.lookback_period).std()
        
        # RSI calculation
        rsi = RSIIndicator(df['Close'], window=self.rsi_period)
        df['rsi'] = rsi.rsi()
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[(df['price_zscore'] < -self.zscore_threshold) & (df['rsi'] < 30)] = 1  # Buy signals
        signals[(df['price_zscore'] > self.zscore_threshold) & (df['rsi'] > 70)] = -1  # Sell signals
        
        return signals

    def get_optimal_parameters(self, market_regime: str) -> dict:
        params = {
            'low_volatility': {'lookback_period': 20, 'zscore_threshold': 2},
            'medium_volatility': {'lookback_period': 15, 'zscore_threshold': 2.5},
            'high_volatility': {'lookback_period': 10, 'zscore_threshold': 3}
        }
        return params.get(market_regime, params['medium_volatility']) 