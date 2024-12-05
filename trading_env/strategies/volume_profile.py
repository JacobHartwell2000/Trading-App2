from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class VolumeProfileStrategy(BaseStrategy):
    def __init__(self, volume_ma_period=20):
        super().__init__()
        self.volume_ma_period = volume_ma_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # VWAP calculation
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Volume momentum
        df['volume_ma'] = df['Volume'].rolling(self.volume_ma_period).mean()
        df['volume_momentum'] = df['Volume'] / df['volume_ma']
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[(df['Close'] < df['vwap']) & (df['volume_momentum'] > 1.5)] = 1  # Buy on high volume dips
        signals[(df['Close'] > df['vwap']) & (df['volume_momentum'] > 1.5)] = -1  # Sell on high volume rips
        
        return signals

    def get_optimal_parameters(self, market_regime: str) -> dict:
        params = {
            'low_volatility': {'volume_ma_period': 20},
            'medium_volatility': {'volume_ma_period': 15},
            'high_volatility': {'volume_ma_period': 10}
        }
        return params.get(market_regime, params['medium_volatility']) 