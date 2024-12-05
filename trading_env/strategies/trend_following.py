from .base_strategy import BaseStrategy
import pandas as pd
from ta.trend import ADXIndicator

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, short_period=20, long_period=50, adx_period=14):
        super().__init__()
        self.short_period = short_period
        self.long_period = long_period
        self.adx_period = adx_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Moving averages
        df['sma_short'] = df['Close'].rolling(self.short_period).mean()
        df['sma_long'] = df['Close'].rolling(self.long_period).mean()
        
        # ADX for trend strength
        adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=self.adx_period)
        df['adx'] = adx.adx()
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[(df['sma_short'] > df['sma_long']) & (df['adx'] > 25)] = 1  # Strong uptrend
        signals[(df['sma_short'] < df['sma_long']) & (df['adx'] > 25)] = -1  # Strong downtrend
        
        return signals

    def get_optimal_parameters(self, market_regime: str) -> dict:
        params = {
            'low_volatility': {'short_period': 20, 'long_period': 50},
            'medium_volatility': {'short_period': 15, 'long_period': 40},
            'high_volatility': {'short_period': 10, 'long_period': 30}
        }
        return params.get(market_regime, params['medium_volatility']) 