from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals for the strategy"""
        pass
    
    @abstractmethod
    def get_optimal_parameters(self, market_regime: str) -> dict:
        """Get optimal strategy parameters based on market regime"""
        pass 