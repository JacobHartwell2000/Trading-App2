from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from scipy.stats import linregress
from logger import log_activity

class MarketRegimeStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.volatility_window = 20
        self.hurst_window = 20

    def get_optimal_parameters(self, regime: str) -> dict:
        """Get optimal parameters for the current market regime"""
        if regime == 'trending':
            return {
                'window_size': 20,
                'volatility_threshold': 0.015
            }
        elif regime == 'ranging':
            return {
                'window_size': 10,
                'volatility_threshold': 0.02
            }
        else:  # neutral or unknown
            return {
                'window_size': 15,
                'volatility_threshold': 0.0175
            }

    def calculate_hurst_exponent(self, series, window):
        """Calculate Hurst exponent to determine market regime"""
        try:
            log_activity("Starting Hurst exponent calculation...")
            
            # Convert to 1D array if necessary
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            elif isinstance(series, np.ndarray) and len(series.shape) > 1:
                series = series.ravel()
            
            if not isinstance(series, (pd.Series, np.ndarray)):
                series = pd.Series(series)
            
            # Ensure we have enough data points
            if len(series) < window:
                log_activity("Insufficient data points for Hurst calculation")
                return None
            
            # Convert to numpy array and handle NaN values
            prices = np.array(series.fillna(method='ffill'))
            returns = np.diff(np.log(prices))
            
            if np.any(np.isnan(returns)):
                log_activity("NaN values found in returns")
                return None
            
            # Calculate tau values using absolute returns for better stability
            tau = []
            lags = range(2, min(window // 2, len(returns) // 2))
            
            for lag in lags:
                # Use absolute returns for more stable calculations
                abs_returns = np.abs(returns)
                lag_returns = abs_returns[lag:]
                base_returns = abs_returns[:-lag]
                
                if len(lag_returns) != len(base_returns):
                    continue
                    
                # Calculate variance of differences
                diff = np.subtract(lag_returns, base_returns)
                tau_value = np.sqrt(np.mean(np.square(diff)))
                
                if not np.isnan(tau_value) and tau_value > 0:
                    tau.append(tau_value)
            
            if not tau or len(tau) < 4:
                log_activity("Insufficient tau values calculated")
                return None
            
            # Perform linear regression with numpy arrays
            x = np.log(np.array(lags[:len(tau)]))
            y = np.log(np.array(tau))
            
            slope, _, r_value, _, _ = linregress(x, y)
            hurst = slope / 2.0
            
            # Add offset to ensure positive Hurst exponent
            hurst = (hurst + 1) / 2
            
            log_activity(f"Hurst calculation completed: {hurst:.3f} (R² = {r_value**2:.3f})")
            
            # Validate Hurst exponent
            if not (0 <= hurst <= 1):
                log_activity(f"Invalid Hurst exponent: {hurst}")
                return None
                
            return hurst
            
        except Exception as e:
            log_activity(f"Error in Hurst calculation: {str(e)}")
            return None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on market regime"""
        try:
            if df is None or df.empty:
                return pd.Series()
            
            # Calculate volatility
            df['volatility'] = df['Close'].pct_change().rolling(self.volatility_window).std()
            
            # Calculate Hurst exponent for each window
            df['hurst'] = df['Close'].rolling(self.hurst_window).apply(
                lambda x: self.calculate_hurst_exponent(x, self.hurst_window)
            )
            
            # Generate signals with null checks
            signals = pd.Series(0, index=df.index)
            
            # Only generate signals where we have valid Hurst and volatility values
            valid_data = df.dropna(subset=['hurst', 'volatility'])
            
            for idx in valid_data.index:
                hurst = valid_data.loc[idx, 'hurst']
                vol = valid_data.loc[idx, 'volatility']
                
                if hurst is not None and vol is not None:
                    if hurst > 0.6 and vol < 0.015:
                        signals[idx] = 1    # Trending market
                    elif hurst < 0.4 and vol > 0.02:
                        signals[idx] = -1   # Mean-reverting market
            
            return signals
            
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return pd.Series(0, index=df.index if df is not None else [])

    def get_market_regime(self, df: pd.DataFrame) -> dict:
        """Determine the current market regime and related metrics"""
        try:
            log_activity("Starting market regime calculation...")
            
            if df is None or df.empty:
                return self._create_regime_dict('unknown', None, None, 'no_data')
            
            if 'Close' not in df.columns:
                return self._create_regime_dict('unknown', None, None, 'missing_close_column')
            
            # Get latest prices and ensure it's a Series
            latest_prices = df['Close'].tail(self.hurst_window).copy()
            log_activity(f"Processing {len(latest_prices)} price points")
            
            if len(latest_prices) < self.hurst_window:
                return self._create_regime_dict('unknown', None, None, 'insufficient_data')
            
            try:
                # Calculate Hurst exponent
                hurst = self.calculate_hurst_exponent(latest_prices, self.hurst_window)
                log_activity(f"Calculated Hurst exponent: {hurst}")
                
                # Calculate volatility (annualized)
                returns = latest_prices.pct_change().dropna()
                volatility = float(returns.std() * np.sqrt(252))
                log_activity(f"Calculated volatility: {volatility:.3f}")
                
                if hurst is not None and not np.isnan(volatility):
                    # Determine regime
                    if hurst > 0.6:
                        regime = 'trending'
                    elif hurst < 0.4:
                        regime = 'ranging'
                    else:
                        regime = 'neutral'
                    
                    log_activity(f"Determined regime: {regime} (Hurst: {hurst:.3f}, Volatility: {volatility:.3f})")
                    
                    return self._create_regime_dict(regime, hurst, volatility, 'success')
                else:
                    return self._create_regime_dict('unknown', None, None, 'calculation_failed')
                    
            except Exception as e:
                log_activity(f"Error in calculations: {str(e)}")
                return self._create_regime_dict('unknown', None, None, f'calculation_error: {str(e)}')
                
        except Exception as e:
            import traceback
            log_activity(f"Error in market regime calculation: {str(e)}")
            log_activity(f"Traceback: {traceback.format_exc()}")
            return self._create_regime_dict('unknown', None, None, f'error: {str(e)}')
            
    def _create_regime_dict(self, regime, hurst, volatility, reason):
        """Create a dictionary for the market regime"""
        return {
            'regime': regime,
            'hurst': float(hurst) if hurst is not None else None,
            'volatility': float(volatility) if volatility is not None else None,
            'reason': reason
        }
            