import yfinance as yf
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import time
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators.openai import OpenAIGenerator
from typing import List
import pandas as pd
from haystack import component
import os
from haystack.components.routers import ConditionalRouter
from haystack.utils import Secret
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from haystack.dataclasses import ChatMessage
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from openai import OpenAI
import base64
from bs4 import BeautifulSoup
import requests
import re
import logging
import json
import joblib
from logger import log_activity, get_activity_log
from typing import Dict

# Add these imports for the strategies
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.volume_profile import VolumeProfileStrategy
from strategies.market_regime import MarketRegimeStrategy
from strategies.options_flow import OptionsFlowStrategy
from strategies.sentiment import SentimentStrategy
from strategies.intermarket import IntermarketStrategy
from strategies.stock_discovery import StockDiscovery

# Alpaca API credentials
API_KEY = 'PK9AFUY7Y7VL5BAP2U46'
API_SECRET = 'GQmm6PjW1B5hbfnraub14XkrWmjnAFgkdarEqfTU'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading for testing

# Add this near your other constants
OPENAI_API_KEY = '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'

# At the top of the file, add debug logging for credentials
log_activity(f"Initializing Alpaca API with URL: {BASE_URL}")
log_activity("Verifying API key format...")
if not API_KEY or len(API_KEY) < 10:
    log_activity("API_KEY appears to be invalid", "error")
if not API_SECRET or len(API_SECRET) < 10:
    log_activity("API_SECRET appears to be invalid", "error")

class StrategyManager:
    def __init__(self):
        self.strategies = {
            'mean_reversion': {'weight': 1.0, 'performance': [], 'sharpe': 0},
            'trend_following': {'weight': 1.0, 'performance': [], 'sharpe': 0},
            'volume_profile': {'weight': 1.0, 'performance': [], 'sharpe': 0},
            'market_regime': {'weight': 1.0, 'performance': [], 'sharpe': 0},
            'options_flow': {'weight': 1.0, 'performance': [], 'sharpe': 0},
            'sentiment': {'weight': 1.0, 'performance': [], 'sharpe': 0},
            'intermarket': {'weight': 1.0, 'performance': [], 'sharpe': 0}
        }
        self.lookback_period = 30  # Days to evaluate strategy performance

    def calculate_strategy_performance(self, strategy_signals, returns):
        """Calculate Sharpe ratio and hit rate for each strategy"""
        for strategy, signal in strategy_signals.items():
            strategy_returns = returns[signal]  # Filter returns where strategy was active
            
            if len(strategy_returns) > 0:
                sharpe = np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std())
                hit_rate = (strategy_returns > 0).mean()
                
                self.strategies[strategy]['performance'].append({
                    'sharpe': sharpe,
                    'hit_rate': hit_rate
                })
                
                # Keep only recent performance
                self.strategies[strategy]['performance'] = \
                    self.strategies[strategy]['performance'][-self.lookback_period:]

    def update_weights(self):
        """Dynamically adjust strategy weights based on performance"""
        total_sharpe = 0
        
        # Calculate recent Sharpe ratio for each strategy
        for strategy in self.strategies:
            if self.strategies[strategy]['performance']:
                recent_sharpe = np.mean([p['sharpe'] for p in 
                    self.strategies[strategy]['performance'][-self.lookback_period:]])
                self.strategies[strategy]['sharpe'] = recent_sharpe
                total_sharpe += max(0, recent_sharpe)  # Only consider positive Sharpe ratios
        
        # Update weights based on relative performance
        if total_sharpe > 0:
            for strategy in self.strategies:
                self.strategies[strategy]['weight'] = \
                    max(0, self.strategies[strategy]['sharpe']) / total_sharpe

class TradingBot:
    def __init__(self, symbols):
        try:
            log_activity("Initializing TradingBot...")
            
            # Log API configuration (mask sensitive data)
            log_activity(f"API Key length: {len(API_KEY)}")
            log_activity(f"API Secret length: {len(API_SECRET)}")
            log_activity(f"Using Alpaca URL: {BASE_URL}")
                
            # Initialize Alpaca API with explicit error handling
            try:
                self.alpaca = tradeapi.REST(
                    API_KEY,
                    API_SECRET,
                    BASE_URL,
                    api_version='v2'
                )
                
                # Test connection by getting account info
                account = self.alpaca.get_account()
                log_activity("Successfully connected to Alpaca API")
                log_activity(f"Account status: {account.status}")
                log_activity(f"Account currency: {account.currency}")
                
            except tradeapi.rest.APIError as e:
                log_activity(f"Alpaca API Error: {str(e)}", "error")
                raise
            except Exception as e:
                log_activity(f"Failed to initialize Alpaca API: {str(e)}", "error")
                raise
            
            # Initialize other attributes
            self.symbols = symbols
            self.models = {}
            self.scaler = None
            self.feature_columns = None
            self.is_scaler_fitted = False
            
            # Load or train models
            log_activity("Loading or training models...")
            self.load_or_train_models()
            
            log_activity("TradingBot initialization completed successfully")
            
        except Exception as e:
            log_activity(f"Failed to initialize TradingBot: {str(e)}", "error")
            raise

    def initialize_strategies(self):
        """Initialize all trading strategies"""
        log_activity("Initializing trading strategies")
        self.strategies = {
            'sentiment': SentimentStrategy(),
            'market_regime': MarketRegimeStrategy(),
            'mean_reversion': {
                'strategy': MeanReversionStrategy(),
                'weight': 1.0,
                'enabled': True
            },
            'trend_following': {
                'strategy': TrendFollowingStrategy(),
                'weight': 1.0,
                'enabled': True
            },
            'volume_profile': {
                'strategy': VolumeProfileStrategy(),
                'weight': 1.0,
                'enabled': True
            },
            'options_flow': {
                'strategy': OptionsFlowStrategy(),
                'weight': 1.0,
                'enabled': True
            },
            'intermarket': {
                'strategy': IntermarketStrategy(),
                'weight': 1.0,
                'enabled': True
            }
        }
        log_activity("Trading strategies initialized successfully")

    def get_market_regime(self, df: pd.DataFrame) -> str:
        """Determine current market regime"""
        volatility = df['Close'].pct_change().std() * np.sqrt(252)
        
        if volatility < 0.15:
            return 'low_volatility'
        elif volatility > 0.25:
            return 'high_volatility'
        return 'medium_volatility'

    def update_strategy_parameters(self, df: pd.DataFrame):
        """Update strategy parameters based on market regime"""
        market_regime = self.get_market_regime(df)
        
        for strategy_name, strategy_dict in self.strategies.items():
            if strategy_dict['enabled']:
                optimal_params = strategy_dict['strategy'].get_optimal_parameters(market_regime)
                for param, value in optimal_params.items():
                    setattr(strategy_dict['strategy'], param, value)

    def calculate_strategy_performance(self, signals: Dict[str, pd.Series], 
                                    returns: pd.Series, window: int = 20):
        """Calculate and update strategy performance metrics"""
        for strategy_name, signal in signals.items():
            if self.strategies[strategy_name]['enabled']:
                # Calculate strategy returns
                strategy_returns = returns * signal.shift(1)
                
                # Calculate metrics
                sharpe = np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std())
                hit_rate = (strategy_returns > 0).mean()
                max_drawdown = (strategy_returns + 1).cumprod().div(
                    (strategy_returns + 1).cumprod().cummax()
                ).min() - 1
                
                # Store metrics
                self.performance_metrics[strategy_name].append({
                    'timestamp': datetime.now(),
                    'sharpe': sharpe,
                    'hit_rate': hit_rate,
                    'max_drawdown': max_drawdown
                })
                
                # Keep only recent performance
                self.performance_metrics[strategy_name] = \
                    self.performance_metrics[strategy_name][-window:]

    def update_strategy_weights(self):
        """Dynamically adjust strategy weights based on performance"""
        total_score = 0
        
        # Calculate performance score for each strategy
        for strategy_name, strategy_dict in self.strategies.items():
            if strategy_dict['enabled'] and self.performance_metrics[strategy_name]:
                metrics = self.performance_metrics[strategy_name][-1]
                
                score = (
                    max(0, metrics['sharpe']) * 0.4 +
                    metrics['hit_rate'] * 0.4 +
                    max(0, 1 + metrics['max_drawdown']) * 0.2
                )
                
                strategy_dict['score'] = score
                total_score += score

        # Update weights based on relative performance
        if total_score > 0:
            for strategy_dict in self.strategies.values():
                if strategy_dict['enabled']:
                    strategy_dict['weight'] = strategy_dict['score'] / total_score
                else:
                    strategy_dict['weight'] = 0

    def generate_combined_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate combined trading signal from all strategies"""
        signals = {}
        weighted_signal = pd.Series(0, index=df.index)
        
        # Generate signals from each strategy
        for strategy_name, strategy_dict in self.strategies.items():
            if strategy_dict['enabled']:
                try:
                    signals[strategy_name] = strategy_dict['strategy'].generate_signals(df)
                    weighted_signal += signals[strategy_name] * strategy_dict['weight']
                except Exception as e:
                    log_activity(f"Error in {strategy_name}: {str(e)}")
                    strategy_dict['enabled'] = False

        # Calculate strategy performance
        returns = df['Close'].pct_change()
        self.calculate_strategy_performance(signals, returns)
        
        # Update weights for next iteration
        self.update_strategy_weights()
        
        return weighted_signal

    def execute_trade(self, symbol, prediction, confidence):
        try:
            # Log analysis
            log_activity("", "analysis", {
                'symbol': symbol,
                'prediction': prediction[0],
                'confidence': confidence,
                'strategy_signals': self.get_strategy_signals(symbol),
                'final_signal': self.final_signal
            })
            
            if self.should_execute_trade(prediction, confidence):
                # Calculate trade parameters
                position_size = self.calculate_position_size(symbol, confidence)
                current_price = float(self.alpaca.get_latest_trade(symbol).price)
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)
                
                # Log trade execution
                log_activity("", "trade", {
                    'symbol': symbol,
                    'side': 'BUY',
                    'position_size': position_size,
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
                
                # Execute the trade
                self.alpaca.submit_order(...)
                
            else:
                # Log reasons for not trading
                reasons = []
                if prediction[0] != 1:
                    reasons.append("ML model predicting SELL")
                if confidence <= 0.7:
                    reasons.append("Confidence below threshold (0.7)")
                if self.final_signal <= 0.5:
                    reasons.append("Combined signal not strong enough")
                    
                log_activity("", "no_trade", {
                    'symbol': symbol,
                    'reasons': reasons
                })
                
        except Exception as e:
            log_activity(f"Error executing trade for {symbol}: {str(e)}")

    def load_or_train_models(self):
        """Load existing models or train new ones"""
        models_path = "saved_models"
        scaler_path = f"{models_path}/scaler.pkl"
        
        try:
            # Try to load existing models
            if os.path.exists(models_path):
                log_activity("Loading existing trading models...")
                self.models = {
                    'rf': joblib.load(f"{models_path}/rf_model.pkl"),
                    'gb': joblib.load(f"{models_path}/gb_model.pkl"),
                    'xgb': joblib.load(f"{models_path}/xgb_model.pkl")
                }
                self.scaler = joblib.load(scaler_path)
                self.feature_columns = joblib.load(f"{models_path}/feature_columns.pkl")
                self.is_scaler_fitted = True
                log_activity("Models loaded successfully!")
                return
        except Exception as e:
            log_activity(f"Error loading models: {str(e)}")
        
        # If loading fails or models don't exist, train new ones
        log_activity("Training new models...")
        self.models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10),
            'gb': GradientBoostingClassifier(n_estimators=200),
            'xgb': xgb.XGBClassifier(objective='binary:logistic')
        }
        self.scaler = StandardScaler()
        self.train_models()
        
        # Save the trained models
        try:
            os.makedirs(models_path, exist_ok=True)
            for name, model in self.models.items():
                joblib.dump(model, f"{models_path}/{name}_model.pkl")
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.feature_columns, f"{models_path}/feature_columns.pkl")
            log_activity("Models saved successfully!")
        except Exception as e:
            log_activity(f"Error saving models: {str(e)}")

    def train_models(self):
        """Train all models with historical data"""
        log_activity("Starting model training...")
        training_data = pd.DataFrame()
        
        # Collect training data from all symbols
        for symbol in self.symbols:
            log_activity(f"Getting training data for {symbol}...")
            df = self.get_historical_data(symbol)
            if df is not None:
                training_data = pd.concat([training_data, df])
        
        if training_data.empty:
            raise ValueError("No training data available")
            
        # Prepare features for training
        X = training_data[self.feature_columns]
        y = training_data['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit scaler
        self.scaler.fit(X_train)
        self.is_scaler_fitted = True
        
        # Scale training data
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train each model
        for name, model in self.models.items():
            try:
                log_activity(f"Training {name} model...")
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                log_activity(f"{name} model accuracy: {score:.2f}")
            except Exception as e:
                log_activity(f"Error training {name} model: {str(e)}")
        
        log_activity("Model training completed!")

    def prepare_features(self, df):
        """Prepare features for analysis"""
        try:
            if df is None or df.empty:
                return None
            
            df = df.copy()
            
            # Add technical indicators
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # RSI
            rsi = RSIIndicator(close=df['Close'], window=14)
            df['RSI'] = rsi.rsi()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_middle'] = bb.bollinger_mavg()
            
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Drop any NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            log_activity(f"Error in prepare_features: {str(e)}")
            return None

    def calculate_position_size(self, symbol, confidence):
        """Calculate position size based on risk management rules"""
        account = self.alpaca.get_account()
        equity = float(account.equity)
        
        # Get current price
        current_price = float(self.alpaca.get_latest_trade(symbol).price)
        
        # Calculate maximum risk amount
        risk_amount = equity * self.risk_per_trade
        
        # Calculate position size based on stop loss
        shares = risk_amount / (current_price * self.stop_loss_pct)
        
        # Adjust position size based on model confidence
        shares = int(shares * confidence)
        
        # Ensure minimum position size
        return max(1, min(shares, int(equity * 0.1 / current_price)))

    def portfolio_diversification(self):
        """Check portfolio diversification"""
        positions = self.alpaca.list_positions()
        sector_exposure = {}
        
        for position in positions:
            # Get sector information (you might need a separate API for this)
            sector = yf.Ticker(position.symbol).info.get('sector', 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + float(position.market_value)
        
        # Calculate sector concentration
        total_value = sum(sector_exposure.values())
        sector_weights = {k: v/total_value for k, v in sector_exposure.items()}
        
        # Return True if no sector exceeds 30% of portfolio
        return all(weight <= 0.3 for weight in sector_weights.values())

    def ensemble_predict(self, features):
        """Ensemble prediction using multiple models"""
        try:
            if self.feature_columns is None:
                log_activity("Error: Feature columns not initialized")
                return None, None
            
            features_df = pd.DataFrame(features, columns=self.feature_columns)
            predictions = []
            probabilities = []
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(features_df)
                    prob = model.predict_proba(features_df)
                    predictions.append(pred)
                    probabilities.append(prob)
                    log_activity(f"{name} model prediction complete")
                except Exception as e:
                    log_activity(f"Error in {name} prediction: {str(e)}")
                    continue
            
            if not predictions:
                log_activity("No successful predictions from any model")
                return None, None
            
            final_pred = np.array([1 if np.mean([p[0] for p in predictions]) > 0.5 else 0])
            final_prob = np.mean([p[0][1] for p in probabilities])
            
            log_activity(f"Ensemble prediction complete: {final_pred[0]} with confidence {final_prob:.2f}")
            return final_pred, final_prob
            
        except Exception as e:
            log_activity(f"Error in ensemble_predict: {str(e)}")
            return None, None

    def enhance_data_processing(self):
        """Add these features to improve data quality"""
        # Market regime detection
        self.add_market_regime()
        
        # Alternative data sources
        self.add_sentiment_analysis()
        self.add_options_flow()
        
        # Economic indicators
        self.add_macro_indicators()
        
        # Enhanced volatility metrics
        self.add_volatility_regime()

    def add_sentiment_analysis(self):
        """Add sentiment analysis from multiple sources"""
        # News sentiment (using NewsAPI or similar)
        # Social media sentiment (Twitter, Reddit)
        # Analyst ratings
        # Insider trading data

    def improve_ml_models(self):
        """Add these ML enhancements"""
        # Deep Learning models
        self.models['lstm'] = self.create_lstm_model()
        self.models['transformer'] = self.create_transformer_model()
        
        # Advanced feature selection
        self.feature_importance_analysis()
        
        # Hyperparameter optimization
        self.optimize_models()
        
        # Online learning capabilities
        self.implement_online_learning()

    def enhance_risk_management(self):
        """Add these risk management features"""
        # Dynamic position sizing
        position_size = self.calculate_dynamic_position_size(
            volatility=self.current_volatility,
            market_regime=self.market_regime,
            correlation_risk=self.portfolio_correlation
        )
        
        # Advanced portfolio optimization
        self.optimize_portfolio_weights()
        
        # Drawdown protection
        self.implement_drawdown_protection()
        
        # Correlation-based position limits
        self.adjust_for_correlation_risk()

    def log_trading_action(self, action_type, details):
        """Log trading actions for later querying"""
        timestamp = datetime.now().isoformat()
        action = {
            "timestamp": timestamp,
            "type": action_type,
            "details": details
        }
        self.trading_history.append(action)
        
        try:
            # Create simplified chat message
            message = ChatMessage(
                content=json.dumps(action),
                role="assistant"
            )
            
            # Use correct parameter name for message writer
            self.message_writer.run(message=message)  # Changed from messages=[message]
        except Exception as e:
            print(f"Error logging message: {str(e)}")

    def query_trading_history(self, query):
        """Query the trading history using natural language"""
        try:
            # Get the latest predictions for context
            latest_predictions = []
            for symbol in self.symbols:
                latest_predictions.append(f"{symbol}: prediction=0, confidence={self.last_predictions.get(symbol, {}).get('confidence', 0):.2f}")
            
            # Create a system status message
            system_status = f"""
            System Status:
            - Total trades executed: {len([x for x in self.trading_history if x['type'] == 'trade_execution'])}
            - Latest predictions:
              {chr(10).join(latest_predictions)}
            - Trading threshold: Requires prediction=1 and confidence>0.7
            - Current market status: {'Open' if self.alpaca.get_clock().is_open else 'Closed'}
            """
            
            # Create the prompt with system status
            prompt = f"""
            Based on this trading system status:
            {system_status}
            
            Please answer this question: {query}
            
            If no trades have been executed, explain why based on the system status.
            """
            
            # Get response from GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful trading assistant that explains the trading system's behavior."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error querying trading history: {str(e)}")
            return f"Error processing query: {str(e)}"

    def get_last_context(self):
        """Return the context used in the last query"""
        return self.last_context

    def get_historical_data(self, symbol):
        """Get historical data for a symbol"""
        try:
            log_activity(f"Fetching historical data for {symbol}")
            end = datetime.now()
            start = end - timedelta(days=365)
            
            df = yf.download(symbol, start=start, end=end)
            
            if df.empty:
                log_activity(f"No data received for {symbol}")
                return None
            
            log_activity(f"Retrieved {len(df)} data points for {symbol}")
            return df  # Return the raw DataFrame without prepare_features
            
        except Exception as e:
            log_activity(f"Error getting historical data for {symbol}: {str(e)}")
            return None

    def force_retrain(self):
        """Force retraining of all models"""
        print("Force retraining models...")
        self.train_models()
        
        # Save the retrained models
        models_path = "saved_models"
        try:
            os.makedirs(models_path, exist_ok=True)
            for name, model in self.models.items():
                joblib.dump(model, f"{models_path}/{name}_model.pkl")
            joblib.dump(self.scaler, f"{models_path}/scaler.pkl")
            joblib.dump(self.feature_columns, f"{models_path}/feature_columns.pkl")
            print("Retrained models saved successfully!")
        except Exception as e:
            print(f"Error saving retrained models: {str(e)}")

    def get_combined_signals(self, df, symbol):
        """Get signals from all strategies and combine them"""
        signals = {}
        
        # Mean Reversion Signals
        signals['mean_reversion'] = self.add_mean_reversion_signals(df)
        
        # Trend Following Signals
        signals['trend_following'] = self.add_trend_signals(df)
        
        # Volume Profile Signals
        signals['volume_profile'] = self.add_volume_analysis(df)
        
        # Market Regime Signals
        signals['market_regime'] = self.detect_market_regime(df)
        
        # Options Flow Signals
        signals['options_flow'] = self.add_options_signals(df)
        
        # Sentiment Signals
        signals['sentiment'] = self.add_sentiment_signals(df)
        
        # Intermarket Signals
        signals['intermarket'] = self.add_intermarket_signals(df, symbol)
        
        return signals

    def get_weighted_signal(self, signals):
        """Combine all strategy signals using current weights"""
        final_signal = 0
        total_weight = 0
        
        for strategy, signal in signals.items():
            weight = self.strategy_manager.strategies[strategy]['weight']
            final_signal += signal * weight
            total_weight += weight
        
        if total_weight > 0:
            final_signal /= total_weight
        
        return final_signal > 0.5  # Convert to binary signal

    def adjust_risk_parameters(self, symbol, active_strategies):
        """Adjust risk parameters based on active strategies"""
        base_stop_loss = self.stop_loss_pct
        base_take_profit = self.take_profit_pct
        
        # Adjust based on active strategies
        if 'mean_reversion' in active_strategies:
            # Tighter stops for mean reversion
            base_stop_loss *= 0.8
            base_take_profit *= 0.8
        
        if 'trend_following' in active_strategies:
            # Wider stops for trend following
            base_stop_loss *= 1.2
            base_take_profit *= 1.5
        
        # Adjust based on market regime
        if 'market_regime' in active_strategies:
            regime = active_strategies['market_regime']
            if regime == 'high_volatility':
                base_stop_loss *= 1.3
                base_take_profit *= 1.3
        
        return base_stop_loss, base_take_profit

    def monitor_strategy_performance(self):
        """Monitor and log strategy performance"""
        log_activity("\nStrategy Performance Summary:")
        for strategy, data in self.strategy_manager.strategies.items():
            if data['performance']:
                recent_perf = data['performance'][-self.strategy_manager.lookback_period:]
                avg_sharpe = np.mean([p['sharpe'] for p in recent_perf])
                avg_hit_rate = np.mean([p['hit_rate'] for p in recent_perf])
                
                log_activity(f"{strategy}:")
                log_activity(f"  Weight: {data['weight']:.2f}")
                log_activity(f"  Sharpe: {avg_sharpe:.2f}")
                log_activity(f"  Hit Rate: {avg_hit_rate:.2%}")

    def discover_new_opportunities(self):
        """Find new trading opportunities"""
        try:
            current_time = time.time()
            if current_time - self.last_discovery_time >= self.discovery_interval:
                log_activity("Running stock discovery...")
                
                # Get new opportunities
                opportunities = self.stock_discovery.find_opportunities()
                
                # Filter and add promising stocks
                for opp in opportunities:
                    if opp['score'] > 0.8:  # High confidence threshold
                        symbol = opp['symbol']
                        if symbol not in self.symbols:
                            log_activity(f"Adding new symbol {symbol} to watchlist (Score: {opp['score']:.2f})")
                            self.symbols.append(symbol)
                            
                            # Log discovery details
                            log_activity("", "discovery", {
                                'symbol': symbol,
                                'score': opp['score'],
                                'analysis': opp['analysis'],
                                'social_data': opp['social_data']
                            })
                
                self.last_discovery_time = current_time
                
        except Exception as e:
            log_activity(f"Error in stock discovery: {str(e)}")

    def get_activity_logs(self):
        """Return recent activity logs"""
        return get_activity_log()

    def get_positions(self):
        """Get current positions with additional details"""
        try:
            positions = self.alpaca.list_positions()
            return [{
                'symbol': pos.symbol,
                'qty': pos.qty,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pl': pos.unrealized_pl,
                'unrealized_plpc': pos.unrealized_plpc,
                'avg_entry_price': pos.avg_entry_price
            } for pos in positions]
        except Exception as e:
            log_activity(f"Error fetching positions: {str(e)}", "error")
            return []

    def get_account_status(self):
        """Get account status and performance metrics"""
        try:
            # Test Alpaca connection first
            log_activity("Testing Alpaca API connection...")
            if not self.alpaca:
                log_activity("Alpaca API not initialized", "error")
                return None
            
            # Verify API credentials and get account info
            log_activity("Attempting to get account information...")
            
            account = self.alpaca.get_account()
            log_activity("Successfully retrieved account information")
            
            # Format the response with explicit type conversion and error handling
            status_data = {
                'equity': float(account.equity or 0),
                'cash': float(account.cash or 0),
                'buying_power': float(account.buying_power or 0),
                'portfolio_value': float(account.portfolio_value or 0),
                'status': account.status,
                'currency': account.currency,
                'account_number': account.account_number
            }
            
            # Validate the data before returning
            for key, value in status_data.items():
                if key not in ['status', 'currency', 'account_number'] and (not isinstance(value, (int, float)) or np.isnan(value)):
                    log_activity(f"Invalid {key} value: {value}", "error")
                    status_data[key] = 0
            
            log_activity("Account status data formatted successfully")
            return status_data
            
        except Exception as e:
            log_activity(f"Error getting account status: {str(e)}", "error")
            return None

    def get_sentiment_analysis(self):
        """Get current market sentiment analysis"""
        try:
            # Initialize sentiment strategy if not already done
            if not hasattr(self, 'sentiment_strategy'):
                self.sentiment_strategy = SentimentStrategy()
            
            # Get sentiment data
            sentiment_data = self.sentiment_strategy.analyze_sentiment()
            
            # Calculate overall sentiment as weighted average
            if sentiment_data:
                news_sentiment = sentiment_data.get('news', 0)
                social_sentiment = sentiment_data.get('social', 0)
                
                # Weight news sentiment slightly higher than social
                overall_sentiment = (news_sentiment * 0.6) + (social_sentiment * 0.4)
                
                return {
                    'news': news_sentiment,
                    'social': social_sentiment,
                    'overall': overall_sentiment
                }
            
            return None
            
        except Exception as e:
            log_activity(f"Error getting sentiment analysis: {str(e)}", "error")
            return None

def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    try:
        log_activity("Starting trading bot...")
        bot = TradingBot(symbols)
        
        while True:
            try:
                print("\nOptions:")
                print("1. Process trades")
                print("2. View trading history")
                print("3. Force retrain models")
                print("4. Discover new stocks")  # New option
                print("5. Exit")
                
                choice = input("Enter your choice (1-5): ")
                
                if choice == '1':
                    # Run discovery before processing trades
                    bot.discover_new_opportunities()
                    
                    # Process trades for each symbol
                    for symbol in symbols:
                        print(f"\nProcessing {symbol}...")
                        historical_data = bot.get_historical_data(symbol)
                        
                        if historical_data is None:
                            print(f"Skipping {symbol} due to data issues")
                            continue
                        
                        if not bot.alpaca.get_clock().is_open:
                            print("Market is closed")
                            continue
                        
                        if bot.feature_columns:
                            latest_data = historical_data.iloc[-1:][bot.feature_columns]
                            prediction, confidence = bot.ensemble_predict(latest_data)
                            
                            if prediction is not None and confidence is not None:
                                print(f"Prediction for {symbol}: {prediction[0]}, Confidence: {confidence:.2f}")
                                bot.execute_trade(symbol, prediction, confidence)
                            else:
                                print(f"Skipping trade for {symbol} due to prediction error")
                        else:
                            print("Feature columns not initialized")
                            continue
                
                elif choice == '2':
                    while True:
                        print("\nTrading History Query (type 'back' to return to main menu)")
                        query = input("Enter your question about trading history: ")
                        
                        if query.lower() == 'back':
                            break
                        
                        answer = bot.query_trading_history(query)
                        print(f"\nAnswer: {answer}\n")
                
                elif choice == '3':
                    bot.force_retrain()
                
                elif choice == '4':
                    # Run stock discovery manually
                    print("\nRunning stock discovery...")
                    bot.discover_new_opportunities()
                    print("\nCurrent watchlist:")
                    for symbol in bot.symbols:
                        print(f"- {symbol}")
                
                elif choice == '5':
                    print("Exiting trading bot...")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
                
                # Add performance monitoring every hour
                if datetime.now().minute == 0:
                    bot.monitor_strategy_performance()
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                print("Waiting 60 seconds before retry...")
                time.sleep(60)
                
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
        return

if __name__ == "__main__":
    main()
