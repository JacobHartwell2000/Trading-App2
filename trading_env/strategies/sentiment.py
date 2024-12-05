from .base_strategy import BaseStrategy
import pandas as pd
import praw
from datetime import datetime, timedelta
from textblob import TextBlob
import numpy as np
from logger import log_activity

class SentimentStrategy(BaseStrategy):
    def __init__(self, sentiment_threshold=0.3, volume_impact_threshold=1.5):
        super().__init__()
        self.sentiment_threshold = sentiment_threshold
        self.volume_impact_threshold = volume_impact_threshold
        
        # Initialize Reddit client - you'll need to create a Reddit app
        self.reddit = praw.Reddit(
            client_id="VF6urbiYah1jIH1Rfdh64w",  # Get from Reddit developer console
            client_secret="-V2513i8RXK8p_Kl_gZFhbfyQdtVtA",  # Get from Reddit developer console
            user_agent="MySentimentAnalyzer/1.0"
        )

    def get_news_sentiment(self, symbol, date):
        """Get sentiment from Reddit r/stocks"""
        try:
            # Convert date to datetime if it's not already
            target_date = pd.to_datetime(date)
            start_time = int((target_date - timedelta(days=1)).timestamp())
            end_time = int(target_date.timestamp())
            
            # Search for posts containing the symbol
            subreddit = self.reddit.subreddit('stocks')
            posts = subreddit.search(f'{symbol}', time_filter='day', sort='relevance')
            
            sentiments = []
            total_score = 0  # Use post score as volume indicator
            
            for post in posts:
                if start_time <= post.created_utc <= end_time:
                    blob = TextBlob(f"{post.title} {post.selftext}")
                    sentiments.append(blob.sentiment.polarity)
                    total_score += post.score
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            return {
                'sentiment_score': avg_sentiment,
                'volume': total_score,
                'impact_score': min(total_score / 100, 2.0)  # Normalize impact
            }
        except Exception as e:
            print(f"Error in news sentiment: {e}")
            return {'sentiment_score': 0.0, 'volume': 0, 'impact_score': 0.0}

    def get_social_sentiment(self, symbol, date):
        """Get sentiment from Reddit r/wallstreetbets"""
        try:
            target_date = pd.to_datetime(date)
            start_time = int((target_date - timedelta(days=1)).timestamp())
            end_time = int(target_date.timestamp())
            
            subreddit = self.reddit.subreddit('wallstreetbets')
            posts = subreddit.search(f'{symbol}', time_filter='day', sort='relevance')
            
            sentiments = []
            total_score = 0
            
            for post in posts:
                if start_time <= post.created_utc <= end_time:
                    # Analyze both title and post content
                    blob = TextBlob(f"{post.title} {post.selftext}")
                    sentiments.append(blob.sentiment.polarity)
                    total_score += post.score
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            return {
                'sentiment_score': avg_sentiment,
                'volume': total_score,
                'impact_score': min(total_score / 100, 2.0)
            }
        except Exception as e:
            print(f"Error in social sentiment: {e}")
            return {'sentiment_score': 0.0, 'volume': 0, 'impact_score': 0.0}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        for date in df.index:
            news_data = self.get_news_sentiment(df['symbol'][0], date)
            social_data = self.get_social_sentiment(df['symbol'][0], date)
            
            # Combined sentiment score
            combined_sentiment = (news_data['sentiment_score'] * news_data['impact_score'] + 
                                social_data['sentiment_score'] * social_data['impact_score']) / 2
            
            # Log sentiment analysis results
            log_activity(f"Sentiment Analysis for {df['symbol'][0]} on {date.strftime('%Y-%m-%d')}:")
            log_activity(f"News Sentiment: {news_data['sentiment_score']:.2f} (Impact: {news_data['impact_score']:.2f})")
            log_activity(f"Social Sentiment: {social_data['sentiment_score']:.2f} (Impact: {social_data['impact_score']:.2f})")
            log_activity(f"Combined Sentiment: {combined_sentiment:.2f}")
            
            # Generate signals based on sentiment
            if combined_sentiment > self.sentiment_threshold:
                signals[date] = 1
                log_activity(f"Generated BUY signal for {df['symbol'][0]} (Sentiment: {combined_sentiment:.2f})")
            elif combined_sentiment < -self.sentiment_threshold:
                signals[date] = -1
                log_activity(f"Generated SELL signal for {df['symbol'][0]} (Sentiment: {combined_sentiment:.2f})")
                
        return signals

    def get_optimal_parameters(self, market_regime: str) -> dict:
        params = {
            'low_volatility': {'sentiment_threshold': 0.3, 'volume_impact_threshold': 1.5},
            'medium_volatility': {'sentiment_threshold': 0.4, 'volume_impact_threshold': 2.0},
            'high_volatility': {'sentiment_threshold': 0.5, 'volume_impact_threshold': 2.5}
        }
        return params.get(market_regime, params['medium_volatility']) 