from typing import Dict, List
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import logging
import praw

class StockDiscovery:
    def __init__(self):
        # Initialize Reddit client (free, just need to register)
          self.reddit = praw.Reddit(
            client_id="VF6urbiYah1jIH1Rfdh64w",  # Get from Reddit developer console
            client_secret="-V2513i8RXK8p_Kl_gZFhbfyQdtVtA",  # Get from Reddit developer console
            user_agent="MySentimentAnalyzer/1.0"
        )
        
    def get_reddit_trending(self) -> List[Dict]:
        """Get trending stocks from Reddit"""
        try:
            trending_stocks = {}
            subreddits = ['wallstreetbets', 'stocks', 'investing']
            
            for sub_name in subreddits:
                subreddit = self.reddit.subreddit(sub_name)
                # Get hot posts from last 24 hours
                for post in subreddit.hot(limit=50):
                    if post.created_utc > (datetime.now() - timedelta(days=1)).timestamp():
                        # Extract stock symbols (e.g., $AAPL or just AAPL)
                        symbols = re.findall(r'[\$]?[A-Z]{1,5}\b', post.title + post.selftext)
                        
                        for symbol in symbols:
                            symbol = symbol.replace('$', '')
                            if self.is_valid_stock(symbol):
                                if symbol not in trending_stocks:
                                    trending_stocks[symbol] = {
                                        'mentions': 0,
                                        'score': 0,
                                        'sentiment': 0
                                    }
                                trending_stocks[symbol]['mentions'] += 1
                                trending_stocks[symbol]['score'] += post.score
                                
                                # Simple sentiment (based on keywords)
                                sentiment = self.simple_sentiment_analysis(post.title + post.selftext)
                                trending_stocks[symbol]['sentiment'] += sentiment
            
            return trending_stocks
        except Exception as e:
            logging.error(f"Error getting Reddit trends: {e}")
            return {}

    def simple_sentiment_analysis(self, text: str) -> float:
        """Simple keyword-based sentiment analysis"""
        positive_words = ['buy', 'bull', 'long', 'up', 'growth', 'positive', 'breakout']
        negative_words = ['sell', 'bear', 'short', 'down', 'crash', 'negative', 'fail']
        
        text = text.lower()
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count + negative_count == 0:
            return 0
        return (positive_count - negative_count) / (positive_count + negative_count)

    def get_finviz_data(self, symbol: str) -> Dict:
        """Scrape FinViz for stock data (free version)"""
        try:
            url = f'https://finviz.com/quote.ashx?t={symbol}'
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic stock data
            data = {}
            snapshot_table = soup.find('table', class_='snapshot-table2')
            if snapshot_table:
                rows = snapshot_table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    for i in range(0, len(cols), 2):
                        if i + 1 < len(cols):
                            data[cols[i].text] = cols[i+1].text
            
            return data
        except Exception as e:
            logging.error(f"Error getting FinViz data for {symbol}: {e}")
            return {}

    def analyze_stock(self, symbol: str) -> Dict:
        """Analyze a single stock using free data sources"""
        try:
            # Get stock data from yfinance
            stock = yf.Ticker(symbol)
            hist = stock.history(period='6mo')
            
            if len(hist) == 0:
                return None
                
            # Calculate basic metrics
            current_price = hist['Close'][-1]
            avg_volume = hist['Volume'].mean()
            current_volume = hist['Volume'][-1]
            
            # Calculate technical indicators
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            
            # Get FinViz data
            finviz_data = self.get_finviz_data(symbol)
            
            return {
                'symbol': symbol,
                'price': current_price,
                'volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': current_volume / avg_volume,
                'above_200_sma': current_price > sma_200,
                'above_50_sma': current_price > sma_50,
                'finviz_data': finviz_data
            }
            
        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {e}")
            return None

    def find_opportunities(self) -> List[Dict]:
        """Find and analyze potential trading opportunities"""
        opportunities = []
        
        try:
            # 1. Get trending stocks from Reddit
            trending = self.get_reddit_trending()
            
            # 2. Analyze each trending stock
            for symbol, data in trending.items():
                if data['mentions'] >= 3:  # Minimum mentions threshold
                    analysis = self.analyze_stock(symbol)
                    
                    if analysis:
                        score = self.calculate_opportunity_score(analysis, data)
                        
                        if score > 0.7:  # Score threshold
                            opportunities.append({
                                'symbol': symbol,
                                'score': score,
                                'analysis': analysis,
                                'social_data': data
                            })
            
            # Sort by score
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            return opportunities[:10]  # Top 10 opportunities
            
        except Exception as e:
            logging.error(f"Error finding opportunities: {e}")
            return []

    def calculate_opportunity_score(self, analysis: Dict, social_data: Dict) -> float:
        """Calculate an opportunity score based on various factors"""
        score = 0.0
        
        try:
            # Technical factors (40%)
            if analysis['above_200_sma']:
                score += 0.2
            if analysis['above_50_sma']:
                score += 0.2
                
            # Volume factors (30%)
            volume_score = min(analysis['volume_ratio'] / 5, 1.0) * 0.3
            score += volume_score
            
            # Social factors (30%)
            sentiment_score = (social_data['sentiment'] + 1) / 2  # Normalize to 0-1
            mention_score = min(social_data['mentions'] / 10, 1.0)
            social_score = (sentiment_score + mention_score) / 2 * 0.3
            score += social_score
            
            return score
            
        except Exception as e:
            logging.error(f"Error calculating score: {e}")
            return 0.0

    def is_valid_stock(self, symbol: str) -> bool:
        """Check if a symbol is a valid stock"""
        try:
            # Basic symbol validation
            if not re.match(r'^[A-Z]{1,5}$', symbol):
                return False
                
            # Check if it exists on Yahoo Finance
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return 'regularMarketPrice' in info
            
        except Exception as e:
            return False