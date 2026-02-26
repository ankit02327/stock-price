"""
US Stocks Current Price Fetcher

Fetches current live stock prices for US stocks using Finnhub API.
Rate-limited to 60 calls/minute for free tier.
"""

import os
import sys
import requests
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from shared.utilities import (
    Config, Constants, standardize_csv_columns, 
    ensure_alphabetical_order, get_currency_for_category
)

class USCurrentFetcher:
    """
    Fetches current live US stock prices using Finnhub API.
    Rate-limited to respect Finnhub free tier limits.
    """
    
    def __init__(self):
        self.config = Config()
        self.data_dir = self.config.data_dir
        self.latest_dir = os.path.join(self.data_dir, 'latest', 'us_stocks')
        self.latest_prices_file = os.path.join(self.latest_dir, 'latest_prices.csv')
        
        # Ensure directories exist
        os.makedirs(self.latest_dir, exist_ok=True)
        
        # API configuration
        self.finnhub_api_key = self.config.finnhub_api_key
        self.currency = get_currency_for_category('us_stocks')
        
        # Rate limiting (Finnhub free tier: 60 calls/minute)
        self.rate_limit_delay = 1.0  # 1 second between calls = 60 calls/minute
        self.max_retries = 3
        
        # Cache configuration
        self.cache_duration = 60  # 60 seconds cache
        self.cache = {}  # {symbol: {data: dict, timestamp: datetime}}
        self.last_request_time = 0
        
        if not self.finnhub_api_key:
            print("WARNING: Finnhub API key not configured. Set FINNHUB_API_KEY environment variable.")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            print(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data for symbol is still valid"""
        if symbol not in self.cache:
            return False
        
        cache_entry = self.cache[symbol]
        cache_time = cache_entry['timestamp']
        
        # Check if cache is still within duration
        return datetime.now() - cache_time < timedelta(seconds=self.cache_duration)
    
    def fetch_price_from_finnhub(self, symbol: str) -> Tuple[float, str]:
        """
        Fetch current stock price from Finnhub API.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Tuple of (price, company_name)
        """
        if not self.finnhub_api_key:
            raise ValueError("Finnhub API key not configured")
        
        for attempt in range(self.max_retries):
            try:
                # Enforce rate limiting
                self._enforce_rate_limit()
                
                # Fetch quote
                quote_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.finnhub_api_key}"
                quote_response = requests.get(quote_url, timeout=10)
                quote_response.raise_for_status()
                
                quote_data = quote_response.json()
                
                if 'c' not in quote_data or quote_data['c'] is None:
                    raise ValueError(f"No price data available for {symbol}")
                
                price = float(quote_data['c'])
                
                # Fetch company profile for name
                profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={self.finnhub_api_key}"
                profile_response = requests.get(profile_url, timeout=10)
                
                company_name = symbol
                if profile_response.status_code == 200:
                    profile_data = profile_response.json()
                    company_name = profile_data.get('name', symbol)
                
                print(f"Successfully fetched {symbol}: ${price} ({company_name})")
                return price, company_name
                
            except Exception as e:
                print(f"Finnhub attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2.0)  # Wait before retry
                else:
                    print(f"All Finnhub attempts failed for {symbol}")
                    raise
    
    def fetch_current_price(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current stock price with caching.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dict with price data
        """
        if not symbol or not symbol.strip():
            raise ValueError("Stock symbol is required")
        
        symbol = symbol.strip().upper()
        
        # Check cache first
        if self._is_cache_valid(symbol):
            cached_data = self.cache[symbol]['data']
            print(f"Returning cached data for {symbol} (age: {(datetime.now() - self.cache[symbol]['timestamp']).seconds}s)")
            return cached_data
        
        timestamp = datetime.now().isoformat()
        
        try:
            price, company_name = self.fetch_price_from_finnhub(symbol)
            
            result = {
                'symbol': symbol,
                'price': price,
                'timestamp': timestamp,
                'source': 'finnhub',
                'company_name': company_name,
                'currency': self.currency
            }
            
            # Cache the result
            self.cache[symbol] = {
                'data': result,
                'timestamp': datetime.now()
            }
            
            # Save OHLCV data to individual file
            try:
                self.save_daily_data(symbol, result)
                print(f"✓ Saved OHLCV data for {symbol} to individual file")
            except Exception as e:
                print(f"Warning: Could not save daily data for {symbol}: {e}")
            
            print(f"Successfully fetched {symbol} price ${price} from Finnhub")
            return result
            
        except Exception as e:
            error_msg = f"Unable to fetch price for {symbol}: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def save_to_csv(self, data: Dict[str, Any]):
        """Save stock data to latest_prices.csv file"""
        try:
            # Create DataFrame with the new data
            new_row = pd.DataFrame([{
                'symbol': data['symbol'],
                'price': data['price'],
                'timestamp': data['timestamp'],
                'source': data['source'],
                'company_name': data['company_name'],
                'currency': data.get('currency', self.currency)
            }])
            
            # Read existing data if file exists
            if os.path.exists(self.latest_prices_file):
                existing_df = pd.read_csv(self.latest_prices_file)
                # Remove any existing entry for this symbol
                existing_df = existing_df[existing_df['symbol'] != data['symbol']]
                # Append new data
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            else:
                updated_df = new_row
            
            # Save updated data
            updated_df.to_csv(self.latest_prices_file, index=False)
            
            # Update dynamic index
            self.update_dynamic_index()
            
            print(f"Saved {data['symbol']} data to {self.latest_prices_file}")
            
        except Exception as e:
            print(f"Error saving to CSV for {data['symbol']}: {str(e)}")
            # Don't raise exception to avoid breaking the main flow
    
    def update_dynamic_index(self):
        """Update the dynamic index CSV file using DynamicIndexManager"""
        try:
            from shared.index_manager import DynamicIndexManager
            index_manager = DynamicIndexManager(self.data_dir)
            
            if os.path.exists(self.latest_prices_file):
                df = pd.read_csv(self.latest_prices_file)
                symbols = df['symbol'].unique().tolist()
                
                # Add each symbol to dynamic index (preserves existing stocks)
                for symbol in symbols:
                    if not index_manager.stock_exists(symbol, 'us_stocks'):
                        # Prepare basic stock info
                        stock_info = {
                            'company_name': symbol,
                            'sector': 'Unknown',
                            'market_cap': '',
                            'headquarters': 'Unknown',
                            'exchange': 'Unknown'
                        }
                        index_manager.add_stock(symbol, stock_info, 'us_stocks')
                
                print(f"Updated dynamic index with {len(symbols)} symbols")
            else:
                print("No latest prices file found, skipping dynamic index update")
                
        except Exception as e:
            print(f"Error updating dynamic index: {str(e)}")
    
    def get_latest_prices(self) -> pd.DataFrame:
        """Get all latest prices from CSV file"""
        try:
            if os.path.exists(self.latest_prices_file):
                return pd.read_csv(self.latest_prices_file)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading latest prices: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch current prices for multiple symbols.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dict with results and statistics
        """
        print(f"Fetching current prices for {len(symbols)} symbols...")
        
        results = {
            'successful': [],
            'failed': [],
            'cached': [],
            'errors': []
        }
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
            
            try:
                # Check cache first
                if self._is_cache_valid(symbol):
                    cached_data = self.cache[symbol]['data']
                    results['cached'].append(cached_data)
                    print(f"{symbol}: Using cached data")
                    continue
                
                # Fetch fresh data
                data = self.fetch_current_price(symbol)
                results['successful'].append(data)
                
            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                results['failed'].append(symbol)
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
            
            # Rate limiting between symbols
            if i < len(symbols):
                time.sleep(self.rate_limit_delay)
        
        # Print summary
        print(f"\n" + "=" * 50)
        print("FETCH SUMMARY")
        print("=" * 50)
        print(f"Total symbols: {len(symbols)}")
        print(f"Successful: {len(results['successful'])}")
        print(f"Cached: {len(results['cached'])}")
        print(f"Failed: {len(results['failed'])}")
        
        if results['errors']:
            print(f"\nErrors:")
            for error in results['errors']:
                print(f"  ❌ {error}")
        
        return results
    
    def save_daily_data(self, symbol: str, ohlcv_data: Dict[str, Any]):
        """
        Append today's OHLCV data to individual stock file in latest directory.
        Only saves if data for today doesn't already exist.
        """
        try:
            from datetime import datetime
            
            # Path to individual file in latest directory
            individual_file = os.path.join(
                self.latest_dir, 'individual_files', f'{symbol}.csv'
            )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(individual_file), exist_ok=True)
            
            # Today's date
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Check if file exists and read it
            if os.path.exists(individual_file):
                df = pd.read_csv(individual_file)
                # Check if today's data already exists
                if 'date' in df.columns and today in df['date'].values:
                    print(f"Today's data already exists for {symbol}, skipping")
                    return
            else:
                # Create new DataFrame with proper columns
                df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close', 'currency'])
            
            # Add new row for today
            new_row = {
                'date': today,
                'open': ohlcv_data.get('open', ohlcv_data.get('price', 0)),
                'high': ohlcv_data.get('high', ohlcv_data.get('price', 0)),
                'low': ohlcv_data.get('low', ohlcv_data.get('price', 0)),
                'close': ohlcv_data.get('close', ohlcv_data.get('price', 0)),
                'volume': ohlcv_data.get('volume', 0),
                'adjusted_close': ohlcv_data.get('adjusted_close', ''),
                'currency': self.currency
            }
            
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Sort by date and save
            df = df.sort_values('date').reset_index(drop=True)
            df.to_csv(individual_file, index=False)
            
            print(f"Saved today's data for {symbol} to {individual_file}")
            
            # Update dynamic index
            try:
                self.update_dynamic_index()
                print(f"✓ Updated dynamic index for {symbol}")
            except Exception as e:
                print(f"Warning: Could not update dynamic index for {symbol}: {e}")
            
        except Exception as e:
            print(f"Error saving daily data for {symbol}: {str(e)}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch current US stock prices using Finnhub")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all symbols from index")
    
    args = parser.parse_args()
    
    fetcher = USCurrentFetcher()
    
    if args.all:
        # Load symbols from index
        past_index = os.path.join(fetcher.data_dir, 'past', 'us_stocks', 'index_us_stocks.csv')
        if os.path.exists(past_index):
            df = pd.read_csv(past_index)
            symbols = df['symbol'].tolist()
            results = fetcher.fetch_multiple_prices(symbols)
        else:
            print("Index file not found. Use --symbols to specify symbols.")
    elif args.symbols:
        results = fetcher.fetch_multiple_prices(args.symbols)
    else:
        print("Please specify --symbols or --all")
        return
    
    # Show latest prices
    latest_prices = fetcher.get_latest_prices()
    if not latest_prices.empty:
        print(f"\nLatest prices ({len(latest_prices)} symbols):")
        for _, row in latest_prices.head(10).iterrows():
            print(f"  {row['symbol']}: ${row['price']:.2f} ({row['source']})")
        if len(latest_prices) > 10:
            print(f"  ... and {len(latest_prices) - 10} more")

if __name__ == "__main__":
    main()
