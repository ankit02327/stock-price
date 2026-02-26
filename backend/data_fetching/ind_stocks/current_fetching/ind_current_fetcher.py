"""
Indian Stocks Current Price Fetcher

Fetches current live stock prices for Indian stocks using real-time data sources.
Implements fallback chain: Upstox API (real-time) → permanent directory (historical)

Rate-limited to respect API limits and ensure reliable data fetching.
All prices are returned in INR currency.
Note: yfinance removed from real-time fetching as it only provides delayed data.
"""

import os
import sys
import requests
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from shared.utilities import (
    Config, Constants, standardize_csv_columns, 
    ensure_alphabetical_order, get_currency_for_category,
    get_live_exchange_rate, convert_usd_to_inr, get_current_timestamp
)
from shared.upstox_token_manager import UpstoxTokenManager
try:
    from .upstox_instruments import get_instruments_fetcher
except ImportError:
    # Fallback for when running as script
    from upstox_instruments import get_instruments_fetcher

class IndianCurrentFetcher:
    """
    Fetches current live Indian stock prices using multiple data sources.
    Implements fallback chain for maximum reliability.
    All prices are returned in INR currency.
    """
    
    def __init__(self):
        self.config = Config()
        self.data_dir = self.config.data_dir
        self.latest_dir = os.path.join(self.data_dir, 'latest', 'ind_stocks')
        
        # Ensure directories exist
        os.makedirs(self.latest_dir, exist_ok=True)
        
        # API configurations
        self.upstox_api_key = self.config.upstox_api_key
        self.currency = get_currency_for_category('ind_stocks')  # INR
        
        # Initialize Upstox token manager
        self.token_manager = UpstoxTokenManager()
        
        # Rate limiting
        self.rate_limit_delay = 2.0  # 2 seconds between calls to avoid rate limits
        self.max_retries = 3
        
        # Cache configuration
        self.cache_duration = 60  # 60 seconds cache
        self.cache = {}  # {symbol: {data: dict, timestamp: datetime}}
        self.last_request_time = 0
        
        
        # Initialize Upstox instruments fetcher for ISIN lookups
        self.instruments_fetcher = get_instruments_fetcher()
        
        # Common Indian stock symbol to instrument key mapping
        self.COMMON_STOCK_MAPPINGS = {
            'RELIANCE': 'NSE_EQ|INE002A01018',
            'TCS': 'NSE_EQ|INE467B01029',
            'HDFCBANK': 'NSE_EQ|INE040A01034',
            'INFY': 'NSE_EQ|INE009A01021',
            'HINDUNILVR': 'NSE_EQ|INE030A01027',
            'ICICIBANK': 'NSE_EQ|INE090A01021',
            'SBIN': 'NSE_EQ|INE062A01020',
            'BHARTIARTL': 'NSE_EQ|INE397D01024',
            'ITC': 'NSE_EQ|INE154A01025',
            'KOTAKBANK': 'NSE_EQ|INE237A01028',
            'LT': 'NSE_EQ|INE018A01030',
            'AXISBANK': 'NSE_EQ|INE238A01034',
            'WIPRO': 'NSE_EQ|INE075A01022',
            'MARUTI': 'NSE_EQ|INE585B01010',
            'TATAMOTORS': 'NSE_EQ|INE155A01022',
            'TATASTEEL': 'NSE_EQ|INE081A01020',
            'HCLTECH': 'NSE_EQ|INE860A01027',
            'ASIANPAINT': 'NSE_EQ|INE021A01026',
            'BAJFINANCE': 'NSE_EQ|INE296A01024',
            'BAJAJ-AUTO': 'NSE_EQ|INE917I01010',
            'TITAN': 'NSE_EQ|INE280A01028',
            'TITANCO': 'NSE_EQ|INE280A01028',
            'ADANIPORTS': 'NSE_EQ|INE742F01042',
            'NESTLEIND': 'NSE_EQ|INE239A01016',
            'POWERGRID': 'NSE_EQ|INE752E01010',
            'NTPC': 'NSE_EQ|INE733E01010',
            'COALINDIA': 'NSE_EQ|INE522F01014',
            'ONGC': 'NSE_EQ|INE213A01029',
            'IOC': 'NSE_EQ|INE242A01010',
            'BPCL': 'NSE_EQ|INE029A01011',
            'HINDALCO': 'NSE_EQ|INE038A01020',
            'JSWSTEEL': 'NSE_EQ|INE019A01038',
            'ULTRACEMCO': 'NSE_EQ|INE481G01011',
            'GRASIM': 'NSE_EQ|INE047A01013',
            'DRREDDY': 'NSE_EQ|INE089A01023',
            'CIPLA': 'NSE_EQ|INE059A01026',
            'SUNPHARMA': 'NSE_EQ|INE044A01036',
            'DIVISLAB': 'NSE_EQ|INE361B01018'
        }
    
    
    def get_instrument_key(self, symbol: str) -> Optional[str]:
        """
        Convert stock symbol to Upstox instrument key format.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'BAJAJ-AUTO')
            
        Returns:
            Instrument key in format 'NSE_EQ|ISIN' or None if not found
        """
        symbol_upper = symbol.strip().upper()
        
        # 1. Try hardcoded mappings first (fastest)
        if symbol_upper in self.COMMON_STOCK_MAPPINGS:
            return self.COMMON_STOCK_MAPPINGS[symbol_upper]
        
        # 2. Try dynamic index (fast, offline)
        try:
            from shared.index_manager import DynamicIndexManager
            index_manager = DynamicIndexManager(self.data_dir)
            isin = index_manager.get_isin(symbol_upper, 'ind_stocks')
            if isin:
                print(f"✓ Found ISIN in dynamic index for {symbol_upper}: {isin}")
                return f"NSE_EQ|{isin}"
        except Exception as e:
            print(f"Warning: Could not lookup ISIN from dynamic index for {symbol_upper}: {e}")
        
        # 3. Try instruments file lookup for dynamic ISIN mapping (slower, online)
        try:
            instrument_key = self.instruments_fetcher.get_instrument_key(symbol_upper)
            if instrument_key:
                print(f"✓ Found ISIN mapping from instruments file for {symbol_upper}: {instrument_key}")
                
                # Save ISIN to dynamic index for future use
                isin = instrument_key.split('|')[1]
                try:
                    from shared.index_manager import DynamicIndexManager
                    index_manager = DynamicIndexManager(self.data_dir)
                    if index_manager.stock_exists(symbol_upper, 'ind_stocks'):
                        index_manager.update_stock_isin(symbol_upper, isin, 'ind_stocks')
                        print(f"✓ Saved ISIN to dynamic index for {symbol_upper}")
                except Exception as e:
                    print(f"Warning: Could not save ISIN to dynamic index: {e}")
                
                return instrument_key
        except Exception as e:
            print(f"Warning: Could not lookup ISIN from instruments file for {symbol_upper}: {e}")
        
        # 4. No ISIN found - return None to skip Upstox
        print(f"⚠ No ISIN mapping found for {symbol_upper}, will skip Upstox")
        return None
    
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
    
    def fetch_price_from_upstox(self, symbol: str) -> Tuple[float, str]:
        """
        Fetch current stock price + today's OHLCV from Upstox API.
        Uses ONLY the market quote LTP endpoint - no historical data.
        Automatically handles token refresh on 401 errors.
        """
        # Get valid token (will refresh if needed)
        access_token = self.token_manager.get_valid_token()
        if not access_token:
            # Log token status for debugging
            token_info = self.token_manager.get_token_info()
            print(f"⚠ Upstox token status: {token_info}")
            raise ValueError("Upstox access token not available. Run OAuth setup script.")
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Get instrument key for symbol
                instrument_key = self.get_instrument_key(symbol)
                if not instrument_key:
                    raise ValueError(f"No ISIN mapping found for {symbol}. Skipping Upstox.")
                
                # Upstox Market Quote API v2 endpoint
                url = "https://api.upstox.com/v2/market-quote/ltp"
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {access_token}'
                }
                
                params = {'symbol': instrument_key}
                
                print(f"Upstox API call for {symbol}: {instrument_key} (attempt {attempt + 1})")
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                # Debug: Print status code and response
                print(f"Upstox response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"Upstox error response: {response.text}")
                
                # Handle 401 error with token refresh
                if response.status_code == 401:
                    if attempt == 0:  # Only try refresh on first attempt
                        print("Upstox returned 401, attempting token refresh...")
                        if self.token_manager.handle_401_error():
                            # Get new token and retry
                            access_token = self.token_manager.get_valid_token()
                            if access_token:
                                continue  # Retry with new token
                            else:
                                raise ValueError("Token refresh failed")
                        else:
                            raise ValueError("Upstox authentication failed. Token refresh unsuccessful.")
                    else:
                        raise ValueError("Upstox authentication failed after token refresh attempt.")
                
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'success' and 'data' in data:
                    # The API returns data with the same instrument_key we sent
                    data_key = instrument_key
                    if data_key in data['data']:
                        price_data = data['data'][data_key]
                        price = float(price_data['last_price'])
                        company_name = symbol  # Use symbol as fallback
                        
                        print(f"✓ Upstox: {symbol} = ₹{price}")
                        return price, company_name
                    else:
                        # Try alternative key format (NSE_EQ:SYMBOL)
                        alt_key = f"NSE_EQ:{symbol}"
                        if alt_key in data['data']:
                            price_data = data['data'][alt_key]
                            price = float(price_data['last_price'])
                            company_name = symbol  # Use symbol as fallback
                            
                            print(f"✓ Upstox: {symbol} = ₹{price} (using alt key)")
                            return price, company_name
                        else:
                            # Debug: print available keys
                            available_keys = list(data['data'].keys())
                            print(f"Available keys in response: {available_keys}")
                            raise ValueError(f"No price data for {symbol} in Upstox response. Available keys: {available_keys}")
                else:
                    error_msg = data.get('message', 'Unknown error')
                    raise ValueError(f"Upstox API error: {error_msg}")
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    # This should be handled above, but just in case
                    if attempt == 0:
                        print("Upstox returned 401, attempting token refresh...")
                        if self.token_manager.handle_401_error():
                            access_token = self.token_manager.get_valid_token()
                            if access_token:
                                continue
                        raise ValueError("Upstox authentication failed. Token refresh unsuccessful.")
                    else:
                        raise ValueError("Upstox authentication failed after token refresh attempt.")
                elif e.response.status_code == 403:
                    raise ValueError("Upstox access forbidden. Ensure API is enabled in Upstox dashboard")
                else:
                    raise ValueError(f"Upstox HTTP error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                print(f"Upstox error for {symbol}: {str(e)}")
                raise
        
        # If we get here, all retries failed
        raise ValueError(f"Upstox API failed after {max_retries} attempts")
    
    def fetch_ohlcv_from_upstox(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch today's OHLCV (Open, High, Low, Close, Volume) from Upstox.
        Uses the full market quote endpoint.
        Automatically handles token refresh on 401 errors.
        """
        # Get valid token (will refresh if needed)
        access_token = self.token_manager.get_valid_token()
        if not access_token:
            raise ValueError("Upstox access token not available. Run OAuth setup script.")
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Get instrument key for symbol
                instrument_key = self.get_instrument_key(symbol)
                if not instrument_key:
                    raise ValueError(f"No ISIN mapping found for {symbol}. Skipping Upstox.")
                
                # Upstox Full Market Quote API endpoint
                url = "https://api.upstox.com/v2/market-quote/quotes"
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {access_token}'
                }
                
                params = {'symbol': instrument_key}
                
                print(f"Upstox OHLCV API call for {symbol}: {instrument_key} (attempt {attempt + 1})")
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                # Handle 401 error with token refresh
                if response.status_code == 401:
                    if attempt == 0:  # Only try refresh on first attempt
                        print("Upstox OHLCV returned 401, attempting token refresh...")
                        if self.token_manager.handle_401_error():
                            # Get new token and retry
                            access_token = self.token_manager.get_valid_token()
                            if access_token:
                                continue  # Retry with new token
                            else:
                                raise ValueError("Token refresh failed")
                        else:
                            raise ValueError("Upstox authentication failed. Token refresh unsuccessful.")
                    else:
                        raise ValueError("Upstox authentication failed after token refresh attempt.")
                
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'success' and 'data' in data:
                    # The API returns data with the same instrument_key we sent
                    data_key = instrument_key
                    if data_key in data['data']:
                        quote_data = data['data'][data_key]
                        ohlc = quote_data.get('ohlc', {})
                        
                        return {
                            'open': float(ohlc.get('open', 0)),
                            'high': float(ohlc.get('high', 0)),
                            'low': float(ohlc.get('low', 0)),
                            'close': float(ohlc.get('close', 0)),
                            'last_price': float(quote_data.get('last_price', 0)),
                            'volume': int(quote_data.get('volume', 0))
                        }
                    else:
                        # Try alternative key format (NSE_EQ:SYMBOL)
                        alt_key = f"NSE_EQ:{symbol}"
                        if alt_key in data['data']:
                            quote_data = data['data'][alt_key]
                            ohlc = quote_data.get('ohlc', {})
                            
                            return {
                                'open': float(ohlc.get('open', 0)),
                                'high': float(ohlc.get('high', 0)),
                                'low': float(ohlc.get('low', 0)),
                                'close': float(ohlc.get('close', 0)),
                                'last_price': float(quote_data.get('last_price', 0)),
                                'volume': int(quote_data.get('volume', 0))
                            }
                
                raise ValueError(f"No OHLCV data for {symbol}")
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    # This should be handled above, but just in case
                    if attempt == 0:
                        print("Upstox OHLCV returned 401, attempting token refresh...")
                        if self.token_manager.handle_401_error():
                            access_token = self.token_manager.get_valid_token()
                            if access_token:
                                continue
                        raise ValueError("Upstox authentication failed. Token refresh unsuccessful.")
                    else:
                        raise ValueError("Upstox authentication failed after token refresh attempt.")
                elif e.response.status_code == 403:
                    raise ValueError("Upstox access forbidden. Ensure API is enabled in Upstox dashboard")
                else:
                    raise ValueError(f"Upstox HTTP error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                print(f"Upstox OHLCV error for {symbol}: {str(e)}")
                raise
        
        # If we get here, all retries failed
        raise ValueError(f"Upstox OHLCV API failed after {max_retries} attempts")
    
    def fetch_batch_prices_upstox(self, symbols: List[str]) -> Dict[str, Tuple[float, str]]:
        """
        Fetch multiple stock prices in a single Upstox API call.
        Upstox supports up to 500 symbols per request.
        
        Args:
            symbols: List of stock symbols (max 500)
        
        Returns:
            Dict mapping symbol to (price, company_name) tuple
        """
        if not self.upstox_access_token:
            raise ValueError("Upstox access token not configured")
        
        if len(symbols) > 500:
            raise ValueError("Upstox batch API supports max 500 symbols")
        
        try:
            url = "https://api.upstox.com/v2/market-quote/ltp"
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.upstox_access_token}'
            }
            
            # Convert symbols to instrument keys and create comma-separated string
            instrument_keys = [self.get_instrument_key(s) for s in symbols]
            params = {'symbol': ','.join(instrument_keys)}
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = {}
            
            if data.get('status') == 'success' and 'data' in data:
                for symbol, instrument_key in zip(symbols, instrument_keys):
                    if instrument_key in data['data']:
                        price_data = data['data'][instrument_key]
                        price = float(price_data['last_price'])
                        results[symbol] = (price, symbol)
            
            return results
            
        except Exception as e:
            print(f"Upstox batch fetch error: {str(e)}")
            raise
    
    def _get_stock_metadata(self, symbol: str) -> Dict[str, str]:
        """Get stock metadata from master dynamic index"""
        try:
            from shared.index_manager import DynamicIndexManager
            
            index_manager = DynamicIndexManager(self.data_dir)
            stock_info = index_manager.get_stock_info(symbol, 'ind_stocks')
            
            if stock_info:
                return {
                    'sector': self._clean_metadata_value(str(stock_info.get('sector', 'N/A'))),
                    'market_cap': self._clean_metadata_value(str(stock_info.get('market_cap', 'N/A'))),
                    'headquarters': self._clean_metadata_value(str(stock_info.get('headquarters', 'N/A'))),
                    'exchange': self._clean_metadata_value(str(stock_info.get('exchange', 'N/A')))
                }
            else:
                # If not in index, try permanent directory as fallback
                permanent_index = os.path.join(
                    self.config.permanent_dir, 'ind_stocks', 'index_ind_stocks.csv'
                )
                if os.path.exists(permanent_index):
                    df = pd.read_csv(permanent_index)
                    row = df[df['symbol'] == symbol]
                    if not row.empty:
                        return {
                            'sector': self._clean_metadata_value(str(row.iloc[0].get('sector', 'N/A'))),
                            'market_cap': self._clean_metadata_value(str(row.iloc[0].get('market_cap', 'N/A'))),
                            'headquarters': self._clean_metadata_value(str(row.iloc[0].get('headquarters', 'N/A'))),
                            'exchange': self._clean_metadata_value(str(row.iloc[0].get('exchange', 'N/A')))
                        }
                
                return {'sector': 'N/A', 'market_cap': 'N/A', 'headquarters': 'N/A', 'exchange': 'N/A'}
                
        except Exception as e:
            print(f"Error fetching metadata for {symbol}: {str(e)}")
            return {'sector': 'N/A', 'market_cap': 'N/A', 'headquarters': 'N/A', 'exchange': 'N/A'}
    
    def _clean_metadata_value(self, value: str) -> str:
        """
        Clean metadata value by handling empty strings, NaN, and whitespace.
        
        Args:
            value: Raw value from CSV
            
        Returns:
            Cleaned value or 'N/A' if empty/invalid
        """
        if not value or value.strip() == '' or value.lower() in ['nan', 'none', 'null']:
            return 'N/A'
        
        # Handle pandas NaN values
        try:
            import pandas as pd
            if pd.isna(value):
                return 'N/A'
        except:
            pass
            
        return value.strip()
    
    def fetch_from_permanent_directory(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch stock data from permanent directory as last resort.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with stock data or None if not found
        """
        try:
            # Use absolute path from config (fixes relative path issues)
            permanent_file = self.config.get_permanent_path(
                'ind_stocks', 'individual_files', f'{symbol}.csv'
            )
            
            if os.path.exists(permanent_file):
                df = pd.read_csv(permanent_file)
                
                if not df.empty:
                    # Get the latest row (most recent data)
                    latest_row = df.iloc[-1]
                    
                    # Try different column name formats (Close or close)
                    close_price = latest_row.get('Close', latest_row.get('close', latest_row.get('adjusted_close')))
                    
                    # Get the date of last data point
                    last_date = latest_row.get('date', latest_row.get('Date', 'unknown'))
                    
                    # Get additional metadata from index files
                    metadata = self._get_stock_metadata(symbol)
                    
                    return {
                        'symbol': symbol,
                        'price': float(close_price),
                        'company_name': symbol,
                        'currency': self.currency,
                        'source': 'permanent',
                        'source_reliable': False,  # Historical data, not real-time
                        'data_date': str(last_date),  # Date of last data in permanent (READ-ONLY fallback)
                        'timestamp': get_current_timestamp(),
                        'sector': metadata['sector'],
                        'market_cap': metadata['market_cap'],
                        'headquarters': metadata['headquarters'],
                        'exchange': metadata['exchange'],
                        'open': float(latest_row['open']) if 'open' in latest_row and pd.notna(latest_row['open']) else None,
                        'high': float(latest_row['high']) if 'high' in latest_row and pd.notna(latest_row['high']) else None,
                        'low': float(latest_row['low']) if 'low' in latest_row and pd.notna(latest_row['low']) else None,
                        'volume': int(latest_row['volume']) if 'volume' in latest_row and pd.notna(latest_row['volume']) else None,
                        'close': float(latest_row['close']) if 'close' in latest_row and pd.notna(latest_row['close']) else None
                    }
        except Exception as e:
            print(f"Error reading permanent data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    
    
    def fetch_current_price(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current stock price with fallback strategy and caching.
        
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
        
        # Try each API in order - Upstox as primary (only real-time API)
        apis = [
            ('upstox', self.fetch_price_from_upstox)
        ]
        
        last_error = None
        
        for api_name, api_func in apis:
            try:
                print(f"Trying {api_name} for symbol {symbol}")
                self._enforce_rate_limit()
                
                price, company_name = api_func(symbol)
                
                # Get additional metadata from index files
                metadata = self._get_stock_metadata(symbol)
                
                # Get additional data from latest CSV files
                additional_data = self._get_latest_day_data(symbol)
                
                # If using Upstox, also fetch and save today's OHLCV data
                ohlcv_data = {}
                if api_name == 'upstox':
                    try:
                        ohlcv_data = self.fetch_ohlcv_from_upstox(symbol)
                        self.save_daily_data(symbol, ohlcv_data)
                        print(f"✓ Fetched and saved OHLCV data for {symbol}")
                    except Exception as e:
                        print(f"Could not fetch OHLCV data for {symbol}: {e}")
                        # Continue with just the price
                
                result = {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': timestamp,
                    'source': api_name,
                    'source_reliable': api_name == 'upstox',  # Indicate if real-time
                    'data_date': None,  # None for live API data (not from permanent fallback)
                    'company_name': company_name,
                    'currency': self.currency,
                    'sector': metadata['sector'],
                    'market_cap': metadata['market_cap'],
                    'headquarters': metadata['headquarters'],
                    'exchange': metadata['exchange'],
                    'open': ohlcv_data.get('open') or additional_data.get('open'),
                    'high': ohlcv_data.get('high') or additional_data.get('high'),
                    'low': ohlcv_data.get('low') or additional_data.get('low'),
                    'volume': ohlcv_data.get('volume') or additional_data.get('volume'),
                    'close': ohlcv_data.get('close') or additional_data.get('close')
                }
                
                # Cache the result
                self.cache[symbol] = {
                    'data': result,
                    'timestamp': datetime.now()
                }
                
                # Add to dynamic index if not already present
                try:
                    from shared.index_manager import DynamicIndexManager
                    index_manager = DynamicIndexManager(self.data_dir)
                    
                    if not index_manager.stock_exists(symbol, 'ind_stocks'):
                        # Prepare stock info for dynamic index
                        stock_info = {
                            'company_name': result.get('company_name', symbol),
                            'sector': result.get('sector', 'N/A'),
                            'market_cap': result.get('market_cap', ''),
                            'headquarters': result.get('headquarters', 'N/A'),
                            'exchange': result.get('exchange', 'NSE')
                        }
                        
                        index_manager.add_stock(symbol, stock_info, 'ind_stocks')
                        print(f"✓ Added {symbol} to dynamic index")
                    else:
                        print(f"✓ {symbol} already exists in dynamic index")
                        
                except Exception as e:
                    print(f"Warning: Could not update dynamic index for {symbol}: {e}")
                
                # Save OHLCV data to individual file
                try:
                    self.save_daily_data(symbol, result)
                    print(f"✓ Saved OHLCV data for {symbol} to individual file")
                except Exception as e:
                    print(f"Warning: Could not save daily data for {symbol}: {e}")
                
                print(f"Successfully fetched {symbol} price ₹{price} from {api_name}")
                return result
                
            except Exception as e:
                last_error = e
                print(f"{api_name} failed for {symbol}: {str(e)}")
                continue
        
        # All APIs failed, try to get data from permanent directory as last resort
        print(f"All APIs failed for {symbol}, trying permanent directory fallback...")
        permanent_file = None
        try:
            permanent_data = self.fetch_from_permanent_directory(symbol)
            if permanent_data:
                print(f"✅ Found {symbol} in permanent directory")
                return permanent_data
        except Exception as e:
            print(f"Permanent directory fallback failed: {e}")
            permanent_file = self.config.get_permanent_path('ind_stocks', 'individual_files', f'{symbol}.csv')
        
        # If everything fails, provide detailed diagnostic info
        if permanent_file is None:
            permanent_file = self.config.get_permanent_path('ind_stocks', 'individual_files', f'{symbol}.csv')
        
        error_details = {
            'upstox_error': str(last_error) if last_error else 'Not attempted',
            'permanent_path_checked': permanent_file,
            'permanent_exists': os.path.exists(permanent_file)
        }
        error_msg = f"Unable to fetch price for {symbol}. All sources failed. Details: {error_details}"
        print(error_msg)
        raise Exception(error_msg)
    
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
                df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'currency'])
            
            # Add new row for today
            new_row = {
                'date': today,
                'open': ohlcv_data.get('open', 0),
                'high': ohlcv_data.get('high', 0),
                'low': ohlcv_data.get('low', 0),
                'close': ohlcv_data.get('close', ohlcv_data.get('last_price', 0)),
                'volume': ohlcv_data.get('volume', 0),
                'currency': 'INR'  # Add currency field to prevent NaN values
            }
            
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Sort by date and save
            df = df.sort_values('date').reset_index(drop=True)
            df.to_csv(individual_file, index=False)
            
            print(f"Saved today's data for {symbol} to {individual_file}")
            
        except Exception as e:
            print(f"Error saving daily data for {symbol}: {str(e)}")
    
    
    def _get_latest_day_data(self, symbol: str) -> Dict:
        """
        Get the latest day's open, high, low, volume data from CSV files.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with additional data fields
        """
        try:
            # Try latest directory first (2025 data)
            latest_path = os.path.join(
                self.config.get_data_path('latest', 'ind_stocks', 'individual_files', f'{symbol}.csv')
            )
            
            # Fallback to permanent directory if latest not available
            permanent_path = os.path.join(
                self.config.get_permanent_path('ind_stocks', 'individual_files', f'{symbol}.csv')
            )
            
            csv_path = None
            if os.path.exists(latest_path):
                csv_path = latest_path
            elif os.path.exists(permanent_path):
                csv_path = permanent_path
            
            if not csv_path:
                print(f"No CSV file found for {symbol}")
                return {}
            
            # Read the latest row from CSV
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                if df.empty:
                    return {}
                
                # Get the most recent row
                latest_row = df.iloc[-1]
                
                return {
                    'open': float(latest_row['open']) if 'open' in latest_row and pd.notna(latest_row['open']) else None,
                    'high': float(latest_row['high']) if 'high' in latest_row and pd.notna(latest_row['high']) else None,
                    'low': float(latest_row['low']) if 'low' in latest_row and pd.notna(latest_row['low']) else None,
                    'volume': int(latest_row['volume']) if 'volume' in latest_row and pd.notna(latest_row['volume']) else None,
                    'close': float(latest_row['close']) if 'close' in latest_row and pd.notna(latest_row['close']) else None
                }
            except ImportError:
                # Fallback to csv module
                import csv
                rows = []
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                
                if not rows:
                    return {}
                
                # Get the last row
                latest_row = rows[-1]
                
                return {
                    'open': float(latest_row.get('open', 0)) if latest_row.get('open') else None,
                    'high': float(latest_row.get('high', 0)) if latest_row.get('high') else None,
                    'low': float(latest_row.get('low', 0)) if latest_row.get('low') else None,
                    'volume': int(latest_row.get('volume', 0)) if latest_row.get('volume') else None,
                    'close': float(latest_row.get('close', 0)) if latest_row.get('close') else None
                }
            
        except Exception as e:
            print(f"Error getting latest day data for {symbol}: {e}")
            return {}

    def fetch_multiple_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch current prices for multiple symbols.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dict with results and statistics
        """
        print(f"Fetching current prices for {len(symbols)} Indian stock symbols...")
        print("Using fallback chain: Upstox → yfinance")
        
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

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch current Indian stock prices using multiple sources")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all symbols from index")
    
    args = parser.parse_args()
    
    fetcher = IndianCurrentFetcher()
    
    if args.all:
        # Load symbols from index
        past_index = os.path.join(fetcher.data_dir, 'past', 'ind_stocks', 'index_ind_stocks.csv')
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
    
    # Show latest prices from individual files
    print(f"\nLatest prices from individual files:")
    for symbol in args.symbols:
        try:
            # Get latest data from individual file
            latest_data = fetcher._get_latest_day_data(symbol)
            if latest_data and 'close' in latest_data:
                print(f"  {symbol}: ₹{latest_data['close']:.2f} (from individual file)")
        except Exception as e:
            print(f"  {symbol}: Error reading individual file - {e}")

if __name__ == "__main__":
    main()