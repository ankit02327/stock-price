"""
Upstox Instruments Fetcher

Downloads and caches the Upstox instruments file to map stock symbols to ISIN codes.
This is required because Upstox API only accepts ISIN codes, not stock symbols.

The instruments file contains all available instruments with their ISIN mappings.
We cache it locally to avoid repeated downloads and improve performance.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path

class UpstoxInstrumentsFetcher:
    """
    Fetches and caches Upstox instruments file for ISIN lookups.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the instruments fetcher.
        
        Args:
            cache_dir: Directory to store cached instruments file
        """
        if cache_dir is None:
            # Default to _cache directory in backend
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            cache_dir = os.path.join(current_dir, '_cache')
        
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, 'upstox_instruments.json')
        self.cache_duration = 24 * 60 * 60  # 24 hours in seconds
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache for symbol -> ISIN mapping
        self._symbol_to_isin_cache = {}
        self._last_refresh = None
        
    def _is_cache_valid(self) -> bool:
        """Check if the cached instruments file is still valid."""
        if not os.path.exists(self.cache_file):
            return False
        
        if self._last_refresh is None:
            # Check file modification time
            file_time = os.path.getmtime(self.cache_file)
            self._last_refresh = datetime.fromtimestamp(file_time)
        
        # Check if cache is older than cache_duration
        return datetime.now() - self._last_refresh < timedelta(seconds=self.cache_duration)
    
    def _download_instruments(self) -> Dict:
        """
        Download the complete instruments file from Upstox.
        
        Returns:
            Dictionary containing all instruments data
        """
        url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json"
        
        print("Downloading Upstox instruments file...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            instruments_data = response.json()
            print(f"✓ Downloaded instruments file with {len(instruments_data.get('data', {}))} instruments")
            
            return instruments_data
            
        except Exception as e:
            print(f"Error downloading instruments file: {e}")
            raise
    
    def _save_instruments_cache(self, instruments_data: Dict):
        """Save instruments data to cache file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(instruments_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Cached instruments file to {self.cache_file}")
        except Exception as e:
            print(f"Error saving instruments cache: {e}")
    
    def _load_instruments_cache(self) -> Optional[Dict]:
        """Load instruments data from cache file."""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                instruments_data = json.load(f)
            print(f"✓ Loaded instruments file from cache")
            return instruments_data
        except Exception as e:
            print(f"Error loading instruments cache: {e}")
            return None
    
    def _build_symbol_mapping(self, instruments_data: Dict):
        """
        Build symbol -> ISIN mapping from instruments data.
        Only includes NSE_EQ instruments.
        """
        self._symbol_to_isin_cache.clear()
        
        data = instruments_data.get('data', {})
        nse_count = 0
        
        for instrument in data.values():
            # Only process NSE equity instruments
            if (instrument.get('exchange') == 'NSE_EQ' and 
                instrument.get('instrument_type') == 'EQ'):
                
                trading_symbol = instrument.get('trading_symbol', '').upper()
                isin = instrument.get('isin', '')
                
                if trading_symbol and isin:
                    self._symbol_to_isin_cache[trading_symbol] = isin
                    nse_count += 1
        
        print(f"✓ Built symbol mapping for {nse_count} NSE stocks")
    
    def refresh_instruments(self) -> bool:
        """
        Refresh the instruments file (download if needed).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Download fresh instruments data
            instruments_data = self._download_instruments()
            
            # Save to cache
            self._save_instruments_cache(instruments_data)
            
            # Build symbol mapping
            self._build_symbol_mapping(instruments_data)
            
            # Update refresh time
            self._last_refresh = datetime.now()
            
            return True
            
        except Exception as e:
            print(f"Error refreshing instruments: {e}")
            return False
    
    def get_isin_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Get ISIN code for a given stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'BAJAJ-AUTO')
            
        Returns:
            ISIN code if found, None otherwise
        """
        symbol_upper = symbol.strip().upper()
        
        # Check if we have a cached mapping
        if self._symbol_to_isin_cache:
            return self._symbol_to_isin_cache.get(symbol_upper)
        
        # Try to load from cache if not already loaded
        if self._is_cache_valid():
            instruments_data = self._load_instruments_cache()
            if instruments_data:
                self._build_symbol_mapping(instruments_data)
                return self._symbol_to_isin_cache.get(symbol_upper)
        
        # Cache is invalid or doesn't exist, try to refresh
        if self.refresh_instruments():
            return self._symbol_to_isin_cache.get(symbol_upper)
        
        return None
    
    def get_all_nse_symbols(self) -> List[str]:
        """
        Get list of all available NSE stock symbols.
        
        Returns:
            List of stock symbols
        """
        if not self._symbol_to_isin_cache:
            # Try to load from cache or refresh
            if self._is_cache_valid():
                instruments_data = self._load_instruments_cache()
                if instruments_data:
                    self._build_symbol_mapping(instruments_data)
            else:
                self.refresh_instruments()
        
        return list(self._symbol_to_isin_cache.keys())
    
    def get_instrument_key(self, symbol: str) -> Optional[str]:
        """
        Get the complete instrument key for Upstox API.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Instrument key in format 'NSE_EQ|ISIN' or None if not found
        """
        isin = self.get_isin_for_symbol(symbol)
        if isin:
            return f"NSE_EQ|{isin}"
        return None

# Global instance for easy access
_instruments_fetcher = None

def get_instruments_fetcher() -> UpstoxInstrumentsFetcher:
    """Get the global instruments fetcher instance."""
    global _instruments_fetcher
    if _instruments_fetcher is None:
        _instruments_fetcher = UpstoxInstrumentsFetcher()
    return _instruments_fetcher

def main():
    """Test the instruments fetcher."""
    fetcher = UpstoxInstrumentsFetcher()
    
    # Test with some common stocks
    test_symbols = ['RELIANCE', 'TCS', 'BAJAJ-AUTO', 'TITAN', 'HDFCBANK']
    
    print("\nTesting ISIN lookups:")
    for symbol in test_symbols:
        isin = fetcher.get_isin_for_symbol(symbol)
        instrument_key = fetcher.get_instrument_key(symbol)
        print(f"{symbol}: {isin} -> {instrument_key}")
    
    print(f"\nTotal NSE symbols available: {len(fetcher.get_all_nse_symbols())}")

if __name__ == "__main__":
    main()
