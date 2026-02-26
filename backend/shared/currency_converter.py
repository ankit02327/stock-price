"""
Currency Converter Module

Handles live USD-INR exchange rate fetching with multiple fallback sources.
Implements caching to avoid excessive API calls.

Fallback chain:
1. forex-python (primary)
2. exchangerate-api.com (fallback #1)
3. hardcoded rate 83.5 (fallback #2)
"""

import os
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CurrencyConverter:
    """
    Currency converter with live exchange rate fetching and caching.
    """
    
    def __init__(self):
        self.cache_duration = 3600  # 1 hour cache
        self.cache = {}  # {rate: float, timestamp: datetime}
        self.hardcoded_rate = 83.5  # Fallback rate
        
    def _is_cache_valid(self) -> bool:
        """Check if cached exchange rate is still valid"""
        if 'rate' not in self.cache or 'timestamp' not in self.cache:
            return False
        
        cache_time = self.cache['timestamp']
        return datetime.now() - cache_time < timedelta(seconds=self.cache_duration)
    
    def _fetch_from_forex_python(self) -> Optional[float]:
        """Fetch USD-INR rate using forex-python library with timeout"""
        try:
            from forex_python.converter import CurrencyRates
            import threading
            import time
            
            result = [None]
            exception = [None]
            
            def fetch_rate():
                try:
                    c = CurrencyRates()
                    rate = c.get_rate('USD', 'INR')
                    result[0] = float(rate)
                except Exception as e:
                    exception[0] = e
            
            # Start the fetch in a separate thread
            thread = threading.Thread(target=fetch_rate)
            thread.daemon = True
            thread.start()
            thread.join(timeout=5)  # 5-second timeout
            
            if thread.is_alive():
                logger.warning("forex-python request timed out")
                return None
            
            if exception[0]:
                raise exception[0]
            
            if result[0]:
                logger.info(f"Fetched USD-INR rate from forex-python: {result[0]}")
                return result[0]
            
            return None
                
        except ImportError:
            logger.warning("forex-python not installed")
            return None
        except Exception as e:
            logger.warning(f"Error with forex-python: {e}")
            return None
    
    def _fetch_from_exchangerate_api(self) -> Optional[float]:
        """Fetch USD-INR rate using exchangerate-api.com (free, no API key)"""
        try:
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            response = requests.get(url, timeout=3)  # 3-second timeout
            response.raise_for_status()
            
            data = response.json()
            rate = data['rates']['INR']
            logger.info(f"Fetched USD-INR rate from exchangerate-api: {rate}")
            return float(rate)
        except Exception as e:
            logger.warning(f"Error with exchangerate-api: {e}")
            return None
    
    def _fetch_from_yahoo_finance(self) -> Optional[float]:
        """Fetch USD-INR rate using Yahoo Finance (web scraping fallback)"""
        try:
            from bs4 import BeautifulSoup
            
            url = "https://finance.yahoo.com/quote/USDINR=X/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find the price element
            price_element = soup.find('fin-streamer', {'data-symbol': 'USDINR=X'})
            if price_element and price_element.get('value'):
                rate = float(price_element.get('value'))
                logger.info(f"Fetched USD-INR rate from Yahoo Finance: {rate}")
                return rate
            
            # Alternative selector
            price_element = soup.find('span', {'data-test': 'qsp-price'})
            if price_element:
                rate_text = price_element.get_text().strip()
                rate = float(rate_text.replace(',', ''))
                logger.info(f"Fetched USD-INR rate from Yahoo Finance (alt): {rate}")
                return rate
                
        except ImportError:
            logger.warning("beautifulsoup4 not installed")
            return None
        except Exception as e:
            logger.warning(f"Error with Yahoo Finance: {e}")
            return None
    
    def get_live_exchange_rate(self) -> float:
        """
        Get live USD-INR exchange rate with fallback chain.
        
        Returns:
            float: USD to INR exchange rate
        """
        # Check cache first
        if self._is_cache_valid():
            logger.debug(f"Using cached exchange rate: {self.cache['rate']}")
            return self.cache['rate']
        
        # Try different sources in order of preference
        sources = [
            ("forex-python", self._fetch_from_forex_python),
            ("exchangerate-api", self._fetch_from_exchangerate_api),
            ("yahoo-finance", self._fetch_from_yahoo_finance)
        ]
        
        for source_name, source_func in sources:
            try:
                rate = source_func()
                if rate and rate > 0:
                    # Cache the result
                    self.cache = {
                        'rate': rate,
                        'timestamp': datetime.now()
                    }
                    logger.info(f"Successfully fetched USD-INR rate from {source_name}: {rate}")
                    return rate
            except Exception as e:
                logger.warning(f"Error with {source_name}: {e}")
                continue
        
        # All sources failed, use hardcoded rate
        logger.warning(f"All exchange rate sources failed, using hardcoded rate: {self.hardcoded_rate}")
        return self.hardcoded_rate
    
    def convert_usd_to_inr(self, usd_amount: float) -> float:
        """
        Convert USD amount to INR.
        
        Args:
            usd_amount: Amount in USD
            
        Returns:
            float: Amount in INR
        """
        if usd_amount <= 0:
            return 0.0
        
        rate = self.get_live_exchange_rate()
        return usd_amount * rate
    
    def convert_inr_to_usd(self, inr_amount: float) -> float:
        """
        Convert INR amount to USD.
        
        Args:
            inr_amount: Amount in INR
            
        Returns:
            float: Amount in USD
        """
        if inr_amount <= 0:
            return 0.0
        
        rate = self.get_live_exchange_rate()
        return inr_amount / rate
    
    def get_exchange_rate_info(self) -> Dict[str, Any]:
        """
        Get exchange rate information including source and cache status.
        
        Returns:
            Dict with rate, source, cached status, and timestamp
        """
        rate = self.get_live_exchange_rate()
        
        info = {
            'rate': rate,
            'cached': self._is_cache_valid(),
            'timestamp': datetime.now().isoformat()
        }
        
        if self._is_cache_valid():
            info['source'] = 'cache'
            info['cache_timestamp'] = self.cache['timestamp'].isoformat()
        else:
            info['source'] = 'live'
        
        return info

# Global instance
currency_converter = CurrencyConverter()

# Convenience functions
def get_live_exchange_rate() -> float:
    """Get live USD-INR exchange rate"""
    return currency_converter.get_live_exchange_rate()

def convert_usd_to_inr(usd_amount: float) -> float:
    """Convert USD to INR"""
    return currency_converter.convert_usd_to_inr(usd_amount)

def convert_inr_to_usd(inr_amount: float) -> float:
    """Convert INR to USD"""
    return currency_converter.convert_inr_to_usd(inr_amount)

def get_exchange_rate_info() -> Dict[str, Any]:
    """Get exchange rate information"""
    return currency_converter.get_exchange_rate_info()

__all__ = [
    'CurrencyConverter',
    'currency_converter',
    'get_live_exchange_rate',
    'convert_usd_to_inr',
    'convert_inr_to_usd',
    'get_exchange_rate_info'
]
