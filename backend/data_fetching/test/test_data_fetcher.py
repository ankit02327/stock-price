"""
Test suite for live stock price fetching functionality.
Tests live price fetching, error handling, caching, and rate limiting.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Import the current fetcher
from current_fetcher import CurrentFetcher, LiveFetcher


class TestLiveFetcher:
    """Test live stock price fetching functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = LiveFetcher()
        # Clear cache for each test
        self.fetcher.cache = {}
    
    def test_fetch_us_stock_price(self):
        """Test fetching price for a valid US stock."""
        symbol = "AAPL"
        
        result = self.fetcher.fetch_live_price(symbol)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'price' in result
        assert 'timestamp' in result
        assert 'source' in result
        assert 'company_name' in result
        
        # Verify values
        assert result['symbol'] == symbol
        assert isinstance(result['price'], (int, float))
        assert result['price'] > 0
        assert result['source'] == 'yfinance'
        assert isinstance(result['company_name'], str)
        assert len(result['company_name']) > 0
    
    def test_fetch_indian_stock_price(self):
        """Test fetching price for a valid Indian stock."""
        symbol = "RELIANCE"
        
        result = self.fetcher.fetch_live_price(symbol)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert result['symbol'] == symbol
        assert isinstance(result['price'], (int, float))
        assert result['price'] > 0
        assert result['source'] == 'yfinance'
    
    def test_fetch_indian_stock_with_suffix(self):
        """Test fetching price for Indian stock with .NS suffix."""
        symbol = "TCS.NS"
        
        result = self.fetcher.fetch_live_price(symbol)
        
        assert isinstance(result, dict)
        assert result['symbol'] == symbol
        assert isinstance(result['price'], (int, float))
        assert result['price'] > 0
    
    def test_invalid_symbol_handling(self):
        """Test handling of invalid stock symbols."""
        invalid_symbols = ["INVALID123", "NONEXISTENT", ""]
        
        for symbol in invalid_symbols:
            if symbol:  # Skip empty string test
                with pytest.raises(Exception):
                    self.fetcher.fetch_live_price(symbol)
    
    def test_empty_symbol_handling(self):
        """Test handling of empty symbol."""
        with pytest.raises(ValueError, match="Stock symbol is required"):
            self.fetcher.fetch_live_price("")
    
    def test_none_symbol_handling(self):
        """Test handling of None symbol."""
        with pytest.raises(ValueError, match="Stock symbol is required"):
            self.fetcher.fetch_live_price(None)
    
    def test_cache_functionality(self):
        """Test that caching works correctly."""
        symbol = "AAPL"
        
        # First request - should hit API
        start_time = time.time()
        result1 = self.fetcher.fetch_live_price(symbol)
        first_duration = time.time() - start_time
        
        # Second request - should use cache
        start_time = time.time()
        result2 = self.fetcher.fetch_live_price(symbol)
        second_duration = time.time() - start_time
        
        # Verify cache hit (faster response)
        assert second_duration < first_duration
        assert result1['timestamp'] == result2['timestamp']
        assert result1['price'] == result2['price']
    
    def test_cache_expiration(self):
        """Test that cache expires after configured duration."""
        symbol = "AAPL"
        
        # Set very short cache duration for testing
        original_duration = self.fetcher.cache_duration
        self.fetcher.cache_duration = 1  # 1 second
        
        try:
            # First request
            result1 = self.fetcher.fetch_live_price(symbol)
            
            # Wait for cache to expire
            time.sleep(1.1)
            
            # Second request should hit API again
            result2 = self.fetcher.fetch_live_price(symbol)
            
            # Timestamps should be different (indicating fresh API call)
            assert result1['timestamp'] != result2['timestamp']
            
        finally:
            # Restore original cache duration
            self.fetcher.cache_duration = original_duration
    
    def test_rate_limiting(self):
        """Test that rate limiting is enforced."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        start_time = time.time()
        
        for symbol in symbols:
            self.fetcher.fetch_live_price(symbol)
        
        total_duration = time.time() - start_time
        
        # Should take at least (min_request_delay * (len(symbols) - 1)) seconds
        min_expected_duration = self.fetcher.min_request_delay * (len(symbols) - 1)
        assert total_duration >= min_expected_duration
    
    @patch('current_fetcher.yf.Ticker')
    def test_yfinance_error_handling(self, mock_ticker):
        """Test handling of yfinance errors."""
        # Mock yfinance to raise an exception
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("Network error")
        mock_ticker.return_value = mock_ticker_instance
        
        symbol = "AAPL"
        
        with pytest.raises(Exception, match="All APIs failed"):
            self.fetcher.fetch_live_price(symbol)
    
    def test_data_format_validation(self):
        """Test that returned data has correct format."""
        symbol = "AAPL"
        
        result = self.fetcher.fetch_live_price(symbol)
        
        # Verify timestamp format (ISO format)
        timestamp = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
        assert isinstance(timestamp, datetime)
        
        # Verify price is reasonable (between $0.01 and $100,000)
        assert 0.01 <= result['price'] <= 100000
        
        # Verify company name is not empty
        assert len(result['company_name'].strip()) > 0
    
    def test_multiple_symbols_different_categories(self):
        """Test fetching prices for symbols from different categories."""
        test_symbols = [
            ("AAPL", "us_stocks"),
            ("RELIANCE", "ind_stocks"),
            ("GOOGL", "us_stocks")
        ]
        
        for symbol, expected_category in test_symbols:
            result = self.fetcher.fetch_live_price(symbol)
            
            assert isinstance(result, dict)
            assert result['symbol'] == symbol
            assert isinstance(result['price'], (int, float))
            assert result['price'] > 0
            
            # Verify categorization
            actual_category = self.fetcher._categorize_stock(symbol)
            assert actual_category == expected_category
    
    def test_csv_saving_functionality(self):
        """Test that data is saved to CSV files."""
        symbol = "AAPL"
        
        # Clear any existing CSV data
        category = self.fetcher._categorize_stock(symbol)
        csv_path = os.path.join(self.fetcher.latest_dir, category, 'latest_prices.csv')
        
        # Remove existing file if it exists
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        # Fetch price (should save to CSV)
        result = self.fetcher.fetch_live_price(symbol)
        
        # Verify CSV file was created
        assert os.path.exists(csv_path)
        
        # Verify CSV contains the data
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            assert len(df) > 0
            assert symbol in df['symbol'].values
        except ImportError:
            # Fallback to csv module
            import csv
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) > 0
                assert any(row['symbol'] == symbol for row in rows)


class TestLiveFetcherIntegration:
    """Integration tests for live fetcher with real API calls."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = LiveFetcher()
        self.fetcher.cache = {}  # Clear cache
    
    def test_real_api_call_us_stock(self):
        """Test real API call for US stock (integration test)."""
        symbol = "AAPL"
        
        result = self.fetcher.fetch_live_price(symbol)
        
        # Verify we got real data
        assert isinstance(result['price'], (int, float))
        assert result['price'] > 100  # AAPL should be > $100
        assert "Apple" in result['company_name'] or "AAPL" in result['company_name']
        assert result['source'] == 'yfinance'
    
    def test_real_api_call_indian_stock(self):
        """Test real API call for Indian stock (integration test)."""
        symbol = "RELIANCE"
        
        result = self.fetcher.fetch_live_price(symbol)
        
        # Verify we got real data
        assert isinstance(result['price'], (int, float))
        assert result['price'] > 0
        assert "Reliance" in result['company_name'] or "RELIANCE" in result['company_name']
        assert result['source'] == 'yfinance'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
