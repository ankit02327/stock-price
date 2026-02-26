#!/usr/bin/env python3
"""
Test script to verify the updated Upstox integration with ISIN mapping.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_fetching', 'ind_stocks', 'current_fetching'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_fetching'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

# Load environment variables
load_dotenv()

def test_upstox_integration():
    """Test the updated Upstox integration."""
    print("=" * 60)
    print("UPSTOX INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Import with absolute path to avoid relative import issues
        import ind_current_fetcher
        IndianCurrentFetcher = ind_current_fetcher.IndianCurrentFetcher
        
        # Initialize fetcher
        fetcher = IndianCurrentFetcher()
        
        # Test cases
        test_cases = [
            # Hardcoded stocks (should use hardcoded ISIN)
            ('RELIANCE', 'Should use hardcoded ISIN'),
            ('TCS', 'Should use hardcoded ISIN'),
            ('INFY', 'Should use hardcoded ISIN'),
            
            # Non-hardcoded stocks (should lookup ISIN from instruments file)
            ('BAJAJ-AUTO', 'Should lookup ISIN from instruments file'),
            ('TITAN', 'Should lookup ISIN from instruments file'),
            ('HDFCBANK', 'Should use hardcoded ISIN'),
            
            # Invalid stocks (should return None)
            ('INVALID123', 'Should return None'),
            ('FAKE', 'Should return None'),
        ]
        
        print("\n1. Testing instrument key generation:")
        print("-" * 50)
        
        for symbol, description in test_cases:
            print(f"\nTesting {symbol}: {description}")
            try:
                instrument_key = fetcher.get_instrument_key(symbol)
                if instrument_key:
                    print(f"  ✓ Instrument key: {instrument_key}")
                else:
                    print(f"  ⚠ No instrument key found (will skip Upstox)")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("\n2. Testing Upstox API calls:")
        print("-" * 50)
        
        # Test with a few stocks that should work
        test_symbols = ['RELIANCE', 'BAJAJ-AUTO', 'INVALID123']
        
        for symbol in test_symbols:
            print(f"\nTesting Upstox API for {symbol}:")
            try:
                result = fetcher.fetch_price_from_upstox(symbol)
                print(f"  ✓ Success: {result}")
            except Exception as e:
                print(f"  ⚠ Expected failure: {e}")
        
        print("\n3. Testing full fetch chain:")
        print("-" * 50)
        
        # Test the full fetch chain
        for symbol in ['RELIANCE', 'BAJAJ-AUTO']:
            print(f"\nTesting full fetch for {symbol}:")
            try:
                result = fetcher.fetch_current_price(symbol)
                print(f"  ✓ Success: Price = ₹{result['price']}, Source = {result['source']}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_upstox_integration()
