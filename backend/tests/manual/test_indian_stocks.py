#!/usr/bin/env python3
"""
Test Indian Stock Fetcher
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data_fetching.ind_stocks.current_fetching.ind_current_fetcher import IndianCurrentFetcher

print("=" * 60)
print("TESTING INDIAN STOCK FETCHER")
print("=" * 60)

fetcher = IndianCurrentFetcher()

# Test stocks
test_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']

for symbol in test_stocks:
    print(f"\nTesting {symbol}...")
    try:
        result = fetcher.fetch_current_price(symbol)
        print(f"✅ SUCCESS!")
        print(f"   Symbol: {result['symbol']}")
        print(f"   Price: ₹{result['price']}")
        print(f"   Source: {result['source']}")
        print(f"   Currency: {result['currency']}")
        print(f"   Timestamp: {result['timestamp']}")
        if 'company_name' in result:
            print(f"   Company: {result['company_name']}")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
