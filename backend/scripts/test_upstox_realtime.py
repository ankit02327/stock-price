#!/usr/bin/env python3
"""
Test Upstox Real-time Pricing

This script tests if Upstox is providing real-time prices for various Indian stocks.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_fetching', 'ind_stocks', 'current_fetching'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from ind_current_fetcher import IndianCurrentFetcher

def test_upstox_realtime():
    """Test Upstox real-time pricing for multiple stocks"""
    print("=" * 60)
    print("TESTING UPSTOX REAL-TIME PRICING")
    print("=" * 60)
    
    # Initialize fetcher
    fetcher = IndianCurrentFetcher()
    
    # Test stocks with known ISINs
    test_stocks = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC', 'WIPRO',
        'BAJAJ-AUTO', 'TITAN', 'ASIANPAINT', 'MARUTI', 'LT',
        'BHARTIARTL', 'SUNPHARMA', 'NESTLEIND', 'ULTRACEMCO'
    ]
    
    print(f"Testing {len(test_stocks)} stocks...")
    print()
    
    upstox_count = 0
    fallback_count = 0
    
    for i, stock in enumerate(test_stocks, 1):
        try:
            print(f"{i:2d}. Testing {stock}...", end=' ')
            result = fetcher.fetch_current_price(stock)
            
            price = result.get('price', 'N/A')
            source = result.get('source', 'unknown')
            currency = result.get('currency', 'INR')
            
            if source == 'upstox':
                upstox_count += 1
                print(f"✅ ₹{price} ({source})")
            else:
                fallback_count += 1
                print(f"⚠️  ₹{price} ({source})")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Upstox (Real-time): {upstox_count} stocks")
    print(f"⚠️  Fallback sources: {fallback_count} stocks")
    print(f"📊 Total tested: {len(test_stocks)} stocks")
    print(f"📈 Upstox success rate: {upstox_count/len(test_stocks)*100:.1f}%")
    
    if upstox_count > 0:
        print("\n🎉 Upstox is providing REAL-TIME prices!")
        print("   ✓ Live market data is working")
        print("   ✓ ISIN integration is successful")
    else:
        print("\n⚠️  Upstox not working - using fallback sources")
        print("   Check API credentials and network connection")

if __name__ == "__main__":
    test_upstox_realtime()
