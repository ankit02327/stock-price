#!/usr/bin/env python3
"""
Simple test script for the Live Stock Price API
"""

import requests
import json
import time
import sys
import os

def check_virtual_environment():
    """Check if running in a virtual environment"""
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None
    )
    
    if not in_venv:
        print("⚠️  WARNING: Not running in a virtual environment!")
        print("   Best practice: Activate virtual environment first:")
        print("   venv\\Scripts\\activate  # Windows")
        print("   source venv/bin/activate  # macOS/Linux")
        print("")
    
    return in_venv

# Configuration
BASE_URL = "http://localhost:5000"
TEST_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "INVALID_SYMBOL"]

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_live_price(symbol):
    """Test live price endpoint for a symbol"""
    print(f"Testing live price for {symbol}...")
    try:
        response = requests.get(f"{BASE_URL}/live_price?symbol={symbol}", timeout=15)
        data = response.json()
        
        if response.status_code == 200:
            if data.get('success') and data.get('data'):
                price_data = data['data']
                print(f"✓ {symbol}: ${price_data['price']} from {price_data['source']} at {price_data['timestamp']}")
                return True
            else:
                print(f"✗ {symbol}: API returned success=False")
                return False
        else:
            print(f"✗ {symbol}: HTTP {response.status_code} - {data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"✗ {symbol}: Error - {e}")
        return False

def test_latest_prices():
    """Test latest prices endpoint"""
    print("Testing latest prices...")
    try:
        response = requests.get(f"{BASE_URL}/latest_prices", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                prices = data.get('data', [])
                print(f"✓ Latest prices: {len(prices)} entries")
                for price in prices[:3]:  # Show first 3
                    print(f"  - {price['symbol']}: ${price['price']} ({price['source']})")
                return True
            else:
                print(f"✗ Latest prices failed: {data.get('message')}")
                return False
        else:
            print(f"✗ Latest prices HTTP error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Latest prices error: {e}")
        return False

def test_symbols():
    """Test symbols endpoint"""
    print("Testing symbols...")
    try:
        response = requests.get(f"{BASE_URL}/symbols", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                symbols = data.get('data', {})
                print(f"✓ Symbols: {sum(len(s) for s in symbols.values())} total")
                for category, symbol_list in symbols.items():
                    print(f"  - {category}: {len(symbol_list)} symbols")
                return True
            else:
                print(f"✗ Symbols failed: {data.get('message')}")
                return False
        else:
            print(f"✗ Symbols HTTP error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Symbols error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Live Stock Price API Test Suite")
    print("=" * 50)
    
    # Check virtual environment
    check_virtual_environment()
    
    # Test health check first
    if not test_health_check():
        print("\n❌ Health check failed. Is the server running?")
        print("Start the server with: py run_server.py")
        sys.exit(1)
    
    print("\n" + "-" * 30)
    
    # Test live prices
    success_count = 0
    for symbol in TEST_SYMBOLS:
        if test_live_price(symbol):
            success_count += 1
        time.sleep(1)  # Rate limiting
    
    print(f"\n✓ {success_count}/{len(TEST_SYMBOLS)} symbols tested successfully")
    
    print("\n" + "-" * 30)
    
    # Test other endpoints
    test_latest_prices()
    test_symbols()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
