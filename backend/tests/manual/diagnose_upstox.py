#!/usr/bin/env python3
"""
Diagnose why Upstox isn't being used
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data_fetching.ind_stocks.current_fetching.ind_current_fetcher import IndianCurrentFetcher

print('=' * 60)
print('UPSTOX DIAGNOSTIC TEST')
print('=' * 60)

fetcher = IndianCurrentFetcher()

# Check configuration
print('\n1. Configuration Check:')
print(f'   Token Manager exists: {hasattr(fetcher, "token_manager")}')
print(f'   Upstox access token: {bool(fetcher.upstox_access_token)}')
print(f'   Token (first 30 chars): {fetcher.upstox_access_token[:30] if fetcher.upstox_access_token else "None"}...')

# Try fetching with verbose output
print('\n2. Fetching RELIANCE...')
try:
    result = fetcher.fetch_current_price('RELIANCE')
    print(f'\n3. Results:')
    print(f'   Symbol: {result["symbol"]}')
    print(f'   Price: ₹{result["price"]}')
    print(f'   Source: {result["source"]}')
    print(f'   Currency: {result["currency"]}')
    
    if result["source"] == "upstox":
        print('\n✅ SUCCESS! Using Upstox live data!')
    else:
        print(f'\n⚠️  WARNING: Using "{result["source"]}" instead of Upstox')
        print('   Upstox API call likely failed in the fetcher.')
        print('   Check the console output above for error messages.')
        
except Exception as e:
    print(f'\n❌ Error: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
