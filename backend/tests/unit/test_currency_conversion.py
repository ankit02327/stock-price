#!/usr/bin/env python3
"""
Test script for real-time currency conversion
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.currency_converter import get_live_exchange_rate, convert_usd_to_inr, convert_inr_to_usd, get_exchange_rate_info

def test_currency_conversion():
    print('🌍 Testing Real-time Currency Conversion')
    print('=' * 40)

    # Test exchange rate info
    print('\n📊 Exchange Rate Information:')
    info = get_exchange_rate_info()
    print(f'Rate: {info["rate"]:.4f} USD/INR')
    print(f'Source: {info["source"]}')
    print(f'Cached: {info["cached"]}')
    print(f'Timestamp: {info["timestamp"]}')

    # Test USD to INR conversion
    print('\n💵 USD to INR Conversion:')
    usd_amounts = [1, 10, 100, 1000]
    for usd in usd_amounts:
        inr = convert_usd_to_inr(usd)
        print(f'${usd} = ₹{inr:.2f}')

    # Test INR to USD conversion
    print('\n💸 INR to USD Conversion:')
    inr_amounts = [83, 830, 8300, 83000]
    for inr in inr_amounts:
        usd = convert_inr_to_usd(inr)
        print(f'₹{inr} = ${usd:.2f}')

    # Test stock price conversion
    print('\n📈 Stock Price Conversion Examples:')
    
    # US stock price in INR
    aapl_price_usd = 248.89
    aapl_price_inr = convert_usd_to_inr(aapl_price_usd)
    print(f'AAPL: ${aapl_price_usd} = ₹{aapl_price_inr:.2f}')
    
    # Indian stock price in USD
    tcs_price_inr = 4158.80
    tcs_price_usd = convert_inr_to_usd(tcs_price_inr)
    print(f'TCS: ₹{tcs_price_inr} = ${tcs_price_usd:.2f}')

    print('\n✅ Currency conversion working!')

if __name__ == "__main__":
    test_currency_conversion()
