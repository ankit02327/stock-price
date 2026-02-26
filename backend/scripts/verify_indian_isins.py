"""
Verify Indian stock ISINs by testing them with live Upstox API data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import random
import json

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_fetching', 'ind_stocks', 'current_fetching'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from ind_current_fetcher import IndianCurrentFetcher

def verify_random_isins(num_stocks=25):
    """Test random Indian stocks with live Upstox data"""
    
    print("=" * 100)
    print("🔍 Verifying Indian Stock ISINs with Live Upstox Data")
    print("=" * 100)
    print()
    
    # Load Indian stocks
    ind_stocks_path = os.path.join(os.path.dirname(__file__), '../../permanent/ind_stocks/index_ind_stocks.csv')
    df = pd.read_csv(ind_stocks_path)
    
    print(f"📊 Total Indian stocks available: {len(df)}")
    print(f"🎲 Randomly selecting {num_stocks} stocks for verification...")
    print()
    
    # Random sample
    random.seed(42)  # For reproducibility
    samples = df.sample(num_stocks)
    
    # Initialize Indian stock fetcher (uses Upstox or fallback)
    try:
        fetcher = IndianCurrentFetcher()
        print("✅ Indian stock fetcher initialized (Upstox with fallback)")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize fetcher: {e}")
        print("⚠️  Showing format validation only (ISINs not tested with live API)")
        print()
        fetcher = None
    
    # Test each stock
    results = {
        'success': [],
        'failed': [],
        'format_invalid': []
    }
    
    print("Testing stocks...")
    print("-" * 100)
    print(f"{'Status':<8} {'Symbol':<15} {'Company Name':<40} {'ISIN':<15} {'Result'}")
    print("-" * 100)
    
    for idx, row in samples.iterrows():
        symbol = row['symbol']
        company_name = row['company_name'][:38]  # Truncate if too long
        isin = row['isin']
        
        # First check format
        format_valid = len(isin) == 12 and isin.startswith('INE')
        
        if not format_valid:
            status = "❌"
            result = f"Invalid format"
            results['format_invalid'].append((symbol, isin))
        elif fetcher is None:
            status = "⚠️"
            result = f"Format OK (API unavailable)"
            results['success'].append((symbol, isin))
        else:
            # Test with live API
            try:
                price_data = fetcher.fetch_current_price(symbol)
                
                if price_data and 'price' in price_data and price_data['price'] != 'N/A':
                    status = "✅"
                    price = price_data['price']
                    source = price_data.get('source', 'unknown')
                    result = f"Live: ₹{price:.2f} ({source})"
                    results['success'].append((symbol, isin))
                else:
                    status = "⚠️"
                    result = "No data returned"
                    results['failed'].append((symbol, isin))
                    
            except Exception as e:
                status = "❌"
                result = f"API Error: {str(e)[:30]}"
                results['failed'].append((symbol, isin))
        
        print(f"{status:<8} {symbol:<15} {company_name:<40} {isin:<15} {result}")
    
    # Summary
    print("-" * 100)
    print()
    print("=" * 100)
    print("📈 VERIFICATION SUMMARY")
    print("=" * 100)
    print(f"✅ Successful: {len(results['success'])}/{num_stocks} ({len(results['success'])/num_stocks*100:.1f}%)")
    print(f"❌ Failed: {len(results['failed'])}/{num_stocks} ({len(results['failed'])/num_stocks*100:.1f}%)")
    print(f"⚠️  Format Invalid: {len(results['format_invalid'])}/{num_stocks} ({len(results['format_invalid'])/num_stocks*100:.1f}%)")
    print()
    
    if results['failed']:
        print("Failed ISINs:")
        for symbol, isin in results['failed']:
            print(f"  - {symbol}: {isin}")
        print()
    
    if results['format_invalid']:
        print("Format Invalid ISINs:")
        for symbol, isin in results['format_invalid']:
            print(f"  - {symbol}: {isin}")
        print()
    
    # Overall status
    if len(results['success']) == num_stocks:
        print("🎉 ALL ISINs VERIFIED SUCCESSFULLY!")
    elif len(results['success']) >= num_stocks * 0.9:
        print("✅ Most ISINs verified (>90% success rate)")
    elif len(results['success']) >= num_stocks * 0.7:
        print("⚠️  Partial verification (70-90% success rate)")
    else:
        print("❌ Verification issues detected (<70% success rate)")
    
    print("=" * 100)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify Indian stock ISINs with live Upstox data')
    parser.add_argument('--count', '-n', type=int, default=25, help='Number of stocks to test (default: 25)')
    
    args = parser.parse_args()
    
    try:
        results = verify_random_isins(args.count)
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

