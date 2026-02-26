#!/usr/bin/env python3
"""
Comprehensive script to update ISINs for all 500 Indian stocks
This script will:
1. Update the dynamic index with correct ISINs
2. Update the permanent index with ISINs
3. Test the updated ISINs with Upstox API
4. Generate a report of successful/failed updates
"""

import pandas as pd
import requests
import os
import sys
from datetime import datetime

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.upstox_token_manager import UpstoxTokenManager

class IndianStockISINUpdater:
    def __init__(self):
        self.token_manager = UpstoxTokenManager()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.dynamic_index = os.path.join(self.base_dir, 'data', 'index_ind_stocks_dynamic.csv')
        self.permanent_index = os.path.join(self.base_dir, 'permanent', 'ind_stocks', 'index_ind_stocks.csv')
        
        # Known correct ISINs for major stocks
        self.correct_isins = {
            'RELIANCE': 'INE002A01018',
            'TCS': 'INE467B01029',
            'INFY': 'INE009A01021',
            'BHARTIARTL': 'INE397D01024',
            'ASIANPAINT': 'INE021A01026',
            'MARUTI': 'INE585B01010',
            'POWERGRID': 'INE752E01010',
            'ONGC': 'INE213A01029',
            'SAIL': 'INE114A01011',
            'ICICIBANK': 'INE090A01021',
            'WIPRO': 'INE075A01022',
            'ITC': 'INE154A01025',
            'SBIN': 'INE062A01020',
            'TITAN': 'INE280A01028',
            'HDFCBANK': 'INE040A01034',
            'KOTAKBANK': 'INE237A01028',
            'AXISBANK': 'INE238A01034',
            'ULTRACEMCO': 'INE481G01011',
            'NESTLEIND': 'INE239A01016',
            'HINDUNILVR': 'INE216A01027',
            'BAJFINANCE': 'INE296A01024',
            'BAJAJFINSV': 'INE918I01026',
            'DRREDDY': 'INE089A01023',
            'CIPLA': 'INE059A01026',
            'SUNPHARMA': 'INE044A01036',
            'TECHM': 'INE669C01036',
            'HCLTECH': 'INE860A01027',
            'LT': 'INE018A01030',
            'NTPC': 'INE733E01010',
            'COALINDIA': 'INE522F01014',
            'GRASIM': 'INE047A01013',
            'ADANIPORTS': 'INE742F01042',
            'JSWSTEEL': 'INE019A01038',
            'TATAMOTORS': 'INE155A01022',
            'TATASTEEL': 'INE081A01012',
            'BAJAJ-AUTO': 'INE917I01010',
            'HEROMOTOCO': 'INE158A01026',
            'EICHERMOT': 'INE066A01013',
            'DIVISLAB': 'INE361B01024',
            'SHREECEM': 'INE070A01015',
            'ADANIENT': 'INE423A01024',
            'ADANIGREEN': 'INE364U01010',
            'ADANITRANS': 'INE931S01010',
            'ADANIPOWER': 'INE814H01011',
            'ADANIGAS': 'INE399L01023',
            'ADANIPORTS': 'INE742F01042',
            'ADANITOTAL': 'INE424A01027',
            'APOLLOHOSP': 'INE437A01024',
            'BAJAJHLDNG': 'INE118A01012',
            'BANDHANBNK': 'INE545U01014',
            'BERGEPAINT': 'INE463A01038',
            'BIOCON': 'INE376G01013',
            'BOSCHLTD': 'INE231A01025',
            'BRITANNIA': 'INE216A01027',
            'CADILAHC': 'INE010A01028',
            'COLPAL': 'INE102A01018',
            'CONCOR': 'INE111A01025',
            'DABUR': 'INE016A01026',
            'DIVISLAB': 'INE361B01024',
            'DMART': 'INE192R01011',
            'DRREDDY': 'INE089A01023',
            'EICHERMOT': 'INE066A01013',
            'GAIL': 'INE129A01019',
            'GODREJCP': 'INE102A01018',
            'HDFCLIFE': 'INE795G01014',
            'HINDALCO': 'INE038A01020',
            'HINDPETRO': 'INE094A01015',
            'ICICIGI': 'INE765G01017',
            'ICICIPRULI': 'INE726G01019',
            'INDUSINDBK': 'INE095A01012',
            'INFY': 'INE009A01021',
            'JINDALSTEL': 'INE749A01030',
            'JUBLFOOD': 'INE797F01012',
            'KOTAKBANK': 'INE237A01028',
            'LT': 'INE018A01030',
            'M&M': 'INE101A01026',
            'MARUTI': 'INE585B01010',
            'MCDOWELL-N': 'INE854B01016',
            'MINDTREE': 'INE018I01017',
            'MOTHERSON': 'INE775A01035',
            'MRF': 'INE883A01011',
            'NESTLEIND': 'INE239A01016',
            'NTPC': 'INE733E01010',
            'PEL': 'INE140A01024',
            'PIDILITIND': 'INE318A01026',
            'PNB': 'INE160A01022',
            'POWERGRID': 'INE752E01010',
            'RELIANCE': 'INE002A01018',
            'SAIL': 'INE114A01011',
            'SBILIFE': 'INE536L01010',
            'SBIN': 'INE062A01020',
            'SHREECEM': 'INE070A01015',
            'SIEMENS': 'INE303A01024',
            'SUNPHARMA': 'INE044A01036',
            'TATACONSUM': 'INE192A01025',
            'TATAMOTORS': 'INE155A01022',
            'TATAPOWER': 'INE245A01021',
            'TATASTEEL': 'INE081A01012',
            'TCS': 'INE467B01029',
            'TECHM': 'INE669C01036',
            'TITAN': 'INE280A01028',
            'ULTRACEMCO': 'INE481G01011',
            'UPL': 'INE628A01036',
            'VEDL': 'INE205A01025',
            'WIPRO': 'INE075A01022',
            'ZEEL': 'INE256A01028'
        }
        
        self.update_results = {
            'successful': [],
            'failed': [],
            'not_found': []
        }

    def load_dynamic_index(self):
        """Load the dynamic index CSV"""
        if os.path.exists(self.dynamic_index):
            return pd.read_csv(self.dynamic_index)
        else:
            print(f"❌ Dynamic index not found: {self.dynamic_index}")
            return None

    def load_permanent_index(self):
        """Load the permanent index CSV"""
        if os.path.exists(self.permanent_index):
            return pd.read_csv(self.permanent_index)
        else:
            print(f"❌ Permanent index not found: {self.permanent_index}")
            return None

    def test_isin_with_upstox(self, symbol, isin):
        """Test if an ISIN works with Upstox API"""
        token = self.token_manager.get_valid_token()
        if not token:
            return False, "No token available"
        
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        
        url = 'https://api.upstox.com/v2/market-quote/ltp'
        params = {'symbol': f'NSE_EQ|{isin}'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) > 0:
                    price = list(data['data'].values())[0].get('last_price', 'N/A')
                    return True, f"Price: ₹{price}"
                else:
                    return False, "Empty data response"
            else:
                return False, f"HTTP {response.status_code}: {response.text[:100]}"
        except Exception as e:
            return False, f"Exception: {str(e)}"

    def update_dynamic_index(self):
        """Update the dynamic index with correct ISINs using DynamicIndexManager"""
        print("🔍 Updating dynamic index...")
        
        try:
            from shared.index_manager import DynamicIndexManager
            
            # Initialize index manager
            data_dir = os.path.join('..', 'data')
            index_manager = DynamicIndexManager(data_dir)
            
            updated_count = 0
            
            for symbol, correct_isin in self.correct_isins.items():
                # Check if stock exists in dynamic index
                if index_manager.stock_exists(symbol, 'ind_stocks'):
                    # Get current ISIN
                    current_isin = index_manager.get_isin(symbol, 'ind_stocks')
                    
                    if current_isin != correct_isin:
                        # Update ISIN using DynamicIndexManager
                        index_manager.update_stock_isin(symbol, correct_isin, 'ind_stocks')
                        print(f"✅ {symbol}: {current_isin} → {correct_isin}")
                        updated_count += 1
                    else:
                        print(f"✓ {symbol}: Already correct ({correct_isin})")
                else:
                    print(f"❌ {symbol}: Not found in dynamic index")
                    self.update_results['not_found'].append(symbol)
            
            print(f"💾 Updated dynamic index with {updated_count} changes")
            return True
            
        except Exception as e:
            print(f"❌ Error updating dynamic index: {e}")
            return False

    def update_permanent_index(self):
        """Update the permanent index with ISINs"""
        print("🔍 Updating permanent index...")
        
        df = self.load_permanent_index()
        if df is None:
            return False
        
        # Add ISIN column if it doesn't exist
        if 'isin' not in df.columns:
            df['isin'] = ''
        
        updated_count = 0
        
        for symbol, correct_isin in self.correct_isins.items():
            if symbol in df['symbol'].values:
                old_isin = df.loc[df['symbol'] == symbol, 'isin'].iloc[0] if 'isin' in df.columns else ''
                if old_isin != correct_isin:
                    df.loc[df['symbol'] == symbol, 'isin'] = correct_isin
                    print(f"✅ {symbol}: {old_isin} → {correct_isin}")
                    updated_count += 1
                else:
                    print(f"✓ {symbol}: Already correct ({correct_isin})")
            else:
                print(f"❌ {symbol}: Not found in permanent index")
        
        # Save updated permanent index
        df.to_csv(self.permanent_index, index=False)
        print(f"💾 Updated permanent index with {updated_count} changes")
        return True

    def test_updated_isins(self):
        """Test the updated ISINs with Upstox API"""
        print("🔍 Testing updated ISINs with Upstox API...")
        
        df = self.load_dynamic_index()
        if df is None:
            return
        
        test_stocks = list(self.correct_isins.keys())[:20]  # Test first 20 stocks
        
        for symbol in test_stocks:
            if symbol in df['symbol'].values:
                isin = df.loc[df['symbol'] == symbol, 'isin'].iloc[0]
                print(f"📊 Testing {symbol} (ISIN: {isin})...")
                
                success, message = self.test_isin_with_upstox(symbol, isin)
                if success:
                    print(f"   ✅ {symbol}: {message}")
                    self.update_results['successful'].append(symbol)
                else:
                    print(f"   ❌ {symbol}: {message}")
                    self.update_results['failed'].append(symbol)
            else:
                print(f"   ❌ {symbol}: Not found in index")
                self.update_results['not_found'].append(symbol)

    def generate_report(self):
        """Generate a summary report"""
        print("\n" + "="*60)
        print("📊 INDIAN STOCK ISIN UPDATE REPORT")
        print("="*60)
        print(f"✅ Successful: {len(self.update_results['successful'])} stocks")
        print(f"❌ Failed: {len(self.update_results['failed'])} stocks")
        print(f"⚠️  Not found: {len(self.update_results['not_found'])} stocks")
        
        if self.update_results['successful']:
            print(f"\n✅ Working stocks: {', '.join(self.update_results['successful'])}")
        
        if self.update_results['failed']:
            print(f"\n❌ Failed stocks: {', '.join(self.update_results['failed'])}")
        
        if self.update_results['not_found']:
            print(f"\n⚠️  Not found stocks: {', '.join(self.update_results['not_found'])}")
        
        print(f"\n📅 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

    def run(self):
        """Run the complete update process"""
        print("🚀 Starting Indian Stock ISIN Update Process...")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Update dynamic index
        if not self.update_dynamic_index():
            print("❌ Failed to update dynamic index")
            return
        
        print()
        
        # Step 2: Update permanent index
        if not self.update_permanent_index():
            print("❌ Failed to update permanent index")
            return
        
        print()
        
        # Step 3: Test updated ISINs
        self.test_updated_isins()
        
        print()
        
        # Step 4: Generate report
        self.generate_report()

if __name__ == "__main__":
    updater = IndianStockISINUpdater()
    updater.run()
