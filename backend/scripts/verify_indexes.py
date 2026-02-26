#!/usr/bin/env python3
"""
Verify Dynamic Indexes

This script verifies that both US and Indian dynamic indexes are properly populated.
"""

import pandas as pd
import os

def verify_indexes():
    """Verify both dynamic indexes"""
    print("=" * 60)
    print("VERIFYING DYNAMIC INDEXES")
    print("=" * 60)
    
    # Check US Dynamic Index
    us_path = os.path.join('..', 'data', 'index_us_stocks_dynamic.csv')
    if os.path.exists(us_path):
        us_df = pd.read_csv(us_path)
        print(f"📊 US Dynamic Index:")
        print(f"   Total stocks: {len(us_df)}")
        print(f"   Sample symbols: {us_df.head(3)['symbol'].tolist()}")
        print(f"   Columns: {us_df.columns.tolist()}")
    else:
        print("❌ US Dynamic Index not found")
    
    print()
    
    # Check Indian Dynamic Index
    ind_path = os.path.join('..', 'data', 'index_ind_stocks_dynamic.csv')
    if os.path.exists(ind_path):
        ind_df = pd.read_csv(ind_path)
        stocks_with_isin = ind_df[ind_df['isin'].str.len() > 0]
        print(f"📊 Indian Dynamic Index:")
        print(f"   Total stocks: {len(ind_df)}")
        print(f"   Stocks with ISIN: {len(stocks_with_isin)}")
        print(f"   ISIN coverage: {len(stocks_with_isin)/len(ind_df)*100:.1f}%")
        print(f"   Columns: {ind_df.columns.tolist()}")
        
        if len(stocks_with_isin) > 0:
            print(f"   Sample with ISIN:")
            for _, row in stocks_with_isin.head(5).iterrows():
                print(f"     {row['symbol']}: {row['isin']}")
    else:
        print("❌ Indian Dynamic Index not found")
    
    print()
    print("✅ Verification complete!")

if __name__ == "__main__":
    verify_indexes()
