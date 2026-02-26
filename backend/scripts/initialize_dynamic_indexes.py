#!/usr/bin/env python3
"""
Initialize Dynamic Indexes from Permanent Indexes

This script initializes the dynamic index files by copying all stocks from the permanent indexes.
It ensures proper column mapping and handles the transition from 6 columns to 8 columns.

Usage:
    python initialize_dynamic_indexes.py
    python initialize_dynamic_indexes.py --us-only
    python initialize_dynamic_indexes.py --ind-only
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def initialize_us_stocks_index():
    """Initialize US stocks dynamic index from permanent index"""
    print("🇺🇸 Initializing US stocks dynamic index...")
    
    # Paths
    permanent_path = os.path.join('..', 'permanent', 'us_stocks', 'index_us_stocks.csv')
    dynamic_path = os.path.join('..', 'data', 'index_us_stocks_dynamic.csv')
    
    # Check if permanent index exists
    if not os.path.exists(permanent_path):
        print(f"❌ Permanent US index not found: {permanent_path}")
        return False
    
    # Read permanent US index
    print(f"📖 Reading permanent US index: {permanent_path}")
    df_permanent = pd.read_csv(permanent_path)
    print(f"   Found {len(df_permanent)} stocks in permanent index")
    
    # Ensure required columns exist
    required_permanent_cols = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange']
    for col in required_permanent_cols:
        if col not in df_permanent.columns:
            print(f"❌ Missing required column in permanent index: {col}")
            return False
    
    # Add missing columns for dynamic index
    df_permanent['currency'] = 'USD'
    df_permanent['isin'] = ''  # Empty string, not 'nan'
    
    # Reorder columns to match dynamic index format
    dynamic_columns = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange', 'currency', 'isin']
    df_dynamic = df_permanent[dynamic_columns]
    
    # Sort alphabetically by symbol
    df_dynamic = df_dynamic.sort_values('symbol').reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(dynamic_path), exist_ok=True)
    
    # Save dynamic index
    df_dynamic.to_csv(dynamic_path, index=False)
    
    print(f"✅ Successfully initialized US stocks dynamic index")
    print(f"   Total stocks: {len(df_dynamic)}")
    print(f"   Saved to: {dynamic_path}")
    
    # Show sample
    print(f"\n📋 Sample US stocks:")
    for i, (_, row) in enumerate(df_dynamic.head(5).iterrows()):
        print(f"   {i+1}. {row['symbol']} - {row['company_name']}")
    
    return True

def initialize_indian_stocks_index():
    """Initialize Indian stocks dynamic index from permanent index"""
    print("\n🇮🇳 Initializing Indian stocks dynamic index...")
    
    # Paths
    permanent_path = os.path.join('..', 'permanent', 'ind_stocks', 'index_ind_stocks.csv')
    dynamic_path = os.path.join('..', 'data', 'index_ind_stocks_dynamic.csv')
    
    # Check if permanent index exists
    if not os.path.exists(permanent_path):
        print(f"❌ Permanent Indian index not found: {permanent_path}")
        return False
    
    # Read permanent Indian index
    print(f"📖 Reading permanent Indian index: {permanent_path}")
    df_permanent = pd.read_csv(permanent_path)
    print(f"   Found {len(df_permanent)} stocks in permanent index")
    
    # Ensure required columns exist
    required_permanent_cols = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange']
    for col in required_permanent_cols:
        if col not in df_permanent.columns:
            print(f"❌ Missing required column in permanent index: {col}")
            return False
    
    # Add missing columns for dynamic index
    df_permanent['currency'] = 'INR'
    df_permanent['isin'] = ''  # Empty string, not 'nan'
    
    # Reorder columns to match dynamic index format
    dynamic_columns = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange', 'currency', 'isin']
    df_dynamic = df_permanent[dynamic_columns]
    
    # Sort alphabetically by symbol
    df_dynamic = df_dynamic.sort_values('symbol').reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(dynamic_path), exist_ok=True)
    
    # Save dynamic index
    df_dynamic.to_csv(dynamic_path, index=False)
    
    print(f"✅ Successfully initialized Indian stocks dynamic index")
    print(f"   Total stocks: {len(df_dynamic)}")
    print(f"   Saved to: {dynamic_path}")
    
    # Show sample
    print(f"\n📋 Sample Indian stocks:")
    for i, (_, row) in enumerate(df_dynamic.head(5).iterrows()):
        print(f"   {i+1}. {row['symbol']} - {row['company_name']}")
    
    return True

def verify_dynamic_indexes():
    """Verify that both dynamic indexes were created correctly"""
    print("\n🔍 Verifying dynamic indexes...")
    
    # Check US stocks
    us_path = os.path.join('..', 'data', 'index_us_stocks_dynamic.csv')
    if os.path.exists(us_path):
        df_us = pd.read_csv(us_path)
        print(f"✅ US stocks dynamic index: {len(df_us)} stocks")
        
        # Verify columns
        expected_cols = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange', 'currency', 'isin']
        if list(df_us.columns) == expected_cols:
            print(f"   ✅ Columns match expected format")
        else:
            print(f"   ❌ Column mismatch. Expected: {expected_cols}, Got: {list(df_us.columns)}")
            return False
        
        # Verify currency
        if (df_us['currency'] == 'USD').all():
            print(f"   ✅ All stocks have USD currency")
        else:
            print(f"   ❌ Some stocks don't have USD currency")
            return False
    else:
        print(f"❌ US stocks dynamic index not found: {us_path}")
        return False
    
    # Check Indian stocks
    ind_path = os.path.join('..', 'data', 'index_ind_stocks_dynamic.csv')
    if os.path.exists(ind_path):
        df_ind = pd.read_csv(ind_path)
        print(f"✅ Indian stocks dynamic index: {len(df_ind)} stocks")
        
        # Verify columns
        expected_cols = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange', 'currency', 'isin']
        if list(df_ind.columns) == expected_cols:
            print(f"   ✅ Columns match expected format")
        else:
            print(f"   ❌ Column mismatch. Expected: {expected_cols}, Got: {list(df_ind.columns)}")
            return False
        
        # Verify currency
        if (df_ind['currency'] == 'INR').all():
            print(f"   ✅ All stocks have INR currency")
        else:
            print(f"   ❌ Some stocks don't have INR currency")
            return False
    else:
        print(f"❌ Indian stocks dynamic index not found: {ind_path}")
        return False
    
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize dynamic indexes from permanent indexes")
    parser.add_argument("--us-only", action="store_true", help="Initialize only US stocks")
    parser.add_argument("--ind-only", action="store_true", help="Initialize only Indian stocks")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing indexes")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("INITIALIZING DYNAMIC INDEXES FROM PERMANENT INDEXES")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = True
    
    if args.verify_only:
        success = verify_dynamic_indexes()
    else:
        if args.us_only:
            success = initialize_us_stocks_index()
        elif args.ind_only:
            success = initialize_indian_stocks_index()
        else:
            # Initialize both
            us_success = initialize_us_stocks_index()
            ind_success = initialize_indian_stocks_index()
            success = us_success and ind_success
            
            if success:
                # Verify both
                success = verify_dynamic_indexes()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Dynamic indexes initialization completed successfully!")
        print("✅ All 500+ stocks are now available in dynamic indexes")
        print("✅ New stocks can be added without losing existing data")
    else:
        print("❌ Dynamic indexes initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
