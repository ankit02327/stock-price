#!/usr/bin/env python3
"""
Restore US Stocks from Permanent Index

This script restores the 500 US stocks from the permanent index to the dynamic index.
It uses the DynamicIndexManager to ensure proper validation and prevent data loss.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def restore_us_index():
    """Restore US stocks from permanent index to dynamic index using DynamicIndexManager"""
    print("=" * 60)
    print("RESTORING US STOCKS FROM PERMANENT INDEX")
    print("=" * 60)
    
    try:
        from shared.index_manager import DynamicIndexManager
        
        # Initialize index manager
        data_dir = os.path.join('..', 'data')
        index_manager = DynamicIndexManager(data_dir)
        
        # Check current count
        current_count = index_manager.get_stock_count('us_stocks')
        print(f"📊 Current US stocks in dynamic index: {current_count}")
        
        if current_count >= 400:
            print("✅ Dynamic index already has sufficient stocks, no restoration needed")
            return True
        
        # Use the initialization method from DynamicIndexManager
        print(f"🔄 Restoring US stocks from permanent index...")
        success = index_manager.initialize_from_permanent('us_stocks')
        
        if success:
            # Verify the restoration
            new_count = index_manager.get_stock_count('us_stocks')
            print(f"✅ Successfully restored US stocks")
            print(f"   Total stocks in dynamic index: {new_count}")
            
            # Show some examples
            symbols = index_manager.get_all_symbols('us_stocks')
            print(f"\n📋 Sample restored stocks:")
            for i, symbol in enumerate(symbols[:5]):
                stock_info = index_manager.get_stock_info(symbol, 'us_stocks')
                company_name = stock_info.get('company_name', symbol) if stock_info else symbol
                print(f"   {i+1}. {symbol} - {company_name}")
            if len(symbols) > 5:
                print(f"   ... and {len(symbols) - 5} more")
            
            return True
        else:
            print("❌ Failed to restore US stocks from permanent index")
            return False
            
    except Exception as e:
        print(f"❌ Error during restoration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = restore_us_index()
        if success:
            print("\n🎉 US index restoration completed successfully!")
        else:
            print("\n❌ US index restoration failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during restoration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
