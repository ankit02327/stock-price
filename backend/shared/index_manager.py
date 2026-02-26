"""
Dynamic Index Manager
Manages the master dynamic index files for stock metadata
"""

import os
import pandas as pd
from typing import Dict, Optional

class DynamicIndexManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.us_index_path = os.path.join(data_dir, 'index_us_stocks_dynamic.csv')
        self.ind_index_path = os.path.join(data_dir, 'index_ind_stocks_dynamic.csv')
    
    def get_index_path(self, category: str) -> str:
        """Get path to dynamic index for category"""
        if category == 'us_stocks':
            return self.us_index_path
        elif category == 'ind_stocks':
            return self.ind_index_path
        else:
            raise ValueError(f"Unknown category: {category}")
    
    def stock_exists(self, symbol: str, category: str) -> bool:
        """Check if stock exists in dynamic index"""
        index_path = self.get_index_path(category)
        if not os.path.exists(index_path):
            return False
        df = pd.read_csv(index_path)
        return symbol.upper() in df['symbol'].str.upper().values
    
    def get_stock_info(self, symbol: str, category: str) -> Optional[Dict]:
        """Get stock metadata from dynamic index"""
        index_path = self.get_index_path(category)
        if not os.path.exists(index_path):
            return None
        df = pd.read_csv(index_path)
        row = df[df['symbol'].str.upper() == symbol.upper()]
        if row.empty:
            return None
        return row.iloc[0].to_dict()
    
    def add_stock(self, symbol: str, stock_info: Dict, category: str):
        """Add new stock to dynamic index (alphabetically sorted)"""
        index_path = self.get_index_path(category)
        
        # Read existing index
        if os.path.exists(index_path):
            df = pd.read_csv(index_path)
            # Ensure ISIN column exists for backward compatibility
            if 'isin' not in df.columns:
                df['isin'] = ''
            # Ensure ISIN column is string type
            df['isin'] = df['isin'].astype(str)
        else:
            df = pd.DataFrame(columns=['symbol', 'company_name', 'sector', 
                                      'market_cap', 'headquarters', 'exchange', 'currency', 'isin'])
        
        # Check if already exists
        if symbol.upper() in df['symbol'].str.upper().values:
            print(f"Stock {symbol} already exists in {category} index")
            return
        
        # Add new stock
        new_row = {
            'symbol': symbol.upper(),
            'company_name': stock_info.get('company_name', symbol),
            'sector': stock_info.get('sector', 'N/A'),
            'market_cap': stock_info.get('market_cap', ''),
            'headquarters': stock_info.get('headquarters', 'N/A'),
            'exchange': stock_info.get('exchange', 'N/A'),
            'currency': 'INR' if category == 'ind_stocks' else 'USD',
            'isin': stock_info.get('isin', '')
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Sort alphabetically by symbol
        df = df.sort_values('symbol').reset_index(drop=True)
        
        # Validate no deletion before saving
        self.validate_no_deletion(category, df)
        
        # Save
        df.to_csv(index_path, index=False)
        print(f"Added {symbol} to {category} dynamic index")
    
    def get_all_symbols(self, category: str) -> list:
        """Get all symbols from dynamic index"""
        index_path = self.get_index_path(category)
        if not os.path.exists(index_path):
            return []
        df = pd.read_csv(index_path)
        return df['symbol'].tolist()
    
    def get_isin(self, symbol: str, category: str) -> Optional[str]:
        """Get ISIN for a stock from dynamic index"""
        index_path = self.get_index_path(category)
        if not os.path.exists(index_path):
            return None
        df = pd.read_csv(index_path)
        # Ensure ISIN column exists for backward compatibility
        if 'isin' not in df.columns:
            return None
        row = df[df['symbol'].str.upper() == symbol.upper()]
        if row.empty:
            return None
        isin = row.iloc[0]['isin']
        return isin if pd.notna(isin) and isin.strip() else None
    
    def validate_no_deletion(self, category: str, new_df: pd.DataFrame):
        """Ensure we're not deleting stocks from dynamic index"""
        index_path = self.get_index_path(category)
        if os.path.exists(index_path):
            old_df = pd.read_csv(index_path)
            old_count = len(old_df)
            new_count = len(new_df)
            
            if new_count < old_count:
                raise ValueError(
                    f"SAFETY CHECK FAILED: Attempting to reduce {category} from "
                    f"{old_count} to {new_count} stocks. This would delete stocks!"
                )
    
    def update_stock_isin(self, symbol: str, isin: str, category: str):
        """Update ISIN for existing stock"""
        index_path = self.get_index_path(category)
        if not os.path.exists(index_path):
            print(f"Index file not found for {category}")
            return
        
        df = pd.read_csv(index_path)
        # Ensure ISIN column exists for backward compatibility
        if 'isin' not in df.columns:
            df['isin'] = ''
        # Ensure ISIN column is string type
        df['isin'] = df['isin'].astype(str)
        
        # Find and update the stock
        mask = df['symbol'].str.upper() == symbol.upper()
        if mask.any():
            df.loc[mask, 'isin'] = isin
            # Validate no deletion before saving
            self.validate_no_deletion(category, df)
            df.to_csv(index_path, index=False)
            print(f"Updated ISIN for {symbol}: {isin}")
        else:
            print(f"Stock {symbol} not found in {category} index")
    
    def get_stock_count(self, category: str) -> int:
        """Get the number of stocks in the dynamic index"""
        index_path = self.get_index_path(category)
        if not os.path.exists(index_path):
            return 0
        try:
            df = pd.read_csv(index_path)
            return len(df)
        except Exception as e:
            print(f"Error reading {category} index: {e}")
            return 0
    
    def initialize_from_permanent(self, category: str) -> bool:
        """
        Initialize dynamic index from permanent index if dynamic is empty or corrupt.
        
        Args:
            category: 'us_stocks' or 'ind_stocks'
            
        Returns:
            True if initialization was successful or not needed
        """
        index_path = self.get_index_path(category)
        
        # Check if dynamic index exists and has reasonable number of stocks
        current_count = self.get_stock_count(category)
        min_expected = 400  # Minimum expected stocks (should be 500+)
        
        if current_count >= min_expected:
            print(f"Dynamic {category} index already has {current_count} stocks, no initialization needed")
            return True
        
        print(f"Dynamic {category} index has only {current_count} stocks, initializing from permanent index...")
        
        # Determine permanent index path
        if category == 'us_stocks':
            permanent_path = os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), 
                                        'permanent', 'us_stocks', 'index_us_stocks.csv')
            currency = 'USD'
        elif category == 'ind_stocks':
            permanent_path = os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), 
                                        'permanent', 'ind_stocks', 'index_ind_stocks.csv')
            currency = 'INR'
        else:
            print(f"Unknown category: {category}")
            return False
        
        # Check if permanent index exists
        if not os.path.exists(permanent_path):
            print(f"Permanent {category} index not found: {permanent_path}")
            return False
        
        try:
            # Read permanent index
            df_permanent = pd.read_csv(permanent_path)
            print(f"Found {len(df_permanent)} stocks in permanent {category} index")
            
            # Ensure required columns exist
            required_cols = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange']
            for col in required_cols:
                if col not in df_permanent.columns:
                    print(f"Missing required column in permanent index: {col}")
                    return False
            
            # Add missing columns for dynamic index
            df_permanent['currency'] = currency
            df_permanent['isin'] = ''  # Empty string, not 'nan'
            
            # Reorder columns to match dynamic index format
            dynamic_columns = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange', 'currency', 'isin']
            df_dynamic = df_permanent[dynamic_columns]
            
            # Sort alphabetically by symbol
            df_dynamic = df_dynamic.sort_values('symbol').reset_index(drop=True)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # Save dynamic index
            df_dynamic.to_csv(index_path, index=False)
            
            print(f"✅ Successfully initialized {category} dynamic index with {len(df_dynamic)} stocks")
            return True
            
        except Exception as e:
            print(f"Error initializing {category} from permanent: {e}")
            return False
    
    def validate_no_deletion(self, category: str, new_df: pd.DataFrame):
        """Ensure we're not deleting stocks from dynamic index"""
        index_path = self.get_index_path(category)
        if os.path.exists(index_path):
            old_df = pd.read_csv(index_path)
            old_count = len(old_df)
            new_count = len(new_df)
            
            if new_count < old_count:
                # Auto-restore from permanent if validation fails
                print(f"⚠️  SAFETY CHECK: Attempting to reduce {category} from {old_count} to {new_count} stocks!")
                print(f"   This would delete {old_count - new_count} stocks!")
                print(f"   Attempting auto-restore from permanent index...")
                
                if self.initialize_from_permanent(category):
                    print(f"✅ Auto-restore successful, retrying operation...")
                    # Re-read the restored index
                    restored_df = pd.read_csv(index_path)
                    if len(restored_df) >= old_count:
                        print(f"✅ Dynamic index restored with {len(restored_df)} stocks")
                        return  # Allow the operation to continue
                
                # If auto-restore failed, raise error
                raise ValueError(
                    f"SAFETY CHECK FAILED: Attempting to reduce {category} from "
                    f"{old_count} to {new_count} stocks. This would delete stocks! "
                    f"Auto-restore from permanent index failed."
                )