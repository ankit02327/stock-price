"""
One-time cleanup script to remove stock CSV files not in dynamic index
Run once after implementing the new index system
"""

import os
import pandas as pd
from pathlib import Path

def cleanup_orphaned_files():
    data_dir = Path(__file__).parent.parent.parent / 'data'
    
    # Load dynamic indexes
    us_index = pd.read_csv(data_dir / 'index_us_stocks_dynamic.csv')
    ind_index = pd.read_csv(data_dir / 'index_ind_stocks_dynamic.csv')
    
    us_symbols = set(us_index['symbol'].str.upper())
    ind_symbols = set(ind_index['symbol'].str.upper())
    
    # Check US stocks
    us_dirs = [
        data_dir / 'latest' / 'us_stocks' / 'individual_files',
        data_dir / 'past' / 'us_stocks' / 'individual_files'
    ]
    
    for dir_path in us_dirs:
        if dir_path.exists():
            for file in dir_path.glob('*.csv'):
                symbol = file.stem.upper()
                if symbol not in us_symbols:
                    print(f"Removing orphaned US stock: {file}")
                    file.unlink()
    
    # Check Indian stocks
    ind_dirs = [
        data_dir / 'latest' / 'ind_stocks' / 'individual_files',
        data_dir / 'past' / 'ind_stocks' / 'individual_files'
    ]
    
    for dir_path in ind_dirs:
        if dir_path.exists():
            for file in dir_path.glob('*.csv'):
                symbol = file.stem.upper()
                if symbol not in ind_symbols:
                    print(f"Removing orphaned Indian stock: {file}")
                    file.unlink()

if __name__ == '__main__':
    cleanup_orphaned_files()
    print("Cleanup complete!")
