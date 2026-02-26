#!/usr/bin/env python3
"""
ARIMA Model Standalone Trainer - FOR A SINGLE STOCK

This script trains an ARIMA model on a single, specified stock.
This is the correct way to train ARIMA, as it requires a
continuous time series.

Usage:
    python backend/training/advanced_models/arima/trainer_single_stock.py --symbol AAPL
"""

import os
import sys
import logging
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from algorithms.optimised.arima.arima import ARIMAModel
from training.advanced_models.arima.config import ARIMAConfig
from prediction.data_loader import DataLoader
from training.display_manager import DisplayManager

logger = logging.getLogger(__name__)

def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'arima_single_stock_training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
    )
    return logging.getLogger(__name__)

def run_single_stock_training(symbol: str, category: str):
    """Run the training pipeline for one stock."""
    
    config = ARIMAConfig()
    display_manager = DisplayManager(model_name=config.MODEL_NAME)
    
    try:
        print(f"\nLoading data for {symbol} ({category})...")
        data_loader = DataLoader()
        
        # 1. Load data for ONE stock
        # This data is already currency-normalized (USD)
        df = data_loader.load_stock_data(symbol, category)
        
        if df is None or len(df) < 252: # Need at least 1 year of data
            print(f"Insufficient data for {symbol}. Skipping.")
            return False
            
        # 2. Get the 'y' (target) series. ARIMA uses the 'close' price.
        # We use 'close' (price) not 'pct_change' for ARIMA,
        # as the 'd' (differencing) parameter handles this.
        y_series = df['close'].values
        # We pass a dummy 'X' because the interface requires it.
        X_dummy = np.empty((len(y_series), 1)) 

        print(f"Data loaded: {len(y_series)} samples.")
        
        # 3. Train the model
        training_start = time.time()
        params = config.get_model_params()
        # Manually enable auto_arima
        params['auto_arima'] = True
        params['max_search_time'] = 300 # 5 min timeout
        
        model = ARIMAModel(**params)
        
        print("Starting ARIMA model.fit()... (This may take several minutes for auto_arima)")
        model.fit(X_dummy, y_series) # Pass dummy X, real y
        
        training_duration = time.time() - training_start
        
        # 4. Save the model
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        model_subdir = os.path.join(models_dir, config.MODEL_NAME)
        os.makedirs(model_subdir, exist_ok=True)
        
        # We must save this model specific to the stock
        model_path = os.path.join(model_subdir, f"{symbol}_model.pkl")
        model.save(model_path)
        
        # 5. Display results
        metrics = model.get_training_metrics()
        train_r2 = metrics.get('r2_score', 0)
        
        print(f"\n--- [METRIC] Training R² (In-Sample): {train_r2:.4f} ---")
        print(f"--- [METRIC] Optimal Order (p,d,q): {metrics.get('order')} ---")

        summary = {
            'model_name': f"{config.MODEL_DISPLAY_NAME} ({symbol})",
            'file_type': '.pkl',
            'model_path': model_path,
            'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'stocks_processed': 1,
            'total_samples': len(y_series),
            'total_time': training_duration,
            'validation_r2': train_r2 # Using train_r2 as validation for this test
        }
        
        display_manager.show_training_complete(summary)
        return True

    except Exception as e:
        logger.error(f"Error during training for {symbol}: {e}", exc_info=True)
        display_manager.show_error(str(e))
        return False

def main():
    parser = argparse.ArgumentParser(description='Train ARIMA model on a single stock.')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to train on (e.g., AAPL)')
    parser.add_argument('--category', type=str, default='us_stocks', help='Stock category (us_stocks or ind_stocks)')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info(f"ARIMA MODEL SINGLE STOCK TRAINING: {args.symbol}")
        logger.info("="*80)
        
        success = run_single_stock_training(args.symbol, args.category)
        sys.exit(0 if success else 1)
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

