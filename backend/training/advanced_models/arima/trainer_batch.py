#!/usr/bin/env python3
"""
ARIMA BATCH TRAINER

This script trains a unique ARIMA model for EVERY stock in the dataset
and saves each one to the models/arima/ directory.

Usage:
    python backend/training/advanced_models/arima/trainer_batch.py
    python backend/training/advanced_models/arima/trainer_batch.py --max-stocks-per-category 5
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

# --- Setup Logging ---
log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'arima_BATCH_training_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
)
logger = logging.getLogger(__name__)

# --- Main Training Function ---

def train_one_stock(symbol, category, data_loader, config, models_dir):
    """
    Trains and saves an ARIMA model for a single stock.
    Returns True on success, False on failure.
    """
    try:
        logger.info(f"--- Processing {symbol} ({category}) ---")
        
        # 1. Load data for ONE stock
        df = data_loader.load_stock_data(symbol, category)
        
        if df is None or len(df) < 252: # Need at least 1 year
            logger.warning(f"Insufficient data for {symbol} ({len(df) if df is not None else 0} days). Skipping.")
            return False, "Insufficient data"
            
        # 2. Get the 'y' (target) series
        y_series = df['close'].values
        X_dummy = np.empty((len(y_series), 1)) # Dummy X

        # 3. Train the model
        params = config.get_model_params()
        params['auto_arima'] = True
        params['max_search_time'] = 300 # 5 min timeout per stock

        model = ARIMAModel(**params)
        model.fit(X_dummy, y_series)
        
        # 4. Save the model
        model_path = os.path.join(models_dir, f"{symbol}_model.pkl")
        model.save(model_path)
        
        metrics = model.get_training_metrics()
        train_r2 = metrics.get('r2_score', 0)
        order = metrics.get('order', 'N/A')
        
        logger.info(f"SUCCESS: Saved {symbol} model. R²={train_r2:.4f}, Order={order}")
        return True, f"R²={train_r2:.4f}"

    except Exception as e:
        logger.error(f"FAILED for {symbol}: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Batch train ARIMA models for all stocks.')
    parser.add_argument('--max-stocks-per-category', type=int, help='Max stocks per category (for testing)')
    args = parser.parse_args()

    config = ARIMAConfig()
    data_loader = DataLoader()
    
    # Path should go to backend/models/arima/ (3 levels up from this script, then into models)
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', config.MODEL_NAME)
    os.makedirs(models_dir, exist_ok=True)

    logger.info("="*80)
    logger.info("STARTING ARIMA BATCH TRAINING")
    logger.info(f"Models will be saved in: {models_dir}")
    logger.info("This will take many hours. Grab a coffee.")
    logger.info("="*80)

    stock_categories = ['us_stocks', 'ind_stocks']
    total_success = 0
    total_failed = 0
    start_time = time.time()

    for category in stock_categories:
        logger.info(f"\n--- Starting Category: {category} ---")
        symbols = data_loader.get_stock_symbols(category)
        
        if args.max_stocks_per_category:
            logger.warning(f"Limiting to {args.max_stocks_per_category} stocks for this category.")
            symbols = symbols[:args.max_stocks_per_category]

        for i, symbol in enumerate(symbols):
            logger.info(f"--- Batch {i+1}/{len(symbols)} ---")
            success, msg = train_one_stock(symbol, category, data_loader, config, models_dir)
            
            if success:
                total_success += 1
            else:
                total_failed += 1

    end_time = time.time()
    total_duration_min = (end_time - start_time) / 60
    
    logger.info("="*80)
    logger.info("ARIMA BATCH TRAINING COMPLETE")
    logger.info(f"Total Time: {total_duration_min:.2f} minutes")
    logger.info(f"Successfully trained: {total_success} models")
    logger.info(f"Failed to train: {total_failed} models")
    logger.info("="*80)

if __name__ == "__main__":
    main()

