"""
Common Trainer Utilities

Shared utilities for all model trainers.
"""

import os
import sys
import json
import logging
import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Any

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prediction.data_loader import DataLoader
from prediction.config import config

logger = logging.getLogger(__name__)


class CommonTrainerMixin:
    """Mixin class providing common functionality for all trainers."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self._stop_progress = False
        self._progress_display_manager = None
        
    def load_validation_stocks(self) -> Dict[str, List[str]]:
        """Load validation stocks from JSON file."""
        validation_file = os.path.join(os.path.dirname(__file__), 'validation_stocks.json')
        try:
            with open(validation_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading validation stocks: {e}")
            return {'us_stocks': [], 'ind_stocks': []}
    
    def _progress_updater(self, start_time: float, update_interval: int = 30):
        """
        Background thread function to display progress updates.
        
        Args:
            start_time: Training start time (from time.time())
            update_interval: Update interval in seconds
        """
        while not self._stop_progress:
            time.sleep(update_interval)
            if not self._stop_progress and self._progress_display_manager:
                elapsed = time.time() - start_time
                self._progress_display_manager.show_training_progress(elapsed)
    
    def train_with_progress(self, model, X: np.ndarray, y: np.ndarray, 
                           display_manager=None, update_interval: int = 30):
        """
        Train a model with periodic progress updates.
        
        Args:
            model: Model instance to train
            X: Training features
            y: Training target
            display_manager: DisplayManager instance for showing updates
            update_interval: Update interval in seconds (default: 30)
            
        Returns:
            Trained model
        """
        self._progress_display_manager = display_manager
        self._stop_progress = False
        
        # Start progress update thread
        start_time = time.time()
        if display_manager:
            progress_thread = threading.Thread(
                target=self._progress_updater,
                args=(start_time, update_interval),
                daemon=True
            )
            progress_thread.start()
        
        try:
            # Train the model (this is the blocking call)
            model.fit(X, y)
        finally:
            # Stop progress updates
            self._stop_progress = True
            if display_manager:
                # Give thread time to exit gracefully
                time.sleep(0.1)
        
        return model
    
    def load_all_stock_data(self, max_stocks: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and combine data from all stocks for training."""
        logger.info("Loading data from all stocks...")
        
        all_X = []
        all_y = []
        stock_symbols = []
        
        categories = ['us_stocks', 'ind_stocks']
        
        for category in categories:
            print(f"\n{'='*80}")
            print(f"Loading {category.upper()} data...")
            print(f"{'='*80}")
            logger.info(f"Loading {category} data...")
            symbols = self.data_loader.get_stock_symbols(category)
            print(f"Found {len(symbols)} symbols in {category}")
            
            if max_stocks and len(symbols) > max_stocks:
                symbols = symbols[:max_stocks]
                logger.info(f"Limited to {max_stocks} stocks for testing")
                print(f"Limited to {max_stocks} stocks for testing")
            
            for i, symbol in enumerate(symbols):
                try:
                    if (i + 1) % 50 == 0:
                        logger.info(f"Processed {i + 1}/{len(symbols)} stocks in {category}")
                        print(f"  Processed {i + 1}/{len(symbols)} stocks in {category}")
                    
                    df = self.data_loader.load_stock_data(symbol, category)
                    if df is None or len(df) < config.MIN_TRAINING_DAYS:
                        continue
                    
                    if not self.data_loader.validate_data_quality(df, symbol):
                        continue
                    
                    # Add market_type feature: 0 for US stocks, 1 for Indian stocks
                    df['market_type'] = 0 if category == 'us_stocks' else 1
                    
                    df_with_features = self.data_loader.create_features(df)
                    if df_with_features is None or len(df_with_features) == 0:
                        continue
                    
                    X, y = self.data_loader.prepare_training_data(df_with_features)
                    if len(X) == 0 or len(y) == 0:
                        continue
                    
                    all_X.append(X)
                    all_y.append(y)
                    stock_symbols.append(f"{symbol}_{category}")
                    
                except Exception as e:
                    logger.warning(f"Error processing {symbol} in {category}: {e}")
                    continue
            
            # Print category summary
            stocks_loaded_this_category = sum(1 for s in stock_symbols if s.endswith(f"_{category}"))
            print(f"\n  Completed {category}: {stocks_loaded_this_category} stocks successfully loaded")
            print(f"{'='*80}\n")
        
        if not all_X:
            logger.error("No valid stock data found")
            return np.array([]), np.array([]), []
        
        logger.info(f"Combining data from {len(all_X)} stocks...")
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        logger.info(f"Combined dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
        return X_combined, y_combined, stock_symbols
    
    def validate_model_generic(self, model, model_name: str, validation_stocks: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generic validation function for any model."""
        logger.info(f"Validating {model_name} on validation stocks...")
        
        validation_results = []
        
        for category, symbols in validation_stocks.items():
            for symbol in symbols:
                try:
                    df = self.data_loader.load_stock_data(symbol, category)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Add market_type feature to match training data (38 features)
                    df['market_type'] = 0 if category == 'us_stocks' else 1
                    
                    df_with_features = self.data_loader.create_features(df)
                    if df_with_features is None or len(df_with_features) == 0:
                        continue
                    
                    X, y = self.data_loader.prepare_training_data(df_with_features)
                    if len(X) == 0 or len(y) == 0:
                        continue
                    
                    split_idx = int(len(X) * 0.8)
                    X_test = X[split_idx:]
                    y_test = y[split_idx:]
                    
                    if len(X_test) == 0:
                        continue
                    
                    y_pred = model.predict(X_test)
                    
                    # Handle shape mismatch
                    if len(y_pred.shape) > 1:
                        y_pred = y_pred.flatten()
                    
                    min_len = min(len(y_test), len(y_pred))
                    if min_len == 0:
                        continue
                    
                    y_test = y_test[:min_len]
                    y_pred = y_pred[:min_len]
                    
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    validation_results.append({
                        'symbol': symbol,
                        'category': category,
                        'r2_score': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'samples': len(X_test)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error validating {symbol}: {e}")
                    continue
        
        if not validation_results:
            return {'test_stocks': 0, 'avg_r2_score': 0.0, 'avg_rmse': 0.0, 'avg_mae': 0.0}
        
        avg_r2 = np.mean([r['r2_score'] for r in validation_results])
        avg_rmse = np.mean([r['rmse'] for r in validation_results])
        avg_mae = np.mean([r['mae'] for r in validation_results])
        
        logger.info(f"Validation: R²={avg_r2:.4f}, RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}")
        
        return {
            'test_stocks': len(validation_results),
            'avg_r2_score': float(avg_r2),
            'avg_rmse': float(avg_rmse),
            'avg_mae': float(avg_mae),
            'results': validation_results
        }

