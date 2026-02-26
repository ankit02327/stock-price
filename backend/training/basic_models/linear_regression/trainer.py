#!/usr/bin/env python3
"""
Linear Regression Model Standalone Trainer

Standalone training script for Linear Regression model.
This script can be run directly to train the Linear Regression model on the full dataset.

Usage:
    python backend/training/basic_models/linear_regression/trainer.py
    python backend/training/basic_models/linear_regression/trainer.py --max-stocks 100
    python backend/training/basic_models/linear_regression/trainer.py --force-retrain
"""

import os
import sys
import logging
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from algorithms.optimised.linear_regression.linear_regression import LinearRegressionModel
from training.basic_models.linear_regression.config import LinearRegressionConfig
from training.display_manager import DisplayManager
from training.common_trainer_utils import CommonTrainerMixin

logger = logging.getLogger(__name__)


class LinearRegressionTrainer(CommonTrainerMixin):
    """Standalone trainer for Linear Regression model."""
    
    def __init__(self):
        super().__init__()
        self.config = LinearRegressionConfig()
        self.model = None
        self.validation_stocks = self.load_validation_stocks()
        
    def create_model(self) -> LinearRegressionModel:
        """Create a new Linear Regression model instance."""
        params = self.config.get_model_params()
        self.model = LinearRegressionModel(**params)
        logger.info(f"Created {self.config.MODEL_DISPLAY_NAME} model with params: {params}")
        return self.model
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              progress_callback=None) -> Dict[str, Any]:
        """
        Train the Linear Regression model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training target (n_samples,)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting {self.config.MODEL_DISPLAY_NAME} training...")
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Track training time
        start_time = time.time()
        
        # Train the model
        self.model.fit(X, y)
        
        training_duration = time.time() - start_time
        
        logger.info(f"{self.config.MODEL_DISPLAY_NAME} training completed in {training_duration:.2f} seconds")
        
        # Return training results
        return {
            'model': self.model,
            'training_duration': training_duration,
            'samples_used': len(X),
            'features_count': X.shape[1] if len(X) > 0 else 0
        }
    
    def save_model(self, model: LinearRegressionModel, save_path: str) -> bool:
        """
        Save the trained model.
        
        Args:
            model: Trained model to save
            save_path: Full path where to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model
            model.save(save_path)
            
            # Get file size
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            
            logger.info(f"Model saved to {save_path} ({file_size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path: str) -> LinearRegressionModel:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        try:
            self.model = LinearRegressionModel().load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def train_model(self, X: np.ndarray, y: np.ndarray, display_manager=None) -> LinearRegressionModel:
        """Train the Linear Regression model."""
        logger.info(f"Starting {self.config.MODEL_DISPLAY_NAME} training...")
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        params = self.config.get_model_params()
        self.model = LinearRegressionModel(**params)
        logger.info(f"Created {self.config.MODEL_DISPLAY_NAME} model")
        
        # Print training start information
        print(f"\n{'='*80}")
        print(f"TRAINING STARTED: {self.config.MODEL_DISPLAY_NAME}")
        print(f"{'='*80}")
        print(f"Dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
        print(f"Expected duration: ~{self.config.EXPECTED_TRAINING_MINUTES} minutes")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Use progress wrapper for training updates
        self.train_with_progress(self.model, X, y, display_manager, update_interval=30)
        
        training_duration = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Actual duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
        print(f"{'='*80}\n")
        
        logger.info(f"{self.config.MODEL_DISPLAY_NAME} training completed in {training_duration:.2f} seconds")
        return self.model
    
    def validate_model(self, model: LinearRegressionModel) -> Dict[str, Any]:
        """Validate the trained model on validation stocks."""
        logger.info(f"Validating {self.config.MODEL_DISPLAY_NAME} on validation stocks...")
        
        validation_results = []
        
        for category, symbols in self.validation_stocks.items():
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
    
    def save_model(self, model: LinearRegressionModel) -> str:
        """Save the trained model."""
        model_subdir = os.path.join(self.models_dir, self.config.MODEL_NAME)
        os.makedirs(model_subdir, exist_ok=True)
        
        model_path = os.path.join(model_subdir, f"{self.config.MODEL_NAME}_model.pkl")
        model.save(model_path)
        
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model saved to {model_path} ({file_size_mb:.2f} MB)")
        
        return model_path
    
    def run_training(self, max_stocks: int = None, force_retrain: bool = False):
        """Run the complete training pipeline."""
        training_info = self.config.get_training_info()
        
        display_manager = DisplayManager(
            model_name=self.config.MODEL_NAME,
            update_interval=30,
            enable_emojis=True
        )
        
        display_manager.show_training_start(
            total_stocks=1000 if not max_stocks else max_stocks,
            expected_duration_min=training_info['expected_minutes']
        )
        
        try:
            print(f"\nLoading full dataset for {self.config.MODEL_DISPLAY_NAME}...")
            start_time = time.time()
            
            X, y, stock_symbols = self.load_all_stock_data(max_stocks)
            
            if len(X) == 0:
                raise ValueError("No training data available")
            
            data_loading_time = time.time() - start_time
            print(f"Data loading completed in {data_loading_time:.2f} seconds")
            
            training_start = time.time()
            model = self.train_model(X, y, display_manager)
            training_duration = time.time() - training_start
            
            model_path = self.save_model(model)
            
            print(f"\nRunning validation on test stocks...")
            validation_metrics = self.validate_model(model)
            
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            summary = {
                'model_name': self.config.MODEL_DISPLAY_NAME,
                'file_type': '.pkl',
                'model_path': model_path,
                'file_size_mb': file_size_mb,
                'stocks_processed': len(stock_symbols),
                'total_samples': len(X),
                'total_time': training_duration,
                'validation_r2': validation_metrics.get('avg_r2_score', 0)
            }
            
            display_manager.show_training_complete(summary)
            
            logger.info(f"{self.config.MODEL_DISPLAY_NAME} training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            display_manager.show_error(str(e))
            return False
    
    @staticmethod
    def get_config() -> LinearRegressionConfig:
        """Get the configuration for this model."""
        return LinearRegressionConfig()


def setup_logging():
    """Setup logging configuration."""
    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'linear_regression_training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main function to run Linear Regression training."""
    parser = argparse.ArgumentParser(description='Train Linear Regression model on full dataset')
    
    parser.add_argument('--max-stocks', 
                       type=int, 
                       help='Maximum number of stocks to use (for testing)')
    
    parser.add_argument('--force-retrain', 
                       action='store_true',
                       help='Force retrain even if model exists')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("LINEAR REGRESSION MODEL TRAINING")
        logger.info("="*80)
        
        trainer = LinearRegressionTrainer()
        
        success = trainer.run_training(
            max_stocks=args.max_stocks,
            force_retrain=args.force_retrain
        )
        
        if success:
            logger.info("Training completed successfully")
            sys.exit(0)
        else:
            logger.error("Training failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

