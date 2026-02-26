#!/usr/bin/env python3
"""
Autoencoder Model Standalone Trainer

Standalone training script for Autoencoder model.

Usage:
    python backend/training/advanced_models/autoencoder/trainer.py
    python backend/training/advanced_models/autoencoder/trainer.py --max-stocks 100
"""

import os
import sys
import logging
import time
import argparse
import numpy as np
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from algorithms.optimised.autoencoders.autoencoder import AutoencoderModel
from training.advanced_models.autoencoder.config import AutoencoderConfig
from training.display_manager import DisplayManager
from training.common_trainer_utils import CommonTrainerMixin

logger = logging.getLogger(__name__)


class AutoencoderTrainer(CommonTrainerMixin):
    """Standalone trainer for Autoencoder model."""
    
    def __init__(self):
        super().__init__()
        self.config = AutoencoderConfig()
        self.model = None
        self.validation_stocks = self.load_validation_stocks()
    
    def train_model(self, X: np.ndarray, y: np.ndarray, display_manager=None) -> AutoencoderModel:
        """Train the Autoencoder model."""
        logger.info(f"Starting {self.config.MODEL_DISPLAY_NAME} training...")
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        params = self.config.get_model_params()
        self.model = AutoencoderModel(**params)
        logger.info(f"Created {self.config.MODEL_DISPLAY_NAME} model")
        
        print(f"\n{self.config.MODEL_DISPLAY_NAME} training in progress...")
        print(f"This may take ~{self.config.EXPECTED_TRAINING_MINUTES} minutes...")
        if self.config.VERBOSE:
            print(f"Model supports verbose output - detailed progress will be shown below...")
        
        start_time = time.time()
        
        # Use progress wrapper for training updates
        self.train_with_progress(self.model, X, y, display_manager, update_interval=30)
        
        training_duration = time.time() - start_time
        
        logger.info(f"{self.config.MODEL_DISPLAY_NAME} training completed in {training_duration:.2f} seconds")
        return self.model
    
    def save_model(self, model: AutoencoderModel) -> str:
        """Save the trained model."""
        model_subdir = os.path.join(self.models_dir, self.config.MODEL_NAME)
        os.makedirs(model_subdir, exist_ok=True)
        
        model_path = os.path.join(model_subdir, f"{self.config.MODEL_NAME}_model.pkl")
        model.save(model_path)
        
        # --- FIX: Calculate total size of all 3 files ---
        try:
            total_size = 0
            total_size += os.path.getsize(f"{model_path}_autoencoder.h5")
            total_size += os.path.getsize(f"{model_path}_encoder.h5")
            total_size += os.path.getsize(f"{model_path}_metadata.pkl")
            file_size_mb = total_size / (1024 * 1024)
            logger.info(f"Model saved to {model_path} (3 files, {file_size_mb:.2f} MB total)")
        except FileNotFoundError:
            logger.warning(f"Could not calculate total size for {model_path}. Main metadata file not found.")
            file_size_mb = 0
        # --- END FIX ---
        
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
            
            print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
            
            training_start = time.time()
            model = self.train_model(X, y, display_manager)
            training_duration = time.time() - training_start
            
            model_path = self.save_model(model)
            file_size_mb = sum([
                os.path.getsize(f"{model_path}_autoencoder.h5"),
                os.path.getsize(f"{model_path}_encoder.h5"),
                os.path.getsize(f"{model_path}_metadata.pkl")
            ]) / (1024 * 1024)
            
            print(f"\nRunning validation...")
            validation_metrics = self.validate_model_generic(model, self.config.MODEL_DISPLAY_NAME, self.validation_stocks)
            
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


def setup_logging():
    """Setup logging configuration."""
    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'autoencoder_training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main function to run Autoencoder training."""
    parser = argparse.ArgumentParser(description='Train Autoencoder model on full dataset')
    parser.add_argument('--max-stocks', type=int, help='Maximum number of stocks to use (for testing)')
    parser.add_argument('--force-retrain', action='store_true', help='Force retrain even if model exists')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("AUTOENCODER MODEL TRAINING")
        logger.info("="*80)
        
        trainer = AutoencoderTrainer()
        success = trainer.run_training(max_stocks=args.max_stocks, force_retrain=args.force_retrain)
        
        sys.exit(0 if success else 1)
            
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

