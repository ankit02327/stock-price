#!/usr/bin/env python3
"""
SVM Model Standalone Trainer

Standalone training script for Support Vector Machine model.
This script can be run directly to train the SVM model on the full dataset.

Usage:
    python backend/training/basic_models/svm/trainer.py
    python backend/training/basic_models/svm/trainer.py --max-stocks 100
    python backend/training/basic_models/svm/trainer.py --force-retrain
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

from algorithms.optimised.svm.svm import SVMModel
from training.basic_models.svm.config import SVMConfig
from training.display_manager import DisplayManager
from training.common_trainer_utils import CommonTrainerMixin

logger = logging.getLogger(__name__)


class SVMTrainer(CommonTrainerMixin):
    """Standalone trainer for SVM model."""
    
    def __init__(self):
        super().__init__()
        self.config = SVMConfig()
        self.model = None
        self.validation_stocks = self.load_validation_stocks()
    
    def train_model(self, X: np.ndarray, y: np.ndarray, display_manager=None) -> SVMModel:
        """Train the SVM model."""
        logger.info(f"Starting {self.config.MODEL_DISPLAY_NAME} training...")
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Create model
        params = self.config.get_model_params()
        self.model = SVMModel(**params)
        logger.info(f"Created {self.config.MODEL_DISPLAY_NAME} model with params: {params}")
        
        # Track training time
        start_time = time.time()
        
        # Use progress wrapper for training updates
        self.train_with_progress(self.model, X, y, display_manager, update_interval=30)
        
        training_duration = time.time() - start_time
        
        logger.info(f"{self.config.MODEL_DISPLAY_NAME} training completed in {training_duration:.2f} seconds")
        
        return self.model
    
    def save_model(self, model: SVMModel) -> str:
        """Save the trained model."""
        # Create model-specific subdirectory
        model_subdir = os.path.join(self.models_dir, self.config.MODEL_NAME)
        os.makedirs(model_subdir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_subdir, f"{self.config.MODEL_NAME}_model.pkl")
        model.save(model_path)
        
        # Get file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model saved to {model_path} ({file_size_mb:.2f} MB)")
        
        return model_path
    
    def run_training(self, max_stocks: int = None, force_retrain: bool = False):
        """Run the complete training pipeline."""
        training_info = self.config.get_training_info()
        
        # Initialize display manager
        display_manager = DisplayManager(
            model_name=self.config.MODEL_NAME,
            update_interval=30,
            enable_emojis=True
        )
        
        # Show startup message
        display_manager.show_training_start(
            total_stocks=1000 if not max_stocks else max_stocks,
            expected_duration_min=training_info['expected_minutes']
        )
        
        try:
            # Load data
            print(f"\nLoading full dataset for {self.config.MODEL_DISPLAY_NAME}...")
            start_time = time.time()
            
            X, y, stock_symbols = self.load_all_stock_data(max_stocks)
            
            if len(X) == 0:
                raise ValueError("No training data available")
            
            data_loading_time = time.time() - start_time
            print(f"Data loading completed in {data_loading_time:.2f} seconds")
            
            # Train model
            training_start = time.time()
            model = self.train_model(X, y, display_manager)
            training_duration = time.time() - training_start
            
            # Save model
            model_path = self.save_model(model)
            
            # Validate model
            print(f"\nRunning validation on test stocks...")
            validation_metrics = self.validate_model_generic(model, self.config.MODEL_DISPLAY_NAME, self.validation_stocks)
            
            # Display completion summary
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


def setup_logging():
    """Setup logging configuration."""
    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'svm_training_{timestamp}.log')
    
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
    """Main function to run SVM training."""
    parser = argparse.ArgumentParser(description='Train SVM model on full dataset')
    
    parser.add_argument('--max-stocks', 
                       type=int, 
                       help='Maximum number of stocks to use (for testing)')
    
    parser.add_argument('--force-retrain', 
                       action='store_true',
                       help='Force retrain even if model exists')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("SVM MODEL TRAINING")
        logger.info("="*80)
        
        # Initialize trainer
        trainer = SVMTrainer()
        
        # Run training
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

