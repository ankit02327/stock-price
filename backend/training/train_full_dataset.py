#!/usr/bin/env python3
"""
Full Dataset Training Script

This script trains all 7 ML models on the full dataset of 936 stocks with sufficient data
(from 1,001 total available stocks) with 5 years of historical data by calling individual standalone trainers.
Stocks are automatically filtered during data loading based on data quality and sufficiency.

Usage:
    python backend/training/train_full_dataset.py
    python backend/training/train_full_dataset.py --model linear_regression
    python backend/training/train_full_dataset.py --force-retrain
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import subprocess
import time

# Setup logging
def setup_logging():
    """Setup comprehensive logging configuration."""
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'full_dataset_training_{timestamp}.log')
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Model trainer paths
MODEL_TRAINERS = {
    'linear_regression': 'backend/training/basic_models/linear_regression/trainer.py',
    'decision_tree': 'backend/training/basic_models/decision_tree/trainer.py',
    'random_forest': 'backend/training/basic_models/random_forest/trainer.py',
    'svm': 'backend/training/basic_models/svm/trainer.py',
    'knn': 'backend/training/advanced_models/knn/trainer.py',
    'arima': 'backend/training/advanced_models/arima/trainer.py',
    'autoencoder': 'backend/training/advanced_models/autoencoder/trainer.py'
}


def run_standalone_trainer(model_name: str, trainer_path: str, force_retrain: bool = False, logger=None) -> bool:
    """
    Run a standalone trainer script.
    
    Args:
        model_name: Name of the model
        trainer_path: Path to the trainer script
        force_retrain: Whether to force retrain
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    if logger:
        logger.info(f"="*80)
        logger.info(f"Starting {model_name.upper()} training...")
        logger.info(f"="*80)
    
    # Build command
    cmd = [sys.executable, trainer_path]
    if force_retrain:
        cmd.append('--force-retrain')
    
    try:
        # Run the trainer
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        
        if logger:
            logger.info(f"✅ {model_name} training completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"❌ {model_name} training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        if logger:
            logger.error(f"❌ Error running {model_name} trainer: {e}")
        return False


def main():
    """Main function to train all models on full dataset."""
    # Change to project root directory for correct relative paths
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description='Train all ML models on full dataset')
    
    parser.add_argument('--model', 
                       choices=list(MODEL_TRAINERS.keys()),
                       help='Train only a specific model')
    
    parser.add_argument('--force-retrain', 
                       action='store_true',
                       help='Force retrain even if model is completed')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("FULL DATASET TRAINING - STANDALONE TRAINERS")
        logger.info("="*80)
        logger.info("Target: 936 stocks with sufficient data (from 1,001 available)")
        logger.info("Dataset: 5 years of historical data (2020-2024)")
        logger.info(f"Models to train: {len(MODEL_TRAINERS)}")
        logger.info("")
        
        results = {}
        start_time = time.time()
        
        # Train models
        if args.model:
            # Train specific model
            trainer_path = MODEL_TRAINERS[args.model]
            success = run_standalone_trainer(args.model, trainer_path, args.force_retrain, logger)
            results[args.model] = success
        else:
            # Train all models sequentially
            logger.info("Training all models sequentially...")
            logger.info("")
            
            for model_name, trainer_path in MODEL_TRAINERS.items():
                success = run_standalone_trainer(model_name, trainer_path, args.force_retrain, logger)
                results[model_name] = success
                logger.info("")  # Blank line between models
        
        total_time = time.time() - start_time
        
        # Print final results
        print_training_results(results, logger, total_time)
        
        # Exit with appropriate code
        all_successful = all(results.values())
        sys.exit(0 if all_successful else 1)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def print_training_results(results: dict, logger, total_time: float):
    """Print training results summary."""
    print("\n" + "="*100)
    print("FULL DATASET TRAINING RESULTS")
    print("="*100)
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"Successfully trained: {successful}/{total} models")
    print(f"Success rate: {successful/total*100:.1f}%")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    
    print("\nModel Results:")
    print("-" * 100)
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_name:<20} | {status}")
    
    print("="*100)
    
    logger.info(f"Training complete: {successful}/{total} models successful")


if __name__ == "__main__":
    main()
