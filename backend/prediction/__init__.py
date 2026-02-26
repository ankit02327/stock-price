"""
Stock Prediction Module

This module provides comprehensive stock price prediction capabilities using
multiple machine learning algorithms from the algorithms/real directory.

Components:
- config.py: Configuration for prediction parameters
- data_loader.py: Data loading and preprocessing  
- predictor.py: Main prediction orchestrator
- prediction_saver.py: Save predictions to CSV files
- run_predictions.py: Standalone execution script

Usage:
    python backend/prediction/run_predictions.py
"""

__version__ = "1.0.0"
__author__ = "Stock Prediction System"

# Import only lightweight components eagerly
from .config import PredictionConfig
from .data_loader import DataLoader
from .prediction_saver import PredictionSaver

# DO NOT import StockPredictor here to avoid loading all ML models eagerly
# Import it directly when needed: from prediction.predictor import StockPredictor

__all__ = [
    'PredictionConfig',
    'DataLoader', 
    'PredictionSaver',
    # 'StockPredictor' removed from eager imports
]
