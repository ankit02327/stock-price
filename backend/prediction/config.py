"""
Prediction Configuration

This module contains all configuration parameters for the stock prediction system.
"""

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import os


class PredictionConfig:
    """
    Configuration class for stock prediction parameters.
    """
    
    def __init__(self):
        # Time horizons for predictions (matching the UI buttons: 1D, 1W, 1M, 1Y, 5Y)
        self.TIME_HORIZONS = {
            '1D': 1,      # 1 day
            '1W': 7,      # 1 week (7 days)
            '1M': 30,     # 1 month (30 days)
            '1Y': 365,    # 1 year (365 days)
            '5Y': 1825    # 5 years (1825 days)
        }
        
        # Model weights for ensemble predictions (updated based on current performance)
        # Only models with R² > 0.8 are included in ensemble until all models are retrained
        self.MODEL_WEIGHTS = {
            'random_forest': 0.40,      # R² = 0.994 (excellent performance)
            'decision_tree': 0.30,      # R² = 0.85 (good performance)
            'linear_regression': 0.30,  # Expected R² > 0.85 after retraining
            # Models excluded from ensemble until retraining completes
            'svm': 0.0,                 # Will be included after fixing and retraining
            'knn': 0.0,                 # Will be included after fixing and retraining
            'ann': 0.0,                 # Will be included after fixing and retraining
            'cnn': 0.0,                 # Will be included after fixing and retraining
            'arima': 0.0,               # Will be included after fixing and retraining
            'autoencoder': 0.0          # Will be included after fixing and retraining
        }
        
        # Data configuration
        self.MIN_TRAINING_DAYS = 252  # Minimum 1 year of data for training
        self.LOOKBACK_DAYS = 60       # Number of days to look back for features
        self.TEST_SIZE = 0.2          # 20% of data for testing
        
        # Feature engineering parameters
        self.MOVING_AVERAGES = [5, 10, 20, 50, 200]  # Moving average windows
        self.RSI_PERIOD = 14          # RSI calculation period
        self.VOLATILITY_WINDOW = 20   # Volatility calculation window
        
        # Model-specific parameters
        self.RIDGE_ALPHA_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
        self.LASSO_ALPHA_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
        self.ELASTICNET_ALPHA_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
        self.ELASTICNET_L1_RATIO_VALUES = [0.1, 0.5, 0.7, 0.9]
        
        self.RANDOM_FOREST_PARAMS = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        self.KNN_PARAMS = {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        }
        
        self.SVR_PARAMS = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        self.LSTM_PARAMS = {
            'units': [50, 100, 150],
            'dropout': [0.1, 0.2, 0.3],
            'recurrent_dropout': [0.1, 0.2, 0.3],
            'epochs': [50, 100, 150],
            'batch_size': [16, 32, 64]
        }
        
        self.ARIMA_PARAMS = {
            'p': [1, 2, 3, 4, 5],
            'd': [0, 1, 2],
            'q': [1, 2, 3, 4, 5]
        }
        
        # Confidence interval parameters
        self.CONFIDENCE_LEVEL = 0.95  # 95% confidence interval
        self.CONFIDENCE_MULTIPLIER = 1.96  # For 95% CI (normal distribution)
        
        # File paths - use absolute paths to avoid issues with working directory
        current_file = os.path.abspath(__file__)
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.PAST_DATA_DIR = os.path.join(self.DATA_DIR, 'past')  # Used for TRAINING (5 years historical)
        self.LATEST_DATA_DIR = os.path.join(self.DATA_DIR, 'latest')  # Not used for training; for live predictions
        self.FUTURE_DATA_DIR = os.path.join(self.DATA_DIR, 'future')  # Output predictions
        self.PERMANENT_DATA_DIR = os.path.join(self.BASE_DIR, 'permanent')  # Fallback data for offline mode
        
        # Index files
        self.US_INDEX_FILE = os.path.join(self.DATA_DIR, 'index_us_stocks_dynamic.csv')
        self.IND_INDEX_FILE = os.path.join(self.DATA_DIR, 'index_ind_stocks_dynamic.csv')
        
        # Output configuration
        self.OUTPUT_COLUMNS = [
            'date',
            'horizon', 
            'predicted_price',
            'current_price',  # Base price used for prediction
            'confidence',  # Overall confidence score (0-100)
            'confidence_low',
            'confidence_high',
            'algorithm_used',
            'currency',
            'last_updated',
            'model_accuracy',
            'data_points_used'
        ]
        
        # Logging configuration
        self.LOG_LEVEL = 'INFO'
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Performance settings
        self.MAX_WORKERS = 4  # For parallel processing
        self.CHUNK_SIZE = 10  # Process stocks in chunks
        
        # Display settings
        self.UPDATE_INTERVAL = 20  # Update every 20 seconds
        self.ENABLE_EMOJIS = True  # Try emojis (auto-fallback if terminal breaks)
        self.SHOW_SAMPLE_STOCKS = 8  # Show first N stocks in progress summaries
        
        # Validation settings
        self.MIN_PRICE = 0.01  # Minimum valid price
        self.MAX_PRICE = 100000  # Maximum valid price
        self.MAX_VOLATILITY = 5.0  # Maximum daily volatility (500%)
        
    def get_time_horizon_days(self, horizon: str) -> int:
        """Get number of days for a given time horizon."""
        return self.TIME_HORIZONS.get(horizon, 1)
    
    def get_model_weight(self, model_name: str) -> float:
        """Get weight for a specific model."""
        return self.MODEL_WEIGHTS.get(model_name, 0.1)
    
    def get_prediction_date(self, horizon: str, base_date: datetime = None) -> datetime:
        """Get the prediction date for a given horizon."""
        if base_date is None:
            base_date = datetime.now()
        
        days = self.get_time_horizon_days(horizon)
        return base_date + timedelta(days=days)
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        # Check if weights sum to 1.0
        total_weight = sum(self.MODEL_WEIGHTS.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Model weights must sum to 1.0, got {total_weight}")
        
        # Check if directories exist
        required_dirs = [self.DATA_DIR, self.PAST_DATA_DIR, self.LATEST_DATA_DIR]
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'time_horizons': self.TIME_HORIZONS,
            'model_weights': self.MODEL_WEIGHTS,
            'min_training_days': self.MIN_TRAINING_DAYS,
            'lookback_days': self.LOOKBACK_DAYS,
            'test_size': self.TEST_SIZE,
            'moving_averages': self.MOVING_AVERAGES,
            'rsi_period': self.RSI_PERIOD,
            'volatility_window': self.VOLATILITY_WINDOW,
            'confidence_level': self.CONFIDENCE_LEVEL,
            'max_workers': self.MAX_WORKERS,
            'chunk_size': self.CHUNK_SIZE
        }


# Global configuration instance
config = PredictionConfig()
