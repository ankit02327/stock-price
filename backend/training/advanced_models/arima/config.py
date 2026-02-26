"""
ARIMA Model Configuration

Configuration settings and hyperparameters for ARIMA model.
"""

from typing import Dict, Any


class ARIMAConfig:
    """Configuration for ARIMA model training."""
    
    # Model Information
    MODEL_NAME = 'arima'
    MODEL_DISPLAY_NAME = 'ARIMA (AutoRegressive Integrated Moving Average)'
    MODEL_CLASS = 'ARIMAModel'
    MODEL_FILE = 'arima.py'
    
    # Training Parameters
    EXPECTED_TRAINING_MINUTES = 25
    STATUS_MESSAGE = 'Training ARIMA with parameter optimization'
    
    # Model is not verbose
    VERBOSE = False
    
    # Hyperparameters
    ORDER = (5, 1, 0)  # (p, d, q)
    SEASONAL_ORDER = (0, 0, 0, 0)  # (P, D, Q, s)
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'order': cls.ORDER,
            'seasonal_order': cls.SEASONAL_ORDER
        }
    
    @classmethod
    def get_training_info(cls) -> Dict[str, Any]:
        """Get training information for display and logging."""
        return {
            'model_name': cls.MODEL_NAME,
            'display_name': cls.MODEL_DISPLAY_NAME,
            'expected_minutes': cls.EXPECTED_TRAINING_MINUTES,
            'status_message': cls.STATUS_MESSAGE,
            'verbose': cls.VERBOSE
        }

