"""
Linear Regression Model Configuration

Configuration settings and hyperparameters for Linear Regression model.
"""

from typing import Dict, Any


class LinearRegressionConfig:
    """Configuration for Linear Regression model training."""
    
    # Model Information
    MODEL_NAME = 'linear_regression'
    MODEL_DISPLAY_NAME = 'Linear Regression'
    MODEL_CLASS = 'LinearRegressionModel'
    MODEL_FILE = 'linear_regression.py'
    
    # Training Parameters
    EXPECTED_TRAINING_MINUTES = 2
    STATUS_MESSAGE = 'Training linear regression (fastest model)'
    
    # Model is not verbose (no detailed progress during training)
    VERBOSE = False
    
    # Hyperparameters
    FIT_INTERCEPT = True
    NORMALIZE = False  # Deprecated in newer sklearn versions
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'fit_intercept': cls.FIT_INTERCEPT,
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

