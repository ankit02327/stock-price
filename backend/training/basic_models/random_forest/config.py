"""
Random Forest Model Configuration

Configuration settings and hyperparameters for Random Forest model.
"""

from typing import Dict, Any


class RandomForestConfig:
    """Configuration for Random Forest model training."""
    
    # Model Information
    MODEL_NAME = 'random_forest'
    MODEL_DISPLAY_NAME = 'Random Forest'
    MODEL_CLASS = 'RandomForestModel'
    MODEL_FILE = 'random_forest.py'
    
    # Training Parameters
    EXPECTED_TRAINING_MINUTES = 8
    STATUS_MESSAGE = 'Training random forest with multiple trees'
    
    # Model supports verbose output
    VERBOSE = True
    
    # Hyperparameters
    N_ESTIMATORS = 100
    MAX_DEPTH = 15
    MIN_SAMPLES_SPLIT = 10
    MIN_SAMPLES_LEAF = 5
    MAX_FEATURES = 'sqrt'
    RANDOM_STATE = 42
    N_JOBS = -1  # Use all available cores
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'n_estimators': cls.N_ESTIMATORS,
            'max_depth': cls.MAX_DEPTH,
            'min_samples_split': cls.MIN_SAMPLES_SPLIT,
            'min_samples_leaf': cls.MIN_SAMPLES_LEAF,
            'max_features': cls.MAX_FEATURES,
            'random_state': cls.RANDOM_STATE,
            'n_jobs': cls.N_JOBS,
            'verbose': 1 if cls.VERBOSE else 0
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

