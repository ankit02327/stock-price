"""
KNN Model Configuration

Configuration settings and hyperparameters for K-Nearest Neighbors model.
"""

from typing import Dict, Any


class KNNConfig:
    """Configuration for KNN model training."""
    
    # Model Information
    MODEL_NAME = 'knn'
    MODEL_DISPLAY_NAME = 'K-Nearest Neighbors (KNN)'
    MODEL_CLASS = 'KNNModel'
    MODEL_FILE = 'knn.py'
    
    # Training Parameters
    EXPECTED_TRAINING_MINUTES = 5
    STATUS_MESSAGE = 'Training KNN with distance calculations'
    
    # Model is not verbose
    VERBOSE = False
    
    # Hyperparameters
    N_NEIGHBORS = 5
    WEIGHTS = 'distance'
    ALGORITHM = 'auto'
    LEAF_SIZE = 30
    P = 2  # Power parameter for Minkowski metric
    N_JOBS = -1  # Use all available cores
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'n_neighbors': cls.N_NEIGHBORS,
            'weights': cls.WEIGHTS,
            'algorithm': cls.ALGORITHM,
            'leaf_size': cls.LEAF_SIZE,
            'p': cls.P,
            'n_jobs': cls.N_JOBS
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

