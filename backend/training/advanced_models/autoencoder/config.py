"""
Autoencoder Model Configuration

Configuration settings and hyperparameters for Autoencoder model.
"""

from typing import Dict, Any


class AutoencoderConfig:
    """Configuration for Autoencoder model training."""
    
    # Model Information
    MODEL_NAME = 'autoencoder'
    MODEL_DISPLAY_NAME = 'Autoencoder'
    MODEL_CLASS = 'AutoencoderModel'
    MODEL_FILE = 'autoencoder.py'
    MODEL_DIR = 'autoencoders'  # Custom directory name in algorithms/optimised
    
    # Training Parameters
    EXPECTED_TRAINING_MINUTES = 18
    STATUS_MESSAGE = 'Training autoencoder with encoding/decoding'
    
    # Model supports verbose output
    VERBOSE = True
    
    # Hyperparameters
    ENCODING_DIM = 16
    HIDDEN_LAYERS = [64, 32]
    ACTIVATION = 'relu'
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'encoding_dim': cls.ENCODING_DIM,
            'hidden_layers': cls.HIDDEN_LAYERS,
            'activation': cls.ACTIVATION,
            'dropout_rate': cls.DROPOUT_RATE,
            'learning_rate': cls.LEARNING_RATE,
            'batch_size': cls.BATCH_SIZE,
            'epochs': cls.EPOCHS,
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

