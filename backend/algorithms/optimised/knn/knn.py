"""
K-Nearest Neighbors for Stock Price Prediction

Optimized implementation using scikit-learn for stock price prediction
based on OHLCV data and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
import os
import joblib
import logging
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ...model_interface import ModelInterface
from ...stock_indicators import StockIndicators


class KNNModel(ModelInterface):
    """
    K-Nearest Neighbors model for stock price prediction.
    
    Uses technical indicators calculated from OHLC data to predict
    future stock prices. Volume is excluded from all calculations.
    """
    
    def __init__(self, n_neighbors: int = 15, weights: str = 'distance',
                 algorithm: str = 'ball_tree', metric: str = 'minkowski',
                 p: int = 2, max_samples: int = 50000, **kwargs):
        super().__init__('K-Nearest Neighbors', **kwargs)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Model parameters
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.max_samples = max_samples  # Maximum samples to store (KNN stores all training data)
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the KNN model on stock data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - stock prices
            
        Returns:
            self: Returns self for method chaining
        """
        self.validate_input(X, y)
        
        # Enhanced data cleaning for infinity and NaN values
        # Replace infinity values with large finite numbers
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y_clean = np.nan_to_num(y, nan=np.nanmean(y), posinf=np.nanmax(y), neginf=np.nanmin(y))
        
        # Additional outlier detection and removal
        # Remove extreme outliers that might cause issues
        if len(X_clean) > 0:
            # Check for extreme values in features
            feature_std = np.std(X_clean, axis=0)
            feature_mean = np.mean(X_clean, axis=0)
            
            # Remove samples with features beyond 10 standard deviations
            outlier_mask = np.all(np.abs(X_clean - feature_mean) <= 10 * feature_std, axis=1)
            X_clean = X_clean[outlier_mask]
            y_clean = y_clean[outlier_mask]
            
            # Final check for any remaining infinity values
            if np.any(np.isinf(X_clean)) or np.any(np.isinf(y_clean)):
                X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=1e6, neginf=-1e6)
                y_clean = np.nan_to_num(y_clean, nan=np.nanmean(y_clean), posinf=np.nanmax(y_clean), neginf=np.nanmin(y_clean))
        
        # Remove any rows where y is still NaN after cleaning
        valid_mask = ~np.isnan(y_clean)
        X_clean = X_clean[valid_mask]
        y_clean = y_clean[valid_mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid data points after cleaning infinity and NaN values")
        
        # Subsample for large datasets (KNN stores all training data in memory)
        original_size = len(X_clean)
        if original_size > self.max_samples:
            logger.info(f"Subsampling from {original_size:,} to {self.max_samples:,} samples for KNN efficiency")
            indices = np.random.choice(original_size, self.max_samples, replace=False)
            X_clean = X_clean[indices]
            y_clean = y_clean[indices]
        
        # Initialize model
        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric,
            p=self.p,
            n_jobs=-1
        )
        
        # Scale features (important for KNN)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train model
        self.model.fit(X_scaled, y_clean)
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        mse = mean_squared_error(y_clean, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_clean, y_pred)
        mae = mean_absolute_error(y_clean, y_pred)
        
        self.set_training_metrics({
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae
        })
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new stock data.
        
        Args:
            X: Features to predict on (n_samples, n_features)
            
        Returns:
            predictions: Predicted stock prices (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.validate_input(X)
        
        # Handle infinity and NaN values in X
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        X_scaled = self.scaler.transform(X_clean)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_with_distances(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with distance information.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (predictions, distances_to_neighbors)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.validate_input(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get distances to neighbors
        distances, indices = self.model.kneighbors(X_scaled)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions, distances
    
    def find_optimal_k(self, X: np.ndarray, y: np.ndarray, 
                       k_range: List[int] = None, cv: int = 5) -> int:
        """
        Find optimal number of neighbors using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            k_range: Range of k values to test
            cv: Number of cross-validation folds
            
        Returns:
            Optimal k value
        """
        if k_range is None:
            k_range = list(range(1, min(21, len(X) // 2), 2))
        
        self.validate_input(X, y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        best_k = k_range[0]
        best_score = -np.inf
        
        for k in k_range:
            knn = KNeighborsRegressor(n_neighbors=k, weights=self.weights)
            scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        
        return best_k
    
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: File path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'metrics': self.training_metrics,
            'params': self.model_params,
            'feature_columns': self.feature_columns
        }, path)
    
    def load(self, path: str) -> 'ModelInterface':
        """
        Load a previously saved model from disk.
        
        Args:
            path: File path to load the model from
            
        Returns:
            self: Returns self for method chaining
        """
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.training_metrics = data['metrics']
        self.model_params = data['params']
        self.feature_columns = data.get('feature_columns')
        self.is_trained = True
        return self
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                             cv: int = 5) -> 'ModelInterface':
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            
        Returns:
            self: Returns self for method chaining
        """
        self.validate_input(X, y)
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': list(range(3, 21, 2)),
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'euclidean', 'manhattan'],
            'p': [1, 2]
        }
        
        # Initialize base model
        base_model = KNeighborsRegressor(n_jobs=-1)
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit grid search
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        self.set_training_metrics({
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'best_params': grid_search.best_params_,
            'cv_score': -grid_search.best_score_
        })
        
        return self


# Example usage and testing
if __name__ == "__main__":
    # Create sample stock data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Generate synthetic OHLC data
    base_price = 100
    returns = np.random.normal(0, 0.02, 200)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices
    })
    
    # Ensure high >= low and high >= close >= low
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    # Create model
    model = KNNModel(n_neighbors=5, weights='distance')
    
    # Add technical indicators
    df_with_features = model._create_technical_indicators(df)
    
    # Prepare training data
    X, y = StockIndicators.prepare_training_data(df_with_features)
    
    if len(X) > 0:
        # Find optimal k
        optimal_k = model.find_optimal_k(X, y)
        print(f"Optimal k: {optimal_k}")
        
        # Update model with optimal k
        model.n_neighbors = optimal_k
        
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X[-10:])  # Predict last 10 days
        
        print(f"KNN Model Results:")
        print(f"Training R²: {model.training_metrics['r2_score']:.4f}")
        print(f"Training RMSE: {model.training_metrics['rmse']:.4f}")
        print(f"Sample predictions: {predictions[:5]}")
        
        # Test save/load
        model.save('test_knn_model.pkl')
        loaded_model = KNNModel().load('test_knn_model.pkl')
        print(f"Model loaded successfully: {loaded_model.is_trained}")
        
        # Clean up
        os.remove('test_knn_model.pkl')
    else:
        print("Insufficient data for training")
