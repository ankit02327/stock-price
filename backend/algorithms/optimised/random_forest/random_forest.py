"""
Random Forest for Stock Price Prediction

Optimized implementation using scikit-learn for stock price prediction
based on OHLCV data and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ...model_interface import ModelInterface
from ...stock_indicators import StockIndicators


class RandomForestModel(ModelInterface):
    """
    Random Forest model for stock price prediction.
    
    Uses technical indicators calculated from OHLC data to predict
    future stock prices. Volume is excluded from all calculations.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', random_state: int = 42, 
                 warm_start: bool = False, **kwargs):
        super().__init__('Random Forest', **kwargs)
        self.model = None
        self.feature_columns = None
        self.feature_importance_ = None
        
        # Model parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.warm_start = warm_start
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the Random Forest model on stock data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - stock prices
            
        Returns:
            self: Returns self for method chaining
        """
        self.validate_input(X, y)
        
        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
            oob_score=True,
            warm_start=self.warm_start
        )
        
        # Train model (Random Forest does not require feature scaling)
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Get feature importance
        self.feature_importance_ = self.model.feature_importances_
        
        self.set_training_metrics({
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'oob_score': self.model.oob_score_
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
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def supports_incremental_learning(self) -> bool:
        """Check if model supports warm_start for incremental learning."""
        return self.warm_start
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Incrementally train on a batch of data using warm_start.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - stock prices
            
        Returns:
            self: Returns self for method chaining
        """
        if not self.supports_incremental_learning():
            raise ValueError("Model does not support incremental learning. Enable warm_start.")
        
        self.validate_input(X, y)
        
        # Incrementally train model using warm_start
        self.model.fit(X, y)
        
        # Update training status
        self.is_trained = True
        
        return self
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using tree variance.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.validate_input(X)
        
        # Get predictions from individual trees
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Mean prediction
        predictions = np.mean(tree_predictions, axis=0)
        
        # Uncertainty as standard deviation across trees
        uncertainty = np.std(tree_predictions, axis=0)
        
        return predictions, uncertainty
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the Random Forest model.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if self.feature_columns is None or self.feature_importance_ is None:
            return {}
        
        return dict(zip(self.feature_columns, self.feature_importance_))
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples
        """
        importance_dict = self.get_feature_importance()
        if not importance_dict:
            return []
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
    
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
            'metrics': self.training_metrics,
            'params': self.model_params,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance_
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
        self.training_metrics = data['metrics']
        self.model_params = data['params']
        self.feature_columns = data.get('feature_columns')
        self.feature_importance_ = data.get('feature_importance')
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
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Initialize base model
        base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        self.feature_importance_ = self.model.feature_importances_
        
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
    model = RandomForestModel(n_estimators=50, max_depth=10)
    
    # Add technical indicators
    df_with_features = model._create_technical_indicators(df)
    
    # Prepare training data
    X, y = StockIndicators.prepare_training_data(df_with_features)
    
    if len(X) > 0:
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X[-10:])  # Predict last 10 days
        
        print(f"Random Forest Model Results:")
        print(f"Training R²: {model.training_metrics['r2_score']:.4f}")
        print(f"Training RMSE: {model.training_metrics['rmse']:.4f}")
        print(f"OOB Score: {model.training_metrics['oob_score']:.4f}")
        print(f"Sample predictions: {predictions[:5]}")
        
        # Get feature importance
        top_features = model.get_top_features(5)
        print(f"Top 5 features: {top_features}")
        
        # Test save/load
        model.save('test_rf_model.pkl')
        loaded_model = RandomForestModel().load('test_rf_model.pkl')
        print(f"Model loaded successfully: {loaded_model.is_trained}")
        
        # Clean up
        os.remove('test_rf_model.pkl')
    else:
        print("Insufficient data for training")
