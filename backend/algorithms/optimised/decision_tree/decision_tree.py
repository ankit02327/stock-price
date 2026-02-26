"""
Decision Tree for Stock Price Prediction

Optimized implementation using scikit-learn for stock price prediction
based on OHLCV data and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
import os
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ...model_interface import ModelInterface
from ...stock_indicators import StockIndicators


class DecisionTreeModel(ModelInterface):
    """
    Decision Tree model for stock price prediction.
    
    Uses technical indicators calculated from OHLC data to predict
    future stock prices. Volume is excluded from all calculations.
    """
    
    def __init__(self, max_depth: int = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: str = None,
                 random_state: int = 42, **kwargs):
        super().__init__('Decision Tree', **kwargs)
        self.model = None
        self.feature_columns = None
        self.feature_importance_ = None
        
        # Model parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _clean_data(self, X: np.ndarray) -> np.ndarray:
        """
        Clean data by handling infinity and NaN values.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cleaned feature matrix
        """
        # Replace infinity with NaN
        X_clean = np.copy(X)
        X_clean[np.isinf(X_clean)] = np.nan
        
        # Replace NaN with median of each feature
        for i in range(X_clean.shape[1]):
            col = X_clean[:, i]
            if np.any(np.isnan(col)):
                median_val = np.nanmedian(col)
                if np.isnan(median_val):
                    median_val = 0.0  # Fallback to 0 if all values are NaN
                X_clean[np.isnan(col), i] = median_val
        
        return X_clean
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the Decision Tree model on stock data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - stock prices
            
        Returns:
            self: Returns self for method chaining
        """
        self.validate_input(X, y)
        
        # Clean data: handle infinity and NaN values
        X_clean = self._clean_data(X)
        y_clean = y[~np.isnan(y) & ~np.isinf(y)]
        
        # Ensure X and y have same length after cleaning
        min_len = min(len(X_clean), len(y_clean))
        X_clean = X_clean[:min_len]
        y_clean = y_clean[:min_len]
        
        if len(X_clean) == 0:
            raise ValueError("No valid data after cleaning")
        
        # Initialize model
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        # Train model with progress updates
        print(f"[TRAINING] Starting decision tree training on {len(X_clean)} samples...")
        print(f"[TRAINING] Features: {X_clean.shape[1]}, Samples: {X_clean.shape[0]:,}")
        print(f"[TRAINING] Training on full dataset...")
        
        # Train the model
        import time
        start_time = time.time()
        self.model.fit(X_clean, y_clean)
        training_time = time.time() - start_time
        print(f"[COMPLETED] Decision tree training completed in {training_time:.1f} seconds")
        
        # Calculate training metrics
        y_pred = self.model.predict(X_clean)
        mse = mean_squared_error(y_clean, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_clean, y_pred)
        mae = mean_absolute_error(y_clean, y_pred)
        
        # Get feature importance
        self.feature_importance_ = self.model.feature_importances_
        
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
        
        # Clean data
        X_clean = self._clean_data(X)
        
        # Make predictions
        predictions = self.model.predict(X_clean)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the Decision Tree model.
        
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
    
    def get_tree_rules(self, feature_names: List[str] = None) -> List[str]:
        """
        Get decision tree rules for interpretability.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            List of decision rules as strings
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.model.n_features_in_)]
        
        # This is a simplified version - full tree export would be more complex
        rules = []
        rules.append(f"Tree depth: {self.model.get_depth()}")
        rules.append(f"Number of leaves: {self.model.get_n_leaves()}")
        rules.append(f"Number of features: {self.model.n_features_in_}")
        
        return rules
    
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
            'max_depth': [3, 5, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Initialize base model
        base_model = DecisionTreeRegressor(random_state=self.random_state)
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=2  # Verbose output to show progress
        )
        
        # Clean features
        X_clean = self._clean_data(X)
        
        # Fit grid search
        grid_search.fit(X_clean, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        # Calculate metrics
        y_pred = self.model.predict(X_clean)
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
    model = DecisionTreeModel(max_depth=10, min_samples_split=5)
    
    # Add technical indicators
    df_with_features = model._create_technical_indicators(df)
    
    # Prepare training data
    X, y = StockIndicators.prepare_training_data(df_with_features)
    
    if len(X) > 0:
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X[-10:])  # Predict last 10 days
        
        print(f"Decision Tree Model Results:")
        print(f"Training R²: {model.training_metrics['r2_score']:.4f}")
        print(f"Training RMSE: {model.training_metrics['rmse']:.4f}")
        print(f"Sample predictions: {predictions[:5]}")
        
        # Get feature importance
        top_features = model.get_top_features(5)
        print(f"Top 5 features: {top_features}")
        
        # Get tree info
        tree_rules = model.get_tree_rules()
        print(f"Tree info: {tree_rules}")
        
        # Test save/load
        model.save('test_dt_model.pkl')
        loaded_model = DecisionTreeModel().load('test_dt_model.pkl')
        print(f"Model loaded successfully: {loaded_model.is_trained}")
        
        # Clean up
        os.remove('test_dt_model.pkl')
    else:
        print("Insufficient data for training")
