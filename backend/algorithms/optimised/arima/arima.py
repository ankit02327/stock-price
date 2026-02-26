"""
ARIMA Time Series Forecasting for Stock Price Prediction

Optimized implementation using statsmodels for stock price prediction
based on OHLCV data and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import sys
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ...model_interface import ModelInterface
from ...stock_indicators import StockIndicators

# Suppress warnings
warnings.filterwarnings('ignore')


class ARIMAModel(ModelInterface):
    """
    ARIMA model for stock price prediction.
    
    Uses time series analysis on stock prices to predict
    future stock prices. Volume is excluded from all calculations.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 auto_arima: bool = True, max_search_time: int = 300, **kwargs):
        super().__init__('ARIMA', **kwargs)
        self.model = None
        self.fitted_model = None
        self.feature_columns = None
        
        # Model parameters
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_arima = auto_arima
        self.max_search_time = max_search_time  # Maximum time for hyperparameter search (seconds)
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _check_stationarity(self, series: pd.Series) -> bool:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to test
            
        Returns:
            True if stationary, False otherwise
        """
        result = adfuller(series.dropna())
        return result[1] < 0.05  # p-value < 0.05 means stationary
    
    def _find_optimal_order(self, series: pd.Series, max_p: int = 3, 
                           max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order using AIC with timeout.
        Reduced search space for efficiency (max_p=3, max_q=3 instead of 5).
        
        Args:
            series: Time series data
            max_p: Maximum p value (reduced to 3)
            max_d: Maximum d value  
            max_q: Maximum q value (reduced to 3)
            
        Returns:
            Optimal (p, d, q) order
        """
        import time
        start_time = time.time()
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    # Check timeout
                    if time.time() - start_time > self.max_search_time:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"ARIMA parameter search timed out after {self.max_search_time}s, using best found: {best_order}")
                        return best_order
                    
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the ARIMA model on stock data.
        
        Args:
            X: Feature matrix (n_samples, n_features) - not used for ARIMA
            y: Target vector (n_samples,) - stock prices
            
        Returns:
            self: Returns self for method chaining
        """
        self.validate_input(X, y)
        
        # ARIMA works on time series, so we use y as the time series
        series = pd.Series(y)
        
        # Find optimal order if auto_arima is enabled
        if self.auto_arima:
            self.order = self._find_optimal_order(series)
        
        # Fit ARIMA model
        self.model = ARIMA(series, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit()
        
        # Calculate training metrics
        fitted_values = self.fitted_model.fittedvalues
        residuals = series - fitted_values
        
        mse = mean_squared_error(series, fitted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(series, fitted_values)
        
        # Calculate R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((series - series.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        self.set_training_metrics({
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'order': self.order
        })
        
        return self
    
    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Make predictions on new stock data.
        
        Args:
            X: Features to predict on (n_samples, n_features) - not used for ARIMA
            steps: Number of steps to predict ahead
            
        Returns:
            predictions: Predicted stock prices (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.validate_input(X)
        
        # ARIMA forecasting
        forecast = self.fitted_model.forecast(steps=steps)
        
        # Return array of predictions
        if steps == 1:
            return np.array([forecast])
        else:
            return np.array(forecast)
    
    def predict_with_confidence(self, X: np.ndarray, steps: int = 1, 
                              confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Features to predict on
            steps: Number of steps to predict ahead
            confidence_level: Confidence level (0-1)
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get forecast with confidence intervals
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int()
        
        # Extract bounds
        lower_bounds = confidence_intervals.iloc[:, 0].values
        upper_bounds = confidence_intervals.iloc[:, 1].values
        
        return forecast.values, lower_bounds, upper_bounds
    
    def get_model_summary(self) -> str:
        """
        Get model summary statistics.
        
        Returns:
            String representation of model summary
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return str(self.fitted_model.summary())
    
    def get_residuals(self) -> np.ndarray:
        """
        Get model residuals.
        
        Returns:
            Array of residuals
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.fitted_model.resid.values
    
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: File path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        joblib.dump({
            'fitted_model': self.fitted_model,
            'metrics': self.training_metrics,
            'params': self.model_params,
            'feature_columns': self.feature_columns,
            'order': self.order,
            'seasonal_order': self.seasonal_order
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
        self.fitted_model = data['fitted_model']
        self.training_metrics = data['metrics']
        self.model_params = data['params']
        self.feature_columns = data.get('feature_columns')
        self.order = data.get('order', (2, 1, 1))
        self.seasonal_order = data.get('seasonal_order', (0, 0, 0, 0))
        self.is_trained = True
        return self


# Example usage and testing
if __name__ == "__main__":
    # Create sample stock data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Generate synthetic OHLC data with trend and seasonality
    base_price = 100
    trend = np.linspace(0, 50, 200)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(200) / 30)  # Monthly seasonality
    noise = np.random.normal(0, 2, 200)
    prices = base_price + trend + seasonality + noise
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices
    })
    
    # Create model
    model = ARIMAModel(auto_arima=True)
    
    # Add technical indicators
    df_with_features = model._create_technical_indicators(df)
    
    # Prepare training data
    X, y = StockIndicators.prepare_training_data(df_with_features)
    
    if len(y) > 0:
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X[-10:], steps=5)  # Predict next 5 days
        
        print(f"ARIMA Model Results:")
        print(f"Training R²: {model.training_metrics['r2_score']:.4f}")
        print(f"Training RMSE: {model.training_metrics['rmse']:.4f}")
        print(f"Optimal Order: {model.training_metrics['order']}")
        print(f"Sample predictions: {predictions}")
        
        # Test save/load
        model.save('test_arima_model.pkl')
        loaded_model = ARIMAModel().load('test_arima_model.pkl')
        print(f"Model loaded successfully: {loaded_model.is_trained}")
        
        # Clean up
        os.remove('test_arima_model.pkl')
    else:
        print("Insufficient data for training")
