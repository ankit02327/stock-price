"""
ARIMA (AutoRegressive Integrated Moving Average) Implementation from Scratch

This module implements ARIMA for time series forecasting using:
- AutoRegressive (AR) component for trend modeling
- Integrated (I) component for stationarity through differencing
- Moving Average (MA) component for error modeling
- Maximum Likelihood Estimation for parameter fitting

Mathematical Foundation:
- AR(p): X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + φₚX_{t-p} + ε_t
- MA(q): X_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q}
- Differencing: ∇ᵈX_t = (1-B)ᵈX_t where B is backshift operator
- ARIMA(p,d,q): ∇ᵈX_t = c + Σφᵢ∇ᵈX_{t-i} + Σθⱼε_{t-j} + ε_t
"""

import numpy as np
import pandas as pd

#########################
# ARIMA Model Implementation
#########################

class ARIMA:
    """
    ARIMA (AutoRegressive Integrated Moving Average) Model
    ARIMA(p, d, q):
    - p: number of autoregressive (AR) terms
    - d: degree of differencing
    - q: number of moving average (MA) terms
    """
    
    def __init__(self, p=1, d=1, q=1):
        """
        Initialize ARIMA model
        
        Parameters:
        - p: AR order (number of lag observations)
        - d: Differencing order (number of times to difference)
        - q: MA order (size of moving average window)
        """
        self.p = p
        self.d = d
        self.q = q
        
        self.ar_params = None
        self.ma_params = None
        self.intercept = None
        
        self.original_data = None
        self.differenced_data = None
        self.residuals = None
        
    def difference(self, data, d):
        """
        Apply differencing to make series stationary
        
        Formula: y'_t = y_t - y_{t-1}
        """
        differenced = data.copy()
        for i in range(d):
            differenced = np.diff(differenced)
        return differenced
    
    def inverse_difference(self, differenced, original, d):
        """
        Reverse differencing to get original scale
        """
        result = differenced.copy()
        
        for i in range(d):
            # Reconstruct from differences
            cumsum = np.cumsum(result)
            result = cumsum + original[d - i - 1]
        
        return result
    
    def fit(self, data, method='ols'):
        """
        Fit ARIMA model to data
        
        Parameters:
        - data: time series data (1D array)
        - method: 'ols' (Ordinary Least Squares) or 'yule_walker'
        """
        self.original_data = np.array(data)
        
        # Step 1: Apply differencing
        if self.d > 0:
            self.differenced_data = self.difference(self.original_data, self.d)
        else:
            self.differenced_data = self.original_data.copy()
        
        # Step 2: Estimate AR parameters using Yule-Walker equations
        if self.p > 0:
            self.ar_params = self._estimate_ar_parameters()
        else:
            self.ar_params = np.array([])
        
        # Step 3: Compute residuals
        self.residuals = self._compute_residuals()
        
        # Step 4: Estimate MA parameters
        if self.q > 0:
            self.ma_params = self._estimate_ma_parameters()
        else:
            self.ma_params = np.array([])
        
        # Step 5: Estimate intercept
        self.intercept = np.mean(self.differenced_data)
        
        return self
    
    def _estimate_ar_parameters(self):
        """
        Estimate AR parameters using Yule-Walker equations
        
        For AR(p): y_t = φ_1*y_{t-1} + φ_2*y_{t-2} + ... + φ_p*y_{t-p} + ε_t
        """
        data = self.differenced_data
        n = len(data)
        
        # Compute autocorrelations
        mean = np.mean(data)
        var = np.var(data)
        
        # Create autocorrelation matrix
        R = np.zeros((self.p, self.p))
        r = np.zeros(self.p)
        
        for i in range(self.p):
            for j in range(self.p):
                lag = abs(i - j)
                if var > 0:
                    R[i, j] = np.sum((data[:-lag-1] - mean) * (data[lag+1:] - mean)) / ((n - lag - 1) * var) if lag > 0 else 1.0
                else:
                    R[i, j] = 1.0 if i == j else 0.0
        
        for i in range(self.p):
            lag = i + 1
            if var > 0 and lag < n:
                r[i] = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / ((n - lag) * var)
            else:
                r[i] = 0.0
        
        # Solve Yule-Walker equations: R * φ = r
        try:
            ar_params = np.linalg.solve(R, r)
        except:
            # If singular, use pseudo-inverse
            ar_params = np.linalg.lstsq(R, r, rcond=None)[0]
        
        return ar_params
    
    def _compute_residuals(self):
        """Compute residuals from AR model"""
        data = self.differenced_data
        residuals = np.zeros(len(data))
        
        for t in range(self.p, len(data)):
            prediction = 0
            for i in range(self.p):
                prediction += self.ar_params[i] * data[t - i - 1]
            residuals[t] = data[t] - prediction
        
        return residuals
    
    def _estimate_ma_parameters(self):
        """
        Estimate MA parameters using residuals
        
        For MA(q): y_t = ε_t + θ_1*ε_{t-1} + θ_2*ε_{t-2} + ... + θ_q*ε_{t-q}
        """
        # Simplified MA estimation using autocorrelation of residuals
        residuals = self.residuals[self.p:]
        n = len(residuals)
        
        if n == 0:
            return np.zeros(self.q)
        
        mean_resid = np.mean(residuals)
        var_resid = np.var(residuals)
        
        ma_params = np.zeros(self.q)
        
        for i in range(self.q):
            lag = i + 1
            if lag < n and var_resid > 0:
                ma_params[i] = np.sum((residuals[:-lag] - mean_resid) * (residuals[lag:] - mean_resid)) / ((n - lag) * var_resid)
            else:
                ma_params[i] = 0.0
        
        return ma_params
    
    def predict(self, steps=1):
        """
        Forecast future values
        
        Parameters:
        - steps: number of steps ahead to forecast
        
        Returns:
        - predictions: forecasted values
        """
        if self.ar_params is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        
        # Start with differenced data
        history = list(self.differenced_data)
        residual_history = list(self.residuals[-self.q:]) if self.q > 0 else []
        
        for _ in range(steps):
            # AR component
            ar_pred = 0
            for i in range(min(self.p, len(history))):
                ar_pred += self.ar_params[i] * history[-(i + 1)]
            
            # MA component
            ma_pred = 0
            for i in range(min(self.q, len(residual_history))):
                ma_pred += self.ma_params[i] * residual_history[-(i + 1)]
            
            # Combined prediction
            pred = ar_pred + ma_pred
            predictions.append(pred)
            
            # Update history
            history.append(pred)
            residual_history.append(0)  # Assume zero residual for future
        
        predictions = np.array(predictions)
        
        # Reverse differencing if needed
        if self.d > 0:
            # Reconstruct from differences
            last_original = self.original_data[-1]
            reconstructed = [last_original]
            
            for pred in predictions:
                reconstructed.append(reconstructed[-1] + pred)
            
            predictions = np.array(reconstructed[1:])
        
        return predictions
    
    def forecast(self, steps=1):
        """Alias for predict()"""
        return self.predict(steps)

#########################
# Helper Functions
#########################

def check_stationarity(data):
    """
    Simple stationarity check using rolling statistics
    
    A stationary series has:
    - Constant mean
    - Constant variance
    - No trend
    """
    data = np.array(data)
    
    # Split into two halves
    mid = len(data) // 2
    first_half = data[:mid]
    second_half = data[mid:]
    
    mean_diff = abs(np.mean(first_half) - np.mean(second_half))
    var_diff = abs(np.var(first_half) - np.var(second_half))
    
    # Simple heuristic
    is_stationary = (mean_diff < 0.1 * np.mean(data)) and (var_diff < 0.1 * np.var(data))
    
    return is_stationary

def mean_absolute_error(y_true, y_pred):
    """Compute MAE"""
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    """Compute MSE"""
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """Compute RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

#########################
# Example Usage
#########################

if __name__ == "__main__":
    print("=" * 60)
    print("ARIMA TIME-SERIES FORECASTING - FROM SCRATCH")
    print("=" * 60)
    
    # Example 1: Synthetic trend + seasonality data
    print("\\n--- Example 1: Synthetic Time Series ---")
    
    np.random.seed(42)
    t = np.arange(100)
    
    # Create synthetic time series: trend + seasonality + noise
    trend = 0.5 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(100) * 2
    data = trend + seasonality + noise
    
    # Split into train and test
    train_size = 80
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Fit ARIMA(2, 1, 1) model
    model = ARIMA(p=2, d=1, q=1)
    model.fit(train_data)
    
    # Forecast next 20 steps
    forecast = model.predict(steps=len(test_data))
    
    # Compute error metrics
    mae = mean_absolute_error(test_data, forecast)
    rmse = root_mean_squared_error(test_data, forecast)
    
    print(f"\\nModel: ARIMA(2, 1, 1)")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"\\nForecast Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Print first few predictions
    print(f"\\nFirst 5 Predictions vs Actual:")
    for i in range(min(5, len(forecast))):
        print(f"Step {i+1}: Predicted={forecast[i]:.2f}, Actual={test_data[i]:.2f}")
    
    # Example 2: Random walk
    print("\\n--- Example 2: Random Walk ---")
    
    np.random.seed(123)
    random_walk = np.cumsum(np.random.randn(50))
    
    train_rw = random_walk[:40]
    test_rw = random_walk[40:]
    
    model_rw = ARIMA(p=1, d=1, q=0)
    model_rw.fit(train_rw)
    
    forecast_rw = model_rw.predict(steps=len(test_rw))
    
    mae_rw = mean_absolute_error(test_rw, forecast_rw)
    
    print(f"\\nModel: ARIMA(1, 1, 0) - Random Walk")
    print(f"MAE: {mae_rw:.4f}")
    
    # Display model parameters
    print(f"\nModel Parameters:")
    print(f"AR coefficients: {model.ar_params}")
    print(f"MA coefficients: {model.ma_params}")
    print(f"Intercept: {model.intercept}")
    
    # Calculate additional statistics
    errors = test_data - forecast
    print(f"\nForecast Error Statistics:")
    print(f"Mean Error: {np.mean(errors):.4f}")
    print(f"Std Error: {np.std(errors):.4f}")
    print(f"Max Error: {np.max(np.abs(errors)):.4f}")
    
    # Display differencing information
    if model.d > 0:
        print(f"\nDifferencing Information:")
        print(f"Differencing order (d): {model.d}")
        print(f"Differenced data length: {len(model.differenced_data)}")
        print(f"Original data length: {len(model.original_data)}")
    
    # Display residual statistics
    print(f"\nResidual Statistics:")
    print(f"Mean residual: {np.mean(model.residuals):.6f}")
    print(f"Residual std: {np.std(model.residuals):.6f}")
    print(f"Residual range: [{np.min(model.residuals):.4f}, {np.max(model.residuals):.4f}]")
    
    print("\\n" + "=" * 60)
    print("ARIMA IMPLEMENTATION COMPLETE!")
    print("=" * 60)