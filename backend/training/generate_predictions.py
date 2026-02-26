#!/usr/bin/env python3
"""
Generate Predictions Script

This script generates predictions for all stocks using the trained models.
It loads pre-trained models and creates prediction files for all stocks.

Usage:
    python backend/training/generate_predictions.py
    python backend/training/generate_predictions.py --category us_stocks
    python backend/training/generate_predictions.py --max-stocks 10
    python backend/training/generate_predictions.py --model linear_regression
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import json

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prediction.data_loader import DataLoader
from prediction.prediction_saver import PredictionSaver
from prediction.config import config

# Import model classes directly
from algorithms.optimised.linear_regression.linear_regression import LinearRegressionModel
from algorithms.optimised.random_forest.random_forest import RandomForestModel
from algorithms.optimised.svm.svm import SVMModel
from algorithms.optimised.knn.knn import KNNModel
from algorithms.optimised.decision_tree.decision_tree import DecisionTreeModel
from algorithms.optimised.arima.arima import ARIMAModel
from algorithms.optimised.autoencoders.autoencoder import AutoencoderModel

# Setup logging
def setup_logging():
    """Setup logging configuration."""
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)


class PredictionGenerator:
    """Generates predictions using trained models."""
    
    # Model class mapping
    MODEL_CLASSES = {
        'linear_regression': LinearRegressionModel,
        'random_forest': RandomForestModel,
        'svm': SVMModel,
        'knn': KNNModel,
        'decision_tree': DecisionTreeModel,
        'arima': ARIMAModel,
        'autoencoder': AutoencoderModel
    }
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.prediction_saver = PredictionSaver()
        self.models = {}
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
    def load_trained_models(self, specific_model: str = None) -> Dict[str, Any]:
        """Load all trained models from disk."""
        logger = logging.getLogger(__name__)
        
        models = {}
        model_names = [specific_model] if specific_model else self.MODEL_CLASSES.keys()
        
        for model_name in model_names:
            # Load from model-specific subdirectory
            model_subdir = os.path.join(self.models_dir, model_name)
            model_path = os.path.join(model_subdir, f"{model_name}_model.pkl")
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                continue
            
            try:
                # Get model class
                model_class = self.MODEL_CLASSES.get(model_name)
                if model_class is None:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                # Load model
                model = model_class().load(model_path)
                models[model_name] = model
                logger.info(f"[SUCCESS] Loaded {model_name}")
                
            except Exception as e:
                logger.error(f"[ERROR] Error loading {model_name}: {e}")
                continue
        
        logger.info(f"Loaded {len(models)} trained models")
        return models
    
    def generate_stock_predictions(self, symbol: str, category: str, models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictions for a single stock using all models."""
        logger = logging.getLogger(__name__)
        
        try:
            # Load stock data
            df = self.data_loader.load_stock_data(symbol, category)
            if df is None or len(df) < config.MIN_TRAINING_DAYS:
                logger.warning(f"Insufficient data for {symbol}")
                return []
            
            # Validate data quality
            if not self.data_loader.validate_data_quality(df, symbol):
                logger.warning(f"Data quality issues for {symbol}")
                return []
            
            # Create features
            df_with_features = self.data_loader.create_features(df)
            if df_with_features is None or len(df_with_features) == 0:
                logger.warning(f"Could not create features for {symbol}")
                return []
            
            # Prepare training data
            X, y = self.data_loader.prepare_training_data(df_with_features)
            if len(X) == 0 or len(y) == 0:
                logger.warning(f"Insufficient training data for {symbol}")
                return []
            
            # Get current price for reference
            current_price = df['close'].iloc[-1]
            currency = 'USD' if category == 'us_stocks' else 'INR'
            
            all_predictions = []
            
            # Generate predictions for each time horizon
            for horizon in config.TIME_HORIZONS.keys():
                horizon_days = config.get_time_horizon_days(horizon)
                
                # Get predictions from all models
                model_predictions = {}
                model_accuracies = {}
                
                for model_name, model in models.items():
                    try:
                        # Get prediction for this horizon
                        if horizon == '1D':
                            # Direct prediction for next day
                            prediction = model.predict(X[-1:].reshape(1, -1))[0]
                        else:
                            # For longer horizons, use trend extrapolation
                            prediction = self._extrapolate_prediction(
                                model, X, y, horizon_days, current_price
                            )
                        
                        # Calculate model accuracy (on training data)
                        y_pred_train = model.predict(X)
                        accuracy = self._calculate_accuracy(y, y_pred_train)
                        
                        model_predictions[model_name] = prediction
                        model_accuracies[model_name] = accuracy
                        
                    except Exception as e:
                        logger.warning(f"Error with {model_name} for {symbol} {horizon}: {e}")
                        continue
                
                if not model_predictions:
                    logger.warning(f"No model predictions for {symbol} {horizon}")
                    continue
                
                # Create ensemble prediction
                ensemble_prediction = self._create_ensemble_prediction(
                    model_predictions, model_accuracies
                )
                
                # Calculate confidence interval
                confidence_low, confidence_high = self._calculate_confidence_interval(
                    model_predictions, ensemble_prediction
                )
                
                # Create prediction dictionary for ensemble
                ensemble_pred = self.prediction_saver.create_prediction_dict(
                    horizon=horizon,
                    predicted_price=ensemble_prediction,
                    confidence_low=confidence_low,
                    confidence_high=confidence_high,
                    algorithm_used='ensemble',
                    currency=currency,
                    model_accuracy=np.mean(list(model_accuracies.values())),
                    data_points_used=len(X)
                )
                all_predictions.append(ensemble_pred)
                
                # Create individual model predictions
                for model_name, pred in model_predictions.items():
                    individual_pred = self.prediction_saver.create_prediction_dict(
                        horizon=horizon,
                        predicted_price=pred,
                        confidence_low=pred * 0.95,  # 5% range
                        confidence_high=pred * 1.05,
                        algorithm_used=model_name,
                        currency=currency,
                        model_accuracy=model_accuracies[model_name],
                        data_points_used=len(X)
                    )
                    all_predictions.append(individual_pred)
            
            return all_predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions for {symbol}: {e}")
            return []
    
    def _extrapolate_prediction(self, model, X: np.ndarray, y: np.ndarray, 
                               horizon_days: int, current_price: float) -> float:
        """
        Extrapolate prediction for longer time horizons.
        
        NOTE: y contains PERCENTAGE CHANGES (not prices)
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector (PERCENTAGE CHANGES)
            horizon_days: Number of days to predict ahead
            current_price: Current stock price
            
        Returns:
            Extrapolated prediction as ABSOLUTE PRICE
        """
        try:
            # y contains percentage changes, not prices
            recent_pct_changes = y[-30:]  # Last 30 days of percentage changes
            if len(recent_pct_changes) < 2:
                return current_price  # Return current price if insufficient data
            
            # Calculate average daily percentage change
            avg_daily_pct_change = np.mean(recent_pct_changes)
            
            # Extrapolate: compound the average daily change over the horizon
            # For simplicity, we'll multiply by horizon_days (linear approximation)
            # This gives us the total expected percentage change
            trend_pct = avg_daily_pct_change * min(horizon_days, 30)  # Cap at 30 days for stability
            
            # Also get model's next-day prediction (percentage change)
            next_day_pct_pred = model.predict(X[-1:].reshape(1, -1))[0]
            
            # Combine trend and model prediction
            # For longer horizons, trust the trend more; for shorter, trust the model more
            trend_weight = min(0.7, horizon_days / 365)  # More weight to trend for longer horizons
            model_weight = 1 - trend_weight
            
            # Both are percentage changes, so we can combine them
            final_pct_prediction = (trend_weight * trend_pct + 
                                   model_weight * next_day_pct_pred)
            
            # Convert percentage change to absolute price
            predicted_price = current_price * (1 + final_pct_prediction / 100)
            
            return predicted_price
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Error in extrapolation: {e}")
            return current_price
    
    def _create_ensemble_prediction(self, model_predictions: Dict[str, float], 
                                   model_accuracies: Dict[str, float]) -> float:
        """Create ensemble prediction using weighted average."""
        try:
            # Use model weights from config, adjusted by accuracy
            total_weight = 0
            weighted_sum = 0
            
            for model_name, prediction in model_predictions.items():
                base_weight = config.get_model_weight(model_name)
                accuracy = model_accuracies.get(model_name, 0.5)
                
                # Adjust weight by accuracy
                adjusted_weight = base_weight * (0.5 + accuracy)  # Boost by accuracy
                weighted_sum += prediction * adjusted_weight
                total_weight += adjusted_weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                # Fallback to simple average
                return np.mean(list(model_predictions.values()))
                
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Error creating ensemble prediction: {e}")
            return np.mean(list(model_predictions.values()))
    
    def _calculate_confidence_interval(self, model_predictions: Dict[str, float], 
                                     ensemble_prediction: float) -> tuple:
        """Calculate confidence interval for predictions."""
        try:
            predictions = list(model_predictions.values())
            
            if len(predictions) < 2:
                # Single prediction - use 10% range
                range_pct = 0.1
                return (ensemble_prediction * (1 - range_pct), 
                       ensemble_prediction * (1 + range_pct))
            
            # Calculate standard deviation
            std_dev = np.std(predictions)
            
            # Use multiplier from config
            multiplier = config.CONFIDENCE_MULTIPLIER
            
            confidence_low = ensemble_prediction - (multiplier * std_dev)
            confidence_high = ensemble_prediction + (multiplier * std_dev)
            
            # Ensure positive values
            confidence_low = max(confidence_low, ensemble_prediction * 0.5)
            confidence_high = max(confidence_high, ensemble_prediction * 1.5)
            
            return confidence_low, confidence_high
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Error calculating confidence interval: {e}")
            # Fallback to 20% range
            range_pct = 0.2
            return (ensemble_prediction * (1 - range_pct), 
                   ensemble_prediction * (1 + range_pct))
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate model accuracy (R² score)."""
        try:
            from sklearn.metrics import r2_score
            return max(0, r2_score(y_true, y_pred))
        except:
            # Fallback to simple correlation
            try:
                correlation = np.corrcoef(y_true, y_pred)[0, 1]
                return max(0, correlation ** 2)
            except:
                return 0.5  # Default accuracy
    
    def generate_all_predictions(self, category: str = None, max_stocks: int = None, 
                                specific_model: str = None) -> Dict[str, Any]:
        """Generate predictions for all stocks."""
        logger = logging.getLogger(__name__)
        
        # Load trained models
        models = self.load_trained_models(specific_model)
        if not models:
            logger.error("No trained models found")
            return {'error': 'No trained models found'}
        
        results = {
            'total_stocks': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'failed_symbols': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'models_used': list(models.keys())
        }
        
        try:
            categories = [category] if category else ['us_stocks', 'ind_stocks']
            
            for cat in categories:
                logger.info(f"Generating predictions for {cat}")
                
                # Get stock symbols
                symbols = self.data_loader.get_stock_symbols(cat)
                
                if max_stocks:
                    symbols = symbols[:max_stocks]
                    logger.info(f"Processing first {max_stocks} stocks for testing")
                
                results['total_stocks'] += len(symbols)
                
                # Process each stock
                for i, symbol in enumerate(symbols, 1):
                    logger.info(f"Processing {symbol} ({i}/{len(symbols)}) in {cat}")
                    
                    try:
                        # Generate predictions
                        predictions = self.generate_stock_predictions(symbol, cat, models)
                        
                        if predictions:
                            # Save predictions
                            success = self.prediction_saver.save_predictions(
                                symbol, cat, predictions
                            )
                            
                            if success:
                                results['successful_predictions'] += 1
                                logger.info(f"[SUCCESS] Generated {len(predictions)} predictions for {symbol}")
                            else:
                                results['failed_predictions'] += 1
                                results['failed_symbols'].append(f"{symbol} ({cat})")
                                logger.error(f"[ERROR] Failed to save predictions for {symbol}")
                        else:
                            results['failed_predictions'] += 1
                            results['failed_symbols'].append(f"{symbol} ({cat})")
                            logger.warning(f"[WARNING] No predictions generated for {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        results['failed_predictions'] += 1
                        results['failed_symbols'].append(f"{symbol} ({cat})")
            
            results['end_time'] = datetime.now().isoformat()
            
            logger.info(f"Prediction generation summary: {results['successful_predictions']}/{results['total_stocks']} successful")
            
            if results['failed_symbols']:
                logger.warning(f"Failed symbols: {results['failed_symbols'][:10]}...")  # Show first 10
            
            return results
            
        except Exception as e:
            logger.error(f"Error in generate_all_predictions: {e}")
            results['end_time'] = datetime.now().isoformat()
            return results


def main():
    """Main function to generate predictions."""
    parser = argparse.ArgumentParser(description='Generate predictions using trained models')
    
    parser.add_argument('--category', 
                       choices=['us_stocks', 'ind_stocks'], 
                       help='Stock category to process')
    
    parser.add_argument('--max-stocks', 
                       type=int, 
                       help='Maximum number of stocks to process (for testing)')
    
    parser.add_argument('--model', 
                       choices=['linear_regression', 'random_forest', 'svm', 'knn', 
                               'decision_tree', 'arima', 'autoencoder'],
                       help='Use only a specific model')
    
    parser.add_argument('--test', 
                       action='store_true',
                       help='Run in test mode (limited stocks)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Set test mode defaults
        if args.test and not args.max_stocks:
            args.max_stocks = 5
        
        # Initialize generator
        logger.info("Initializing Prediction Generator...")
        generator = PredictionGenerator()
        
        start_time = datetime.now()
        logger.info(f"Starting prediction generation at {start_time}")
        
        # Generate predictions
        results = generator.generate_all_predictions(
            category=args.category,
            max_stocks=args.max_stocks,
            specific_model=args.model
        )
        
        # Print results
        print_prediction_results(results, logger)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"Prediction generation completed at {end_time}")
        logger.info(f"Total duration: {duration}")
        
    except KeyboardInterrupt:
        logger.info("Prediction generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in prediction generation: {str(e)}")
        sys.exit(1)


def print_prediction_results(results: dict, logger):
    """Print prediction results in a formatted way."""
    print("\n" + "="*80)
    print("PREDICTION GENERATION RESULTS")
    print("="*80)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Models used: {', '.join(results['models_used'])}")
    print(f"Total stocks processed: {results['total_stocks']}")
    print(f"Successful predictions: {results['successful_predictions']}")
    print(f"Failed predictions: {results['failed_predictions']}")
    
    success_rate = (results['successful_predictions'] / results['total_stocks'] * 100) if results['total_stocks'] > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    print(f"Start time: {results['start_time']}")
    print(f"End time: {results['end_time']}")
    
    if results['failed_symbols']:
        print(f"\nFailed symbols ({len(results['failed_symbols'])}):")
        for symbol in results['failed_symbols'][:10]:  # Show first 10
            print(f"  - {symbol}")
        if len(results['failed_symbols']) > 10:
            print(f"  ... and {len(results['failed_symbols']) - 10} more")
    
    print("="*80)
    
    # Save results to file
    results_file = f"prediction_generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
    except Exception as e:
        logger.warning(f"Could not save results file: {e}")


if __name__ == "__main__":
    main()
