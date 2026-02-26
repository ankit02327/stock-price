#!/usr/bin/env python3
"""
Simple Model Training Status Checker

This is the ONLY status checking script for the project.
Shows current training status of all ML models in a clean format.

Usage:
    python status.py                    # Table format (default)
    python status.py --json             # JSON format
    python status.py --simple           # Simple one-line format
    python status.py --help             # Show help
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from training.enhanced_model_trainer import EnhancedModelTrainer
except ImportError as e:
    print(f"Error importing trainer: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def format_r2_score(r2_score):
    """Format R² score for display."""
    if r2_score is None:
        return "N/A"
    elif r2_score > 1000:
        return f"{r2_score:.0f}"
    elif r2_score > 0:
        return f"{r2_score:.3f}"
    else:
        return f"{r2_score:.1f}"


def format_date(date_str):
    """Format date for display."""
    if date_str:
        return date_str[:10]  # Just the date part
    return "N/A"


def format_error(error_str):
    """Format error message for display."""
    if not error_str:
        return ""
    return error_str[:50] + "..." if len(error_str) > 50 else error_str


def print_status_table(summary):
    """Print status in table format."""
    print("="*100)
    print("ML MODEL TRAINING STATUS")
    print("="*100)
    print(f"Total Models: {summary['total_models']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pending: {summary['pending']}")
    print("")
    
    print("Model Details:")
    print("-" * 100)
    print(f"{'Model Name':<20} | {'Trained':<8} | {'Stocks':<8} | {'R² Score':<10} | {'Date':<12} | {'Error'}")
    print("-" * 100)
    
    for name, details in summary['models'].items():
        model_details = details['details']
        
        # Support both old and new format
        status = model_details.get('status', 'pending')
        trained = (status == 'completed') or model_details.get('trained', False)
        stocks_trained = model_details.get('stocks_trained', 0)
        
        # Try to get R² from validation_metrics first, then fall back to r2_score
        validation_metrics = model_details.get('validation_metrics', {})
        r2_score = validation_metrics.get('avg_r2_score') if validation_metrics else model_details.get('r2_score')
        
        trained_date = model_details.get('last_updated', model_details.get('trained_date', ''))
        error = model_details.get('error_message', model_details.get('error', ''))
        
        trained_str = "Yes" if trained else "No"
        r2_str = format_r2_score(r2_score)
        date_str = format_date(trained_date)
        error_str = format_error(error)
        
        print(f"{name:<20} | {trained_str:<8} | {stocks_trained:<8} | {r2_str:<10} | {date_str:<12} | {error_str}")
    
    print("="*100)


def print_status_json(summary):
    """Print status as JSON."""
    # Convert to simple format for JSON output
    models = {}
    for name, details in summary['models'].items():
        model_details = details['details']
        models[name] = {
            'trained': model_details.get('trained', False),
            'stocks_trained': model_details.get('stocks_trained', 0),
            'r2_score': model_details.get('r2_score'),
            'trained_date': model_details.get('trained_date'),
            'error': model_details.get('error')
        }
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_models': summary['total_models'],
            'completed': summary['completed'],
            'failed': summary['failed'],
            'pending': summary['pending']
        },
        'models': models
    }
    
    print(json.dumps(output, indent=2))


def print_status_simple(summary):
    """Print status in simple one-line format."""
    print(f"Models: {summary['completed']}/{summary['total_models']} completed, {summary['failed']} failed, {summary['pending']} pending")
    
    for name, details in summary['models'].items():
        model_details = details['details']
        trained = model_details.get('trained', False)
        stocks_trained = model_details.get('stocks_trained', 0)
        r2_score = model_details.get('r2_score')
        
        status = "✅" if trained and not model_details.get('error') else "❌" if model_details.get('error') else "⏳"
        r2_str = format_r2_score(r2_score)
        
        print(f"{status} {name}: {stocks_trained} stocks, R²={r2_str}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Check ML model training status')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--simple', action='store_true', help='Simple one-line format')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer and get status
        trainer = EnhancedModelTrainer()
        summary = trainer.get_training_summary()
        
        if args.json:
            print_status_json(summary)
        elif args.simple:
            print_status_simple(summary)
        else:
            print_status_table(summary)
            
    except Exception as e:
        print(f"Error checking status: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
