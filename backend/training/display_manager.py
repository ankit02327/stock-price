#!/usr/bin/env python3
"""
Display Manager for Training Progress

Handles all display formatting with emoji support and progress tracking.
"""

import sys
import time
from typing import Dict, List, Optional, Any
from datetime import datetime


class DisplayManager:
    """Manages all training progress display with emoji support and stage tracking."""
    
    # Stage definitions with emojis
    STAGES = {
        'loading': {'name': 'Loading', 'emoji': '⏳', 'description': 'Reading CSV files from disk'},
        'validation': {'name': 'Validation', 'emoji': '✓', 'description': 'Checking data quality'},
        'feature_engineering': {'name': 'Feature Engineering', 'emoji': '🔧', 'description': 'Creating 37 technical indicators'},
        'preprocessing': {'name': 'Preprocessing', 'emoji': '📊', 'description': 'Cleaning and scaling data'},
        'training': {'name': 'Training', 'emoji': '🎯', 'description': 'Model fitting'},
        'validation_final': {'name': 'Validation', 'emoji': '✅', 'description': 'Testing model accuracy'}
    }
    
    def __init__(self, model_name: str, update_interval: int = 20, enable_emojis: bool = None):
        """
        Initialize display manager.
        
        Args:
            model_name: Name of the model being trained
            update_interval: Update interval in seconds
            enable_emojis: Whether to try using emojis (auto-fallback if errors, None=auto-detect)
        """
        self.model_name = model_name.upper().replace('_', ' ')
        self.update_interval = update_interval
        
        # Auto-detect emoji support: disable on Windows by default
        if enable_emojis is None:
            import platform
            enable_emojis = platform.system() != 'Windows'
        
        self.enable_emojis = enable_emojis
        self.emojis_working = enable_emojis
        
        # Test emoji support
        if self.enable_emojis:
            self._test_emoji_support()
        
        # Separators (use ASCII-safe characters)
        self.separator_double = '=' * 80
        self.separator_single = '-' * 80
        
    def _test_emoji_support(self):
        """Test if terminal supports emojis and disable if not."""
        try:
            # Try to encode some common emojis
            test_emojis = '🎯📊🔧✓✅⏳'
            # Get the stdout encoding, default to ascii on Windows
            encoding = sys.stdout.encoding or 'ascii'
            test_emojis.encode(encoding)
            self.emojis_working = True
        except (UnicodeEncodeError, AttributeError, TypeError):
            self.emojis_working = False
            print("Note: Terminal doesn't support emojis, using text-only display")
    
    def _get_emoji(self, stage_key: str) -> str:
        """Get emoji for stage, or empty string if emojis disabled."""
        if not self.emojis_working:
            return ''
        return self.STAGES.get(stage_key, {}).get('emoji', '')
    
    def _safe_print(self, text: str):
        """Safely print text, catching emoji errors."""
        try:
            print(text)
        except UnicodeEncodeError:
            # Fallback: remove emojis and try again
            self.emojis_working = False
            # Simple emoji removal (remove all non-ASCII)
            ascii_text = ''.join(char for char in text if ord(char) < 128)
            print(ascii_text)
    
    def create_progress_bar(self, percentage: float, length: int = 50) -> str:
        """Create a progress bar string."""
        filled = int(length * percentage / 100)
        bar = '█' * filled + '─' * (length - filled)
        return f"[{bar}]"
    
    def format_time(self, seconds: float) -> str:
        """Format time in a readable way."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hr"
    
    def show_training_progress(self, elapsed_time: float, stage: str = "Training"):
        """
        Show periodic training progress update.
        
        Args:
            elapsed_time: Time elapsed in seconds
            stage: Current stage of training (default: "Training")
        """
        emoji_progress = '⚙️' if self.emojis_working else '[PROGRESS]'
        time_str = self.format_time(elapsed_time)
        
        # Create a simple progress indicator (rotating dots)
        dot_count = int(elapsed_time / 10) % 4
        dots = '.' * dot_count
        
        self._safe_print(f"{emoji_progress} {stage}... [Elapsed: {time_str}] - Model fitting in progress{dots}")
    
    def show_training_start(self, total_stocks: int, expected_duration_min: int):
        """Show training start message."""
        emoji_start = '🚀' if self.emojis_working else '>>>'
        emoji_data = '📊' if self.emojis_working else '[DATA]'
        emoji_time = '⏱️' if self.emojis_working else '[TIME]'
        emoji_model = '🎯' if self.emojis_working else '[MODEL]'
        
        self._safe_print(f"\n{self.separator_double}")
        self._safe_print(f"{emoji_start} STARTING {self.model_name} TRAINING")
        self._safe_print(f"{self.separator_single}")
        self._safe_print(f"{emoji_data} DATA TARGET:")
        self._safe_print(f"   Stocks: ~{total_stocks:,} stocks (US + Indian)")
        self._safe_print(f"   Historical Data: 5 years per stock")
        self._safe_print(f"   Features: 37 technical indicators per stock")
        self._safe_print(f"   Total Samples: ~{total_stocks * 1000:,}+ data points")
        self._safe_print(f"")
        self._safe_print(f"{emoji_time} TIMING ESTIMATES:")
        self._safe_print(f"   Start Time: {datetime.now().strftime('%H:%M:%S')}")
        self._safe_print(f"   Expected Duration: ~{expected_duration_min} minutes")
        self._safe_print(f"   Updates: Every {self.update_interval} seconds")
        self._safe_print(f"")
        self._safe_print(f"{emoji_model} MODEL INFO:")
        self._safe_print(f"   Model: {self.model_name}")
        self._safe_print(f"   Progress tracking: Real-time updates during training")
        self._safe_print(f"{self.separator_double}\n")
    
    def show_training_complete(self, summary: Dict[str, Any]):
        """
        Show final training completion summary.
        
        Args:
            summary: Dictionary with:
                - model_name: Full model name
                - file_type: File extension (.pkl, .h5, etc.)
                - model_path: Full path to saved model
                - file_size_mb: File size in MB
                - stocks_processed: Total stocks processed
                - total_samples: Total samples used
                - total_time: Total time in seconds
                - validation_r2: Validation R² score
        """
        emoji_success = '✅' if self.emojis_working else '[SUCCESS]'
        emoji_model = '📦' if self.emojis_working else '[MODEL]'
        emoji_file = '💾' if self.emojis_working else '[FILE]'
        emoji_folder = '📁' if self.emojis_working else '[PATH]'
        emoji_size = '📊' if self.emojis_working else '[SIZE]'
        emoji_stocks = '📈' if self.emojis_working else '[DATA]'
        emoji_samples = '📊' if self.emojis_working else '[SAMPLES]'
        emoji_time = '⏱️' if self.emojis_working else '[TIME]'
        emoji_score = '🎯' if self.emojis_working else '[SCORE]'
        emoji_status = '✅' if self.emojis_working else '[OK]'
        emoji_check = '✓' if self.emojis_working else 'OK'
        
        model_name = summary.get('model_name', self.model_name)
        file_type = summary.get('file_type', '.pkl')
        model_path = summary.get('model_path', '')
        file_size = summary.get('file_size_mb', 0)
        stocks = summary.get('stocks_processed', 0)
        samples = summary.get('total_samples', 0)
        total_time = summary.get('total_time', 0)
        r2_score = summary.get('validation_r2', 0)
        
        self._safe_print(f"\n{self.separator_double}")
        self._safe_print(f"{emoji_success} TRAINING COMPLETED SUCCESSFULLY")
        self._safe_print(f"{self.separator_double}")
        self._safe_print(f"Model Details:")
        self._safe_print(f"  {emoji_model} Model Name: {model_name}")
        self._safe_print(f"  {emoji_file} File Type: {file_type} (Pickle)" if file_type == '.pkl' else f"  {emoji_file} File Type: {file_type}")
        self._safe_print(f"  {emoji_folder} Saved Path: {model_path}")
        if file_size > 0:
            self._safe_print(f"  {emoji_size} File Size: {file_size:.1f} MB")
        self._safe_print(f"")
        self._safe_print(f"Training Summary:")
        self._safe_print(f"  {emoji_stocks} Stocks Processed: {stocks:,} / {stocks:,} (100%)")
        self._safe_print(f"  {emoji_samples} Total Samples: {samples:,}")
        self._safe_print(f"  {emoji_time}  Total Time: {self.format_time(total_time)}")
        if r2_score > 0:
            self._safe_print(f"  {emoji_score} Validation R²: {r2_score:.4f}")
        self._safe_print(f"  {emoji_status} Status: Model ready for predictions")
        self._safe_print(f"")
        self._safe_print(f"Model saved successfully {emoji_check}")
        self._safe_print(f"{self.separator_double}\n")
    
    def show_error(self, error_message: str):
        """Show error message."""
        emoji_error = '❌' if self.emojis_working else '[ERROR]'
        
        self._safe_print(f"\n{self.separator_single}")
        self._safe_print(f"{emoji_error} Error: {error_message}")
        self._safe_print(f"{self.separator_single}\n")


# Example usage
if __name__ == "__main__":
    # Test the display manager
    dm = DisplayManager("Linear Regression")
    
    print("Testing emoji support...")
    print(f"Emojis working: {dm.emojis_working}")
    
    print("\n\nTesting training start display...")
    dm.show_training_start(total_stocks=1000, expected_duration_min=25)
    
    print("\n\nTesting completion summary...")
    summary = {
        'model_name': 'Linear Regression',
        'file_type': '.pkl',
        'model_path': 'backend/models/linear_regression/linear_regression_model.pkl',
        'file_size_mb': 2.3,
        'stocks_processed': 1000,
        'total_samples': 1245678,
        'total_time': 1530,  # 25.5 minutes
        'validation_r2': 0.9456
    }
    dm.show_training_complete(summary)

