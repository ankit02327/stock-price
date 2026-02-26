# Model Training Guide

**Last Updated**: October 24, 2025 | **Status**: 4/7 models trained

---

## Overview

Complete guide for training machine learning models on stock price data. The system includes 7 algorithms with standalone trainers, supporting both basic and advanced models.

## Current Training Status

✅ **Trained**: Linear Regression (R²=-0.002), Decision Tree (R²=0.001), Random Forest (R²=0.024), SVM (R²=-0.0055)  
🔄 **Next**: KNN (K-Nearest Neighbors)  
⏳ **Pending**: ARIMA, Autoencoder

## Storage Structure

```
backend/models/
├── {model_name}/{model_name}_model.pkl    # Classical ML (joblib)
├── autoencoder/                           # Deep learning (3 files)
│   ├── autoencoder_model.pkl_autoencoder.h5
│   ├── autoencoder_model.pkl_encoder.h5
│   └── autoencoder_model.pkl_metadata.pkl
└── model_status.json
```

## Training Architecture

```
backend/training/
├── basic_models/{model}/           # Linear Regression, Decision Tree, Random Forest, SVM
│   ├── config.py                   # Hyperparameters
│   └── trainer.py                  # Standalone trainer
├── advanced_models/{model}/        # KNN, ARIMA, Autoencoder
│   ├── config.py
│   └── trainer.py
├── common_trainer_utils.py         # Shared utilities
├── display_manager.py              # Progress tracking
└── validation_stocks.json          # Validation data
```

## Models Overview (7 Total)

| Model | Type | Training Time | Model Size | Memory Peak | Status |
|-------|------|---------------|------------|-------------|--------|
| Linear Regression | Basic | ~2 min | ~3 MB | 2-3 GB | ✅ Trained |
| Decision Tree | Basic | ~3 min | ~150 MB | 3-4 GB | ✅ Trained |
| Random Forest | Basic | ~8 min | ~9.5 GB | 6-8 GB | ✅ Trained |
| SVM | Basic | ~15 min | ~10 MB | 4-6 GB | ✅ Trained |
| KNN | Advanced | ~5 min | ~10 MB | 4-5 GB | 🔄 Next |
| ARIMA | Advanced | ~25 min | ~170 MB | 3-5 GB | ⏳ Pending |
| Autoencoder | Advanced | ~18 min | ~1 MB (x3 files) | 4-6 GB | ⏳ Pending |

**Total Disk Space Required**: ~10-11 GB for all 7 models

## Prerequisites

### Hardware Requirements
- **Minimum**: 8GB RAM, 4+ CPU cores, 15GB free disk space
- **Recommended**: 16GB RAM, 8+ CPU cores, 20GB free disk space

### Software Requirements
- **Python**: 3.8 or higher
- **Node.js**: 16 or higher (for frontend)
- **Git**: For cloning repository

### Platform-Specific Setup

#### Windows
```powershell
# Install Python (if not installed)
winget install Python.Python.3.11

# Install Node.js (if not installed)
winget install OpenJS.NodeJS

# Verify installations
python --version
node --version
npm --version
```

#### macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Node.js
brew install python@3.11 node

# Verify installations
python3 --version
node --version
npm --version
```

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3.11 python3.11-pip python3.11-venv

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installations
python3 --version
node --version
npm --version
```

## Training Setup

### 1. Environment Setup
```bash
# Navigate to backend directory
cd ml/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Data Availability
```bash
# Check system status
python status.py

# Verify data files exist
ls permanent/us_stocks/individual_files/ | wc -l  # Should show 501
ls permanent/ind_stocks/individual_files/ | wc -l  # Should show 500
```

## Training Commands

### Individual Model Training

#### Basic Models
```bash
# Linear Regression (fastest - good for testing)
python backend/training/basic_models/linear_regression/trainer.py

# Decision Tree
python backend/training/basic_models/decision_tree/trainer.py

# Random Forest (best expected performance)
python backend/training/basic_models/random_forest/trainer.py

# SVM (Support Vector Machine)
python backend/training/basic_models/svm/trainer.py
```

#### Advanced Models
```bash
# KNN (K-Nearest Neighbors)
python backend/training/advanced_models/knn/trainer.py

# ARIMA (Time Series)
python backend/training/advanced_models/arima/trainer.py

# Autoencoder (Neural Network)
python backend/training/advanced_models/autoencoder/trainer.py
```

### Training Options
```bash
# Test with limited data (faster)
python backend/training/basic_models/linear_regression/trainer.py --max-stocks 100

# Force retrain (ignore existing model)
python backend/training/basic_models/random_forest/trainer.py --force-retrain

# Train all models sequentially
python backend/training/train_full_dataset.py
```

## Training Data

### Dataset Information
- **Period**: 2020-2024 (5 years of historical data)
- **Stocks Available**: 1,001 total (501 US + 500 Indian stocks)
- **Stocks Used for Training**: 936 stocks with sufficient historical data (65 filtered out)
- **Samples**: ~1,200 rows per stock → ~1,057 after feature engineering
- **Features**: 38 technical indicators
- **Total Training Samples**: ~950K rows from 936 stocks
- **Memory**: X=365MB, y=10MB, Peak=2-8GB depending on model

### Feature Engineering (38 Features)

**Price (2)**: price_change, price_change_abs  
**Moving Averages (10)**: ma_5/10/20/50/200 + ratios  
**Volatility (1)**: volatility  
**Momentum (1)**: rsi  
**Intraday (2)**: hl_ratio, oc_ratio  
**Position (1)**: price_position  
**Lagged (5)**: close_lag_1/2/3/5/10  
**Rolling Stats (9)**: close_std/min/max for 5/10/20 days  
**Time (3)**: day_of_week, month, quarter  
**Raw OHLC (3)**: open, high, low

**Note**: Volume exists but is NOT USED in calculations.

## Model Architecture

### Training Pipeline
1. **Data Loading**: Load stocks from permanent directory (936 with sufficient data from 1,001 available)
2. **Data Filtering**: Exclude stocks with insufficient historical data points
3. **Feature Engineering**: Generate 38 technical indicators
4. **Preprocessing**: Handle missing values, normalize features
5. **Training**: Single-pass training on combined dataset
6. **Validation**: Test on separate validation stocks
7. **Saving**: Store model and metadata

### Prediction Method

Models predict **percentage change** rather than absolute prices:
```
Current: $100 | Model: +2.5% | Final: $100 × 1.025 = $102.50
```

### Expected Performance

| Model | Time | Size | Memory | Target R² | Current R² | Status |
|-------|------|------|--------|-----------|------------|--------|
| Linear Regression | 2-3m | 3MB | 2-3GB | 0.85 | -0.002 | ✅ Needs fix |
| Decision Tree | 3-5m | 150MB | 3-4GB | 0.85 | 0.001 | ✅ Needs fix |
| Random Forest | 8-12m | 34MB | 6-8GB | 0.90+ | 0.024 | ✅ Trained |
| SVM | 15-25m | 10MB | 4-6GB | 0.80 | -0.0055 | ✅ Trained |
| KNN | 5-8m | 10MB | 4-5GB | 0.80 | - | 🔄 Next |
| ARIMA | 25-40m | 170MB | 3-5GB | 0.70 | - | ⏳ Pending |
| Autoencoder | 18-30m | 1MB | 4-6GB | 0.75 | - | ⏳ Pending |

**Requirements**: 8GB RAM min, 16GB recommended | 15GB disk | 4+ cores

## Output Locations

### Model Files
```
backend/models/
├── linear_regression/
│   └── linear_regression_model.pkl
├── decision_tree/
│   └── decision_tree_model.pkl
├── random_forest/
│   └── random_forest_model.pkl
├── svm/
│   └── svm_model.pkl
├── knn/
│   └── knn_model.pkl
├── arima/
│   └── arima_model.pkl
├── autoencoder/
│   ├── autoencoder_model.pkl_autoencoder.h5
│   ├── autoencoder_model.pkl_encoder.h5
│   └── autoencoder_model.pkl_metadata.pkl
└── model_status.json
```

### Logs
```
backend/logs/
├── linear_regression_training_2024-10-24_14-30-15.log
├── decision_tree_training_2024-10-24_14-35-22.log
├── random_forest_training_2024-10-24_14-45-18.log
└── svm_training_2024-10-24_15-02-41.log
```

## Monitoring Training Progress

### Real-time Monitoring
```bash
# Windows PowerShell
Get-Content backend\logs\random_forest_training_*.log -Tail 50 -Wait

# macOS/Linux
tail -f backend/logs/random_forest_training_*.log
```

### Check Training Status
```bash
# View current model status
python status.py

# Check specific model
python status.py --model random_forest
```

## Testing Trained Models

### 1. Start Backend Server
```bash
cd ml/backend
python main.py
```

### 2. Start Frontend (New Terminal)
```bash
cd ml/frontend
npm install
npm run dev
```

### 3. Test Predictions
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000
- **Test Stocks**: AAPL, MSFT, TSLA (US) | RELIANCE, TCS, INFY (Indian)
- **Horizons**: 1D, 1W, 1M, 1Y, 5Y

### 4. Verify Results
- Predictions should be reasonable (±20% for 1D/1W)
- Confidence scores should be displayed
- No error messages in browser console

## Troubleshooting

### Common Issues

| Issue | Fix |
|-------|-----|
| Low R² (~0.02) | Review feature engineering, check hyperparameters |
| Out of Memory | Close apps, ensure 8GB+ free, train one at a time |
| Training Hangs | Check logs in `backend/logs/`, verify data files exist |
| Model Not Found | Run trainer: `python backend/training/.../trainer.py` |
| Python Import Errors | Ensure virtual environment activated, run `pip install -r requirements.txt` |

#### Recent Fixes (Oct 23, 2025)
- **Decision Tree**: Removed unnecessary StandardScaler (tree models don't need scaling)
- **Review Workflow**: Code review before training catches bugs early
- **Validation**: Using separate validation stocks for unbiased testing

### Debug Commands
```bash
# Test individual components
python -c "import pandas, numpy, sklearn, tensorflow; print('All imports successful')"

# Check data loading
python -c "from prediction.data_loader import DataLoader; dl = DataLoader(); print('DataLoader works')"

# Verify model imports
python -c "from algorithms.optimised.random_forest.random_forest import RandomForestModel; print('Model import works')"
```

## Configuration

### Hyperparameter Tuning
Edit configuration files in `backend/training/{basic|advanced}_models/{model}/config.py`:

```python
# Example: Random Forest Configuration
class RandomForestConfig:
    N_ESTIMATORS = 100        # Number of trees
    MAX_DEPTH = 15            # Maximum tree depth
    MIN_SAMPLES_SPLIT = 10    # Minimum samples to split
    MIN_SAMPLES_LEAF = 5      # Minimum samples per leaf
    RANDOM_STATE = 42         # For reproducibility
    VERBOSE = True            # Show training progress
```

After editing config, retrain with `--force-retrain` flag.

## Best Practices

### Training Workflow
1. **Review First**: Read trainer code and config before training
2. **Start Small**: Test with `--max-stocks 100` first
3. **Monitor Resources**: Keep Task Manager/htop open
4. **Train Sequentially**: One model at a time for stability
5. **Validate Each**: Test predictions after each model

### Performance Optimization
- **Memory**: Close unnecessary applications
- **Storage**: Ensure 15GB+ free space
- **CPU**: Use all available cores (models are parallelized)
- **Testing**: Use limited stocks for initial testing

## Next Steps

### Immediate Actions
1. **Train KNN**: Next model in sequence
2. **Fix Low R²**: Investigate Linear Regression and Decision Tree
3. **Validate Models**: Test predictions on frontend

### Future Improvements
- **Hyperparameter Optimization**: Automated tuning
- **Model Ensemble**: Combine predictions from multiple models
- **Automated Retraining**: Schedule periodic model updates
- **Additional Models**: LSTM, XGBoost, etc.

## Related Documentation

- [Main README](../README.md) - Project overview and installation
- [Offline Mode](OFFLINE_MODE.md) - Running without API keys
- [Backend API](../backend/README.md) - API endpoints and usage
- [Upstox Integration](UPSTOX_INTEGRATION.md) - Indian market API setup

---

**Ready to train?** Start with Linear Regression for a quick test, then move to Random Forest for best performance!
