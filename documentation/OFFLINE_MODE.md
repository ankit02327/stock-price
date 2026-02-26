# Offline Mode Guide

**The system works completely offline without any API keys!**

## ✨ What Works Offline

### ✅ Fully Functional (No API Keys Required)

- **Stock Information Cards**
  - Display metadata for 1,001 stocks (500 Indian + 501 US)
  - Show company name, sector, market cap
  - Historical prices from permanent directory
  - ML predictions use 936 stocks with sufficient training data

- **Historical Charts**
  - Complete 5-year OHLCV data (2020-2024)
  - Interactive Recharts visualization
  - Support for 1W, 1M, 1Y, 5Y time periods

- **ML Predictions**
  - All trained models work with offline data
  - Predictions based on permanent directory
  - Visual indicators show data source and date

- **Search Functionality**
  - Full-text search across 1,001 stocks
  - Symbol and company name matching
  - Fast index-based lookup

- **Technical Indicators**
  - 38 indicators calculated from OHLC data
  - Moving averages, RSI, volatility
  - All computed client-side

### ❌ Requires API Keys

- **Live Price Fetching**
  - Finnhub API (US stocks)
  - Upstox API (Indian stocks)
  - Real-time price updates

- **Currency Conversion (Live)**
  - Real-time USD/INR rates
  - Falls back to static rate if API unavailable

---

## 🚀 Quick Setup

See [Main README](../README.md) for detailed installation instructions.

**Summary**:
1. Backend: `cd ml/backend && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt`
2. Create empty `.env` file (required)
3. Frontend: `cd ml/frontend && npm install`
4. Run: Backend with `python main.py`, Frontend with `npm run dev`

Access: http://localhost:5173

## 🔍 Visual Indicators

### Info Card (Offline Data)

When using offline data, you'll see an amber warning box:

```
┌────────────────────────────────────┐
│ 🗄️ Using offline data from 2024-12-31 │
└────────────────────────────────────┘
```

### Prediction Card (Offline Data)

Similarly, predictions show:

```
┌────────────────────────────────────────────────┐
│ 🗄️ Prediction based on offline data from 2024-12-31 │
└────────────────────────────────────────────────┘
```

### Live Data Indicators

When APIs are configured and working:
- Info card shows: "Live data from finnhub/upstox"
- No warning boxes
- Real-time timestamp displayed

---

## 🔄 Data Synchronization

### Info Card & Prediction Card Sync

Both cards always use the **same data source**:

**Scenario 1: Online (APIs Available)**
- Info Card: Uses live API → Shows current price
- Prediction Card: Uses same price → Prediction based on live data
- ✅ **Both synchronized on live data**

**Scenario 2: Offline (No APIs)**
- Info Card: Falls back to permanent → Shows price from 2024-12-31
- Prediction Card: Uses same permanent data → Prediction from 2024-12-31
- ✅ **Both synchronized on offline data**

**Scenario 3: Partial (Some stocks offline)**
- System automatically detects per-symbol
- Each stock uses best available source
- Visual indicators show which source is used

---

## 🔧 How Offline Mode Works

### Data Loading Hierarchy

The system uses a three-tier fallback strategy:

```
1st Priority: data/past/{category}/individual_files/{SYMBOL}.csv
     ↓ (if not found)
2nd Priority: permanent/{category}/individual_files/{SYMBOL}.csv
     ↓ (if not found)
3rd: Error message with clear diagnostics
```

### Key Principles

1. **Permanent Directory is READ-ONLY**
   - Never written to
   - Contains 1001 pre-loaded stocks (2020-2024 data)
   - Serves as ultimate fallback for offline usage

2. **Data/Past is Primary**
   - Used for training and predictions when available
   - Can be updated with new data
   - Falls back to permanent if empty

3. **Automatic Fallback**
   - No configuration needed
   - System automatically detects missing data
   - Seamless transition between sources

---

## 🔄 Transitioning to Live Mode

### Adding API Keys

When ready for live data, create `backend/.env`:

```bash
# US Stocks (Finnhub)
FINNHUB_API_KEY=your_finnhub_key_here

# Indian Stocks (Upstox - requires OAuth)
UPSTOX_API_KEY=your_upstox_key_here
UPSTOX_CLIENT_ID=your_client_id_here
UPSTOX_CLIENT_SECRET=your_client_secret_here
```

### Upstox Daily Authentication

Upstox tokens expire daily at 3:30 AM IST. Run:

```bash
cd ml/backend
python scripts/setup_upstox_oauth.py
```

See [QUICK_AUTH_GUIDE.md](QUICK_AUTH_GUIDE.md) for detailed instructions.

---

## 📊 Data Organization

### Directory Structure

```
ml/
├── data/                     # Dynamic data
│   ├── past/                 # Historical data (2020-2024)
│   │   ├── us_stocks/
│   │   │   └── individual_files/
│   │   │       └── AAPL.csv
│   │   └── ind_stocks/
│   │       └── individual_files/
│   │           └── RELIANCE.csv
│   ├── latest/               # Live updates (2025+)
│   └── future/               # Predictions output
│
├── permanent/                # READ-ONLY fallback
│   ├── us_stocks/
│   │   ├── index_us_stocks.csv
│   │   └── individual_files/
│   │       └── [501 CSV files]
│   └── ind_stocks/
│       ├── index_ind_stocks.csv
│       └── individual_files/
│           └── [500 CSV files]
```

### File Format

Each CSV file contains:
```csv
date,open,high,low,close,volume,adjusted_close,currency
2024-12-31,150.00,151.00,149.50,150.25,5000000,,USD
```

---

## 🛠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Could not load stock data | Verify `permanent/` directory exists, check files |
| Backend won't start | Create empty `.env` file |
| Charts show no data | Check backend logs for `/historical` endpoint |
| Frontend build errors | Run `rm -rf node_modules package-lock.json && npm install` |
| Python import errors | Activate venv and run `pip install -r requirements.txt` |

---

## Related Documentation

- [Main README](../README.md) | [Training Guide](TRAINING.md) | [Upstox Integration](UPSTOX_INTEGRATION.md)