import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Brain, AlertTriangle, ChevronDown, ChevronRight, Database, Clock } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Skeleton } from './ui/skeleton';
import { Alert, AlertDescription } from './ui/alert';
import { Button } from './ui/button';
import { PredictionResult } from '../services/stockService';
import { formatPrice, Currency } from '../utils/currency';

interface StockPredictionProps {
  prediction: PredictionResult | null;
  currentPrice?: number;
  loading: boolean;
  symbol: string;
  error: string;
  currency: Currency;
  selectedHorizon?: string;
  selectedModel?: string;
  onHorizonChange?: (horizon: string) => void;
  onModelChange?: (model: string) => void;
  dataTimestamp?: string;
  dataSource?: string;
}

const HORIZON_OPTIONS = [
  { key: '1D', label: '1D' },
  { key: '1W', label: '1W' },
  { key: '1M', label: '1M' },
  { key: '1Y', label: '1Y' },
  { key: '5Y', label: '5Y' },
];

const MODEL_OPTIONS = {
  basic: [
    { key: 'decision_tree', label: 'Decision Tree' },
    { key: 'linear_regression', label: 'Linear Regression' },
    { key: 'random_forest', label: 'Random Forest' },
    { key: 'svm', label: 'Support Vector Machine' },
  ],
  advanced: [
    { key: 'arima', label: 'AutoRegressive Integrated Moving Average' },
    { key: 'autoencoder', label: 'Autoencoder' },
    { key: 'knn', label: 'K-Nearest Neighbors' },
  ]
};

export function StockPrediction({ 
  prediction, 
  currentPrice, 
  loading, 
  symbol, 
  error,
  currency,
  selectedHorizon: propSelectedHorizon = '1D',
  selectedModel: propSelectedModel = 'linear_regression',
  onHorizonChange,
  onModelChange,
  dataTimestamp,
  dataSource
}: StockPredictionProps) {
  const [selectedHorizon, setSelectedHorizon] = useState(propSelectedHorizon);
  const [selectedModel, setSelectedModel] = useState<string>(propSelectedModel);
  const [showBasic, setShowBasic] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Sync with parent props
  useEffect(() => {
    setSelectedHorizon(propSelectedHorizon);
  }, [propSelectedHorizon]);

  useEffect(() => {
    setSelectedModel(propSelectedModel);
  }, [propSelectedModel]);

  const handleHorizonChange = (horizon: string) => {
    setSelectedHorizon(horizon);
    if (onHorizonChange) {
      onHorizonChange(horizon);
    }
  };

  const handleModelToggle = (model: string) => {
    setSelectedModel(model);
    if (onModelChange) {
      onModelChange(model);
    }
  };
  
  // Calculate prediction values if available
  const displayPrediction = prediction;
  // Always use the currentPrice prop (same source as Stock Info Card)
  // This ensures both cards are synchronized on the same price data
  const predictionCurrentPrice = currentPrice || 0;
  const isPositiveChange = displayPrediction ? displayPrediction.predictedPrice > predictionCurrentPrice : false;
  const change = displayPrediction?.predictedPrice ? displayPrediction.predictedPrice - predictionCurrentPrice : 0;
  const changePercent = (predictionCurrentPrice && change !== 0) ? ((change / predictionCurrentPrice) * 100) : 0;

  const getConfidenceColor = (confidence: number | undefined | null) => {
    if (!confidence || isNaN(confidence)) return 'text-muted-foreground';
    if (confidence >= 61) return 'text-green-600';
    if (confidence >= 31) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBadgeVariant = (confidence: number | undefined | null): "default" | "secondary" | "destructive" | "outline" => {
    if (!confidence || isNaN(confidence)) return 'outline';
    if (confidence >= 61) return 'default';
    if (confidence >= 31) return 'secondary';
    return 'destructive';
  };

  const getProgressBarColor = (confidence: number | undefined | null) => {
    if (!confidence || isNaN(confidence)) return 'bg-gray-400';
    if (confidence >= 61) return 'bg-green-600';
    if (confidence >= 31) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getProgressBarStyle = (confidence: number | undefined | null) => {
    if (!confidence || isNaN(confidence)) return { backgroundColor: '#9ca3af' };
    if (confidence >= 61) return { backgroundColor: '#16a34a' }; // green-600
    if (confidence >= 31) return { backgroundColor: '#f97316' }; // orange-500
    return { backgroundColor: '#ef4444' }; // red-500
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="card-title-scaled card-title-with-icon flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Brain className="card-icon-scaled" />
            <span>AI Price Prediction for {symbol}</span>
          </div>
          <div className="flex gap-1">
            {HORIZON_OPTIONS.map(option => (
              <Button
                key={option.key}
                variant={selectedHorizon === option.key ? 'default' : 'outline'}
                size="sm"
                onClick={() => handleHorizonChange(option.key)}
              >
                {option.label}
              </Button>
            ))}
          </div>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {loading ? (
          // Show loading skeleton
          <div className="space-y-3">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : error ? (
          // Show error message
          <Alert className="prediction-alert">
            <AlertTriangle className="prediction-alert-icon" />
            <AlertDescription className="prediction-alert-text">{error}</AlertDescription>
          </Alert>
        ) : !displayPrediction || !symbol || !currentPrice ? (
          // Show placeholder when no stock is selected OR when waiting for live price
          <div className="text-center py-8">
            <p className="text-muted-foreground text-xl">
              {!symbol ? 'Select a stock to see price predictions' : 'Loading live price data...'}
            </p>
          </div>
        ) : (
          // Show actual prediction (only when we have live price)
          <>
            {/* Prediction Alert */}
            <Alert className="prediction-alert">
              <AlertTriangle className="prediction-alert-icon" />
              <AlertDescription className="prediction-alert-text">
                This prediction is based on historical data analysis using machine learning. 
                Market conditions can change rapidly and predictions may not reflect future performance.
              </AlertDescription>
            </Alert>

            {/* Data source sync indicator */}
            {displayPrediction && !displayPrediction.sourceReliable && displayPrediction.dataDate && (
              <div className="mb-3 px-3 py-2 bg-amber-50 border border-amber-200 rounded-md">
                <div className="flex items-center gap-2 text-sm text-amber-700">
                  <Database className="h-4 w-4" />
                  <span>Prediction based on offline data from {displayPrediction.dataDate}</span>
                </div>
              </div>
            )}

            {/* Main Prediction */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="prediction-metadata-label text-muted-foreground">Predicted Price</span>
                <div className="flex items-center gap-2">
                  {isPositiveChange ? (
                    <TrendingUp className="prediction-icon text-green-600" />
                  ) : (
                    <TrendingDown className="prediction-icon text-red-600" />
                  )}
                  <Badge variant={getConfidenceBadgeVariant(displayPrediction.confidence)} className="prediction-confidence-badge">
                    {displayPrediction.confidence ? displayPrediction.confidence.toFixed(1) : '0.0'}% confidence
                  </Badge>
                </div>
              </div>
              
              <div className="prediction-price-main">
                {formatPrice(displayPrediction.predictedPrice, currency)}
              </div>
              
              {currentPrice && (
                <div className={`prediction-change-info ${
                  isPositiveChange ? 'text-green-600' : 'text-red-600'
                }`}>
                  {isPositiveChange ? '+' : ''}{formatPrice(change, currency)} ({changePercent.toFixed(2)}%)
                </div>
              )}

              {/* Price data source indicator */}
              <div className="prediction-metadata-label text-muted-foreground mt-2 flex items-center gap-2 text-sm">
                <Database className="h-3 w-3" />
                <span>
                  {displayPrediction?.dataSource === 'upstox' ? 'Using live price data (Upstox)' :
                   displayPrediction?.dataSource === 'finnhub' ? 'Using live price data (finnhub)' :
                   displayPrediction?.dataSource === 'permanent' ? 'Using historical price data' :
                   dataSource === 'upstox' ? 'Using live price data (Upstox)' :
                   dataSource === 'finnhub' ? 'Using live price data (finnhub)' :
                   dataSource === 'permanent' ? 'Using historical price data' :
                   'Using live price data'}
                </span>
                <Clock className="h-3 w-3 ml-2" />
                <span>{dataTimestamp ? new Date(dataTimestamp).toLocaleTimeString() : new Date().toLocaleTimeString()}</span>
              </div>
            </div>

            {/* Confidence Indicator */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="prediction-metadata-label text-muted-foreground">Prediction Confidence</span>
                <span className={`prediction-metadata-value ${getConfidenceColor(displayPrediction.confidence)}`}>
                  {displayPrediction.confidence ? displayPrediction.confidence.toFixed(1) : '0.0'}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2" style={{ backgroundColor: '#e5e7eb' }}>
                <div 
                  className={`h-2 rounded-full transition-all duration-300 ${getProgressBarColor(displayPrediction.confidence)}`}
                  style={{ 
                    width: `${displayPrediction.confidence || 0}%`,
                    ...getProgressBarStyle(displayPrediction.confidence)
                  }}
                />
              </div>
            </div>
          </>
        )}
      </CardContent>
      
      {/* Model Selection */}
      <div className="px-6 pb-16">
        <div className="space-y-4">
          <h3 className="text-2xl font-bold text-foreground">Models</h3>
          
          {/* Basic Models */}
          <div className="space-y-2">
            <button
              onClick={() => setShowBasic(!showBasic)}
              className="flex items-center gap-2 text-base font-medium text-muted-foreground uppercase tracking-wide hover:text-foreground transition-colors"
            >
              {showBasic ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              Basic Models
            </button>
            {showBasic && (
              <div className="space-y-2">
                {/* Single row - all 4 buttons */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.basic.map(model => (
                    <Button
                      key={model.key}
                      variant={selectedModel === model.key ? 'default' : 'outline'}
                      size="default"
                      onClick={() => handleModelToggle(model.key)}
                      className="text-lg model-toggle-button"
                    >
                      {model.label}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Advanced Models */}
          <div className="space-y-2 mb-12">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-base font-medium text-muted-foreground uppercase tracking-wide hover:text-foreground transition-colors"
            >
              {showAdvanced ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              Advanced Models
            </button>
            {showAdvanced && (
              <div className="space-y-2">
                {/* Single row - all 3 advanced models */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.advanced.map(model => (
                    <Button
                      key={model.key}
                      variant={selectedModel === model.key ? 'default' : 'outline'}
                      size="default"
                      onClick={() => handleModelToggle(model.key)}
                      className="text-lg model-toggle-button"
                    >
                      {model.label}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Bottom separator and spacing */}
          <div className="border-t border-gray-200 mt-6 pt-6"></div>
        </div>
      </div>
    </Card>
  );
}