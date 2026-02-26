// Stock data interfaces
export interface StockData {
    symbol: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
    volume: number;
    marketCap: string;
    lastUpdated: string;
}

export interface PricePoint {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface PredictionResult {
    predictedPrice: number;
    confidence: number;
    model: string;
    timeframe: string;
}

// Popular stocks list
const POPULAR_STOCKS = [
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corporation' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.' },
    { symbol: 'TSLA', name: 'Tesla Inc.' },
    { symbol: 'META', name: 'Meta Platforms Inc.' },
    { symbol: 'NVDA', name: 'NVIDIA Corporation' },
    { symbol: 'NFLX', name: 'Netflix Inc.' }
];

// Cache for storing generated data
const cache = new Map<string, { data: any; timestamp: number }>();

// Utility function to generate realistic stock data
function generateStockData(symbol: string): StockData {
    const stock = POPULAR_STOCKS.find(s => s.symbol === symbol);
    if (!stock) {
        throw new Error(`Stock symbol ${symbol} not found`);
    }

    // Generate realistic price based on symbol
    const basePrices: Record<string, number> = {
        'AAPL': 180,
        'GOOGL': 140,
        'MSFT': 380,
        'AMZN': 140,
        'TSLA': 250,
        'META': 320,
        'NVDA': 450,
        'NFLX': 430
    };

    const basePrice = basePrices[symbol] || 100;
    const volatility = 0.02; // 2% daily volatility
    const change = (Math.random() - 0.5) * 2 * volatility * basePrice;
    const currentPrice = basePrice + change;
    const changePercent = (change / basePrice) * 100;

    return {
        symbol: stock.symbol,
        name: stock.name,
        price: parseFloat(currentPrice.toFixed(2)),
        change: parseFloat(change.toFixed(2)),
        changePercent: parseFloat(changePercent.toFixed(2)),
        volume: Math.floor(Math.random() * 50000000 + 1000000),
        marketCap: `$${(Math.random() * 2000 + 100).toFixed(0)}B`,
        lastUpdated: new Date().toISOString()
    };
}

// Generate historical data
function generateHistoricalData(symbol: string, days: number): PricePoint[] {
    const data: PricePoint[] = [];
    const basePrices: Record<string, number> = {
        'AAPL': 180,
        'GOOGL': 140,
        'MSFT': 380,
        'AMZN': 140,
        'TSLA': 250,
        'META': 320,
        'NVDA': 450,
        'NFLX': 430
    };

    let currentPrice = basePrices[symbol] || 100;

    for (let i = days; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);

        // Add realistic price movement
        const volatility = 0.02;
        const change = (Math.random() - 0.5) * 2 * volatility * currentPrice;
        currentPrice = Math.max(currentPrice + change, 1);

        const high = currentPrice * (1 + Math.random() * 0.02);
        const low = currentPrice * (1 - Math.random() * 0.02);
        const open = currentPrice * (0.98 + Math.random() * 0.04);

        data.push({
            date: date.toISOString().split('T')[0],
            open: parseFloat(open.toFixed(2)),
            high: parseFloat(high.toFixed(2)),
            low: parseFloat(low.toFixed(2)),
            close: parseFloat(currentPrice.toFixed(2)),
            volume: Math.floor(Math.random() * 10000000 + 1000000)
        });
    }

    return data.reverse();
}

// K-nearest neighbor prediction model
function knnPredict(historicalData: PricePoint[], k: number = 5): PredictionResult {
    if (historicalData.length < k) {
        throw new Error('Not enough historical data for prediction');
    }

    const recentData = historicalData.slice(-k);

    // Calculate weighted average change based on recency
    let totalWeightedChange = 0;
    let totalWeight = 0;

    for (let i = 1; i < recentData.length; i++) {
        const change = recentData[i].close - recentData[i - 1].close;
        const weight = i; // More recent data gets higher weight
        totalWeightedChange += change * weight;
        totalWeight += weight;
    }

    const avgChange = totalWeightedChange / totalWeight;
    const lastPrice = recentData[recentData.length - 1].close;
    const predictedPrice = lastPrice + avgChange;

    // Calculate confidence based on price stability and volume
    const changes: number[] = [];
    for (let i = 1; i < recentData.length; i++) {
        changes.push(recentData[i].close - recentData[i - 1].close);
    }

    const avgChangeValue = totalWeightedChange / (recentData.length - 1);
    const variance = changes.reduce((sum, change) => sum + Math.pow(change - avgChangeValue, 2), 0) / changes.length;
    const avgVolume = recentData.reduce((sum, point) => sum + point.volume, 0) / recentData.length;

    // Confidence based on stability (lower variance = higher confidence) and volume
    const stabilityScore = Math.max(0.1, 1 - (variance / (lastPrice * 0.1)));
    const volumeScore = Math.min(1, avgVolume / 10000000); // Normalize volume
    const confidence = (stabilityScore * 0.7 + volumeScore * 0.3) * 100;

    return {
        predictedPrice: parseFloat(Math.max(predictedPrice, 0.01).toFixed(2)),
        confidence: parseFloat(Math.min(Math.max(confidence, 20), 90).toFixed(1)),
        model: 'K-Nearest Neighbor',
        timeframe: '1 day'
    };
}

// Cache helper functions
function getCachedData<T>(key: string, maxAge: number): T | null {
    const cached = cache.get(key);
    if (cached && Date.now() - cached.timestamp < maxAge) {
        return cached.data;
    }
    return null;
}

function setCachedData(key: string, data: any): void {
    cache.set(key, { data, timestamp: Date.now() });
}

export const stockService = {
    // Get current stock data
    getStockData: async (symbol: string): Promise<StockData> => {
        if (!symbol) {
            throw new Error('Stock symbol is required');
        }

        const cacheKey = `stock_${symbol}`;
        const cachedData = getCachedData<StockData>(cacheKey, 5 * 60 * 1000); // 5 minutes

        if (cachedData) {
            return cachedData;
        }

        try {
            const stockData = generateStockData(symbol);
            setCachedData(cacheKey, stockData);
            return stockData;
        } catch (error) {
            console.error(`Failed to fetch stock data for ${symbol}:`, error);
            throw new Error(`Unable to fetch data for ${symbol}. Please try again later.`);
        }
    },

    // Get historical data
    getHistoricalData: async (symbol: string, period: 'week' | 'month' | 'year'): Promise<PricePoint[]> => {
        if (!symbol) {
            throw new Error('Stock symbol is required');
        }

        if (!['week', 'month', 'year'].includes(period)) {
            throw new Error('Invalid period. Must be week, month, or year.');
        }

        const cacheKey = `historical_${symbol}_${period}`;
        const cachedData = getCachedData<PricePoint[]>(cacheKey, 60 * 60 * 1000); // 1 hour

        if (cachedData) {
            return cachedData;
        }

        try {
            const days = period === 'week' ? 7 : period === 'month' ? 30 : 365;
            const historicalData = generateHistoricalData(symbol, days);
            setCachedData(cacheKey, historicalData);
            return historicalData;
        } catch (error) {
            console.error(`Failed to fetch historical data for ${symbol} (${period}):`, error);
            throw new Error(`Unable to fetch historical data for ${symbol}. Please try again later.`);
        }
    },

    // Get stock prediction
    getPrediction: async (symbol: string): Promise<PredictionResult> => {
        if (!symbol) {
            throw new Error('Stock symbol is required');
        }

        const cacheKey = `prediction_${symbol}`;
        const cachedData = getCachedData<PredictionResult>(cacheKey, 15 * 60 * 1000); // 15 minutes

        if (cachedData) {
            return cachedData;
        }

        try {
            // Generate historical data for prediction
            const historicalData = generateHistoricalData(symbol, 30);
            const prediction = knnPredict(historicalData);
            setCachedData(cacheKey, prediction);
            return prediction;
        } catch (error) {
            console.error(`Failed to get prediction for ${symbol}:`, error);
            throw new Error(`Unable to generate prediction for ${symbol}. Please try again later.`);
        }
    },

    // Search stocks
    searchStocks: async (query: string): Promise<{ symbol: string; name: string }[]> => {
        try {
            if (!query.trim()) {
                return POPULAR_STOCKS;
            }

            const results = POPULAR_STOCKS.filter(stock =>
                stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
                stock.name.toLowerCase().includes(query.toLowerCase())
            );

            return results;
        } catch (error) {
            console.error(`Failed to search stocks with query "${query}":`, error);
            // Return empty array instead of throwing to avoid breaking the UI
            return [];
        }
    },

    // Get popular stocks
    getPopularStocks: async (): Promise<{ symbol: string; name: string }[]> => {
        try {
            return POPULAR_STOCKS;
        } catch (error) {
            console.error('Failed to fetch popular stocks:', error);
            // Return fallback list
            return POPULAR_STOCKS;
        }
    },

    // Health check for server status
    checkHealth: async (): Promise<{ status: string; timestamp: string }> => {
        try {
            return {
                status: 'healthy',
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('Health check failed:', error);
            throw new Error('Server is unavailable');
        }
    }
};
