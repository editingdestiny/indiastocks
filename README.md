# 📊 Indian Stock Market Dashboard

A comprehensive, modern, and responsive dashboard for analyzing Indian stock market data from NSE (National Stock Exchange) with 10 years of historical data, advanced analytics, predictive modeling, and fundamental analysis.

![Dashboard Preview](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Dash](https://img.shields.io/badge/Dash-Latest-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Key Features

### 📈 Price Analysis
- **Interactive Charts**: Real-time closing price history, trading volume, and OHLC candlestick charts
- **Multiple Timeframes**: 1 Day, 1 Week, 1 Month, 3 Months, 6 Months, 1 Year, 3 Years, 5 Years, and All Time
- **Comprehensive Metrics**: Latest close price, returns, annualized returns, high/low/average prices, volatility
- **2000+ Stocks**: Complete coverage of NSE-listed stocks with daily updates

### 🏆 Market Performers
- **Top 10 & Bottom 10**: Real-time ranking by selected timeframe
- **Color-Coded Returns**: Visual indicators (green for gains, red for losses)
- **Dynamic Updates**: Automatically refreshes with latest data

### 🤖 Predictive Analysis (LSTM AI Model)
- **90-Day Price Predictions**: Advanced LSTM neural network for price forecasting
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Prediction Tracking**: Historical accuracy tracking and performance metrics
- **Visual Forecasts**: Interactive charts showing predicted vs actual prices
- **Confidence Metrics**: Model performance indicators and validation results

### 🔄 Backtesting Engine
- **Strategy Testing**: Test trading strategies on historical data
- **Multiple Algorithms**: Buy-and-Hold, Moving Average Crossover, RSI-based, MACD-based strategies
- **Performance Metrics**: Total return, Sharpe ratio, max drawdown, win rate
- **Visual Results**: Equity curves and trade markers on price charts
- **Risk Analysis**: Comprehensive risk-adjusted returns

### 💼 Fundamental Analysis
- **Top 10 Best Fundamentals**: Stocks with strongest financial health
- **Bottom 10 Weakest Fundamentals**: Identify underperforming stocks
- **Composite Scoring**: Multi-factor analysis including:
  - Return on Equity (ROE) - 30% weight
  - Profit Margins - 20% weight
  - Return on Assets (ROA) - 15% weight
  - Current Ratio - 10% weight
  - Debt to Equity - 15% weight
  - Earnings Growth - 10% weight
- **Detailed Metrics**: P/E ratios, market cap, EPS, dividend yield, financial ratios
- **4 Category Views**: Valuation, Profitability, Dividend & Growth, Financial Health

### 🔄 Automatic Data Updates
- **Daily Price Updates**: Automated cron job runs weekdays at 6:15 PM
- **Smart Cache Invalidation**: Dashboard automatically refreshes after data updates
- **Batch Processing**: Efficient downloading of 2000+ stocks in batches
- **Backup System**: Automatic timestamped backups before each update
- **Error Handling**: Robust retry logic and logging

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- 4GB+ RAM recommended
- Port 8060 available

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/editingdestiny/indiastocks.git
   cd indiastocks
   ```

2. **Add your initial data** (Optional - first-time setup)
   - Place your `nse_all_10y.csv` file in the project root
   - Or let the update script fetch latest data on first run

3. **Build and start the application**
   ```bash
   docker-compose up -d --build
   ```

4. **Access the dashboard**
   - Open your browser: `http://localhost:8060/indiastock/`
   - Wait for initial data load (~30 seconds)

5. **Update fundamental data** (One-time setup)
   ```bash
   docker exec -w /app indiastock python update_fundamentals.py
   ```

## 🐳 Docker Architecture

The application runs in a containerized environment with:

### Services
- **Dashboard**: Dash web application on port 8060
- **Cron Service**: Automated daily data updates
- **Data Processing**: Pandas-based ETL pipeline

### Container Structure
```
indiastock/
├── Dashboard (Dash + Flask)
│   ├── Port: 8060
│   └── Path: /indiastock/
├── LSTM Models (TensorFlow)
├── Data Cache (4-hour TTL)
├── Cron Jobs (Daily updates)
└── Backup System
```

### Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker logs indiastock --tail 100 -f

# Restart container
docker restart indiastock

# Stop services
docker-compose down

# Force cache refresh
docker exec indiastock touch /app/.cache_invalidate
docker restart indiastock

# Manual data update
docker exec indiastock python /app/update_daily.py

# Update fundamentals
docker exec indiastock python /app/update_fundamentals.py

# Check update logs
docker exec indiastock tail -50 /app/logs/update_daily.log
```

## 📁 Project Structure

```
indiastocks/
├── dashboard.py              # Main Dash application
├── predictive_analysis.py    # LSTM prediction module
├── lstm_model.py            # Neural network architecture
├── backtesting.py           # Strategy backtesting engine
├── fundamentals.py          # Fundamental analysis module
├── prediction_tracker.py    # Prediction accuracy tracking
├── update_daily.py          # Daily price update script
├── update_fundamentals.py   # Fundamental data fetcher
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Docker Compose setup
├── start_services.sh        # Service orchestration
├── run_update.sh            # Update wrapper script
├── docker/
│   └── update_cron          # Cron schedule (Mon-Fri 18:15)
├── logs/
│   └── update_daily.log     # Update logs
├── backups/                 # Timestamped data backups
├── nse_all_10y.csv         # Historical price data
├── fundamentals_data.csv   # Fundamental metrics (2002 stocks)
└── prediction_history.json # Prediction tracking data
```

## 🛠️ Technology Stack

### Frontend & Visualization
- **Dash (Plotly)**: Interactive web dashboards
- **Plotly.js**: Advanced charting library
- **HTML/CSS**: Responsive design with custom styling

### Backend & Processing
- **Python 3.11**: Core language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **yfinance**: Yahoo Finance data fetching

### Machine Learning
- **TensorFlow 2.x**: Deep learning framework
- **LSTM Networks**: Time series prediction
- **Scikit-learn**: Model evaluation and preprocessing

### Infrastructure
- **Docker**: Containerization
- **Cron**: Scheduled tasks
- **Linux (Debian)**: Base OS

## 📊 Data Architecture

### Price Data Format
```
CSV Structure (Multi-level headers):
Level 0: Ticker (RELIANCE.NS, TCS.NS, ...)
Level 1: OHLC + Volume (Open, High, Low, Close, Volume)
Level 2: Dates (2015-11-12, 2015-11-13, ...)

Rows: 2473+ daily records
Columns: 10,000+ (2002 stocks × 5 metrics)
Size: ~271MB
```

### Fundamental Data Format
```
CSV Structure (Flat):
ticker, trailing_pe, forward_pe, price_to_book, market_cap,
enterprise_value, trailing_eps, forward_eps, dividend_yield,
payout_ratio, profit_margins, operating_margins, return_on_equity,
return_on_assets, revenue_growth, earnings_growth, current_ratio,
quick_ratio, debt_to_equity, book_value, fifty_two_week_high,
fifty_two_week_low, beta, shares_outstanding, last_updated

Rows: 2002 stocks
Size: ~500KB
```

## 🎯 Feature Highlights

### 1. Real-Time Cache Management
- 4-hour cache timeout
- Signal-based invalidation after updates
- Automatic refresh on data changes
- Zero downtime during updates

### 2. LSTM Prediction Model
- **Architecture**: 3-layer LSTM (128, 64, 32 units)
- **Training**: 80/20 train-test split
- **Features**: 60-day lookback window
- **Normalization**: MinMax scaling (0-1 range)
- **Optimization**: Adam optimizer with MSE loss

### 3. Technical Indicators
- **Moving Averages**: 20-day, 50-day, 200-day SMA
- **RSI**: 14-day Relative Strength Index
- **MACD**: 12/26/9 periods
- **Bollinger Bands**: 20-day, 2 standard deviations
- **Volume Analysis**: Average volume trends

### 4. Backtesting Strategies

#### Buy and Hold
- Simple baseline strategy
- Long-only position
- Benchmark for comparison

#### Moving Average Crossover
- Golden Cross: 50-day crosses above 200-day (Buy)
- Death Cross: 50-day crosses below 200-day (Sell)

#### RSI Strategy
- Oversold: RSI < 30 (Buy signal)
- Overbought: RSI > 70 (Sell signal)

#### MACD Strategy
- Bullish: MACD crosses above signal (Buy)
- Bearish: MACD crosses below signal (Sell)

### 5. Fundamental Scoring Algorithm
```python
Score = (ROE × 30%) + (Profit Margin × 20%) + (ROA × 15%) +
        (Current Ratio × 10%) + (Debt/Equity × 15%) + (Earnings Growth × 10%)

Criteria:
- ROE: Higher is better (capped at 30%)
- Profit Margin: Higher is better (capped at 20%)
- ROA: Higher is better (capped at 15%)
- Current Ratio: Optimal 1.5-3.0 (10 points), >1.0 (5 points)
- Debt/Equity: <0.5 (15 pts), <1.0 (10 pts), <2.0 (5 pts)
- Earnings Growth: Higher is better (capped at 10%)
```

## 🔧 Configuration

### Environment Variables
```bash
# Container timezone
TZ=Asia/Kolkata

# Cache settings (in dashboard.py)
CACHE_TIMEOUT=14400  # 4 hours in seconds

# Update schedule (in docker/update_cron)
15 18 * * 1-5  # Mon-Fri at 6:15 PM
```

### Ports
- **8060**: Main dashboard interface
- **Internal**: All processing happens within container

### Data Update Schedule
```
Monday-Friday: 18:15 IST (6:15 PM)
- Fetches previous day's closing data
- Creates timestamped backup
- Updates nse_all_10y.csv
- Triggers cache invalidation
- Logs to /app/logs/update_daily.log
```

## 📈 Usage Guide

### 1. Price Analysis Tab
1. Select a stock from the dropdown (2000+ options)
2. Choose your timeframe (1D to All Time)
3. View comprehensive metrics and charts
4. Analyze volume trends and OHLC patterns

### 2. Predictive Analysis Tab
1. Select a stock
2. Wait for LSTM model training (~30 seconds)
3. View 90-day price predictions
4. Analyze technical indicators
5. Review prediction confidence metrics

### 3. Backtesting Tab
1. Select a stock
2. Choose a backtesting strategy
3. View performance metrics:
   - Total return
   - Sharpe ratio
   - Maximum drawdown
   - Win rate
   - Number of trades
4. Analyze equity curve and trade markers

### 4. Fundamentals Tab
1. View top 10 best fundamental stocks
2. View bottom 10 weakest stocks
3. Select a stock for detailed analysis
4. Review 4 categories of metrics:
   - Valuation (P/E, P/B, Market Cap)
   - Profitability (ROE, ROA, Margins)
   - Dividend & Growth (Yield, Growth rates)
   - Financial Health (Ratios, Debt levels)

## 🎨 Design Philosophy

### Modern UI/UX
- **Color Scheme**: Purple/Blue gradient (#667eea → #764ba2)
- **Card Design**: Gradient backgrounds with layered shadows
- **Typography**: Responsive font sizing (clamp)
- **Icons**: Emoji-based visual indicators
- **Animations**: Smooth transitions and hover effects

### Responsive Design
```css
Mobile (< 768px): Single column, stacked charts
Tablet (768-1024px): 2-column grid, optimized spacing
Desktop (> 1024px): Full layout with side-by-side charts
```

### Accessibility
- High contrast ratios (WCAG AA compliant)
- Clear visual hierarchy
- Readable font sizes
- Color-blind friendly palettes

## 🔍 Performance Metrics

### Data Loading
- Initial load: ~5-10 seconds (2473 rows, 10,000+ columns)
- Cache hit: <100ms
- Stock selection: <500ms

### Model Training
- LSTM training: 20-40 seconds per stock
- Prediction generation: 1-2 seconds
- Technical indicators: <1 second

### Data Updates
- Daily update: ~2-3 minutes (2002 stocks)
- Batch size: 50 stocks per request
- Cache invalidation: <100ms

## 🚨 Troubleshooting

### Dashboard not loading
```bash
# Check container status
docker ps | grep indiastock

# View logs
docker logs indiastock --tail 50

# Restart container
docker restart indiastock
```

### Data not updating
```bash
# Check cron service
docker exec indiastock service cron status

# View update logs
docker exec indiastock cat /app/logs/update_daily.log

# Manual update
docker exec indiastock python /app/update_daily.py
```

### Cache not refreshing
```bash
# Force cache invalidation
docker exec indiastock touch /app/.cache_invalidate

# Restart dashboard
docker restart indiastock
```

### Fundamentals not showing
```bash
# Check if fundamentals file exists
docker exec indiastock ls -lh /app/fundamentals_data.csv

# Update fundamentals data
docker exec indiastock python /app/update_fundamentals.py
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Contribution
- [ ] Additional technical indicators
- [ ] More backtesting strategies
- [ ] Enhanced prediction models (GRU, Transformer)
- [ ] Portfolio optimization features
- [ ] Real-time alerts and notifications
- [ ] Mobile app (React Native)
- [ ] API rate limiting and caching
- [ ] User authentication
- [ ] Watchlist management
- [ ] Export to PDF/Excel

### Development Setup
```bash
# Clone and create branch
git clone https://github.com/editingdestiny/indiastocks.git
cd indiastocks
git checkout -b feature/your-feature-name

# Make changes and test
docker-compose up --build

# Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for functions
- Comment complex logic
- Test before submitting PR

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NSE India**: Data source
- **Yahoo Finance**: Real-time data API
- **Plotly/Dash**: Visualization framework
- **TensorFlow**: Machine learning framework
- **Docker Community**: Containerization platform
- **Open Source Community**: Various Python libraries

## 📞 Support & Contact

### Reporting Issues
- GitHub Issues: [Create an issue](https://github.com/editingdestiny/indiastocks/issues)
- Provide clear description and steps to reproduce
- Include logs and error messages

### Feature Requests
- Open a GitHub issue with [Feature Request] tag
- Describe the feature and use case
- Explain expected behavior

### Questions
- Check existing issues and documentation first
- Open a discussion on GitHub
- Provide context and examples

## 🗺️ Roadmap

### Version 2.0 (Planned)
- [ ] Real-time WebSocket data streaming
- [ ] Advanced options analytics
- [ ] Sector-wise analysis
- [ ] Market breadth indicators
- [ ] Heat maps and correlation matrices
- [ ] Portfolio tracking and management
- [ ] Multi-user support with authentication
- [ ] Custom alerts and notifications
- [ ] Mobile-responsive improvements
- [ ] API rate limiting and optimization

### Version 3.0 (Future)
- [ ] News sentiment analysis
- [ ] Social media sentiment tracking
- [ ] Automated trading signals
- [ ] Integration with broker APIs
- [ ] Advanced risk management tools
- [ ] Machine learning model marketplace
- [ ] Community-driven strategies
- [ ] Educational content and tutorials

## 📊 Statistics

- **Stocks Covered**: 2,002 NSE-listed companies
- **Historical Data**: 10 years (2015-2025)
- **Daily Updates**: Automated on weekdays
- **Fundamental Metrics**: 23 per stock
- **Technical Indicators**: 8+ indicators
- **Backtesting Strategies**: 4 built-in
- **Prediction Horizon**: 90 days
- **Update Frequency**: Daily at 6:15 PM IST

---

**Made with ❤️ for Indian stock market investors and traders**

*Disclaimer: This dashboard is for educational and informational purposes only. Not financial advice. Always do your own research and consult with financial advisors before making investment decisions.*
