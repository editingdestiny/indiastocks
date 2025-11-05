# 📊 Indian Stock Market Dashboard

A beautiful, modern, and responsive dashboard for analyzing Indian stock market data from NSE (National Stock Exchange) with 10 years of historical data.

![Dashboard Preview](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Dash](https://img.shields.io/badge/Dash-Latest-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## ✨ Features

### 📈 Price Analysis
- **Interactive Charts**: Closing price history, trading volume, and OHLC candlestick charts
- **Multiple Timeframes**: 1 Month, 3 Months, 6 Months, 1 Year, 3 Years, 5 Years, and All Time
- **Key Metrics**: Latest close price, returns, annualized returns, high/low/average prices
- **2000+ Stocks**: Comprehensive coverage of NSE-listed stocks

### 🏆 Market Performers
- Top 10 performing stocks by selected timeframe
- Worst 10 performing stocks
- Color-coded returns (green for positive, red for negative)

### 🎨 Modern UI/UX
- **Glossy Professional Design**: Gradient backgrounds, modern cards with shadows
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile
- **Color-Coded Metrics**: Visual indicators for gains and losses
- **Tab-Based Navigation**: Clean separation between Price Analysis and Predictive Analysis (coming soon)

### 🔮 Coming Soon
- Price prediction using machine learning models
- Trend analysis and forecasting
- Risk assessment metrics
- Buy/Sell recommendations

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- 10 years of NSE historical data (CSV format)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/editingdestiny/indiastock-dashboard.git
   cd indiastock-dashboard
   ```

2. **Add your data**
   - Place your `nse_all_10y.csv` file in the project root
   - The CSV should have multi-level headers: Ticker, Price Type (Open/High/Low/Close/Volume), Date

3. **Start the application**
   ```bash
   docker-compose up -d
   ```

4. **Access the dashboard**
   - Open your browser and navigate to: `http://localhost:8060/indiastock/`

## 🐳 Docker Setup

The application runs in a Docker container with the following services:
- **Dashboard**: Dash web application (port 8060)
- **API**: FastAPI backend (port 8010)
- **Cron**: Automated daily data updates

### Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop services
docker-compose down
```

## 📁 Project Structure

```
indiastock/
├── dashboard.py          # Main Dash application
├── api.py               # FastAPI backend
├── update_daily.py      # Daily data update script
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── start_services.sh    # Service startup script
├── run_update.sh        # Update script wrapper
├── docker/
│   └── update_cron      # Cron job configuration
└── nse_all_10y.csv     # Historical data (not included in repo)
```

## 🛠️ Technology Stack

- **Frontend**: Dash (Plotly) - Interactive web dashboards
- **Backend**: FastAPI - Modern Python API framework
- **Charts**: Plotly - Interactive visualizations
- **Data Processing**: Pandas - Data manipulation
- **Deployment**: Docker & Docker Compose
- **Web Server**: Uvicorn (ASGI)

## 📊 Data Format

The dashboard expects a CSV file with the following structure:
- **Multi-level headers**: 
  - Level 0: Ticker symbols (e.g., RELIANCE.NS, TCS.NS)
  - Level 1: Price types (Open, High, Low, Close, Volume)
  - Level 2: Dates
- **Rows**: Daily price data

## 🎨 Design Features

- **Color Scheme**: Purple/Blue gradient primary theme (#667eea → #764ba2)
- **Modern Cards**: Gradient backgrounds with multi-layer shadows
- **Responsive**: Mobile-first design with CSS media queries
- **Accessibility**: Clear contrast ratios and readable fonts
- **Icons**: Emoji-based visual indicators

## 🔧 Configuration

### Environment Variables (Optional)
You can customize the following in your environment:
- `TZ`: Timezone (default: Asia/Kolkata)
- Data update schedule via `docker/update_cron`

### Ports
- Dashboard: 8060
- API: 8010

## 📈 Usage

1. **Select a Stock**: Choose from 2000+ NSE-listed stocks
2. **Choose Timeframe**: Select your analysis period
3. **View Analytics**: 
   - Market performers at the top
   - Detailed stock metrics and statistics
   - Interactive charts: OHLC, Closing Price, Volume
4. **Switch Tabs**: Navigate between Price Analysis and Predictive Analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- NSE (National Stock Exchange of India) for data
- Plotly/Dash community for the amazing framework
- All contributors and users of this dashboard

## 📞 Contact

For questions, suggestions, or issues, please open an issue on GitHub.

---

**Made with ❤️ for the Indian stock market community**
