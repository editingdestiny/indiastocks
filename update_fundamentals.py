"""
Update Fundamentals Data
Fetches fundamental data for all stocks and saves to CSV
"""

import yfinance as yf
import pandas as pd
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_fundamentals_for_all_stocks():
    """Fetch fundamental data for all stocks in nse_all_10y.csv"""
    
    # Read the main CSV to get list of stocks
    logger.info("Reading stock list from nse_all_10y.csv...")
    df = pd.read_csv('nse_all_10y.csv')
    
    # Get unique stock tickers from column names
    stocks = []
    for col in df.columns:
        if col not in ['Date', 'date']:
            ticker = col.split('_')[0] if '_' in col else col
            if ticker not in stocks and ticker.endswith('.NS'):
                stocks.append(ticker)
    
    logger.info(f"Found {len(stocks)} unique stocks")
    
    fundamentals_data = []
    
    for i, ticker in enumerate(stocks, 1):
        try:
            logger.info(f"Fetching {i}/{len(stocks)}: {ticker}")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fund_data = {
                'ticker': ticker,
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'price_to_book': info.get('priceToBook'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'trailing_eps': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'profit_margins': info.get('profitMargins'),
                'operating_margins': info.get('operatingMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'book_value': info.get('bookValue'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'beta': info.get('beta'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'last_updated': datetime.now().isoformat()
            }
            
            fundamentals_data.append(fund_data)
            
            # Rate limiting - be nice to Yahoo Finance
            if i % 10 == 0:
                time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            # Add empty record so we track which stocks failed
            fundamentals_data.append({
                'ticker': ticker,
                'last_updated': datetime.now().isoformat()
            })
    
    # Save to CSV
    fundamentals_df = pd.DataFrame(fundamentals_data)
    output_file = 'fundamentals_data.csv'
    fundamentals_df.to_csv(output_file, index=False)
    logger.info(f"Saved fundamental data to {output_file}")
    logger.info(f"Total stocks processed: {len(fundamentals_data)}")
    
    return fundamentals_df

if __name__ == '__main__':
    fetch_fundamentals_for_all_stocks()
