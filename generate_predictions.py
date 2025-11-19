#!/usr/bin/env python3
"""
generate_predictions.py

Generates LSTM predictions for all stocks and saves to CSV.
Runs daily at midnight via cron to pre-compute predictions.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import lstm_model
import predictive_analysis as pred
from prediction_tracker import prediction_tracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_FILE = "/app/nse_all_10y.csv"
OUTPUT_FILE = "/app/predictions_cache.csv"
PREDICTION_HISTORY_FILE = "/app/prediction_history.json"
BACKUP_DIR = "/app/backups"
PREDICTION_DAYS = 90
MIN_DATA_POINTS = 100  # Minimum data points required for training

# Target dates for predictions (starting from Oct 1, 2025)
PREDICTION_START_DATE = datetime(2025, 10, 1)  # Oct 1, 2025

def load_stock_data():
    """Load the main stock data CSV file with multi-level headers."""
    csv_path = '/app/nse_all_10y.csv'
    logger.info(f"Loading stock data from {csv_path}...")
    
    try:
        # Read with multi-level header (3 rows: Ticker, Price type, Date)
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Extract the date column (first column is Ticker/Price/Date)
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col], errors='coerce')
        
        # Get list of stock tickers (unique values from level 0, excluding 'Ticker')
        tickers = [col for col in df.columns.get_level_values(0).unique() if col != 'Ticker']
        logger.info(f"Found {len(tickers)} tickers")
        
        return df, dates, tickers
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None, None, []


def get_stock_prices(df, dates, ticker):
    """Extract close prices for a specific ticker from multi-level column structure."""
    try:
        # Find the Close column for this ticker
        ticker_cols = [col for col in df.columns if col[0] == ticker]
        if not ticker_cols:
            return None
        
        close_col = [col for col in ticker_cols if col[1] == 'Close']
        if not close_col:
            return None
        
        close_col = close_col[0]
        
        # Create a DataFrame with Date and Close columns
        stock_df = pd.DataFrame({
            'Date': dates,
            'Close': pd.to_numeric(df[close_col], errors='coerce')
        })
        
        # Drop NaN values
        stock_df = stock_df.dropna(subset=['Close'])
        
        if len(stock_df) < MIN_DATA_POINTS:
            return None
        
        return stock_df
        
    except Exception as e:
        logger.error(f"Error extracting data for {ticker}: {e}")
        return None


def generate_prediction_for_stock(stock_df, ticker):
    """Generate LSTM prediction for a single stock."""
    try:
        if stock_df is None or len(stock_df) < MIN_DATA_POINTS:
            logger.warning(f"Insufficient data for {ticker}")
            return None
        
        # Get the last known date and price
        last_date = pd.to_datetime(stock_df['Date'].iloc[-1])
        last_price = float(stock_df['Close'].iloc[-1])
        
        # Train LSTM model and generate predictions
        logger.info(f"  Training LSTM model for {ticker}...")
        lstm_results = lstm_model.train_lstm_model(
            stock_df,
            seq_length=60,
            forecast_days=PREDICTION_DAYS,
            epochs=20,  # Reduced from 50 to 20 for faster generation
            batch_size=32
        )
        
        # Check if prediction was successful
        if not lstm_results.get('success'):
            logger.warning(f"  ✗ LSTM training failed for {ticker}: {lstm_results.get('message', 'Unknown error')}")
            return None
        
        # Extract predictions
        forecast_dates = lstm_results['forecast_dates']
        forecast_values = lstm_results['forecast_values']
        
        # Get today's date for tracking
        prediction_made_date = datetime.now()
        
        # Store predictions in tracker for accuracy monitoring
        # Only store predictions for dates >= Oct 1, 2025
        model_metrics = lstm_results.get('metrics', {})
        for pred_date, pred_price in zip(forecast_dates, forecast_values):
            pred_date_dt = pd.to_datetime(pred_date)
            if pred_date_dt >= PREDICTION_START_DATE:
                prediction_tracker.save_prediction(
                    ticker=ticker,
                    prediction_date=prediction_made_date,
                    target_date=pred_date_dt,
                    predicted_price=pred_price,
                    model_info={
                        'rmse': model_metrics.get('rmse', 0),
                        'mae': model_metrics.get('mae', 0),
                        'train_loss': model_metrics.get('train_loss', 0),
                        'val_loss': model_metrics.get('val_loss', 0)
                    }
                )
        
        # Convert dates to strings for cache file
        pred_dates_str = ','.join([pd.to_datetime(d).strftime('%Y-%m-%d') for d in forecast_dates])
        pred_prices_str = ','.join([f'{p:.2f}' for p in forecast_values])
        
        # Store result as dictionary for the cache
        result = {
            'ticker': ticker,
            'last_date': last_date.strftime('%Y-%m-%d'),
            'last_price': last_price,
            'prediction_dates': pred_dates_str,
            'predicted_prices': pred_prices_str,
            'generated_at': prediction_made_date.strftime('%Y-%m-%d %H:%M:%S'),
            'rmse': model_metrics.get('rmse', 0),
            'mae': model_metrics.get('mae', 0)
        }
        
        logger.info(f"  ✓ Generated {PREDICTION_DAYS}-day predictions for {ticker} (RMSE: {model_metrics.get('rmse', 0):.2f})")
        return result
        
    except Exception as e:
        logger.error(f"Error generating prediction for {ticker}: {e}")
        return None


def backup_existing_predictions():
    """Backup existing predictions file."""
    if not os.path.exists(OUTPUT_FILE):
        return
    
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(BACKUP_DIR, f"predictions_cache.{timestamp}.bak")
        
        import shutil
        shutil.copy2(OUTPUT_FILE, backup_path)
        logger.info(f"Backed up predictions to {backup_path}")
    except Exception as e:
        logger.warning(f"Could not backup predictions: {e}")


def save_predictions(predictions_list):
    """Save predictions to CSV file."""
    try:
        # Backup existing file
        backup_existing_predictions()
        
        # Create DataFrame
        df = pd.DataFrame(predictions_list)
        
        # Save to CSV
        df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Saved {len(predictions_list)} predictions to {OUTPUT_FILE}")
        
        # Create cache ready signal for dashboard
        cache_signal_file = "/app/.predictions_cache_ready"
        with open(cache_signal_file, 'w') as f:
            f.write(str(datetime.now()))
        logger.info("Created predictions cache signal")
        
        return True
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        return False


def update_actual_prices_from_csv(df, dates):
    """Update actual prices in prediction tracker from current CSV data."""
    logger.info("Updating actual prices for past predictions...")
    
    try:
        # Get all tickers with predictions
        tickers_to_update = list(prediction_tracker.predictions.keys())
        logger.info(f"Checking {len(tickers_to_update)} tickers for actual prices")
        
        updated_count = 0
        for ticker in tickers_to_update:
            # Get stock data
            stock_data = get_stock_prices(df, dates, ticker)
            if stock_data is not None:
                # Batch update actual prices
                prediction_tracker.batch_update_actual_prices(ticker, stock_data)
                updated_count += 1
        
        logger.info(f"Updated actual prices for {updated_count} tickers")
        return True
    except Exception as e:
        logger.error(f"Error updating actual prices: {e}")
        return False


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Starting prediction generation")
    logger.info(f"Prediction start date: {PREDICTION_START_DATE.strftime('%Y-%m-%d')}")
    logger.info("="*60)
    
    # Load data
    df, dates, stock_list = load_stock_data()
    if df is None:
        logger.error("Failed to load stock data")
        return 1
    
    # First, update actual prices for any past predictions
    update_actual_prices_from_csv(df, dates)
    
    # Generate predictions for all stocks
    predictions_list = []
    failed_stocks = []
    
    for i, ticker in enumerate(stock_list, 1):
        try:
            logger.info(f"\n[{i}/{len(stock_list)}] Processing {ticker}")
            
            # Get stock data
            stock_data = get_stock_prices(df, dates, ticker)
            if stock_data is None:
                logger.warning(f"  ✗ Insufficient data for {ticker}")
                failed_stocks.append(ticker)
                continue
            
            # Generate prediction
            result = generate_prediction_for_stock(stock_data, ticker)
            if result:
                predictions_list.append(result)
            else:
                failed_stocks.append(ticker)
            
            # Progress update every 50 stocks
            if i % 50 == 0:
                success_rate = (len(predictions_list) / i) * 100
                logger.info(f"Progress: {i}/{len(stock_list)} ({success_rate:.1f}% success)")
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            failed_stocks.append(ticker)
            continue
    
    # Save results
    if predictions_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generated {len(predictions_list)} predictions")
        logger.info(f"Failed: {len(failed_stocks)} stocks")
        logger.info(f"Success rate: {(len(predictions_list)/len(stock_list)*100):.1f}%")
        logger.info(f"{'='*60}")
        
        if save_predictions(predictions_list):
            logger.info("✓ Prediction generation completed successfully")
            logger.info(f"✓ Prediction history saved to {PREDICTION_HISTORY_FILE}")
            return 0
        else:
            logger.error("✗ Failed to save predictions")
            return 1
    else:
        logger.error("No predictions generated")
        return 1


if __name__ == '__main__':
    start_time = datetime.now()
    logger.info(f"Started at {start_time}")
    
    try:
        exit_code = main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit_code = 1
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Completed at {end_time}")
    logger.info(f"Total duration: {duration}")
    
    sys.exit(exit_code)
