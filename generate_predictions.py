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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_FILE = "/app/nse_all_10y.csv"
OUTPUT_FILE = "/app/predictions_cache.csv"
BACKUP_DIR = "/app/backups"
PREDICTION_DAYS = 90
MIN_DATA_POINTS = 100  # Minimum data points required for training

def load_stock_data():
    """Load the main stock data file."""
    try:
        logger.info(f"Loading data from {DATA_FILE}")
        df = pd.read_csv(DATA_FILE, header=[0, 1, 2])
        
        # Extract dates
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col], errors='coerce')
        
        # Extract stock list
        stock_list = [col for col in df.columns.get_level_values(0).unique() if col != 'Ticker']
        
        logger.info(f"Loaded {len(df)} rows with {len(stock_list)} stocks")
        return df, dates, stock_list
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None


def get_stock_prices(df, dates, ticker):
    """Extract close prices for a specific stock."""
    try:
        close_col = (ticker, 'Close', 'Close')
        if close_col not in df.columns:
            return None
        
        stock_data = pd.DataFrame({
            'Date': dates,
            'Close': df[close_col].values
        })
        
        stock_data = stock_data.dropna()
        return stock_data if len(stock_data) >= MIN_DATA_POINTS else None
    except Exception as e:
        logger.error(f"Error extracting data for {ticker}: {e}")
        return None


def generate_prediction_for_stock(stock_data, ticker):
    """Generate LSTM prediction for a single stock."""
    try:
        logger.info(f"Training model for {ticker}")
        
        # Prepare data
        close_prices = stock_data['Close'].values
        
        # Train LSTM model
        model, scaler, X_test, y_test, train_size = lstm_model.train_lstm_model(
            close_prices, 
            lookback=60
        )
        
        if model is None:
            logger.warning(f"Model training failed for {ticker}")
            return None
        
        # Generate predictions
        last_sequence = close_prices[-60:]
        predictions = lstm_model.predict_future(
            model, 
            scaler, 
            last_sequence, 
            days=PREDICTION_DAYS
        )
        
        if predictions is None or len(predictions) == 0:
            logger.warning(f"Prediction generation failed for {ticker}")
            return None
        
        # Get last date and generate future dates
        last_date = stock_data['Date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=PREDICTION_DAYS,
            freq='D'
        )
        
        # Calculate basic metrics
        last_price = close_prices[-1]
        predicted_change = ((predictions[-1] - last_price) / last_price) * 100
        
        # Create result dictionary
        result = {
            'ticker': ticker,
            'last_date': last_date.strftime('%Y-%m-%d'),
            'last_price': float(last_price),
            'predicted_price_30d': float(predictions[29]) if len(predictions) > 29 else None,
            'predicted_price_60d': float(predictions[59]) if len(predictions) > 59 else None,
            'predicted_price_90d': float(predictions[-1]),
            'predicted_change_pct': float(predicted_change),
            'prediction_dates': ','.join([d.strftime('%Y-%m-%d') for d in future_dates]),
            'prediction_prices': ','.join([f"{p:.2f}" for p in predictions]),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"✓ {ticker}: Last={last_price:.2f}, 90d={predictions[-1]:.2f}, Change={predicted_change:.2f}%")
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
        
        # Create cache invalidation signal for dashboard
        cache_signal_file = "/app/.predictions_cache_ready"
        with open(cache_signal_file, 'w') as f:
            f.write(str(datetime.now()))
        logger.info("Created predictions cache signal")
        
        return True
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        return False


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Starting prediction generation")
    logger.info("="*60)
    
    # Load data
    df, dates, stock_list = load_stock_data()
    if df is None:
        logger.error("Failed to load stock data")
        return 1
    
    # Generate predictions for all stocks
    predictions_list = []
    failed_stocks = []
    
    for i, ticker in enumerate(stock_list, 1):
        try:
            logger.info(f"\n[{i}/{len(stock_list)}] Processing {ticker}")
            
            # Get stock data
            stock_data = get_stock_prices(df, dates, ticker)
            if stock_data is None:
                logger.warning(f"Insufficient data for {ticker}")
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
