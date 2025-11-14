#!/usr/bin/env python3
"""
create_partial_cache.py

Extracts the latest predictions from prediction_history.json and creates
predictions_cache.csv for the dashboard to use instantly.
"""
import json
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PREDICTION_HISTORY_FILE = "/home/sd22750/indiastock/prediction_history.json"
CACHE_OUTPUT_FILE = "/home/sd22750/indiastock/predictions_cache.csv"
SIGNAL_FILE = "/home/sd22750/indiastock/.predictions_cache_ready"

def load_prediction_history():
    """Load the prediction history JSON file."""
    try:
        with open(PREDICTION_HISTORY_FILE, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded prediction history with {len(data)} tickers")
        return data
    except Exception as e:
        logger.error(f"Error loading prediction history: {e}")
        return {}

def extract_latest_predictions(history_data):
    """
    Extract the latest prediction for each ticker.
    
    For each ticker, find the most recent prediction_date and get all
    target_date predictions from that run.
    """
    cache_records = []
    
    for ticker, predictions in history_data.items():
        if not predictions:
            continue
        
        # Find the latest prediction_date
        latest_prediction_date = max(
            pred['prediction_date'] for pred in predictions
        )
        
        # Get all predictions from that date
        latest_predictions = [
            pred for pred in predictions 
            if pred['prediction_date'] == latest_prediction_date
        ]
        
        if not latest_predictions:
            continue
        
        # Sort by target_date
        latest_predictions.sort(key=lambda x: x['target_date'])
        
        # Extract prediction dates and prices
        prediction_dates = [pred['target_date'] for pred in latest_predictions]
        predicted_prices = [pred['predicted_price'] for pred in latest_predictions]
        
        # Get metadata from first prediction
        first_pred = latest_predictions[0]
        model_info = first_pred.get('model_info', {})
        
        # Build cache record matching the format expected by predictive_analysis.py
        # Need to calculate predicted prices at 30, 60, 90 days
        price_30d = predicted_prices[29] if len(predicted_prices) > 29 else predicted_prices[-1]
        price_60d = predicted_prices[59] if len(predicted_prices) > 59 else predicted_prices[-1]
        price_90d = predicted_prices[89] if len(predicted_prices) > 89 else predicted_prices[-1]
        
        # Calculate change percentage from last_price to 90-day prediction
        last_price = predicted_prices[0] if predicted_prices else 0.0
        change_pct = ((price_90d - last_price) / last_price * 100) if last_price > 0 else 0.0
        
        cache_record = {
            'ticker': ticker,
            'last_date': prediction_dates[0] if prediction_dates else '',
            'last_price': last_price,
            'predicted_price_30d': price_30d,
            'predicted_price_60d': price_60d,
            'predicted_price_90d': price_90d,
            'predicted_change_pct': change_pct,
            'prediction_dates': ','.join(prediction_dates),
            'prediction_prices': ','.join([f'{p:.2f}' for p in predicted_prices]),
            'generated_at': latest_prediction_date,
            'rmse': model_info.get('rmse', 0.0),
            'mae': model_info.get('mae', 0.0)
        }
        
        cache_records.append(cache_record)
    
    logger.info(f"Extracted {len(cache_records)} ticker predictions")
    return cache_records

def save_cache(cache_records):
    """Save cache records to CSV file."""
    try:
        df = pd.DataFrame(cache_records)
        df.to_csv(CACHE_OUTPUT_FILE, index=False)
        logger.info(f"✓ Saved {len(cache_records)} predictions to {CACHE_OUTPUT_FILE}")
        
        # Create signal file
        with open(SIGNAL_FILE, 'w') as f:
            f.write(str(datetime.now()))
        logger.info(f"✓ Created cache ready signal: {SIGNAL_FILE}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        return False

def main():
    logger.info("="*60)
    logger.info("Creating partial predictions cache")
    logger.info("="*60)
    
    # Load prediction history
    history_data = load_prediction_history()
    if not history_data:
        logger.error("No prediction history found")
        return 1
    
    # Extract latest predictions
    cache_records = extract_latest_predictions(history_data)
    if not cache_records:
        logger.error("No predictions to cache")
        return 1
    
    # Save to cache file
    if save_cache(cache_records):
        logger.info("="*60)
        logger.info("✓ Partial cache created successfully!")
        logger.info(f"  Tickers cached: {len(cache_records)}")
        logger.info(f"  Dashboard will now load these predictions instantly")
        logger.info("="*60)
        return 0
    else:
        logger.error("Failed to create cache")
        return 1

if __name__ == '__main__':
    exit(main())
