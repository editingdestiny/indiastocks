#!/usr/bin/env python3
"""
Fix the predictions cache CSV to match the expected format.
"""
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_FILE = "/home/sd22750/indiastock/predictions_cache.csv"

def fix_cache():
    """Fix the cache file format."""
    try:
        # Load existing cache
        df = pd.read_csv(CACHE_FILE)
        logger.info(f"Loaded cache with {len(df)} records")
        
        # Rename column if needed
        if 'predicted_prices' in df.columns and 'prediction_prices' not in df.columns:
            df = df.rename(columns={'predicted_prices': 'prediction_prices'})
            logger.info("Renamed predicted_prices to prediction_prices")
        
        # Add missing columns if they don't exist
        if 'predicted_price_30d' not in df.columns:
            logger.info("Adding predicted_price columns...")
            
            def extract_price_at_index(prices_str, index, default=None):
                try:
                    prices = [float(p) for p in prices_str.split(',')]
                    return prices[index] if len(prices) > index else (prices[-1] if prices else default)
                except:
                    return default
            
            df['predicted_price_30d'] = df['prediction_prices'].apply(lambda x: extract_price_at_index(x, 29))
            df['predicted_price_60d'] = df['prediction_prices'].apply(lambda x: extract_price_at_index(x, 59))
            df['predicted_price_90d'] = df['prediction_prices'].apply(lambda x: extract_price_at_index(x, 89))
        
        # Add predicted_change_pct if missing
        if 'predicted_change_pct' not in df.columns:
            logger.info("Calculating predicted_change_pct...")
            df['predicted_change_pct'] = ((df['predicted_price_90d'] - df['last_price']) / df['last_price'] * 100)
        
        # Save fixed cache
        df.to_csv(CACHE_FILE, index=False)
        logger.info(f"✓ Fixed cache saved with {len(df)} records")
        logger.info(f"  Columns: {list(df.columns)}")
        
        return True
    except Exception as e:
        logger.error(f"Error fixing cache: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    fix_cache()
