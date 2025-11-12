"""
Prediction Tracker Module
Tracks LSTM predictions and compares with actual prices to calculate accuracy
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PredictionTracker:
    def __init__(self, storage_file='/app/prediction_history.json'):
        self.storage_file = storage_file
        self.predictions = self._load_predictions()
    
    def _load_predictions(self):
        """Load prediction history from file"""
        try:
            if Path(self.storage_file).exists():
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} prediction records")
                    return data
            return {}
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return {}
    
    def _save_predictions(self):
        """Save prediction history to file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.predictions, f, indent=2)
            logger.info(f"Saved {len(self.predictions)} prediction records")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def save_prediction(self, ticker, prediction_date, target_date, predicted_price, model_info=None):
        """
        Save a new prediction
        
        Args:
            ticker: Stock ticker symbol
            prediction_date: Date when prediction was made
            target_date: Date for which prediction is made
            predicted_price: Predicted price value
            model_info: Optional dict with model details (RMSE, MAE, etc)
        """
        if ticker not in self.predictions:
            self.predictions[ticker] = []
        
        prediction_record = {
            'prediction_date': prediction_date.strftime('%Y-%m-%d') if isinstance(prediction_date, datetime) else prediction_date,
            'target_date': target_date.strftime('%Y-%m-%d') if isinstance(target_date, datetime) else target_date,
            'predicted_price': float(predicted_price),
            'actual_price': None,
            'error': None,
            'accuracy': None,
            'model_info': model_info or {}
        }
        
        self.predictions[ticker].append(prediction_record)
        self._save_predictions()
        logger.info(f"Saved prediction for {ticker}: {predicted_price:.2f} on {target_date}")
    
    def update_actual_price(self, ticker, target_date, actual_price):
        """
        Update actual price for a prediction and calculate accuracy
        
        Args:
            ticker: Stock ticker symbol
            target_date: Date of the prediction
            actual_price: Actual observed price
        """
        if ticker not in self.predictions:
            return
        
        target_date_str = target_date.strftime('%Y-%m-%d') if isinstance(target_date, datetime) else target_date
        
        for pred in self.predictions[ticker]:
            if pred['target_date'] == target_date_str and pred['actual_price'] is None:
                pred['actual_price'] = float(actual_price)
                pred['error'] = abs(pred['predicted_price'] - actual_price)
                pred['accuracy'] = 100 * (1 - pred['error'] / actual_price) if actual_price != 0 else 0
                logger.info(f"Updated actual price for {ticker} on {target_date}: {actual_price:.2f}, accuracy: {pred['accuracy']:.2f}%")
        
        self._save_predictions()
    
    def batch_update_actual_prices(self, ticker, price_df):
        """
        Batch update actual prices from a DataFrame
        
        Args:
            ticker: Stock ticker symbol
            price_df: DataFrame with 'Date' and 'Close' columns
        """
        if ticker not in self.predictions:
            return
        
        updated_count = 0
        for pred in self.predictions[ticker]:
            if pred['actual_price'] is None:
                target_date = pd.to_datetime(pred['target_date'])
                matching_rows = price_df[price_df['Date'] == target_date]
                
                if not matching_rows.empty:
                    actual_price = matching_rows.iloc[0]['Close']
                    pred['actual_price'] = float(actual_price)
                    pred['error'] = abs(pred['predicted_price'] - actual_price)
                    pred['accuracy'] = 100 * (1 - pred['error'] / actual_price) if actual_price != 0 else 0
                    updated_count += 1
        
        if updated_count > 0:
            self._save_predictions()
            logger.info(f"Batch updated {updated_count} actual prices for {ticker}")
    
    def get_prediction_history(self, ticker, days=30):
        """
        Get prediction history for a ticker
        
        Args:
            ticker: Stock ticker symbol
            days: Number of recent days to include
        
        Returns:
            DataFrame with prediction history (trading days only)
        """
        if ticker not in self.predictions or not self.predictions[ticker]:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.predictions[ticker])
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df['target_date'] = pd.to_datetime(df['target_date'])
        
        # Filter out weekends (only show trading days)
        df = df[df['target_date'].dt.weekday < 5]  # Monday=0 to Friday=4
        
        # Filter by recent predictions
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['prediction_date'] >= cutoff_date]
        
        # Sort by target date
        df = df.sort_values('target_date', ascending=False)
        
        return df
    
    def get_accuracy_metrics(self, ticker, days=30):
        """
        Calculate accuracy metrics for a ticker
        
        Args:
            ticker: Stock ticker symbol
            days: Number of recent days to include
        
        Returns:
            Dictionary with accuracy metrics
        """
        df = self.get_prediction_history(ticker, days)
        
        if df.empty:
            return {
                'total_predictions': 0,
                'verified_predictions': 0,
                'mean_accuracy': 0,
                'mean_error': 0,
                'median_accuracy': 0,
                'best_accuracy': 0,
                'worst_accuracy': 0
            }
        
        # Filter only verified predictions (with actual prices)
        verified_df = df[df['actual_price'].notna()]
        
        if verified_df.empty:
            return {
                'total_predictions': len(df),
                'verified_predictions': 0,
                'mean_accuracy': 0,
                'mean_error': 0,
                'median_accuracy': 0,
                'best_accuracy': 0,
                'worst_accuracy': 0
            }
        
        return {
            'total_predictions': len(df),
            'verified_predictions': len(verified_df),
            'mean_accuracy': verified_df['accuracy'].mean(),
            'mean_error': verified_df['error'].mean(),
            'median_accuracy': verified_df['accuracy'].median(),
            'best_accuracy': verified_df['accuracy'].max(),
            'worst_accuracy': verified_df['accuracy'].min(),
            'mae': verified_df['error'].mean(),  # Mean Absolute Error
            'rmse': (verified_df['error'] ** 2).mean() ** 0.5  # Root Mean Square Error
        }
    
    def cleanup_old_predictions(self, days=90):
        """Remove predictions older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        
        removed_count = 0
        for ticker in list(self.predictions.keys()):
            original_count = len(self.predictions[ticker])
            self.predictions[ticker] = [
                pred for pred in self.predictions[ticker]
                if pred['prediction_date'] >= cutoff_str
            ]
            removed_count += original_count - len(self.predictions[ticker])
            
            # Remove ticker if no predictions left
            if not self.predictions[ticker]:
                del self.predictions[ticker]
        
        if removed_count > 0:
            self._save_predictions()
            logger.info(f"Cleaned up {removed_count} old predictions")

# Global tracker instance
prediction_tracker = PredictionTracker()
