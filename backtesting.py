"""
Backtesting module for LSTM model validation.
Trains on historical data and tests on last 6 months.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data_for_backtesting(df_tuple, stock_ticker, test_months=6):
    """
    Split data into train (all data before last N months) and test (last N months).
    
    Args:
        df_tuple: Tuple of (df, dates) from load_data()
        stock_ticker: Stock symbol
        test_months: Number of months for testing (default 6)
    
    Returns:
        train_df, test_df, split_date
    """
    try:
        # Unpack the tuple
        df, dates = df_tuple
        
        # Get the stock data columns
        close_col = [col for col in df.columns if col[0] == stock_ticker and col[1] == 'Close']
        if not close_col:
            logger.warning(f"No data found for {stock_ticker}")
            return None, None, None
        
        close_col = close_col[0]
        
        # Create a series with dates as index
        close_prices = pd.Series(
            pd.to_numeric(df[close_col], errors='coerce').values,
            index=dates
        )
        close_prices = close_prices.dropna()
        
        if len(close_prices) < 180:  # Need at least 6 months of data
            logger.warning(f"Insufficient data for {stock_ticker}: only {len(close_prices)} days")
            return None, None, None
        
        # Calculate split date (6 months ago from last date)
        last_date = close_prices.index[-1]
        split_date = last_date - pd.DateOffset(months=test_months)
        
        # Split data
        train_df = close_prices[close_prices.index < split_date]
        test_df = close_prices[close_prices.index >= split_date]
        
        logger.info(f"Split data for {stock_ticker}:")
        logger.info(f"  Train: {len(train_df)} days ({train_df.index[0]} to {train_df.index[-1]})")
        logger.info(f"  Test:  {len(test_df)} days ({test_df.index[0]} to {test_df.index[-1]})")
        
        return train_df, test_df, split_date
        
    except Exception as e:
        logger.error(f"Error splitting data: {e}", exc_info=True)
        return None, None, None


def create_sequences(data, seq_length):
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def build_lstm_model(seq_length, n_features=1):
    """Build LSTM model architecture."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_and_predict(train_df, test_df, seq_length=30, epochs=20, batch_size=32):
    """
    Train LSTM model on training data and predict on test data.
    
    Args:
        train_df: Training data (pandas Series)
        test_df: Test data (pandas Series)
        seq_length: Sequence length for LSTM
        epochs: Training epochs
        batch_size: Batch size
    
    Returns:
        dict with predictions, actuals, metrics, training_history
    """
    try:
        # Prepare data
        train_data = train_df.values.reshape(-1, 1)
        test_data = test_df.values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # Create sequences for training
        X_train, y_train = create_sequences(train_scaled, seq_length)
        
        if len(X_train) < 50:
            logger.warning("Insufficient training sequences")
            return None
        
        # Build and train model
        logger.info("Building LSTM model...")
        model = build_lstm_model(seq_length)
        
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        logger.info(f"Training on {len(X_train)} sequences...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop]
        )
        
        # Predict on test data
        logger.info("Generating predictions...")
        predictions = []
        
        # Use last seq_length days from training as initial sequence
        current_sequence = train_scaled[-seq_length:].copy()
        
        for i in range(len(test_df)):
            # Predict next value
            pred = model.predict(current_sequence.reshape(1, seq_length, 1), verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence with actual value for next prediction
            if i < len(test_scaled) - 1:
                current_sequence = np.append(current_sequence[1:], test_scaled[i].reshape(1, 1), axis=0)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        
        # Calculate metrics
        actuals = test_df.values
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(actuals.flatten()) > 0
        pred_direction = np.diff(predictions.flatten()) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        logger.info(f"Backtesting metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, Dir Acc={directional_accuracy:.2f}%")
        
        return {
            'success': True,
            'predictions': predictions.flatten(),
            'actuals': actuals.flatten(),
            'dates': test_df.index,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'directional_accuracy': directional_accuracy
            },
            'training_history': {
                'loss': history.history['loss'],
                'mae': history.history['mae'],
                'epochs': len(history.history['loss'])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in train_and_predict: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def calculate_prediction_errors(actuals, predictions):
    """Calculate various error metrics."""
    errors = predictions - actuals
    percent_errors = (errors / actuals) * 100
    
    return {
        'errors': errors,
        'percent_errors': percent_errors,
        'abs_errors': np.abs(errors),
        'abs_percent_errors': np.abs(percent_errors)
    }


def create_backtesting_charts(results):
    """
    Create visualization charts for backtesting results.
    
    Args:
        results: Dictionary from train_and_predict
    
    Returns:
        dict with plotly figures
    """
    if not results or not results.get('success'):
        return None
    
    predictions = results['predictions']
    actuals = results['actuals']
    dates = results['dates']
    metrics = results['metrics']
    
    # Calculate errors
    error_data = calculate_prediction_errors(actuals, predictions)
    
    # 1. Predicted vs Actual Price Chart
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Scatter(
        x=dates,
        y=actuals,
        mode='lines',
        name='Actual Price',
        line=dict(color='#10b981', width=2),
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
    ))
    
    fig_comparison.add_trace(go.Scatter(
        x=dates,
        y=predictions,
        mode='lines',
        name='Predicted Price',
        line=dict(color='#667eea', width=2, dash='dash'),
        hovertemplate='<b>Predicted</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
    ))
    
    fig_comparison.update_layout(
        title='Backtesting: Predicted vs Actual Prices',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        font=dict(family='Arial, sans-serif'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # 2. Prediction Error Distribution
    fig_error = go.Figure()
    
    fig_error.add_trace(go.Histogram(
        x=error_data['percent_errors'],
        nbinsx=30,
        name='Error Distribution',
        marker=dict(color='#667eea', opacity=0.7),
        hovertemplate='Error: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))
    
    fig_error.update_layout(
        title='Prediction Error Distribution',
        xaxis_title='Prediction Error (%)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400,
        font=dict(family='Arial, sans-serif'),
        showlegend=False
    )
    
    # 3. Error Over Time
    fig_error_time = go.Figure()
    
    fig_error_time.add_trace(go.Scatter(
        x=dates,
        y=error_data['percent_errors'],
        mode='lines+markers',
        name='Prediction Error',
        line=dict(color='#ef4444', width=1),
        marker=dict(size=4),
        hovertemplate='<b>Error</b><br>Date: %{x}<br>Error: %{y:.2f}%<extra></extra>'
    ))
    
    fig_error_time.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig_error_time.update_layout(
        title='Prediction Error Over Time',
        xaxis_title='Date',
        yaxis_title='Prediction Error (%)',
        template='plotly_white',
        height=400,
        font=dict(family='Arial, sans-serif'),
        showlegend=False
    )
    
    # 4. Training Loss Chart
    if 'training_history' in results:
        fig_training = go.Figure()
        
        epochs = list(range(1, len(results['training_history']['loss']) + 1))
        
        fig_training.add_trace(go.Scatter(
            x=epochs,
            y=results['training_history']['loss'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#667eea', width=2),
            marker=dict(size=6)
        ))
        
        fig_training.update_layout(
            title='Model Training Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss (MSE)',
            template='plotly_white',
            height=400,
            font=dict(family='Arial, sans-serif'),
            showlegend=False
        )
    else:
        fig_training = None
    
    return {
        'comparison': fig_comparison,
        'error_dist': fig_error,
        'error_time': fig_error_time,
        'training': fig_training,
        'metrics': metrics
    }


def run_backtesting(df_tuple, stock_ticker, test_months=6, seq_length=30, epochs=20):
    """
    Complete backtesting pipeline.
    
    Args:
        df_tuple: Tuple of (df, dates) from load_data()
        stock_ticker: Stock to backtest
        test_months: Months to use for testing
        seq_length: LSTM sequence length
        epochs: Training epochs
    
    Returns:
        dict with results and charts
    """
    logger.info(f"Starting backtesting for {stock_ticker}...")
    
    # Split data
    train_df, test_df, split_date = split_data_for_backtesting(df_tuple, stock_ticker, test_months)
    
    if train_df is None or test_df is None:
        return {
            'success': False,
            'error': 'Insufficient data for backtesting'
        }
    
    # Train and predict
    results = train_and_predict(train_df, test_df, seq_length, epochs)
    
    if not results or not results.get('success'):
        return results
    
    # Create charts
    charts = create_backtesting_charts(results)
    
    return {
        'success': True,
        'charts': charts,
        'split_date': split_date,
        'train_period': f"{train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}",
        'test_period': f"{test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}",
        'train_days': len(train_df),
        'test_days': len(test_df)
    }
