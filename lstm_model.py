"""
LSTM Deep Learning Model for Stock Price Prediction
Uses TensorFlow/Keras for time series forecasting
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available for LSTM predictions")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM predictions will be disabled.")


def create_sequences(data, seq_length):
    """
    Create sequences for LSTM training
    
    Args:
        data: Array of values
        seq_length: Length of each sequence
    
    Returns:
        X, y arrays for training
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def build_lstm_model(seq_length, n_features=1):
    """
    Build LSTM model architecture
    
    Args:
        seq_length: Length of input sequences
        n_features: Number of features (default 1 for univariate)
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_lstm_model(df, seq_length=60, forecast_days=30, epochs=50, batch_size=32):
    """
    Train LSTM model and make predictions
    
    Args:
        df: DataFrame with 'Date' and 'Close' columns
        seq_length: Number of previous days to use for prediction
        forecast_days: Number of days to forecast
        epochs: Training epochs
        batch_size: Batch size for training
    
    Returns:
        Dictionary with predictions, model info, and evaluation metrics
    """
    if not TENSORFLOW_AVAILABLE:
        return {
            'error': 'TensorFlow not installed',
            'message': 'LSTM predictions require TensorFlow. Install with: pip install tensorflow'
        }
    
    try:
        # Prepare data
        df_clean = df[['Date', 'Close']].dropna().copy()
        df_clean = df_clean.sort_values('Date')
        
        if len(df_clean) < seq_length + 50:
            return {
                'error': 'Insufficient data',
                'message': f'Need at least {seq_length + 50} data points for LSTM training'
            }
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_clean['Close'].values.reshape(-1, 1))
        
        # Create sequences
        X, y = create_sequences(scaled_data, seq_length)
        
        # Split into train and test (80-20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train model
        model = build_lstm_model(seq_length)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        logger.info(f"Training LSTM model with {len(X_train)} samples...")
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate on test set
        test_predictions = model.predict(X_test, verbose=0)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mse = np.mean((test_predictions - y_test_actual) ** 2)
        mae = np.mean(np.abs(test_predictions - y_test_actual))
        rmse = np.sqrt(mse)
        
        # Make future predictions
        last_sequence = scaled_data[-seq_length:]
        future_predictions = []
        
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_days):
            # Predict next value
            next_pred = model.predict(current_sequence.reshape(1, seq_length, 1), verbose=0)
            future_predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform predictions
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # Create future dates
        last_date = df_clean['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        # Prepare historical predictions for plotting
        train_predictions = model.predict(X_train, verbose=0)
        train_predictions = scaler.inverse_transform(train_predictions)
        
        result = {
            'success': True,
            'forecast_dates': future_dates.tolist(),
            'forecast_values': future_predictions.flatten().tolist(),
            'test_predictions': test_predictions.flatten().tolist(),
            'test_actual': y_test_actual.flatten().tolist(),
            'test_dates': df_clean['Date'].iloc[train_size + seq_length:].tolist(),
            'metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'train_loss': float(history.history['loss'][-1]),
                'val_loss': float(history.history['val_loss'][-1])
            },
            'training_info': {
                'epochs_trained': len(history.history['loss']),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'sequence_length': seq_length
            }
        }
        
        logger.info(f"LSTM model trained successfully. RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}", exc_info=True)
        return {
            'error': 'Training failed',
            'message': str(e)
        }


def create_lstm_charts(df, lstm_results, ticker):
    """
    Create visualization charts for LSTM predictions
    
    Args:
        df: Original DataFrame
        lstm_results: Results from train_lstm_model
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary of plotly figures
    """
    charts = {}
    
    if 'error' in lstm_results:
        # Return empty chart with error message
        fig_error = go.Figure()
        fig_error.add_annotation(
            text=f"⚠️ {lstm_results['message']}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#ef4444')
        )
        fig_error.update_layout(
            title="LSTM Prediction Error",
            template='plotly_white'
        )
        charts['lstm_forecast'] = fig_error
        return charts
    
    # 1. Future Forecast Chart
    fig_forecast = go.Figure()
    
    # Historical data
    df_clean = df[['Date', 'Close']].dropna()
    last_90_days = df_clean.tail(90)
    
    fig_forecast.add_trace(go.Scatter(
        x=last_90_days['Date'],
        y=last_90_days['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#667eea', width=2)
    ))
    
    # Future predictions
    fig_forecast.add_trace(go.Scatter(
        x=lstm_results['forecast_dates'],
        y=lstm_results['forecast_values'],
        mode='lines+markers',
        name='LSTM Forecast',
        line=dict(color='#10b981', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig_forecast.update_layout(
        title={
            'text': f'{ticker} - LSTM Price Forecast (Next {len(lstm_results["forecast_dates"])} Days)',
            'font': {'size': 18, 'color': '#2d3748'}
        },
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_white',
        hovermode='x unified',
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font={'color': '#4a5568'},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    charts['lstm_forecast'] = fig_forecast
    
    # 2. Model Performance Chart (Test Set) - only if test data is available
    if 'test_dates' in lstm_results and 'test_actual' in lstm_results and 'test_predictions' in lstm_results:
        fig_performance = go.Figure()
        
        fig_performance.add_trace(go.Scatter(
            x=lstm_results['test_dates'],
            y=lstm_results['test_actual'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#667eea', width=2)
        ))
        
        fig_performance.add_trace(go.Scatter(
            x=lstm_results['test_dates'],
            y=lstm_results['test_predictions'],
            mode='lines',
            name='LSTM Predictions',
            line=dict(color='#ef4444', width=2, dash='dot')
        ))
        
        fig_performance.update_layout(
            title={
                'text': f'{ticker} - LSTM Model Performance on Test Data',
                'font': {'size': 18, 'color': '#2d3748'}
            },
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            template='plotly_white',
            hovermode='x unified',
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='white',
            font={'color': '#4a5568'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        charts['lstm_performance'] = fig_performance
    
    return charts


def is_lstm_available():
    """Check if LSTM functionality is available"""
    return TENSORFLOW_AVAILABLE
