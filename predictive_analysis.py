"""
Predictive Analysis Module for Stock Market Dashboard
Provides technical indicators, forecasting, and trading signals
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_moving_averages(df, periods=[20, 50, 200]):
    """
    Calculate Simple Moving Averages (SMA) for different periods
    
    Args:
        df: DataFrame with 'Close' column
        periods: List of periods for moving averages
    
    Returns:
        DataFrame with additional SMA columns
    """
    result = df.copy()
    for period in periods:
        if len(df) >= period:
            result[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    return result


def calculate_ema(df, periods=[12, 26]):
    """
    Calculate Exponential Moving Averages (EMA)
    
    Args:
        df: DataFrame with 'Close' column
        periods: List of periods for EMAs
    
    Returns:
        DataFrame with additional EMA columns
    """
    result = df.copy()
    for period in periods:
        if len(df) >= period:
            result[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    return result


def calculate_rsi(df, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        df: DataFrame with 'Close' column
        period: RSI period (default 14)
    
    Returns:
        DataFrame with RSI column
    """
    result = df.copy()
    
    # Calculate price changes
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    return result


def calculate_macd(df):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        df: DataFrame with 'Close' column
    
    Returns:
        DataFrame with MACD, Signal, and Histogram columns
    """
    result = df.copy()
    
    # Calculate EMAs
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD line
    result['MACD'] = ema_12 - ema_26
    
    # Signal line (9-day EMA of MACD)
    result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    
    # MACD Histogram
    result['MACD_Hist'] = result['MACD'] - result['MACD_Signal']
    
    return result


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands
    
    Args:
        df: DataFrame with 'Close' column
        period: Moving average period
        std_dev: Number of standard deviations
    
    Returns:
        DataFrame with Bollinger Bands columns
    """
    result = df.copy()
    
    # Middle band (SMA)
    result['BB_Middle'] = df['Close'].rolling(window=period).mean()
    
    # Standard deviation
    rolling_std = df['Close'].rolling(window=period).std()
    
    # Upper and lower bands
    result['BB_Upper'] = result['BB_Middle'] + (rolling_std * std_dev)
    result['BB_Lower'] = result['BB_Middle'] - (rolling_std * std_dev)
    
    return result


def linear_regression_forecast(df, forecast_days=30):
    """
    Simple linear regression forecast
    
    Args:
        df: DataFrame with 'Date' and 'Close' columns
        forecast_days: Number of days to forecast
    
    Returns:
        DataFrame with forecast
    """
    try:
        from sklearn.linear_model import LinearRegression
        
        # Prepare data
        df_clean = df[['Date', 'Close']].dropna().copy()
        df_clean['Days'] = (df_clean['Date'] - df_clean['Date'].min()).dt.days
        
        X = df_clean[['Days']].values
        y = df_clean['Close'].values
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate future dates
        last_day = df_clean['Days'].max()
        future_days = np.array([[last_day + i] for i in range(1, forecast_days + 1)])
        last_date = df_clean['Date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        # Predict
        forecast_values = model.predict(future_days)
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast_values
        })
        
        return forecast_df, model.coef_[0], model.intercept_
        
    except ImportError:
        # If sklearn not available, return simple extrapolation
        last_30_days = df.tail(30)
        avg_change = last_30_days['Close'].diff().mean()
        
        last_date = df['Date'].max()
        last_price = df['Close'].iloc[-1]
        
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        forecast_values = [last_price + (avg_change * i) for i in range(1, forecast_days + 1)]
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast_values
        })
        
        return forecast_df, avg_change, last_price


def generate_trading_signals(df):
    """
    Generate buy/sell signals based on technical indicators
    
    Args:
        df: DataFrame with technical indicators
    
    Returns:
        Dictionary with signals and recommendations
    """
    signals = {
        'overall': 'NEUTRAL',
        'strength': 0,
        'indicators': []
    }
    
    latest = df.iloc[-1]
    score = 0
    
    # RSI Signal
    if 'RSI' in df.columns and not pd.isna(latest['RSI']):
        if latest['RSI'] < 30:
            signals['indicators'].append({'name': 'RSI', 'signal': 'BUY', 'value': f"{latest['RSI']:.2f}", 'reason': 'Oversold (< 30)'})
            score += 2
        elif latest['RSI'] > 70:
            signals['indicators'].append({'name': 'RSI', 'signal': 'SELL', 'value': f"{latest['RSI']:.2f}", 'reason': 'Overbought (> 70)'})
            score -= 2
        else:
            signals['indicators'].append({'name': 'RSI', 'signal': 'NEUTRAL', 'value': f"{latest['RSI']:.2f}", 'reason': 'Normal range'})
    
    # MACD Signal
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                signals['indicators'].append({'name': 'MACD', 'signal': 'BUY', 'value': f"{latest['MACD']:.2f}", 'reason': 'MACD above signal line'})
                score += 1
            else:
                signals['indicators'].append({'name': 'MACD', 'signal': 'SELL', 'value': f"{latest['MACD']:.2f}", 'reason': 'MACD below signal line'})
                score -= 1
    
    # Moving Average Crossover
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        if not pd.isna(latest['SMA_20']) and not pd.isna(latest['SMA_50']):
            if latest['SMA_20'] > latest['SMA_50']:
                signals['indicators'].append({'name': 'MA Cross', 'signal': 'BUY', 'value': 'SMA20 > SMA50', 'reason': 'Short-term uptrend'})
                score += 1
            else:
                signals['indicators'].append({'name': 'MA Cross', 'signal': 'SELL', 'value': 'SMA20 < SMA50', 'reason': 'Short-term downtrend'})
                score -= 1
    
    # Bollinger Bands Signal
    if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
        if not pd.isna(latest['BB_Upper']) and not pd.isna(latest['BB_Lower']):
            if latest['Close'] < latest['BB_Lower']:
                signals['indicators'].append({'name': 'Bollinger', 'signal': 'BUY', 'value': 'Below lower band', 'reason': 'Potentially oversold'})
                score += 1
            elif latest['Close'] > latest['BB_Upper']:
                signals['indicators'].append({'name': 'Bollinger', 'signal': 'SELL', 'value': 'Above upper band', 'reason': 'Potentially overbought'})
                score -= 1
            else:
                signals['indicators'].append({'name': 'Bollinger', 'signal': 'NEUTRAL', 'value': 'Within bands', 'reason': 'Normal range'})
    
    # Overall signal
    signals['strength'] = score
    if score >= 3:
        signals['overall'] = 'STRONG BUY'
    elif score >= 1:
        signals['overall'] = 'BUY'
    elif score <= -3:
        signals['overall'] = 'STRONG SELL'
    elif score <= -1:
        signals['overall'] = 'SELL'
    else:
        signals['overall'] = 'NEUTRAL'
    
    return signals


def create_prediction_charts(df, ticker):
    """
    Create comprehensive prediction charts
    
    Args:
        df: DataFrame with all indicators
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary of plotly figures
    """
    charts = {}
    
    # 1. Price with Moving Averages
    fig_ma = go.Figure()
    
    fig_ma.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name='Close Price',
        line=dict(color='#667eea', width=2)
    ))
    
    if 'SMA_20' in df.columns:
        fig_ma.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_20'],
            mode='lines', name='SMA 20',
            line=dict(color='#f59e0b', width=1.5, dash='dash')
        ))
    
    if 'SMA_50' in df.columns:
        fig_ma.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_50'],
            mode='lines', name='SMA 50',
            line=dict(color='#10b981', width=1.5, dash='dash')
        ))
    
    if 'SMA_200' in df.columns:
        fig_ma.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_200'],
            mode='lines', name='SMA 200',
            line=dict(color='#ef4444', width=1.5, dash='dash')
        ))
    
    fig_ma.update_layout(
        title={'text': f'{ticker} - Price with Moving Averages', 'font': {'size': 18, 'color': '#2d3748'}},
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_white',
        hovermode='x unified',
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font={'color': '#4a5568'},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    charts['moving_averages'] = fig_ma
    
    # 2. RSI Chart
    if 'RSI' in df.columns:
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=df['Date'], y=df['RSI'],
            mode='lines', name='RSI',
            line=dict(color='#8b5cf6', width=2)
        ))
        
        # Add overbought/oversold lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        
        fig_rsi.update_layout(
            title={'text': f'{ticker} - RSI (Relative Strength Index)', 'font': {'size': 18, 'color': '#2d3748'}},
            xaxis_title='Date',
            yaxis_title='RSI',
            template='plotly_white',
            hovermode='x unified',
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='white',
            font={'color': '#4a5568'}
        )
        
        charts['rsi'] = fig_rsi
    
    # 3. MACD Chart
    if 'MACD' in df.columns:
        fig_macd = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.05,
                                 subplot_titles=(f'{ticker} - MACD', 'MACD Histogram'))
        
        fig_macd.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD'],
            mode='lines', name='MACD',
            line=dict(color='#3b82f6', width=2)
        ), row=1, col=1)
        
        fig_macd.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD_Signal'],
            mode='lines', name='Signal',
            line=dict(color='#ef4444', width=2)
        ), row=1, col=1)
        
        # Histogram
        colors = ['#10b981' if val >= 0 else '#ef4444' for val in df['MACD_Hist']]
        fig_macd.add_trace(go.Bar(
            x=df['Date'], y=df['MACD_Hist'],
            name='Histogram',
            marker_color=colors
        ), row=2, col=1)
        
        fig_macd.update_layout(
            template='plotly_white',
            hovermode='x unified',
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='white',
            font={'color': '#4a5568'},
            showlegend=True,
            height=500
        )
        
        charts['macd'] = fig_macd
    
    # 4. Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig_bb = go.Figure()
        
        fig_bb.add_trace(go.Scatter(
            x=df['Date'], y=df['BB_Upper'],
            mode='lines', name='Upper Band',
            line=dict(color='rgba(239, 68, 68, 0.5)', width=1)
        ))
        
        fig_bb.add_trace(go.Scatter(
            x=df['Date'], y=df['BB_Middle'],
            mode='lines', name='Middle (SMA 20)',
            line=dict(color='#f59e0b', width=1.5)
        ))
        
        fig_bb.add_trace(go.Scatter(
            x=df['Date'], y=df['BB_Lower'],
            mode='lines', name='Lower Band',
            line=dict(color='rgba(16, 185, 129, 0.5)', width=1),
            fill='tonexty', fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        fig_bb.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'],
            mode='lines', name='Close Price',
            line=dict(color='#667eea', width=2)
        ))
        
        fig_bb.update_layout(
            title={'text': f'{ticker} - Bollinger Bands', 'font': {'size': 18, 'color': '#2d3748'}},
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            template='plotly_white',
            hovermode='x unified',
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='white',
            font={'color': '#4a5568'}
        )
        
        charts['bollinger_bands'] = fig_bb
    
    return charts
