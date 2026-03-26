import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
from pathlib import Path
import logging
import time
import os
import predictive_analysis as pred
import lstm_model
import backtesting
from prediction_tracker import prediction_tracker
import fundamentals

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, 
                url_base_pathname='/indiastock/',
                serve_locally=True,
                suppress_callback_exceptions=True,
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
                ])

server = app.server

# Add custom CSS for responsive charts
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @media (max-width: 768px) {
                .top-charts {
                    flex-direction: column !important;
                }
                .chart-half {
                    width: 100% !important;
                    min-width: 100% !important;
                }
            }
            @media (min-width: 769px) {
                .top-charts {
                    flex-direction: row !important;
                    flex-wrap: nowrap !important;
                }
                .chart-half {
                    width: 48% !important;
                    flex: 1 1 48% !important;
                }
            }
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.1); opacity: 0.8; }
            }
            @keyframes shimmer {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes rotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
            }
            .loading-brain {
                animation: pulse 2s ease-in-out infinite;
                display: inline-block;
            }
            .loading-progress {
                animation: shimmer 2s ease-in-out infinite;
            }
            .loading-dot {
                animation: bounce 1.4s ease-in-out infinite;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Global variable to store the data with timestamp
cached_df = None
stock_list = []
cache_timestamp = None
CACHE_TIMEOUT = 14400  # 4 hours in seconds
CACHE_SIGNAL_FILE = "/app/.cache_invalidate"

def check_cache_invalidation():
    """Check if cache should be invalidated due to data update."""
    global cache_timestamp
    
    if not os.path.exists(CACHE_SIGNAL_FILE):
        return False
    
    try:
        # Check if signal file is newer than cache
        signal_mtime = os.path.getmtime(CACHE_SIGNAL_FILE)
        if cache_timestamp is None or signal_mtime > cache_timestamp:
            logger.info(f"Cache invalidation signal detected (signal time: {signal_mtime}, cache time: {cache_timestamp})")
            # Remove the signal file after reading
            os.remove(CACHE_SIGNAL_FILE)
            logger.info("Removed cache invalidation signal file")
            return True
    except Exception as e:
        logger.warning(f"Error checking cache invalidation signal: {e}")
    
    return False

def load_data():
    """Load stock data from nse_all_10y.csv with multi-index columns."""
    global cached_df, stock_list, cache_timestamp
    
    # Check for cache invalidation signal first
    if check_cache_invalidation():
        logger.info("Cache invalidated by update signal, forcing reload...")
        cached_df = None
        cache_timestamp = None
    
    # Check if cache is still valid (within 4 hours)
    import time
    current_time = time.time()
    if cached_df is not None and cache_timestamp is not None:
        if current_time - cache_timestamp < CACHE_TIMEOUT:
            return cached_df, stock_list
        else:
            logger.info("Cache expired, reloading data...")
    
    logger.info("=== Loading stock data ===")
    file_path = "/app/nse_all_10y.csv"
    
    try:
        # Read with multi-level header (3 rows: Ticker, Price type, Date)
        df = pd.read_csv(file_path, header=[0, 1, 2])
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        logger.info(f"Shape: {df.shape}")
        
        # Extract the date column (first column is Ticker/Price/Date)
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col], errors='coerce')
        
        # Get list of stock tickers (unique values from level 0, excluding 'Ticker')
        stock_list = [col for col in df.columns.get_level_values(0).unique() if col != 'Ticker']
        logger.info(f"Found {len(stock_list)} stocks")
        logger.info(f"First 10 stocks: {stock_list[:10]}")
        
        cached_df = (df, dates)
        cache_timestamp = current_time
        logger.info(f"Cache will expire at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_timestamp + CACHE_TIMEOUT))}")
        return cached_df, stock_list
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}", exc_info=True)
        return (None, None), []

# Load initial data
_, stock_list = load_data()

# Get the latest date from the data
def get_latest_date():
    """Get the latest date available in the dataset."""
    (df, dates), _ = load_data()
    if dates is not None and not dates.empty:
        latest_date = dates.max()
        return latest_date.strftime('%B %d, %Y')
    return "Unknown"

# Don't cache this - always get fresh date
def get_current_latest_date():
    """Get the current latest date without caching."""
    return get_latest_date()

latest_data_date = get_latest_date()

# Layout of the dashboard
app.layout = html.Div([
    # Header with gradient background
    html.Div([
        html.H1("Indian Stock Market Dashboard", 
                style={
                    'textAlign': 'center', 
                    'color': 'white', 
                    'marginBottom': '5px', 
                    'fontSize': 'clamp(20px, 5vw, 32px)',
                    'fontWeight': '700',
                    'textShadow': '0 2px 4px rgba(0,0,0,0.2)'
                }),
        html.P("NSE 10-Year Historical Data", 
               style={
                   'textAlign': 'center', 
                   'color': 'rgba(255,255,255,0.9)', 
                   'fontSize': 'clamp(12px, 3vw, 16px)',
                   'margin': '0',
                   'marginBottom': '5px'
               }),
        html.P(id='latest-data-date',
               children=f"Latest Data: {latest_data_date}",
               style={
                   'textAlign': 'center', 
                   'color': 'rgba(255,255,255,0.8)', 
                   'fontSize': 'clamp(10px, 2.5vw, 13px)',
                   'margin': '0',
                   'fontStyle': 'italic'
               })
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'padding': '20px 15px',
        'marginBottom': '15px',
        'boxShadow': '0 4px 15px rgba(0,0,0,0.15)',
        'borderRadius': '0 0 15px 15px'
    }),
    
    # Market Performers Table (Top section)
    html.Div([
        html.H3(id='performers-heading', style={
            'textAlign': 'center', 
            'marginBottom': '15px', 
            'fontSize': 'clamp(16px, 3.5vw, 20px)',
            'color': '#667eea',
            'fontWeight': '600'
        }),
        dcc.Loading(
            id="loading-performers",
            type="circle",
            color="#667eea",
            children=html.Div(id='top-performers-table', style={'fontSize': 'clamp(9px, 2vw, 11px)', 'maxHeight': '250px', 'overflowY': 'auto'})
        )
    ], style={
        'maxWidth': '1200px', 
        'marginLeft': 'auto', 
        'marginRight': 'auto', 
        'marginBottom': '20px', 
        'padding': '20px', 
        'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
        'borderRadius': '12px', 
        'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
        'border': '1px solid rgba(102, 126, 234, 0.1)'
    }),
    
    # Filters (under top performers)
    html.Div([
        html.Div([
            html.Label("Select Stock Ticker:", style={
                'fontWeight': '600', 
                'fontSize': 'clamp(12px, 3vw, 16px)',
                'color': '#4a5568',
                'marginBottom': '8px',
                'display': 'block'
            }),
            dcc.Dropdown(
                id='stock-selector',
                options=[{'label': stock, 'value': stock} for stock in stock_list],
                value=stock_list[0] if stock_list else None,
                placeholder="Select a stock ticker...",
                style={
                    'width': '100%', 
                    'fontSize': 'clamp(11px, 2.5vw, 14px)',
                    'borderRadius': '8px'
                }
            ),
        ], style={'width': '55%'}, className='mobile-full-width'),
        
        html.Div(id='timeframe-container', children=[
            html.Label("Select Timeframe:", style={
                'fontWeight': '600', 
                'fontSize': 'clamp(12px, 3vw, 16px)',
                'color': '#4a5568',
                'marginBottom': '8px',
                'display': 'block'
            }),
            dcc.Dropdown(
                id='timeframe-selector',
                options=[
                    {'label': 'Today', 'value': 'TODAY'},
                    {'label': '1 Month', 'value': '1M'},
                    {'label': '3 Months', 'value': '3M'},
                    {'label': '6 Months', 'value': '6M'},
                    {'label': '1 Year', 'value': '1Y'},
                    {'label': '3 Years', 'value': '3Y'},
                    {'label': '5 Years', 'value': '5Y'},
                    {'label': 'All Time', 'value': 'ALL'}
                ],
                value='1Y',
                style={
                    'width': '100%', 
                    'fontSize': 'clamp(11px, 2.5vw, 14px)',
                    'borderRadius': '8px'
                }
            ),
        ], style={'width': '15%'}, className='mobile-full-width'),
    ], style={
        'maxWidth': '1200px', 
        'marginLeft': 'auto', 
        'marginRight': 'auto', 
        'marginBottom': '25px', 
        'padding': '20px',
        'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
        'borderRadius': '12px',
        'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
        'border': '1px solid rgba(102, 126, 234, 0.1)',
        'display': 'flex', 
        'flexWrap': 'wrap', 
        'gap': '15px'
    }),
    
    # Tabs (after dropdowns)
    dcc.Tabs(id='tabs', value='tab-analysis', children=[
        dcc.Tab(
            label='📊 Price Analysis', 
            value='tab-analysis', 
            style={
                'padding': '12px 24px', 
                'fontWeight': '600',
                'fontSize': 'clamp(13px, 3vw, 16px)',
                'borderRadius': '8px 8px 0 0',
                'backgroundColor': '#f8f9fa',
                'border': 'none'
            }, 
            selected_style={
                'padding': '12px 24px', 
                'fontWeight': '700',
                'fontSize': 'clamp(13px, 3vw, 16px)',
                'borderRadius': '8px 8px 0 0',
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'border': 'none',
                'boxShadow': '0 -2px 8px rgba(102, 126, 234, 0.3)'
            }
        ),
        dcc.Tab(
            label='🔮 Predictive Analysis', 
            value='tab-prediction', 
            style={
                'padding': '12px 24px', 
                'fontWeight': '600',
                'fontSize': 'clamp(13px, 3vw, 16px)',
                'borderRadius': '8px 8px 0 0',
                'backgroundColor': '#f8f9fa',
                'border': 'none'
            }, 
            selected_style={
                'padding': '12px 24px', 
                'fontWeight': '700',
                'fontSize': 'clamp(13px, 3vw, 16px)',
                'borderRadius': '8px 8px 0 0',
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'border': 'none',
                'boxShadow': '0 -2px 8px rgba(102, 126, 234, 0.3)'
            }
        ),
        dcc.Tab(
            label='📈 Backtesting', 
            value='tab-backtesting', 
            style={
                'padding': '12px 24px', 
                'fontWeight': '600',
                'fontSize': 'clamp(13px, 3vw, 16px)',
                'borderRadius': '8px 8px 0 0',
                'backgroundColor': '#f8f9fa',
                'border': 'none'
            }, 
            selected_style={
                'padding': '12px 24px', 
                'fontWeight': '700',
                'fontSize': 'clamp(13px, 3vw, 16px)',
                'borderRadius': '8px 8px 0 0',
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'border': 'none',
                'boxShadow': '0 -2px 8px rgba(102, 126, 234, 0.3)'
            }
        ),
        dcc.Tab(
            label='💼 Fundamentals', 
            value='tab-fundamentals', 
            style={
                'padding': '12px 24px', 
                'fontWeight': '600',
                'fontSize': 'clamp(13px, 3vw, 16px)',
                'borderRadius': '8px 8px 0 0',
                'backgroundColor': '#f8f9fa',
                'border': 'none'
            }, 
            selected_style={
                'padding': '12px 24px', 
                'fontWeight': '700',
                'fontSize': 'clamp(13px, 3vw, 16px)',
                'borderRadius': '8px 8px 0 0',
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'border': 'none',
                'boxShadow': '0 -2px 8px rgba(102, 126, 234, 0.3)'
            }
        ),
        dcc.Tab(
            label='🎯 Prediction Tracking', 
            value='tab-prediction-tracking', 
            style={
                'padding': '12px 24px', 
                'fontWeight': '600',
                'fontSize': 'clamp(13px, 3vw, 16px)',
                'borderRadius': '8px 8px 0 0',
                'backgroundColor': '#f8f9fa',
                'border': 'none'
            }, 
            selected_style={
                'padding': '12px 24px', 
                'fontWeight': '700',
                'fontSize': 'clamp(13px, 3vw, 16px)',
                'borderRadius': '8px 8px 0 0',
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'border': 'none',
                'boxShadow': '0 -2px 8px rgba(102, 126, 234, 0.3)'
            }
        ),
    ], style={
        'maxWidth': '1200px', 
        'marginLeft': 'auto', 
        'marginRight': 'auto', 
        'marginBottom': '0'
    }),
    
    # Tab content with loading spinner positioned near tabs
    dcc.Loading(
        id="loading-tab-content",
        type="dot",
        color="#667eea",
        fullscreen=True,
        style={'minHeight': '200px'},
        children=html.Div(id='tabs-content', style={'width': '100%'})
    ),
    
    # Footer
    html.Div([
        html.P("Data source: Yahoo Finance",
               style={
                   'textAlign': 'center', 
                   'color': '#718096', 
                   'marginTop': '50px',
                   'fontSize': 'clamp(11px, 2.5vw, 14px)',
                   'fontWeight': '500'
               })
    ])
], style={
    'background': 'linear-gradient(to bottom, #e6e9f0 0%, #eef1f5 100%)',
    'minHeight': '100vh', 
    'paddingBottom': '40px'
})

def get_stock_data(ticker):
    """Get stock data for a specific ticker."""
    (df, dates), _ = load_data()
    if df is None:
        return None
    
    # Find columns for this ticker
    ticker_cols = [col for col in df.columns if col[0] == ticker]
    if not ticker_cols:
        return None
    
    # Extract Close, Volume, Open, High, Low data
    close_col = [col for col in ticker_cols if col[1] == 'Close'][0]
    volume_col = [col for col in ticker_cols if col[1] == 'Volume'][0]
    open_col = [col for col in ticker_cols if col[1] == 'Open'][0]
    high_col = [col for col in ticker_cols if col[1] == 'High'][0]
    low_col = [col for col in ticker_cols if col[1] == 'Low'][0]
    
    stock_df = pd.DataFrame({
        'Date': dates,
        'Close': pd.to_numeric(df[close_col], errors='coerce'),
        'Volume': pd.to_numeric(df[volume_col], errors='coerce').fillna(0),
        'Open': pd.to_numeric(df[open_col], errors='coerce'),
        'High': pd.to_numeric(df[high_col], errors='coerce'),
        'Low': pd.to_numeric(df[low_col], errors='coerce')
    })
    
    # Drop rows where Close is NaN
    stock_df = stock_df.dropna(subset=['Close'])
    
    return stock_df

def create_prediction_history_ui(ticker):
    """Create UI components for prediction history and accuracy"""
    if not ticker:
        return html.P("Select a stock to view prediction history", style={
            'textAlign': 'center',
            'color': '#718096',
            'fontSize': 'clamp(11px, 2.2vw, 13px)',
            'padding': '20px'
        })
    
    # Update actual prices for this ticker
    stock_df = get_stock_data(ticker)
    if stock_df is not None and not stock_df.empty:
        prediction_tracker.batch_update_actual_prices(ticker, stock_df)
    
    # Get accuracy metrics
    metrics = prediction_tracker.get_accuracy_metrics(ticker, days=90)
    
    # Get prediction history
    history_df = prediction_tracker.get_prediction_history(ticker, days=90)
    
    if metrics['total_predictions'] == 0:
        return html.Div([
            html.P("🔮 No prediction history yet", style={
                'textAlign': 'center',
                'color': '#667eea',
                'fontSize': 'clamp(12px, 2.5vw, 14px)',
                'fontWeight': '600',
                'marginBottom': '10px'
            }),
            html.P("Predictions will be recorded when the LSTM model generates forecasts. Once target dates pass, actual prices will be compared to show accuracy.", style={
                'textAlign': 'center',
                'color': '#718096',
                'fontSize': 'clamp(10px, 2vw, 12px)',
                'padding': '10px 20px',
                'lineHeight': '1.5'
            })
        ])
    
    # Accuracy Summary Cards
    accuracy_cards = html.Div([
        html.Div([
            html.P("Total Predictions", style={'fontSize': 'clamp(9px, 1.8vw, 11px)', 'color': '#718096', 'marginBottom': '3px'}),
            html.P(f"{metrics['total_predictions']}", style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#667eea', 'fontWeight': 'bold'})
        ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'background': '#f7fafc', 'borderRadius': '6px', 'margin': '3px'}),
        
        html.Div([
            html.P("Verified", style={'fontSize': 'clamp(9px, 1.8vw, 11px)', 'color': '#718096', 'marginBottom': '3px'}),
            html.P(f"{metrics['verified_predictions']}", style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#10b981', 'fontWeight': 'bold'})
        ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'background': '#f7fafc', 'borderRadius': '6px', 'margin': '3px'}),
        
        html.Div([
            html.P("Mean Accuracy", style={'fontSize': 'clamp(9px, 1.8vw, 11px)', 'color': '#718096', 'marginBottom': '3px'}),
            html.P(f"{metrics['mean_accuracy']:.1f}%" if metrics['verified_predictions'] > 0 else "N/A", 
                   style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#667eea', 'fontWeight': 'bold'})
        ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'background': '#f7fafc', 'borderRadius': '6px', 'margin': '3px'}),
        
        html.Div([
            html.P("Mean Error", style={'fontSize': 'clamp(9px, 1.8vw, 11px)', 'color': '#718096', 'marginBottom': '3px'}),
            html.P(f"₹{metrics['mean_error']:.2f}" if metrics['verified_predictions'] > 0 else "N/A", 
                   style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#ef4444', 'fontWeight': 'bold'})
        ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'background': '#f7fafc', 'borderRadius': '6px', 'margin': '3px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'marginBottom': '15px'})
    
    # Create history table
    if history_df.empty:
        table_content = html.P("No detailed history available", style={'textAlign': 'center', 'color': '#718096', 'padding': '10px'})
    else:
        # Split into past (verifiable) and future (pending) predictions
        from datetime import datetime
        today = pd.Timestamp(datetime.now().date())
        
        # Sort by target_date first to ensure proper ordering
        history_df = history_df.sort_values('target_date')
        
        # Filter out weekends (only show trading days)
        history_df['is_trading_day'] = history_df['target_date'].dt.weekday < 5  # Monday=0 to Friday=4
        history_df = history_df[history_df['is_trading_day']].copy()
        
        # Group by target_date first to remove duplicates
        history_grouped = history_df.groupby('target_date').agg({
            'predicted_price': 'mean',
            'actual_price': 'first',
            'error': 'mean',
            'accuracy': 'mean'
        }).reset_index()
        
        # Filter to show current month only
        current_month_start = pd.Timestamp(today.year, today.month, 1)
        current_month_end = (current_month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
        
        display_df = history_grouped[
            (history_grouped['target_date'] >= current_month_start) & 
            (history_grouped['target_date'] <= current_month_end)
        ].copy()
        
        if not display_df.empty:
            past_count = len(display_df[display_df['target_date'] <= today])
            future_count = len(display_df[display_df['target_date'] > today])
            all_future = future_count == len(display_df)
            
            if past_count > 0 and future_count > 0:
                table_title = f"Current Month Predictions ({past_count} Past + {future_count} Future)"
            elif past_count > 0:
                table_title = f"Current Month Predictions ({past_count} Verified)"
            else:
                table_title = f"Current Month Predictions ({future_count} Upcoming)"
        else:
            table_title = "Current Month Predictions"
            all_future = False
        
        if not display_df.empty:
            display_df = display_df[['target_date', 'predicted_price', 'actual_price', 'error', 'accuracy']].copy()
            display_df['target_date'] = display_df['target_date'].dt.strftime('%Y-%m-%d')
        
        # Create table with title
        table_title_elem = html.Div(table_title, style={
            'fontSize': 'clamp(11px, 2.2vw, 13px)',
            'color': '#667eea',
            'fontWeight': '600',
            'marginBottom': '10px',
            'marginTop': '10px'
        })
        
        # Create table header
        table_header = html.Tr([
            html.Th("Date", style={'padding': '8px', 'textAlign': 'left', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568', 'fontWeight': '600'}),
            html.Th("Predicted", style={'padding': '8px', 'textAlign': 'right', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568', 'fontWeight': '600'}),
            html.Th("Actual", style={'padding': '8px', 'textAlign': 'right', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568', 'fontWeight': '600'}),
            html.Th("Error", style={'padding': '8px', 'textAlign': 'right', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568', 'fontWeight': '600'}),
            html.Th("Accuracy", style={'padding': '8px', 'textAlign': 'right', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568', 'fontWeight': '600'}),
        ])
        
        # Create table rows
        table_rows = []
        for _, row in display_df.iterrows():
            accuracy_color = '#10b981' if row['accuracy'] and row['accuracy'] >= 90 else '#f59e0b' if row['accuracy'] and row['accuracy'] >= 80 else '#ef4444' if row['accuracy'] else '#9ca3af'
            
            table_rows.append(html.Tr([
                html.Td(row['target_date'], style={'padding': '8px', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568'}),
                html.Td(f"₹{row['predicted_price']:.2f}", style={'padding': '8px', 'textAlign': 'right', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#667eea', 'fontWeight': '600'}),
                html.Td(f"₹{row['actual_price']:.2f}" if pd.notna(row['actual_price']) else "Pending", 
                        style={'padding': '8px', 'textAlign': 'right', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568'}),
                html.Td(f"₹{row['error']:.2f}" if pd.notna(row['error']) else "-", 
                        style={'padding': '8px', 'textAlign': 'right', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#ef4444'}),
                html.Td(f"{row['accuracy']:.1f}%" if pd.notna(row['accuracy']) else "-", 
                        style={'padding': '8px', 'textAlign': 'right', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': accuracy_color, 'fontWeight': '600'}),
            ], style={'borderBottom': '1px solid #e2e8f0'}))
        
        table_elem = html.Table([
            html.Thead(table_header, style={'borderBottom': '2px solid #cbd5e0'}),
            html.Tbody(table_rows)
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '5px'})
        
        table_content = html.Div([table_title_elem, table_elem])
        
        # Add info message if all predictions are future
        if all_future:
            info_message = html.P(
                "ℹ️ All predictions are for future dates. Accuracy will be calculated once actual prices become available.",
                style={
                    'textAlign': 'center',
                    'color': '#667eea',
                    'fontSize': 'clamp(10px, 2vw, 12px)',
                    'marginTop': '15px',
                    'padding': '10px',
                    'background': '#f0f4ff',
                    'borderRadius': '6px',
                    'border': '1px solid #cbd5e0'
                }
            )
            table_content = html.Div([table_content, info_message])
    
    return html.Div([
        accuracy_cards,
        table_content
    ])

# Callback to render tab content
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('stock-selector', 'value'),
     Input('timeframe-selector', 'value')]
)
def render_tab_content(tab, selected_stock, timeframe):
    if tab == 'tab-analysis':
        # Get the data for the selected stock
        if not selected_stock:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Please select a stock")
            stock_info_content = ""
            price_fig = empty_fig
            volume_fig = empty_fig
            candlestick_fig = empty_fig
        else:
            stock_df = get_stock_data(selected_stock)
            ticker = selected_stock
            
            if stock_df is None or stock_df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No data available for this stock")
                stock_info_content = f"No data for {selected_stock}"
                price_fig = empty_fig
                volume_fig = empty_fig
                candlestick_fig = empty_fig
            else:
                # Filter by timeframe
                latest_date = stock_df['Date'].max()
                if timeframe == 'TODAY':
                    # For today, get last 2 trading days to calculate daily return
                    unique_dates = sorted(stock_df['Date'].unique())
                    if len(unique_dates) >= 2:
                        start_date = unique_dates[-2]  # Yesterday or last trading day
                    else:
                        start_date = latest_date
                elif timeframe == '1M':
                    start_date = latest_date - pd.DateOffset(months=1)
                elif timeframe == '3M':
                    start_date = latest_date - pd.DateOffset(months=3)
                elif timeframe == '6M':
                    start_date = latest_date - pd.DateOffset(months=6)
                elif timeframe == '1Y':
                    start_date = latest_date - pd.DateOffset(years=1)
                elif timeframe == '3Y':
                    start_date = latest_date - pd.DateOffset(years=3)
                elif timeframe == '5Y':
                    start_date = latest_date - pd.DateOffset(years=5)
                else:  # ALL
                    start_date = stock_df['Date'].min()
                
                stock_df = stock_df[stock_df['Date'] >= start_date]
                
                # Calculate returns
                latest_close = stock_df['Close'].iloc[-1]
                earliest_close = stock_df['Close'].iloc[0]
                pct_change = ((latest_close - earliest_close) / earliest_close) * 100 if earliest_close != 0 else 0
                
                # For TODAY timeframe, don't calculate annualized return (doesn't make sense for 1 day)
                if timeframe == 'TODAY':
                    annualized_return = 0  # Not applicable for single day
                else:
                    days = (stock_df['Date'].max() - stock_df['Date'].min()).days
                    years = days / 365.25
                    annualized_return = (((latest_close / earliest_close) ** (1 / years)) - 1) * 100 if years > 0 and earliest_close != 0 else 0
                
                timeframe_labels = {
                    'TODAY': 'Today', '1M': '1 Month', '3M': '3 Months', '6M': '6 Months',
                    '1Y': '1 Year', '3Y': '3 Years', '5Y': '5 Years', 'ALL': 'All Time'
                }
                
                # Build metrics list conditionally
                metrics_divs = [
                    html.Div([
                        html.Div("Latest Close", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': 'black', 'marginBottom': '4px'}),
                        html.Div(f"₹{latest_close:.2f}", style={'fontSize': 'clamp(18px, 4vw, 24px)', 'fontWeight': '700', 'color': "black"}),
                    ], style={'textAlign': 'center', 'padding': '15px', 'background': 'white', 'borderRadius': '10px', 'color': 'black', 'boxShadow': '0 4px 12px rgba(102, 126, 234, 0.3)'}),
                    
                    html.Div([
                        html.Div(f"{timeframe_labels.get(timeframe, timeframe)} Return", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '4px'}),
                        html.Div(f"{pct_change:+.2f}%", style={'fontSize': 'clamp(18px, 4vw, 24px)', 'fontWeight': '700', 'color': '#10b981' if pct_change >= 0 else '#ef4444'}),
                    ], style={'textAlign': 'center', 'padding': '15px', 'background': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'border': '1px solid #e2e8f0'}),
                ]
                
                # Only show annualized return for timeframes other than TODAY
                if timeframe != 'TODAY':
                    metrics_divs.append(
                        html.Div([
                            html.Div("Annualized Return", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '4px'}),
                            html.Div(f"{annualized_return:+.2f}%", style={'fontSize': 'clamp(18px, 4vw, 24px)', 'fontWeight': '700', 'color': '#10b981' if annualized_return >= 0 else '#ef4444'}),
                        ], style={'textAlign': 'center', 'padding': '15px', 'background': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'border': '1px solid #e2e8f0'})
                    )
                
                stock_info_content = html.Div([
                    html.Div(metrics_divs, style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))', 'gap': '15px', 'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Div([
                            html.Span("📈 High: ", style={'color': '#718096', 'fontSize': 'clamp(11px, 2.2vw, 14px)'}),
                            html.Span(f"₹{stock_df['Close'].max():.2f}", style={'fontWeight': '700', 'color': '#10b981', 'fontSize': 'clamp(11px, 2.2vw, 14px)'}),
                        ], style={'padding': '10px', 'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.06)'}),
                        
                        html.Div([
                            html.Span("📉 Low: ", style={'color': '#718096', 'fontSize': 'clamp(11px, 2.2vw, 14px)'}),
                            html.Span(f"₹{stock_df['Close'].min():.2f}", style={'fontWeight': '700', 'color': '#ef4444', 'fontSize': 'clamp(11px, 2.2vw, 14px)'}),
                        ], style={'padding': '10px', 'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.06)'}),
                        
                        html.Div([
                            html.Span("📊 Average: ", style={'color': '#718096', 'fontSize': 'clamp(11px, 2.2vw, 14px)'}),
                            html.Span(f"₹{stock_df['Close'].mean():.2f}", style={'fontWeight': '700', 'color': '#3b82f6', 'fontSize': 'clamp(11px, 2.2vw, 14px)'}),
                        ], style={'padding': '10px', 'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.06)'}),
                        
                        html.Div([
                            html.Span("📅 Trading Days: ", style={'color': '#718096', 'fontSize': 'clamp(11px, 2.2vw, 14px)'}),
                            html.Span(f"{len(stock_df)}", style={'fontWeight': '700', 'color': '#6366f1', 'fontSize': 'clamp(11px, 2.2vw, 14px)'}),
                        ], style={'padding': '10px', 'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.06)'}),
                        
                        html.Div([
                            html.Span("📦 Avg Volume: ", style={'color': '#718096', 'fontSize': 'clamp(11px, 2.2vw, 14px)'}),
                            html.Span(f"{stock_df['Volume'].mean():,.0f}", style={'fontWeight': '700', 'color': '#8b5cf6', 'fontSize': 'clamp(11px, 2.2vw, 14px)'}),
                        ], style={'padding': '10px', 'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.06)'}),
                    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))', 'gap': '10px'})
                ])
                
                # Create charts
                price_fig = go.Figure(data=[go.Scatter(
                    x=stock_df['Date'],
                    y=stock_df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#667eea', width=3),
                    fill='tonexty',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                )])
                price_fig.update_layout(
                    title={'text': f'{ticker} Closing Price History', 'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial, sans-serif'}},
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    template='plotly_white',
                    hovermode='x unified',
                    plot_bgcolor='rgba(248, 249, 250, 0.8)',
                    paper_bgcolor='white',
                    font={'color': '#4a5568'}
                )
                
                volume_fig = go.Figure(data=[go.Bar(
                    x=stock_df['Date'],
                    y=stock_df['Volume'],
                    name='Volume',
                    marker_color='#8b5cf6',
                    marker_line_color='#7c3aed',
                    marker_line_width=0.5
                )])
                volume_fig.update_layout(
                    title={'text': f'{ticker} Trading Volume', 'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial, sans-serif'}},
                    xaxis_title='Date',
                    yaxis_title='Volume (Log Scale)',
                    yaxis_type='log',
                    template='plotly_white',
                    plot_bgcolor='rgba(248, 249, 250, 0.8)',
                    paper_bgcolor='white',
                    font={'color': '#4a5568'}
                )
                
                candlestick_fig = go.Figure(data=[go.Candlestick(
                    x=stock_df['Date'],
                    open=stock_df['Open'],
                    high=stock_df['High'],
                    low=stock_df['Low'],
                    close=stock_df['Close'],
                    name='OHLC',
                    increasing_line_color='#10b981',
                    decreasing_line_color='#ef4444'
                )])
                candlestick_fig.update_layout(
                    title={'text': f'{ticker} OHLC Chart', 'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial, sans-serif'}},
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    template='plotly_white',
                    xaxis_rangeslider_visible=False,
                    plot_bgcolor='rgba(248, 249, 250, 0.8)',
                    paper_bgcolor='white',
                    font={'color': '#4a5568'}
                )
        
        return html.Div([
            # Stock info section
            dcc.Loading(
                id="loading-info",
                type="circle",
                color="#667eea",
                children=html.Div(stock_info_content, style={
                    'margin': '15px', 
                    'padding': '20px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)',
                    'maxWidth': '1200px', 
                    'marginLeft': 'auto', 
                    'marginRight': 'auto'
                })
            ),
            
            # Candlestick chart
            html.Div([
                html.H3("OHLC Candlestick Chart", style={
                    'textAlign': 'center', 
                    'fontSize': 'clamp(15px, 3.5vw, 19px)',
                    'color': '#667eea',
                    'fontWeight': '600',
                    'marginBottom': '15px'
                }),
                dcc.Loading(
                    id="loading-candlestick",
                    type="circle",
                    color="#667eea",
                    children=dcc.Graph(figure=candlestick_fig, style={'height': 'clamp(250px, 50vw, 350px)'}, config={'displayModeBar': False})
                )
            ], style={
                'maxWidth': '1200px', 
                'marginLeft': 'auto', 
                'marginRight': 'auto', 
                'marginBottom': '20px', 
                'padding': '20px',
                'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                'borderRadius': '12px',
                'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                'border': '1px solid rgba(102, 126, 234, 0.1)'
            }),
            
            # Price and Volume charts
            html.Div([
                html.Div([
                    html.H3("Closing Price History", style={
                        'textAlign': 'center', 
                        'fontSize': 'clamp(15px, 3.5vw, 19px)',
                        'color': '#667eea',
                        'fontWeight': '600',
                        'marginBottom': '15px'
                    }),
                    dcc.Loading(
                        id="loading-price",
                        type="circle",
                        color="#667eea",
                        children=dcc.Graph(figure=price_fig, style={'height': 'clamp(250px, 50vw, 350px)'}, config={'displayModeBar': False})
                    )
                ], style={
                    'width': '48%',
                    'minWidth': '300px',
                    'flex': '1 1 48%',
                    'padding': '20px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)'
                }, className='chart-half'),
                
                html.Div([
                    html.H3("Trading Volume", style={
                        'textAlign': 'center', 
                        'fontSize': 'clamp(15px, 3.5vw, 19px)',
                        'color': '#667eea',
                        'fontWeight': '600',
                        'marginBottom': '15px'
                    }),
                    dcc.Loading(
                        id="loading-volume",
                        type="circle",
                        color="#667eea",
                        children=dcc.Graph(figure=volume_fig, style={'height': 'clamp(250px, 50vw, 350px)'}, config={'displayModeBar': False})
                    )
                ], style={
                    'width': '48%',
                    'minWidth': '300px',
                    'flex': '1 1 48%',
                    'padding': '20px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)'
                }, className='chart-half'),
            ], style={
                'maxWidth': '1200px', 
                'marginLeft': 'auto', 
                'marginRight': 'auto', 
                'marginBottom': '20px', 
                'padding': '0 15px',
                'display': 'flex', 
                'flexDirection': 'row',
                'justifyContent': 'space-between',
                'gap': '20px', 
                'flexWrap': 'nowrap'
            }, className='top-charts'),
        ])
    
    elif tab == 'tab-prediction':
        # Get the data for the selected stock
        if not selected_stock:
            return html.Div([
                html.Div([
                    html.Div("🔮", style={'fontSize': '60px', 'textAlign': 'center', 'marginBottom': '20px'}),
                    html.H3("Predictive Analysis", style={
                        'textAlign': 'center', 
                        'marginBottom': '15px', 
                        'fontSize': 'clamp(18px, 4vw, 26px)', 
                        'color': '#667eea',
                        'fontWeight': '700'
                    }),
                    html.P("Please select a stock to view predictive analysis", style={
                        'textAlign': 'center', 
                        'fontSize': 'clamp(14px, 3vw, 18px)', 
                        'color': '#718096',
                        'marginBottom': '30px'
                    })
                ], style={
                    'maxWidth': '900px', 
                    'marginLeft': 'auto', 
                    'marginRight': 'auto',
                    'margin': '40px auto',
                    'padding': '50px 40px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '15px',
                    'boxShadow': '0 10px 30px rgba(102, 126, 234, 0.2), 0 4px 8px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.15)'
                })
            ])
        
        # Get stock data
        stock_df = get_stock_data(selected_stock)
        if stock_df is None or stock_df.empty:
            return html.Div("No data available for this stock")
        
        # For Predictive Analysis, always use ALL data regardless of timeframe
        # Technical indicators and LSTM need sufficient historical data to be accurate
        stock_df_filtered = stock_df
        
        # Calculate technical indicators
        df_with_indicators = pred.calculate_moving_averages(stock_df_filtered, [20, 50, 200])
        df_with_indicators = pred.calculate_ema(df_with_indicators, [12, 26])
        df_with_indicators = pred.calculate_rsi(df_with_indicators)
        df_with_indicators = pred.calculate_macd(df_with_indicators)
        df_with_indicators = pred.calculate_bollinger_bands(df_with_indicators)
        
        # Generate trading signals
        signals = pred.generate_trading_signals(df_with_indicators)
        
        # Create charts
        charts = pred.create_prediction_charts(df_with_indicators, selected_stock)
        
        # Generate forecast
        forecast_df, slope, intercept = pred.linear_regression_forecast(df_with_indicators, forecast_days=30)
        
        # Train LSTM model if available (with reduced parameters for faster training)
        lstm_results = None
        lstm_charts = None
        prediction_source = "cache"  # Track if predictions came from cache or on-demand training
        
        # Try to load from cache first (much faster!)
        cached_pred = pred.get_cached_prediction(selected_stock)
        
        if cached_pred and cached_pred.get('from_cache'):
            # Use pre-computed predictions from cache
            logger.info(f"Using cached predictions for {selected_stock}")
            lstm_results = {
                'success': True,
                'forecast_dates': cached_pred['prediction_dates'],
                'forecast_values': cached_pred['prediction_prices'],
                'metrics': {
                    'rmse': 0,  # Metrics not available in cache
                    'mae': 0
                },
                'training_info': {
                    'train_samples': 0
                },
                'from_cache': True
            }
            prediction_source = "cache"
        elif lstm_model.is_lstm_available():
            # Fallback to on-demand training if cache miss
            logger.info(f"Cache miss for {selected_stock}, training LSTM on-demand")
            prediction_source = "on-demand"
            try:
                lstm_results = lstm_model.train_lstm_model(
                    df_with_indicators,
                    seq_length=30,  # Reduced from 60 for faster training
                    forecast_days=30,
                    epochs=20,  # Reduced from 50 for faster training
                    batch_size=32
                )
                if lstm_results and lstm_results.get('success'):
                    # Save predictions to tracker
                    from datetime import datetime
                    prediction_date = datetime.now()
                    model_info = {
                        'rmse': lstm_results['metrics']['rmse'],
                        'mae': lstm_results['metrics']['mae'],
                        'train_samples': lstm_results['training_info']['train_samples']
                    }
                    
                    # Perform proper backtesting: Train on historical data and predict on validation period
                    # This gives us real past predictions to compare with actual prices
                    last_date = df_with_indicators['Date'].max()
                    backtest_start = last_date - pd.DateOffset(months=2)  # Last 2 months for validation
                    
                    # Split data: everything before backtest_start for training
                    train_cutoff = backtest_start - pd.DateOffset(days=1)
                    backtest_df = df_with_indicators[df_with_indicators['Date'] <= train_cutoff].copy()
                    
                    if len(backtest_df) >= 100:  # Need sufficient data for backtesting
                        logger.info(f"Running backtest for {selected_stock} from {backtest_start}")
                        
                        # Train model on historical data only
                        backtest_results = lstm_model.train_lstm_model(
                            backtest_df,
                            seq_length=30,
                            forecast_days=60,  # Predict next 60 days (covers our 2-month validation period)
                            epochs=20,
                            batch_size=32
                        )
                        
                        if backtest_results and backtest_results.get('success'):
                            # Save backtest predictions
                            backtest_model_info = {
                                'type': 'backtest',
                                'rmse': backtest_results['metrics']['rmse'],
                                'mae': backtest_results['metrics']['mae'],
                                'train_samples': backtest_results['training_info']['train_samples']
                            }
                            
                            for target_date, predicted_price in zip(backtest_results['forecast_dates'], backtest_results['forecast_values']):
                                # Only save predictions within our validation window
                                if target_date >= backtest_start and target_date <= last_date:
                                    if target_date.weekday() < 5:  # Trading days only
                                        prediction_tracker.save_prediction(
                                            ticker=selected_stock,
                                            prediction_date=train_cutoff,
                                            target_date=target_date,
                                            predicted_price=predicted_price,
                                            model_info=backtest_model_info
                                        )
                    
                    # Save future forecasted values (only for trading days - weekdays)
                    for target_date, predicted_price in zip(lstm_results['forecast_dates'], lstm_results['forecast_values']):
                        # Skip weekends (Saturday=5, Sunday=6)
                        if target_date.weekday() < 5:  # Monday=0 to Friday=4
                            prediction_tracker.save_prediction(
                                ticker=selected_stock,
                                prediction_date=prediction_date,
                                target_date=target_date,
                                predicted_price=predicted_price,
                                model_info=model_info
                            )
                    
                    lstm_charts = lstm_model.create_lstm_charts(
                        df_with_indicators,
                        lstm_results,
                        selected_stock
                    )
            except Exception as e:
                logging.warning(f"LSTM training failed: {str(e)}")
                lstm_results = {'error': 'Training failed', 'message': str(e)}
                lstm_charts = None
        
        # Create LSTM charts for cached predictions too
        if lstm_results and lstm_results.get('success') and lstm_results.get('from_cache') and lstm_charts is None:
            lstm_charts = lstm_model.create_lstm_charts(
                df_with_indicators,
                lstm_results,
                selected_stock
            )
        
        # Build the UI
        return html.Div([
            # Add prediction source indicator at the top
            html.Div([
                html.Div([
                    html.Span("⚡ " if prediction_source == "cache" else "⏳ ", style={'fontSize': '16px'}),
                    html.Span(
                        "Predictions loaded from cache (instant)" if prediction_source == "cache" 
                        else "Model trained on-demand (cache unavailable)",
                        style={
                            'fontSize': 'clamp(11px, 2.2vw, 13px)', 
                            'color': '#10b981' if prediction_source == "cache" else '#f59e0b',
                            'fontWeight': '500',
                            'fontStyle': 'italic'
                        }
                    )
                ], style={
                    'textAlign': 'center',
                    'padding': '8px 16px',
                    'background': 'rgba(16, 185, 129, 0.1)' if prediction_source == "cache" else 'rgba(245, 158, 11, 0.1)',
                    'borderRadius': '8px',
                    'marginBottom': '20px',
                    'border': f'1px solid {"rgba(16, 185, 129, 0.3)" if prediction_source == "cache" else "rgba(245, 158, 11, 0.3)"}'
                })
            ]),
            
            # Trading Signals Section
            html.Div([
                html.H3("📊 Trading Signals", style={
                    'textAlign': 'center', 
                    'fontSize': 'clamp(16px, 3.5vw, 20px)',
                    'color': "black",
                    'fontWeight': '600',
                    'marginBottom': '20px'
                }),
                
                # Overall Signal
                html.Div([
                    html.Div([
                        html.Div("Overall Signal", style={'fontSize': 'clamp(12px, 2.5vw, 14px)', 'color': '#718096', 'marginBottom': '5px'}),
                        html.Div(signals['overall'], style={
                            'fontSize': 'clamp(20px, 4vw, 28px)', 
                            'fontWeight': '700',
                            'color': '#10b981' if 'BUY' in signals['overall'] else '#ef4444' if 'SELL' in signals['overall'] else '#6b7280'
                        }),
                        html.Div(f"Signal Strength: {signals['strength']}", style={
                            'fontSize': 'clamp(11px, 2.2vw, 13px)', 
                            'color': '#718096',
                            'marginTop': '5px'
                        })
                    ], style={
                        'textAlign': 'center',
                        'padding': '20px',
                        'background': 'white',
                        'borderRadius': '12px',
                        'color': 'black',
                        'boxShadow': '0 4px 12px rgba(102, 126, 234, 0.3)'
                    })
                ], style={'marginBottom': '20px'}),
                
                # Individual Indicators
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Span(f"{ind['name']}: ", style={'fontWeight': '600', 'color': '#4a5568'}),
                                html.Span(ind['signal'], style={
                                    'fontWeight': '700',
                                    'color': '#10b981' if ind['signal'] == 'BUY' else '#ef4444' if ind['signal'] == 'SELL' else '#6b7280',
                                    'marginLeft': '10px'
                                })
                            ], style={'marginBottom': '5px'}),
                            html.Div([
                                html.Span(f"Value: {ind['value']}", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096'}),
                                html.Span(f" • {ind['reason']}", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096'})
                            ])
                        ], style={
                            'padding': '12px',
                            'background': 'white',
                            'borderRadius': '8px',
                            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
                            'border': '1px solid #e2e8f0',
                            'marginBottom': '10px'
                        })
                        for ind in signals['indicators']
                    ])
                ])
            ], style={
                'maxWidth': '1200px',
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'marginBottom': '25px',
                'padding': '20px',
                'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                'borderRadius': '12px',
                'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                'border': '1px solid rgba(102, 126, 234, 0.1)'
            }),
            
            # LSTM Deep Learning Predictions (MOVED TO TOP)
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("🧠 LSTM Deep Learning Forecast", style={
                            'fontSize': 'clamp(14px, 3vw, 18px)',
                            'color': '#667eea',
                            'fontWeight': '600',
                            'marginBottom': '10px'
                        }),
                        html.P([
                            html.Strong("What it is: "),
                            "Long Short-Term Memory (LSTM) is an advanced deep learning neural network specifically designed for time series prediction. It learns complex patterns from 30 days of historical price data to forecast the next 30 days. The model uses 2 LSTM layers with 50 units each and dropout regularization."
                        ], style={
                            'fontSize': 'clamp(10px, 2vw, 12px)',
                            'color': '#4a5568',
                            'marginBottom': '8px',
                            'lineHeight': '1.5'
                        }),
                        html.P([
                            html.Strong("How it works: "),
                            "The model is trained on 80% of historical data and validated on 20% to prevent overfitting. It uses past price sequences to identify trends, seasonality, and momentum. RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) metrics indicate prediction accuracy - lower values mean better predictions."
                        ], style={
                            'fontSize': 'clamp(10px, 2vw, 12px)',
                            'color': '#4a5568',
                            'marginBottom': '12px',
                            'lineHeight': '1.5'
                        }),
                    ]),
                    
                    # LSTM Results
                    html.Div([
                        # Show LSTM results if available and successful
                        html.Div([
                            # Model Performance Metrics (Compact)
                            html.Div([
                                html.Div([
                                    html.P("RMSE", style={'fontSize': 'clamp(9px, 1.8vw, 11px)', 'color': '#718096', 'marginBottom': '3px'}),
                                    html.P(f"₹{lstm_results['metrics']['rmse']:.2f}", style={'fontSize': 'clamp(12px, 2.5vw, 14px)', 'color': '#667eea', 'fontWeight': 'bold'})
                                ], style={'flex': '1', 'textAlign': 'center', 'padding': '8px', 'background': '#f7fafc', 'borderRadius': '6px', 'margin': '3px'}),
                                html.Div([
                                    html.P("MAE", style={'fontSize': 'clamp(9px, 1.8vw, 11px)', 'color': '#718096', 'marginBottom': '3px'}),
                                    html.P(f"₹{lstm_results['metrics']['mae']:.2f}", style={'fontSize': 'clamp(12px, 2.5vw, 14px)', 'color': '#667eea', 'fontWeight': 'bold'})
                                ], style={'flex': '1', 'textAlign': 'center', 'padding': '8px', 'background': '#f7fafc', 'borderRadius': '6px', 'margin': '3px'}),
                                html.Div([
                                    html.P("Samples", style={'fontSize': 'clamp(9px, 1.8vw, 11px)', 'color': '#718096', 'marginBottom': '3px'}),
                                    html.P(f"{lstm_results['training_info']['train_samples']}", style={'fontSize': 'clamp(12px, 2.5vw, 14px)', 'color': '#667eea', 'fontWeight': 'bold'})
                                ], style={'flex': '1', 'textAlign': 'center', 'padding': '8px', 'background': '#f7fafc', 'borderRadius': '6px', 'margin': '3px'}),
                            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'marginBottom': '15px'}),
                            
                            # Compact Charts
                            html.Div([
                                dcc.Graph(
                                    figure=lstm_charts.get('forecast'),
                                    config={'displayModeBar': False},
                                    style={'height': '300px'}
                                ) if lstm_charts and 'forecast' in lstm_charts else html.Div(),
                            ])
                            
                        ]) if lstm_results and lstm_charts and 'metrics' in lstm_results and 'success' in lstm_results and lstm_results['success'] else html.Div([
                            html.P(f"⚠️ {lstm_results.get('message', 'LSTM training failed')}", style={
                                'textAlign': 'center',
                                'color': '#ef4444',
                                'fontSize': 'clamp(11px, 2.2vw, 13px)',
                                'padding': '15px'
                            })
                        ]) if lstm_results and 'error' in lstm_results else html.Div([
                            html.Div("🧠", style={
                                'fontSize': '48px',
                                'textAlign': 'center',
                                'marginBottom': '15px'
                            }),
                            html.P("Training LSTM model and running backtests...", style={
                                'textAlign': 'center',
                                'color': '#667eea',
                                'fontSize': 'clamp(13px, 2.8vw, 16px)',
                                'fontWeight': '600',
                                'marginBottom': '10px'
                            }),
                            html.P("This may take 1-2 minutes. Please wait...", style={
                                'textAlign': 'center',
                                'color': '#718096',
                                'fontSize': 'clamp(11px, 2.2vw, 13px)',
                                'marginBottom': '15px'
                            }),
                            html.Ul([
                                html.Li("Training LSTM neural network", style={'marginBottom': '5px'}),
                                html.Li("Running 2-month backtest validation", style={'marginBottom': '5px'}),
                                html.Li("Generating 30-day forecasts", style={'marginBottom': '5px'})
                            ], style={
                                'textAlign': 'left',
                                'color': '#4a5568',
                                'fontSize': 'clamp(11px, 2.2vw, 13px)',
                                'maxWidth': '400px',
                                'margin': '0 auto 15px',
                                'listStyle': 'none',
                                'paddingLeft': '0'
                            }),
                            html.P("💡 Tip: LSTM networks learn patterns from historical price movements to predict future trends", style={
                                'textAlign': 'center',
                                'color': '#718096',
                                'fontSize': 'clamp(10px, 2vw, 12px)',
                                'fontStyle': 'italic',
                                'padding': '12px',
                                'background': '#f7fafc',
                                'borderRadius': '8px',
                                'maxWidth': '450px',
                                'margin': '0 auto'
                            })
                        ], style={'padding': '30px 20px'}) if lstm_model.is_lstm_available() else html.Div([
                            html.P("⚠️ TensorFlow not available.", style={
                                'textAlign': 'center',
                                'color': '#ef4444',
                                'fontSize': 'clamp(11px, 2.2vw, 13px)',
                                'padding': '15px'
                            })
                        ])
                    ])
                ], style={
                    'padding': '15px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)',
                    'marginBottom': '15px'
                })
            ], style={
                'maxWidth': '1200px',
                'marginLeft': 'auto',
                'marginRight': 'auto',
            }),
            
            # Prediction History & Accuracy Section
            html.Div([
                html.Div([
                    html.H4("📊 Prediction History & Accuracy", style={
                        'fontSize': 'clamp(14px, 3vw, 18px)',
                        'color': '#667eea',
                        'fontWeight': '600',
                        'marginBottom': '10px'
                    }),
                    html.P([
                        html.Strong("What it shows: "),
                        "Historical record of LSTM predictions vs actual prices. This tracks how accurate our model has been over time, helping you understand prediction reliability."
                    ], style={
                        'fontSize': 'clamp(10px, 2vw, 12px)',
                        'color': '#4a5568',
                        'marginBottom': '12px',
                        'lineHeight': '1.5'
                    }),
                    
                    # Get prediction history and accuracy
                    html.Div(id='prediction-history-content', children=create_prediction_history_ui(selected_stock))
                    
                ], style={
                    'padding': '15px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)',
                    'marginBottom': '15px'
                })
            ], style={
                'maxWidth': '1200px',
                'marginLeft': 'auto',
                'marginRight': 'auto',
            }),
            
            # Technical Indicators - Compact Cards
            html.Div([
                # Moving Averages Card
                html.Div([
                    html.Div([
                        html.H4("📈 Moving Averages", style={
                            'fontSize': 'clamp(13px, 2.8vw, 16px)',
                            'color': '#667eea',
                            'fontWeight': '600',
                            'marginBottom': '8px'
                        }),
                        html.P([
                            html.Strong("What it shows: "),
                            "Moving Averages (MA) smooth out price fluctuations to reveal underlying trends. We track three timeframes: SMA 20 (short-term momentum), SMA 50 (medium-term trend), and SMA 200 (long-term trend). These act as dynamic support and resistance levels."
                        ], style={
                            'fontSize': 'clamp(10px, 2vw, 12px)',
                            'color': '#4a5568',
                            'marginBottom': '8px',
                            'lineHeight': '1.5'
                        }),
                        html.P([
                            html.Strong("How to interpret: "),
                            "When price is above the MA, it indicates an uptrend. A 'Golden Cross' (SMA 50 crossing above SMA 200) is a strong bullish signal, while a 'Death Cross' (SMA 50 crossing below SMA 200) suggests bearish conditions."
                        ], style={
                            'fontSize': 'clamp(10px, 2vw, 12px)',
                            'color': '#4a5568',
                            'marginBottom': '10px',
                            'lineHeight': '1.5'
                        }),
                    ]),
                    dcc.Graph(figure=charts['moving_averages'], config={'displayModeBar': False}, style={'height': '300px'})
                ], style={
                    'marginBottom': '15px',
                    'padding': '15px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)'
                }),
                
                # RSI Card
                html.Div([
                    html.Div([
                        html.H4("📊 RSI (Relative Strength Index)", style={
                            'fontSize': 'clamp(13px, 2.8vw, 16px)',
                            'color': '#667eea',
                            'fontWeight': '600',
                            'marginBottom': '8px'
                        }),
                        html.P([
                            html.Strong("What it shows: "),
                            "RSI measures the magnitude and velocity of price changes on a scale of 0-100. It identifies whether a stock is potentially overbought or oversold based on recent price momentum over a 14-day period."
                        ], style={
                            'fontSize': 'clamp(10px, 2vw, 12px)',
                            'color': '#4a5568',
                            'marginBottom': '8px',
                            'lineHeight': '1.5'
                        }),
                        html.P([
                            html.Strong("How to interpret: "),
                            "RSI below 30 suggests the stock is oversold and may be due for a bounce (potential buy opportunity). RSI above 70 indicates overbought conditions and possible price correction (potential sell signal). RSI between 30-70 shows neutral momentum."
                        ], style={
                            'fontSize': 'clamp(10px, 2vw, 12px)',
                            'color': '#4a5568',
                            'marginBottom': '10px',
                            'lineHeight': '1.5'
                        }),
                    ]),
                    dcc.Graph(figure=charts.get('rsi'), config={'displayModeBar': False}, style={'height': '300px'})
                ], style={
                    'marginBottom': '15px',
                    'padding': '15px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)'
                }) if 'rsi' in charts else html.Div(),
                
                # MACD Card
                html.Div([
                    html.Div([
                        html.H4("📈 MACD (Moving Average Convergence Divergence)", style={
                            'fontSize': 'clamp(13px, 2.8vw, 16px)',
                            'color': '#667eea',
                            'fontWeight': '600',
                            'marginBottom': '8px'
                        }),
                        html.P([
                            html.Strong("What it shows: "),
                            "MACD reveals the relationship between two exponential moving averages (12-day and 26-day EMAs). It consists of the MACD line, signal line (9-day EMA of MACD), and histogram (difference between MACD and signal line) to show momentum changes."
                        ], style={
                            'fontSize': 'clamp(10px, 2vw, 12px)',
                            'color': '#4a5568',
                            'marginBottom': '8px',
                            'lineHeight': '1.5'
                        }),
                        html.P([
                            html.Strong("How to interpret: "),
                            "When the MACD line crosses above the signal line, it's a bullish signal suggesting upward momentum. Crossing below indicates bearish momentum. The histogram shows the strength of the trend - larger bars mean stronger momentum."
                        ], style={
                            'fontSize': 'clamp(10px, 2vw, 12px)',
                            'color': '#4a5568',
                            'marginBottom': '10px',
                            'lineHeight': '1.5'
                        }),
                    ]),
                    dcc.Graph(figure=charts.get('macd'), config={'displayModeBar': False}, style={'height': '300px'})
                ], style={
                    'marginBottom': '15px',
                    'padding': '15px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)'
                }) if 'macd' in charts else html.Div(),
                
                # Bollinger Bands Card
                html.Div([
                    html.Div([
                        html.H4("📊 Bollinger Bands", style={
                            'fontSize': 'clamp(13px, 2.8vw, 16px)',
                            'color': '#667eea',
                            'fontWeight': '600',
                            'marginBottom': '8px'
                        }),
                        html.P([
                            html.Strong("What it shows: "),
                            "Bollinger Bands consist of three lines: a 20-day simple moving average (middle band) and two bands set at 2 standard deviations above and below. The bands expand during high volatility and contract during low volatility periods."
                        ], style={
                            'fontSize': 'clamp(10px, 2vw, 12px)',
                            'color': '#4a5568',
                            'marginBottom': '8px',
                            'lineHeight': '1.5'
                        }),
                        html.P([
                            html.Strong("How to interpret: "),
                            "When price touches the lower band, it may be oversold (potential buy). When it touches the upper band, it may be overbought (potential sell). Price bouncing between bands indicates ranging market. Band width expansion signals increasing volatility."
                        ], style={
                            'fontSize': 'clamp(10px, 2vw, 12px)',
                            'color': '#4a5568',
                            'marginBottom': '10px',
                            'lineHeight': '1.5'
                        }),
                    ]),
                    dcc.Graph(figure=charts.get('bollinger_bands'), config={'displayModeBar': False}, style={'height': '300px'})
                ], style={
                    'marginBottom': '15px',
                    'padding': '15px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15), 0 2px 4px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)'
                }) if 'bollinger_bands' in charts else html.Div()
                
            ], style={
                'maxWidth': '1200px',
                'marginLeft': 'auto',
                'marginRight': 'auto'
            })
        ])
    
    elif tab == 'tab-backtesting':
        # Get the data for the selected stock
        if not selected_stock:
            return html.Div([
                html.Div([
                    html.Div("📈", style={'fontSize': '60px', 'textAlign': 'center', 'marginBottom': '20px'}),
                    html.H3("Backtesting", style={
                        'textAlign': 'center', 
                        'marginBottom': '15px', 
                        'fontSize': 'clamp(18px, 4vw, 26px)', 
                        'color': '#667eea',
                        'fontWeight': '700'
                    }),
                    html.P("Please select a stock to run backtesting", style={
                        'textAlign': 'center', 
                        'fontSize': 'clamp(14px, 3vw, 18px)', 
                        'color': '#718096',
                        'marginBottom': '15px'
                    }),
                    html.P("Backtesting validates LSTM predictions by training on historical data and testing on the last 6 months.", style={
                        'textAlign': 'center', 
                        'fontSize': 'clamp(12px, 2.5vw, 14px)', 
                        'color': '#9ca3af',
                        'maxWidth': '600px',
                        'marginLeft': 'auto',
                        'marginRight': 'auto'
                    })
                ], style={
                    'maxWidth': '900px', 
                    'marginLeft': 'auto', 
                    'marginRight': 'auto',
                    'margin': '40px auto',
                    'padding': '50px 40px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '15px',
                    'boxShadow': '0 10px 30px rgba(102, 126, 234, 0.2), 0 4px 8px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.15)'
                })
            ])
        
        # Get stock data
        stock_df = get_stock_data(selected_stock)
        if stock_df is None or stock_df.empty:
            return html.Div("No data available for this stock")
        
        # Run backtesting
        try:
            # Get full data (not filtered by timeframe for backtesting)
            backtest_results = backtesting.run_backtesting(
                cached_df if cached_df else None,
                selected_stock,
                test_months=6,
                seq_length=30,
                epochs=20
            )
            
            if not backtest_results or not backtest_results.get('success'):
                error_msg = backtest_results.get('error', 'Unknown error') if backtest_results else 'Backtesting failed'
                return html.Div([
                    html.Div([
                        html.Div("⚠️", style={'fontSize': '50px', 'textAlign': 'center', 'marginBottom': '15px'}),
                        html.H4("Backtesting Error", style={
                            'textAlign': 'center', 
                            'color': '#ef4444',
                            'marginBottom': '10px'
                        }),
                        html.P(error_msg, style={
                            'textAlign': 'center', 
                            'color': '#718096'
                        })
                    ], style={
                        'maxWidth': '600px',
                        'marginLeft': 'auto',
                        'marginRight': 'auto',
                        'padding': '40px',
                        'background': 'white',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'
                    })
                ])
            
            charts = backtest_results['charts']
            metrics = charts['metrics']
            
            # Build backtesting UI
            return html.Div([
                # Header Info
                html.Div([
                    html.H3(f"📈 Backtesting Results: {selected_stock}", style={
                        'textAlign': 'center',
                        'fontSize': 'clamp(16px, 3.5vw, 22px)',
                        'color': '#667eea',
                        'fontWeight': '700',
                        'marginBottom': '15px'
                    }),
                    html.Div([
                        html.Div([
                            html.Div("Training Period", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '5px'}),
                            html.Div(backtest_results['train_period'], style={'fontSize': 'clamp(11px, 2.2vw, 14px)', 'fontWeight': '600', 'color': '#4a5568'}),
                            html.Div(f"({backtest_results['train_days']} days)", style={'fontSize': 'clamp(9px, 1.8vw, 11px)', 'color': '#9ca3af', 'marginTop': '2px'})
                        ], style={
                            'textAlign': 'center',
                            'padding': '15px',
                            'background': 'white',
                            'borderRadius': '10px',
                            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
                            'border': '1px solid #e2e8f0'
                        }),
                        html.Div([
                            html.Div("Testing Period", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '5px'}),
                            html.Div(backtest_results['test_period'], style={'fontSize': 'clamp(11px, 2.2vw, 14px)', 'fontWeight': '600', 'color': '#4a5568'}),
                            html.Div(f"({backtest_results['test_days']} days)", style={'fontSize': 'clamp(9px, 1.8vw, 11px)', 'color': '#9ca3af', 'marginTop': '2px'})
                        ], style={
                            'textAlign': 'center',
                            'padding': '15px',
                            'background': 'white',
                            'borderRadius': '10px',
                            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
                            'border': '1px solid #e2e8f0'
                        })
                    ], style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                        'gap': '15px',
                        'marginBottom': '20px'
                    })
                ], style={
                    'maxWidth': '1200px',
                    'marginLeft': 'auto',
                    'marginRight': 'auto',
                    'marginBottom': '25px',
                    'padding': '20px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)'
                }),
                
                # Performance Metrics
                html.Div([
                    html.H4("🎯 Performance Metrics", style={
                        'textAlign': 'center',
                        'fontSize': 'clamp(14px, 3vw, 18px)',
                        'color': '#667eea',
                        'fontWeight': '600',
                        'marginBottom': '15px'
                    }),
                    html.Div([
                        html.Div([
                            html.Div("RMSE", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '5px'}),
                            html.Div(f"₹{metrics['rmse']:.2f}", style={'fontSize': 'clamp(16px, 3.5vw, 20px)', 'fontWeight': '700', 'color': '#667eea'}),
                            html.Div("Root Mean Square Error", style={'fontSize': 'clamp(9px, 1.8vw, 10px)', 'color': '#9ca3af', 'marginTop': '2px'})
                        ], style={
                            'textAlign': 'center',
                            'padding': '15px',
                            'background': 'white',
                            'borderRadius': '10px',
                            'boxShadow': '0 4px 12px rgba(102, 126, 234, 0.2)',
                            'border': '1px solid rgba(102, 126, 234, 0.1)'
                        }),
                        html.Div([
                            html.Div("MAE", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '5px'}),
                            html.Div(f"₹{metrics['mae']:.2f}", style={'fontSize': 'clamp(16px, 3.5vw, 20px)', 'fontWeight': '700', 'color': '#10b981'}),
                            html.Div("Mean Absolute Error", style={'fontSize': 'clamp(9px, 1.8vw, 10px)', 'color': '#9ca3af', 'marginTop': '2px'})
                        ], style={
                            'textAlign': 'center',
                            'padding': '15px',
                            'background': 'white',
                            'borderRadius': '10px',
                            'boxShadow': '0 4px 12px rgba(16, 185, 129, 0.2)',
                            'border': '1px solid rgba(16, 185, 129, 0.1)'
                        }),
                        html.Div([
                            html.Div("MAPE", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '5px'}),
                            html.Div(f"{metrics['mape']:.2f}%", style={'fontSize': 'clamp(16px, 3.5vw, 20px)', 'fontWeight': '700', 'color': '#f59e0b'}),
                            html.Div("Mean Absolute % Error", style={'fontSize': 'clamp(9px, 1.8vw, 10px)', 'color': '#9ca3af', 'marginTop': '2px'})
                        ], style={
                            'textAlign': 'center',
                            'padding': '15px',
                            'background': 'white',
                            'borderRadius': '10px',
                            'boxShadow': '0 4px 12px rgba(245, 158, 11, 0.2)',
                            'border': '1px solid rgba(245, 158, 11, 0.1)'
                        }),
                        html.Div([
                            html.Div("Direction Accuracy", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '5px'}),
                            html.Div(f"{metrics['directional_accuracy']:.1f}%", style={'fontSize': 'clamp(16px, 3.5vw, 20px)', 'fontWeight': '700', 'color': '#8b5cf6'}),
                            html.Div("Trend Prediction Accuracy", style={'fontSize': 'clamp(9px, 1.8vw, 10px)', 'color': '#9ca3af', 'marginTop': '2px'})
                        ], style={
                            'textAlign': 'center',
                            'padding': '15px',
                            'background': 'white',
                            'borderRadius': '10px',
                            'boxShadow': '0 4px 12px rgba(139, 92, 246, 0.2)',
                            'border': '1px solid rgba(139, 92, 246, 0.1)'
                        })
                    ], style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))',
                        'gap': '15px'
                    })
                ], style={
                    'maxWidth': '1200px',
                    'marginLeft': 'auto',
                    'marginRight': 'auto',
                    'marginBottom': '25px',
                    'padding': '20px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 20px rgba(102, 126, 234, 0.15)',
                    'border': '1px solid rgba(102, 126, 234, 0.1)'
                }),
                
                # Charts
                html.Div([
                    # Predicted vs Actual
                    dcc.Graph(figure=charts['comparison'], config={'displayModeBar': False}),
                    
                    # Error Distribution and Error Over Time side by side
                    html.Div([
                        html.Div([
                            dcc.Graph(figure=charts['error_dist'], config={'displayModeBar': False})
                        ], style={'width': '48%'}),
                        html.Div([
                            dcc.Graph(figure=charts['error_time'], config={'displayModeBar': False})
                        ], style={'width': '48%'})
                    ], style={
                        'display': 'flex',
                        'justifyContent': 'space-between',
                        'gap': '20px',
                        'flexWrap': 'wrap'
                    }),
                    
                    # Training Loss
                    dcc.Graph(figure=charts['training'], config={'displayModeBar': False}) if charts.get('training') else html.Div()
                ], style={
                    'maxWidth': '1200px',
                    'marginLeft': 'auto',
                    'marginRight': 'auto'
                }),
                
                # Info Box
                html.Div([
                    html.H4("📚 Understanding the Metrics", style={
                        'fontSize': 'clamp(13px, 2.8vw, 16px)',
                        'color': '#667eea',
                        'fontWeight': '600',
                        'marginBottom': '12px'
                    }),
                    html.Ul([
                        html.Li([
                            html.Strong("RMSE (Root Mean Square Error): "),
                            "Measures average prediction error. Lower is better. Values close to typical daily price movements indicate good performance."
                        ], style={'marginBottom': '8px', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568'}),
                        html.Li([
                            html.Strong("MAE (Mean Absolute Error): "),
                            "Average absolute difference between predicted and actual prices. Lower is better and easier to interpret than RMSE."
                        ], style={'marginBottom': '8px', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568'}),
                        html.Li([
                            html.Strong("MAPE (Mean Absolute Percentage Error): "),
                            "Error as a percentage of actual price. Values under 10% are excellent, under 20% are good."
                        ], style={'marginBottom': '8px', 'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568'}),
                        html.Li([
                            html.Strong("Directional Accuracy: "),
                            "Percentage of time the model correctly predicted whether price would go up or down. Above 50% is better than random, above 60% is very good."
                        ], style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#4a5568'})
                    ], style={'paddingLeft': '20px'})
                ], style={
                    'maxWidth': '1200px',
                    'marginLeft': 'auto',
                    'marginRight': 'auto',
                    'marginTop': '25px',
                    'padding': '20px',
                    'background': 'white',
                    'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                    'border': '1px solid #e2e8f0'
                })
            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'})
            
        except Exception as e:
            logger.error(f"Backtesting error: {e}", exc_info=True)
            return html.Div([
                html.Div([
                    html.Div("⚠️", style={'fontSize': '50px', 'textAlign': 'center', 'marginBottom': '15px'}),
                    html.H4("Backtesting Error", style={
                        'textAlign': 'center', 
                        'color': '#ef4444',
                        'marginBottom': '10px'
                    }),
                    html.P(str(e), style={
                        'textAlign': 'center', 
                        'color': '#718096'
                    })
                ], style={
                    'maxWidth': '600px',
                    'marginLeft': 'auto',
                    'marginRight': 'auto',
                    'padding': '40px',
                    'background': 'white',
                    'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'
                })
            ])

    elif tab == 'tab-fundamentals':
        # Fundamentals tab
        
        # Create top/bottom tables section
        tables_section = html.Div([
            html.Div([
                html.H2("📈 Fundamental Rankings", style={
                    'textAlign': 'center',
                    'color': '#667eea',
                    'fontSize': 'clamp(20px, 4.5vw, 28px)',
                    'fontWeight': '700',
                    'marginBottom': '30px'
                }),
                fundamentals.create_top_bottom_tables()
            ], style={
                'maxWidth': '1400px',
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'padding': '30px 20px',
                'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                'borderRadius': '15px',
                'boxShadow': '0 10px 30px rgba(102, 126, 234, 0.2), 0 4px 8px rgba(0,0,0,0.1)',
                'border': '1px solid rgba(102, 126, 234, 0.15)',
                'marginBottom': '30px'
            })
        ])
        
        if not selected_stock:
            return html.Div([
                tables_section,
                html.Div([
                    html.Div("💼", style={'fontSize': '60px', 'textAlign': 'center', 'marginBottom': '20px'}),
                    html.H3("Individual Stock Analysis", style={
                        'textAlign': 'center', 
                        'marginBottom': '15px', 
                        'fontSize': 'clamp(18px, 4vw, 26px)', 
                        'color': '#667eea',
                        'fontWeight': '700'
                    }),
                    html.P("Please select a stock above to view detailed fundamental analysis", style={
                        'textAlign': 'center', 
                        'fontSize': 'clamp(14px, 3vw, 18px)', 
                        'color': '#718096'
                    })
                ], style={
                    'maxWidth': '900px', 
                    'marginLeft': 'auto', 
                    'marginRight': 'auto',
                    'margin': '40px auto',
                    'padding': '50px 40px',
                    'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                    'borderRadius': '15px',
                    'boxShadow': '0 10px 30px rgba(102, 126, 234, 0.2), 0 4px 8px rgba(0,0,0,0.1)',
                    'border': '1px solid rgba(102, 126, 234, 0.15)'
                })
            ])
        
        # Fetch fundamental data
        fundamental_data = fundamentals.get_fundamental_data(selected_stock)
        
        return html.Div([
            tables_section,
            html.Div([
                html.H2(f"📊 {selected_stock} - Detailed Analysis", style={
                    'textAlign': 'center',
                    'color': '#667eea',
                    'fontSize': 'clamp(18px, 4vw, 24px)',
                    'fontWeight': '700',
                    'marginBottom': '30px'
                }),
                fundamentals.create_fundamentals_ui(selected_stock, fundamental_data)
            ], style={
                'maxWidth': '1200px',
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'padding': '30px 20px',
                'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                'borderRadius': '15px',
                'boxShadow': '0 10px 30px rgba(102, 126, 234, 0.2), 0 4px 8px rgba(0,0,0,0.1)',
                'border': '1px solid rgba(102, 126, 234, 0.15)'
            })
        ], style={
            'maxWidth': '1400px',
            'marginLeft': 'auto',
            'marginRight': 'auto',
            'marginTop': '20px'
        })
    
    elif tab == 'tab-prediction-tracking':
        # Prediction Tracking tab
        from prediction_tracker import prediction_tracker
        
        if not selected_stock:
            return html.Div([
                html.Div("🎯", style={'fontSize': '60px', 'textAlign': 'center', 'marginBottom': '20px'}),
                html.H3("Prediction Accuracy Tracking", style={
                    'textAlign': 'center', 
                    'marginBottom': '15px', 
                    'fontSize': 'clamp(18px, 4vw, 26px)', 
                    'color': '#667eea',
                    'fontWeight': '700'
                }),
                html.P("Please select a stock above to view prediction accuracy tracking", style={
                    'textAlign': 'center', 
                    'fontSize': 'clamp(14px, 3vw, 18px)', 
                    'color': '#718096'
                }),
                html.Div([
                    html.H4("📊 How it works:", style={'color': '#667eea', 'marginBottom': '15px'}),
                    html.Ul([
                        html.Li("Every night at midnight, the model generates predictions for Oct 1, 2025 onwards"),
                        html.Li("Each day's predictions are stored with timestamps"),
                        html.Li("As actual prices become available, accuracy is calculated"),
                        html.Li("Track how predictions improve as we get closer to the target date"),
                    ], style={'fontSize': '16px', 'lineHeight': '1.8', 'color': '#718096'})
                ], style={
                    'maxWidth': '600px',
                    'margin': '30px auto',
                    'padding': '25px',
                    'background': '#f8f9fa',
                    'borderRadius': '10px',
                    'border': '1px solid #e2e8f0'
                })
            ], style={
                'maxWidth': '900px', 
                'marginLeft': 'auto', 
                'marginRight': 'auto',
                'margin': '40px auto',
                'padding': '50px 40px',
                'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                'borderRadius': '15px',
                'boxShadow': '0 10px 30px rgba(102, 126, 234, 0.2), 0 4px 8px rgba(0,0,0,0.1)',
                'border': '1px solid rgba(102, 126, 234, 0.15)'
            })
        
        # Get prediction history for selected stock
        pred_history = prediction_tracker.get_prediction_history(selected_stock, days=90)
        accuracy_metrics = prediction_tracker.get_accuracy_metrics(selected_stock, days=90)
        
        if pred_history.empty:
            return html.Div([
                html.Div("📭", style={'fontSize': '60px', 'textAlign': 'center', 'marginBottom': '20px'}),
                html.H3(f"No predictions yet for {selected_stock}", style={
                    'textAlign': 'center', 
                    'marginBottom': '15px', 
                    'fontSize': 'clamp(18px, 4vw, 26px)', 
                    'color': '#667eea',
                    'fontWeight': '700'
                }),
                html.P("Predictions will appear here after the nightly generation runs", style={
                    'textAlign': 'center', 
                    'fontSize': 'clamp(14px, 3vw, 18px)', 
                    'color': '#718096'
                })
            ], style={
                'maxWidth': '900px', 
                'marginLeft': 'auto', 
                'marginRight': 'auto',
                'margin': '40px auto',
                'padding': '50px 40px',
                'background': 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                'borderRadius': '15px',
                'boxShadow': '0 10px 30px rgba(102, 126, 234, 0.2), 0 4px 8px rgba(0,0,0,0.1)',
                'border': '1px solid rgba(102, 126, 234, 0.15)'
            })
        
        # Create accuracy metrics cards
        metrics_cards = html.Div([
            html.Div([
                html.Div("📊", style={'fontSize': '30px', 'marginBottom': '10px'}),
                html.Div(f"{accuracy_metrics['verified_predictions']}", style={
                    'fontSize': '32px', 'fontWeight': 'bold', 'color': '#667eea'
                }),
                html.Div("Verified Predictions", style={'fontSize': '14px', 'color': '#718096'})
            ], style={
                'flex': '1',
                'padding': '20px',
                'background': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'textAlign': 'center'
            }),
            html.Div([
                html.Div("🎯", style={'fontSize': '30px', 'marginBottom': '10px'}),
                html.Div(f"{accuracy_metrics['mean_accuracy']:.2f}%", style={
                    'fontSize': '32px', 'fontWeight': 'bold', 'color': '#10b981'
                }),
                html.Div("Mean Accuracy", style={'fontSize': '14px', 'color': '#718096'})
            ], style={
                'flex': '1',
                'padding': '20px',
                'background': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'textAlign': 'center'
            }),
            html.Div([
                html.Div("💰", style={'fontSize': '30px', 'marginBottom': '10px'}),
                html.Div(f"₹{accuracy_metrics['mae']:.2f}", style={
                    'fontSize': '32px', 'fontWeight': 'bold', 'color': '#f59e0b'
                }),
                html.Div("Mean Error (MAE)", style={'fontSize': '14px', 'color': '#718096'})
            ], style={
                'flex': '1',
                'padding': '20px',
                'background': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'textAlign': 'center'
            })
        ], style={
            'display': 'flex',
            'gap': '20px',
            'marginBottom': '30px',
            'flexWrap': 'wrap'
        })
        
        # Create prediction history table
        table_data = pred_history[['target_date', 'prediction_date', 'predicted_price', 'actual_price', 'accuracy']].copy()
        table_data['target_date'] = table_data['target_date'].dt.strftime('%Y-%m-%d')
        table_data['prediction_date'] = table_data['prediction_date'].dt.strftime('%Y-%m-%d')
        table_data['predicted_price'] = table_data['predicted_price'].apply(lambda x: f"₹{x:.2f}")
        table_data['actual_price'] = table_data['actual_price'].apply(
            lambda x: f"₹{x:.2f}" if pd.notna(x) else "Pending"
        )
        table_data['accuracy'] = table_data['accuracy'].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "-"
        )
        
        table = html.Table([
            html.Thead(html.Tr([
                html.Th("Target Date", style={'padding': '12px', 'textAlign': 'left', 'borderBottom': '2px solid #667eea', 'color': '#667eea', 'fontWeight': '700'}),
                html.Th("Predicted On", style={'padding': '12px', 'textAlign': 'left', 'borderBottom': '2px solid #667eea', 'color': '#667eea', 'fontWeight': '700'}),
                html.Th("Predicted Price", style={'padding': '12px', 'textAlign': 'right', 'borderBottom': '2px solid #667eea', 'color': '#667eea', 'fontWeight': '700'}),
                html.Th("Actual Price", style={'padding': '12px', 'textAlign': 'right', 'borderBottom': '2px solid #667eea', 'color': '#667eea', 'fontWeight': '700'}),
                html.Th("Accuracy", style={'padding': '12px', 'textAlign': 'right', 'borderBottom': '2px solid #667eea', 'color': '#667eea', 'fontWeight': '700'})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row['target_date'], style={'padding': '12px', 'borderBottom': '1px solid #e2e8f0'}),
                    html.Td(row['prediction_date'], style={'padding': '12px', 'borderBottom': '1px solid #e2e8f0'}),
                    html.Td(row['predicted_price'], style={'padding': '12px', 'textAlign': 'right', 'borderBottom': '1px solid #e2e8f0'}),
                    html.Td(row['actual_price'], style={'padding': '12px', 'textAlign': 'right', 'borderBottom': '1px solid #e2e8f0'}),
                    html.Td(row['accuracy'], style={
                        'padding': '12px', 
                        'textAlign': 'right', 
                        'borderBottom': '1px solid #e2e8f0',
                        'color': '#10b981' if row['accuracy'] != '-' and float(row['accuracy'].rstrip('%')) >= 90 else '#f59e0b' if row['accuracy'] != '-' else '#718096',
                        'fontWeight': '600'
                    })
                ]) for _, row in table_data.iterrows()
            ])
        ], style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'background': 'white',
            'borderRadius': '10px',
            'overflow': 'hidden',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
        })
        
        return html.Div([
            html.H2(f"🎯 {selected_stock} - Prediction Accuracy Tracking", style={
                'textAlign': 'center',
                'color': '#667eea',
                'fontSize': 'clamp(20px, 4.5vw, 28px)',
                'fontWeight': '700',
                'marginBottom': '30px'
            }),
            metrics_cards,
            html.Div([
                html.H3("📋 Prediction History", style={
                    'color': '#667eea',
                    'fontSize': '20px',
                    'fontWeight': '700',
                    'marginBottom': '20px'
                }),
                table
            ], style={
                'padding': '20px',
                'background': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
            })
        ], style={
            'maxWidth': '1200px',
            'marginLeft': 'auto',
            'marginRight': 'auto',
            'marginTop': '20px',
            'padding': '30px 20px'
        })
    
    # No data case
    return html.Div("No data available", style={"textAlign": "center", "padding": "20px"})


# Callback to update the latest data date display
@app.callback(
    Output('latest-data-date', 'children'),
    Input('tabs', 'value')  # Triggers on any tab change
)
def update_latest_date(tab):
    """Update the latest data date whenever tabs change (ensures fresh data display)."""
    current_date = get_latest_date()
    return f"Latest Data: {current_date}"


# Callback to toggle timeframe selector visibility based on active tab
@app.callback(
    Output('timeframe-container', 'style'),
    Input('tabs', 'value')
)
def toggle_timeframe_visibility(tab):
    """Hide timeframe selector on Predictive Analysis, Backtesting, Fundamentals, and Prediction Tracking tabs, show on Price Analysis tab."""
    if tab in ['tab-prediction', 'tab-backtesting', 'tab-fundamentals', 'tab-prediction-tracking']:
        return {'display': 'none'}
    else:
        return {'width': '15%', 'display': 'block'}


# Callback for top performers table (outside tabs, always visible)
@app.callback(
    [Output('performers-heading', 'children'),
     Output('top-performers-table', 'children')],
    Input('timeframe-selector', 'value')
)
def update_top_performers(timeframe):
    """Update the top performers heading and table based on selected timeframe."""
    timeframe_labels = {
        'TODAY': 'Today', '1M': '1 Month', '3M': '3 Months', '6M': '6 Months',
        '1Y': '1 Year', '3Y': '3 Years', '5Y': '5 Years', 'ALL': 'All Time'
    }
    heading = f"Market Performers ({timeframe_labels.get(timeframe, timeframe)})"
    table = calculate_top_performers(timeframe, top_n=10)
    return heading, table

def calculate_top_performers(timeframe, top_n=10):
    """Calculate top 10 and worst 10 performing stocks for a given timeframe."""
    (df, dates), stocks = load_data()
    if df is None:
        return html.Div("No data available")
    
    returns = []
    
    # Filter dates based on timeframe
    latest_date = dates.max()
    if timeframe == 'TODAY':
        # For today, we need yesterday's close and today's close
        # Get the last 2 trading days
        unique_dates = sorted(dates.dropna().unique())
        if len(unique_dates) >= 2:
            start_date = unique_dates[-2]  # Yesterday (or last trading day)
        else:
            start_date = latest_date
    elif timeframe == '1M':
        start_date = latest_date - pd.DateOffset(months=1)
    elif timeframe == '3M':
        start_date = latest_date - pd.DateOffset(months=3)
    elif timeframe == '6M':
        start_date = latest_date - pd.DateOffset(months=6)
    elif timeframe == '1Y':
        start_date = latest_date - pd.DateOffset(years=1)
    elif timeframe == '3Y':
        start_date = latest_date - pd.DateOffset(years=3)
    elif timeframe == '5Y':
        start_date = latest_date - pd.DateOffset(years=5)
    else:  # ALL
        start_date = dates.min()
    
    date_mask = dates >= start_date
    
    # Calculate returns for ALL stocks
    for ticker in stocks:
        try:
            ticker_cols = [col for col in df.columns if col[0] == ticker]
            if not ticker_cols:
                continue
            
            close_col = [col for col in ticker_cols if col[1] == 'Close'][0]
            close_prices = pd.to_numeric(df[close_col], errors='coerce')
            
            # Get filtered close prices
            filtered_closes = close_prices[date_mask].dropna()
            if len(filtered_closes) < 2:
                continue
            
            start_price = filtered_closes.iloc[0]
            end_price = filtered_closes.iloc[-1]
            
            if start_price > 0:
                return_pct = ((end_price - start_price) / start_price) * 100
                returns.append({'Ticker': ticker, 'Return': return_pct})
        except:
            continue
    
    if not returns:
        return html.Div("Calculating...", style={'textAlign': 'center', 'padding': '20px'})
    
    # Sort by returns
    returns_df = pd.DataFrame(returns).sort_values('Return', ascending=False)
    
    # Get top 10 and worst 10
    top_10 = returns_df.head(top_n)
    worst_10 = returns_df.tail(top_n).sort_values('Return', ascending=True)
    
    # Create two tables side by side
    return html.Div([
        # Top Performers Table
        html.Div([
            html.H4("Top 10 Performers", style={'textAlign': 'left', 'color': '#27ae60', 'marginBottom': '10px', 'fontSize': 'clamp(12px, 2.5vw, 16px)'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Rank", style={'padding': '8px', 'borderBottom': '2px solid #ddd', 'backgroundColor': '#e8f5e9', 'textAlign': 'center', 'width': '15%'}),
                    html.Th("Ticker", style={'padding': '8px', 'borderBottom': '2px solid #ddd', 'backgroundColor': '#e8f5e9', 'textAlign': 'left', 'width': '50%'}),
                    html.Th("Return", style={'padding': '8px', 'borderBottom': '2px solid #ddd', 'backgroundColor': '#e8f5e9', 'textAlign': 'right', 'width': '35%'})
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(str(i+1), style={'padding': '6px', 'borderBottom': '1px solid #eee', 'textAlign': 'center'}),
                        html.Td(row['Ticker'].replace('.NS', ''), style={'padding': '6px', 'borderBottom': '1px solid #eee', 'fontWeight': 'bold', 'textAlign': 'left'}),
                        html.Td(f"{row['Return']:+.2f}%", style={'padding': '6px', 'borderBottom': '1px solid #eee', 'color': '#27ae60', 'fontWeight': 'bold', 'textAlign': 'right'})
                    ]) for i, (idx, row) in enumerate(top_10.iterrows())
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': 'clamp(10px, 2vw, 12px)', 'display': 'table'})
        ], style={'width': '48%', 'verticalAlign': 'top'}, className='performer-table'),
        
        # Worst Performers Table
        html.Div([
            html.H4("Worst 10 Performers", style={'textAlign': 'left', 'color': '#e74c3c', 'marginBottom': '10px', 'fontSize': 'clamp(12px, 2.5vw, 16px)'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Rank", style={'padding': '8px', 'borderBottom': '2px solid #ddd', 'backgroundColor': '#ffebee', 'textAlign': 'center', 'width': '15%'}),
                    html.Th("Ticker", style={'padding': '8px', 'borderBottom': '2px solid #ddd', 'backgroundColor': '#ffebee', 'textAlign': 'left', 'width': '50%'}),
                    html.Th("Return", style={'padding': '8px', 'borderBottom': '2px solid #ddd', 'backgroundColor': '#ffebee', 'textAlign': 'right', 'width': '35%'})
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(str(i+1), style={'padding': '6px', 'borderBottom': '1px solid #eee', 'textAlign': 'center'}),
                        html.Td(row['Ticker'].replace('.NS', ''), style={'padding': '6px', 'borderBottom': '1px solid #eee', 'fontWeight': 'bold', 'textAlign': 'left'}),
                        html.Td(f"{row['Return']:+.2f}%", style={'padding': '6px', 'borderBottom': '1px solid #eee', 'color': '#e74c3c', 'fontWeight': 'bold', 'textAlign': 'right'})
                    ]) for i, (idx, row) in enumerate(worst_10.iterrows())
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': 'clamp(10px, 2vw, 12px)', 'display': 'table'})
        ], style={'width': '48%', 'verticalAlign': 'top'}, className='performer-table'),
    ], style={'display': 'flex', 'gap': '2%', 'flexWrap': 'wrap', 'justifyContent': 'space-between'})

if __name__ == '__main__':
    import logging
    logging.getLogger('werkzeug').setLevel(logging.DEBUG)
    
    # Add external CSS for responsive design
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                /* Mobile responsive styles */
                @media (max-width: 768px) {
                    .chart-half {
                        width: 100% !important;
                    }
                    
                    .top-charts {
                        flex-direction: column !important;
                    }
                    
                    .performer-table {
                        width: 100% !important;
                        display: block !important;
                    }
                    
                    .mobile-full-width {
                        width: 100% !important;
                        margin-right: 0 !important;
                    }
                    
                    /* Make dropdowns touch-friendly */
                    .Select-control {
                        min-height: 44px !important;
                    }
                }
                
                /* Desktop styles */
                @media (min-width: 769px) {
                    .chart-half {
                        width: 48% !important;
                    }
                    
                    .performer-table {
                        width: 48% !important;
                    }
                }
                
                /* Ensure tables are scrollable on small screens */
                table {
                    width: 100%;
                    overflow-x: auto;
                    display: block;
                }
                
                @media (max-width: 768px) {
                    table {
                        font-size: 10px !important;
                    }
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    app.run(host='0.0.0.0', port=8060, debug=True)