import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
from pathlib import Path
import logging
import time
import predictive_analysis as pred
import lstm_model
import backtesting

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, 
                requests_pathname_prefix='/indiastock/',
                routes_pathname_prefix='/indiastock/',
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

def load_data():
    """Load stock data from nse_all_10y.csv with multi-index columns."""
    global cached_df, stock_list, cache_timestamp
    
    # Check if cache is still valid (within 1 hour)
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
        html.P(f"Latest Data: {latest_data_date}", 
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
    ], style={
        'maxWidth': '1200px', 
        'marginLeft': 'auto', 
        'marginRight': 'auto', 
        'marginBottom': '0'
    }),
    
    # Tab content with loading spinner positioned near tabs
    dcc.Loading(
        id="loading-tab-content",
        type="circle",
        color="#667eea",
        style={'minHeight': '200px'},
        parent_style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'flex-start',
            'paddingTop': '50px',
            'minHeight': '300px'
        },
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
                if timeframe == '1M':
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
                
                days = (stock_df['Date'].max() - stock_df['Date'].min()).days
                years = days / 365.25
                annualized_return = (((latest_close / earliest_close) ** (1 / years)) - 1) * 100 if years > 0 and earliest_close != 0 else 0
                
                timeframe_labels = {
                    '1M': '1 Month', '3M': '3 Months', '6M': '6 Months',
                    '1Y': '1 Year', '3Y': '3 Years', '5Y': '5 Years', 'ALL': 'All Time'
                }
                
                stock_info_content = html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Latest Close", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': 'black', 'marginBottom': '4px'}),
                            html.Div(f"₹{latest_close:.2f}", style={'fontSize': 'clamp(18px, 4vw, 24px)', 'fontWeight': '700', 'color': "black"}),
                        ], style={'textAlign': 'center', 'padding': '15px', 'background': 'white', 'borderRadius': '10px', 'color': 'black', 'boxShadow': '0 4px 12px rgba(102, 126, 234, 0.3)'}),
                        
                        html.Div([
                            html.Div(f"{timeframe_labels.get(timeframe, timeframe)} Return", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '4px'}),
                            html.Div(f"{pct_change:+.2f}%", style={'fontSize': 'clamp(18px, 4vw, 24px)', 'fontWeight': '700', 'color': '#10b981' if pct_change >= 0 else '#ef4444'}),
                        ], style={'textAlign': 'center', 'padding': '15px', 'background': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'border': '1px solid #e2e8f0'}),
                        
                        html.Div([
                            html.Div("Annualized Return", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '4px'}),
                            html.Div(f"{annualized_return:+.2f}%", style={'fontSize': 'clamp(18px, 4vw, 24px)', 'fontWeight': '700', 'color': '#10b981' if annualized_return >= 0 else '#ef4444'}),
                        ], style={'textAlign': 'center', 'padding': '15px', 'background': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'border': '1px solid #e2e8f0'}),
                    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))', 'gap': '15px', 'marginBottom': '20px'}),
                    
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
        
        # Apply timeframe filter
        if timeframe != 'ALL':
            stock_df_filtered = stock_df.copy()
            latest_date = stock_df['Date'].max()
            if timeframe == '1M':
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
            stock_df_filtered = stock_df_filtered[stock_df_filtered['Date'] >= start_date]
        else:
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
        if lstm_model.is_lstm_available():
            try:
                lstm_results = lstm_model.train_lstm_model(
                    df_with_indicators,
                    seq_length=30,  # Reduced from 60 for faster training
                    forecast_days=30,
                    epochs=20,  # Reduced from 50 for faster training
                    batch_size=32
                )
                if lstm_results:
                    lstm_charts = lstm_model.create_lstm_charts(
                        df_with_indicators,
                        lstm_results,
                        selected_stock
                    )
            except Exception as e:
                logging.warning(f"LSTM training failed: {str(e)}")
                lstm_results = {'error': 'Training failed', 'message': str(e)}
                lstm_charts = None
        
        # Build the UI
        return html.Div([
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
                            html.P("🔄 Training LSTM model... This may take 1-2 minutes.", style={
                                'textAlign': 'center',
                                'color': '#667eea',
                                'fontSize': 'clamp(11px, 2.2vw, 13px)',
                                'padding': '15px'
                            })
                        ]) if lstm_model.is_lstm_available() else html.Div([
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

    # No data case
    return html.Div("No data available", style={"textAlign": "center", "padding": "20px"})


# Callback to toggle timeframe selector visibility based on active tab
@app.callback(
    Output('timeframe-container', 'style'),
    Input('tabs', 'value')
)
def toggle_timeframe_visibility(tab):
    """Hide timeframe selector on Predictive Analysis and Backtesting tabs, show on Price Analysis tab."""
    if tab in ['tab-prediction', 'tab-backtesting']:
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
        '1M': '1 Month', '3M': '3 Months', '6M': '6 Months',
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
    if timeframe == '1M':
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