"""
Fundamentals Module
Loads and displays fundamental stock data like P/E ratio, market cap, etc. from CSV file
"""

import pandas as pd
import logging
from dash import html
import os

logger = logging.getLogger(__name__)

# Load fundamentals data once at module import
FUNDAMENTALS_FILE = 'fundamentals_data.csv'
_fundamentals_df = None

def load_fundamentals_data():
    """Load fundamentals data from CSV file"""
    global _fundamentals_df
    
    if _fundamentals_df is not None:
        return _fundamentals_df
    
    if not os.path.exists(FUNDAMENTALS_FILE):
        logger.warning(f"{FUNDAMENTALS_FILE} not found. Run update_fundamentals.py first.")
        return None
    
    try:
        _fundamentals_df = pd.read_csv(FUNDAMENTALS_FILE)
        logger.info(f"Loaded fundamental data for {len(_fundamentals_df)} stocks")
        return _fundamentals_df
    except Exception as e:
        logger.error(f"Error loading fundamentals data: {e}")
        return None

def get_fundamental_data(ticker):
    """
    Get fundamental data for a stock ticker from CSV
    
    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE.NS')
    
    Returns:
        Dictionary with fundamental metrics
    """
    try:
        df = load_fundamentals_data()
        if df is None:
            return None
        
        # Find the ticker in the dataframe
        stock_data = df[df['ticker'] == ticker]
        
        if stock_data.empty:
            logger.warning(f"No fundamental data found for {ticker}")
            return None
        
        # Convert to dictionary
        row = stock_data.iloc[0]
        fundamentals = {
            'trailing_pe': row.get('trailing_pe'),
            'forward_pe': row.get('forward_pe'),
            'price_to_book': row.get('price_to_book'),
            'market_cap': row.get('market_cap'),
            'enterprise_value': row.get('enterprise_value'),
            'trailing_eps': row.get('trailing_eps'),
            'forward_eps': row.get('forward_eps'),
            'dividend_yield': row.get('dividend_yield'),
            'payout_ratio': row.get('payout_ratio'),
            'profit_margins': row.get('profit_margins'),
            'operating_margins': row.get('operating_margins'),
            'return_on_equity': row.get('return_on_equity'),
            'return_on_assets': row.get('return_on_assets'),
            'revenue_growth': row.get('revenue_growth'),
            'earnings_growth': row.get('earnings_growth'),
            'current_ratio': row.get('current_ratio'),
            'quick_ratio': row.get('quick_ratio'),
            'debt_to_equity': row.get('debt_to_equity'),
            'book_value': row.get('book_value'),
            'fifty_two_week_high': row.get('fifty_two_week_high'),
            'fifty_two_week_low': row.get('fifty_two_week_low'),
            'beta': row.get('beta'),
            'shares_outstanding': row.get('shares_outstanding')
        }
        
        return fundamentals
        
    except Exception as e:
        logger.error(f"Error getting fundamental data for {ticker}: {e}")
        return None

def format_large_number(num):
    """Format large numbers with K, M, B, T suffixes"""
    if num is None:
        return "N/A"
    
    try:
        num = float(num)
        if num >= 1e12:
            return f"₹{num/1e12:.2f}T"
        elif num >= 1e9:
            return f"₹{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"₹{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"₹{num/1e3:.2f}K"
        else:
            return f"₹{num:.2f}"
    except:
        return "N/A"

def format_percentage(num):
    """Format percentage values"""
    if num is None:
        return "N/A"
    try:
        return f"{float(num) * 100:.2f}%"
    except:
        return "N/A"

def format_ratio(num):
    """Format ratio values"""
    if num is None:
        return "N/A"
    try:
        return f"{float(num):.2f}"
    except:
        return "N/A"

def calculate_fundamental_score(row):
    """
    Calculate a composite fundamental score for a stock
    Higher score = Better fundamentals
    
    Scoring criteria:
    - ROE: Higher is better (weight: 30%)
    - Profit Margin: Higher is better (weight: 20%)
    - ROA: Higher is better (weight: 15%)
    - Current Ratio: 1.5-3.0 is ideal (weight: 10%)
    - Debt to Equity: Lower is better, penalize >1 (weight: 15%)
    - Earnings Growth: Higher is better (weight: 10%)
    """
    score = 0
    valid_metrics = 0
    
    # ROE Score (30% weight) - normalize to 0-30 range
    if pd.notna(row.get('return_on_equity')) and row['return_on_equity'] is not None:
        roe = float(row['return_on_equity'])
        if roe > 0:
            score += min(roe * 100, 30)  # Cap at 30
            valid_metrics += 1
    
    # Profit Margin Score (20% weight)
    if pd.notna(row.get('profit_margins')) and row['profit_margins'] is not None:
        pm = float(row['profit_margins'])
        if pm > 0:
            score += min(pm * 100, 20)  # Cap at 20
            valid_metrics += 1
    
    # ROA Score (15% weight)
    if pd.notna(row.get('return_on_assets')) and row['return_on_assets'] is not None:
        roa = float(row['return_on_assets'])
        if roa > 0:
            score += min(roa * 100, 15)  # Cap at 15
            valid_metrics += 1
    
    # Current Ratio Score (10% weight) - ideal is 1.5-3.0
    if pd.notna(row.get('current_ratio')) and row['current_ratio'] is not None:
        cr = float(row['current_ratio'])
        if 1.5 <= cr <= 3.0:
            score += 10
            valid_metrics += 1
        elif cr > 1.0:
            score += 5
            valid_metrics += 1
    
    # Debt to Equity Score (15% weight) - lower is better
    if pd.notna(row.get('debt_to_equity')) and row['debt_to_equity'] is not None:
        de = float(row['debt_to_equity'])
        if de < 0.5:
            score += 15
            valid_metrics += 1
        elif de < 1.0:
            score += 10
            valid_metrics += 1
        elif de < 2.0:
            score += 5
            valid_metrics += 1
    
    # Earnings Growth Score (10% weight)
    if pd.notna(row.get('earnings_growth')) and row['earnings_growth'] is not None:
        eg = float(row['earnings_growth'])
        if eg > 0:
            score += min(eg * 50, 10)  # Cap at 10
            valid_metrics += 1
    
    # Return None if not enough valid metrics
    if valid_metrics < 3:
        return None
    
    return round(score, 2)

def get_top_bottom_fundamentals(n=10):
    """
    Get stocks with highest and lowest fundamental scores
    
    Args:
        n: Number of stocks to return in each list
    
    Returns:
        Tuple of (top_stocks, bottom_stocks) DataFrames
    """
    df = load_fundamentals_data()
    if df is None:
        return None, None
    
    # Calculate scores for all stocks
    df['fundamental_score'] = df.apply(calculate_fundamental_score, axis=1)
    
    # Filter out stocks with no score
    scored_df = df[df['fundamental_score'].notna()].copy()
    
    if scored_df.empty:
        return None, None
    
    # Sort by score
    scored_df = scored_df.sort_values('fundamental_score', ascending=False)
    
    # Get top and bottom n stocks
    top_stocks = scored_df.head(n)[['ticker', 'fundamental_score', 'return_on_equity', 
                                      'profit_margins', 'debt_to_equity', 'market_cap']]
    bottom_stocks = scored_df.tail(n)[['ticker', 'fundamental_score', 'return_on_equity', 
                                         'profit_margins', 'debt_to_equity', 'market_cap']]
    
    return top_stocks, bottom_stocks

def create_top_bottom_tables():
    """Create UI for top and bottom fundamental stocks"""
    top_stocks, bottom_stocks = get_top_bottom_fundamentals(10)
    
    if top_stocks is None or bottom_stocks is None:
        return html.Div([
            html.P("Unable to load fundamental rankings", style={
                'textAlign': 'center',
                'color': '#ef4444',
                'padding': '20px'
            })
        ])
    
    # Top Stocks Table
    top_table = html.Div([
        html.H3("🏆 Top 10 Stocks by Fundamental Score", style={
            'color': '#10b981',
            'fontSize': 'clamp(16px, 3.5vw, 20px)',
            'fontWeight': '700',
            'marginBottom': '15px',
            'textAlign': 'center'
        }),
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Rank", style={'width': '10%', 'padding': '12px', 'background': '#10b981', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Th("Ticker", style={'width': '25%', 'padding': '12px', 'background': '#10b981', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Th("Score", style={'width': '15%', 'padding': '12px', 'background': '#10b981', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Th("ROE", style={'width': '15%', 'padding': '12px', 'background': '#10b981', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Th("Profit Margin", style={'width': '15%', 'padding': '12px', 'background': '#10b981', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Th("Debt/Equity", style={'width': '20%', 'padding': '12px', 'background': '#10b981', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(str(i+1), style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Td(row['ticker'].replace('.NS', ''), style={'padding': '10px', 'fontWeight': 'bold', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Td(f"{row['fundamental_score']:.2f}", style={'padding': '10px', 'textAlign': 'center', 'color': '#10b981', 'fontWeight': 'bold', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Td(format_percentage(row['return_on_equity']), style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Td(format_percentage(row['profit_margins']), style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Td(format_ratio(row['debt_to_equity']), style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                ], style={'background': '#f0fdf4' if i % 2 == 0 else 'white'})
                for i, (_, row) in enumerate(top_stocks.iterrows())
            ])
        ], style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'borderRadius': '8px',
            'overflow': 'hidden'
        })
    ], style={'marginBottom': '30px'})
    
    # Bottom Stocks Table
    bottom_table = html.Div([
        html.H3("⚠️ Bottom 10 Stocks by Fundamental Score", style={
            'color': '#ef4444',
            'fontSize': 'clamp(16px, 3.5vw, 20px)',
            'fontWeight': '700',
            'marginBottom': '15px',
            'textAlign': 'center'
        }),
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Rank", style={'width': '10%', 'padding': '12px', 'background': '#ef4444', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Th("Ticker", style={'width': '25%', 'padding': '12px', 'background': '#ef4444', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Th("Score", style={'width': '15%', 'padding': '12px', 'background': '#ef4444', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Th("ROE", style={'width': '15%', 'padding': '12px', 'background': '#ef4444', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Th("Profit Margin", style={'width': '15%', 'padding': '12px', 'background': '#ef4444', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Th("Debt/Equity", style={'width': '20%', 'padding': '12px', 'background': '#ef4444', 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(str(i+1), style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Td(row['ticker'].replace('.NS', ''), style={'padding': '10px', 'fontWeight': 'bold', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Td(f"{row['fundamental_score']:.2f}", style={'padding': '10px', 'textAlign': 'center', 'color': '#ef4444', 'fontWeight': 'bold', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Td(format_percentage(row['return_on_equity']), style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Td(format_percentage(row['profit_margins']), style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    html.Td(format_ratio(row['debt_to_equity']), style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                ], style={'background': '#fef2f2' if i % 2 == 0 else 'white'})
                for i, (_, row) in enumerate(bottom_stocks.iterrows())
            ])
        ], style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'borderRadius': '8px',
            'overflow': 'hidden'
        })
    ])
    
    return html.Div([top_table, bottom_table])

def create_fundamentals_ui(ticker, fundamentals):
    """
    Create UI components for fundamental analysis
    
    Args:
        ticker: Stock ticker symbol
        fundamentals: Dictionary with fundamental metrics
    
    Returns:
        Dash HTML components
    """
    if not fundamentals:
        return html.Div([
            html.P("Unable to fetch fundamental data", style={
                'textAlign': 'center',
                'color': '#ef4444',
                'padding': '20px'
            })
        ])
    
    # Check if all values are None (no actual data)
    has_data = any(v is not None for v in fundamentals.values())
    if not has_data:
        return html.Div([
            html.P("No fundamental data available for this stock", style={
                'textAlign': 'center',
                'color': '#ef4444',
                'padding': '20px'
            })
        ])
    
    # Valuation Metrics
    valuation_section = html.Div([
        html.H4("📊 Valuation Metrics", style={
            'color': '#667eea',
            'fontSize': 'clamp(14px, 3vw, 18px)',
            'fontWeight': '600',
            'marginBottom': '15px'
        }),
        html.Div([
            html.Div([
                html.P("Trailing P/E", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_ratio(fundamentals['trailing_pe']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#667eea', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("Forward P/E", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_ratio(fundamentals['forward_pe']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#667eea', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("P/B Ratio", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_ratio(fundamentals['price_to_book']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#667eea', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("Market Cap", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_large_number(fundamentals['market_cap']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#667eea', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '20px'})
    ])
    
    # Profitability Metrics
    profitability_section = html.Div([
        html.H4("💰 Profitability & Returns", style={
            'color': '#667eea',
            'fontSize': 'clamp(14px, 3vw, 18px)',
            'fontWeight': '600',
            'marginBottom': '15px'
        }),
        html.Div([
            html.Div([
                html.P("EPS (TTM)", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_ratio(fundamentals['trailing_eps']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#10b981', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("Profit Margin", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_percentage(fundamentals['profit_margins']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#10b981', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("ROE", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_percentage(fundamentals['return_on_equity']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#10b981', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("ROA", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_percentage(fundamentals['return_on_assets']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#10b981', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '20px'})
    ])
    
    # Dividend & Growth
    dividend_growth_section = html.Div([
        html.H4("📈 Dividend & Growth", style={
            'color': '#667eea',
            'fontSize': 'clamp(14px, 3vw, 18px)',
            'fontWeight': '600',
            'marginBottom': '15px'
        }),
        html.Div([
            html.Div([
                html.P("Dividend Yield", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_percentage(fundamentals['dividend_yield']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#8b5cf6', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("Earnings Growth", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_percentage(fundamentals['earnings_growth']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#8b5cf6', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("Revenue Growth", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_percentage(fundamentals['revenue_growth']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#8b5cf6', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("Beta", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_ratio(fundamentals['beta']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#8b5cf6', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '20px'})
    ])
    
    # Financial Health
    health_section = html.Div([
        html.H4("🏥 Financial Health", style={
            'color': '#667eea',
            'fontSize': 'clamp(14px, 3vw, 18px)',
            'fontWeight': '600',
            'marginBottom': '15px'
        }),
        html.Div([
            html.Div([
                html.P("Current Ratio", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_ratio(fundamentals['current_ratio']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#3b82f6', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("Quick Ratio", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_ratio(fundamentals['quick_ratio']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#3b82f6', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("Debt/Equity", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_ratio(fundamentals['debt_to_equity']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#3b82f6', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
            
            html.Div([
                html.P("Book Value", style={'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}),
                html.P(format_ratio(fundamentals['book_value']), style={'fontSize': 'clamp(14px, 2.8vw, 16px)', 'color': '#3b82f6', 'fontWeight': 'bold'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '12px', 'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'})
    ])
    
    return html.Div([
        valuation_section,
        profitability_section,
        dividend_growth_section,
        health_section
    ])
