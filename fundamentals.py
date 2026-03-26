"""
Fundamentals Module
Fetches and displays fundamental stock data on-demand from Yahoo Finance.
No stale CSV required — data is fetched live per stock.
"""

import pandas as pd
import logging
import time
from dash import html
import os

logger = logging.getLogger(__name__)

# ─── On-demand cache (per-process, lives for container uptime) ───────────────
_cache = {}          # ticker → (timestamp, data_dict)
_CACHE_TTL = 3600    # 1 hour before refreshing same ticker

# ─── Curated list for top/bottom rankings (large-cap NSE stocks) ────────────
# These are fetched in batch for ranking — refreshed periodically
_LARGE_CAP_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFOSYS.NS', 'ICICIBANK.NS',
    'HINDUSTANLU.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'LT.NS',
    'AXISBANK.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
    'NESTLEIND.NS', 'ONGC.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIENT.NS',
    'POWERGRID.NS', 'NTPC.NS', 'COALINDIA.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
    'JSWSTEEL.NS', 'HDFCLIFE.NS', 'SHRIRAMFIN.NS', 'DLF.NS', 'GRASIM.NS',
    'ADANIPOWER.NS', 'BERGEPAINT.NS', 'EICHERMOT.NS', 'HAVELLS.NS', 'HCLTECH.NS',
    'HEROMOTOCO.NS', 'INDUSINDBK.NS', 'JINDALSTEL.NS', 'LODHA.NS', 'M&M.NS',
    'NAUKRI.NS', 'OFSS.NS', 'PERSISTENT.NS', 'PIDILITIND.NS', 'SBILIFE.NS',
    'TATACONSUM.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS',
    'APOLLOHOSP.NS', 'Bajaj AUTO.NS', 'CIPLA.NS', 'DRREDDY.NS', 'FEDERALBNK.NS',
]

_rankings_cache = {}  # 'top_bottom' → (timestamp, (top_df, bottom_df))
_RANKINGS_TTL = 1800  # 30 minutes

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fetch_from_yfinance(ticker):
    """Fetch raw info dict from Yahoo Finance. Returns None on failure."""
    try:
        import yfinance as yf
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.get_info()
        if not info or info.get('regularMarketPrice') is None:
            logger.warning(f"No yfinance data for {ticker}")
            return None
        return info
    except Exception as e:
        logger.error(f"yfinance error for {ticker}: {e}")
        return None


def _extract_fundamentals(info):
    """Convert yfinance info dict to our fundamentals dict."""
    if info is None:
        return None

    def safe(val):
        return val if val is not None else None

    return {
        'trailing_pe':         safe(info.get('trailingPE')),
        'forward_pe':         safe(info.get('forwardPE')),
        'price_to_book':      safe(info.get('priceToBook')),
        'market_cap':         safe(info.get('marketCap')),
        'enterprise_value':   safe(info.get('enterpriseValue')),
        'trailing_eps':       safe(info.get('trailingEps')),
        'forward_eps':        safe(info.get('forwardEps')),
        'dividend_yield':     safe(info.get('dividendYield')),
        'payout_ratio':       safe(info.get('payoutRatio')),
        'profit_margins':     safe(info.get('profitMargins')),
        'operating_margins':  safe(info.get('operatingMargins')),
        'return_on_equity':   safe(info.get('returnOnEquity')),
        'return_on_assets':   safe(info.get('returnOnAssets')),
        'revenue_growth':     safe(info.get('revenueGrowth')),
        'earnings_growth':    safe(info.get('earningsGrowth')),
        'current_ratio':      safe(info.get('currentRatio')),
        'quick_ratio':        safe(info.get('quickRatio')),
        'debt_to_equity':     safe(info.get('debtToEquity')),
        'book_value':         safe(info.get('bookValue')),
        'fifty_two_week_high': safe(info.get('fiftyTwoWeekHigh')),
        'fifty_two_week_low':  safe(info.get('fiftyTwoWeekLow')),
        'beta':               safe(info.get('beta')),
        'shares_outstanding':  safe(info.get('sharesOutstanding')),
    }


def _calculate_score(fundamentals):
    """
    Score a fundamentals dict (0–100). Higher = better fundamentals.
    Returns None if not enough data.
    """
    if fundamentals is None:
        return None

    score = 0
    valid = 0

    roe = fundamentals.get('return_on_equity')
    if roe is not None and roe > 0:
        score += min(roe * 100, 30)
        valid += 1

    pm = fundamentals.get('profit_margins')
    if pm is not None and pm > 0:
        score += min(pm * 100, 20)
        valid += 1

    roa = fundamentals.get('return_on_assets')
    if roa is not None and roa > 0:
        score += min(roa * 100, 15)
        valid += 1

    cr = fundamentals.get('current_ratio')
    if cr is not None:
        if 1.5 <= cr <= 3.0:
            score += 10
            valid += 1
        elif cr > 1.0:
            score += 5
            valid += 1

    de = fundamentals.get('debt_to_equity')
    if de is not None:
        if de < 0.5:
            score += 15
            valid += 1
        elif de < 1.0:
            score += 10
            valid += 1
        elif de < 2.0:
            score += 5
            valid += 1

    eg = fundamentals.get('earnings_growth')
    if eg is not None and eg > 0:
        score += min(eg * 50, 10)
        valid += 1

    return round(score, 2) if valid >= 3 else None


# ─── Public API ───────────────────────────────────────────────────────────────

def get_fundamental_data(ticker):
    """
    Get fundamental data for a stock ticker — fetched live from Yahoo Finance.
    Results are cached for CACHE_TTL seconds.

    Args:
        ticker: Stock ticker symbol (e.g. 'RELIANCE.NS')

    Returns:
        Dictionary with fundamental metrics, or None on failure.
    """
    now = time.time()

    if ticker in _cache:
        ts, data = _cache[ticker]
        if now - ts < _CACHE_TTL:
            logger.info(f"Cache hit for {ticker}")
            return data
        # Expired — refresh
        logger.info(f"Cache expired for {ticker}, refreshing...")

    info = _fetch_from_yfinance(ticker)
    fundamentals = _extract_fundamentals(info)
    _cache[ticker] = (now, fundamentals)
    return fundamentals


def get_top_bottom_fundamentals(n=10):
    """
    Get top N and bottom N stocks by fundamental score from the large-cap list.
    Results are cached for RANKINGS_TTL seconds.

    Returns:
        Tuple of (top_df, bottom_df) — each with columns:
        ticker, fundamental_score, return_on_equity, profit_margins,
        debt_to_equity, market_cap
    """
    now = time.time()
    key = f'top_bottom_{n}'

    if key in _rankings_cache:
        ts, result = _rankings_cache[key]
        if now - ts < _RANKINGS_TTL:
            logger.info("Returning cached top/bottom rankings")
            return result

    rows = []
    for ticker in _LARGE_CAP_TICKERS:
        # Try cache first
        fundamentals = get_fundamental_data(ticker)
        score = _calculate_score(fundamentals)

        row = {
            'ticker': ticker,
            'fundamental_score': score,
            'return_on_equity': fundamentals.get('return_on_equity') if fundamentals else None,
            'profit_margins': fundamentals.get('profit_margins') if fundamentals else None,
            'debt_to_equity': fundamentals.get('debt_to_equity') if fundamentals else None,
            'market_cap': fundamentals.get('market_cap') if fundamentals else None,
        }
        rows.append(row)
        time.sleep(0.05)  # be kind to yfinance

    df = pd.DataFrame(rows)
    scored = df[df['fundamental_score'].notna()].sort_values('fundamental_score', ascending=False)
    top_stocks = scored.head(n)
    bottom_stocks = scored.tail(n)

    _rankings_cache[key] = (now, (top_stocks, bottom_stocks))
    return top_stocks, bottom_stocks


# ─── Formatting helpers ───────────────────────────────────────────────────────

def format_large_number(num):
    """Format large numbers with K, M, B, T suffixes."""
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
    """Format percentage values."""
    if num is None:
        return "N/A"
    try:
        return f"{float(num) * 100:.2f}%"
    except:
        return "N/A"


def format_ratio(num):
    """Format ratio values."""
    if num is None:
        return "N/A"
    try:
        return f"{float(num):.2f}"
    except:
        return "N/A"


# ─── UI builders ─────────────────────────────────────────────────────────────

def create_top_bottom_tables():
    """Create UI for top and bottom fundamental stocks."""
    top_stocks, bottom_stocks = get_top_bottom_fundamentals(10)

    if top_stocks is None or bottom_stocks is None or top_stocks.empty:
        return html.Div([
            html.P("Unable to load fundamental rankings — try again shortly.", style={
                'textAlign': 'center',
                'color': '#ef4444',
                'padding': '20px'
            })
        ])

    def make_table(stocks, color, label, row_bg):
        rows = []
        for i, (_, row) in enumerate(stocks.iterrows()):
            rows.append(html.Tr([
                html.Td(str(i + 1), style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                html.Td(row['ticker'].replace('.NS', ''), style={'padding': '10px', 'fontWeight': 'bold', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                html.Td(f"{row['fundamental_score']:.2f}" if row['fundamental_score'] else "N/A",
                        style={'padding': '10px', 'textAlign': 'center', 'color': color, 'fontWeight': 'bold', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                html.Td(format_percentage(row['return_on_equity']),
                        style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                html.Td(format_percentage(row['profit_margins']),
                        style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                html.Td(format_ratio(row['debt_to_equity']),
                        style={'padding': '10px', 'textAlign': 'center', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
            ], style={'background': row_bg if i % 2 == 0 else 'white'}))

        return html.Div([
            html.H3(label, style={
                'color': color,
                'fontSize': 'clamp(16px, 3.5vw, 20px)',
                'fontWeight': '700',
                'marginBottom': '15px',
                'textAlign': 'center'
            }),
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Rank", style={'padding': '12px', 'background': color, 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                        html.Th("Ticker", style={'padding': '12px', 'background': color, 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                        html.Th("Score", style={'padding': '12px', 'background': color, 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                        html.Th("ROE", style={'padding': '12px', 'background': color, 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                        html.Th("Margin", style={'padding': '12px', 'background': color, 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                        html.Th("D/E", style={'padding': '12px', 'background': color, 'color': 'white', 'fontSize': 'clamp(10px, 2vw, 12px)'}),
                    ])
                ),
                html.Tbody(rows)
            ], style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'borderRadius': '8px',
                'overflow': 'hidden'
            })
        ], style={'marginBottom': '30px'})

    top_table = make_table(top_stocks, '#10b981', '🏆 Top 10 Stocks by Fundamental Score', '#f0fdf4')
    bottom_table = make_table(bottom_stocks, '#ef4444', '⚠️ Bottom 10 Stocks by Fundamental Score', '#fef2f2')

    return html.Div([top_table, bottom_table])


def create_fundamentals_ui(ticker, fundamentals):
    """
    Create UI components for fundamental analysis.

    Args:
        ticker: Stock ticker symbol
        fundamentals: Dictionary with fundamental metrics (from get_fundamental_data)

    Returns:
        Dash HTML components
    """
    if not fundamentals:
        return html.Div([
            html.P("No fundamental data available for this stock. Yahoo Finance may not have data for it.",
                   style={'textAlign': 'center', 'color': '#ef4444', 'padding': '20px'})
        ])

    has_data = any(v is not None for v in fundamentals.values())
    if not has_data:
        return html.Div([
            html.P("Fundamental data is empty for this stock — try selecting a different one.",
                   style={'textAlign': 'center', 'color': '#ef4444', 'padding': '20px'})
        ])

    CARD = {'flex': '1', 'textAlign': 'center', 'padding': '12px',
            'background': '#f7fafc', 'borderRadius': '8px', 'margin': '5px', 'minWidth': '120px'}
    LABEL = {'fontSize': 'clamp(10px, 2vw, 12px)', 'color': '#718096', 'marginBottom': '3px'}
    VALUE = {'fontSize': 'clamp(14px, 2.8vw, 16px)', 'fontWeight': 'bold'}

    def metric(label, value, color='#667eea'):
        return html.Div([
            html.P(label, style=LABEL),
            html.P(value if value else "N/A", style={**VALUE, 'color': color})
        ], style=CARD)

    def row(*children):
        return html.Div(list(children),
                        style={'display': 'flex', 'flexWrap': 'wrap',
                               'justifyContent': 'space-around', 'marginBottom': '20px'})

    def section(heading, *children):
        return html.Div([
            html.H4(heading, style={
                'color': '#667eea',
                'fontSize': 'clamp(14px, 3vw, 18px)',
                'fontWeight': '600',
                'marginBottom': '15px'
            }),
            *children
        ])

    return html.Div([
        section("📊 Valuation Metrics",
                row(metric("Trailing P/E", format_ratio(fundamentals['trailing_pe'])),
                    metric("Forward P/E", format_ratio(fundamentals['forward_pe'])),
                    metric("P/B Ratio", format_ratio(fundamentals['price_to_book'])),
                    metric("Market Cap", format_large_number(fundamentals['market_cap'])))),

        section("💰 Profitability & Returns",
                row(metric("EPS (TTM)", format_ratio(fundamentals['trailing_eps'])),
                    metric("Profit Margin", format_percentage(fundamentals['profit_margins'])),
                    metric("ROE", format_percentage(fundamentals['return_on_equity'])),
                    metric("ROA", format_percentage(fundamentals['return_on_assets'])))),

        section("📈 Dividend & Growth",
                row(metric("Dividend Yield", format_percentage(fundamentals['dividend_yield'])),
                    metric("Earnings Growth", format_percentage(fundamentals['earnings_growth'])),
                    metric("Revenue Growth", format_percentage(fundamentals['revenue_growth'])),
                    metric("Beta", format_ratio(fundamentals['beta'])))),

        section("🏥 Financial Health",
                row(metric("Current Ratio", format_ratio(fundamentals['current_ratio'])),
                    metric("Quick Ratio", format_ratio(fundamentals['quick_ratio'])),
                    metric("Debt/Equity", format_ratio(fundamentals['debt_to_equity'])),
                    metric("Book Value/Share", format_ratio(fundamentals['book_value'])))),
    ])
