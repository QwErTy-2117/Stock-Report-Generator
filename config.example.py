"""Configuration file for PDF Stock Report Generator.

Copy this file to config.py and update with your own settings.
DO NOT commit config.py to version control!
"""

import os
from pathlib import Path

# ==============================================================================
# API CONFIGURATION
# ==============================================================================

# Data Provider Selection
# Options: 'yfinance', 'alpha_vantage', 'iex', 'finnhub'
DATA_SOURCE = 'yfinance'

# Alpha Vantage API Key (required if using alpha_vantage)
# Get your free API key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY_HERE'

# IEX Cloud API Key (required if using iex)
# Get your API key at: https://iexcloud.io/
IEX_API_KEY = 'YOUR_IEX_API_KEY_HERE'

# Finnhub API Key (required if using finnhub)
# Get your free API key at: https://finnhub.io/
FINNHUB_API_KEY = 'YOUR_FINNHUB_API_KEY_HERE'

# ==============================================================================
# DATA FETCHING CONFIGURATION
# ==============================================================================

# Historical Data Period
# Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'
DATA_PERIOD = '1y'

# Data Interval
# Options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
DATA_INTERVAL = '1d'

# Maximum retries for API requests
MAX_RETRIES = 3

# Request timeout in seconds
REQUEST_TIMEOUT = 30

# ==============================================================================
# TECHNICAL INDICATORS CONFIGURATION
# ==============================================================================

# Enable/disable specific technical indicators
ENABLE_INDICATORS = True

# Technical indicators to include in reports
# Options: 'SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'VWAP', 'OBV'
INDICATORS = [
    'SMA',   # Simple Moving Average
    'EMA',   # Exponential Moving Average
    'RSI',   # Relative Strength Index
    'MACD',  # Moving Average Convergence Divergence
    'BB',    # Bollinger Bands
]

# Moving Average periods
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands configuration
BB_PERIOD = 20
BB_STD_DEV = 2

# ==============================================================================
# CHART CONFIGURATION
# ==============================================================================

# Chart style
# Options: 'default', 'seaborn', 'ggplot', 'bmh', 'classic', 'dark_background'
CHART_STYLE = 'seaborn'

# Figure size (width, height) in inches
FIGURE_SIZE = (12, 8)

# DPI (resolution)
DPI = 300

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'volume': '#9467bd',
    'grid': '#cccccc',
}

# Chart types to include
CHART_TYPES = [
    'price',      # Price chart with candlesticks/line
    'volume',     # Volume bars
    'indicators', # Technical indicators
    'comparison', # Comparative analysis (if multiple stocks)
]

# ==============================================================================
# PDF CONFIGURATION
# ==============================================================================

# PDF page size
# Options: 'A4', 'Letter', 'Legal'
PAGE_SIZE = 'A4'

# Page orientation
# Options: 'portrait', 'landscape'
ORIENTATION = 'portrait'

# Margins (in inches)
MARGINS = {
    'top': 1.0,
    'bottom': 1.0,
    'left': 1.0,
    'right': 1.0,
}

# Font settings
FONT_FAMILY = 'Helvetica'
FONT_SIZE = {
    'title': 24,
    'heading': 18,
    'subheading': 14,
    'body': 12,
    'caption': 10,
}

# Report sections to include
REPORT_SECTIONS = [
    'cover',          # Cover page with logo and title
    'summary',        # Executive summary
    'price_chart',    # Price history chart
    'volume_chart',   # Volume analysis
    'indicators',     # Technical indicators
    'statistics',     # Statistical analysis
    'recommendation', # Optional: Trading recommendation
]

# Include company information
INCLUDE_COMPANY_INFO = True

# Include market data (market cap, P/E ratio, etc.)
INCLUDE_MARKET_DATA = True

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

# Output directory for generated reports
OUTPUT_DIR = Path('reports')

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Report filename format
# Available variables: {symbol}, {date}, {time}
FILENAME_FORMAT = '{symbol}_report_{date}.pdf'

# Date format for filenames
DATE_FORMAT = '%Y%m%d'

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

# Enable logging
ENABLE_LOGGING = True

# Log level
# Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL = 'INFO'

# Log file path
LOG_FILE = 'stock_report_generator.log'

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==============================================================================
# BATCH PROCESSING CONFIGURATION
# ==============================================================================

# Enable batch processing
BATCH_PROCESSING = True

# Number of concurrent workers for batch processing
MAX_WORKERS = 4

# Delay between requests (seconds) to avoid rate limiting
REQUEST_DELAY = 0.5

# ==============================================================================
# ADVANCED OPTIONS
# ==============================================================================

# Cache data to reduce API calls
ENABLE_CACHE = True

# Cache expiration time (seconds)
CACHE_EXPIRATION = 3600  # 1 hour

# Cache directory
CACHE_DIR = Path('.cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Enable progress bars
SHOW_PROGRESS = True

# Validate stock symbols before processing
VALIDATE_SYMBOLS = True

# ==============================================================================
# CUSTOM BRANDING (Optional)
# ==============================================================================

# Company/User name to display on reports
BRAND_NAME = 'Stock Report Generator'

# Logo file path (optional)
LOGO_PATH = None  # Set to Path('path/to/logo.png') if you have a logo

# Footer text
FOOTER_TEXT = 'Generated by PDF Stock Report Generator'

# Disclaimer text
DISCLAIMER = (
    'This report is for informational purposes only and does not constitute '
    'financial advice. Always conduct your own research and consult with a '
    'qualified financial advisor before making investment decisions.'
)

# ==============================================================================
# ENVIRONMENT VARIABLES (Alternative to hardcoding API keys)
# ==============================================================================

# You can also set API keys via environment variables:
# export ALPHA_VANTAGE_API_KEY="your_key_here"
# export IEX_API_KEY="your_key_here"
# export FINNHUB_API_KEY="your_key_here"

if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == 'YOUR_ALPHA_VANTAGE_API_KEY_HERE':
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')

if not IEX_API_KEY or IEX_API_KEY == 'YOUR_IEX_API_KEY_HERE':
    IEX_API_KEY = os.getenv('IEX_API_KEY', '')

if not FINNHUB_API_KEY or FINNHUB_API_KEY == 'YOUR_FINNHUB_API_KEY_HERE':
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
