"""Market data analysis services.

MDB-Engine Integration:
- Logging: Uses `get_logger(__name__)` from mdb_engine.observability for structured logging
"""
import math
import pandas as pd
import time
from typing import Tuple, Optional, Dict, Any
import vectorbt as vbt
from datetime import datetime, timedelta
from alpaca_trade_api.rest import TimeFrame
from mdb_engine.observability import get_logger
from ..core.config import ALPACA_KEY, ALPACA_SECRET, ALPACA_URL
import alpaca_trade_api as tradeapi

logger = get_logger(__name__)

# Initialize Alpaca API
if not ALPACA_KEY or not ALPACA_SECRET:
    logger.warning("⚠️ Alpaca API credentials not configured! Trading features will not work.")
    logger.warning(f"ALPACA_KEY present: {bool(ALPACA_KEY)}, ALPACA_SECRET present: {bool(ALPACA_SECRET)}")
    logger.warning(f"ALPACA_URL: {ALPACA_URL}")
else:
    logger.info(f"✅ Alpaca API configured - URL: {ALPACA_URL}")

api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version='v2') if ALPACA_KEY and ALPACA_SECRET else None

# Firecrawl integration for historical news search
try:
    from firecrawl import Firecrawl
    from ..core.config import FIRECRAWL_API_KEY
    firecrawl_client = Firecrawl(api_key=FIRECRAWL_API_KEY) if FIRECRAWL_API_KEY else None
except ImportError:
    logger.warning("firecrawl-py not installed - historical news search will be limited")
    firecrawl_client = None
except Exception as e:
    logger.warning(f"Firecrawl initialization failed: {e}")
    firecrawl_client = None

async def get_market_data(symbol: str, days: int = 100, query_template: Optional[str] = None, db=None) -> Tuple[Optional[pd.DataFrame], list, list]:
    """Fetch market data and news for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        days: Number of days of historical data to fetch
        query_template: Optional Firecrawl query template (legacy)
        db: Optional MongoDB database instance for caching
        
    Returns:
        Tuple of (bars DataFrame, headlines list, news_objects list)
        news_objects contains full news data with URLs for explanation feature
    """
    if not api:
        logger.error("❌ Alpaca API not initialized - cannot fetch market data")
        logger.error(f"ALPACA_KEY present: {bool(ALPACA_KEY)}, ALPACA_SECRET present: {bool(ALPACA_SECRET)}")
        logger.error(f"ALPACA_URL: {ALPACA_URL}")
        return None, [], []
    
    try:
        # Try to get more data than requested to ensure we have enough
        # Alpaca API limit is 10000 bars, but we'll request more than needed
        requested_limit = max(days, 250)  # Request at least 250 days for proper analysis
        
        # Try multiple approaches to get data
        bars = None
        
        # Approach 1: Use limit parameter with 'iex' feed (free for paper trading)
        try:
            large_limit = max(requested_limit, 500)
            raw_bars = api.get_bars(symbol, TimeFrame.Day, limit=large_limit, feed='iex')
            bars = raw_bars.df
            if bars is not None and not bars.empty:
                logger.debug(f"Got {len(bars)} bars for {symbol} using limit parameter")
                if len(bars) <= 5:
                    logger.warning(f"Very few rows returned for {symbol}: {len(bars)} bars")
        except Exception as e:
            logger.debug(f"Limit parameter failed for {symbol}, trying date range: {e}")
            bars = None
        
        # Approach 2: If limit didn't work or returned too little, use date range
        if bars is None or bars.empty or (bars is not None and len(bars) < 50):
            try:
                from datetime import timezone
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=requested_limit + 30)
                bars_date = api.get_bars(
                    symbol, 
                    TimeFrame.Day, 
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    feed='iex'
                ).df
                
                if bars_date is not None and not bars_date.empty:
                    if bars is None or bars.empty or len(bars_date) > len(bars):
                        bars = bars_date
                        logger.debug(f"Using date range data for {symbol}: {len(bars)} bars")
            except Exception as e:
                logger.debug(f"Date range failed for {symbol}: {e}")
        
        # Approach 3: Try with a longer historical period if we still don't have enough
        if (bars is None or bars.empty or (bars is not None and len(bars) < 50)) and requested_limit < 1000:
            try:
                from datetime import timezone
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=365)
                bars_year = api.get_bars(
                    symbol,
                    TimeFrame.Day,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    feed='iex'
                ).df
                
                if bars_year is not None and not bars_year.empty:
                    if bars is None or bars.empty or len(bars_year) > len(bars):
                        bars = bars_year
                        logger.debug(f"Using 1-year data for {symbol}: {len(bars)} bars")
            except Exception as e:
                logger.debug(f"1-year range failed for {symbol}: {e}")
        
        if bars is None or bars.empty:
            logger.error(f"❌ No data returned for {symbol} after all attempts")
            return None, [], []
        
        # Critical check: if we only have 1 row, something is very wrong
        if len(bars) == 1:
            logger.error(f"CRITICAL: Only 1 row of data for {symbol}! API or data processing issue.")
        
        # Handle MultiIndex if present
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index()
        
        # Check if index is a DatetimeIndex or has time/date in name - reset it to make it a column
        index_is_datetime = isinstance(bars.index, pd.DatetimeIndex)
        index_has_time_name = bars.index.name and ('time' in str(bars.index.name).lower() or 'date' in str(bars.index.name).lower())
        
        if index_is_datetime or index_has_time_name:
            bars = bars.reset_index()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        bars.columns = [c.lower() for c in bars.columns]
        
        missing_cols = [c for c in required_cols if c not in bars.columns]
        if missing_cols:
            logger.error(f"Missing required columns for {symbol}: {missing_cols}")
            return None, [], []
        
        # Handle timestamp column - it might be in index or as a column
        timestamp_col = None
        for col in ['timestamp', 'time', 'date', 'datetime']:
            if col in bars.columns:
                timestamp_col = col
                break
        
        # Sort by timestamp to ensure chronological order
        before_sort_len = len(bars)
        
        if timestamp_col:
            bars = bars.sort_values(timestamp_col).reset_index(drop=True)
            # Remove duplicates based on timestamp
            bars_before_dedup = len(bars)
            bars = bars.drop_duplicates(subset=[timestamp_col], keep='first')
            if bars_before_dedup != len(bars):
                logger.warning(f"Removed {bars_before_dedup - len(bars)} duplicate rows for {symbol}")
        else:
            bars = bars.sort_index().reset_index(drop=True)
        
        # If we lost a lot of data, something is wrong
        if before_sort_len > 10 and len(bars) < before_sort_len * 0.5:
            logger.warning(f"Lost more than 50% of data during processing for {symbol}: {before_sort_len} -> {len(bars)}")
        
        # Final check - ensure we have enough rows
        if len(bars) == 0:
            logger.error(f"After processing, no data rows remain for {symbol}")
            return None, [], []
        
        logger.debug(f"Fetched {len(bars)} days of data for {symbol}")
        if len(bars) < 14:
            logger.warning(f"Insufficient data for {symbol}: only {len(bars)} days (need 14+)")
        
        news_items = []
        news_objects = []
        
        try:
            news = api.get_news(symbol=symbol, limit=5) if api else None
            if news:
                for n in news:
                    headline_text = f"- {n.headline}"
                    news_items.append(headline_text)
                    news_obj = {
                        "headline": n.headline,
                        "url": getattr(n, 'url', None) or getattr(n, 'source_url', None),
                        "summary": getattr(n, 'summary', None) or getattr(n, 'content', None) or "",
                        "author": getattr(n, 'author', None) or "Unknown",
                        "created_at": getattr(n, 'created_at', None) or getattr(n, 'published_at', None)
                    }
                    news_objects.append(news_obj)
                logger.debug(f"Fetched {len(news_items)} news items from Alpaca for {symbol}")
        except Exception as e:
            logger.warning(f"Alpaca news fetch failed for {symbol}: {e}")
        
        return bars, news_items, news_objects
    except Exception as e:
        logger.error(f"Data fail {symbol}: {e}")
        return None, [], [], {}

def analyze_technicals(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate technical indicators using vectorbt.
    
    Architecture: Uses vectorbt for vectorized computation.
    Indicators are computed on-the-fly, not stored in MongoDB.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dict with technical indicators
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    if len(df) < 14:
        raise ValueError(f"Insufficient data: need at least 14 days, got {len(df)}")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    
    # Calculate indicators using vectorbt (vectorized, fast)
    try:
        # RSI calculation
        rsi_series = vbt.RSI.run(df['close'], window=14).rsi
        
        # SMA-200 calculation (use available data if less than 200)
        sma_period = min(200, len(df))
        sma_200_series = vbt.MA.run(df['close'], window=sma_period).ma
        
        # ATR calculation
        atr_series = vbt.ATR.run(df['high'], df['low'], df['close'], window=14).atr
        
    except Exception as e:
        logger.error(f"vectorbt indicator calculation failed: {e}")
        raise
    
    # Get latest values
    current = df.iloc[-1]
    rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    sma_val = float(sma_200_series.iloc[-1]) if not pd.isna(sma_200_series.iloc[-1]) else float(current['close'])
    atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 1.0
    
    trend = "UP" if current['close'] > sma_val else "DOWN"
    
    techs = {
        "price": round(current['close'], 2),
        "rsi": round(rsi_val, 2),
        "atr": round(atr_val, 2),
        "trend": trend,
        "sma": round(sma_val, 2),
        "sma_200": round(sma_val, 2)  # Alias for compatibility
    }
    
    return techs
