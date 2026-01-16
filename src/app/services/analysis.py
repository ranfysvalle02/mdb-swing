"""Market data analysis services."""
import math
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
from alpaca_trade_api.rest import TimeFrame
from mdb_engine.observability import get_logger
from ..core.config import ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, FIRECRAWL_API_KEY
from .indicators import rsi, sma, atr
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

# Initialize Firecrawl if API key is available
firecrawl_client = None
if FIRECRAWL_API_KEY:
    try:
        from firecrawl import Firecrawl
        firecrawl_client = Firecrawl(api_key=FIRECRAWL_API_KEY)
        logger.info("Firecrawl client initialized")
    except ImportError:
        logger.warning("firecrawl-py not installed. Install with: pip install firecrawl-py")
    except Exception as e:
        logger.warning(f"Failed to initialize Firecrawl: {e}")
else:
    logger.warning("FIRECRAWL_API_KEY not set. Article content scraping will be skipped.")

def scrape_article_content(url: str, max_chars: int = 500) -> Optional[str]:
    """Scrape article content using Firecrawl and return first ~500 characters."""
    if not firecrawl_client or not url:
        return None
    
    try:
        response = firecrawl_client.scrape(
            url,
            formats=["markdown"],
            only_main_content=True,
            timeout=10000  # 10 seconds timeout
        )
        
        # Handle different response structures
        content = None
        if response:
            # Check if response has success attribute and it's False
            if hasattr(response, 'success') and not response.success:
                logger.warning(f"Firecrawl scrape failed for {url}: {getattr(response, 'error', 'Unknown error')}")
                return None
            
            # Try to get markdown content from various possible structures
            if hasattr(response, 'data') and hasattr(response.data, 'markdown'):
                content = response.data.markdown
            elif hasattr(response, 'markdown'):
                content = response.markdown
            elif isinstance(response, dict):
                content = response.get('markdown') or response.get('data', {}).get('markdown')
        
        if content:
            # Get first ~500 characters, try to end at a sentence boundary
            if len(content) > max_chars:
                truncated = content[:max_chars]
                # Try to find the last sentence boundary
                last_period = truncated.rfind('.')
                last_newline = truncated.rfind('\n')
                cutoff = max(last_period, last_newline)
                if cutoff > max_chars * 0.7:  # Only use cutoff if it's not too early
                    content = truncated[:cutoff + 1]
                else:
                    content = truncated + "..."
            return content
    except Exception as e:
        logger.warning(f"Failed to scrape article {url}: {e}")
        return None
    
    return None

def get_market_data(symbol: str, days: int = 100) -> Tuple[Optional[pd.DataFrame], list]:
    """Fetch market data and news for a symbol."""
    if not api:
        logger.error("Alpaca API not initialized - cannot fetch market data")
        return None, []
    
    try:
        # Try to get more data than requested to ensure we have enough
        # Alpaca API limit is 10000 bars, but we'll request more than needed
        requested_limit = max(days, 250)  # Request at least 250 days for proper analysis
        
        # Try multiple approaches to get data
        bars = None
        
        # Approach 1: Use limit parameter with 'iex' feed (free for paper trading)
        # Paper trading accounts don't have SIP access, so we use 'iex' feed
        try:
            # Use same approach as backtest which works
            large_limit = max(requested_limit, 500)  # Request at least 500 days
            logger.info(f"🔍 Requesting {large_limit} days for {symbol} using limit parameter with 'iex' feed")
            # Use 'iex' feed which is free for paper trading (avoids SIP subscription error)
            raw_bars = api.get_bars(symbol, TimeFrame.Day, limit=large_limit, feed='iex')
            bars = raw_bars.df
            logger.info(f"✅ Got {len(bars) if bars is not None else 0} bars using limit={large_limit} for {symbol}")
            logger.info(f"   Raw bars type: {type(raw_bars)}, DataFrame shape: {bars.shape if bars is not None else 'None'}")
            if bars is not None and not bars.empty:
                logger.info(f"   Index type: {type(bars.index)}, Index name: {bars.index.name}")
                logger.info(f"   First few rows index: {list(bars.index[:min(5, len(bars))])}")
                logger.info(f"   Columns: {list(bars.columns)}")
                if len(bars) <= 5:
                    logger.warning(f"   ⚠️ Very few rows returned! Full data:\n{bars}")
        except Exception as e:
            logger.error(f"❌ get_bars with limit failed for {symbol}: {e}", exc_info=True)
            bars = None
        
        # Approach 2: If limit didn't work or returned too little, use date range
        if bars is None or bars.empty or (bars is not None and len(bars) < 50):
            try:
                from datetime import timezone
                # Use UTC for consistency with Alpaca API
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=requested_limit + 30)  # Add buffer
                
                logger.info(f"🔄 Trying date range for {symbol}: {start_date.date()} to {end_date.date()} with 'iex' feed")
                bars_date = api.get_bars(
                    symbol, 
                    TimeFrame.Day, 
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    feed='iex'  # Use 'iex' feed for paper trading
                ).df
                
                logger.info(f"   Date range returned {len(bars_date) if bars_date is not None else 0} bars")
                
                if bars_date is not None and not bars_date.empty:
                    if bars is None or bars.empty or len(bars_date) > len(bars):
                        bars = bars_date
                        logger.info(f"✅ Using date range data: {len(bars)} bars")
                    else:
                        logger.info(f"   Keeping limit data ({len(bars)} bars) over date range ({len(bars_date)} bars)")
            except Exception as e:
                logger.error(f"❌ get_bars with date range failed for {symbol}: {e}", exc_info=True)
        
        # Approach 3: Try with a longer historical period if we still don't have enough
        if (bars is None or bars.empty or (bars is not None and len(bars) < 50)) and requested_limit < 1000:
            try:
                # Try getting 1 year of data
                from datetime import timezone
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=365)
                
                logger.info(f"🔄 Trying 1-year range for {symbol}: {start_date.date()} to {end_date.date()} with 'iex' feed")
                bars_year = api.get_bars(
                    symbol,
                    TimeFrame.Day,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    feed='iex'  # Use 'iex' feed for paper trading
                ).df
                
                logger.info(f"   1-year range returned {len(bars_year) if bars_year is not None else 0} bars")
                
                if bars_year is not None and not bars_year.empty:
                    if bars is None or bars.empty or len(bars_year) > len(bars):
                        bars = bars_year
                        logger.info(f"✅ Using 1-year data: {len(bars)} bars")
                    else:
                        logger.info(f"   Keeping existing data ({len(bars)} bars) over 1-year ({len(bars_year)} bars)")
            except Exception as e:
                logger.error(f"❌ Failed to get 1-year data for {symbol}: {e}", exc_info=True)
        
        if bars is None or bars.empty:
            logger.error(f"❌ No data returned for {symbol} after all attempts")
            return None, []
        
        # Critical check: if we only have 1 row, something is very wrong
        if len(bars) == 1:
            logger.error(f"🚨 CRITICAL: Only 1 row of data for {symbol}! This suggests an API or data processing issue.")
            logger.error(f"   DataFrame info:\n{bars.info()}")
            logger.error(f"   Full DataFrame:\n{bars}")
            # Don't return yet - let's see what happens in processing
        
        # Log the structure we received BEFORE processing
        logger.info(f"📊 Raw data for {symbol}: shape={bars.shape}, columns={list(bars.columns)}, index_type={type(bars.index)}, index_name={bars.index.name}")
        if len(bars) > 0:
            logger.info(f"   Sample index values: {list(bars.index[:3])}")
            logger.info(f"   Sample data:\n{bars.head(3)}")
        
        # Handle MultiIndex if present
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index()
            logger.debug("Reset MultiIndex")
        
        # Check if index is a DatetimeIndex or has time/date in name - reset it to make it a column
        index_is_datetime = isinstance(bars.index, pd.DatetimeIndex)
        index_has_time_name = bars.index.name and ('time' in str(bars.index.name).lower() or 'date' in str(bars.index.name).lower())
        
        if index_is_datetime or index_has_time_name:
            bars = bars.reset_index()
            logger.debug(f"Reset index (datetime={index_is_datetime}, has_time_name={index_has_time_name})")
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        bars.columns = [c.lower() for c in bars.columns]
        
        logger.info(f"📊 After reset_index: shape={bars.shape}, columns={list(bars.columns)}")
        if len(bars) > 0:
            logger.info(f"   Sample after reset:\n{bars.head(3)}")
        
        missing_cols = [c for c in required_cols if c not in bars.columns]
        if missing_cols:
            logger.error(f"Missing required columns for {symbol}: {missing_cols}. Available: {list(bars.columns)}")
            return None, []
        
        # Handle timestamp column - it might be in index or as a column
        timestamp_col = None
        for col in ['timestamp', 'time', 'date', 'datetime']:
            if col in bars.columns:
                timestamp_col = col
                break
        
        # Sort by timestamp to ensure chronological order
        before_sort_len = len(bars)
        logger.info(f"📊 Before sorting: {before_sort_len} rows")
        
        if timestamp_col:
            # Sort by timestamp
            bars = bars.sort_values(timestamp_col).reset_index(drop=True)
            logger.info(f"   After sort by {timestamp_col}: {len(bars)} rows")
            
            # Remove duplicates based on timestamp, but keep the first occurrence
            bars_before_dedup = len(bars)
            bars = bars.drop_duplicates(subset=[timestamp_col], keep='first')
            if bars_before_dedup != len(bars):
                logger.warning(f"⚠️ Removed {bars_before_dedup - len(bars)} duplicate timestamp rows for {symbol}")
                logger.warning(f"   This might indicate a data quality issue!")
        else:
            # If no timestamp column, sort by index
            bars = bars.sort_index().reset_index(drop=True)
            logger.info(f"   Sorted by index: {len(bars)} rows")
        
        logger.info(f"📊 After sorting/dedup: {before_sort_len} -> {len(bars)} rows")
        
        # If we lost a lot of data, something is wrong
        if before_sort_len > 10 and len(bars) < before_sort_len * 0.5:
            logger.error(f"🚨 Lost more than 50% of data during processing! {before_sort_len} -> {len(bars)}")
        
        # Final check - ensure we have enough rows
        if len(bars) == 0:
            logger.error(f"❌ After processing, no data rows remain for {symbol}")
            return None, []
        
        logger.info(f"✅ Final: Fetched {len(bars)} days of data for {symbol}")
        if len(bars) < 14:
            logger.warning(f"Only {len(bars)} days of data for {symbol} - may not be enough for full analysis")
            # If we have some data but not enough, still return it with a warning
            # The route handler will show an appropriate message
        
        news = api.get_news(symbol=symbol, limit=3)
        if not news:
            return bars, ["No recent news."]
        
        # Scrape article content for each news item
        news_items = []
        for n in news:
            headline = n.headline
            # Get article URL if available
            article_url = getattr(n, 'url', None) or getattr(n, 'source_url', None)
            
            if article_url and firecrawl_client:
                content = scrape_article_content(article_url)
                if content:
                    news_items.append(f"- {headline}\n  Content: {content}")
                else:
                    news_items.append(f"- {headline}")
            else:
                news_items.append(f"- {headline}")
        
        return bars, news_items
    except Exception as e:
        logger.error(f"Data fail {symbol}: {e}")
        return None, []

def analyze_technicals(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate technical indicators: RSI, SMA, and ATR."""
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
    
    # Calculate RSI (needs at least 14 periods)
    if len(df) >= 14:
        df['rsi'] = rsi(df['close'], length=14)
    else:
        df['rsi'] = 50.0  # Default neutral RSI
    
    # Calculate SMA 200 (needs at least 200 periods, but use available data if less)
    sma_period = min(200, len(df))
    if len(df) >= sma_period:
        df['sma_200'] = sma(df['close'], length=sma_period)
    else:
        # If not enough data for SMA, use simple average of available data
        df['sma_200'] = df['close'].mean()
    
    # Calculate ATR (needs at least 14 periods)
    if len(df) >= 14:
        df['atr'] = atr(df['high'], df['low'], df['close'], length=14)
    else:
        # Use price range as proxy for ATR if insufficient data
        df['atr'] = (df['high'] - df['low']).mean()
    
    current = df.iloc[-1]
    
    sma_val = current['sma_200'] if not math.isnan(current['sma_200']) else current['close']
    trend = "UP" if current['close'] > sma_val else "DOWN"
    
    return {
        "price": round(current['close'], 2),
        "rsi": round(current['rsi'], 2) if not math.isnan(current['rsi']) else 50,
        "atr": round(current['atr'], 2) if not math.isnan(current['atr']) else 1,
        "trend": trend,
        "sma": round(sma_val, 2)
    }
