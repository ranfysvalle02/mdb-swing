"""Market data analysis services."""
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import vectorbt as vbt
from datetime import datetime, timedelta
from alpaca_trade_api.rest import TimeFrame
from mdb_engine.observability import get_logger
from ..core.config import ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, FIRECRAWL_API_KEY
import alpaca_trade_api as tradeapi
import asyncio

logger = get_logger(__name__)

if not ALPACA_KEY or not ALPACA_SECRET:
    logger.warning("Alpaca API credentials not configured! Trading features will not work.")
    logger.warning(f"ALPACA_KEY present: {bool(ALPACA_KEY)}, ALPACA_SECRET present: {bool(ALPACA_SECRET)}")
    logger.warning(f"ALPACA_URL: {ALPACA_URL}")
else:
    logger.info(f"Alpaca API configured - URL: {ALPACA_URL}")

api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version='v2') if ALPACA_KEY and ALPACA_SECRET else None

FIRECRAWL_API_URL = "https://api.firecrawl.dev/v2"

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available - Firecrawl API calls will fail")

async def search_news_headlines(symbol: str, limit: int = 5) -> list:
    """Fetch news headlines using yfinance (free, no rate limits).
    
    This provides a preview of what news articles the AI will analyze.
    Fast and efficient for watchlist cards.
    
    Uses yfinance to get news directly from Yahoo Finance - no API rate limits!
    
    Args:
        symbol: Stock ticker symbol
        limit: Maximum number of headlines to return (default 5)
        
    Returns:
        List of dicts with 'text' and 'url' keys, or empty list if fetch fails
    """
    logger.info(f"[NEWS] Fetching news headlines for {symbol} using yfinance")
    
    try:
        import yfinance as yf
        logger.info(f"[NEWS] yfinance imported successfully for {symbol}")
    except ImportError as import_err:
        logger.error(f"[NEWS] yfinance not installed - cannot fetch news for {symbol}: {import_err}")
        return []
    
    try:
        # Create the Ticker object
        logger.info(f"[NEWS] Creating yfinance Ticker for {symbol}")
        ticker = yf.Ticker(symbol)
        
        # Get the raw news list (returns a list of dictionaries)
        # Run in executor to avoid blocking the event loop (yfinance is synchronous)
        import asyncio
        
        def fetch_news():
            """Helper function to fetch news synchronously."""
            # Note: Logger calls inside executor might not show up immediately
            # But errors will be caught and logged after executor returns
            try:
                news_data = ticker.news
                return news_data
            except Exception as e:
                # Return error info so we can log it outside executor
                return {"_error": str(e), "_exception": e}
        
        loop = asyncio.get_event_loop()
        logger.info(f"[NEWS] Running fetch_news in executor for {symbol}")
        raw_news = await loop.run_in_executor(None, fetch_news)
        
        # Check if executor returned an error
        if isinstance(raw_news, dict) and "_error" in raw_news:
            error_msg = raw_news.get("_error", "Unknown error")
            logger.error(f"[NEWS] yfinance error in executor for {symbol}: {error_msg}", exc_info=True)
            return []
        
        logger.info(f"[NEWS] Raw news result for {symbol}: type={type(raw_news)}, length={len(raw_news) if raw_news else 0}")
        
        if not raw_news:
            logger.warning(f"[NEWS] No news found for {symbol} - raw_news is empty or None")
            return []
        
        if not isinstance(raw_news, list):
            logger.error(f"[NEWS] Unexpected news format for {symbol}: expected list, got {type(raw_news)}")
            logger.error(f"[NEWS] Raw value: {raw_news}")
            return []
        
        logger.info(f"[NEWS] Found {len(raw_news)} news items for {symbol}")
        
        # Extract headlines from news results with summary
        # yfinance news structure: article['content']['title'], article['content']['summary'], article['content']['canonicalUrl']['url']
        headlines = []
        for idx, article in enumerate(raw_news[:limit]):
            try:
                if not isinstance(article, dict):
                    logger.warning(f"[NEWS] Article {idx+1} for {symbol} is not a dict: {type(article)}")
                    continue
                
                # Extract data from nested structure: content.title, content.summary, content.canonicalUrl.url
                content = article.get('content', {})
                if not isinstance(content, dict):
                    # Fallback: try top-level
                    title = article.get('title', '').strip()
                    summary = article.get('summary', '').strip()
                    link = article.get('link')
                else:
                    title = content.get('title', '').strip()
                    summary = content.get('summary', '').strip()
                    
                    # Extract URL from nested structure: prefer canonicalUrl, fallback to clickThroughUrl
                    link = None
                    canonical_url = content.get('canonicalUrl', {})
                    if isinstance(canonical_url, dict):
                        link = canonical_url.get('url')
                    
                    # Fallback to clickThroughUrl if canonicalUrl not available
                    if not link:
                        click_through = content.get('clickThroughUrl', {})
                        if isinstance(click_through, dict):
                            link = click_through.get('url')
                
                # Final fallback: try top-level link
                if not link:
                    link = article.get('link')
                
                # Use summary if available, otherwise use title
                # Summary is often more informative for quick scanning
                display_text = summary if summary else title
                
                
                if display_text:
                    headlines.append({
                        'text': display_text,
                        'title': title,  # Keep original title for tooltip/hover
                        'summary': summary if summary else None,  # Keep summary if available
                        'url': link if link else None
                    })
                else:
                    logger.warning(f"[NEWS] Article {idx+1} for {symbol} has no title or summary. Content keys: {list(content.keys()) if isinstance(content, dict) else 'N/A'}")
            except Exception as e:
                logger.error(f"[NEWS] Error processing article {idx+1} for {symbol}: {e}", exc_info=True)
                continue
        
        logger.info(f"[NEWS] Successfully extracted {len(headlines)} headlines for {symbol} via yfinance")
        return headlines
        
    except Exception as e:
        logger.error(f"[NEWS] Failed to fetch news for {symbol} using yfinance: {e}", exc_info=True)
        import traceback
        logger.error(f"[NEWS] Full traceback: {traceback.format_exc()}")
        return []


async def fetch_article_content(url: str, max_chars: int = 500) -> Optional[str]:
    """Fetch article content using Firecrawl scrape API v2 via direct HTTP calls.
    
    Features:
    - Uses Firecrawl scrape API v2 to extract markdown content from articles
    - Makes direct HTTP POST requests to https://api.firecrawl.dev/v2/scrape
    - Comprehensive error handling and logging
    - Returns first max_chars characters of article content
    
    Args:
        url: Article URL to scrape
        max_chars: Maximum characters to return (first N characters)
        
    Returns:
        Article content as string (first max_chars), or None if fetch fails
    """
    if not FIRECRAWL_API_KEY or not url:
        return None
    
    if not HTTPX_AVAILABLE:
        logger.error(f"httpx not available - cannot scrape article for {url[:80]}")
        return None
    
    try:
        
        logger.debug(f"Fetching article content from {url[:80]}...")
        
        # Prepare request payload according to v2 API spec
        payload = {
            "url": url,
            "formats": [{"type": "markdown"}]  # v2 format: array of objects
        }
        
        # Make direct HTTP POST request to Firecrawl v2 API
        headers = {
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{FIRECRAWL_API_URL}/scrape",
                json=payload,
                headers=headers
            )
            
            # Check for errors
            if response.status_code != 200:
                logger.warning(f"Firecrawl scrape API error for {url[:80]}: {response.status_code} - {response.text[:200]}")
                return None
            
            # Parse JSON response
            result = response.json()
            
            # Check if request was successful
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                logger.warning(f"Firecrawl scrape failed for {url[:80]}: {error_msg}")
                return None
            
            # Extract markdown content from response
            # v2 format: { "success": true, "data": { "markdown": "..." } }
            data = result.get("data", {})
            content = data.get("markdown") or data.get("content")
            
            if content:
                content = content.strip()
                original_len = len(content)
                if len(content) > max_chars:
                    content = content[:max_chars] + "..."
                logger.debug(f"Successfully fetched {original_len} chars from article (returning {len(content)} chars)")
                return content
            else:
                logger.warning(f"No content found in Firecrawl response for {url[:80]}")
                return None
        
    except httpx.TimeoutException:
        logger.warning(f"Firecrawl API timeout for {url[:80]}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Firecrawl API request error for {url[:80]}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch article content from {url[:80]}: {e}")
        import traceback
        logger.debug(f"Firecrawl error traceback: {traceback.format_exc()}")
        return None


async def fetch_cnn_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch CNN market data for a symbol using Firecrawl.
    
    Scrapes CNN Markets page (https://www.cnn.com/markets/stocks/{SYMBOL}) to extract:
    - Analyst ratings (buy/hold/sell percentages)
    - Price targets (high, median, low)
    - Financial metrics (revenue, net income, EPS)
    - Current price and market cap
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with market data or None if fetch fails
    """
    if not FIRECRAWL_API_KEY:
        logger.debug("Firecrawl API key not configured - skipping CNN data")
        return None
    
    if not HTTPX_AVAILABLE:
        logger.debug("httpx not available - skipping CNN data")
        return None
    
    cnn_url = f"https://www.cnn.com/markets/stocks/{symbol.upper()}"
    
    try:
        logger.info(f"[CNN] Fetching market data for {symbol} from CNN...")
        
        # Use Firecrawl to scrape CNN page
        payload = {
            "url": cnn_url,
            "formats": [{"type": "markdown"}]
        }
        
        headers = {
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{FIRECRAWL_API_URL}/scrape",
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.debug(f"CNN scrape failed for {symbol}: {response.status_code}")
                return None
            
            result = response.json()
            if not result.get("success", False):
                logger.debug(f"CNN scrape failed for {symbol}: {result.get('error', 'Unknown')}")
                return None
            
            # Extract markdown content
            data = result.get("data", {})
            content = data.get("markdown") or data.get("content")
            
            if not content:
                logger.debug(f"No content found in CNN response for {symbol}")
                return None
            
            # Parse CNN data from markdown content
            # Look for analyst ratings, price targets, financials
            market_data = {
                "analyst_ratings": {},
                "price_targets": {},
                "financials": {},
                "current_price": None,
                "market_cap": None
            }
            
            # Extract analyst ratings (e.g., "buy 67%, hold 27%, sell 6%")
            import re
            rating_pattern = r'(buy|hold|sell)\s*(\d+)%'
            ratings = re.findall(rating_pattern, content.lower())
            if ratings:
                for rating_type, pct in ratings:
                    market_data["analyst_ratings"][rating_type] = int(pct)
            
            # Extract price targets (e.g., "High $151.00", "Median $115.00", "Low $80.00")
            target_patterns = [
                (r'high\s*\$?([\d,]+\.?\d*)', 'high'),
                (r'median\s*\$?([\d,]+\.?\d*)', 'median'),
                (r'low\s*\$?([\d,]+\.?\d*)', 'low')
            ]
            for pattern, key in target_patterns:
                match = re.search(pattern, content.lower())
                if match:
                    value_str = match.group(1).replace(',', '')
                    try:
                        market_data["price_targets"][key] = float(value_str)
                    except ValueError:
                        pass
            
            # Extract current price
            price_match = re.search(r'current\s*price[:\s]*\$?([\d,]+\.?\d*)', content.lower())
            if not price_match:
                price_match = re.search(r'\$([\d,]+\.?\d*)\s*\(current\)', content.lower())
            if price_match:
                try:
                    market_data["current_price"] = float(price_match.group(1).replace(',', ''))
                except ValueError:
                    pass
            
            # Extract market cap
            cap_match = re.search(r'market\s*cap[:\s]*\$?([\d.]+)\s*([BMK])', content.lower())
            if cap_match:
                value = float(cap_match.group(1))
                unit = cap_match.group(2).upper()
                multiplier = {'B': 1_000_000_000, 'M': 1_000_000, 'K': 1_000}.get(unit, 1)
                market_data["market_cap"] = value * multiplier
            
            # Extract financial metrics (revenue, net income, EPS)
            revenue_match = re.search(r'total\s*revenue[:\s]*\$?([\d.]+)\s*([BMK])', content.lower())
            if revenue_match:
                value = float(revenue_match.group(1))
                unit = revenue_match.group(2).upper()
                multiplier = {'B': 1_000_000_000, 'M': 1_000_000, 'K': 1_000}.get(unit, 1)
                market_data["financials"]["revenue"] = value * multiplier
            
            net_income_match = re.search(r'net\s*income[:\s]*\$?([\d.]+)\s*([BMK])', content.lower())
            if net_income_match:
                value = float(net_income_match.group(1))
                unit = net_income_match.group(2).upper()
                multiplier = {'B': 1_000_000_000, 'M': 1_000_000, 'K': 1_000}.get(unit, 1)
                market_data["financials"]["net_income"] = value * multiplier
            
            eps_match = re.search(r'earnings\s*per\s*share[:\s]*\$?([\d.]+)', content.lower())
            if eps_match:
                try:
                    market_data["financials"]["eps"] = float(eps_match.group(1))
                except ValueError:
                    pass
            
            # Only return if we got at least some useful data
            if market_data["analyst_ratings"] or market_data["price_targets"] or market_data.get("current_price"):
                logger.info(f"[CNN] Successfully fetched market data for {symbol}")
                return market_data
            else:
                logger.debug(f"[CNN] No useful data extracted for {symbol}")
                return None
        
    except httpx.TimeoutException:
        logger.debug(f"CNN API timeout for {symbol}")
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch CNN data for {symbol}: {e}")
        return None

async def get_market_data(symbol: str, days: int = 100, db=None, fetch_firecrawl_content: bool = False) -> Tuple[Optional[pd.DataFrame], list, list]:
    """Fetch market data and news for a symbol.
    
    Uses caching to reduce API calls. Market data is cached for 5 minutes.
    
    Args:
        symbol: Stock ticker symbol
        days: Number of days of historical data to fetch
        db: Optional MongoDB database instance for caching
        fetch_firecrawl_content: If True, fetch full article content via Firecrawl (slow).
                                 If False, only fetch headlines/summaries (fast).
                                 Default False for page load performance.
        
    Returns:
        Tuple of (bars DataFrame, headlines list, news_objects list)
        news_objects contains full news data with URLs for explanation feature
    """
    try:
        from .watchlist_cache import get_cache
        cache = get_cache()
        cached_data = cache.get_market_data(symbol, ttl_seconds=300)  # 5-minute TTL
        if cached_data:
            # If article content is requested but cached data doesn't have it, skip cache
            if fetch_firecrawl_content:
                bars_cached, headlines_cached, news_objects_cached = cached_data
                has_article_content = any(
                    news_obj.get('article_content') for news_obj in (news_objects_cached or [])
                )
                if not has_article_content:
                    logger.debug(f"Cache exists for {symbol} but missing article content, fetching fresh...")
                    cached_data = None  # Skip cache, fetch fresh with article content
                else:
                    logger.debug(f"Using cached market data for {symbol} (with article content)")
                    return cached_data
            else:
                logger.debug(f"Using cached market data for {symbol}")
                return cached_data
    except Exception as e:
        logger.debug(f"Cache check failed for {symbol}: {e}")
    
    if not api:
        logger.error("Alpaca API not initialized - cannot fetch market data")
        logger.error(f"ALPACA_KEY present: {bool(ALPACA_KEY)}, ALPACA_SECRET present: {bool(ALPACA_SECRET)}")
        logger.error(f"ALPACA_URL: {ALPACA_URL}")
        return None, [], []
    
    try:
        requested_limit = max(days, 250)
        bars = None
        
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
            logger.error(f"No data returned for {symbol} after all attempts")
            return None, [], []
        
        if len(bars) == 1:
            logger.error(f"CRITICAL: Only 1 row of data for {symbol}! API or data processing issue.")
        
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index()
        
        index_is_datetime = isinstance(bars.index, pd.DatetimeIndex)
        index_has_time_name = bars.index.name and ('time' in str(bars.index.name).lower() or 'date' in str(bars.index.name).lower())
        
        if index_is_datetime or index_has_time_name:
            bars = bars.reset_index()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        bars.columns = [c.lower() for c in bars.columns]
        
        missing_cols = [c for c in required_cols if c not in bars.columns]
        if missing_cols:
            logger.error(f"Missing required columns for {symbol}: {missing_cols}")
            return None, [], []
        
        timestamp_col = None
        for col in ['timestamp', 'time', 'date', 'datetime']:
            if col in bars.columns:
                timestamp_col = col
                break
        
        before_sort_len = len(bars)
        
        if timestamp_col:
            bars = bars.sort_values(timestamp_col).reset_index(drop=True)
            bars_before_dedup = len(bars)
            bars = bars.drop_duplicates(subset=[timestamp_col], keep='first')
            if bars_before_dedup != len(bars):
                logger.warning(f"Removed {bars_before_dedup - len(bars)} duplicate rows for {symbol}")
        else:
            bars = bars.sort_index().reset_index(drop=True)
        
        if before_sort_len > 10 and len(bars) < before_sort_len * 0.5:
            logger.warning(f"Lost more than 50% of data during processing for {symbol}: {before_sort_len} -> {len(bars)}")
        
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
                news_list = []
                url_to_index = {}
                fetch_tasks = []
                
                for idx, n in enumerate(news):
                    url = getattr(n, 'url', None) or getattr(n, 'source_url', None)
                    news_obj = {
                        "headline": n.headline,
                        "url": url,
                        "summary": getattr(n, 'summary', None) or getattr(n, 'content', None) or "",
                        "author": getattr(n, 'author', None) or "Unknown",
                        "created_at": getattr(n, 'created_at', None) or getattr(n, 'published_at', None),
                        "article_content": None
                    }
                    news_list.append(news_obj)
                    
                    if url and fetch_firecrawl_content:
                        # Only fetch article content if explicitly requested (for AI analysis)
                        # Skip on page load for performance
                        url_to_index[url] = idx
                        fetch_tasks.append((url, fetch_article_content(url, max_chars=500)))
                    elif url:
                        # Store URL but don't fetch content (fast path for page load)
                        url_to_index[url] = idx
                        news_list[idx]["article_content"] = None
                
                if fetch_tasks:
                    logger.info(f"Fetching article content for {len(fetch_tasks)} news articles...")
                    tasks_only = [task for _, task in fetch_tasks]
                    urls_only = [url for url, _ in fetch_tasks]
                    
                    content_results = await asyncio.gather(*tasks_only, return_exceptions=True)
                    
                    for url, content in zip(urls_only, content_results):
                        idx = url_to_index[url]
                        try:
                            if isinstance(content, Exception):
                                logger.warning(f"Error fetching content for article {idx}: {content}")
                                content = None
                            news_list[idx]["article_content"] = content
                        except Exception as e:
                            logger.warning(f"Error processing content result for article {idx}: {e}")
                            news_list[idx]["article_content"] = None
                
                for news_obj in news_list:
                    if news_obj["article_content"]:
                        content_text = f"Headline: {news_obj['headline']}\nContent: {news_obj['article_content']}"
                        news_items.append(content_text)
                    else:
                        if news_obj["summary"]:
                            news_items.append(f"Headline: {news_obj['headline']}\nSummary: {news_obj['summary']}")
                        else:
                            news_items.append(f"- {news_obj['headline']}")
                    
                    news_objects.append(news_obj)
                
                logger.info(f"Fetched {len(news_items)} news items for {symbol} ({len([n for n in news_objects if n.get('article_content')])} with article content)")
        except Exception as e:
            logger.warning(f"Alpaca news fetch failed for {symbol}: {e}")
        
        try:
            from .watchlist_cache import get_cache
            cache = get_cache()
            cache.set_market_data(symbol, (bars, news_items, news_objects))
        except Exception as e:
            logger.debug(f"Cache store failed for {symbol}: {e}")
        
        return bars, news_items, news_objects
    except Exception as e:
        logger.error(f"Data fail {symbol}: {e}")
        return None, [], []

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
    
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    required_cols = ['open', 'high', 'low', 'close']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    
    try:
        rsi_series = vbt.RSI.run(df['close'], window=14).rsi
        sma_period = min(200, len(df))
        sma_200_series = vbt.MA.run(df['close'], window=sma_period).ma
        atr_series = vbt.ATR.run(df['high'], df['low'], df['close'], window=14).atr
    except Exception as e:
        logger.error(f"vectorbt indicator calculation failed: {e}")
        raise
    
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
        "sma_200": round(sma_val, 2)
    }
    
    return techs


def analyze_comprehensive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive technical indicators using vectorbt.
    
    Extended version of analyze_technicals with additional indicators:
    - MACD, Bollinger Bands, Stochastic, Multiple MAs, Volume metrics
    
    Architecture: Uses vectorbt for vectorized computation.
    Indicators are computed on-the-fly, not stored in MongoDB.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dict with comprehensive technical indicators
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    if len(df) < 20:
        raise ValueError(f"Insufficient data: need at least 20 days, got {len(df)}")
    
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    
    stats = {}
    current = df.iloc[-1]
    
    try:
        stats['price'] = round(float(current['close']), 2)
        stats['open'] = round(float(current['open']), 2)
        stats['high'] = round(float(current['high']), 2)
        stats['low'] = round(float(current['low']), 2)
        stats['volume'] = int(current['volume']) if 'volume' in df.columns else 0
        
        if len(df) >= 2:
            prev_close = float(df.iloc[-2]['close'])
            stats['change_pct'] = round(((stats['price'] - prev_close) / prev_close) * 100, 2) if prev_close > 0 else 0
        
        rsi_series = vbt.RSI.run(df['close'], window=14).rsi
        stats['rsi'] = round(float(rsi_series.iloc[-1]), 2) if not pd.isna(rsi_series.iloc[-1]) else None
        
        periods = [20, 50, 100, 200]
        for period in periods:
            if len(df) >= period:
                try:
                    sma = vbt.MA.run(df['close'], window=period).ma
                    sma_val = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None
                    if sma_val:
                        stats[f'sma_{period}'] = round(sma_val, 2)
                        stats[f'price_vs_sma_{period}_pct'] = round(((stats['price'] - sma_val) / sma_val) * 100, 2)
                except Exception as e:
                    logger.debug(f"Failed to calculate SMA-{period}: {e}")
        
        ema_12_series = None
        ema_26_series = None
        try:
            if len(df) >= 26:
                ema_12_series = vbt.MA.run(df['close'], window=12, ewm=True).ma
                ema_26_series = vbt.MA.run(df['close'], window=26, ewm=True).ma
                stats['ema_12'] = round(float(ema_12_series.iloc[-1]), 2) if not pd.isna(ema_12_series.iloc[-1]) else None
                stats['ema_26'] = round(float(ema_26_series.iloc[-1]), 2) if not pd.isna(ema_26_series.iloc[-1]) else None
        except Exception as e:
            logger.debug(f"Failed to calculate EMA: {e}")
        
        if stats.get('ema_12') and stats.get('ema_26') and ema_12_series is not None and ema_26_series is not None:
            try:
                macd_line = stats['ema_12'] - stats['ema_26']
                macd_series = ema_12_series - ema_26_series
                if len(macd_series) >= 9:
                    signal_series = vbt.MA.run(macd_series, window=9, ewm=True).ma
                    signal_line = float(signal_series.iloc[-1]) if not pd.isna(signal_series.iloc[-1]) else None
                    stats['macd'] = round(macd_line, 2)
                    stats['macd_signal'] = round(signal_line, 2) if signal_line else None
                    stats['macd_histogram'] = round(macd_line - (signal_line or 0), 2) if signal_line else None
            except Exception as e:
                logger.debug(f"Failed to calculate MACD: {e}")
        
        if len(df) >= 20:
            try:
                bb = vbt.BBANDS.run(df['close'], window=20)
                stats['bb_upper'] = round(float(bb.upper.iloc[-1]), 2) if not pd.isna(bb.upper.iloc[-1]) else None
                stats['bb_middle'] = round(float(bb.middle.iloc[-1]), 2) if not pd.isna(bb.middle.iloc[-1]) else None
                stats['bb_lower'] = round(float(bb.lower.iloc[-1]), 2) if not pd.isna(bb.lower.iloc[-1]) else None
                if stats.get('bb_upper') and stats.get('bb_lower') and stats.get('bb_middle'):
                    bb_width = ((stats['bb_upper'] - stats['bb_lower']) / stats['bb_middle']) * 100
                    stats['bb_width_pct'] = round(bb_width, 2)
                    if stats['bb_upper'] != stats['bb_lower']:
                        stats['bb_percent_b'] = round(((stats['price'] - stats['bb_lower']) / (stats['bb_upper'] - stats['bb_lower'])) * 100, 2)
            except Exception as e:
                logger.debug(f"Failed to calculate Bollinger Bands: {e}")
        
        if len(df) >= 14:
            try:
                stoch = vbt.STOCH.run(df['high'], df['low'], df['close'], k_window=14)
                stats['stoch_k'] = round(float(stoch.percent_k.iloc[-1]), 2) if not pd.isna(stoch.percent_k.iloc[-1]) else None
                stats['stoch_d'] = round(float(stoch.percent_d.iloc[-1]), 2) if not pd.isna(stoch.percent_d.iloc[-1]) else None
            except Exception as e:
                logger.debug(f"Failed to calculate Stochastic: {e}")
        
        try:
            atr_series = vbt.ATR.run(df['high'], df['low'], df['close'], window=14).atr
            stats['atr'] = round(float(atr_series.iloc[-1]), 2) if not pd.isna(atr_series.iloc[-1]) else None
            if stats.get('atr') and stats.get('price'):
                stats['atr_pct'] = round((stats['atr'] / stats['price']) * 100, 2)
        except Exception as e:
            logger.debug(f"Failed to calculate ATR: {e}")
        
        if 'volume' in df.columns and len(df) >= 20:
            try:
                vol_sma = vbt.MA.run(df['volume'], window=20).ma
                avg_volume = float(vol_sma.iloc[-1]) if not pd.isna(vol_sma.iloc[-1]) else None
                if avg_volume and avg_volume > 0:
                    stats['volume_ratio'] = round(stats['volume'] / avg_volume, 2)
                    stats['avg_volume'] = int(avg_volume)
            except Exception as e:
                logger.debug(f"Failed to calculate volume metrics: {e}")
        
        if len(df) >= 20:
            try:
                recent_high = float(df['high'].tail(20).max())
                recent_low = float(df['low'].tail(20).min())
                stats['recent_high'] = round(recent_high, 2)
                stats['recent_low'] = round(recent_low, 2)
                if recent_high != recent_low:
                    stats['price_position_in_range'] = round(((stats['price'] - recent_low) / (recent_high - recent_low)) * 100, 2)
            except Exception as e:
                logger.debug(f"Failed to calculate price range: {e}")
        
        sma_200 = stats.get('sma_200') or stats.get('sma_100') or stats.get('sma_50')
        stats['trend'] = "UP" if sma_200 and stats['price'] > sma_200 else "DOWN"
        
    except Exception as e:
        logger.error(f"Comprehensive stats calculation failed: {e}", exc_info=True)
        raise
    
    return stats
