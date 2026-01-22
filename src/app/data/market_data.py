"""Market Data Ingestion - High-Performance Cache."""
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import pickle
try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
from mdb_engine.observability import get_logger

logger = get_logger(__name__)

# Cache directory for market data (not MongoDB)
CACHE_DIR = Path("/tmp/mdb_swing_cache") if Path("/tmp").exists() else Path("./cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)


async def load_daily_bars(
    symbols: List[str],
    days: int = 250,
    cache_ttl_minutes: int = 15
) -> Dict[str, pd.DataFrame]:
    """Load daily OHLCV bars for symbols.
    
    Architecture: Loads from cache/API, NOT MongoDB.
    OHLCV data is transient computation fuel.
    
    Args:
        symbols: List of stock symbols
        days: Number of days of history to load
        cache_ttl_minutes: Cache TTL in minutes
    
    Returns:
        Dict mapping symbol to DataFrame with columns: open, high, low, close, volume
    """
    from ..services.analysis import api, get_market_data
    
    results = {}
    
    for symbol in symbols:
        # Try cache first (Parquet preferred, fallback to pickle)
        cache_parquet = CACHE_DIR / f"{symbol}_{days}d.parquet"
        cache_pickle = CACHE_DIR / f"{symbol}_{days}d.pkl"
        
        cache_file = cache_parquet if PARQUET_AVAILABLE and cache_parquet.exists() else (cache_pickle if cache_pickle.exists() else None)
        
        if cache_file and cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(minutes=cache_ttl_minutes):
                try:
                    if cache_file.suffix == '.parquet' and PARQUET_AVAILABLE:
                        df = pd.read_parquet(cache_file)
                    else:
                        with open(cache_file, 'rb') as f:
                            df = pickle.load(f)
                    logger.debug(f"Loaded {symbol} from cache ({len(df)} bars)")
                    results[symbol] = df
                    continue
                except Exception as e:
                    logger.warning(f"Cache read failed for {symbol}: {e}")
        
        # Load from API
        try:
            bars, _, _, _ = await get_market_data(symbol, days=days)
            if bars is not None and not bars.empty:
                # Cache the result (Parquet preferred for performance)
                try:
                    if PARQUET_AVAILABLE:
                        cache_parquet = CACHE_DIR / f"{symbol}_{days}d.parquet"
                        bars.to_parquet(cache_parquet, compression='snappy')
                        # Also remove old pickle if exists
                        cache_pickle = CACHE_DIR / f"{symbol}_{days}d.pkl"
                        if cache_pickle.exists():
                            cache_pickle.unlink()
                    else:
                        cache_pickle = CACHE_DIR / f"{symbol}_{days}d.pkl"
                        with open(cache_pickle, 'wb') as f:
                            pickle.dump(bars, f)
                except Exception as e:
                    logger.debug(f"Cache write failed for {symbol}: {e}")
                
                results[symbol] = bars
            else:
                logger.warning(f"No data returned for {symbol}")
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
    
    return results


def load_ai_scores(
    symbols: List[str],
    date: Optional[str] = None
) -> Dict[str, Optional[float]]:
    """Load AI scores for symbols.
    
    Architecture: AI scores are stored in MongoDB (signals collection),
    but we load them here for computation.
    
    Args:
        symbols: List of stock symbols
    
    Returns:
        Dict mapping symbol to AI score (or None if not available)
    """
    # This would query MongoDB signals collection for latest AI scores
    # For now, return None - will be populated by analysis pipeline
    return {symbol: None for symbol in symbols}
