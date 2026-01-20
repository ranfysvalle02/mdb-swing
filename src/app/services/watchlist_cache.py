"""Watchlist caching service for market data and evaluation results.

This module provides caching for:
- Market data (bars, news) with 5-minute TTL
- Evaluation results per config hash with 1-minute TTL

Cache invalidation:
- Market data expires after TTL (prices change frequently)
- Evaluation cache invalidated when config hash changes
- Manual cache clear available for debugging
"""
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from mdb_engine.observability import get_logger

logger = get_logger(__name__)


class WatchlistCache:
    """In-memory cache for watchlist data with TTL support."""
    
    def __init__(self):
        # Cache structure: key -> (data, timestamp)
        self.market_data_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.evaluation_cache: Dict[str, Tuple[Dict, datetime]] = {}
        self._cache_stats = {
            'market_data_hits': 0,
            'market_data_misses': 0,
            'evaluation_hits': 0,
            'evaluation_misses': 0,
        }
    
    def get_market_data(self, symbol: str, ttl_seconds: int = 300) -> Optional[Any]:
        """Get cached market data if available and not expired.
        
        Args:
            symbol: Stock symbol (uppercased automatically)
            ttl_seconds: Time-to-live in seconds (default: 300 = 5 minutes)
            
        Returns:
            Cached data tuple (bars, news_items, news_objects) or None if expired/not found
        """
        key = symbol.upper()
        if key in self.market_data_cache:
            data, timestamp = self.market_data_cache[key]
            age = datetime.now() - timestamp
            if age < timedelta(seconds=ttl_seconds):
                self._cache_stats['market_data_hits'] += 1
                logger.debug(f"Cache hit for market data: {symbol}")
                return data
            else:
                # Expired - remove from cache
                del self.market_data_cache[key]
                logger.debug(f"Cache expired for market data: {symbol} (age: {age.total_seconds():.1f}s)")
        
        self._cache_stats['market_data_misses'] += 1
        return None
    
    def set_market_data(self, symbol: str, data: Any):
        """Cache market data with current timestamp.
        
        Args:
            symbol: Stock symbol (uppercased automatically)
            data: Data tuple (bars, news_items, news_objects)
        """
        key = symbol.upper()
        self.market_data_cache[key] = (data, datetime.now())
        logger.debug(f"Cached market data for {symbol}")
    
    def get_evaluation(self, symbol: str, config_hash: str, ttl_seconds: int = 60) -> Optional[Dict]:
        """Get cached evaluation result if available and not expired.
        
        Args:
            symbol: Stock symbol (uppercased automatically)
            config_hash: Configuration hash (from calculate_watchlist_config_hash)
            ttl_seconds: Time-to-live in seconds (default: 60 = 1 minute)
            
        Returns:
            Cached evaluation dict or None if expired/not found
        """
        key = f"{symbol.upper()}:{config_hash}"
        if key in self.evaluation_cache:
            result, timestamp = self.evaluation_cache[key]
            age = datetime.now() - timestamp
            if age < timedelta(seconds=ttl_seconds):
                self._cache_stats['evaluation_hits'] += 1
                logger.debug(f"Cache hit for evaluation: {symbol} (config: {config_hash[:8]}...)")
                return result
            else:
                # Expired - remove from cache
                del self.evaluation_cache[key]
                logger.debug(f"Cache expired for evaluation: {symbol} (age: {age.total_seconds():.1f}s)")
        
        self._cache_stats['evaluation_misses'] += 1
        return None
    
    def set_evaluation(self, symbol: str, config_hash: str, result: Dict):
        """Cache evaluation result with current timestamp.
        
        Args:
            symbol: Stock symbol (uppercased automatically)
            config_hash: Configuration hash
            result: Evaluation result dictionary
        """
        key = f"{symbol.upper()}:{config_hash}"
        self.evaluation_cache[key] = (result, datetime.now())
        logger.debug(f"Cached evaluation for {symbol} (config: {config_hash[:8]}...)")
    
    def invalidate_config(self, config_hash: str):
        """Invalidate all evaluations for a specific config hash.
        
        Args:
            config_hash: Configuration hash to invalidate
        """
        keys_to_remove = [k for k in self.evaluation_cache.keys() if k.endswith(f":{config_hash}")]
        for key in keys_to_remove:
            del self.evaluation_cache[key]
        logger.info(f"Invalidated {len(keys_to_remove)} evaluation cache entries for config hash {config_hash[:8]}...")
    
    def clear_all(self):
        """Clear all caches (useful for debugging)."""
        market_count = len(self.market_data_cache)
        eval_count = len(self.evaluation_cache)
        self.market_data_cache.clear()
        self.evaluation_cache.clear()
        logger.info(f"Cleared all caches: {market_count} market data entries, {eval_count} evaluation entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache hit/miss counts and current cache sizes
        """
        total_market_requests = self._cache_stats['market_data_hits'] + self._cache_stats['market_data_misses']
        total_eval_requests = self._cache_stats['evaluation_hits'] + self._cache_stats['evaluation_misses']
        
        market_hit_rate = (self._cache_stats['market_data_hits'] / total_market_requests * 100) if total_market_requests > 0 else 0
        eval_hit_rate = (self._cache_stats['evaluation_hits'] / total_eval_requests * 100) if total_eval_requests > 0 else 0
        
        return {
            'market_data': {
                'hits': self._cache_stats['market_data_hits'],
                'misses': self._cache_stats['market_data_misses'],
                'hit_rate': round(market_hit_rate, 1),
                'cached_entries': len(self.market_data_cache),
            },
            'evaluation': {
                'hits': self._cache_stats['evaluation_hits'],
                'misses': self._cache_stats['evaluation_misses'],
                'hit_rate': round(eval_hit_rate, 1),
                'cached_entries': len(self.evaluation_cache),
            },
        }


# Global cache instance
_cache = WatchlistCache()


def get_cache() -> WatchlistCache:
    """Get the global cache instance."""
    return _cache
