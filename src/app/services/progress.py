"""Progress calculation utilities for watchlist cards.

This module provides reusable functions for calculating progress percentages
for RSI and SMA-200 proximity indicators used in watchlist cards.

MDB-Engine Integration:
- Logging: Uses `get_logger(__name__)` from mdb_engine.observability
"""
from typing import Optional
from mdb_engine.observability import get_logger

logger = get_logger(__name__)

# Constants for progress calculations
MAX_PROGRESS_PCT = 100.0
MIN_PROGRESS_PCT = 0.0
MAX_PROGRESS_BELOW_SMA = 30.0  # Maximum progress % for stocks below SMA-200 (fails uptrend requirement)
RSI_OVERBOUGHT_THRESHOLD = 70.0  # RSI level considered overbought


def calculate_rsi_progress(rsi: Optional[float], rsi_threshold: float = 35.0) -> Optional[float]:
    """Calculate RSI progress percentage for UI display.
    
    Progress represents how close RSI is to the oversold threshold.
    Lower RSI = higher progress (closer to oversold).
    
    Args:
        rsi: Current RSI value (0-100)
        rsi_threshold: RSI threshold for entry criteria (default: 35)
    
    Returns:
        Progress percentage (0-100), or None if RSI is None
        - 0 RSI = 100% progress (extremely oversold)
        - threshold RSI = 50% progress (at threshold)
        - 70 RSI = 0% progress (overbought)
    """
    if rsi is None:
        return None
    
    # Invert: lower RSI = higher progress (closer to oversold)
    # Scale: 0 RSI = 100%, threshold RSI = 50%, 70 RSI = 0%
    progress = ((rsi_threshold - rsi) / rsi_threshold) * MAX_PROGRESS_PCT
    return max(MIN_PROGRESS_PCT, min(MAX_PROGRESS_PCT, progress))


def calculate_sma_progress(
    price_vs_sma_pct: Optional[float],
    sma_proximity_pct: float = 3.0
) -> Optional[float]:
    """Calculate SMA-200 proximity progress percentage for UI display.
    
    Progress represents how close price is to meeting entry criteria:
    - Price > SMA-200 (uptrend) AND within proximity threshold
    
    Args:
        price_vs_sma_pct: Percentage difference from SMA-200
            - Positive = price above SMA-200
            - Negative = price below SMA-200
        sma_proximity_pct: Maximum allowed percentage above SMA-200 (default: 3.0)
    
    Returns:
        Progress percentage (0-100), or None if price_vs_sma_pct is None
        - Price below SMA: 0-30% (fails uptrend requirement)
        - Price above SMA AND within proximity: 100% (meets criteria)
        - Price above SMA but beyond proximity: scales down from 100% to 0%
    """
    if price_vs_sma_pct is None or sma_proximity_pct <= 0:
        return None
    
    # Strategy requires: price > SMA-200 (uptrend) AND within proximity
    if price_vs_sma_pct <= 0:
        # Price below SMA: fails uptrend requirement, show low progress (0-30%)
        # Closer to SMA = higher progress, but never 100% (uptrend required)
        # Formula: 30 - (distance_below / proximity_threshold) * 30
        # At SMA (0% below): 30% progress
        # At proximity_threshold below: 0% progress
        distance_below = abs(price_vs_sma_pct)
        progress = MAX_PROGRESS_BELOW_SMA - (distance_below / sma_proximity_pct) * MAX_PROGRESS_BELOW_SMA
        return max(MIN_PROGRESS_PCT, min(MAX_PROGRESS_BELOW_SMA, progress))
    
    elif price_vs_sma_pct <= sma_proximity_pct:
        # Price above SMA AND within proximity: meets criteria (100%)
        return MAX_PROGRESS_PCT
    
    else:
        # Price above SMA but beyond proximity: scale down from 100% to 0%
        # Formula: 100 - ((distance_beyond_proximity / proximity_threshold) * 100)
        # At proximity threshold: 100% progress
        # At 2x proximity threshold: 0% progress
        distance_beyond = price_vs_sma_pct - sma_proximity_pct
        progress = MAX_PROGRESS_PCT - ((distance_beyond / sma_proximity_pct) * MAX_PROGRESS_PCT)
        return max(MIN_PROGRESS_PCT, min(MAX_PROGRESS_PCT, progress))
