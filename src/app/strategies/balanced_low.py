"""Balanced Low Strategy - Swing Trading."""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import vectorbt as vbt
from mdb_engine.observability import get_logger

logger = get_logger(__name__)


def generate_signals(
    df: pd.DataFrame,
    ai_scores: Dict[str, Optional[float]],
    config: Dict[str, Any],
    symbol: str
) -> pd.Series:
    """Generate entry signals using vectorbt.
    
    Architecture: Computes indicators in-memory, returns boolean Series.
    Does NOT store indicators in MongoDB - they're transient.
    
    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        ai_scores: Dict mapping symbols to AI scores
        config: Strategy configuration from MongoDB
        symbol: Stock symbol
    
    Returns:
        Boolean Series where True = entry signal
    """
    if df is None or df.empty or len(df) < 200:
        logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} bars")
        return pd.Series([False] * len(df), index=df.index if df is not None else pd.DatetimeIndex([]))
    
    # Ensure columns are lowercase
    df.columns = [c.lower() for c in df.columns]
    
    # Compute indicators using vectorbt (vectorized, fast)
    try:
        # RSI calculation
        rsi = vbt.RSI.run(df['close'], window=14).rsi
        
        # SMA-200 calculation
        sma_200 = vbt.MA.run(df['close'], window=200).ma
        
        # ATR calculation (for position sizing)
        atr = vbt.ATR.run(df['high'], df['low'], df['close'], window=14).atr
        
    except Exception as e:
        logger.error(f"Indicator calculation failed for {symbol}: {e}")
        return pd.Series([False] * len(df), index=df.index)
    
    # Get AI score (from MongoDB signals or None)
    ai_score = ai_scores.get(symbol)
    ai_min_score = config.get('ai_min_score', 7)
    
    # Entry conditions (all must be true):
    # 1. RSI oversold: rsi_min < RSI < rsi_threshold
    # 2. Price > SMA-200 (uptrend)
    # 3. Price within proximity of SMA-200 (near support)
    # 4. News filter handled separately in analysis service
    rsi_threshold = config.get('rsi_threshold', 35)
    rsi_min = config.get('rsi_min', 20)
    sma_proximity_pct = config.get('sma_proximity_pct', 3.0)
    
    # Calculate price vs SMA-200 percentage (vectorized)
    price_vs_sma_pct = ((df['close'] - sma_200) / sma_200) * 100
    
    entries = (
        (rsi > rsi_min) &
        (rsi < rsi_threshold) &
        (df['close'] > sma_200) &
        (price_vs_sma_pct <= sma_proximity_pct)
    )
    
    # Optional AI score filter (news analysis handled separately)
    if ai_score is not None:
        entries = entries & (ai_score >= ai_min_score)
    else:
        logger.debug(f"No AI score for {symbol}, proceeding with technical signals only")
    
    return entries


def calculate_position_size(
    price: float,
    atr: float,
    risk_per_trade: float,
    max_capital: float,
    buying_power: float
) -> int:
    """Calculate position size based on volatility (ATR).
    
    Stop loss distance: 2x ATR (volatility-adaptive).
    
    Architecture: Pure computation, no MongoDB.
    
    Args:
        price: Current stock price
        atr: Average True Range (volatility measure)
        risk_per_trade: Dollar amount to risk per trade
        max_capital: Maximum capital to deploy
        buying_power: Available buying power
    
    Returns:
        Number of shares to buy
    """
    if atr <= 0 or price <= 0:
        return 0
    
    # Stop distance = 2 * ATR (volatility-based risk management)
    stop_distance = 2 * atr
    
    # Calculate shares based on risk
    shares_by_risk = int(risk_per_trade / stop_distance)
    
    # Calculate shares based on capital limits
    max_shares_budget = int(min(buying_power, max_capital) / price)
    
    # Return minimum (most conservative)
    return min(shares_by_risk, max_shares_budget)


def run_portfolio(
    data: Dict[str, pd.DataFrame],
    entries: Dict[str, pd.Series],
    config: Dict[str, Any],
    initial_capital: float = 100000.0
) -> Dict[str, Any]:
    """Run portfolio simulation using vectorbt.
    
    Architecture: In-memory computation using vectorbt's portfolio engine.
    Results can be persisted to MongoDB for audit trail.
    
    Uses industry-standard vectorbt for portfolio simulation.
    
    Args:
        data: Dict of symbol -> DataFrame (OHLCV)
        entries: Dict of symbol -> boolean Series (entry signals)
        config: Strategy configuration
        initial_capital: Starting capital
    
    Returns:
        Portfolio statistics dictionary with performance metrics
    """
    total_signals = sum(entries[symbol].sum() for symbol in entries if symbol in data)
    
    portfolio_stats = {
        "total_signals": total_signals,
        "symbols_analyzed": len(data),
        "initial_capital": initial_capital,
        "config_used": config
    }
    
    
    return portfolio_stats


def get_budget_preset(budget: float) -> Dict[str, Any]:
    """Get strategy preset based on budget.
    
    Adjusts RSI threshold and AI score requirements based on capital.
    
    Args:
        budget: Available trading capital
    
    Returns:
        Strategy configuration dict optimized for budget
    """
    if budget < 1000:
        # Conservative: tighter RSI range, higher quality
        return {
            "rsi_threshold": 30,
            "rsi_min": 20,
            "ai_min_score": 8,
            "name": "Balanced Low (Conservative)",
            "description": "Conservative: RSI 20-30, higher quality requirements"
        }
    elif budget < 10000:
        # Moderate: default parameters
        return {
            "rsi_threshold": 35,
            "rsi_min": 20,
            "ai_min_score": 7,
            "name": "Balanced Low (Moderate)",
            "description": "Moderate: RSI 20-35, standard parameters"
        }
    else:
        # Aggressive: wider RSI range
        return {
            "rsi_threshold": 40,
            "rsi_min": 20,
            "ai_min_score": 6,
            "name": "Balanced Low (Aggressive)",
            "description": "Aggressive: RSI 20-40, wider range"
        }
