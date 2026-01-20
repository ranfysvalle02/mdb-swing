"""Trading execution services.

MDB-Engine Integration:
- Logging: Uses `get_logger(__name__)` from mdb_engine.observability for structured logging
- Database: Uses scoped database via mdb-engine for trade history (when needed)
"""
import math
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from alpaca_trade_api.rest import TimeFrame
# Indicators now computed via analyze_technicals() using vectorbt
from mdb_engine.observability import get_logger
from ..core.config import ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, STRATEGY_CONFIG
from .analysis import api, get_market_data, analyze_technicals

logger = get_logger(__name__)

def calculate_position_size(price: float, atr: float, risk_per_trade_dollars: float = None) -> int:
    """Calculate position size based on volatility (ATR).
    
    Args:
        price: Current stock price
        atr: Average True Range (volatility measure)
        risk_per_trade_dollars: Optional risk amount per trade. If None, uses strategy config.
        
    Returns:
        Number of shares to buy, calculated to risk the specified amount based on stop distance (2×ATR)
    """
    if risk_per_trade_dollars is None:
        risk_per_trade_dollars = STRATEGY_CONFIG['risk_per_trade']
    stop_distance = 2 * atr
    if stop_distance == 0:
        return 0
    
    shares_risk = math.floor(risk_per_trade_dollars / stop_distance)
    
    buying_power = 10000.0  # Default fallback
    if api:
        try:
            acct = api.get_account()
            buying_power = float(acct.buying_power)
        except Exception as e:
            logger.warning(f"Could not get account info: {e}, using default buying power")
    
    # Use strategy config max capital
    max_capital = STRATEGY_CONFIG['max_capital']
    max_shares_budget = math.floor(min(buying_power, max_capital) / price)
    
    return int(min(shares_risk, max_shares_budget))

async def place_order(symbol: str, side: str, reason: str, score: int, db, qty: Optional[int] = None) -> Optional[Any]:
    """Place a trading order.
    
    Args:
        symbol: Stock symbol
        side: 'buy' or 'sell'
        reason: Reason for the order
        score: AI score (0-10)
        db: Database connection
        qty: Optional quantity. If None, calculates based on risk/budget.
    
    Returns:
        Order object if successful, None if failed (with detailed logging)
    """
    if api is None:
        logger.error("Cannot place order: Alpaca API not initialized")
        return None
    
    logger.info(f"Placing {side} order for {symbol}...")
    
    # Get market data
    try:
        df, headlines, _, _ = await get_market_data(symbol)
        if df is None or df.empty:
            logger.error(f"Cannot place order for {symbol}: No market data returned")
            return None
        logger.debug(f"Got market data for {symbol}: {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to get market data for {symbol}: {e}", exc_info=True)
        return None
    
    # Analyze technicals
    try:
        techs = analyze_technicals(df)
        if not techs or 'price' not in techs:
            logger.error(f"Technical analysis returned invalid data for {symbol}")
            return None
        logger.debug(f"Technical analysis for {symbol}: price={techs.get('price')}, atr={techs.get('atr')}")
    except Exception as e:
        logger.error(f"Technical analysis failed for {symbol}: {e}", exc_info=True)
        return None
    
    # Calculate or use provided position size
    if qty is None:
        try:
            qty = calculate_position_size(techs['price'], techs['atr'])
            if qty < 1:
                logger.warning(f"Skipping {symbol}: Position size 0 (Risk/Budget) - price={techs['price']}, atr={techs['atr']}")
                return None
            logger.info(f"Calculated position size for {symbol}: {qty} shares")
        except Exception as e:
            logger.error(f"Position size calculation failed for {symbol}: {e}", exc_info=True)
            return None
    else:
        # Validate provided quantity
        if qty < 1 or qty > 99:
            logger.error(f"Invalid quantity {qty} for {symbol}. Must be between 1-99.")
            return None
        logger.info(f"Using provided quantity for {symbol}: {qty} shares")

    # Calculate order prices
    limit_price = techs['price']
    stop_price = round(techs['price'] - (2 * techs['atr']), 2)
    take_profit = round(techs['price'] + (3 * techs['atr']), 2)
    
    logger.info(f"Submitting {side} order for {symbol}: qty={qty}, limit={limit_price}, stop={stop_price}, target={take_profit}")
    
    # Submit order to Alpaca
    try:
        order = api.submit_order(
            symbol=symbol, qty=qty, side=side, type='limit', limit_price=limit_price,
            time_in_force='day', order_class='bracket',
            take_profit={'limit_price': take_profit},
            stop_loss={'stop_price': stop_price}
        )
        logger.info(f"Order submitted successfully for {symbol}: {getattr(order, 'id', 'N/A')}")
        
        # Log to DB using mdb-engine scoped database
        trade_record = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "action": side,
            "qty": qty,
            "price": limit_price,
            "reason": reason,
            "score": score,
            "entry_price": limit_price,
            "stop_loss": stop_price,
            "take_profit": take_profit
        }
        await db.history.insert_one(trade_record)
        
        # Store initial analysis to radar_history for outcome tracking
        # MDB-Engine Pattern: Get EmbeddingService for non-route context
        try:
            from .radar import RadarService
            from mdb_engine.embeddings.dependencies import get_embedding_service_for_app
            from ..core.engine import engine, APP_SLUG
            
            embedding_service = get_embedding_service_for_app(APP_SLUG, engine)
            radar_service = RadarService(db, embedding_service=embedding_service)
            
            # Create verdict dict for storage (MongoDB can't serialize objects)
            verdict_dict = {
                'score': score,
                'reason': reason,
                'risk_level': 'MEDIUM',
                'action': 'BUY'
            }
            analysis_data = {
                'symbol': symbol,
                'techs': techs,
                'headlines': headlines,
                'verdict': verdict_dict,  # Use dict instead of object
                'timestamp': datetime.now(),
                'strategy': 'balanced_low'  # Default, could be passed as parameter
            }
            
            # Store with outcome=None initially (will be updated when position closes)
            await radar_service.store_to_history(symbol, analysis_data, outcome=None)
        except Exception as e:
            logger.warning(f"Could not store analysis to radar_history: {e}")
        
        return order
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Alpaca API order submission failed for {symbol}: {error_msg}", exc_info=True)
        # Check for common Alpaca API errors
        if "insufficient buying power" in error_msg.lower():
            logger.error(f"Insufficient buying power for {symbol}")
        elif "invalid symbol" in error_msg.lower():
            logger.error(f"Invalid symbol: {symbol}")
        elif "market closed" in error_msg.lower():
            logger.error(f"Market is closed - cannot place order for {symbol}")
        return None

async def update_trade_outcome(symbol: str, entry_price: float, exit_price: float, 
                               pnl: float, days_held: int, exit_reason: str, db) -> bool:
    """Update trade outcome in radar_history.
    
    Args:
        symbol: Stock symbol
        entry_price: Entry price of the trade
        exit_price: Exit price of the trade
        pnl: Profit/loss amount
        days_held: Number of days position was held
        exit_reason: Reason for exit (stop_loss, take_profit, manual_close)
        db: MongoDB database instance
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        from .radar import RadarService
        # MDB-Engine Pattern: Get EmbeddingService for non-route context
        from mdb_engine.embeddings.dependencies import get_embedding_service_for_app
        from ..core.engine import engine, APP_SLUG
        
        embedding_service = get_embedding_service_for_app(APP_SLUG, engine)
        radar_service = RadarService(db, embedding_service=embedding_service)
        
        # Find the most recent historical record for this symbol without an outcome
        # Match by symbol and approximate entry price (within 1%)
        price_tolerance = entry_price * 0.01
        
        history_record = await db.radar_history.find_one({
            "symbol": symbol,
            "outcome": None,
            "analysis.techs.price": {
                "$gte": entry_price - price_tolerance,
                "$lte": entry_price + price_tolerance
            }
        }, sort=[("timestamp", -1)])
        
        if history_record:
            outcome = {
                "pnl": pnl,
                "days_held": days_held,
                "exit_reason": exit_reason,
                "profitable": pnl > 0,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "updated_at": datetime.now()
            }
            
            await db.radar_history.update_one(
                {"_id": history_record["_id"]},
                {"$set": {"outcome": outcome}}
            )
            
            logger.info(f"Updated trade outcome for {symbol}: P&L=${pnl:.2f}, {days_held} days, {exit_reason}")
            return True
        else:
            logger.warning(f"Could not find historical record to update for {symbol}")
            return False
    except Exception as e:
        logger.error(f"Error updating trade outcome for {symbol}: {e}", exc_info=True)
        return False
