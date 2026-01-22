"""Trading execution services.

MDB-Engine Integration:
- Logging: Uses `get_logger(__name__)` from mdb_engine.observability for structured logging
- Database: Uses scoped database via mdb-engine for trade history (when needed)
"""
import math
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from alpaca_trade_api.rest import TimeFrame
from mdb_engine.observability import get_logger
from ..core.config import ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, STRATEGY_CONFIG
from .analysis import api, get_market_data, analyze_technicals
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import alpaca_trade_api as tradeapi

logger = get_logger(__name__)

def get_api_client(db=None):
    """Get Alpaca API client, preferring account manager over global config.
    
    Args:
        db: Optional database connection for account manager
        
    Returns:
        Alpaca REST API client or None
    """
    # Try account manager first if db is provided
    if db:
        try:
            from .alpaca_accounts import AlpacaAccountManager
            account_manager = AlpacaAccountManager(db)
            # This is sync, but we need async - routes should pass client directly
            # For now, fall back to global api
            pass
        except Exception:
            pass
    
    # Fall back to global api
    return api

def calculate_position_size(price: float, atr: float, risk_per_trade_dollars: float = None) -> int:
    """Calculate position size based on volatility (ATR).
    
    Args:
        price: Current stock price
        atr: Average True Range (volatility measure)
        risk_per_trade_dollars: Optional risk amount per trade. If None, uses strategy config or default.
        
    Returns:
        Number of shares to buy, calculated to risk the specified amount based on stop distance (2×ATR)
    """
    if risk_per_trade_dollars is None:
        risk_per_trade_dollars = STRATEGY_CONFIG.get('risk_per_trade', 100.0)  # Default $100 risk per trade
    
    stop_distance = 2 * atr
    if stop_distance == 0:
        return 0
    
    shares_risk = math.floor(risk_per_trade_dollars / stop_distance)
    
    buying_power = 10000.0  # Default fallback
    # Note: This function doesn't have access to db/account_manager
    # Routes should pass buying_power if available
    if api:
        try:
            acct = api.get_account()
            buying_power = float(acct.buying_power)
        except Exception as e:
            logger.warning(f"Could not get account info: {e}, using default buying power")
    
    max_capital = STRATEGY_CONFIG.get('max_capital', 10000.0)  # Default $10k max capital
    max_shares_budget = math.floor(min(buying_power, max_capital) / price)
    
    return int(min(shares_risk, max_shares_budget))

async def place_order(symbol: str, side: str, reason: str, score: int, db, qty: Optional[int] = None, api_client: Optional['tradeapi.REST'] = None) -> Optional[Any]:
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
    # Use provided API client or fall back to global
    trading_api = api_client if api_client else api
    if trading_api is None:
        logger.error("Cannot place order: Alpaca API not initialized")
        return None
    
    logger.info(f"Placing {side} order for {symbol}...")
    
    try:
        df, headlines, news_objects = await get_market_data(symbol)
        if df is None or df.empty:
            logger.error(f"Cannot place order for {symbol}: No market data returned")
            return None
        logger.debug(f"Got market data for {symbol}: {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to get market data for {symbol}: {e}", exc_info=True)
        return None
    
    try:
        techs = analyze_technicals(df)
        if not techs or 'price' not in techs:
            logger.error(f"Technical analysis returned invalid data for {symbol}")
            return None
        logger.debug(f"Technical analysis for {symbol}: price={techs.get('price')}, atr={techs.get('atr')}")
    except Exception as e:
        logger.error(f"Technical analysis failed for {symbol}: {e}", exc_info=True)
        return None
    
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
        if qty < 1 or qty > 99:
            logger.error(f"Invalid quantity {qty} for {symbol}. Must be between 1-99.")
            return None
        logger.info(f"Using provided quantity for {symbol}: {qty} shares")

    # Use current market price for stop/take profit calculations
    estimated_price = techs['price']
    stop_price = round(estimated_price - (2 * techs['atr']), 2)
    take_profit = round(estimated_price + (3 * techs['atr']), 2)
    
    logger.info(f"Submitting MARKET {side} order for {symbol}: qty={qty}, estimated_price=${estimated_price:.2f}, stop={stop_price}, target={take_profit}")
    
    try:
        # Submit market order (no limit_price needed)
        order = trading_api.submit_order(
            symbol=symbol, qty=qty, side=side, type='market',
            time_in_force='day', order_class='bracket',
            take_profit={'limit_price': take_profit},
            stop_loss={'stop_price': stop_price}
        )
        order_id = getattr(order, 'id', 'N/A')
        logger.info(f"Market order submitted successfully for {symbol}: {order_id}")
        
        # Market orders typically fill immediately, but we need to get the actual fill price
        # Retry mechanism to get fill price (handle Alpaca API propagation delay)
        actual_fill_price = estimated_price  # Fallback to estimated price
        max_retries = 5
        retry_delay = 0.2  # 200ms between retries
        
        for attempt in range(max_retries):
            try:
                # Get updated order status
                updated_order = trading_api.get_order(order_id)
                if updated_order:
                    # Check if order is filled
                    order_status = getattr(updated_order, 'status', '').lower()
                    filled_avg_price = getattr(updated_order, 'filled_avg_price', None)
                    
                    if order_status == 'filled' and filled_avg_price:
                        actual_fill_price = float(filled_avg_price)
                        logger.info(f"Order {order_id} filled at ${actual_fill_price:.2f} (attempt {attempt + 1})")
                        break
                    elif order_status in ['filled', 'partially_filled']:
                        # Order is filled or partially filled
                        if filled_avg_price:
                            actual_fill_price = float(filled_avg_price)
                            logger.info(f"Order {order_id} {order_status} at ${actual_fill_price:.2f}")
                        break
                    elif order_status in ['canceled', 'cancelled', 'rejected', 'expired']:
                        logger.warning(f"Order {order_id} was {order_status} - cannot get fill price")
                        break
                    elif attempt < max_retries - 1:
                        # Order still pending, wait and retry
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        logger.debug(f"Order {order_id} still {order_status}, retrying to get fill price... (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    logger.debug(f"Error getting order status for {order_id}: {e}, retrying...")
                else:
                    logger.warning(f"Could not get fill price for order {order_id} after {max_retries} attempts, using estimated price ${estimated_price:.2f}")
        
        # Recalculate stop/take profit based on actual fill price (if different from estimated)
        if abs(actual_fill_price - estimated_price) > 0.01:  # If fill price differs significantly
            price_adjustment = actual_fill_price - estimated_price
            stop_price = round(stop_price + price_adjustment, 2)
            take_profit = round(take_profit + price_adjustment, 2)
            logger.info(f"Adjusted stop/take profit based on actual fill price: stop=${stop_price:.2f}, target=${take_profit:.2f}")
        
        trade_record = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "action": side,
            "qty": qty,
            "price": actual_fill_price,  # Use actual fill price
            "reason": reason,
            "score": score,
            "entry_price": actual_fill_price,
            "stop_loss": stop_price,
            "take_profit": take_profit
        }
        await db.history.insert_one(trade_record)
        
        try:
            from .radar import RadarService
            from mdb_engine.embeddings.dependencies import get_embedding_service_for_app
            from ..core.engine import engine, APP_SLUG
            
            embedding_service = get_embedding_service_for_app(APP_SLUG, engine)
            radar_service = RadarService(db, embedding_service=embedding_service)
            
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
                'verdict': verdict_dict,
                'timestamp': datetime.now(),
                'strategy': 'balanced_low'
            }
            
            await radar_service.store_to_history(symbol, analysis_data, outcome=None)
        except Exception as e:
            logger.warning(f"Could not store analysis to radar_history: {e}")
        
        return order
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Alpaca API order submission failed for {symbol}: {error_msg}", exc_info=True)
        if "insufficient buying power" in error_msg.lower():
            logger.error(f"Insufficient buying power for {symbol}")
        elif "invalid symbol" in error_msg.lower():
            logger.error(f"Invalid symbol: {symbol}")
        elif "market closed" in error_msg.lower():
            logger.error(f"Market is closed - cannot place order for {symbol}")
        return None

async def get_current_market_price(symbol: str, api_client: Optional['tradeapi.REST'] = None) -> Optional[float]:
    """Get current market price for a symbol.
    
    Tries multiple methods:
    1. Latest minute bar close price (fastest, most accurate during market hours)
    2. Latest daily bar close price (fallback)
    3. get_market_data() (last resort)
    
    Args:
        symbol: Stock symbol
        api_client: Optional Alpaca API client. If None, uses global api.
        
    Returns:
        Current market price or None if unavailable
    """
    trading_api = api_client if api_client else api
    if trading_api is None:
        logger.debug(f"Cannot get current price for {symbol}: API not initialized")
        return None
    
    try:
        # Try to get latest minute bar (most accurate for current price during market hours)
        try:
            bars = trading_api.get_bars(symbol, TimeFrame.Minute, limit=1, feed='iex')
            if bars and len(bars) > 0:
                price = float(bars[-1].c)  # Close price of latest bar
                logger.debug(f"Got current price for {symbol} from latest minute bar: ${price:.2f}")
                return price
        except Exception as e:
            logger.debug(f"Could not get latest minute bar for {symbol}: {e}")
        
        # Fallback: get latest daily bar close price
        try:
            bars = trading_api.get_bars(symbol, TimeFrame.Day, limit=1, feed='iex')
            if bars and len(bars) > 0:
                price = float(bars[-1].c)  # Close price of latest bar
                logger.debug(f"Got current price for {symbol} from latest daily bar: ${price:.2f}")
                return price
        except Exception as e:
            logger.debug(f"Could not get latest daily bar for {symbol}: {e}")
        
        # Last resort: use get_market_data (slower but reliable)
        try:
            df, _, _ = await get_market_data(symbol, days=1)
            if df is not None and not df.empty:
                price = float(df.iloc[-1]['close'])
                logger.debug(f"Got current price for {symbol} from market data: ${price:.2f}")
                return price
        except Exception as e:
            logger.debug(f"Could not get price from market data for {symbol}: {e}")
        
        logger.warning(f"Could not determine current price for {symbol}")
        return None
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}", exc_info=True)
        return None


def get_market_status(api_client: Optional['tradeapi.REST'] = None) -> Dict[str, Any]:
    """Get market status (open/closed) from Alpaca API.
    
    Args:
        api_client: Optional Alpaca API client. If None, uses global api.
        
    Returns:
        Dict with keys:
        - is_open: bool indicating if market is currently open
        - next_open: Optional datetime of next market open
        - next_close: Optional datetime of next market close
    """
    trading_api = api_client if api_client else api
    if trading_api is None:
        logger.debug("Cannot get market status: API not initialized")
        # Default to market open for graceful degradation
        return {"is_open": True, "next_open": None, "next_close": None}
    
    try:
        clock = trading_api.get_clock()
        if clock:
            is_open = getattr(clock, 'is_open', False)
            next_open = getattr(clock, 'next_open', None)
            next_close = getattr(clock, 'next_close', None)
            
            logger.debug(f"Market status: is_open={is_open}, next_open={next_open}, next_close={next_close}")
            return {
                "is_open": is_open,
                "next_open": next_open,
                "next_close": next_close
            }
        else:
            logger.warning("Clock API returned None, assuming market is open")
            return {"is_open": True, "next_open": None, "next_close": None}
    except Exception as e:
        logger.warning(f"Could not get market status: {e}, assuming market is open")
        # Graceful degradation: assume market is open
        return {"is_open": True, "next_open": None, "next_close": None}


def analyze_pending_order_status(
    order: Any,
    current_price: Optional[float],
    market_open: bool
) -> Dict[str, Any]:
    """Analyze why a pending order hasn't filled yet.
    
    Handles both market and limit orders intelligently.
    
    Args:
        order: Alpaca order object
        current_price: Current market price (or None if unavailable)
        market_open: Whether market is currently open
        
    Returns:
        Dict with keys:
        - status_message: Human-readable explanation
        - price_diff_pct: Percentage difference (None for market orders)
        - can_fill: bool indicating if order should fill soon
    """
    side = getattr(order, 'side', '').lower()
    order_status = getattr(order, 'status', '').lower()
    order_type = getattr(order, 'type', '').lower()
    limit_price = None
    if hasattr(order, 'limit_price') and order.limit_price:
        limit_price = float(order.limit_price)
    
    # Handle partially filled orders
    if order_status == 'partially_filled':
        filled_qty = getattr(order, 'filled_qty', 0)
        qty = getattr(order, 'qty', 0)
        return {
            "status_message": f"Partially filled ({filled_qty}/{qty} shares) - executing remaining shares",
            "price_diff_pct": None,
            "can_fill": True
        }
    
    # Handle market orders (they should fill immediately when market is open)
    if order_type == 'market':
        if not market_open:
            return {
                "status_message": "Market is closed - order will execute at market open",
                "price_diff_pct": None,
                "can_fill": False
            }
        else:
            # Market orders should fill immediately when market is open
            if order_status in ['new', 'pending_new', 'accepted']:
                return {
                    "status_message": "Market order executing - should fill immediately",
                    "price_diff_pct": None,
                    "can_fill": True
                }
            elif order_status == 'filled':
                return {
                    "status_message": "Order filled",
                    "price_diff_pct": None,
                    "can_fill": True
                }
            else:
                return {
                    "status_message": f"Market order pending (status: {order_status})",
                    "price_diff_pct": None,
                    "can_fill": True
                }
    
    # Handle limit orders (legacy support)
    # If market is closed, that's the primary reason
    if not market_open:
        return {
            "status_message": "Market is closed - order will execute when market opens",
            "price_diff_pct": None,
            "can_fill": False
        }
    
    # If we don't have current price or limit price, can't analyze price difference
    if current_price is None or limit_price is None:
        return {
            "status_message": "Waiting for execution",
            "price_diff_pct": None,
            "can_fill": None
        }
    
    # Analyze price difference for limit orders
    if side == 'buy':
        # For buy orders, we want limit_price >= current_price for fill
        price_diff = limit_price - current_price
        price_diff_pct = (price_diff / current_price) * 100 if current_price > 0 else 0
        
        if price_diff >= 0:
            # Limit price is at or above current price - should fill
            if price_diff_pct > 0.5:
                return {
                    "status_message": f"Price reached - limit ${limit_price:.2f} is {price_diff_pct:.1f}% above current ${current_price:.2f}",
                    "price_diff_pct": price_diff_pct,
                    "can_fill": True
                }
            else:
                return {
                    "status_message": f"Price reached - should fill soon",
                    "price_diff_pct": price_diff_pct,
                    "can_fill": True
                }
        else:
            # Limit price is below current price - waiting for price to drop
            return {
                "status_message": f"Waiting for price - limit ${limit_price:.2f} is {abs(price_diff_pct):.1f}% below current ${current_price:.2f}",
                "price_diff_pct": price_diff_pct,
                "can_fill": False
            }
    else:
        # For sell orders, we want limit_price <= current_price for fill
        price_diff = current_price - limit_price
        price_diff_pct = (price_diff / limit_price) * 100 if limit_price > 0 else 0
        
        if price_diff >= 0:
            # Current price is at or above limit - should fill
            if price_diff_pct > 0.5:
                return {
                    "status_message": f"Price reached - current ${current_price:.2f} is {price_diff_pct:.1f}% above limit ${limit_price:.2f}",
                    "price_diff_pct": price_diff_pct,
                    "can_fill": True
                }
            else:
                return {
                    "status_message": f"Price reached - should fill soon",
                    "price_diff_pct": price_diff_pct,
                    "can_fill": True
                }
        else:
            # Current price is below limit - waiting for price to rise
            return {
                "status_message": f"Waiting for price - limit ${limit_price:.2f} is {abs(price_diff_pct):.1f}% above current ${current_price:.2f}",
                "price_diff_pct": price_diff_pct,
                "can_fill": False
            }


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
        from mdb_engine.embeddings.dependencies import get_embedding_service_for_app
        from ..core.engine import engine, APP_SLUG
        
        embedding_service = get_embedding_service_for_app(APP_SLUG, engine)
        radar_service = RadarService(db, embedding_service=embedding_service)
        
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
