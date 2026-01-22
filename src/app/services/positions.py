"""Position management service."""
from typing import Optional, Dict, Any, List
from datetime import datetime
from mdb_engine.observability import get_logger
from .analysis import get_market_data, analyze_technicals

logger = get_logger(__name__)


class PositionMetrics:
    """Calculated metrics for a position."""
    def __init__(
        self,
        entry_price: float,
        current_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        qty: int = 0,
        unrealized_pl: float = 0.0,
        market_value: float = 0.0
    ):
        self.entry_price = entry_price
        self.current_price = current_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.qty = qty
        self.unrealized_pl = unrealized_pl
        self.market_value = market_value
        
        # Calculate derived metrics
        self.pl_pct = (unrealized_pl / (market_value - unrealized_pl)) * 100 if market_value != unrealized_pl else 0
        self.distance_to_stop = current_price - stop_loss if stop_loss else None
        self.distance_to_target = take_profit - current_price if take_profit else None
        
        # Calculate progress percentages
        self.progress_to_stop = self._calculate_stop_progress()
        self.progress_to_profit = self._calculate_profit_progress()
        
        # Determine visual states
        self.is_near_stop = self.progress_to_stop >= 70 if self.progress_to_stop else False
        self.is_near_profit = self.progress_to_profit >= 70 if self.progress_to_profit else False
        self.is_at_profit = self.progress_to_profit >= 95 if self.progress_to_profit else False
        self.is_at_stop = self.progress_to_stop >= 95 if self.progress_to_stop else False
        
        # Calculate double-ended bar position
        self.bar_position = self._calculate_bar_position()
        self.is_at_loss = unrealized_pl < 0
    
    def _calculate_stop_progress(self) -> Optional[float]:
        """Calculate progress toward stop loss (0-100%)."""
        if not self.stop_loss or not self.entry_price:
            return None
        
        if self.current_price <= self.entry_price:
            # Moving toward stop loss
            return max(0, min(100, ((self.entry_price - self.current_price) / (self.entry_price - self.stop_loss)) * 100))
        else:
            # Above entry, moving away from stop loss
            return 0
    
    def _calculate_profit_progress(self) -> Optional[float]:
        """Calculate progress toward take profit (0-100%)."""
        if not self.take_profit or not self.entry_price:
            return None
        
        if self.current_price >= self.entry_price:
            # Moving toward take profit
            return max(0, min(100, ((self.current_price - self.entry_price) / (self.take_profit - self.entry_price)) * 100))
        else:
            # Below entry, moving away from take profit
            return 0
    
    def _calculate_bar_position(self) -> Optional[Dict[str, float]]:
        """Calculate position on double-ended bar (stop loss <-> take profit).
        
        Returns dict with:
        - position_pct: Current price position (0-100%, where 0=stop_loss, 100=take_profit)
        - entry_pct: Entry price position (0-100%)
        - stop_width: Width of red zone (0-50%)
        - profit_width: Width of green zone (0-50%)
        """
        if not self.stop_loss or not self.take_profit:
            return None
        
        total_range = self.take_profit - self.stop_loss
        if total_range <= 0:
            return None
        
        # Calculate positions as percentages
        current_pos = ((self.current_price - self.stop_loss) / total_range) * 100
        entry_pos = ((self.entry_price - self.stop_loss) / total_range) * 100
        
        # Clamp to 0-100%
        current_pos = max(0, min(100, current_pos))
        entry_pos = max(0, min(100, entry_pos))
        
        # Calculate zone widths (entry is the center point)
        stop_width = entry_pos  # Red zone from 0 to entry
        profit_width = 100 - entry_pos  # Green zone from entry to 100
        
        return {
            'current_pct': current_pos,
            'entry_pct': entry_pos,
            'stop_width': stop_width,
            'profit_width': profit_width
        }


class SellSignal:
    """Sell signal detection result."""
    def __init__(self, signal_type: str, reason: str):
        self.signal_type = signal_type  # profit_target, near_target, overbought, consider_sell
        self.reason = reason


async def calculate_position_metrics(
    position: Any,
    trade_record: Optional[Dict[str, Any]] = None,
    db = None
) -> PositionMetrics:
    """Calculate metrics for a position.
    
    Args:
        position: Alpaca position object
        trade_record: Trade record from database (optional)
    
    Returns:
        PositionMetrics object with all calculated values
    """
    pl = float(position.unrealized_pl)
    market_value = float(position.market_value)
    current_price = float(position.current_price)
    avg_entry_price = float(position.avg_entry_price)
    
    # Get entry/stop/take_profit from trade record or use defaults
    entry_price = trade_record.get("entry_price") if trade_record else avg_entry_price
    stop_loss = trade_record.get("stop_loss") if trade_record else None
    take_profit = trade_record.get("take_profit") if trade_record else None
    
    # If take_profit is missing, calculate it using current ATR (entry + 3*ATR)
    if not take_profit:
        try:
            bars, _, _ = await get_market_data(position.symbol, days=100, db=db)
            if bars is not None and not bars.empty and len(bars) >= 14:
                techs = analyze_technicals(bars)
                atr = techs.get('atr', 0)
                if atr > 0:
                    # Calculate take profit: entry + 3*ATR (standard strategy)
                    take_profit = round(entry_price + (3 * atr), 2)
                    logger.debug(f"Calculated take_profit for {position.symbol}: ${take_profit} (entry: ${entry_price}, ATR: ${atr})")
        except Exception as e:
            logger.debug(f"Could not calculate take_profit for {position.symbol}: {e}")
    
    # If stop_loss is missing, calculate it using current ATR (entry - 2*ATR)
    if not stop_loss:
        try:
            bars, _, _ = await get_market_data(position.symbol, days=100, db=db)
            if bars is not None and not bars.empty and len(bars) >= 14:
                techs = analyze_technicals(bars)
                atr = techs.get('atr', 0)
                if atr > 0:
                    # Calculate stop loss: entry - 2*ATR (standard strategy)
                    stop_loss = round(entry_price - (2 * atr), 2)
                    logger.debug(f"Calculated stop_loss for {position.symbol}: ${stop_loss} (entry: ${entry_price}, ATR: ${atr})")
        except Exception as e:
            logger.debug(f"Could not calculate stop_loss for {position.symbol}: {e}")
    
    return PositionMetrics(
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        qty=int(position.qty),
        unrealized_pl=pl,
        market_value=market_value
    )


async def detect_sell_signal(
    symbol: str,
    current_price: float,
    take_profit: Optional[float],
    unrealized_pl: float,
    db = None
) -> Optional[SellSignal]:
    """Detect sell signals based on technical analysis.
    
    Args:
        symbol: Stock symbol
        current_price: Current stock price
        take_profit: Take profit target price
        unrealized_pl: Unrealized profit/loss
    
    Returns:
        SellSignal if signal detected, None otherwise
    """
    try:
        bars, _, _ = await get_market_data(symbol, days=100, db=db)
        if bars is None or bars.empty or len(bars) < 14:
            return None
        
        techs = analyze_technicals(bars)
        rsi = techs.get('rsi', 50)
        
        # Check sell signals in priority order
        if take_profit and current_price >= take_profit:
            return SellSignal("profit_target", "Profit target reached! Time to sell high.")
        
        if take_profit and current_price >= take_profit * 0.98:  # Within 2% of target
            return SellSignal("near_target", "Near profit target - consider selling high")
        
        if rsi > 70:
            return SellSignal("overbought", "RSI overbought - stock may be overvalued")
        
        if rsi > 65 and unrealized_pl > 0:  # Profitable and getting overbought
            return SellSignal("consider_sell", "RSI rising, profit locked - consider selling high")
        
        return None
    except Exception as e:
        logger.debug(f"Could not analyze sell signals for {symbol}: {e}")
        return None
