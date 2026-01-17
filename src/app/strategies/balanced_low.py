"""Balanced Low Strategy - PRIMARY STRATEGY for Swing Trading System.

This is the PRIMARY and ONLY active strategy in the system. All trading decisions
are made using the Balanced Low approach, which focuses on buying stocks at
"balanced lows" - temporary oversold conditions in stocks that are still in
long-term uptrends.

STRATEGY OVERVIEW:
------------------
The Balanced Low strategy identifies swing trading opportunities by finding stocks
that are temporarily oversold (RSI < 35) but still in an uptrend (price > 200-day SMA).
This creates a "balanced low" - not a crash, but a healthy pullback in a rising stock.

SWING TRADING CONTEXT:
----------------------
- Time Horizon: Multi-day to multi-week positions (not day trading)
- Data: Uses daily bars (not intraday)
- Entry: Buy at balanced lows when oversold in uptrend
- Exit: Stop loss (entry - 2×ATR) or take profit (entry + 3×ATR)
- Risk/Reward: 1:1.5 ratio (risk $1 to make $1.50)

ENTRY CRITERIA:
---------------
1. RSI < 35 (oversold but not extreme - "balanced")
   - RSI < 30 might indicate panic selling or fundamental issues
   - RSI 30-35 indicates healthy pullback in uptrend
   
2. Price > 200-day SMA (confirms uptrend)
   - Ensures we're buying dips in rising stocks, not catching falling knives
   - 200-day SMA filters out stocks in long-term downtrends
   
3. AI Score >= 7 (validates opportunity)
   - AI analyzes news sentiment, technical strength, and swing potential
   - Score 7-10 indicates good swing trading opportunity
   - Score < 7 means wait for better setup

EXIT CRITERIA:
--------------
- Stop Loss: Entry price - (2 × ATR)
  - ATR (Average True Range) measures volatility
  - 2×ATR provides buffer for normal price swings
  - Protects against trend reversal
  
- Take Profit: Entry price + (3 × ATR)
  - 3×ATR target captures swing move potential
  - Risk/Reward ratio: 1:1.5 (for every $1 risked, potential $1.50 reward)
  
- Manual Exit: Can sell at any time via UI

POSITION SIZING:
----------------
- Risk per trade: $50 (configurable)
- Position size calculated based on stop distance (2×ATR)
- Formula: shares = risk_amount / stop_distance
- Ensures consistent risk regardless of stock price or volatility

RISK MANAGEMENT:
----------------
- Maximum capital deployed: $5,000 (configurable)
- Each trade risks $50 maximum
- Stop loss prevents large losses
- Position sizing based on volatility (ATR)

WHY "BALANCED LOW"?
-------------------
- "Low" = Stock is oversold (RSI < 35), temporarily cheap
- "Balanced" = Not extreme (RSI not < 30), not in freefall, still in uptrend
- This creates opportunity: buying quality stocks on sale, not distressed assets

This strategy is designed for swing traders who want to:
- Capture multi-day to multi-week moves
- Buy quality stocks at temporary lows
- Use technical analysis + AI validation
- Manage risk with defined stop losses
"""
from typing import Dict, Any
from .base import Strategy


class BalancedLowStrategy(Strategy):
    """PRIMARY STRATEGY: Balanced Low Swing Trading Approach.
    
    The Eye focuses on balanced lows - oversold but stable, in uptrend, with upside potential.
    This is the ONLY active strategy in the system. All trading decisions use this approach.
    
    STRATEGY PHILOSOPHY:
    --------------------
    Buy quality stocks when they're temporarily "on sale" (oversold) but still in an uptrend.
    This is NOT buying crashes or distressed stocks - it's buying healthy pullbacks in rising stocks.
    
    This strategy seeks stocks that are:
    - At a balanced low (RSI < 35, oversold but not extreme)
      * RSI 30-35 = healthy pullback in uptrend
      * RSI < 30 = might indicate panic or fundamental issues (too extreme)
    - In an uptrend (price > 200-day SMA)
      * Confirms we're buying dips in rising stocks
      * Filters out stocks in long-term downtrends
    - Have clear upside potential (validated by AI score >= 7)
      * AI analyzes news, technicals, and swing potential
      * Score 7-10 = good swing opportunity
    - News is neutral/positive (no catastrophic events)
      * AI validates no major negative news
      * Catastrophic news = automatic veto (score 0)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Balanced Low strategy with configuration.
        
        Args:
            config: Strategy configuration dict. If None, uses defaults.
            
        Configuration Parameters:
        - rsi_threshold (35): Maximum RSI for entry (oversold threshold)
          * RSI < 35 = oversold but not extreme ("balanced")
          * Lower values (30) = more conservative, fewer signals
          * Higher values (40) = more aggressive, more signals
          
        - ai_score_required (7): Minimum AI score for trade entry
          * Score 7-10 = good swing opportunity
          * Score 4-6 = wait for better setup
          * Score 0-3 = avoid (poor opportunity or risk)
          
        - risk_per_trade (50.0): Dollars risked per trade
          * Used for position sizing calculation
          * Higher = larger positions, more risk
          * Lower = smaller positions, less risk
          
        - max_capital (5000.0): Maximum capital to deploy
          * Limits total exposure across all positions
          * Prevents over-leveraging
        """
        self.config = config or {
            "rsi_threshold": 35,  # Oversold but not extreme - "balanced low"
            "ai_score_required": 7,  # Good opportunity threshold
            "risk_per_trade": 50.0,  # $50 risk per trade
            "max_capital": 5000.0,  # Max capital deployed
        }
    
    def check_technical_signal(self, techs: Dict[str, Any]) -> bool:
        """Check if technical conditions indicate a balanced low opportunity.
        
        This is the FIRST filter - checks technical indicators only.
        AI validation happens separately and must also pass (score >= 7).
        
        Balanced low signal requires BOTH:
        1. RSI < threshold (oversold but not extreme)
           - Default: RSI < 35
           - Why 35? Oversold enough to be "on sale" but not so extreme as to indicate panic
           - RSI < 30 might indicate fundamental issues or panic selling
           
        2. Trend is UP (price > 200-day SMA)
           - Confirms stock is in long-term uptrend
           - We're buying dips in rising stocks, not catching falling knives
           - 200-day SMA filters out stocks in downtrends
        
        Args:
            techs: Dictionary with keys:
                - price: Current stock price
                - rsi: Relative Strength Index (0-100)
                - atr: Average True Range (volatility measure)
                - trend: "UP" or "DOWN" (based on price vs 200-day SMA)
                - sma: 200-day Simple Moving Average value
                
        Returns:
            True if technical conditions are met (RSI oversold + uptrend)
            False otherwise
            
        Note:
            This is a technical filter only. AI validation (score >= 7) is still required
            for actual trade entry. Both conditions must pass.
        """
        rsi_threshold = self.config.get("rsi_threshold", 35)
        rsi_value = techs.get('rsi', 50)
        trend_value = techs.get('trend')
        
        # Both conditions must be true:
        # 1. RSI is oversold (below threshold) - stock is "on sale"
        # 2. Trend is UP - we're buying dips in rising stocks, not catching falling knives
        return rsi_value < rsi_threshold and trend_value == "UP"
    
    def get_ai_prompt(self) -> str:
        """Get AI prompt focused on balanced low analysis with actionable insights."""
        return (
            "You are the Eye of Sauron, watching the markets with unwavering focus. "
            "Your current focus: identifying balanced lows with upside potential for SWING TRADING.\n\n"
            "SWING TRADING CONTEXT: This is for multi-day to multi-week positions, not day trading. "
            "We use daily bars and hold until stop loss or take profit.\n\n"
            "A 'balanced low' means:\n"
            "- Stock is oversold (RSI < 35) but NOT in freefall\n"
            "- Stock is in an UPTREND (price > 200-day moving average)\n"
            "- News is neutral or positive (no catastrophic events)\n"
            "- There is clear upside potential for a swing move (days to weeks)\n\n"
            "Score 0-10 based on:\n"
            "- How balanced the low is (not too extreme, not too mild)\n"
            "- Strength of uptrend (stronger trend = higher score)\n"
            "- News sentiment (positive = higher, catastrophic = 0)\n"
            "- Upside potential for swing trading (more room for multi-day move = higher)\n\n"
            "VETO (score 0) if: fraud, bankruptcy, major scandal, or downtrend.\n"
            "BUY (score 7-10) if: balanced low + uptrend + good news + swing trading upside potential.\n"
            "WAIT (score 4-6) if: close but not quite balanced enough for swing entry.\n\n"
            "IMPORTANT: Your 'reason' field must be:\n"
            "- A clear, actionable insight (1-2 sentences max)\n"
            "- Focus on swing trading opportunity (multi-day to multi-week potential)\n"
            "- Be specific: mention RSI level, trend strength, news impact, or swing setup\n"
            "- Use swing trading language: 'oversold bounce', 'uptrend pullback', 'swing entry', 'multi-day move'\n"
            "- Examples: 'Oversold bounce setup in strong uptrend - good swing entry' or 'RSI 32 with positive news suggests multi-day recovery potential'\n"
        )
    
    def get_name(self) -> str:
        """Get strategy name."""
        return "Balanced Low"
    
    def get_description(self) -> str:
        """Get strategy description."""
        return "Buy stocks when they reach a balanced 'low' - oversold but in uptrend with upside potential"
    
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return {
            **self.config,
            "name": self.get_name(),
            "description": self.get_description(),
            "color": "green"  # Green for buy opportunities
        }
    
    def get_rsi_threshold(self) -> float:
        """Get RSI threshold for this strategy.
        
        Returns:
            Maximum RSI value for entry (default: 35)
            RSI < this value = oversold condition ("balanced low")
        """
        return self.config.get("rsi_threshold", 35)
    
    def get_ai_score_required(self) -> int:
        """Get minimum AI score required for trade.
        
        Returns:
            Minimum AI score (0-10) for trade entry (default: 7)
            Score 7-10 = good swing opportunity
            Score < 7 = wait for better setup
        """
        return self.config.get("ai_score_required", 7)
    
    def get_risk_per_trade(self) -> float:
        """Get risk per trade in dollars.
        
        This is used for position sizing calculation:
        - Position size = risk_amount / stop_distance
        - Stop distance = 2 × ATR (volatility-based)
        - Ensures consistent risk regardless of stock price
        
        Returns:
            Dollars risked per trade (default: 50.0)
        """
        return self.config.get("risk_per_trade", 50.0)
    
    def get_max_capital(self) -> float:
        """Get maximum capital to deploy across all positions.
        
        This limits total exposure and prevents over-leveraging.
        
        Returns:
            Maximum capital to deploy (default: 5000.0)
        """
        return self.config.get("max_capital", 5000.0)
