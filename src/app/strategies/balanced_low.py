"""Balanced Low Strategy - Focus: Buy stocks at balanced lows with upside potential."""
from typing import Dict, Any
from .base import Strategy


class BalancedLowStrategy(Strategy):
    """The Eye focuses on balanced lows - oversold but stable, in uptrend, with upside potential.
    
    This strategy seeks stocks that are:
    - At a balanced low (RSI < 35, oversold but not extreme)
    - In an uptrend (price > 200-day SMA)
    - Have clear upside potential (validated by AI)
    - News is neutral/positive (no catastrophic events)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Balanced Low strategy with configuration.
        
        Args:
            config: Strategy configuration dict. If None, uses defaults.
        """
        self.config = config or {
            "rsi_threshold": 35,
            "ai_score_required": 7,
            "risk_per_trade": 50.0,
            "max_capital": 5000.0,
        }
    
    def check_technical_signal(self, techs: Dict[str, Any]) -> bool:
        """Check if technical conditions indicate a balanced low opportunity.
        
        Balanced low signal: RSI < threshold AND trend is UP
        
        Args:
            techs: Dictionary with keys: price, rsi, atr, trend, sma
            
        Returns:
            True if technical conditions are met
        """
        rsi_threshold = self.config.get("rsi_threshold", 35)
        return techs.get('rsi', 50) < rsi_threshold and techs.get('trend') == "UP"
    
    def get_ai_prompt(self) -> str:
        """Get AI prompt focused on balanced low analysis."""
        return (
            "You are the Eye of Sauron, watching the markets with unwavering focus. "
            "Your current focus: identifying balanced lows with upside potential.\n\n"
            "A 'balanced low' means:\n"
            "- Stock is oversold (RSI < 35) but NOT in freefall\n"
            "- Stock is in an UPTREND (price > 200-day moving average)\n"
            "- News is neutral or positive (no catastrophic events)\n"
            "- There is clear upside potential (room to grow)\n\n"
            "Score 0-10 based on:\n"
            "- How balanced the low is (not too extreme, not too mild)\n"
            "- Strength of uptrend (stronger trend = higher score)\n"
            "- News sentiment (positive = higher, catastrophic = 0)\n"
            "- Upside potential (more room to grow = higher)\n\n"
            "VETO (score 0) if: fraud, bankruptcy, major scandal, or downtrend.\n"
            "BUY (score 7-10) if: balanced low + uptrend + good news + upside potential.\n"
            "WAIT (score 4-6) if: close but not quite balanced enough.\n"
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
        """Get RSI threshold for this strategy."""
        return self.config.get("rsi_threshold", 35)
    
    def get_ai_score_required(self) -> int:
        """Get minimum AI score required for trade."""
        return self.config.get("ai_score_required", 7)
    
    def get_risk_per_trade(self) -> float:
        """Get risk per trade in dollars."""
        return self.config.get("risk_per_trade", 50.0)
    
    def get_max_capital(self) -> float:
        """Get maximum capital to deploy."""
        return self.config.get("max_capital", 5000.0)
