"""Example Strategy - Template for creating new strategies.

This is an example strategy that demonstrates how to create a new pluggable strategy.
Copy this file and modify it to create your own strategy.

Convention:
- File name: {strategy_name}.py (e.g., momentum_breakout.py)
- Class name: {Name}Strategy (e.g., MomentumBreakoutStrategy)
- Must inherit from Strategy base class
- Must implement all abstract methods
"""
from typing import Dict, Any
from .base import Strategy


class ExampleStrategy(Strategy):
    """Example strategy - demonstrates the pluggable strategy pattern.
    
    This strategy is automatically discovered by the StrategyRegistry.
    To use it, set CURRENT_STRATEGY=example_strategy environment variable.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Example strategy with configuration.
        
        Args:
            config: Strategy configuration dict. If None, uses defaults.
        """
        self.config = config or {
            "rsi_threshold": 40,  # Example threshold
            "ai_score_required": 6,  # Example score requirement
            "risk_per_trade": 50.0,
            "max_capital": 5000.0,
        }
    
    def check_technical_signal(self, techs: Dict[str, Any]) -> bool:
        """Check if technical conditions indicate an opportunity.
        
        Args:
            techs: Dictionary with keys: price, rsi, atr, trend, sma
            
        Returns:
            True if technical conditions are met
        """
        rsi_threshold = self.config.get("rsi_threshold", 40)
        return techs.get('rsi', 50) < rsi_threshold and techs.get('trend') == "UP"
    
    def get_ai_prompt(self) -> str:
        """Get AI prompt focused on example strategy analysis."""
        return (
            "You are the Eye of Sauron, watching the markets with unwavering focus. "
            "Your current focus: identifying example opportunities for SWING TRADING.\n\n"
            "SWING TRADING CONTEXT: This is for multi-day to multi-week positions, not day trading. "
            "We use daily bars and hold until stop loss or take profit.\n\n"
            "Example strategy criteria:\n"
            "- RSI < 40 (oversold but not extreme)\n"
            "- Stock is in an UPTREND (price > 200-day SMA)\n"
            "- News is neutral or positive\n"
            "- There is clear upside potential for a swing move\n\n"
            "Score 0-10 based on opportunity quality.\n\n"
            "IMPORTANT: Your 'reason' field must be:\n"
            "- A clear, actionable insight (1-2 sentences max)\n"
            "- Focus on swing trading opportunity\n"
            "- Use swing trading language: 'oversold bounce', 'uptrend pullback', 'swing entry'\n"
        )
    
    def get_name(self) -> str:
        """Get strategy name."""
        return "Example Strategy"
    
    def get_description(self) -> str:
        """Get strategy description."""
        return "Example strategy demonstrating the pluggable pattern"
    
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return {
            **self.config,
            "name": self.get_name(),
            "description": self.get_description(),
            "color": "blue"  # Color for UI display
        }
    
    def get_rsi_threshold(self) -> float:
        """Get RSI threshold for this strategy."""
        return self.config.get("rsi_threshold", 40)
    
    def get_ai_score_required(self) -> int:
        """Get minimum AI score required for trade."""
        return self.config.get("ai_score_required", 6)
    
    def get_risk_per_trade(self) -> float:
        """Get risk per trade in dollars."""
        return self.config.get("risk_per_trade", 50.0)
    
    def get_max_capital(self) -> float:
        """Get maximum capital to deploy."""
        return self.config.get("max_capital", 5000.0)
