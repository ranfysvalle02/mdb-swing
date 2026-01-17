"""Base strategy interface for the Eye of Sauron."""
from abc import ABC, abstractmethod
from typing import Dict, Any


class Strategy(ABC):
    """Base class for Eye strategies - pluggable lenses that focus the Eye's energy.
    
    The Eye of Sauron watches markets continuously. Strategies define what patterns
    the Eye seeks - they are lenses that focus the Eye's watchful gaze.
    """
    
    @abstractmethod
    def check_technical_signal(self, techs: Dict[str, Any]) -> bool:
        """Check if technical conditions are met for this strategy.
        
        Args:
            techs: Dictionary containing technical indicators (price, rsi, atr, trend, sma)
            
        Returns:
            True if technical conditions are met, False otherwise
        """
        pass
    
    @abstractmethod
    def get_ai_prompt(self) -> str:
        """Get strategy-specific AI analysis prompt.
        
        Returns:
            System prompt that guides AI analysis for this strategy
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name.
        
        Returns:
            Human-readable strategy name
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration.
        
        Returns:
            Dictionary containing strategy-specific configuration (thresholds, risk, etc.)
        """
        return {}
    
    def get_description(self) -> str:
        """Get strategy description.
        
        Returns:
            Human-readable description of what this strategy seeks
        """
        return f"{self.get_name()} strategy"
