"""AI service for market analysis - reusable AI engine for the Eye of Sauron.

This module provides the EyeAI class, a strategy-agnostic AI analysis engine that works
with any trading strategy by accepting strategy-specific prompts. The Eye watches markets,
strategies define what it seeks, and EyeAI provides the intelligence.

Key Features:
- Strategy-agnostic: Works with any strategy via custom prompts
- Azure OpenAI Integration: Uses GPT-4o for intelligent analysis
- Structured Output: Returns TradeVerdict with score, action, reason, risk_level
- Reusable: Single AI engine instance can be shared across strategies

MDB-Engine Pattern: Uses mdb-engine's observability for structured logging.
"""
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from mdb_engine.observability import get_logger
from typing import Dict, Any
from ..models import TradeVerdict
from ..core.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_MODEL_NAME
)

logger = get_logger(__name__)

class EyeAI:
    """Reusable AI analysis engine for the Eye of Sauron.
    
    Works with any strategy by accepting strategy-specific prompts.
    The Eye watches, strategies define what it seeks.
    """
    
    def __init__(self):
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required")
        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            model_name=AZURE_OPENAI_MODEL_NAME,
            api_version="2024-08-01-preview",  # Updated for structured output support
            temperature=0.2
        )
        
    def analyze(self, ticker: str, techs: Dict[str, Any], headlines: str, 
                strategy_prompt: str) -> TradeVerdict:
        """Analyze a trading opportunity using strategy-specific prompt.
        
        Args:
            ticker: Stock symbol
            techs: Dictionary with technical indicators (price, rsi, atr, trend, sma)
            headlines: Recent news headlines
            strategy_prompt: Strategy-specific system prompt
            
        Returns:
            TradeVerdict with score, action, reason, and risk_level
        """
        price = techs.get('price', 0)
        rsi = techs.get('rsi', 50)
        atr = techs.get('atr', 0)
        trend = techs.get('trend', 'UNKNOWN')
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", strategy_prompt),
            ("human", (
                f"Ticker: {ticker} | Price: ${price:.2f}\n"
                f"RSI: {rsi:.1f} (oversold if < 35)\n"
                f"Trend: {trend} (UP = uptrend, DOWN = downtrend)\n"
                f"ATR: ${atr:.2f} (volatility measure)\n"
                f"Recent News Headlines:\n{headlines}\n\n"
                f"Analyze this opportunity according to the strategy focus."
            ))
        ])
        chain = prompt | self.llm.with_structured_output(TradeVerdict)
        return chain.invoke({})


# Module-level AI engine instance
# Initialized on module import, used by routes that need direct AI access
# Most code should use EyeAI instances created via Eye class instead
_ai_engine = None
try:
    _ai_engine = EyeAI()
except Exception as e:
    logger.warning(f"Azure OpenAI configuration missing. The Eye cannot see: {e}")
