"""AI service for market analysis - reusable AI engine for FLUX.

This module provides the EyeAI class, a strategy-agnostic AI analysis engine that works
with any trading strategy by accepting strategy-specific prompts. FLUX analyzes markets,
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
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from typing import List
from ..core.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_MODEL_NAME
)

logger = get_logger(__name__)


class TradeVerdict(BaseModel):
    """AI verdict for a trading opportunity with detailed insights."""
    score: int = Field(description="Bullish score 0-10.")
    action: str = Field(description="One of: 'BUY', 'WAIT', 'SELL_NOW'. Note: SELL_NOW means 'avoid buying' for stocks without positions.")
    reason: str = Field(description="Concise strategic reasoning (max 20 words).")
    risk_level: str = Field(description="Low, Medium, or High based on volatility.")
    key_factors: List[str] = Field(default_factory=list, description="List of 3-5 key factors supporting the bounce-back opportunity (e.g., 'RSI 28 in sweet spot', 'Price 2% above SMA-200 support', 'Earnings catalyst in 5 days'). Each factor should be a concise bullet point (max 15 words).")
    risks: List[str] = Field(default_factory=list, description="List of 2-4 key risk factors to watch (e.g., 'Earnings volatility risk', 'Market sentiment concerns', 'RSI near extreme oversold'). Each risk should be concise (max 15 words). If no significant risks, return empty list.")
    opportunities: List[str] = Field(default_factory=list, description="List of 2-3 potential opportunities or catalysts (e.g., 'Sector rotation tailwind', 'Upcoming product launch', 'Analyst upgrade potential'). Each opportunity should be concise (max 15 words). If no clear opportunities, return empty list.")
    catalyst: Optional[str] = Field(default=None, description="Primary catalyst driving the bounce-back opportunity (e.g., 'Earnings in 5 days', 'Sector rotation', 'Oversold bounce from support'). Max 20 words. If no clear catalyst, return null.")


class StrategyConfig(BaseModel):
    """AI-generated strategy configuration from goal description."""
    rsi_threshold: int = Field(description="Maximum RSI for entry (20-50, typical: 30-40). Lower = more conservative.")
    rsi_min: int = Field(description="Minimum RSI to avoid extreme oversold (15-25, typical: 20).")
    sma_proximity_pct: float = Field(description="Maximum percentage above SMA-200 for entry (0-5, typical: 3.0). Lower = closer to support (more conservative).")
    ai_score_required: int = Field(description="Minimum AI score required (0-10, typical: 6-9). Higher = more conservative.")
    risk_per_trade: float = Field(description="Dollar amount to risk per trade (25-200, typical: 50-100).")
    max_capital: float = Field(description="Maximum capital to deploy (1000-50000, typical: 2500-10000).")
    reasoning: str = Field(description="Brief explanation of why these parameters match the goal.")


class EyeAI:
    """Reusable AI analysis engine for FLUX.
    
    Works with any strategy by accepting strategy-specific prompts.
    FLUX analyzes, strategies define what it seeks.
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
        
        # Build context string
        context_parts = [
            f"Ticker: {ticker} | Price: ${price:.2f}",
            f"RSI: {rsi:.1f} (oversold if < 35)",
            f"Trend: {trend} (UP = uptrend, DOWN = downtrend)",
            f"ATR: ${atr:.2f} (volatility measure)"
        ]
        
        # Add enrichment data if available
        analyst_rec = techs.get("analyst_recommendation")
        if analyst_rec:
            context_parts.append(f"Analyst Recommendation: {analyst_rec}")
        
        pe_ratio = techs.get("pe_ratio")
        if pe_ratio:
            context_parts.append(f"P/E Ratio: {pe_ratio:.1f}")
        
        earnings_in_days = techs.get("earnings_in_days")
        if earnings_in_days is not None:
            if earnings_in_days <= 3:
                context_parts.append(f"⚠️ Earnings in {earnings_in_days} day(s) - HIGH VOLATILITY RISK")
            else:
                context_parts.append(f"Earnings in {earnings_in_days} days")
        
        eps_beats = techs.get("eps_beats_count")
        if eps_beats is not None:
            eps_beats_pct = techs.get("eps_beats_pct", 0)
            context_parts.append(f"EPS Quality: {eps_beats}/4 quarters beat estimates ({eps_beats_pct:.0f}%)")
        
        market_cap = techs.get("market_cap")
        if market_cap:
            market_cap_m = market_cap / 1_000_000
            context_parts.append(f"Market Cap: ${market_cap_m:.0f}M")
        
        context_parts.append(f"\nRecent News Headlines:\n{headlines}")
        context_parts.append("\nAnalyze this opportunity according to the strategy focus.")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", strategy_prompt),
            ("human", "\n".join(context_parts))
        ])
        chain = prompt | self.llm.with_structured_output(TradeVerdict)
        return chain.invoke({})
    
    def generate_strategy_config(self, goal: str, budget: Optional[float] = None) -> StrategyConfig:
        """Generate strategy configuration parameters from a goal description.
        
        Uses AI to interpret the user's trading goal and generate appropriate
        strategy parameters (RSI thresholds, risk settings, etc.).
        
        Args:
            goal: User's trading goal description (e.g., "I want conservative swing trades with high probability")
            budget: Optional budget amount to help inform capital allocation
            
        Returns:
            StrategyConfig with recommended parameters
        """
        budget_context = ""
        if budget:
            budget_context = f"\n\nUser's available budget: ${budget:,.2f}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a trading strategy configuration expert. Your job is to analyze a user's trading goal "
                "and generate appropriate strategy parameters for a swing trading system.\n\n"
                "STRATEGY CONTEXT:\n"
                "- This is a 'Balanced Low' swing trading strategy\n"
                "- It buys stocks that are oversold (low RSI) but have high probability to bounce back\n"
                "- Positions are held for days to weeks until stop loss or take profit\n"
                "- Uses RSI (Relative Strength Index) to identify oversold conditions\n"
                "- Uses AI scoring (0-10) to assess bounce-back probability\n\n"
                "PARAMETER GUIDELINES:\n"
                "- rsi_threshold (20-50): Maximum RSI for entry. Lower = more conservative (fewer signals). Typical: 30-40\n"
                "- rsi_min (15-25): Minimum RSI to avoid extreme oversold. Typical: 20\n"
                "- ai_score_required (0-10): Minimum AI score. Higher = more conservative. Typical: 6-9\n"
                "- risk_per_trade (25-200): Dollar risk per trade. Typical: 50-100\n"
                "- max_capital (1000-50000): Max capital to deploy. Should scale with budget. Typical: 2500-10000\n\n"
                "INTERPRET THE GOAL:\n"
                "- 'Conservative' / 'Safe' / 'Low risk' → Lower RSI threshold (30), Higher AI score (8-9), Lower risk per trade\n"
                "- 'Aggressive' / 'More opportunities' → Higher RSI threshold (40), Lower AI score (6-7), Higher risk per trade\n"
                "- 'Balanced' / 'Moderate' → Middle values (RSI 35, Score 7, Risk $50)\n"
                "- Budget considerations: If budget < $5000, use lower max_capital. If > $10000, can use higher max_capital\n\n"
                "Generate parameters that best match the user's goal and risk tolerance."
            )),
            ("human", (
                f"User's trading goal:\n{goal}{budget_context}\n\n"
                "Generate strategy parameters that align with this goal. Consider:\n"
                "- Risk tolerance (conservative vs aggressive)\n"
                "- Desired trade frequency (more vs fewer signals)\n"
                "- Capital constraints (if budget provided)\n"
                "- Overall trading style described"
            ))
        ])
        chain = prompt | self.llm.with_structured_output(StrategyConfig)
        return chain.invoke({})


# Module-level AI engine instance
# Initialized on module import, used by routes that need direct AI access
# Most code should use EyeAI instances created via Eye class instead
_ai_engine = None
try:
    _ai_engine = EyeAI()
except Exception as e:
    logger.warning(f"Azure OpenAI configuration missing. FLUX cannot analyze: {e}")
