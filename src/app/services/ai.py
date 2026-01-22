"""AI service for market analysis."""
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from mdb_engine.observability import get_logger
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
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
    action: str = Field(description="One of: 'BUY' or 'NOT BUY'. NOT BUY means avoid buying this stock (not a sell signal - sells come from positions, stop loss, or take profit).")
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
                strategy_prompt: str, strategy_config: Optional[Dict[str, Any]] = None,
                cnn_data: Optional[Dict[str, Any]] = None) -> TradeVerdict:
        """Analyze a trading opportunity using strategy-specific prompt.
        
        Args:
            ticker: Stock symbol
            techs: Dictionary with technical indicators (price, rsi, atr, trend, sma)
            headlines: Recent news headlines
            strategy_prompt: Strategy-specific system prompt
            strategy_config: Optional dictionary with strategy parameters (rsi_threshold, rsi_min, sma_proximity_pct, ai_score_required)
            cnn_data: Optional CNN market data (analyst ratings, price targets, financials)
            
        Returns:
            TradeVerdict with score, action, reason, and risk_level
        """
        price = techs.get('price', 0)
        rsi = techs.get('rsi', 50)
        atr = techs.get('atr', 0)
        trend = techs.get('trend', 'UNKNOWN')
        
        context_parts = [
            f"Ticker: {ticker} | Price: ${price:.2f}",
            f"RSI: {rsi:.1f} (oversold if < 35, extreme oversold if < 20)",
            f"Trend: {trend} (UP = uptrend, DOWN = downtrend)",
            f"ATR: ${atr:.2f} (volatility measure)"
        ]
        
        # Add CNN market data if available (analyst ratings, price targets)
        if cnn_data:
            cnn_parts = []
            
            # Analyst ratings
            ratings = cnn_data.get("analyst_ratings", {})
            if ratings:
                buy_pct = ratings.get("buy", 0)
                hold_pct = ratings.get("hold", 0)
                sell_pct = ratings.get("sell", 0)
                if buy_pct or hold_pct or sell_pct:
                    cnn_parts.append(f"\nANALYST CONSENSUS (CNN Markets):")
                    if buy_pct:
                        cnn_parts.append(f"- Buy: {buy_pct}%")
                    if hold_pct:
                        cnn_parts.append(f"- Hold: {hold_pct}%")
                    if sell_pct:
                        cnn_parts.append(f"- Sell: {sell_pct}%")
                    if buy_pct >= 60:
                        cnn_parts.append(f"  → Strong Buy consensus ({buy_pct}% buy) - bullish signal")
                    elif buy_pct >= 40:
                        cnn_parts.append(f"  → Moderate Buy consensus ({buy_pct}% buy)")
            
            # Price targets
            targets = cnn_data.get("price_targets", {})
            if targets and price:
                cnn_parts.append(f"\nPRICE TARGETS (CNN Markets):")
                if targets.get("high"):
                    high_target = targets["high"]
                    upside_high = ((high_target - price) / price) * 100
                    cnn_parts.append(f"- High: ${high_target:.2f} ({upside_high:+.1f}% upside)")
                if targets.get("median"):
                    median_target = targets["median"]
                    upside_median = ((median_target - price) / price) * 100
                    cnn_parts.append(f"- Median: ${median_target:.2f} ({upside_median:+.1f}% upside)")
                if targets.get("low"):
                    low_target = targets["low"]
                    upside_low = ((low_target - price) / price) * 100
                    cnn_parts.append(f"- Low: ${targets['low']:.2f} ({upside_low:+.1f}% upside)")
                
                # Highlight significant upside potential
                if targets.get("median") and ((targets["median"] - price) / price) * 100 > 20:
                    cnn_parts.append(f"  → Significant upside potential: median target {((targets['median'] - price) / price) * 100:.1f}% above current price")
            
            # Financial metrics
            financials = cnn_data.get("financials", {})
            if financials:
                if financials.get("revenue"):
                    revenue_b = financials["revenue"] / 1_000_000_000
                    cnn_parts.append(f"\nFINANCIALS (CNN Markets):")
                    cnn_parts.append(f"- Revenue: ${revenue_b:.2f}B")
                if financials.get("net_income"):
                    income_b = financials["net_income"] / 1_000_000_000
                    cnn_parts.append(f"- Net Income: ${income_b:.2f}B")
                if financials.get("eps"):
                    cnn_parts.append(f"- EPS: ${financials['eps']:.2f}")
            
            if cnn_parts:
                context_parts.extend(cnn_parts)
        
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
        
        # Add SMA proximity calculation if available
        sma_val = techs.get('sma') or techs.get('sma_200')
        if sma_val and price:
            price_vs_sma_pct = ((price - sma_val) / sma_val) * 100
            context_parts.append(f"Price vs SMA-200: {price_vs_sma_pct:.2f}% ({'above' if price_vs_sma_pct > 0 else 'below'} SMA-200)")
        
        context_parts.append(f"\nRecent News Headlines:\n{headlines}")
        context_parts.append("\nAnalyze this opportunity according to the strategy focus. Use the strategy configuration parameters to weight your analysis.")
        
        # Include strategy config in context if provided
        if strategy_config:
            config_context = [
                "\nSTRATEGY CONFIGURATION (use these to weight your analysis):",
                f"- RSI Threshold: {strategy_config.get('rsi_threshold', 'N/A')}",
                f"- RSI Minimum: {strategy_config.get('rsi_min', 'N/A')}",
                f"- SMA Proximity: {strategy_config.get('sma_proximity_pct', 'N/A')}%",
                f"- AI Score Required: {strategy_config.get('ai_score_required', 'N/A')}/10"
            ]
            context_parts.extend(config_context)
        
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
                "- ai_score_required (0-10): Minimum AI score. Higher = more conservative. Typical: 6-9\n\n"
                "INTERPRET THE GOAL:\n"
                "- 'Conservative' / 'Safe' / 'Low risk' → Lower RSI threshold (30), Higher AI score (8-9)\n"
                "- 'Aggressive' / 'More opportunities' → Higher RSI threshold (40), Lower AI score (6-7)\n"
                "- 'Balanced' / 'Moderate' → Middle values (RSI 35, Score 7)\n\n"
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


_ai_engine = None
try:
    _ai_engine = EyeAI()
except Exception as e:
    logger.warning(f"Azure OpenAI configuration missing. FLUX cannot analyze: {e}")
