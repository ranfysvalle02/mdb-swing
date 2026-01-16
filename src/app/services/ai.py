"""AI service for market analysis."""
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from mdb_engine.observability import get_logger
from ..models import TradeVerdict
from ..core.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_MODEL_NAME
)

logger = get_logger(__name__)

class SauronEye:
    """The watching eye that analyzes market signals for swing trading opportunities."""
    
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
        
    def analyze(self, ticker: str, price: float, rsi: float, atr: float, trend: str, headlines: str) -> TradeVerdict:
        """Analyze a trading opportunity using AI."""
        system_prompt = (
            "You are the Eye of Sauron, watching the markets with unwavering focus. "
            "Analyze Technicals + Sentiment to identify swing trading opportunities. "
            "Technicals: RSI < 30 is Oversold (Buy zone), > 70 Overbought. "
            "ATR indicates volatility. If news is catastrophic (fraud/bankruptcy), VETO the trade (Score 0). "
            "Your purpose: Extract capital from the markets using precise swing strategies."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", (
                f"Ticker: {ticker} | Price: ${price}\n"
                f"Technicals: RSI={rsi}, ATR={atr}, Trend={trend}\n"
                f"News:\n{headlines}"
            ))
        ])
        chain = prompt | self.llm.with_structured_output(TradeVerdict)
        return chain.invoke({})

# Initialize the Eye
eye = None
try:
    eye = SauronEye()
except Exception as e:
    logger.warning(f"Azure OpenAI configuration missing. The Eye cannot see: {e}")
