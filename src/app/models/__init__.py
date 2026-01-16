"""Pydantic models."""
from pydantic import BaseModel, Field

class TradeVerdict(BaseModel):
    """AI verdict for a trading opportunity."""
    score: int = Field(description="Bullish score 0-10.")
    action: str = Field(description="One of: 'BUY', 'WAIT', 'SELL_NOW'.")
    reason: str = Field(description="Concise strategic reasoning (max 20 words).")
    risk_level: str = Field(description="Low, Medium, or High based on volatility.")
