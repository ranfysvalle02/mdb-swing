"""Application configuration."""
import os
from typing import List, Dict, Any

# Environment Variables
SYMBOLS_ENV = os.getenv("TARGET_SYMBOLS", "NVDA,AMD,MSFT,GOOGL,AMZN,TSLA,COIN,AAPL")
SYMBOLS: List[str] = [s.strip() for s in SYMBOLS_ENV.split(',')]
MAX_CAPITAL_DEPLOYED = float(os.getenv("MAX_CAPITAL_DEPLOYED", "5000.00"))

# Strategy Presets
STRATEGY_PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "name": "Conservative",
        "description": "Low risk, high confidence trades only",
        "rsi_threshold": 30,
        "ai_score_required": 9,
        "risk_per_trade": 25.0,
        "max_capital": 2500.0,
        "color": "blue"
    },
    "moderate": {
        "name": "Moderate",
        "description": "Balanced risk/reward (default)",
        "rsi_threshold": 35,
        "ai_score_required": 8,
        "risk_per_trade": 50.0,
        "max_capital": 5000.0,
        "color": "yellow"
    },
    "aggressive": {
        "name": "Aggressive",
        "description": "Higher risk, more opportunities",
        "rsi_threshold": 40,
        "ai_score_required": 7,
        "risk_per_trade": 100.0,
        "max_capital": 10000.0,
        "color": "red"
    }
}

# Current strategy preset (default: moderate)
CURRENT_PRESET = os.getenv("STRATEGY_PRESET", "moderate")
STRATEGY_CONFIG = STRATEGY_PRESETS.get(CURRENT_PRESET, STRATEGY_PRESETS["moderate"])

# Alpaca Configuration
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
# Remove /v2 suffix if present (api_version is handled separately)
if ALPACA_URL and ALPACA_URL.endswith('/v2'):
    ALPACA_URL = ALPACA_URL[:-3]

# MongoDB Configuration
# Use service name 'atlas-local' when running in Docker, 'localhost' for local dev
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_USER = os.getenv("MONGO_USER", "admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "secret")
# Allow override via MONGO_URI, or construct from components
if os.getenv("MONGO_URI"):
    MONGO_URI = os.getenv("MONGO_URI")
else:
    # Construct URI with authentication for Atlas Local
    MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:27017/"
MONGO_DB = os.getenv("MONGO_DB", "sauron_eye")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o")

# Firecrawl Configuration
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
