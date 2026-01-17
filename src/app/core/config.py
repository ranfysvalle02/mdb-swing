"""Application configuration.

This module manages all application configuration including:
- Environment variable loading
- Strategy configuration (with database override support)
- API credentials (Alpaca, Azure OpenAI, MongoDB)
- Configuration priority: Database > Environment Variables > Defaults

Configuration Priority:
1. Database (strategy_config collection) - highest priority, loaded in async routes
2. Environment variables - loaded at module import
3. Default values - hardcoded fallbacks

MDB-Engine Pattern: Configuration values are used throughout the app via imports.
"""
import os
from typing import List, Dict, Any, Optional
from mdb_engine.observability import get_logger

logger = get_logger(__name__)

# Environment Variables
SYMBOLS_ENV = os.getenv("TARGET_SYMBOLS", "NVDA,AMD,MSFT,GOOGL,AMZN,TSLA,COIN,AAPL")
SYMBOLS: List[str] = [s.strip() for s in SYMBOLS_ENV.split(',')]
MAX_CAPITAL_DEPLOYED = float(os.getenv("MAX_CAPITAL_DEPLOYED", "5000.00"))

# Strategy Selection
# PRIMARY STRATEGY: Balanced Low (ONLY active strategy)
# Other strategies exist as pseudocode placeholders for future implementation
# Convention: Set via environment variable, defaults to Balanced Low
CURRENT_STRATEGY = os.getenv("CURRENT_STRATEGY", "balanced_low")

# Strategy Configuration (legacy - strategies now define their own configs)
# Kept for backward compatibility and default values
# NOTE: Balanced Low is the PRIMARY and ONLY active strategy
STRATEGY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "balanced_low": {
        "name": "Balanced Low",
        "description": "Buy stocks when they reach a balanced 'low' - oversold but in uptrend with upside potential",
        "rsi_threshold": 35,  # Oversold but not extreme (balanced)
        "ai_score_required": 7,  # Good opportunity, not perfect
        "risk_per_trade": 50.0,  # $50 risk per trade
        "max_capital": 5000.0,  # Max capital deployed
        "color": "green"  # Green for "buy" opportunities
    }
}

# Current strategy config (for backward compatibility and easy access)
# PRIMARY STRATEGY: Balanced Low (only active strategy)
STRATEGY_CONFIG = STRATEGY_CONFIGS.get(CURRENT_STRATEGY, STRATEGY_CONFIGS["balanced_low"])


async def get_strategy_from_db(db) -> Optional[Dict[str, Any]]:
    """Load active strategy configuration from MongoDB.
    
    Args:
        db: MongoDB database instance (scoped via mdb-engine)
        
    Returns:
        Dictionary with strategy config or None if not found
    """
    try:
        active_config = await db.strategy_config.find_one({"active": True})
        if active_config:
            # Remove MongoDB-specific fields and convert to dict
            config = {
                "rsi_threshold": active_config.get("rsi_threshold", 35),
                "ai_score_required": active_config.get("ai_score_required", 7),
                "risk_per_trade": active_config.get("risk_per_trade", 50.0),
                "max_capital": active_config.get("max_capital", 5000.0),
                "name": active_config.get("name", "Balanced Low"),
                "description": active_config.get("description", "Buy stocks at balanced lows"),
                "color": active_config.get("color", "green"),
                "preset": active_config.get("preset", "Custom")
            }
            return config
        return None
    except Exception as e:
        logger.warning(f"Could not load strategy from database: {e}")
        return None

def get_strategy_instance(db=None, strategy_name: Optional[str] = None):
    """Get the current strategy instance.
    
    Args:
        db: Optional MongoDB database instance (not used in sync context)
        strategy_name: Optional strategy name override (defaults to CURRENT_STRATEGY env var)
    
    Returns:
        Strategy instance based on CURRENT_STRATEGY setting or provided name
    """
    # Original behavior - direct import (most reliable, no registry dependency)
    from ..strategies import BalancedLowStrategy
    
    # In sync context, always use env vars/defaults
    # DB config is loaded in async routes via get_strategy_from_db()
    if CURRENT_STRATEGY == "balanced_low" or not strategy_name:
        config = STRATEGY_CONFIGS.get("balanced_low", {})
    else:
        logger.warning(f"Unknown strategy '{strategy_name or CURRENT_STRATEGY}', defaulting to 'balanced_low'")
        config = STRATEGY_CONFIGS.get("balanced_low", {})
    
    return BalancedLowStrategy(config=config)

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

# Firecrawl removed - using headlines only for balanced low strategy
