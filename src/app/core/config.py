"""Application configuration."""
import os
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from mdb_engine.observability import get_logger

logger = get_logger(__name__)

# Timezone handling for 6pm ET final scan cutoff
try:
    from zoneinfo import ZoneInfo
    HAS_ZONEINFO = True
except ImportError:
    # Python < 3.9 fallback
    try:
        import pytz
        ZoneInfo = pytz.timezone
        HAS_ZONEINFO = True
    except ImportError:
        logger.warning("No timezone library available - using UTC for final scan time")
        ZoneInfo = None
        HAS_ZONEINFO = False

SYMBOLS_ENV = os.getenv("TARGET_SYMBOLS", "NVDA,AMD,MSFT,GOOGL,AMZN")
SYMBOLS: List[str] = [s.strip().upper() for s in SYMBOLS_ENV.split(',') if s.strip()][:5]

CURRENT_STRATEGY = os.getenv("CURRENT_STRATEGY", "balanced_low")

STRATEGY_CONFIG: Dict[str, Any] = {
    "name": "Balanced Low",
    "description": "Swing trading: oversold stocks in uptrends. RSI 20-35, price > SMA-200, near support (<3%), clean news.",
    "rsi_threshold": 35,
    "rsi_min": 20,
    "sma_proximity_pct": 3.0,
    "ai_score_required": 7,
    "color": "green",
    "pe_ratio_max": float(os.getenv("PE_RATIO_MAX", "50.0")),
    "market_cap_min": float(os.getenv("MARKET_CAP_MIN", "300000000")),
    "earnings_days_min": int(os.getenv("EARNINGS_DAYS_MIN", "3")),
}


async def get_strategy_from_db(db) -> Optional[Dict[str, Any]]:
    """Load active strategy configuration from MongoDB."""
    try:
        active_config = await db.strategy_config.find_one({"active": True})
        if active_config:
            # Remove MongoDB-specific fields and convert to dict
            config = {
                "rsi_threshold": active_config.get("rsi_threshold", 35),
                "rsi_min": active_config.get("rsi_min", 20),
                "sma_proximity_pct": active_config.get("sma_proximity_pct", 3.0),
                "ai_score_required": active_config.get("ai_score_required", 7),
                "name": active_config.get("name", "Balanced Low"),
                "description": active_config.get("description", "Safety First swing trading - oversold stocks in uptrends with clean news"),
                "color": active_config.get("color", "green"),
                "preset": active_config.get("preset", "Custom"),
            }
            return config
        return None
    except Exception as e:
        logger.warning(f"Could not load strategy from database: {e}")
        return None

async def get_strategy_config(db=None, budget: Optional[float] = None) -> Dict[str, Any]:
    """Get current strategy configuration."""
    if budget:
        from ..strategies.balanced_low import get_budget_preset
        return get_budget_preset(budget)
    
    if db:
        try:
            db_config = await get_strategy_from_db(db)
            if db_config:
                return {
                    "rsi_threshold": db_config.get("rsi_threshold", 35),
                    "rsi_min": db_config.get("rsi_min", 20),
                    "sma_proximity_pct": db_config.get("sma_proximity_pct", 3.0),
                    "ai_min_score": db_config.get("ai_score_required", 7),
                    "ai_score_required": db_config.get("ai_score_required", 7),  # Alias for compatibility
                    "name": db_config.get("name", "Balanced Low"),
                    "description": db_config.get("description", "Safety First swing trading - oversold stocks in uptrends with clean news"),
                    "color": db_config.get("color", "green")
                }
        except Exception as e:
            logger.debug(f"Could not load config from DB: {e}")
    
    return {
        "rsi_threshold": STRATEGY_CONFIG.get("rsi_threshold", 35),
        "rsi_min": STRATEGY_CONFIG.get("rsi_min", 20),
        "sma_proximity_pct": STRATEGY_CONFIG.get("sma_proximity_pct", 3.0),
        "ai_min_score": STRATEGY_CONFIG.get("ai_score_required", 7),
        "ai_score_required": STRATEGY_CONFIG.get("ai_score_required", 7),  # Alias for compatibility
        "name": STRATEGY_CONFIG.get("name", "Balanced Low"),
        "description": STRATEGY_CONFIG.get("description", "Safety First swing trading - oversold stocks in uptrends with clean news"),
        "color": STRATEGY_CONFIG.get("color", "green")
    }

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
if ALPACA_URL and ALPACA_URL.endswith('/v2'):
    ALPACA_URL = ALPACA_URL[:-3]

MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_USER = os.getenv("MONGO_USER", "admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "secret")
if os.getenv("MONGO_URI"):
    MONGO_URI = os.getenv("MONGO_URI")
else:
    MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:27017/"
MONGO_DB = os.getenv("MONGO_DB", "sauron_eye")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o")

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_SEARCH_QUERY_TEMPLATE = os.getenv(
    "FIRECRAWL_SEARCH_QUERY_TEMPLATE",
    "{symbol} stock news earnings financial analysis market"
)


def is_after_final_scan_time() -> bool:
    """Check if current time is >= 6pm ET."""
    try:
        if HAS_ZONEINFO and ZoneInfo:
            et_tz = ZoneInfo("America/New_York")
            et_now = datetime.now(et_tz)
        else:
            et_now = datetime.utcnow()
            from datetime import timedelta
            et_now = et_now - timedelta(hours=4, minutes=30)
        return et_now.hour >= 18
    except Exception as e:
        logger.warning(f"Error checking final scan time: {e}, defaulting to False")
        return False


def get_today_date_et() -> str:
    """Get today's date in YYYY-MM-DD format in ET timezone."""
    try:
        if HAS_ZONEINFO and ZoneInfo:
            et_tz = ZoneInfo("America/New_York")
            et_now = datetime.now(et_tz)
        else:
            et_now = datetime.utcnow()
        return et_now.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"Error getting today's date in ET: {e}, using UTC")
        return datetime.utcnow().strftime("%Y-%m-%d")


def calculate_watchlist_config_hash(config: Dict[str, Any]) -> str:
    """Calculate hash of watchlist-relevant parameters."""
    relevant_params = {
        'rsi_threshold': config.get('rsi_threshold', 35),
        'rsi_min': config.get('rsi_min', 20),
        'sma_proximity_pct': config.get('sma_proximity_pct', 3.0),
        'earnings_days_min': config.get('earnings_days_min', 3),
        'pe_ratio_max': config.get('pe_ratio_max', 50.0),
        'market_cap_min': config.get('market_cap_min', 300_000_000),
    }
    param_str = json.dumps(relevant_params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


async def get_active_custom_strategy(db) -> Optional[Dict[str, Any]]:
    """Get the currently active custom strategy from MongoDB."""
    try:
        active_strategy = await db.custom_strategies.find_one({"is_active": True})
        if active_strategy:
            strategy = {
                "name": active_strategy.get("name"),
                "label": active_strategy.get("label"),
                "conditions": active_strategy.get("conditions", []),
                "description": active_strategy.get("description", ""),
                "created_at": active_strategy.get("created_at"),
                "updated_at": active_strategy.get("updated_at"),
            }
            return strategy
        return None
    except Exception as e:
        logger.warning(f"Could not load active custom strategy from database: {e}")
        return None


def calculate_custom_strategy_hash(strategy_config: Dict[str, Any]) -> str:
    """Calculate hash of custom strategy conditions for cache invalidation."""
    conditions = strategy_config.get('conditions', [])
    sorted_conditions = sorted(conditions, key=lambda x: x.get('metric', ''))
    param_str = json.dumps(sorted_conditions, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()
