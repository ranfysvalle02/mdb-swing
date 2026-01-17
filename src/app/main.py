"""Main application entry point.

MDB-Engine Showcase:
This file demonstrates mdb-engine's key features:

1. Automatic FastAPI App Creation:
   - engine.create_app() handles all boilerplate
   - Manifest-based configuration (CORS, auth, etc.)
   - Automatic lifecycle management

2. Declarative WebSocket Configuration:
   - WebSocket settings defined in manifest.json
   - register_message_handler() for message routing
   - Custom WebSocket handlers can override defaults

3. Observability:
   - get_logger() provides structured logging
   - Health checks via engine.get_health_status()
   - Metrics via engine.get_metrics()

4. Scoped Database Access:
   - engine.get_scoped_db(APP_SLUG) for database access
   - Automatic connection pooling and lifecycle management
"""
from pathlib import Path
from typing import Dict, Any
from mdb_engine.observability import get_logger
from mdb_engine.routing.websockets import register_message_handler

from fastapi.responses import HTMLResponse, JSONResponse
from .core.engine import engine, get_manifest_path, APP_SLUG
from .core.config import get_strategy_instance
from .api import routes

# MDB-Engine Pattern: Structured logging via get_logger()
# Provides consistent log formatting and context
logger = get_logger(__name__)


def register_websocket_message_handlers():
    """Register WebSocket message handlers using mdb-engine conveniences."""
    
    async def handle_realtime_message(websocket, message: Dict[str, Any]):
        """Handle incoming WebSocket messages for real-time stock analysis.
        
        This handler processes incoming client messages if needed.
        The actual streaming is handled by the WebSocket route handler.
        """
        # For now, we don't handle incoming messages - the client just connects
        # and receives streaming updates. If we need to handle commands from
        # the client, we can add that logic here.
        logger.debug(f"Received WebSocket message: {message}")
    
    # Register handler for the "realtime" WebSocket channel (defined in manifest)
    register_message_handler(APP_SLUG, "realtime", handle_realtime_message)


async def on_startup(app, engine, manifest):
    """Additional startup tasks beyond what create_app() handles.
    
    This callback runs after engine is fully initialized and routes are registered.
    """
    # Register WebSocket message handlers (for incoming messages if needed)
    register_websocket_message_handlers()
    
    logger.info("✅ Startup complete - WebSocket handler registered")
    
    # Log startup information
    strategy = get_strategy_instance()
    logger.info("👁️ SAURON'S EYE IS WATCHING...")
    logger.info(f"👉 Focus: {strategy.get_name()} - {strategy.get_description()}")
    logger.info("👉 UI: http://localhost:5000")
    logger.info("👉 The Eye scans on-demand via UI")


async def on_shutdown(app, engine, manifest=None):
    """Cleanup on shutdown.
    
    MDB-Engine Pattern: Shutdown callback signature matches mdb-engine's expectations.
    The manifest parameter is optional but may be passed by mdb-engine.
    """
    logger.info("👁️ The Eye closes... watching ceases...")


# MDB-Engine Pattern: Automatic FastAPI App Creation
# engine.create_app() handles all the boilerplate:
# - MongoDB connection initialization
# - Manifest loading and validation (indexes, WebSockets, CORS)
# - FastAPI app setup with proper middleware
# - Lifecycle callbacks (startup/shutdown)
# - Health checks and metrics endpoints
# 
# This replaces ~100+ lines of boilerplate code!
app = engine.create_app(
    slug=APP_SLUG,
    manifest=get_manifest_path(),  # Declarative config: indexes, WebSockets, CORS
    title="Sauron's Eye - Market Watcher",
    description="AI-Augmented Swing Trading Bot - Balanced Low Buy System",
    version="1.0.0",
    on_startup=on_startup,      # Custom startup logic
    on_shutdown=on_shutdown,    # Custom cleanup logic
)

# MDB-Engine Pattern: Custom WebSocket Handler
# WebSocket configuration is declared in manifest.json, but we can override
# the handler for custom streaming logic (real-time stock analysis)
from fastapi import WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time stock analysis updates.
    
    MDB-Engine Pattern: Custom WebSocket handler with scoped database access.
    - WebSocket path configured in manifest.json
    - Uses engine.get_scoped_db() for database access
    - Connection lifecycle managed automatically
    
    This handler streams stock analysis data to connected clients.
    We register this here to override mdb-engine's default handler.
    """
    logger.info("🔌 [MAIN] Custom WebSocket handler called for /ws")
    try:
        # MDB-Engine: Get scoped database - connection managed automatically
        db = engine.get_scoped_db(APP_SLUG)
        await routes._get_trending_stocks_with_analysis_streaming(websocket, db)
    except Exception as e:
        logger.error(f"❌ [MAIN] WebSocket handler error: {e}", exc_info=True)
        raise

logger.info("✅ Custom WebSocket handler registered for /ws (registered after app creation)")


# Register routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend dashboard."""
    frontend_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return frontend_path.read_text()
    return "<h1>👁️ Sauron's Eye - Market Watcher</h1><p>Frontend not found</p>"

# Core routes for Balanced Low Buy System
app.get("/api/balance", response_class=HTMLResponse)(routes.get_balance)
app.post("/api/analyze", response_class=HTMLResponse)(routes.analyze_symbol)
# Backtest route removed - keeping UI focused
app.post("/api/trade", response_class=HTMLResponse)(routes.execute_trade)
app.post("/api/panic", response_class=HTMLResponse)(routes.panic_close)
app.get("/api/positions", response_class=HTMLResponse)(routes.get_positions)
app.post("/api/quick-buy", response_class=HTMLResponse)(routes.quick_buy)
app.post("/api/quick-sell", response_class=HTMLResponse)(routes.quick_sell)
app.post("/api/cancel-order", response_class=HTMLResponse)(routes.cancel_order)
app.get("/api/logs", response_class=HTMLResponse)(routes.get_trade_logs)
app.get("/api/strategy-config", response_class=HTMLResponse)(routes.get_strategy_config)
app.get("/api/strategy")(routes.get_strategy_api)
app.post("/api/strategy")(routes.update_strategy_api)
app.get("/api/strategy/presets")(routes.get_strategy_presets)
app.get("/api/discover-stocks", response_class=HTMLResponse)(routes.discover_stocks)
app.post("/api/explanation")(routes.get_explanation)
# Position chart endpoint removed - keeping UI focused

# MDB-Engine Pattern: Built-in Observability Endpoints
# These endpoints are automatically available via mdb-engine:
# - Health checks: Database connectivity, index status
# - Metrics: Connection pool stats, query performance
@app.get("/health")
async def health_check():
    """Health check endpoint using mdb-engine.
    
    Returns database connectivity status and index health.
    Useful for monitoring and load balancer health checks.
    """
    health = await engine.get_health_status()
    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)

@app.get("/metrics")
async def metrics():
    """Metrics endpoint using mdb-engine.
    
    Returns connection pool statistics and performance metrics.
    Useful for monitoring and debugging.
    """
    return JSONResponse(content=engine.get_metrics())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
