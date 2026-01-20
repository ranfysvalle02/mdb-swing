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

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .core.engine import engine, get_manifest_path, APP_SLUG
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
    # Clean up old conflicting indexes
    try:
        db = engine.get_scoped_db(APP_SLUG)
        # Drop old conflicting index if it exists
        try:
            await db.strategy_config.drop_index("flux_active_idx")
            logger.info("✅ Dropped old conflicting index: flux_active_idx")
        except Exception as e:
            # Index doesn't exist or already dropped - that's fine
            logger.debug(f"Old index cleanup: {e}")
        
        try:
            await db.strategies.drop_index("flux_active_idx")
            logger.info("✅ Dropped old conflicting index from strategies: flux_active_idx")
        except Exception as e:
            # Index doesn't exist or already dropped - that's fine
            logger.debug(f"Old strategies index cleanup: {e}")
    except Exception as e:
        logger.warning(f"Index cleanup warning (non-fatal): {e}")
    
    # Register WebSocket message handlers (for incoming messages if needed)
    register_websocket_message_handlers()
    
    logger.info("✅ Startup complete - WebSocket handler registered")
    
    # Log startup information
    logger.info("⚡ FLUX IS ACTIVE...")
    logger.info("👉 Focus: Balanced Low - Buy stocks at balanced lows")
    logger.info("👉 UI: http://localhost:5000")
    logger.info("👉 FLUX scans on-demand via UI")


async def on_shutdown(app, engine, manifest=None):
    """Cleanup on shutdown.
    
    MDB-Engine Pattern: Shutdown callback signature matches mdb-engine's expectations.
    The manifest parameter is optional but may be passed by mdb-engine.
    """
    logger.info("⚡ FLUX shutting down...")


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
    title="FLUX - Swing Trading Evolved",
    description="AI-Augmented Swing Trading Bot - Balanced Low Buy System",
    version="1.0.0",
    on_startup=on_startup,      # Custom startup logic
    on_shutdown=on_shutdown,    # Custom cleanup logic
)

# HTMX Gold Standard: Security Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from .core.security import CSRFMiddleware, get_csrf_token_for_template
# Import templates early to ensure filters are registered
from .core.templates import templates

# Add CSRF protection middleware
app.add_middleware(CSRFMiddleware)

# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses.
    
    HTMX Best Practice: Security headers protect against common attacks.
    """
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Content Security Policy - allow HTMX, Alpine.js, and required external resources
        # HTMX Gold Standard: Balance security with functionality
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
            "https://cdn.tailwindcss.com https://unpkg.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' "
            "https://cdn.tailwindcss.com https://fonts.googleapis.com; "
            "font-src 'self' data: https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' ws: wss: https://unpkg.com https://cdn.jsdelivr.net; "
            "frame-src 'self' https://*.tradingview.com;"
        )
        
        response.headers["Content-Security-Policy"] = csp
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Update template context processor to include CSRF token per-request
@app.middleware("http")
async def add_csrf_to_context(request: Request, call_next):
    """Add CSRF token to template context for each request.
    
    HTMX Gold Standard: CSRF tokens available in all templates via csrf_token variable.
    """
    # Store CSRF token in request state for template access
    request.state.csrf_token = get_csrf_token_for_template(request)
    
    # Make CSRF token available to templates via request state
    # Templates can access it via request.state.csrf_token in context processors
    response = await call_next(request)
    return response

# MDB-Engine Pattern: Custom WebSocket Handler
# WebSocket configuration is declared in manifest.json, but we can override
# the handler for custom streaming logic (real-time stock analysis)
from fastapi import WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, symbols: str = None):
    """WebSocket endpoint for real-time stock analysis updates.
    
    MDB-Engine Pattern: Custom WebSocket handler with scoped database access.
    - WebSocket path configured in manifest.json
    - Uses engine.get_scoped_db() for database access
    - Connection lifecycle managed automatically
    
    Args:
        symbols: Optional comma-separated list of symbols to analyze
                 If not provided, uses intelligent discovery
    
    This handler streams stock analysis data to connected clients.
    We register this here to override mdb-engine's default handler.
    """
    logger.info(f"🔌 [MAIN] Custom WebSocket handler called for /ws (symbols: {symbols or 'auto'})")
    try:
        # MDB-Engine: Get scoped database - connection managed automatically
        db = engine.get_scoped_db(APP_SLUG)
        custom_symbols = None
        if symbols:
            custom_symbols = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        await routes._get_trending_stocks_with_analysis_streaming(websocket, db, custom_symbols=custom_symbols)
    except Exception as e:
        logger.error(f"❌ [MAIN] WebSocket handler error: {e}", exc_info=True)
        raise

logger.info("✅ Custom WebSocket handler registered for /ws (registered after app creation)")

# Mount static files
frontend_static_path = Path(__file__).parent.parent.parent / "frontend" / "static"
if frontend_static_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_static_path)), name="static")
    logger.info(f"✅ Static files mounted from {frontend_static_path}")

# Register routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend dashboard."""
    frontend_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return frontend_path.read_text()
    return "<h1>⚡ FLUX - Swing Trading Evolved</h1><p>Frontend not found</p>"

# Core routes for Balanced Low Buy System
app.get("/api/balance", response_class=HTMLResponse)(routes.get_balance)
app.post("/api/analyze", response_class=HTMLResponse)(routes.analyze_symbol)
app.post("/api/analyze-preview", response_class=HTMLResponse)(routes.analyze_preview)
app.post("/api/analyze-rejection", response_class=HTMLResponse)(routes.analyze_rejection)
app.post("/api/trade", response_class=HTMLResponse)(routes.execute_trade)
app.post("/api/panic", response_class=HTMLResponse)(routes.panic_close)
app.get("/api/positions", response_class=HTMLResponse)(routes.get_positions)
app.get("/api/transactions", response_class=HTMLResponse)(routes.get_transactions)
app.post("/api/quick-buy", response_class=HTMLResponse)(routes.quick_buy)
app.post("/api/quick-sell", response_class=HTMLResponse)(routes.quick_sell)
app.post("/api/cancel-order", response_class=HTMLResponse)(routes.cancel_order)
app.get("/api/strategy-display", response_class=HTMLResponse)(routes.get_strategy_display_html)
app.get("/api/strategy")(routes.get_strategy_api)
app.post("/api/strategy")(routes.update_strategy_api)
app.post("/api/strategy/param", response_class=HTMLResponse)(routes.update_strategy_parameter_api)
app.get("/api/firecrawl-query")(routes.get_firecrawl_query)
app.get("/api/timeout-error", response_class=HTMLResponse)(routes.get_timeout_error)
app.post("/api/firecrawl-query")(routes.update_firecrawl_query)
app.get("/api/watch-list", response_class=HTMLResponse)(routes.get_watch_list)
app.post("/api/watch-list")(routes.update_watch_list)
app.get("/api/tickers/search")(routes.search_tickers)
app.get("/api/logo/{ticker}", response_class=HTMLResponse)(routes.get_logo)
app.post("/api/explanation")(routes.get_explanation)
app.get("/api/latest-scan")(routes.get_latest_scan)
app.get("/api/analysis-preview")(routes.get_analysis_preview)

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
