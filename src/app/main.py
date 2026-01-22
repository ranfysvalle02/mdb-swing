"""Main application entry point."""
from pathlib import Path
from typing import Dict, Any
from fastapi import Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from mdb_engine.observability import get_logger
from mdb_engine.routing.websockets import register_message_handler
from .core.engine import engine, get_manifest_path, APP_SLUG
from .core.security import CSRFMiddleware, get_csrf_token_for_template
from .core.templates import templates
from .api import routes

logger = get_logger(__name__)


def register_websocket_message_handlers():
    """Register WebSocket message handlers."""
    async def handle_realtime_message(websocket, message: Dict[str, Any]):
        logger.debug(f"Received WebSocket message: {message}")
    
    register_message_handler(APP_SLUG, "realtime", handle_realtime_message)


async def on_startup(app, engine, manifest):
    """Startup tasks."""
    try:
        db = engine.get_scoped_db(APP_SLUG)
        try:
            await db.strategy_config.drop_index("flux_active_idx")
            logger.info("Dropped old conflicting index: flux_active_idx")
        except Exception:
            logger.debug("Old index cleanup: index not found")
        
        try:
            await db.strategies.drop_index("flux_active_idx")
            logger.info("Dropped old conflicting index from strategies: flux_active_idx")
        except Exception:
            logger.debug("Old strategies index cleanup: index not found")
    except Exception as e:
        logger.warning(f"Index cleanup warning (non-fatal): {e}")
    
    register_websocket_message_handlers()
    
    try:
        engine.register_websocket_routes(app, APP_SLUG)
        logger.info("✅ WebSocket routes registered")
    except Exception as e:
        logger.warning(f"Failed to register WebSocket routes: {e}", exc_info=True)
    
    logger.info("⚡ FLUX IS ACTIVE...")
    logger.info("👉 Focus: Balanced Low - Buy stocks at balanced lows")
    logger.info("👉 UI: http://localhost:5000")
    logger.info("👉 FLUX scans on-demand via UI")


async def on_shutdown(app, engine, manifest=None):
    """Shutdown cleanup."""
    logger.info("⚡ FLUX shutting down...")


app = engine.create_app(
    slug=APP_SLUG,
    manifest=get_manifest_path(),
    title="FLUX - Swing Trading Evolved",
    description="AI-Augmented Swing Trading Bot - Balanced Low Buy System",
    version="1.0.0",
    on_startup=on_startup,
    on_shutdown=on_shutdown,
)

app.add_middleware(CSRFMiddleware)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
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

@app.middleware("http")
async def add_csrf_to_context(request: Request, call_next):
    """Add CSRF token to template context for each request."""
    request.state.csrf_token = get_csrf_token_for_template(request)
    response = await call_next(request)
    return response

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, symbols: str = None):
    """WebSocket endpoint for real-time stock analysis updates."""
    logger.info(f"WebSocket handler called for /ws (symbols: {symbols or 'auto'})")
    try:
        db = engine.get_scoped_db(APP_SLUG)
        custom_symbols = None
        if symbols:
            custom_symbols = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        await routes._get_trending_stocks_with_analysis_streaming(websocket, db, custom_symbols=custom_symbols)
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}", exc_info=True)
        raise

frontend_static_path = Path(__file__).parent.parent.parent / "frontend" / "static"
if frontend_static_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_static_path)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend dashboard."""
    frontend_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return frontend_path.read_text()
    return "<h1>⚡ FLUX - Swing Trading Evolved</h1><p>Frontend not found</p>"
app.get("/api/balance", response_class=HTMLResponse)(routes.get_balance)
app.get("/api/balance-compact", response_class=HTMLResponse)(routes.get_balance_compact)
app.post("/api/analyze", response_class=HTMLResponse)(routes.analyze_symbol)
app.post("/api/analyze-preview", response_class=HTMLResponse)(routes.analyze_preview)
app.post("/api/all-stats", response_class=HTMLResponse)(routes.get_all_stats)
app.post("/api/analyze-rejection", response_class=HTMLResponse)(routes.analyze_rejection)
app.post("/api/trade", response_class=HTMLResponse)(routes.execute_trade)
app.post("/api/panic", response_class=HTMLResponse)(routes.panic_close)
app.get("/api/positions", response_class=HTMLResponse)(routes.get_positions)
app.get("/api/transactions", response_class=HTMLResponse)(routes.get_transactions)
app.post("/api/quick-buy", response_class=HTMLResponse)(routes.quick_buy)
app.post("/api/quick-sell", response_class=HTMLResponse)(routes.quick_sell)
app.post("/api/buy-confirmation", response_class=HTMLResponse)(routes.show_buy_confirmation_modal)
app.post("/api/buy-confirm", response_class=HTMLResponse)(routes.confirm_buy_order)
app.post("/api/buy-order-details")(routes.get_buy_order_details)
app.post("/api/inline-buy-interface", response_class=HTMLResponse)(routes.get_inline_buy_interface)
app.post("/api/cancel-order", response_class=HTMLResponse)(routes.cancel_order)
app.get("/api/strategy-display", response_class=HTMLResponse)(routes.get_strategy_display_html)
app.get("/api/strategy")(routes.get_strategy_api)
app.post("/api/strategy")(routes.update_strategy_api)
app.post("/api/strategy/param", response_class=HTMLResponse)(routes.update_strategy_parameter_api)
app.get("/api/firecrawl-query")(routes.get_firecrawl_query)
app.get("/api/timeout-error", response_class=HTMLResponse)(routes.get_timeout_error)
app.post("/api/firecrawl-query")(routes.update_firecrawl_query)
app.get("/api/watch-list", response_class=HTMLResponse)(routes.get_watch_list)
app.get("/api/signal-hunter-section", response_class=HTMLResponse)(routes.get_signal_hunter_section)
app.post("/api/watch-list", response_model=None)(routes.update_watch_list)
app.get("/api/tickers/search")(routes.search_tickers)
app.get("/api/logo/{ticker}", response_class=HTMLResponse)(routes.get_logo)
app.post("/api/explanation")(routes.get_explanation)
app.get("/api/latest-scan")(routes.get_latest_scan)
app.get("/api/analysis-preview")(routes.get_analysis_preview)
app.get("/api/strategies")(routes.get_strategies)
app.get("/api/strategies/{label}")(routes.get_strategy)
app.post("/api/strategies")(routes.create_strategy)
app.put("/api/strategies/{label}")(routes.update_strategy)
app.delete("/api/strategies/{label}")(routes.delete_strategy)
app.post("/api/strategies/{label}/activate")(routes.activate_strategy)
app.get("/api/strategy-builder", response_class=HTMLResponse)(routes.get_strategy_builder)
app.get("/api/strategy-list", response_class=HTMLResponse)(routes.get_strategy_list)
app.get("/api/strategy-selector", response_class=HTMLResponse)(routes.get_strategy_selector)
app.get("/api/strategy-selector-modal", response_class=HTMLResponse)(routes.get_strategy_selector_modal)
app.get("/api/signal-hunter-modal", response_class=HTMLResponse)(routes.get_signal_hunter_modal)
app.post("/api/signal-hunter", response_class=HTMLResponse)(routes.search_signal_hunter)
app.get("/api/finviz-filters")(routes.get_finviz_filters_api)
app.get("/api/alpaca-accounts", response_class=HTMLResponse)(routes.get_alpaca_accounts)
app.post("/api/alpaca-accounts/create", response_class=HTMLResponse)(routes.create_alpaca_account)
app.post("/api/alpaca-accounts/set-active", response_class=HTMLResponse)(routes.set_active_alpaca_account)
app.post("/api/alpaca-accounts/delete", response_class=HTMLResponse)(routes.delete_alpaca_account)
app.post("/api/alpaca-accounts/test", response_class=HTMLResponse)(routes.test_alpaca_account)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health = await engine.get_health_status()
    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)

@app.get("/metrics")
async def metrics():
    """Metrics endpoint."""
    return JSONResponse(content=engine.get_metrics())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
