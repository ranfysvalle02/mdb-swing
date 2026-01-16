"""Main application entry point."""
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from mdb_engine.observability import get_logger

from .core.engine import engine, get_manifest_path
from .core.config import SYMBOLS, STRATEGY_CONFIG
from .services.analysis import get_market_data, analyze_technicals
from .services.trading import place_order
from .services.ai import eye
from .services.analysis import api as alpaca_api
from fastapi.responses import HTMLResponse
from .api import routes

logger = get_logger(__name__)

# Create FastAPI app with automatic lifecycle management
app = engine.create_app(
    slug="sauron_eye",
    manifest=get_manifest_path(),
    title="Sauron's Eye - Market Watcher"
)

# Register routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend dashboard."""
    frontend_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return frontend_path.read_text()
    return "<h1>Sauron's Eye - Market Watcher</h1><p>Frontend not found</p>"

app.get("/api/balance", response_class=HTMLResponse)(routes.get_balance)
app.post("/api/analyze", response_class=HTMLResponse)(routes.analyze_symbol)
app.post("/api/backtest", response_class=HTMLResponse)(routes.backtest_symbol)
app.post("/api/trade", response_class=HTMLResponse)(routes.execute_trade)
app.get("/api/positions", response_class=HTMLResponse)(routes.get_positions)
app.get("/api/logs", response_class=HTMLResponse)(routes.get_trade_logs)
app.get("/api/pending-trades", response_class=HTMLResponse)(routes.get_pending_trades)
app.post("/api/approve-trade", response_class=HTMLResponse)(routes.approve_trade)
app.post("/api/reject-trade", response_class=HTMLResponse)(routes.reject_trade)
app.get("/api/strategy-config", response_class=HTMLResponse)(routes.get_strategy_config)
app.post("/api/panic", response_class=HTMLResponse)(routes.panic_close)
app.post("/api/debug", response_class=HTMLResponse)(routes.debug_symbol)
app.get("/api/discover-stocks", response_class=HTMLResponse)(routes.discover_stocks)
app.get("/api/watchlist", response_class=HTMLResponse)(routes.get_watchlist)
app.post("/api/watchlist/add", response_class=HTMLResponse)(routes.add_to_watchlist)
app.post("/api/watchlist/remove", response_class=HTMLResponse)(routes.remove_from_watchlist)
app.get("/api/auto-setup")(routes.auto_setup_watchlist)
app.get("/api/search-trending")(routes.search_trending_stocks)
app.get("/api/trending-with-analysis")(routes.get_trending_stocks_with_analysis)
app.post("/api/quick-buy")(routes.quick_buy)
app.post("/api/quick-sell")(routes.quick_sell)
app.post("/api/cancel-orders")(routes.cancel_orders)
app.get("/api/open-orders")(routes.get_open_orders)
app.websocket("/ws")(routes._get_trending_stocks_with_analysis_streaming)

# Scheduler (will be started in startup)
scheduler: Optional[AsyncIOScheduler] = None

async def scanner_job():
    """The Eye of Sauron scans the markets, seeking signals to extract capital."""
    logger.info("👁️ The Eye watches... scanning markets for swing signals...")
    db = engine.get_scoped_db("sauron_eye")
    
    # Get watchlist from database, fallback to config if empty
    try:
        watchlist_doc = await db.watchlist.find_one({})
        if watchlist_doc and watchlist_doc.get('symbols'):
            symbols_to_scan = watchlist_doc['symbols']
            logger.info(f"📋 Scanning {len(symbols_to_scan)} stocks from watchlist: {', '.join(symbols_to_scan)}")
        else:
            # Fallback to config symbols if watchlist is empty
            symbols_to_scan = SYMBOLS
            logger.info(f"📋 Watchlist empty, using config symbols: {', '.join(symbols_to_scan)}")
    except Exception as e:
        logger.warning(f"Could not fetch watchlist, using config: {e}")
        symbols_to_scan = SYMBOLS
    
    for sym in symbols_to_scan:
        try:
            # Skip if we already have a position
            if alpaca_api:
                try:
                    pos = alpaca_api.get_position(sym)
                    if float(pos.qty) > 0:
                        logger.info(f"Skipping {sym}: Already have position")
                        continue
                except Exception:
                    # No position exists, continue
                    pass

            bars, headlines = get_market_data(sym)
            if bars is None or bars.empty:
                logger.debug(f"No data for {sym}, skipping")
                continue
            
            if len(bars) < 14:
                logger.debug(f"Insufficient data for {sym} ({len(bars)} days), skipping")
                continue
            
            try:
                techs = analyze_technicals(bars)
            except Exception as e:
                logger.error(f"Technical analysis failed for {sym}: {e}")
                continue
            
            # Use strategy preset thresholds
            rsi_threshold = STRATEGY_CONFIG['rsi_threshold']
            ai_score_required = STRATEGY_CONFIG['ai_score_required']
            
            if techs['rsi'] < rsi_threshold and techs['trend'] == "UP":
                logger.info(f"👁️ Signal detected: {sym} - The Eye focuses...")
                
                if eye:
                    try:
                        verdict = eye.analyze(sym, techs['price'], techs['rsi'], techs['atr'], techs['trend'], "\n".join(headlines))
                        
                        if verdict.score >= ai_score_required:
                            logger.info(f"🔥 The Eye commands: EXTRACT CAPITAL from {sym} (Score: {verdict.score}, Required: {ai_score_required})")
                            # Create pending trade instead of auto-executing
                            from datetime import datetime
                            pending_trade = {
                                "timestamp": datetime.now(),
                                "symbol": sym,
                                "action": "buy",
                                "price": techs['price'],
                                "rsi": techs['rsi'],
                                "atr": techs['atr'],
                                "trend": techs['trend'],
                                "reason": verdict.reason,
                                "score": verdict.score,
                                "risk_level": verdict.risk_level,
                                "status": "pending",
                                "headlines": "\n".join(headlines),
                                "strategy_preset": STRATEGY_CONFIG['name'].lower(),
                                "rsi_threshold_used": rsi_threshold,
                                "score_threshold_used": ai_score_required
                            }
                            await db.pending_trades.insert_one(pending_trade)
                            logger.info(f"✅ Pending trade created for {sym} - awaiting manual approval (Strategy: {STRATEGY_CONFIG['name']})")
                        else:
                            logger.info(f"⚫ The Eye sees weakness: {sym} rejected (Score: {verdict.score})")
                    except Exception as e:
                        logger.error(f"AI analysis failed for {sym}: {e}", exc_info=True)
                else:
                    logger.warning("⚠️ The Eye is blind - Azure OpenAI configuration missing, skipping trade.")

        except Exception as e:
            logger.error(f"👁️ The Eye encountered an error watching {sym}: {e}", exc_info=True)

@app.on_event("startup")
async def startup():
    """Initialize application on startup."""
    global scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scanner_job,
        trigger=IntervalTrigger(minutes=15),
        id='scanner_job'
    )
    scheduler.start()
    logger.info("👁️ SAURON'S EYE IS WATCHING...")
    logger.info("👉 UI: http://localhost:5000")
    logger.info("👉 The Eye scans every 15 minutes for swing signals...")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global scheduler
    if scheduler:
        scheduler.shutdown()
    logger.info("👁️ The Eye closes... watching ceases...")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
