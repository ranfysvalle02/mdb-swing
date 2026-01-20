"""API route handlers for Balanced Low Buy System.

This module contains all FastAPI route handlers for the FLUX trading bot.
Routes are organized by functionality:
- Market scanning and analysis (analyze_symbol)
- Position management (get_positions, quick_buy, quick_sell)
- Trade execution (execute_trade, panic_close)
- Strategy configuration (get_strategy_config, update_strategy_api)
- WebSocket streaming (get_trending_stocks_with_analysis_streaming)

MDB-Engine Integration:
- Database Access: All routes use `Depends(get_scoped_db)` for MongoDB access
  - get_scoped_db() from core.engine provides scoped database connection
  - Automatic connection lifecycle management via mdb-engine
- Logging: Uses `get_logger(__name__)` from mdb_engine.observability
- Embeddings: Uses `Depends(get_embedding_service)` for vector search
  - EmbeddingService from mdb_engine.dependencies for semantic search

Routes return HTMLResponse for HTMX integration, or JSONResponse for API endpoints.
"""
from fastapi import Depends, Form, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import json
from mdb_engine.observability import get_logger
from mdb_engine.dependencies import get_embedding_service
from mdb_engine.embeddings import EmbeddingService
from ..core.engine import get_scoped_db
from ..services.trading import place_order
from ..services.analysis import api
from ..services.analysis import get_market_data, analyze_technicals
from alpaca_trade_api.rest import TimeFrame
from ..services.ai import EyeAI, _ai_engine
from ..services.radar import RadarService
from ..services.positions import calculate_position_metrics, detect_sell_signal
from ..services.logo import get_logo_html
from ..api.templates import empty_positions, pending_order_card, position_card, lucide_init_script, toast_notification, htmx_response, htmx_modal_wrapper, error_response, transaction_history_item, empty_transactions, transactions_view
from ..core.templates import templates
from ..core.config import ALPACA_KEY, ALPACA_SECRET, STRATEGY_CONFIG, get_strategy_from_db, get_strategy_config as get_strategy_config_dict, FIRECRAWL_SEARCH_QUERY_TEMPLATE
from ..services.ai_prompts import get_balanced_low_prompt

logger = get_logger(__name__)


def error_response_html(message: str, status_code: int = 200) -> HTMLResponse:
    """Create a standardized HTML error response for HTMX.
    
    HTMX Best Practice: Return HTML fragments for errors, not JSON.
    This ensures errors can be swapped into the DOM seamlessly.
    
    Args:
        message: Error message to display
        status_code: HTTP status code (default 200 for HTMX compatibility)
        
    Returns:
        HTMLResponse with error message template
    """
    return HTMLResponse(
        content=templates.get_template("partials/error_message.html").render(message=message),
        status_code=status_code
    )


def error_response_json(message: str, detail: Optional[str] = None, status_code: int = 400) -> JSONResponse:
    """Create a standardized JSON error response for API endpoints.
    
    Args:
        message: Error message
        detail: Optional detailed error information
        status_code: HTTP status code
        
    Returns:
        JSONResponse with error structure
    """
    content = {"error": message}
    if detail:
        content["detail"] = detail
    return JSONResponse(content=content, status_code=status_code)


async def get_timeout_error() -> HTMLResponse:
    """Get timeout error template for HTMX timeout handling.
    
    HTMX Best Practice: Server-rendered HTML template for error messages.
    This endpoint provides the timeout error template that can be swapped
    into the target element when a request times out.
    """
    return HTMLResponse(content=templates.get_template("partials/timeout_error.html").render())

async def _get_target_symbols(db: Any, force_discovery: bool = False) -> List[str]:
    """Get target symbols from watch list.
    
    Returns all symbols from watch list (no limit).
    
    Args:
        db: MongoDB database instance
        force_discovery: Ignored (kept for compatibility)
    """
    try:
        # Get watch list from database
        watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
        if watch_list_settings and watch_list_settings.get("value"):
            symbols_str = watch_list_settings.get("value", "")
            symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]  # No limit
            if symbols:
                logger.info(f"📋 Using watch list symbols ({len(symbols)} total): {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
                return symbols
        
        # Fallback to defaults
        from ..core.config import SYMBOLS
        logger.info(f"📋 Using default symbols: {SYMBOLS}")
        return SYMBOLS
    except Exception as e:
        logger.error(f"Error getting watch list: {e}", exc_info=True)
        # Final fallback
        from ..core.config import SYMBOLS
        return SYMBOLS

async def _safe_send_json(websocket: WebSocket, data: Dict[str, Any]) -> bool:
    """Safely send JSON via WebSocket with automatic datetime sanitization.
    
    MDB-Engine Pattern: Ensures all WebSocket messages are JSON-serializable.
    This is a wrapper around websocket.send_json() that automatically sanitizes
    datetime objects and handles serialization errors gracefully.
    
    Args:
        websocket: WebSocket connection
        data: Data dictionary to send
        
    Returns:
        True if message was sent successfully, False if WebSocket is closed
    """
    try:
        # Check if WebSocket is still open
        if websocket.client_state.name != "CONNECTED":
            logger.debug(f"[WS] WebSocket not connected (state: {websocket.client_state.name}), skipping send")
            return False
        
        # First attempt: sanitize and send
        sanitized_data = _sanitize_for_json(data)
        await websocket.send_json(sanitized_data)
        return True
    except RuntimeError as e:
        # WebSocket closed - this is expected when client disconnects
        error_str = str(e).lower()
        if 'close message' in error_str or 'cannot call "send"' in error_str:
            logger.debug(f"[WS] WebSocket closed, cannot send message: {e}")
            return False
        raise
    except (TypeError, ValueError) as e:
        # If sanitization didn't catch everything, try one more deep sanitization pass
        error_str = str(e).lower()
        if 'not json serializable' in error_str or 'datetime' in error_str or 'bool_' in error_str:
            logger.warning(f"First sanitization pass failed, attempting deep sanitization: {e}")
            try:
                # Check WebSocket state again before retry
                if websocket.client_state.name != "CONNECTED":
                    return False
                # Deep sanitization: use json.dumps with default=str to convert any remaining non-serializable objects
                json_str = json.dumps(data, default=str, ensure_ascii=False)
                fallback_data = json.loads(json_str)
                await websocket.send_json(fallback_data)
                return True
            except RuntimeError as e2:
                error_str2 = str(e2).lower()
                if 'close message' in error_str2 or 'cannot call "send"' in error_str2:
                    logger.debug(f"[WS] WebSocket closed during retry: {e2}")
                    return False
                raise
            except Exception as e2:
                logger.error(f"Deep sanitization also failed: {e2}", exc_info=True)
                raise
        else:
            raise
    except Exception as e:
        # Catch any other errors and check if it's a WebSocket closure
        error_str = str(e).lower()
        if 'close message' in error_str or 'cannot call "send"' in error_str:
            logger.debug(f"[WS] WebSocket closed: {e}")
            return False
        raise

def _convert_timestamp_to_iso(timestamp: Union[datetime, str, None]) -> Optional[str]:
    """Convert various timestamp types to ISO format string for JSON serialization.
    
    Handles: datetime, pandas Timestamp, MongoDB Timestamp, None
    """
    if timestamp is None:
        return None
    try:
        # Handle pandas Timestamp
        if hasattr(timestamp, 'isoformat'):
            return timestamp.isoformat()
        # Handle datetime
        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        # Handle string (already converted)
        if isinstance(timestamp, str):
            return timestamp
        # Try to convert to string as fallback
        return str(timestamp)
    except Exception as e:
        logger.warning(f"Failed to convert timestamp {timestamp} to ISO: {e}")
        return None

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert datetime objects and numpy/pandas types to JSON-serializable formats.
    
    MDB-Engine Pattern: Ensure all data sent via WebSocket is JSON-serializable.
    This handles nested dictionaries and lists containing datetime objects, numpy/pandas bool_,
    and other non-serializable types.
    
    Args:
        obj: Any object (dict, list, datetime, numpy bool_, etc.)
        
    Returns:
        Object with all non-serializable objects converted to JSON-serializable formats
    """
    # Handle None
    if obj is None:
        return None
    
    # Handle datetime objects (check this first before other types)
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle BSON datetime objects from MongoDB and pandas Timestamp
    # Check type name first (faster than attribute checks)
    obj_type = type(obj)
    obj_type_name = obj_type.__name__
    
    # Handle numpy/pandas bool_ types (not JSON serializable)
    if obj_type_name == 'bool_' or (hasattr(obj, 'dtype') and 'bool' in str(obj.dtype)):
        return bool(obj)
    
    # Handle numpy/pandas integer types (int8, int16, int32, int64, etc.)
    if obj_type_name.startswith('int') and hasattr(obj, 'item'):
        try:
            return int(obj.item())
        except (AttributeError, ValueError):
            return int(obj)
    
    # Handle numpy/pandas float types (float16, float32, float64, etc.)
    if obj_type_name.startswith('float') and hasattr(obj, 'item'):
        try:
            return float(obj.item())
        except (AttributeError, ValueError):
            return float(obj)
    
    # Check for datetime-like type names
    if 'datetime' in obj_type_name.lower() or 'timestamp' in obj_type_name.lower():
        try:
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif hasattr(obj, 'strftime'):
                return obj.strftime('%Y-%m-%dT%H:%M:%S.%f')
            else:
                return str(obj)
        except Exception:
            return str(obj)
    
    # Handle objects with isoformat method (pandas Timestamp, datetime subclasses, etc.)
    if hasattr(obj, 'isoformat') and callable(getattr(obj, 'isoformat', None)):
        try:
            # Check if it's actually datetime-like by checking for year attribute
            if hasattr(obj, 'year') or hasattr(obj, 'date'):
                result = obj.isoformat()
                return result
        except (AttributeError, TypeError, ValueError):
            pass
    
    # Handle dictionaries - recurse into all values
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    
    # Handle lists and tuples - recurse into all items
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    
    # Handle sets - convert to list and recurse
    if isinstance(obj, set):
        return [_sanitize_for_json(item) for item in obj]
    
    return obj

def check_alpaca_config() -> bool:
    """Check if Alpaca API is properly configured."""
    if not ALPACA_KEY or not ALPACA_SECRET:
        logger.warning("Alpaca API credentials not configured")
        return False
    return True

async def get_balance() -> HTMLResponse:
    """Get account balance.
    
    MDB-Engine Pattern: Returns HTMLResponse (not JSON) for htmx integration.
    HTMX expects HTML fragments that can be swapped into the DOM.
    This endpoint is polled every 10 seconds via hx-trigger="every 10s".
    """
    if not check_alpaca_config():
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message="Alpaca API not configured"
        ))
    
    if api is None:
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message="Alpaca API not initialized"
        ))
    
    try:
        acct = api.get_account()
        pl = float(acct.equity) - float(acct.last_equity)
        pl_color = "text-green-400" if pl >= 0 else "text-red-400"
        pl_icon = "fa-arrow-up" if pl >= 0 else "fa-arrow-down"
        return HTMLResponse(content=templates.get_template("pages/account_balance.html").render(
            equity=float(acct.equity),
            pl_abs=abs(pl),
            buying_power=float(acct.buying_power),
            pl_color=pl_color,
            pl_icon=pl_icon
        ))
    except Exception as e:
        logger.error(f"Failed to get balance: {e}", exc_info=True)
        error_msg = str(e)[:50] if str(e) else "Unknown error"
        return error_response_html(f"API Error: {error_msg}")

def _check_has_position(symbol: str) -> bool:
    """Check if user has an open position for the given symbol."""
    try:
        if not check_alpaca_config() or api is None:
            return False
        positions = api.list_positions()
        return any(p.symbol.upper() == symbol.upper() for p in positions)
    except Exception as e:
        logger.debug(f"Could not check position for {symbol}: {e}")
        return False


def _adjust_action_for_context(action: str, has_position: bool) -> str:
    """Adjust AI action based on context (position vs no position).
    
    If AI says SELL_NOW but user doesn't have a position, change to AVOID.
    This makes the recommendation context-aware and smarter.
    """
    if action == "SELL_NOW" and not has_position:
        return "AVOID"
    return action


async def analyze_symbol(
    ticker: str = Form(...),
    db: Any = Depends(get_scoped_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> HTMLResponse:
    """Analyze a symbol using AI - uses cached data if available."""
    symbol = ticker.upper().strip()
    if not symbol:
        # HTMX Gold Standard: Use template instead of f-string HTML
        return HTMLResponse(content=templates.get_template("partials/status_message.html").render(
            message="Enter Symbol",
            color="gray"
        ))
    
    strategy_id = "balanced_low"
    radar_service = RadarService(db, embedding_service=embedding_service)
    
    # Check if user has a position for context-aware recommendations
    has_position = _check_has_position(symbol)
    
    # Check cache first - if fresh, use it immediately (no loading!)
    cached = await radar_service.get_cached_analysis(symbol, strategy_id)
    if cached and cached.get('fresh'):
        analysis_data = cached['analysis']
        logger.info(f"✅ Using cached analysis for {symbol} (instant response)")
        
        # Extract data from cached analysis
        techs = analysis_data.get('techs', {})
        verdict_data = analysis_data.get('verdict', {})
        
        # Handle verdict (dict from cache)
        from types import SimpleNamespace
        original_action = verdict_data['action']
        adjusted_action = _adjust_action_for_context(original_action, has_position)
        verdict = SimpleNamespace(
            score=verdict_data['score'],
            reason=verdict_data['reason'],
            risk_level=verdict_data['risk_level'],
            action=adjusted_action,
            key_factors=verdict_data['key_factors'],
            risks=verdict_data['risks'],
            opportunities=verdict_data['opportunities'],
            catalyst=verdict_data['catalyst']
        )
        
        color = "text-green-400" if verdict.score > 6 else "text-red-400"
        risk_badge = "badge-success" if verdict.risk_level.upper() == "LOW" else "badge-warning" if verdict.risk_level.upper() == "MEDIUM" else "badge-danger"
        action_color = "text-green-400" if verdict.action == "BUY" else "text-yellow-400" if verdict.action in ["WAIT", "AVOID"] else "text-red-400"
        
        import time
        tradingview_id = f"tradingview_{int(time.time())}_{symbol}"
        
        cached_meta = await radar_service.get_cached_analysis(symbol, strategy_id)
        headlines_count = len(analysis_data.get('headlines', [])) if cached_meta else 0
        bars_count = 0  # We don't store bars count in cache, estimate from techs
        
        # Use summary-first template for better UX
        analysis_content = templates.get_template("pages/analysis_result_summary.html").render(
            symbol=symbol,
            verdict=verdict,
            techs=techs,
            color=color,
            risk_badge=risk_badge,
            action_color=action_color,
            tradingview_id=tradingview_id,
            strategy_config=STRATEGY_CONFIG,
            headlines_count=headlines_count,
            bars_count=bars_count
        )
        
        return HTMLResponse(content=analysis_content)
    
    # Cache miss or stale - do full analysis
    logger.info(f"🔄 Cache miss for {symbol}, doing full analysis...")
    
    # This shows the user what's happening with polished loading state
    import asyncio
    progress_html = templates.get_template("components/ai_analysis_progress.html").render(
        symbol=symbol
    )
    
    # Small delay to ensure loading state is visible (polished UX)
    # This prevents the "flash" when analysis is very fast
    await asyncio.sleep(0.3)  # 300ms minimum display time for loading state
    
    bars, headlines, news_objects = await get_market_data(symbol, days=500, db=db)
    if bars is None or bars.empty:
        error_content = templates.get_template("partials/error_message.html").render(
            message="Data Error: Unable to fetch market data for this symbol. Symbol not found or market closed. Check if the symbol is valid."
        )
        return HTMLResponse(content=error_content)
    
    if len(bars) < 14:
        # Log detailed info for debugging
        logger.error(f"🚨 Still only {len(bars)} days for {symbol} after requesting 500 days!")
        logger.error(f"   This suggests an API issue. Check logs for details.")
        
        insufficient_content = templates.get_template("partials/insufficient_data.html").render(
            symbol=symbol,
            days_available=len(bars)
        )
        return HTMLResponse(content=insufficient_content)
    
    try:
        techs = analyze_technicals(bars)
    except ValueError as e:
        logger.error(f"Technical analysis failed for {symbol}: {e}")
        error_content = templates.get_template("partials/error_message.html").render(
            message=f"Analysis Error: {str(e)}"
        )
        return HTMLResponse(content=error_content)
    except Exception as e:
        logger.error(f"Unexpected error analyzing {symbol}: {e}", exc_info=True)
        error_content = templates.get_template("partials/error_message.html").render(
            message=f"Unexpected Error: {str(e)[:100]}"
        )
        return HTMLResponse(content=error_content)
    
    ai_engine = _ai_engine
    if ai_engine:
        strategy_prompt = get_balanced_low_prompt()
        # Combine headlines for AI analysis
        headlines_text = "\n".join(headlines) if headlines else "No recent news found."
        logger.info(f"🤖 [AI] Analyzing {symbol} with {len(headlines)} headlines")
        
        # Wrap AI analysis in timeout (60 seconds)
        try:
            verdict = await asyncio.wait_for(
                asyncio.to_thread(
                    ai_engine.analyze,
                    ticker=symbol,
                    techs=techs,
                    headlines=headlines_text,
                    strategy_prompt=strategy_prompt
                ),
                timeout=60.0
            )
            logger.info(f"✅ [AI] Analysis complete for {symbol}: Score {verdict.score}/10, Action: {verdict.action}")
        except asyncio.TimeoutError:
            logger.error(f"⏱️ [AI] Analysis timed out for {symbol} after 60 seconds")
            error_content = templates.get_template("partials/error_message.html").render(
                message=f"AI analysis timed out after 60 seconds. The AI service may be slow or overloaded. Please try again."
            )
            return HTMLResponse(content=error_content)
        except Exception as e:
            logger.error(f"❌ [AI] Analysis failed for {symbol}: {e}", exc_info=True)
            error_content = templates.get_template("partials/error_message.html").render(
                message=f"AI analysis failed: {str(e)[:200]}"
            )
            return HTMLResponse(content=error_content)
        
        # Adjust action based on position context
        original_action = verdict.action
        adjusted_action = _adjust_action_for_context(original_action, has_position)
        if original_action != adjusted_action:
            logger.info(f"🔄 Adjusted action for {symbol}: {original_action} → {adjusted_action} (has_position={has_position})")
            # Create new verdict with adjusted action, preserving all insight fields
            from types import SimpleNamespace
            verdict = SimpleNamespace(
                score=verdict.score,
                reason=verdict.reason,
                risk_level=verdict.risk_level,
                action=adjusted_action,
                key_factors=verdict.key_factors,
                risks=verdict.risks,
                opportunities=verdict.opportunities,
                catalyst=verdict.catalyst
            )
        
        # Cache the analysis for future requests (store original action, we'll adjust on retrieval)
        verdict_dict = {
            'score': verdict.score,
            'reason': verdict.reason,
            'risk_level': verdict.risk_level,
            'action': original_action,  # Store original, adjust on retrieval
            'key_factors': verdict.key_factors,
            'risks': verdict.risks,
            'opportunities': verdict.opportunities,
            'catalyst': verdict.catalyst
        }
        analysis_data = {
            'techs': techs,
            'verdict': verdict_dict,
            'headlines': headlines,
            'news_objects': news_objects
        }
        await radar_service.cache_analysis(symbol, strategy_id, analysis_data)
        
        color = "text-green-400" if verdict.score > 6 else "text-red-400"
        risk_badge = "badge-success" if verdict.risk_level.upper() == "LOW" else "badge-warning" if verdict.risk_level.upper() == "MEDIUM" else "badge-danger"
        action_color = "text-green-400" if verdict.action == "BUY" else "text-yellow-400" if verdict.action in ["WAIT", "AVOID"] else "text-red-400"
        
        import time
        tradingview_id = f"tradingview_{int(time.time())}_{symbol}"
        
        # Use summary-first template for better UX - shows key info immediately, details on demand
        analysis_content = templates.get_template("pages/analysis_result_summary.html").render(
            symbol=symbol,
            verdict=verdict,
            techs=techs,
            color=color,
            risk_badge=risk_badge,
            action_color=action_color,
            tradingview_id=tradingview_id,
            strategy_config=STRATEGY_CONFIG,
            headlines_count=len(headlines) if headlines else 0,
            bars_count=len(bars) if bars is not None and not bars.empty else 0
        )
        
        # Return just the content (for HTMX swap into existing modal)
        return HTMLResponse(content=analysis_content)
    else:
        no_ai_content = templates.get_template("pages/analysis_no_ai.html").render()
        return HTMLResponse(content=no_ai_content)

async def analyze_preview(ticker: str = Form(...), db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Show pre-computed analysis preview for both candidates and non-candidates.
    
    Unified preview that shows technical indicators without AI analysis.
    Includes "Analyze with AI" button for on-demand AI scoring.
    """
    symbol = ticker.upper().strip()
    if not symbol:
        error_content = templates.get_template("partials/error_message.html").render(
            message="Invalid symbol"
        )
        modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
            symbol="Unknown",
            title="Analysis Error",
            content=error_content
        )
        return HTMLResponse(content=modal_html)
    
    try:
        from ..core.config import get_strategy_config
        from ..services.analysis import analyze_technicals, get_market_data
        
        strategy_config = await get_strategy_config(db)
        rsi_threshold = strategy_config.get('rsi_threshold', 35)
        rsi_min = strategy_config.get('rsi_min', 20)
        sma_proximity_pct = strategy_config.get('sma_proximity_pct', 3.0)
        
        bars, headlines, news_objects = await get_market_data(symbol, days=100, db=db)
        
        if bars is None or bars.empty or len(bars) < 14:
            insufficient_content = templates.get_template("partials/insufficient_data.html").render(
                symbol=symbol,
                days_available=len(bars) if bars is not None and not bars.empty else 0
            )
            modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
                symbol=symbol,
                title=f"{symbol} Analysis",
                content=insufficient_content
            )
            return HTMLResponse(content=modal_html)
        
        # Analyze technicals (pre-computed, no AI)
        techs = analyze_technicals(bars)
        price = techs.get('price')
        rsi = techs.get('rsi')
        trend = techs.get('trend')
        sma_200 = techs.get('sma')
        atr = techs.get('atr')
        
        change_pct = None
        if bars is not None and not bars.empty and len(bars) >= 2:
            prev_close = float(bars.iloc[-2]['close'])
            if prev_close > 0:
                change_pct = ((price - prev_close) / prev_close) * 100
        
        price_vs_sma_pct = None
        if price and sma_200 and sma_200 > 0:
            price_vs_sma_pct = ((price - sma_200) / sma_200) * 100
        
        # Check if meets criteria (RSI in sweet spot: rsi_min < RSI < rsi_threshold AND uptrend AND within SMA-200 proximity)
        meets_criteria = (
            rsi is not None and 
            rsi > rsi_min and
            rsi < rsi_threshold and
            trend == "UP" and
            (price_vs_sma_pct is None or price_vs_sma_pct <= sma_proximity_pct)
        )
        
        # Build rejection reason if not a candidate
        rejection_reason = None
        if not meets_criteria:
            reasons = []
            if rsi is not None:
                if rsi <= rsi_min:
                    reasons.append(f"RSI {rsi:.1f} ≤ {rsi_min} (too extreme oversold)")
                elif rsi >= rsi_threshold:
                    reasons.append(f"RSI {rsi:.1f} ≥ {rsi_threshold} (not oversold)")
            if trend != "UP":
                if trend == "DOWN":
                    reasons.append(f"Downtrend (Price ${price:.2f} < SMA-200 ${sma_200:.2f})")
                else:
                    reasons.append("Trend unclear")
            if price_vs_sma_pct is not None and price_vs_sma_pct > sma_proximity_pct:
                reasons.append(f"Price {price_vs_sma_pct:.1f}% above SMA-200 (exceeds {sma_proximity_pct}% proximity limit)")
            rejection_reason = "; ".join(reasons) if reasons else "Does not meet entry criteria"
        
        # Render preview content
        preview_content = templates.get_template("pages/analysis_preview.html").render(
            symbol=symbol,
            price=price,
            change_pct=change_pct,
            rsi=rsi,
            trend=trend,
            sma_200=sma_200,
            atr=atr,
            price_vs_sma_pct=price_vs_sma_pct,
            sma_proximity_pct=sma_proximity_pct,
            meets_criteria=meets_criteria,
            rejection_reason=rejection_reason,
            rsi_threshold=rsi_threshold
        )
        
        # Wrap in modal
        modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
            symbol=symbol,
            title=f"{symbol} Analysis Preview",
            content=preview_content
        )
        
        return HTMLResponse(content=modal_html)
        
    except Exception as e:
        logger.error(f"Error showing preview for {symbol}: {e}", exc_info=True)
        error_content = templates.get_template("partials/status_message.html").render(
            message=f"Could not load preview for {symbol}. Please try again later.",
            color="red"
        )
        modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
            symbol=symbol,
            title=f"{symbol} - Preview Error",
            content=error_content
        )
        return HTMLResponse(content=modal_html)

async def analyze_rejection(ticker: str = Form(...), db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Analyze why a symbol doesn't meet entry criteria - simple explanation."""
    symbol = ticker.upper().strip()
    if not symbol:
        error_content = templates.get_template("partials/error_message.html").render(
            message="Invalid symbol"
        )
        modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
            symbol="Unknown",
            title="Analysis Error",
            content=error_content
        )
        return HTMLResponse(content=modal_html)
    
    try:
        from ..core.config import get_strategy_config
        from ..services.analysis import analyze_technicals, get_market_data
        
        strategy_config = await get_strategy_config(db)
        rsi_threshold = strategy_config.get('rsi_threshold', 35)
        
        bars, headlines, news_objects = await get_market_data(symbol, days=100, db=db)
        
        if bars is None or bars.empty or len(bars) < 14:
            insufficient_content = templates.get_template("partials/insufficient_data.html").render(
                symbol=symbol,
                days_available=len(bars) if bars is not None and not bars.empty else 0
            )
            modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
                symbol=symbol,
                title=f"{symbol} - Why Not a Candidate",
                content=insufficient_content
            )
            return HTMLResponse(content=modal_html)
        
        # Analyze technicals
        techs = analyze_technicals(bars)
        price = techs.get('price')
        rsi = techs.get('rsi')
        trend = techs.get('trend')
        sma_200 = techs.get('sma_200')
        
        # Simple explanations
        reasons = []
        fixes = []
        
        if rsi is None:
            reasons.append("RSI (Relative Strength Index) couldn't be calculated")
            fixes.append("We need at least 14 days of price data to calculate RSI")
        elif rsi >= rsi_threshold:
            reasons.append(f"RSI is {rsi:.1f}, which is too high (we need it below {rsi_threshold})")
            fixes.append(f"Wait for the stock to become more oversold (RSI below {rsi_threshold})")
            if rsi > 70:
                fixes.append("The stock is currently overbought - it's been rising too fast")
        
        if trend != "UP":
            if trend == "DOWN":
                reasons.append(f"Price is below the 200-day average (downtrend)")
                fixes.append("Wait for the stock to start an uptrend (price above 200-day average)")
            else:
                reasons.append("Trend is unclear or neutral")
                fixes.append("Wait for a clear uptrend to develop")
        
        if sma_200 and price:
            price_vs_sma = ((price - sma_200) / sma_200) * 100
            if price < sma_200:
                reasons.append(f"Price (${price:.2f}) is {abs(price_vs_sma):.1f}% below the 200-day average (${sma_200:.2f})")
        
        earnings_in_days = techs.get('earnings_in_days')
        earnings_days_min = strategy_config.get('earnings_days_min', 3)
        if earnings_in_days is not None and earnings_in_days < earnings_days_min:
            reasons.append(f"Earnings announcement in {earnings_in_days} day(s) - high volatility risk")
            fixes.append(f"Wait until after earnings (at least {earnings_days_min} days away)")
        
        pe_ratio = techs.get('pe_ratio')
        pe_ratio_max = strategy_config.get('pe_ratio_max', 50.0)
        if pe_ratio and pe_ratio > pe_ratio_max:
            reasons.append(f"P/E ratio {pe_ratio:.1f} is too high (prefer < {pe_ratio_max:.0f} for quality bounce)")
            fixes.append("This stock may be overvalued even if oversold - wait for better entry")
        
        market_cap = techs.get('market_cap')
        market_cap_min = strategy_config.get('market_cap_min', 300_000_000)
        if market_cap and market_cap < market_cap_min:
            market_cap_m = market_cap / 1_000_000
            reasons.append(f"Market cap ${market_cap_m:.0f}M is too small (prefer > ${market_cap_min/1_000_000:.0f}M for liquidity)")
            fixes.append("Smaller stocks may have liquidity issues - prefer larger market cap")
        
        analyst_rec = techs.get('analyst_recommendation')
        if analyst_rec and analyst_rec in ['Sell', 'Strong Sell']:
            reasons.append(f"Analyst recommendation: {analyst_rec} (prefer Buy/Strong Buy for oversold stocks)")
            fixes.append("Analyst sentiment is bearish - wait for more positive outlook")
        
        # Render rejection content
        rejection_content = templates.get_template("pages/analysis_rejection.html").render(
            symbol=symbol,
            reasons=reasons,
            fixes=fixes,
            rsi=rsi,
            trend=trend,
            rsi_threshold=rsi_threshold
        )
        
        # Wrap in modal (HTMX pattern: just-in-time modal)
        modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
            symbol=symbol,
            title=f"{symbol} - Why Not a Candidate",
            content=rejection_content
        )
        
        return HTMLResponse(content=modal_html)
        
    except Exception as e:
        logger.error(f"Error analyzing rejection for {symbol}: {e}", exc_info=True)
        error_content = templates.get_template("partials/status_message.html").render(
            message=f"Could not analyze {symbol}. Please try again later.",
            color="red"
        )
        modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
            symbol=symbol,
            title=f"{symbol} - Analysis Error",
            content=error_content
        )
        return HTMLResponse(content=modal_html)

async def execute_trade(ticker: str = Form(...), action: str = Form(...), db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Execute a manual trade."""
    if not check_alpaca_config():
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message="Alpaca API not configured"
        ))
    
    if api is None:
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message="Alpaca API not initialized"
        ))
    
    sym = ticker.upper()
    
    if action == 'buy':
        try:
            order = await place_order(sym, 'buy', "Manual Override", 10, db)
            if order:
                return HTMLResponse(content=templates.get_template("partials/success_message.html").render(
                    message="Order Placed Successfully"
                ))
            return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                message="Order Failed - Check logs"
            ))
        except Exception as e:
            logger.error(f"Failed to place order: {e}", exc_info=True)
            return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                message=f"Error: {str(e)[:50]}"
            ))
    else:
        try:
            api.close_position(sym)
            return HTMLResponse(content=templates.get_template("partials/success_message.html").render(
                message="Position Sold"
            ))
        except Exception as e:
            logger.error(f"Failed to close position: {e}", exc_info=True)
            error_msg = str(e)[:50] if str(e) else "No Position"
            return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                message=error_msg
            ))

async def get_positions(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get current active positions only.
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) for automatic connection management.
    Returns HTML with htmx buttons for closing positions (hx-post="/api/quick-sell").
    This endpoint is polled every 5 seconds via hx-trigger="every 5s".
    
    Note: Pending orders are now shown in the Transactions tab.
    """
    if not check_alpaca_config():
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message="Alpaca API not configured"
        ))
    
    if api is None:
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message="Alpaca API not initialized"
        ))
    
    try:
        pos = api.list_positions()
        
        # Empty state
        if not pos:
            return HTMLResponse(content=empty_positions())
        
        html_parts = []
        
        # Active positions only
        for p in pos:
            trade_record = await db.history.find_one(
                {"symbol": p.symbol, "action": "buy"},
                sort=[("timestamp", -1)]
            )
            
            metrics = await calculate_position_metrics(p, trade_record, db=db)
            
            # Detect sell signals using service layer
            sell_signal = await detect_sell_signal(
                symbol=p.symbol,
                current_price=metrics.current_price,
                take_profit=metrics.take_profit,
                unrealized_pl=metrics.unrealized_pl,
                db=db
            )
            
            # Generate card using template helper
            html_parts.append(position_card(
                symbol=p.symbol,
                metrics=metrics,
                sell_signal=sell_signal
            ))
        
        html = "".join(html_parts) + lucide_init_script()
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Failed to get positions: {e}", exc_info=True)
        error_msg = str(e)[:50] if str(e) else "Unknown error"
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"API Error: {error_msg}"
        ))

async def get_transactions(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get pending orders and transaction history.
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) for automatic connection management.
    Returns HTML with pending orders section and transaction history section.
    This endpoint is polled every 5 seconds via hx-trigger="every 5s".
    
    Filters out bracket order children (SELL orders for symbols with active positions).
    """
    if not check_alpaca_config():
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message="Alpaca API not configured"
        ))
    
    if api is None:
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message="Alpaca API not initialized"
        ))
    
    try:
        # Get active positions to filter out bracket children
        pos = api.list_positions()
        position_symbols = {p.symbol for p in pos} if pos else set()
        
        # Get pending orders
        pending_orders = []
        try:
            all_orders = api.list_orders(status='open')
            # Filter for pending statuses (orders that haven't been filled yet)
            pending_statuses = ['new', 'pending_new', 'accepted', 'pending_replace', 'pending_cancel', 'partially_filled']
            pending_orders = [o for o in all_orders if o.status in pending_statuses]
            
            # Filter out SELL orders for symbols with active positions
            # These are bracket order children (take profit/stop loss) and shouldn't show as pending
            pending_orders = [
                o for o in pending_orders 
                if not (o.side.lower() == 'sell' and o.symbol in position_symbols)
            ]
            logger.debug(f"Found {len(pending_orders)} pending orders (excluded bracket children)")
        except Exception as e:
            logger.warning(f"Could not fetch pending orders: {e}", exc_info=True)
        
        # Get transaction history from database
        transaction_history = []
        try:
            history_records = await db.history.find().sort("timestamp", -1).limit(50).to_list(length=50)
            transaction_history = history_records if history_records else []
            logger.debug(f"Found {len(transaction_history)} transaction history records")
        except Exception as e:
            logger.warning(f"Could not fetch transaction history: {e}", exc_info=True)
        
        # Empty state
        if not pending_orders and not transaction_history:
            return HTMLResponse(content=empty_transactions())
        
        # Fetch trade records for pending orders
        trade_records_map = {}
        for order in pending_orders:
            try:
                trade_record = await db.history.find_one(
                    {"symbol": order.symbol, "action": order.side.lower()},
                    sort=[("timestamp", -1)]
                )
                if trade_record:
                    key = (order.symbol, order.side.lower())
                    trade_records_map[key] = trade_record
            except Exception as e:
                logger.debug(f"Could not fetch trade record for {order.symbol}: {e}")
        
        # Build HTML using transactions_view helper
        html = transactions_view(
            pending_orders=pending_orders,
            transaction_history=transaction_history,
            trade_records_map=trade_records_map
        )
        
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Failed to get transactions: {e}", exc_info=True)
        error_msg = str(e)[:50] if str(e) else "Unknown error"
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"API Error: {error_msg}"
        ))

async def cancel_order(order_id: str = Form(...), db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Cancel a pending order.
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) for automatic connection management.
    Returns HTML with hx-swap-oob to update transactions list and show toast notification.
    """
    if not check_alpaca_config() or api is None:
        return HTMLResponse(content=toast_notification(
            message="Alpaca API not configured",
            type="error"
        ), status_code=400)
    
    try:
        logger.info(f"Canceling order: {order_id}")
        
        # Cancel the order via Alpaca API
        api.cancel_order(order_id)
        
        transactions_response = await get_transactions(db=db)
        transactions_html = transactions_response.body.decode() if hasattr(transactions_response.body, 'decode') else str(transactions_response.body)
        
        # Return HTML with hx-swap-oob to update transactions and show success toast
        success_toast = toast_notification("Order canceled successfully", "success", 3000)
        response_content = htmx_response(
            updates={
                "transactions-list": transactions_html,
                "toast-container": success_toast
            }
        )
        response = HTMLResponse(content=response_content)
        response.headers["HX-Trigger"] = "refreshTransactions"
        return response
        
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id}: {e}", exc_info=True)
        error_msg = str(e)[:150] if str(e) else "Unknown error occurred"
        return HTMLResponse(content=toast_notification(
            message=f"Cancel failed: {error_msg}",
            type="error"
        ), status_code=500)

async def get_strategy_display_html(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get strategy configuration display view - Balanced Low Buy System.
    
    Returns an inline-editable display of current strategy configuration.
    Loads from database first, falls back to env vars/defaults.
    """
    try:
        # Load config from database or fallback
        try:
            config = await get_strategy_from_db(db)
            if not config:
                raise ValueError("No active config found")
        except Exception as config_error:
            logger.warning(f"Could not load strategy config from DB, using defaults: {config_error}")
            from ..core.config import STRATEGY_CONFIG
            config = {
                "rsi_threshold": STRATEGY_CONFIG.get("rsi_threshold", 35),
                "rsi_min": STRATEGY_CONFIG.get("rsi_min", 20),
                "sma_proximity_pct": STRATEGY_CONFIG.get("sma_proximity_pct", 3.0),
                "ai_score_required": STRATEGY_CONFIG.get("ai_score_required", 7),
                "name": STRATEGY_CONFIG.get("name", "Balanced Low"),
                "description": STRATEGY_CONFIG.get("description", "Buy stocks at balanced lows"),
                "color": STRATEGY_CONFIG.get("color", "green"),
                "preset": "Custom"
            }
        
        color = config.get('color', 'green')
        color_classes = {
            'green': 'border-green-500/30 bg-green-500/10',
            'blue': 'border-blue-500/30 bg-blue-500/10',
            'yellow': 'border-yellow-500/30 bg-yellow-500/10',
            'red': 'border-red-500/30 bg-red-500/10'
        }
        border_class = color_classes.get(color, 'border-green-500/30')
        
        rsi_threshold = config.get('rsi_threshold', 35)
        rsi_min = config.get('rsi_min', 20)
        sma_proximity_pct = config.get('sma_proximity_pct', 3.0)
        ai_score_required = config.get('ai_score_required', 7)
        name = config.get('name', 'Balanced Low')
        
        # Render template
        return HTMLResponse(content=templates.get_template("pages/strategy_display.html").render(
            name=name,
            rsi_threshold=rsi_threshold,
            rsi_min=rsi_min,
            sma_proximity_pct=sma_proximity_pct,
            ai_score_required=ai_score_required,
            border_class=border_class,
            current_preset='Custom'
        ))
    except Exception as e:
        logger.error(f"Error loading strategy display: {e}", exc_info=True)
        from ..core.config import STRATEGY_CONFIG
        fallback_config = STRATEGY_CONFIG
        
        fallback_color_classes = {
            'green': 'border-green-500/30 bg-green-500/10',
            'blue': 'border-blue-500/30 bg-blue-500/10',
            'yellow': 'border-yellow-500/30 bg-yellow-500/10',
            'red': 'border-red-500/30 bg-red-500/10'
        }
        fallback_border_class = fallback_color_classes.get(fallback_config.get('color', 'green'), 'border-yellow-500/30')
        
        return HTMLResponse(content=templates.get_template("pages/strategy_display.html").render(
            name=fallback_config.get('name', 'Balanced Low'),
            rsi_threshold=fallback_config.get('rsi_threshold', 35),
            rsi_min=fallback_config.get('rsi_min', 20),
            sma_proximity_pct=fallback_config.get('sma_proximity_pct', 3.0),
            ai_score_required=fallback_config.get('ai_score_required', 7),
            border_class=fallback_border_class,
            current_preset='Custom'
        ))

async def get_strategy_api(db: Any = Depends(get_scoped_db)) -> JSONResponse:
    """Get current active strategy config from MongoDB or env vars."""
    try:
        db_config = await get_strategy_from_db(db)
        if db_config:
            # Remove goal from response
            response_config = {k: v for k, v in db_config.items() if k != 'goal'}
            return JSONResponse(content={
                "success": True,
                "source": "database",
                "config": response_config
            })
        # Fallback to defaults
        from ..core.config import STRATEGY_CONFIG
        fallback_config = {k: v for k, v in STRATEGY_CONFIG.items() if k != 'goal'}
        return JSONResponse(content={
            "success": True,
            "source": "defaults",
            "config": fallback_config
        })
    except Exception as e:
        logger.error(f"Error getting strategy config: {e}", exc_info=True)
        return JSONResponse(content={
            "success": False,
            "error": str(e)[:100]
        }, status_code=500)

async def update_strategy_api(request: Request, db: Any = Depends(get_scoped_db)) -> JSONResponse:
    """Update strategy configuration in MongoDB.
    
    HTMX Gold Standard: Returns HTMLResponse for HTMX requests, JSONResponse for API requests.
    """
    # Check if this is an HTMX request
    is_htmx = request.headers.get("HX-Request") == "true"
    
    try:
        # Handle both form data (HTMX) and JSON (API)
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form_data = await request.form()
            body = dict(form_data)
            # Convert string values to appropriate types
            for key in ['rsi_threshold', 'rsi_min', 'ai_score_required']:
                if key in body:
                    try:
                        body[key] = int(body[key])
                    except (ValueError, TypeError):
                        pass
            for key in ['sma_proximity_pct']:
                if key in body:
                    try:
                        body[key] = float(body[key])
                    except (ValueError, TypeError):
                        pass
        else:
            body = await request.json()
        
        # Build config from custom parameters
        config = {
            "rsi_threshold": body.get('rsi_threshold', STRATEGY_CONFIG.get('rsi_threshold', 35)),
            "rsi_min": body.get('rsi_min', STRATEGY_CONFIG.get('rsi_min', 20)),
            "sma_proximity_pct": body.get('sma_proximity_pct', STRATEGY_CONFIG.get('sma_proximity_pct', 3.0)),
            "ai_score_required": body.get('ai_score_required', STRATEGY_CONFIG.get('ai_score_required', 7)),
            "preset": "Custom"
        }
        
        # Validate parameters
        if not (0 < config['rsi_threshold'] <= 100):
            error_msg = "rsi_threshold must be between 0 and 100"
            if is_htmx:
                return error_response(error_msg, status_code=400, target_id="toast-container")
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        if not (15 <= config.get('rsi_min', 20) <= 25):
            error_msg = "rsi_min must be between 15 and 25"
            if is_htmx:
                return error_response(error_msg, status_code=400, target_id="toast-container")
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        if not (0 <= config.get('sma_proximity_pct', 3.0) <= 5.0):
            error_msg = "sma_proximity_pct must be between 0 and 5"
            if is_htmx:
                return error_response(error_msg, status_code=400, target_id="toast-container")
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        if not (0 <= config['ai_score_required'] <= 10):
            error_msg = "ai_score_required must be between 0 and 10"
            if is_htmx:
                return error_response(error_msg, status_code=400, target_id="toast-container")
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        
        # Set all existing configs to inactive
        await db.strategy_config.update_many(
            {"active": True},
            {"$set": {"active": False}}
        )
        
        # Insert new active config
        config_doc = {
            **config,
            "active": True,
            "created_at": datetime.now(),
            "name": STRATEGY_CONFIG.get('name', 'Balanced Low'),
            "description": STRATEGY_CONFIG.get('description', 'Buy stocks at balanced lows')
        }
        
        await db.strategy_config.insert_one(config_doc)
        
        logger.info(f"Strategy updated: RSI<{config['rsi_threshold']}, Score≥{config['ai_score_required']}")
        
        if is_htmx:
            # HTMX Gold Standard: Return updated strategy display HTML
            # Strategy changes affect watchlist evaluation, so trigger re-evaluation
            logger.info(f"✅ Strategy updated: RSI {config.get('rsi_min', 20)}-{config['rsi_threshold']}, SMA proximity {config.get('sma_proximity_pct', 3.0)}%")
            
            # Get updated strategy display HTML
            strategy_display_html = await get_strategy_display_html(db)
            strategy_display_content = strategy_display_html.body.decode('utf-8') if isinstance(strategy_display_html.body, bytes) else str(strategy_display_html.body)
            
            # Return HTML with hx-swap-oob to update strategy display
            # Use HX-Trigger header to trigger watchlist re-evaluation
            response = HTMLResponse(content=strategy_display_content)
            response.headers["HX-Trigger"] = "strategySaved"
            return response
        
        return JSONResponse(content={
            "success": True,
            "config": config_doc
        })
    except Exception as e:
        logger.error(f"Error updating strategy config: {e}", exc_info=True)
        error_msg = str(e)[:100]
        if is_htmx:
            return error_response(f"Failed to save: {error_msg}", status_code=500, target_id="toast-container")
        return JSONResponse(content={"success": False, "error": error_msg}, status_code=500)


async def update_strategy_parameter_api(request: Request, db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Update a single strategy parameter.
    
    HTMX endpoint for inline editing of individual parameters.
    """
    is_htmx = request.headers.get("HX-Request") == "true"
    
    try:
        # Handle both form data (HTMX) and JSON (API)
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form_data = await request.form()
            body = dict(form_data)
        else:
            body = await request.json()
        
        param_name = body.get('param')
        param_value = body.get('value')
        
        # Log incoming request for debugging
        logger.debug(f"Strategy param update request: param={param_name}, value={param_value}, content_type={content_type}")
        
        # Validate required fields (handle both None and empty string)
        if not param_name or param_value is None or param_value == '':
            error_msg = f"Missing 'param' or 'value' (got param={param_name}, value={param_value})"
            logger.warning(error_msg)
            if is_htmx:
                # Return current strategy display with 200 OK (HTMX doesn't swap on error codes)
                display_response = await get_strategy_display_html(db)
                display_html = display_response.body.decode('utf-8') if isinstance(display_response.body, bytes) else str(display_response.body)
                return HTMLResponse(content=display_html, status_code=200)
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        
        # Validate parameter name
        valid_params = ['rsi_threshold', 'rsi_min', 'sma_proximity_pct', 'ai_score_required']
        if param_name not in valid_params:
            error_msg = f"Invalid parameter name. Must be one of: {', '.join(valid_params)}"
            if is_htmx:
                # Return current strategy display with 200 OK
                display_response = await get_strategy_display_html(db)
                display_html = display_response.body.decode('utf-8') if isinstance(display_response.body, bytes) else str(display_response.body)
                return HTMLResponse(content=display_html, status_code=200)
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        
        # Convert and validate value
        try:
            # Handle string values from form data
            if isinstance(param_value, str):
                param_value = param_value.strip()
                if not param_value:
                    raise ValueError("Empty value")
            
            if param_name in ['rsi_threshold', 'rsi_min', 'ai_score_required']:
                param_value = int(param_value)
            elif param_name == 'sma_proximity_pct':
                param_value = float(param_value)
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid value for {param_name}: {param_value} ({type(param_value).__name__}) - {str(e)}"
            logger.warning(error_msg)
            if is_htmx:
                # Return current strategy display with 200 OK
                display_response = await get_strategy_display_html(db)
                display_html = display_response.body.decode('utf-8') if isinstance(display_response.body, bytes) else str(display_response.body)
                return HTMLResponse(content=display_html, status_code=200)
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        
        # Validate ranges
        if param_name == 'rsi_threshold' and not (0 < param_value <= 100):
            error_msg = "rsi_threshold must be between 0 and 100"
            logger.warning(f"Validation failed: {error_msg}")
            if is_htmx:
                display_response = await get_strategy_display_html(db)
                display_html = display_response.body.decode('utf-8') if isinstance(display_response.body, bytes) else str(display_response.body)
                return HTMLResponse(content=display_html, status_code=200)
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        elif param_name == 'rsi_min' and not (15 <= param_value <= 25):
            error_msg = "rsi_min must be between 15 and 25"
            logger.warning(f"Validation failed: {error_msg}")
            if is_htmx:
                display_response = await get_strategy_display_html(db)
                display_html = display_response.body.decode('utf-8') if isinstance(display_response.body, bytes) else str(display_response.body)
                return HTMLResponse(content=display_html, status_code=200)
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        elif param_name == 'sma_proximity_pct' and not (0 <= param_value <= 5.0):
            error_msg = "sma_proximity_pct must be between 0 and 5"
            logger.warning(f"Validation failed: {error_msg}")
            if is_htmx:
                display_response = await get_strategy_display_html(db)
                display_html = display_response.body.decode('utf-8') if isinstance(display_response.body, bytes) else str(display_response.body)
                return HTMLResponse(content=display_html, status_code=200)
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        elif param_name == 'ai_score_required' and not (0 <= param_value <= 10):
            error_msg = "ai_score_required must be between 0 and 10"
            logger.warning(f"Validation failed: {error_msg}")
            if is_htmx:
                display_response = await get_strategy_display_html(db)
                display_html = display_response.body.decode('utf-8') if isinstance(display_response.body, bytes) else str(display_response.body)
                return HTMLResponse(content=display_html, status_code=200)
            return JSONResponse(content={"success": False, "error": error_msg}, status_code=400)
        
        # Get current config
        current_config = await get_strategy_config_dict(db)
        
        # Update the parameter
        config = {
            "rsi_threshold": current_config.get('rsi_threshold', 35),
            "rsi_min": current_config.get('rsi_min', 20),
            "sma_proximity_pct": current_config.get('sma_proximity_pct', 3.0),
            "ai_score_required": current_config.get('ai_score_required', 7),
            "preset": "Custom"
        }
        config[param_name] = param_value
        
        # Set all existing configs to inactive
        await db.strategy_config.update_many(
            {"active": True},
            {"$set": {"active": False}}
        )
        
        # Insert new active config
        config_doc = {
            **config,
            "active": True,
            "created_at": datetime.now(),
            "name": STRATEGY_CONFIG.get('name', 'Balanced Low'),
            "description": STRATEGY_CONFIG.get('description', 'Buy stocks at balanced lows')
        }
        
        await db.strategy_config.insert_one(config_doc)
        
        logger.info(f"Strategy parameter updated: {param_name} = {param_value}")
        
        if is_htmx:
            # Return updated strategy display HTML (will replace innerHTML of #strategy-display)
            display_response = await get_strategy_display_html(db)
            
            # Decode response body properly
            if isinstance(display_response.body, bytes):
                display_html = display_response.body.decode('utf-8')
            elif hasattr(display_response.body, 'decode'):
                display_html = display_response.body.decode('utf-8')
            else:
                display_html = str(display_response.body)
            
            # Staff Engineer Solution: Strategy changes require user to manually re-evaluate watchlist
            # Show re-evaluate button instead of auto-refreshing watchlist
            logger.info(f"✅ Strategy parameter updated: {param_name}={param_value} (new config: RSI {config.get('rsi_min', 20)}-{config.get('rsi_threshold', 35)}, SMA proximity {config.get('sma_proximity_pct', 3.0)}%)")
            response = HTMLResponse(content=display_html)
            # Trigger strategyChanged event to show re-evaluate button (user must click to re-evaluate)
            response.headers["HX-Trigger"] = "strategyChanged"
            return response
        
        return JSONResponse(content={
            "success": True,
            "config": config_doc
        })
    except Exception as e:
        logger.error(f"Error updating strategy parameter: {e}", exc_info=True)
        error_msg = f"Failed to update parameter: {str(e)[:100]}"
        if is_htmx:
            # Return current strategy display on error with 200 OK (so HTMX can swap it)
            try:
                display_response = await get_strategy_display_html(db)
                display_html = display_response.body.decode('utf-8') if isinstance(display_response.body, bytes) else str(display_response.body)
                return HTMLResponse(content=display_html, status_code=200)
            except Exception:
                # Fallback: return minimal valid HTML
                return HTMLResponse(content="<div id='strategy-display-content'>Error updating parameter</div>", status_code=200)
        return JSONResponse(content={"success": False, "error": error_msg}, status_code=500)


async def generate_strategy_params_api(request: Request) -> JSONResponse:
    """Generate strategy parameters from goal description using AI."""
    try:
        body = await request.json()
        goal = body.get('goal', '').strip()
        budget = body.get('budget')  # Optional budget parameter
        
        if not goal:
            return JSONResponse(
                content={"success": False, "error": "Goal description is required"},
                status_code=400
            )
        
        if len(goal) > 280:
            return JSONResponse(
                content={"success": False, "error": "Goal must be 280 characters or less"},
                status_code=400
            )
        
        # Use AI to generate parameters
        try:
            from ..services.ai import EyeAI, StrategyConfig
            ai_engine = EyeAI()
            ai_config = ai_engine.generate_strategy_config(goal, budget)
            
            # Convert Pydantic model to dict
            config_dict = {
                "rsi_threshold": ai_config.rsi_threshold,
                "rsi_min": ai_config.rsi_min,
                "ai_score_required": ai_config.ai_score_required,
                "reasoning": ai_config.reasoning
            }
            
            logger.info(f"Generated strategy params from goal: {goal[:50]}... -> RSI<{config_dict['rsi_threshold']}, Score≥{config_dict['ai_score_required']}")
            
            return JSONResponse(content={
                "success": True,
                "config": config_dict
            })
        except Exception as ai_error:
            logger.error(f"AI parameter generation failed: {ai_error}", exc_info=True)
            # Fallback to default parameters
            return JSONResponse(content={
                "success": True,
                "config": {
                    "rsi_threshold": 35,
                    "rsi_min": 20,
                    "ai_score_required": 7,
                    "reasoning": "Using default parameters (AI generation unavailable)"
                },
                "warning": "AI generation unavailable, using default parameters"
            })
    except Exception as e:
        logger.error(f"Error generating strategy params: {e}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": str(e)[:100]},
            status_code=500
        )


async def get_firecrawl_query(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get current Firecrawl search query template - returns HTML for HTMX."""
    try:
        settings = await db.app_settings.find_one({"key": "firecrawl_search_query_template"})
        if settings:
            query_template = settings.get("value", FIRECRAWL_SEARCH_QUERY_TEMPLATE)
        else:
            query_template = FIRECRAWL_SEARCH_QUERY_TEMPLATE
        
        return HTMLResponse(content=templates.get_template("pages/firecrawl_query.html").render(
            query_template=query_template
        ))
    except Exception as e:
        logger.error(f"Error getting Firecrawl query: {e}", exc_info=True)
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"Error loading search settings: {str(e)[:100]}"
        ))

async def get_analysis_preview(db: Any = Depends(get_scoped_db)) -> JSONResponse:
    """Get preview of symbols that will be analyzed with current prices and pre-calculated technical indicators.
    
    Returns list of symbols with:
    - Current market prices (from Alpaca)
    - Pre-calculated technical indicators (RSI, SMA-200, ATR, Trend) using vectorbt
    - Quick signal preview (meets criteria check)
    
    This provides a "glass box" view - users see what will be analyzed before clicking ANALYZE.
    """
    try:
        symbols = await _get_target_symbols(db)
        
        if not symbols:
            return JSONResponse(content={
                "symbols": [],
                "message": "No symbols configured"
            })
        
        config = await get_strategy_config_dict(db)
        rsi_threshold = config.get('rsi_threshold', 35)
        sma_proximity_pct = config.get('sma_proximity_pct', 3.0)
        
        preview_data = []
        for symbol in symbols:
            try:
                current_price = None
                change_pct = None
                rsi = None
                sma_200 = None
                atr = None
                trend = None
                meets_criteria = False
                rejection_reason = None
                
                if not api:
                    logger.error(f"❌ Alpaca API not initialized - cannot fetch data for {symbol}")
                    rejection_reason = "Alpaca API not configured"
                else:
                    try:
                        logger.info(f"📊 Fetching market data for {symbol}")
                        bars, _, _ = await get_market_data(symbol, days=250, db=db)
                        
                        if bars is None or bars.empty:
                            logger.error(f"❌ No bars returned for {symbol} - bars is {bars}")
                            rejection_reason = "No market data returned from API"
                        elif len(bars) < 14:
                            logger.error(f"❌ Insufficient bars for {symbol}: {len(bars)} bars (need 14+)")
                            rejection_reason = f"Insufficient data: only {len(bars)} bars (need 14+)"
                        else:
                            logger.info(f"✅ Got {len(bars)} bars for {symbol}")
                        
                        if bars is not None and not bars.empty and len(bars) >= 14:
                            current_price = float(bars.iloc[-1]['close'])
                            
                            if len(bars) >= 2:
                                prev_close = float(bars.iloc[-2]['close'])
                                if prev_close > 0:
                                    change_pct = ((current_price - prev_close) / prev_close) * 100
                            
                            try:
                                techs = analyze_technicals(bars)
                                rsi = techs.get('rsi')
                                sma_200 = techs.get('sma')
                                atr = techs.get('atr')
                                trend = techs.get('trend')
                                
                                reasons = []
                                if rsi is not None:
                                    if rsi >= rsi_threshold:
                                        reasons.append(f"RSI {rsi:.1f} ≥ {rsi_threshold} (not oversold)")
                                else:
                                    reasons.append("RSI unavailable")
                                
                                if trend != "UP":
                                    if trend == "DOWN":
                                        reasons.append(f"Downtrend (Price ${current_price:.2f} < SMA-200 ${sma_200:.2f})")
                                    else:
                                        reasons.append("Trend unclear")
                                
                                # Check SMA-200 proximity
                                price_vs_sma_pct = None
                                if current_price and sma_200 and sma_200 > 0:
                                    price_vs_sma_pct = ((current_price - sma_200) / sma_200) * 100
                                    if price_vs_sma_pct > sma_proximity_pct:
                                        reasons.append(f"Price {price_vs_sma_pct:.1f}% above SMA-200 (exceeds {sma_proximity_pct}% proximity limit)")
                                
                                # Entry criteria: RSI < threshold AND uptrend AND within SMA-200 proximity
                                meets_criteria = (
                                    rsi is not None and rsi < rsi_threshold and
                                    trend == "UP" and
                                    (price_vs_sma_pct is None or price_vs_sma_pct <= sma_proximity_pct)
                                )
                                
                                rejection_reason = "; ".join(reasons) if reasons and not meets_criteria else None
                                
                            except Exception as e:
                                logger.error(f"❌ Could not calculate indicators for {symbol}: {e}", exc_info=True)
                                rejection_reason = f"Indicator calculation failed: {str(e)[:50]}"
                                meets_criteria = False
                        else:
                            # Bars were None or empty - already logged above
                            if not rejection_reason:
                                rejection_reason = "No market data available"
                                
                    except Exception as e:
                        logger.error(f"❌ Could not get market data for {symbol}: {e}", exc_info=True)
                        rejection_reason = f"API error: {str(e)[:100]}"
                        # Fallback: try to get just price
                        try:
                            if api:
                                logger.info(f"🔄 Trying fallback price fetch for {symbol}")
                                bars = api.get_bars(symbol, TimeFrame.Day, limit=1, feed='iex').df
                                if bars is not None and not bars.empty:
                                    current_price = float(bars.iloc[-1]['close'])
                                    logger.info(f"✅ Fallback succeeded for {symbol}: ${current_price}")
                                else:
                                    logger.error(f"❌ Fallback returned empty data for {symbol}")
                        except Exception as fallback_error:
                            logger.error(f"❌ Fallback price fetch failed for {symbol}: {fallback_error}", exc_info=True)
                
                preview_data.append({
                    "symbol": symbol,
                    "price": round(current_price, 2) if current_price else None,
                    "change_pct": round(change_pct, 2) if change_pct else None,
                    "rsi": round(rsi, 2) if rsi is not None else None,
                    "sma_200": round(sma_200, 2) if sma_200 is not None else None,
                    "atr": round(atr, 2) if atr is not None else None,
                    "trend": trend,
                    "price_vs_sma_pct": round(price_vs_sma_pct, 2) if price_vs_sma_pct is not None else None,
                    "meets_criteria": meets_criteria,
                    "rejection_reason": rejection_reason or (None if current_price else "Unable to fetch market data"),
                    "available": current_price is not None
                })
            except Exception as e:
                logger.error(f"Could not get preview data for {symbol}: {e}", exc_info=True)
                preview_data.append({
                    "symbol": symbol,
                    "price": None,
                    "change_pct": None,
                    "rsi": None,
                    "sma_200": None,
                    "atr": None,
                    "trend": None,
                    "meets_criteria": False,
                    "rejection_reason": f"Data unavailable: {str(e)[:50]}",
                    "available": False
                })
        
        # Count stocks that meet entry criteria
        ready_count = sum(1 for s in preview_data if s.get("meets_criteria", False))
        
        logger.info(f"📊 Analysis preview: {len(preview_data)} symbols, {ready_count} ready, {sum(1 for s in preview_data if s['available'])} available, api={api is not None}")
        logger.info(f"📊 Preview data sample: {preview_data[0] if preview_data else 'No data'}")
        return JSONResponse(content={
            "symbols": preview_data,
            "total": len(preview_data),
            "available_count": sum(1 for s in preview_data if s["available"]),
            "ready_count": ready_count,
            "can_analyze": ready_count > 0,
            "rsi_threshold": rsi_threshold
        })
    except Exception as e:
        logger.error(f"Error getting analysis preview: {e}", exc_info=True)
        return JSONResponse(content={
            "symbols": [],
            "error": str(e)
        }, status_code=500)


async def get_watch_list(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get current watch list display with integrated pre-calculated indicators.
    
    Combines watchlist management with preview indicators for a streamlined UX.
    """
    try:
        # Get watch list from database
        watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
        if watch_list_settings and watch_list_settings.get("value"):
            symbols_str = watch_list_settings.get("value", "")
            symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
        else:
            # Fallback to default from config
            from ..core.config import SYMBOLS
            symbols = SYMBOLS.copy() if SYMBOLS else []
        
        # Ensure symbols is always a list
        if not isinstance(symbols, list):
            symbols = []
        
        from ..core.config import get_strategy_config, calculate_watchlist_config_hash
        strategy_config = await get_strategy_config(db)
        rsi_threshold = strategy_config.get('rsi_threshold', 35)
        rsi_min = strategy_config.get('rsi_min', 20)
        sma_proximity_pct = strategy_config.get('sma_proximity_pct', 3.0)
        
        # Calculate config hash for smart re-evaluation detection
        config_hash = calculate_watchlist_config_hash(strategy_config)
        
        # Log watchlist re-evaluation with current strategy parameters
        logger.info(f"📋 Re-evaluating watchlist with strategy params: RSI {rsi_min}-{rsi_threshold}, SMA proximity {sma_proximity_pct}% (config hash: {config_hash[:8]}...)")
        
        # Get preview data for symbols (pre-calculated indicators)
        # Parallelize data fetching for better performance
        preview_data = []
        ready_count = 0
        if symbols:
            try:
                import asyncio
                import time
                from ..services.analysis import get_market_data, analyze_technicals
                from ..services.progress import calculate_rsi_progress, calculate_sma_progress
                
                logger.info(f"📋 Processing {len(symbols)} symbols in parallel for watchlist")
                start_time = time.time()
                
                async def process_symbol(symbol: str) -> dict:
                    """Process a single symbol and return preview data."""
                    try:
                        # Check evaluation cache first
                        from ..services.watchlist_cache import get_cache
                        cache = get_cache()
                        cached_eval = cache.get_evaluation(symbol, config_hash, ttl_seconds=60)
                        if cached_eval:
                            logger.debug(f"Using cached evaluation for {symbol}")
                            return cached_eval
                        
                        bars, _, _ = await get_market_data(symbol, days=100, db=db)
                        current_price = None
                        rsi = None
                        trend = None
                        meets_criteria = False
                        
                        if bars is not None and not bars.empty and len(bars) >= 14:
                            current_price = float(bars.iloc[-1]['close'])
                            current_volume = float(bars.iloc[-1].get('volume', 0)) if 'volume' in bars.columns else None
                            techs = analyze_technicals(bars)
                            rsi = techs.get('rsi')
                            trend = techs.get('trend', 'NEUTRAL')
                            sma_200 = techs.get('sma')
                            atr = techs.get('atr')
                            
                            # Get additional metrics
                            earnings_in_days = techs.get('earnings_in_days')
                            pe_ratio = techs.get('pe_ratio')
                            market_cap = techs.get('market_cap')
                            
                            # Calculate price vs SMA percentage
                            price_vs_sma_pct = None
                            if current_price and sma_200:
                                price_vs_sma_pct = round(((current_price - sma_200) / sma_200) * 100, 1)
                            
                            # Calculate progress percentages for UI using utility functions
                            rsi_progress_pct = calculate_rsi_progress(rsi, rsi_threshold)
                            sma_progress_pct = calculate_sma_progress(price_vs_sma_pct, sma_proximity_pct)
                            
                            passes_filters = True
                            earnings_days_min = strategy_config.get('earnings_days_min', 3)
                            if earnings_in_days is not None and earnings_in_days < earnings_days_min:
                                passes_filters = False
                            
                            pe_ratio_max = strategy_config.get('pe_ratio_max', 50.0)
                            if pe_ratio and pe_ratio > pe_ratio_max:
                                passes_filters = False
                            
                            market_cap_min = strategy_config.get('market_cap_min', 300_000_000)
                            if market_cap and market_cap < market_cap_min:
                                passes_filters = False
                            
                            # Check if meets entry criteria (RSI in sweet spot: rsi_min < RSI < rsi_threshold AND uptrend AND within SMA-200 proximity)
                            # This matches the strategy's generate_signals logic exactly
                            meets_criteria = (
                                passes_filters and
                                current_price is not None and
                                rsi is not None and
                                rsi > rsi_min and  # Not too extreme (avoid RSI < rsi_min)
                                rsi < rsi_threshold and  # Oversold (RSI < threshold)
                                trend == "UP" and
                                (price_vs_sma_pct is None or price_vs_sma_pct <= sma_proximity_pct)
                            )
                            
                            # Build rejection reason if not a candidate
                            rejection_reason = None
                            if not meets_criteria:
                                reasons = []
                                if not passes_filters:
                                    if earnings_in_days is not None and earnings_in_days < earnings_days_min:
                                        reasons.append(f"Earnings in {earnings_in_days} day(s)")
                                    if pe_ratio and pe_ratio > pe_ratio_max:
                                        reasons.append(f"P/E {pe_ratio:.1f} > {pe_ratio_max:.0f}")
                                    if market_cap and market_cap < market_cap_min:
                                        reasons.append(f"Market cap < ${market_cap_min/1_000_000:.0f}M")
                                if rsi is not None:
                                    if rsi <= rsi_min:
                                        reasons.append(f"RSI {rsi:.1f} ≤ {rsi_min} (too extreme oversold)")
                                    elif rsi >= rsi_threshold:
                                        reasons.append(f"RSI {rsi:.1f} ≥ {rsi_threshold} (not oversold)")
                                if trend != "UP":
                                    if trend == "DOWN":
                                        reasons.append(f"Downtrend (Price ${current_price:.2f} < SMA-200 ${sma_200:.2f})")
                                    else:
                                        reasons.append("Trend unclear")
                                if price_vs_sma_pct is not None and price_vs_sma_pct > sma_proximity_pct:
                                    reasons.append(f"Price {price_vs_sma_pct:.1f}% above SMA-200 (exceeds {sma_proximity_pct}% proximity limit)")
                                rejection_reason = "; ".join(reasons) if reasons else "Does not meet entry criteria"
                        else:
                            sma_200 = None
                            atr = None
                            price_vs_sma_pct = None
                            current_volume = None
                            earnings_in_days = None
                            pe_ratio = None
                            market_cap = None
                            rsi_progress_pct = None
                            sma_progress_pct = None
                            rejection_reason = "Insufficient data"
                        
                        result = {
                            "symbol": symbol,
                            "price": round(current_price, 2) if current_price else None,
                            "rsi": round(rsi, 2) if rsi else None,
                            "trend": trend,
                            "sma_200": round(sma_200, 2) if sma_200 else None,
                            "atr": round(atr, 2) if atr else None,
                            "price_vs_sma_pct": price_vs_sma_pct,
                            "volume": current_volume,
                            "earnings_in_days": earnings_in_days,
                            "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                            "market_cap": market_cap,
                            "rsi_progress_pct": round(rsi_progress_pct, 1) if rsi_progress_pct is not None else None,
                            "sma_progress_pct": round(sma_progress_pct, 1) if sma_progress_pct is not None else None,
                            "meets_criteria": meets_criteria,
                            "rejection_reason": rejection_reason if not meets_criteria else None
                        }
                        
                        # Cache the evaluation result
                        try:
                            cache.set_evaluation(symbol, config_hash, result)
                        except Exception as e:
                            logger.debug(f"Failed to cache evaluation for {symbol}: {e}")
                        
                        return result
                    except Exception as e:
                        logger.error(f"Error processing symbol {symbol} in watchlist: {e}", exc_info=True)
                        error_result = {
                            "symbol": symbol,
                            "price": None,
                            "rsi": None,
                            "trend": None,
                            "sma_200": None,
                            "atr": None,
                            "price_vs_sma_pct": None,
                            "volume": None,
                            "earnings_in_days": None,
                            "pe_ratio": None,
                            "market_cap": None,
                            "rsi_progress_pct": None,
                            "sma_progress_pct": None,
                            "meets_criteria": False,
                            "error": str(e)[:100]  # Include error message for UI display
                        }
                        # Don't cache error results
                        return error_result
                
                # Process all symbols in parallel with timeout protection
                tasks = [process_symbol(symbol) for symbol in symbols]
                # Use gather with return_exceptions to handle individual failures gracefully
                # Each symbol has a 30s timeout (via get_market_data internal timeouts)
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                elapsed_time = time.time() - start_time
                logger.info(f"✅ Processed {len(symbols)} symbols in {elapsed_time:.2f}s ({elapsed_time/len(symbols):.2f}s per symbol avg)")
                
                # Collect results and count ready symbols
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Exception in parallel processing: {result}", exc_info=True)
                        continue
                    preview_data.append(result)
                    if result.get("meets_criteria", False):
                        ready_count += 1
                
                # Sort preview_data to match original symbol order
                symbol_to_data = {item["symbol"]: item for item in preview_data}
                preview_data = [symbol_to_data.get(symbol, {
                    "symbol": symbol,
                    "price": None,
                    "rsi": None,
                    "trend": None,
                    "sma_200": None,
                    "atr": None,
                    "price_vs_sma_pct": None,
                    "meets_criteria": False,
                    "error": None
                }) for symbol in symbols]
                
            except Exception as e:
                logger.warning(f"Could not load preview data: {e}", exc_info=True)
        
        # Prepare preview data for template with pre-calculated styling
        # Also generate logo SVGs for each symbol
        from ..services.logo import get_svg_initials
        template_preview_data = []
        if symbols and preview_data:
            for preview in preview_data:
                symbol = preview["symbol"]
                price = preview.get("price")
                rsi = preview.get("rsi")
                trend = preview.get("trend")
                # Generate logo SVG for this symbol
                preview["logo_svg"] = get_svg_initials(symbol)
                sma_200 = preview.get("sma_200")
                atr = preview.get("atr")
                price_vs_sma_pct = preview.get("price_vs_sma_pct")
                meets_criteria = preview.get("meets_criteria", False)
                
                # Pre-calculate styling to reduce template complexity
                rsi_color = "text-gray-400"
                if rsi is not None:
                    if rsi < 35:
                        rsi_color = "text-green-400"
                    elif rsi < 50:
                        rsi_color = "text-yellow-400"
                    else:
                        rsi_color = "text-red-400"
                
                trend_icon = "minus"
                trend_color = "text-gray-400"
                if trend == "UP":
                    trend_icon = "trending-up"
                    trend_color = "text-green-400"
                elif trend == "DOWN":
                    trend_icon = "trending-down"
                    trend_color = "text-red-400"
                
                # Price vs SMA styling
                price_vs_sma_color = "text-gray-400"
                price_vs_sma_icon = "minus"
                if price_vs_sma_pct is not None:
                    if price_vs_sma_pct > 0:
                        price_vs_sma_color = "text-green-400"
                        price_vs_sma_icon = "arrow-up"
                    else:
                        price_vs_sma_color = "text-red-400"
                        price_vs_sma_icon = "arrow-down"
                
                border_class = "border-blue-500/30"
                bg_class = "bg-blue-500/5"
                if meets_criteria:
                    border_class = "border-green-500/40"
                    bg_class = "bg-green-500/10"
                elif price is None:
                    border_class = "border-gray-500/20"
                    bg_class = "bg-gray-500/5"
                
                template_preview_data.append({
                    "symbol": symbol,
                    "price": price,
                    "rsi": rsi,
                    "trend": trend,
                    "sma_200": sma_200,
                    "atr": atr,
                    "price_vs_sma_pct": price_vs_sma_pct,
                    "volume": preview.get("volume"),
                    "earnings_in_days": preview.get("earnings_in_days"),
                    "pe_ratio": preview.get("pe_ratio"),
                    "market_cap": preview.get("market_cap"),
                    "rsi_progress_pct": preview.get("rsi_progress_pct"),
                    "sma_progress_pct": preview.get("sma_progress_pct"),
                    "meets_criteria": meets_criteria,
                    "rejection_reason": preview.get("rejection_reason"),
                    "error": preview.get("error"),  # Include error if present
                    "rsi_color": rsi_color,
                    "trend_icon": trend_icon,
                    "trend_color": trend_color,
                    "price_vs_sma_color": price_vs_sma_color,
                    "price_vs_sma_icon": price_vs_sma_icon,
                    "border_class": border_class,
                    "bg_class": bg_class,
                    "logo_svg": preview.get("logo_svg", get_svg_initials(symbol))
                })
        
        # Render template
        response = HTMLResponse(content=templates.get_template("pages/watchlist.html").render(
            symbols=symbols,
            preview_data=template_preview_data if template_preview_data else None,
            ready_count=ready_count,
            rsi_threshold=rsi_threshold,
            sma_proximity_pct=sma_proximity_pct,
            config_hash=config_hash,
            evaluation_timestamp=datetime.now().isoformat()
        ))
        # Add config hash to response headers for frontend access
        response.headers["X-Config-Hash"] = config_hash
        response.headers["X-Evaluation-Timestamp"] = datetime.now().isoformat()
        return response
    except Exception as e:
        logger.error(f"Error loading watch list: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Watchlist error traceback: {error_details}")
        # Return watchlist template with empty data rather than error message
        # This ensures the UI doesn't break
        try:
            # Get config hash even in error case
            from ..core.config import get_strategy_config, calculate_watchlist_config_hash
            try:
                error_strategy_config = await get_strategy_config(db)
                error_config_hash = calculate_watchlist_config_hash(error_strategy_config)
            except:
                error_config_hash = "unknown"
            
            response = HTMLResponse(content=templates.get_template("pages/watchlist.html").render(
                symbols=symbols if 'symbols' in locals() else [],
                preview_data=None,
                ready_count=0,
                rsi_threshold=35,
                config_hash=error_config_hash,
                evaluation_timestamp=None
            ))
            response.headers["X-Config-Hash"] = error_config_hash
            return response
        except Exception as template_error:
            logger.error(f"Error rendering watchlist template: {template_error}", exc_info=True)
            return HTMLResponse(content=templates.get_template("partials/status_message.html").render(
                message=f"Error loading watch list: {str(e)[:100]}",
                color="red"
            ))


def get_logo(ticker: str, size: str = Query("14", description="Logo size in pixels")) -> HTMLResponse:
    """Get company logo HTML fragment (SVG initials).
    
    Returns SVG initials synchronously - no API calls, instant rendering.
    
    Args:
        ticker: Stock ticker symbol
        size: Size in pixels (default: 14, for w-14 h-14)
        
    Returns:
        HTMLResponse with SVG initials element
    """
    try:
        ticker = ticker.upper().strip() if ticker else "?"
        
        # Parse size (handle both "14" and "w-14 h-14" formats)
        try:
            if isinstance(size, str) and size.replace('-', '').replace('w', '').replace('h', '').replace(' ', '').isdigit():
                # Extract number from "w-14 h-14" or just "14"
                size_num = int(''.join(filter(str.isdigit, size.split()[0] if ' ' in size else size)))
            else:
                size_num = int(size) if size.isdigit() else 14
        except (ValueError, AttributeError):
            size_num = 14
        
        # Use size for CSS class
        css_class = f"company-logo w-{size_num} h-{size_num} rounded-full"
        from ..services.logo import get_logo_html
        logo_html = get_logo_html(ticker, css_class=css_class)
        return HTMLResponse(content=logo_html)
    
    except Exception as e:
        logger.warning(f"Error generating logo for {ticker}: {e}")
        # Fallback to SVG initials using template
        from ..services.logo import get_svg_initials
        svg = get_svg_initials(ticker if ticker else "?")
        return HTMLResponse(content=templates.get_template("components/logo.html").render(
            symbol=ticker if ticker else "?",
            logo_svg=svg,
            size_class="w-14 h-14"
        ))


async def search_tickers(query: str = None) -> JSONResponse:
    """Search for available tickers using Alpaca API.
    
    Returns a list of tickers that match the query (if provided) or popular tickers.
    Uses Alpaca's list_assets API to get tradable stocks.
    """
    try:
        if not api:
            # Fallback to popular tickers if Alpaca API not available
            popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "NFLX"]
            if query:
                query_upper = query.upper()
                filtered = [t for t in popular_tickers if query_upper in t]
                return JSONResponse(content={"tickers": filtered[:20]})
            return JSONResponse(content={"tickers": popular_tickers[:20]})
        
        # Use Alpaca API to search for assets
        try:
            # Get all assets (stocks only, tradable)
            assets = api.list_assets(status='active', asset_class='us_equity')
            
            # Filter by query if provided
            if query:
                query_upper = query.upper().strip()
                filtered = [
                    asset.symbol for asset in assets 
                    if query_upper in asset.symbol.upper() and asset.tradable
                ]
                # Sort by match position (exact matches first)
                filtered.sort(key=lambda x: (not x.startswith(query_upper), x))
                return JSONResponse(content={"tickers": filtered[:50]})
            else:
                # Return popular/well-known tickers if no query
                popular_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "NFLX", 
                                  "JPM", "V", "JNJ", "WMT", "MA", "PG", "DIS", "BAC", "XOM", "CVX"]
                # Also include some from Alpaca's list
                alpaca_symbols = [asset.symbol for asset in assets[:30] if asset.tradable]
                combined = list(dict.fromkeys(popular_symbols + alpaca_symbols))  # Remove duplicates, preserve order
                return JSONResponse(content={"tickers": combined[:50]})
        except Exception as e:
            logger.warning(f"Error fetching tickers from Alpaca: {e}")
            # Fallback to popular tickers
            popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "NFLX"]
            if query:
                query_upper = query.upper()
                filtered = [t for t in popular_tickers if query_upper in t]
                return JSONResponse(content={"tickers": filtered[:20]})
            return JSONResponse(content={"tickers": popular_tickers[:20]})
    except Exception as e:
        logger.error(f"Error searching tickers: {e}", exc_info=True)
        return JSONResponse(
            content={"tickers": [], "error": str(e)[:100]},
            status_code=500
        )


async def update_watch_list(request: Request, db: Any = Depends(get_scoped_db)) -> JSONResponse:
    """Update watch list - supports add, remove, and bulk update operations.
    
    No longer limited to 5 tickers - can have many tickers in watchlist.
    """
    try:
        body = await request.json()
        action = body.get("action", "bulk")  # "add", "remove", or "bulk"
        
        # Get current watch list
        watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
        if watch_list_settings and watch_list_settings.get("value"):
            current_symbols_str = watch_list_settings.get("value", "")
            current_symbols = [s.strip().upper() for s in current_symbols_str.split(',') if s.strip()]
        else:
            # Fallback to default from config
            from ..core.config import SYMBOLS
            current_symbols = SYMBOLS.copy()
        
        if action == "add":
            # Add a single ticker
            symbol = body.get("symbol", "").strip().upper()
            if not symbol:
                return JSONResponse(
                    content={"success": False, "error": "Symbol is required"},
                    status_code=400
                )
            
            # Validate symbol format
            import re
            if not re.match(r'^[A-Z]{1,5}$', symbol):
                return JSONResponse(
                    content={"success": False, "error": f"Invalid symbol format: {symbol}"},
                    status_code=400
                )
            
            # Check if already exists
            if symbol in current_symbols:
                return JSONResponse(
                    content={"success": False, "error": f"{symbol} is already in watch list"},
                    status_code=400
                )
            
            # Add symbol (no limit)
            current_symbols.append(symbol)
            
        elif action == "remove":
            # Remove a single ticker
            symbol = body.get("symbol", "").strip().upper()
            if not symbol:
                return JSONResponse(
                    content={"success": False, "error": "Symbol is required"},
                    status_code=400
                )
            
            if symbol not in current_symbols:
                return JSONResponse(
                    content={"success": False, "error": f"{symbol} is not in watch list"},
                    status_code=400
                )
            
            # Remove symbol
            current_symbols.remove(symbol)
            
        else:
            # Bulk update (comma-separated)
            symbols_str = body.get("symbols", "").strip()
            
            if not symbols_str:
                return JSONResponse(
                    content={"success": False, "error": "Symbols cannot be empty"},
                    status_code=400
                )
            
            # Parse and validate symbols
            symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
            
            if len(symbols) == 0:
                return JSONResponse(
                    content={"success": False, "error": "At least one symbol is required"},
                    status_code=400
                )
            
            # Validate symbol format (1-5 uppercase letters)
            import re
            for symbol in symbols:
                if not re.match(r'^[A-Z]{1,5}$', symbol):
                    return JSONResponse(
                        content={"success": False, "error": f"Invalid symbol format: {symbol}"},
                        status_code=400
                    )
            
            # Remove duplicates while preserving order
            seen = set()
            current_symbols = []
            for symbol in symbols:
                if symbol not in seen:
                    seen.add(symbol)
                    current_symbols.append(symbol)
        
        # Store in MongoDB app_settings collection
        symbols_value = ", ".join(current_symbols)
        await db.app_settings.update_one(
            {"key": "watch_list"},
            {"$set": {"value": symbols_value, "updated_at": datetime.now()}},
            upsert=True
        )
        
        logger.info(f"Watch list updated: {symbols_value}")
        
        # Check if this is an HTMX request by checking headers
        is_htmx = request.headers.get("HX-Request") == "true"
        
        # For HTMX remove requests, return updated watchlist HTML directly
        if is_htmx and action == "remove":
            # Return the updated watchlist HTML by calling get_watch_list
            # Import here to avoid circular dependency
            watchlist_html_response = await get_watch_list(db)
            
            # Add undo toast notification
            from .templates import toast_notification
            removed_symbol = body.get("symbol", "").strip().upper()
            undo_toast = toast_notification(
                message=f"{removed_symbol} removed from watchlist",
                type="success",
                duration=5000,
                target_id="toast-container",
                undo_action="/api/watch-list",
                undo_symbol=removed_symbol
            )
            
            # Append toast to response using hx-swap-oob
            # The toast HTML already has hx-swap-oob="beforeend" so it will be appended
            response_content = watchlist_html_response.body.decode('utf-8') + undo_toast
            
            return HTMLResponse(content=response_content)
        
        # For non-HTMX requests (API calls), return JSON
        response_content = {
            "success": True,
            "message": "Watch list updated successfully",
            "symbols": symbols_value,
            "symbols_list": current_symbols
        }
        
        response = JSONResponse(content=response_content)
        # Add HX-Trigger header to reload watchlist after remove (for non-HTML responses)
        if action == "remove" and not is_htmx:
            response.headers["HX-Trigger"] = "watchlistUpdated"
        
        return response
    except Exception as e:
        logger.error(f"Error updating watch list: {e}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": str(e)[:100]},
            status_code=500
        )


async def update_target_symbols(request: Request, db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Update target symbols for initial search."""
    try:
        body = await request.json()
        symbols_str = body.get("symbols", "").strip()
        
        if not symbols_str:
            return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                message="Error: Symbols cannot be empty"
            ))
        
        # Parse and validate symbols
        symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
        
        if len(symbols) > 5:
            return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                message="Error: Maximum 5 symbols allowed"
            ))
        
        if len(symbols) == 0:
            return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                message="Error: At least one symbol is required"
            ))
        
        # Validate symbol format (basic check - alphanumeric, 1-5 chars)
        for symbol in symbols:
            if not symbol.isalnum() or len(symbol) < 1 or len(symbol) > 5:
                return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                    message=f"Error: Invalid symbol format: {symbol}"
                ))
        
        # Store in MongoDB app_settings collection
        symbols_value = ", ".join(symbols)
        await db.app_settings.update_one(
            {"key": "target_symbols"},
            {"$set": {"value": symbols_value, "updated_at": datetime.now()}},
            upsert=True
        )
        
        logger.info(f"Target symbols updated: {symbols_value}")
        
        # Return HTML response for HTMX to update the UI
        success_html = templates.get_template("partials/success_message.html").render(
            message="Target symbols updated successfully!",
            details="Changes will take effect on the next analysis scan."
        )
        # HTMX Gold Standard: Use HX-Trigger header instead of deprecated htmx_trigger.html
        response = HTMLResponse(content=success_html)
        response.headers["HX-Trigger"] = "refreshWatchList"
        return response
    except Exception as e:
        logger.error(f"Error updating target symbols: {e}", exc_info=True)
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"Error: {str(e)[:100]}"
        ))

async def update_firecrawl_query(request: Request, db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Update Firecrawl search query template."""
    try:
        body = await request.json()
        query_template = body.get("query_template", "").strip()
        
        if not query_template:
            return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                message="Error: Search query cannot be empty"
            ))
        
        # Validate that {symbol} placeholder exists
        if "{symbol}" not in query_template:
            return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                message="Error: Query must contain {symbol} placeholder"
            ))
        
        # Store in MongoDB app_settings collection
        await db.app_settings.update_one(
            {"key": "firecrawl_search_query_template"},
            {"$set": {"value": query_template, "updated_at": datetime.now()}},
            upsert=True
        )
        
        logger.info(f"Firecrawl search query template updated: {query_template}")
        
        # Return HTML response for HTMX to update the UI
        success_html = templates.get_template("partials/success_message.html").render(
            message="Search query updated successfully!",
            details="Changes will take effect on the next market scan."
        )
        # HTMX Gold Standard: Use HX-Trigger header instead of deprecated htmx_trigger.html
        response = HTMLResponse(content=success_html)
        response.headers["HX-Trigger"] = "refreshFirecrawlQuery"
        return response
    except Exception as e:
        logger.error(f"Error updating Firecrawl query: {e}", exc_info=True)
        return HTMLResponse(content=templates.get_template("partials/status_message.html").render(
            message=f"Error: {str(e)[:100]}",
            color="red"
        ))

async def panic_close() -> HTMLResponse:
    """Close all positions and cancel all orders."""
    if not check_alpaca_config():
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message="Alpaca API not configured"
        ))
    
    if api is None:
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message="Alpaca API not initialized"
        ))
    
    try:
        api.cancel_all_orders()
        api.close_all_positions()
        logger.critical("⚫ FLUX closes all positions - All capital extracted")
        # HTMX Gold Standard: Use template instead of f-string HTML
        return HTMLResponse(content=templates.get_template("partials/status_message.html").render(
            message="All positions closed",
            color="red"
        ))
    except Exception as e:
        logger.error(f"Failed to panic close: {e}", exc_info=True)
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"Error: {str(e)[:50]}"
        ))

async def _search_trending_stocks_internal() -> dict:
    """Internal function to search trending stocks - returns curated list.
    
    Returns a curated list of popular tech stocks for discovery.
    Firecrawl integration removed - using curated list for reliability.
    """
    # Curated list of popular tech stocks for discovery
    fallback_stocks = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "AMD", "META"]
    return {
        "success": True,
        "stocks": fallback_stocks,
        "source": "curated"
    }

async def search_trending_stocks() -> JSONResponse:
    """Public endpoint to search trending stocks."""
    result = await _search_trending_stocks_internal()
    return JSONResponse(content=result)

async def _get_trending_stocks_with_analysis_internal() -> dict:
    """Internal function to get trending stocks with analysis - returns dict."""
    try:
        # Get trending stocks
        search_data = await _search_trending_stocks_internal()
        
        if not search_data.get('success') or not search_data.get('stocks'):
            # Fallback to curated list
            symbols = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "AMD"]
        else:
            symbols = search_data['stocks'][:6]  # Top 6 trending stocks
        
        # Analyze each stock
        stocks_with_analysis = []
        for symbol in symbols:
            try:
                bars, headlines, news_objects = await get_market_data(symbol, days=500, db=db)
                if bars is None or bars.empty or len(bars) < 14:
                    continue
                
                techs = analyze_technicals(bars)
                
                # Get AI analysis if available
                ai_score = None
                ai_reason = None
                risk_level = "UNKNOWN"
                
                if _ai_engine:
                    try:
                        strategy_prompt = get_balanced_low_prompt()
                        verdict = _ai_engine.analyze(
                            ticker=symbol,
                            techs=techs,
                            headlines="\n".join(headlines[:3]) if headlines else "No recent news",
                            strategy_prompt=strategy_prompt
                        )
                        ai_score = verdict.score
                        ai_reason = verdict.reason[:100] + "..." if len(verdict.reason) > 100 else verdict.reason
                        risk_level = verdict.risk_level
                    except Exception as e:
                        logger.warning(f"AI analysis failed for {symbol}: {e}")
                
                stocks_with_analysis.append({
                    "symbol": symbol,
                    "price": techs['price'],
                    "rsi": techs['rsi'],
                    "atr": techs['atr'],
                    "trend": techs['trend'],
                    "sma": techs['sma'],
                    "ai_score": ai_score,
                    "ai_reason": ai_reason,
                    "risk_level": risk_level
                })
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                continue
        
        return {
            "success": True,
            "stocks": stocks_with_analysis,
            "source": search_data.get('source', 'fallback')
        }
        
    except Exception as e:
        logger.error(f"Failed to get trending stocks with analysis: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }

async def get_trending_stocks_with_analysis() -> JSONResponse:
    """Public endpoint to get trending stocks with analysis."""
    result = await _get_trending_stocks_with_analysis_internal()
    return JSONResponse(content=result)

async def quick_buy(symbol: str = Form(...), qty: Optional[int] = Form(10), db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Quick buy order from stock card - returns HTML with hx-swap-oob for multiple updates.
    
    HTMX Pattern: Returns HTML with hx-swap-oob="true" to update multiple elements:
    - Updates #positions-list with fresh positions
    - Adds toast notification to #toast-container
    This demonstrates htmx's out-of-band swap capability.
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) for database access.
    
    Args:
        symbol: Stock symbol to buy
        qty: Number of shares (default: 10, range: 1-99)
    """
    if not check_alpaca_config() or api is None:
        return HTMLResponse(content=toast_notification(
            message="Alpaca API not configured",
            type="error"
        ), status_code=400)
    
    try:
        # Validate symbol
        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("Symbol is required")
        
        # Validate quantity
        if qty is None:
            qty = 10  # Default
        try:
            qty = int(qty)
            if qty < 1 or qty > 99:
                raise ValueError("Quantity must be between 1-99")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid quantity: {qty}. Must be between 1-99.")
        
        logger.info(f"Quick buy requested for {symbol}, quantity: {qty}")
        
        # Check API is ready
        if api is None:
            raise ValueError("Alpaca API not initialized - check API credentials")
        
        # Place order with specified quantity
        order = await place_order(symbol, 'buy', f"Quick buy from card ({qty} shares)", 10, db, qty=qty)
        
        if order is None:
            # place_order returns None on failure - check logs for details
            error_msg = f"Order placement failed for {symbol}. Check server logs for details."
            logger.error(f"place_order returned None for {symbol}")
            return HTMLResponse(content=toast_notification(
                message=f"{error_msg}. Possible reasons: No market data, position size too small, or API error",
                type="error"
            ), status_code=500)
            return HTMLResponse(content=error_html, status_code=500)
        
        # Order succeeded - get updated positions
        logger.info(f"Buy order placed successfully for {symbol}: {order}")
        
        # Get updated positions HTML - pass db explicitly
        positions_response = await get_positions(db=db)
        positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
        
        # Return HTML with hx-swap-oob to update positions and show success toast
        order_id = getattr(order, 'id', 'N/A')
        # Build toast message with order ID if available
        toast_message = f"Buy order placed for {symbol}"
        if order_id != 'N/A':
            # Use proper HTML structure instead of raw HTML string to avoid parsing issues
            toast_message = f"Buy order placed for {symbol}\nOrder ID: {order_id}"
        
        success_toast = toast_notification(
            message=toast_message,
            type="success",
            duration=5000
        )
        
        response_content = htmx_response(
            updates={
                "positions-list": positions_html,
                "toast-container": success_toast
            }
        )
        response = HTMLResponse(content=response_content)
        response.headers["HX-Trigger"] = "refreshPositions"
        return response
        
    except Exception as e:
        logger.error(f"Quick buy failed for {symbol}: {e}", exc_info=True)
        error_msg = str(e)[:150] if str(e) else "Unknown error occurred"
        return HTMLResponse(content=toast_notification(
            message=f"Buy failed: {error_msg}. Check server logs for details",
            type="error"
        ), status_code=500)
        return HTMLResponse(content=error_html, status_code=500)

async def get_latest_scan(
    date: Optional[str] = None,
    db: Any = Depends(get_scoped_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> JSONResponse:
    """Get latest scan (or final scan if after 6pm ET).
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) and Depends(get_embedding_service)
    for dependency injection. Follows existing route patterns.
    
    Args:
        date: Optional date string (YYYY-MM-DD). If None, uses today.
        
    Returns:
        JSONResponse with scan data or empty if not found
    """
    try:
        # MDB-Engine Pattern: Initialize RadarService with injected dependencies
        radar_service = RadarService(db, embedding_service=embedding_service)
        
        scan = await radar_service.get_latest_scan(date=date)
        
        if scan:
            # Convert MongoDB document to JSON-serializable dict
            scan_dict = {
                "date": scan.get("date"),
                "timestamp": scan.get("timestamp").isoformat() if scan.get("timestamp") else None,
                "is_final": scan.get("is_final", False),
                "strategy_id": scan.get("strategy_id"),
                "stocks": scan.get("stocks", []),
                "metadata": scan.get("metadata", {})
            }
            return JSONResponse(content=scan_dict)
        else:
            return JSONResponse(content={"stocks": [], "date": date, "message": "No scan found"})
    except Exception as e:
        logger.error(f"Error getting latest scan: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)[:100], "stocks": []},
            status_code=500
        )

async def quick_sell(symbol: str = Form(...), db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Quick sell/close position - returns HTML with hx-swap-oob for multiple updates.
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) for database access.
    """
    if not check_alpaca_config() or api is None:
        return HTMLResponse(content=toast_notification(
            message="Alpaca API not configured",
            type="error"
        ), status_code=400)
    
    try:
        api.close_position(symbol.upper())
        
        # Get updated positions HTML - pass db explicitly
        positions_response = await get_positions(db=db)
        positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
        
        # Return HTML with hx-swap-oob to update positions and show success toast
        success_toast = toast_notification(f"Position sold for {symbol}", "success", 5000)
        response_content = htmx_response(
            updates={
                "positions-list": positions_html,
                "toast-container": success_toast
            }
        )
        response = HTMLResponse(content=response_content)
        response.headers["HX-Trigger"] = "refreshPositions"
        return response
    except Exception as e:
        error_msg = str(e)
        if "position does not exist" in error_msg.lower():
            error_toast = toast_notification("No position to sell", "warning", 5000)
            return HTMLResponse(content=htmx_response(updates={"toast-container": error_toast}), status_code=404)
        logger.error(f"Quick sell failed: {e}", exc_info=True)
        error_toast = toast_notification(f"Error: {str(e)[:100]}", "error", 5000)
        return HTMLResponse(content=htmx_response(updates={"toast-container": error_toast}), status_code=500)

async def cancel_orders(symbol: str = Form(None)) -> HTMLResponse:
    """Cancel orders - all or for specific symbol - returns HTML toast notification."""
    if not check_alpaca_config() or api is None:
        return HTMLResponse(content=toast_notification(
            message="Alpaca API not configured",
            type="error"
        ), status_code=400)
    
    try:
        if symbol:
            # Cancel orders for specific symbol
            orders = api.list_orders(status='open', symbols=[symbol.upper()])
            for order in orders:
                api.cancel_order(order.id)
            message = f"Cancelled orders for {symbol}"
        else:
            # Cancel all orders
            api.cancel_all_orders()
            message = "All orders cancelled"
        
        success_toast = toast_notification(message, "success", 5000)
        return HTMLResponse(content=htmx_response(updates={"toast-container": success_toast}))
    except Exception as e:
        logger.error(f"Cancel orders failed: {e}", exc_info=True)
        error_toast = toast_notification(f"Error: {str(e)[:100]}", "error", 5000)
        return HTMLResponse(content=htmx_response(updates={"toast-container": error_toast}), status_code=500)

async def get_open_orders() -> JSONResponse:
    """Get all open orders."""
    if not check_alpaca_config() or api is None:
        return JSONResponse(content={"success": False, "orders": []})
    
    try:
        orders = api.list_orders(status='open')
        orders_data = []
        for order in orders:
            orders_data.append({
                "id": str(order.id),
                "symbol": order.symbol,
                "side": order.side,
                "qty": float(order.qty),
                "type": order.type,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "status": order.status
            })
        return JSONResponse(content={"success": True, "orders": orders_data})
    except Exception as e:
        logger.error(f"Get open orders failed: {e}", exc_info=True)
        return JSONResponse(content={"success": False, "orders": []})

async def _get_trending_stocks_with_analysis_streaming(websocket: WebSocket, db: Any, custom_symbols: Optional[List[str]] = None) -> None:
    """Stream stock analysis with progress updates via WebSocket.
    
    Simplified version - analyzes sample stocks without Firecrawl discovery.
    Includes cache integration and historical context.
    
    Args:
        websocket: WebSocket connection
        db: MongoDB database instance
    """
    logger.info("🔌 [WS] WebSocket connection attempt received")
    
    try:
        await websocket.accept()
        logger.info("✅ [WS] WebSocket connection accepted")
    except Exception as e:
        logger.error(f"❌ [WS] Failed to accept WebSocket: {e}", exc_info=True)
        return
    
    # Initialize RadarService and strategy
    # MDB-Engine Pattern: Get EmbeddingService for WebSocket context
    # Note: WebSockets can't use Depends(), so we get it directly
    from mdb_engine.embeddings.dependencies import get_embedding_service_for_app
    from ..core.engine import engine, APP_SLUG
    embedding_service = get_embedding_service_for_app(APP_SLUG, engine)
    radar_service = RadarService(db, embedding_service=embedding_service)
    strategy_id = "balanced_low"  # Current strategy
    
    # Track scan metadata for daily_scans collection
    import time
    scan_start_time = time.time()
    cache_hits = 0
    cache_misses = 0
    
    try:
        
        # Get target symbols (use custom symbols if provided, otherwise auto-discover)
        if custom_symbols:
            symbols = custom_symbols[:5]  # Limit to 5
            logger.info(f"📊 [WS] Using custom symbols: {symbols}")
        else:
            symbols = await _get_target_symbols(db)
        total = len(symbols)
        stocks_with_analysis = []
        
        logger.info(f"📊 [WS] Starting analysis of {total} symbols: {symbols}")
        
        # Send initial progress
        sent = await _safe_send_json(websocket, {
            "type": "progress",
            "message": f"Analyzing {total} stocks...",
            "current": 0,
            "total": total,
            "percentage": 0
        })
        if not sent:
            logger.info(f"[WS] WebSocket closed immediately, stopping")
            return
        logger.info("✅ [WS] Sent initial progress message")
        
        import asyncio
        await asyncio.sleep(0.1)  # Small delay for UI update
        
        # Track failures for reporting
        failed_stocks = []
        
        # Analyze each stock with progress updates
        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"🔍 [WS] [{idx}/{total}] Starting analysis for {symbol}")
            
            try:
                # Check cache before expensive operations
                cached = await radar_service.get_cached_analysis(symbol, strategy_id)
                cache_hit = cached and cached['fresh']
                
                if cache_hit:
                    logger.info(f"✅ [WS] [{idx}/{total}] Cache hit for {symbol}")
                    cache_hits += 1
                    cached_analysis = cached['analysis']
                    config = await get_strategy_config_dict(db)
                    ai_score = cached_analysis.get('verdict', {}).get('score') if isinstance(cached_analysis.get('verdict'), dict) else (cached_analysis.get('verdict').score if hasattr(cached_analysis.get('verdict'), 'score') else None)
                    confidence_boost = cached_analysis.get('confidence', {}).get('boost', 0.0)
                    adjusted_score = (ai_score or 0) + confidence_boost
                    rsi = cached_analysis.get('techs', {}).get('rsi', 0)
                    meets_criteria = rsi < config.get('rsi_threshold', 35) and adjusted_score >= config.get('ai_score_required', 7)
                    
                    stock_data = {
                        "symbol": symbol,
                        "price": cached_analysis.get('techs', {}).get('price', 0),
                        "rsi": rsi,
                        "atr": cached_analysis.get('techs', {}).get('atr', 0),
                        "trend": cached_analysis.get('techs', {}).get('trend', 'UNKNOWN'),
                        "sma": cached_analysis.get('techs', {}).get('sma', 0),
                        "ai_score": ai_score,
                        "ai_reason": cached_analysis.get('verdict', {}).get('reason', '')[:100] if isinstance(cached_analysis.get('verdict'), dict) else (cached_analysis.get('verdict').reason[:100] if hasattr(cached_analysis.get('verdict'), 'reason') else ''),
                        "risk_level": cached_analysis.get('verdict', {}).get('risk_level', 'UNKNOWN') if isinstance(cached_analysis.get('verdict'), dict) else (cached_analysis.get('verdict').risk_level if hasattr(cached_analysis.get('verdict'), 'risk_level') else 'UNKNOWN'),
                        "cache_hit": True,
                        "headlines": cached_analysis.get('headlines', []),
                        "news_objects": cached_analysis.get('news_objects', []),
                        "similar_signals": [
                            {
                                "timestamp": _convert_timestamp_to_iso(s.get('timestamp')),
                                "score": s.get('analysis', {}).get('verdict', {}).get('score', 0) if isinstance(s.get('analysis'), dict) else (s.get('score', 0) if isinstance(s, dict) else 0),
                                "profitable": s.get('outcome', {}).get('profitable', False) if s.get('outcome') else None
                            }
                            for s in cached_analysis.get('similar_signals', [])
                        ],
                        "confidence_boost": confidence_boost,
                        "adjusted_score": adjusted_score,
                        "meets_criteria": meets_criteria,
                        "rsi_threshold": config.get('rsi_threshold', 35),
                        "ai_score_required": config.get('ai_score_required', 7)
                    }
                    # Sanitize stock_data BEFORE appending (so completion message is also sanitized)
                    sanitized_stock_data = _sanitize_for_json(stock_data)
                    stocks_with_analysis.append(sanitized_stock_data)
                    
                    sent = await _safe_send_json(websocket, {
                        "type": "stock_complete",
                        "stock": sanitized_stock_data,
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100)
                    })
                    if not sent:
                        logger.info(f"[WS] WebSocket closed, stopping analysis")
                        return
                    continue
                
                # Track cache miss
                cache_misses += 1
                
                # Send progress update
                sent = await _safe_send_json(websocket, {
                    "type": "progress",
                    "message": f"Analyzing {symbol}...",
                    "current": idx,
                    "total": total,
                    "percentage": int((idx / total) * 100),
                    "symbol": symbol,
                    "cache_hit": False
                })
                if not sent:
                    logger.info(f"[WS] WebSocket closed, stopping analysis")
                    return
                logger.info(f"✅ [WS] [{idx}/{total}] Sent progress update for {symbol}")
                
                # Fetch market data
                logger.info(f"📈 [WS] [{idx}/{total}] Fetching market data for {symbol}...")
                import time
                start_time = time.time()
                
                try:
                    bars, headlines, news_objects = await get_market_data(symbol, days=500, db=db)
                    fetch_time = time.time() - start_time
                    logger.info(f"⏱️ [WS] [{idx}/{total}] Market data fetch for {symbol} took {fetch_time:.2f}s")
                except Exception as e:
                    error_msg = f"Failed to fetch market data: {str(e)[:100]}"
                    logger.error(f"❌ [WS] [{idx}/{total}] {symbol}: {error_msg}", exc_info=True)
                    failed_stocks.append({"symbol": symbol, "reason": error_msg})
                    sent = await _safe_send_json(websocket, {
                        "type": "error",
                        "message": f"Error fetching data for {symbol}: {str(e)[:50]}",
                        "symbol": symbol,
                        "current": idx,
                        "total": total
                    })
                    if not sent:
                        return
                    continue
                
                if bars is None:
                    logger.warning(f"⚠️ [WS] [{idx}/{total}] {symbol}: bars is None")
                    failed_stocks.append({"symbol": symbol, "reason": "No market data returned"})
                    sent = await _safe_send_json(websocket, {
                        "type": "progress",
                        "message": f"Skipping {symbol} - no data returned",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "skipped": True
                    })
                    if not sent:
                        return
                    continue
                
                if bars.empty:
                    logger.warning(f"⚠️ [WS] [{idx}/{total}] {symbol}: bars is empty")
                    failed_stocks.append({"symbol": symbol, "reason": "Empty market data"})
                    sent = await _safe_send_json(websocket, {
                        "type": "progress",
                        "message": f"Skipping {symbol} - empty data",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "skipped": True
                    })
                    if not sent:
                        return
                    continue
                
                if len(bars) < 14:
                    logger.warning(f"⚠️ [WS] [{idx}/{total}] {symbol}: Only {len(bars)} days of data (need 14+)")
                    failed_stocks.append({"symbol": symbol, "reason": f"Insufficient data ({len(bars)} days, need 14+)"})
                    sent = await _safe_send_json(websocket, {
                        "type": "progress",
                        "message": f"Skipping {symbol} - insufficient data ({len(bars)} days)",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "skipped": True
                    })
                    if not sent:
                        return
                    continue
                
                logger.info(f"✅ [WS] [{idx}/{total}] {symbol}: Got {len(bars)} days of data, {len(headlines)} headlines")
                
                # Send technical analysis progress
                sent = await _safe_send_json(websocket, {
                    "type": "progress",
                    "message": f"Calculating indicators for {symbol}...",
                    "current": idx,
                    "total": total,
                    "percentage": int((idx / total) * 100),
                    "symbol": symbol,
                    "stage": "technicals"
                })
                if not sent:
                    return
                logger.info(f"✅ [WS] [{idx}/{total}] Sent technicals progress for {symbol}")
                
                # Calculate technical indicators
                logger.info(f"🧮 [WS] [{idx}/{total}] Calculating technical indicators for {symbol}...")
                techs_start = time.time()
                
                try:
                    techs = analyze_technicals(bars)
                    techs_time = time.time() - techs_start
                    logger.info(f"✅ [WS] [{idx}/{total}] Technicals calculated for {symbol} in {techs_time:.2f}s: RSI={techs['rsi']:.1f}, Trend={techs['trend']}, Price=${techs['price']:.2f}")
                except Exception as e:
                    error_msg = f"Technical analysis failed: {str(e)[:100]}"
                    logger.error(f"❌ [WS] [{idx}/{total}] Technical analysis failed for {symbol}: {e}", exc_info=True)
                    failed_stocks.append({"symbol": symbol, "reason": error_msg})
                    sent = await _safe_send_json(websocket, {
                        "type": "error",
                        "message": f"Technical analysis failed for {symbol}: {str(e)[:50]}",
                        "symbol": symbol,
                        "current": idx,
                        "total": total
                    })
                    if not sent:
                        return
                    continue
                
                # Send AI analysis progress
                sent = await _safe_send_json(websocket, {
                    "type": "progress",
                    "message": f"AI analyzing {symbol}...",
                    "current": idx,
                    "total": total,
                    "percentage": int((idx / total) * 100),
                    "symbol": symbol,
                    "stage": "ai"
                })
                if not sent:
                    return
                logger.info(f"✅ [WS] [{idx}/{total}] Sent AI progress for {symbol}")
                
                # Find similar historical signals for context
                analysis_data_for_search = {
                    'techs': techs,
                    'headlines': headlines
                }
                similar_signals = await radar_service.find_similar_signals(symbol, analysis_data_for_search, limit=5)
                
                # Get AI analysis if available
                ai_score = None
                ai_reason = None
                risk_level = "UNKNOWN"
                verdict = None
                
                if _ai_engine:
                    logger.info(f"🤖 [WS] [{idx}/{total}] Running AI analysis for {symbol}...")
                    ai_start = time.time()
                    try:
                        strategy_prompt = get_balanced_low_prompt()
                        logger.debug(f"   Strategy: Balanced Low, Prompt length: {len(strategy_prompt)}")
                        
                        # Include historical context in prompt if available
                        if similar_signals:
                            profitable_count = sum(
                                1 for s in similar_signals
                                if s.get('outcome', {}).get('profitable', False)
                            )
                            if profitable_count > 0:
                                strategy_prompt += f"\n\nHistorical Context: {profitable_count} out of {len(similar_signals)} similar signals were profitable."
                        
                        verdict = _ai_engine.analyze(
                            ticker=symbol,
                            techs=techs,
                            headlines="\n".join(headlines[:3]) if headlines else "No recent news",
                            strategy_prompt=strategy_prompt
                        )
                        ai_score = verdict.score
                        ai_reason = verdict.reason[:100] + "..." if len(verdict.reason) > 100 else verdict.reason
                        risk_level = verdict.risk_level
                        
                        ai_time = time.time() - ai_start
                        logger.info(f"✅ [WS] [{idx}/{total}] AI analysis complete for {symbol} in {ai_time:.2f}s: Score={ai_score}/10, Risk={risk_level}, Action={verdict.action}")
                    except Exception as e:
                        logger.error(f"❌ [WS] [{idx}/{total}] AI analysis failed for {symbol}: {e}", exc_info=True)
                        logger.error(f"   Error type: {type(e).__name__}, Error: {str(e)}")
                else:
                    logger.warning(f"⚠️ [WS] [{idx}/{total}] AI engine not available, skipping AI analysis for {symbol}")
                
                # Calculate confidence boost from historical patterns
                confidence = await radar_service.get_signal_confidence(symbol, {
                    'techs': techs,
                    'headlines': headlines,
                    'verdict': verdict
                })
                
                # Store analysis to cache and history
                if verdict:
                    # Convert TradeVerdict (Pydantic model) to dict for MongoDB storage
                    verdict_dict = verdict.model_dump()
                    
                    analysis_data = {
                        'symbol': symbol,
                        'techs': techs,
                        'headlines': headlines,
                        'verdict': verdict_dict,  # Use dict instead of object
                        'timestamp': datetime.now(),
                        'strategy': strategy_id,
                        'confidence': confidence,
                        'similar_signals': similar_signals[:5],
                        'news_objects': news_objects
                    }
                    await radar_service.cache_analysis(symbol, strategy_id, analysis_data)
                    await radar_service.store_to_history(symbol, analysis_data)
                
                # Include analysis even if scoring failed - allows users to retry
                # Only skip if we have no data at all (bars, techs, etc.)
                config = await get_strategy_config_dict(db)
                adjusted_score = (ai_score or 0) + confidence.get('boost', 0.0)
                meets_criteria = techs.get('rsi', 50) < config.get('rsi_threshold', 35) and adjusted_score >= config.get('ai_min_score', 7) if ai_score is not None else False
                
                # Log if scoring failed but include analysis anyway
                if ai_score is None:
                    logger.info(f"⚠️ [WS] [{idx}/{total}] {symbol} - Analysis complete but AI scoring unavailable (included for retry)")
                elif ai_score == 0:
                    logger.info(f"⚠️ [WS] [{idx}/{total}] {symbol} - Score is 0/10 but including analysis for review/retry")
                
                stock_data = {
                    "symbol": symbol,
                    "price": techs['price'],
                    "rsi": techs['rsi'],
                    "atr": techs['atr'],
                    "trend": techs['trend'],
                    "sma": techs['sma'],
                    "ai_score": ai_score,
                    "ai_reason": ai_reason,
                    "risk_level": risk_level,
                    "cache_hit": False,
                    "headlines": headlines,
                    "news_objects": news_objects,
                    "similar_signals": [
                        {
                            "timestamp": _convert_timestamp_to_iso(s.get('timestamp')),
                            "score": s.get('analysis', {}).get('verdict', {}).get('score', 0),
                            "profitable": s.get('outcome', {}).get('profitable', False) if s.get('outcome') else None
                        }
                        for s in similar_signals[:5]
                    ],
                    "confidence_boost": confidence.get('boost', 0.0),
                    "win_rate": confidence.get('win_rate', 0.0),
                    "similar_count": confidence.get('similar_count', 0),
                    "adjusted_score": adjusted_score,
                    "meets_criteria": meets_criteria,
                    "rsi_threshold": config.get('rsi_threshold', 35),
                    "ai_score_required": config.get('ai_min_score', 7)
                }
                
                # Sanitize stock_data BEFORE appending (so completion message is also sanitized)
                sanitized_stock_data = _sanitize_for_json(stock_data)
                stocks_with_analysis.append(sanitized_stock_data)
                logger.info(f"✅ [WS] [{idx}/{total}] {symbol} analysis complete, added to results")
                
                # Send completed stock
                try:
                    sent = await _safe_send_json(websocket, {
                        "type": "stock_complete",
                        "stock": sanitized_stock_data,
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100)
                    })
                    if not sent:
                        logger.info(f"[WS] WebSocket closed, stopping analysis")
                        return
                    logger.info(f"✅ [WS] [{idx}/{total}] Sent stock_complete for {symbol}")
                except Exception as e:
                    logger.error(f"❌ [WS] Failed to send stock_complete for {symbol}: {e}", exc_info=True)
                
            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "rate limit" in error_msg.lower() or "rate limit exceeded" in error_msg.lower()
                
                logger.error(f"❌ [WS] [{idx}/{total}] Exception analyzing {symbol}: {e}", exc_info=True)
                logger.error(f"   Error type: {type(e).__name__}")
                logger.error(f"   Error message: {error_msg}")
                
                failed_stocks.append({
                    "symbol": symbol,
                    "reason": f"Exception: {error_msg[:100]}"
                })
                
                await _safe_send_json(websocket, {
                    "type": "error",
                    "message": f"Error analyzing {symbol}: {str(e)[:50]}",
                    "symbol": symbol,
                    "current": idx,
                    "total": total,
                    "is_rate_limit": is_rate_limit,
                    "error_type": "rate_limit" if is_rate_limit else "general"
                })
                continue
        
        # Send completion
        # Note: stocks_with_analysis is already sanitized (we sanitize before appending)
        success_count = len(stocks_with_analysis)
        failed_count = len(failed_stocks)
        logger.info(f"🎉 [WS] Analysis complete! Processed {success_count}/{total} stocks successfully, {failed_count} failed")
        logger.info(f"📊 [WS] Scanned {total} stocks | {success_count} with complete analysis (including those without scores for retry)")
        
        if failed_stocks:
            logger.warning(f"⚠️ [WS] Failed stocks: {', '.join([f['symbol'] for f in failed_stocks])}")
            for failure in failed_stocks:
                logger.warning(f"   - {failure['symbol']}: {failure['reason']}")
        
        # Count stocks with scores vs without scores
        stocks_with_scores = sum(1 for s in stocks_with_analysis if s.get('ai_score') is not None)
        stocks_without_scores = success_count - stocks_with_scores
        if stocks_without_scores > 0:
            logger.info(f"ℹ️ [WS] {stocks_without_scores} stocks included without scores (available for retry)")
        
        try:
            warnings = []
            if success_count < len(symbols):
                warnings.append(f"Only {success_count} of {len(symbols)} stocks analyzed successfully")
                if failed_stocks:
                    failed_symbols = [f['symbol'] for f in failed_stocks]
                    warnings.append(f"Failed: {', '.join(failed_symbols)}")
            
            sent = await _safe_send_json(websocket, {
                "type": "complete",
                "stocks": stocks_with_analysis,  # Already sanitized
                "total": success_count,
                "expected": len(symbols),
                "failed": failed_stocks,  # Include failure details
                "source": "sample",
                "warnings": warnings
            })
            logger.info("✅ [WS] Sent completion message")
            
            # Save scan to daily_scans collection
            scan_duration = time.time() - scan_start_time
            scan_metadata = {
                "symbols_scanned": len(stocks_with_analysis),
                "duration_seconds": round(scan_duration, 2),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "success_count": success_count,
                "failed_count": len(failed_stocks)
            }
            save_result = await radar_service.save_daily_scan(
                stocks=stocks_with_analysis,
                strategy_id=strategy_id,
                metadata=scan_metadata
            )
            if save_result:
                logger.info(f"✅ [WS] Saved daily scan: {len(stocks_with_analysis)} stocks, duration: {scan_duration:.2f}s")
            else:
                logger.warning("⚠️ [WS] Failed to save daily scan")
        except Exception as e:
            logger.error(f"❌ [WS] Failed to send completion message: {e}", exc_info=True)
        
    except WebSocketDisconnect:
        logger.info("🔌 [WS] WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"❌ [WS] Fatal WebSocket error: {e}", exc_info=True)
        logger.error(f"   Error type: {type(e).__name__}")
        try:
            await _safe_send_json(websocket, {
                "type": "error",
                "message": f"Analysis failed: {str(e)[:100]}"
            })
        except:
            logger.error("❌ [WS] Failed to send error message to client")

async def get_explanation(
    symbol: str = Form(...), 
    db: Any = Depends(get_scoped_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> HTMLResponse:
    """Get detailed explanation for a stock analysis including calculations, data, insights, and news.
    
    MDB-Engine Pattern: Dependency Injection
    - db: Scoped database via get_scoped_db()
    - embedding_service: EmbeddingService via get_embedding_service()
    """
    try:
        symbol = symbol.upper().strip()
        strategy_id = "balanced_low"  # Current strategy
        # MDB-Engine Pattern: Inject EmbeddingService via dependency injection
        radar_service = RadarService(db, embedding_service=embedding_service)
        
        # Try to get from cache first
        cached = await radar_service.get_cached_analysis(symbol, strategy_id)
        bars_count = 0
        if cached and cached['fresh']:
            analysis_data = cached['analysis']
            # Estimate bars count from techs if available
            bars_count = 200  # Default estimate
            # Ensure news_objects exists in cached data
            if 'news_objects' not in analysis_data:
                analysis_data['news_objects'] = []
        else:
            # Fetch fresh data
            bars, headlines, news_objects = get_market_data(symbol, days=500, db=db)
            if bars is None or bars.empty or len(bars) < 14:
                return HTMLResponse(content=templates.get_template("partials/status_message.html").render(
                    message=f"Insufficient data for {symbol}",
                    color="red"
                ), status_code=400)
            
            bars_count = len(bars)
            techs = analyze_technicals(bars)
            
            # Get AI analysis
            if not _ai_engine:
                return HTMLResponse(content=templates.get_template("partials/status_message.html").render(
                    message="AI engine not available",
                    color="red"
                ), status_code=503)
            
            strategy_prompt = get_balanced_low_prompt()
            verdict = _ai_engine.analyze(
                ticker=symbol,
                techs=techs,
                headlines="\n".join(headlines),
                strategy_prompt=strategy_prompt
            )
            
            # Find similar signals
            analysis_data_for_search = {
                'techs': techs,
                'headlines': headlines
            }
            similar_signals = await radar_service.find_similar_signals(symbol, analysis_data_for_search, limit=5)
            
            # Get confidence
            confidence = await radar_service.get_signal_confidence(symbol, {
                'techs': techs,
                'headlines': headlines,
                'verdict': verdict
            })
            
            analysis_data = {
                'symbol': symbol,
                'techs': techs,
                'headlines': headlines,
                'verdict': {
                    'score': verdict.score,
                    'reason': verdict.reason,
                    'risk_level': verdict.risk_level,
                    'action': verdict.action
                },
                'timestamp': datetime.now(),
                'strategy': strategy_id,
                'confidence': confidence,
                'similar_signals': similar_signals[:5],
                'news_objects': news_objects
            }
        
        # Build explanation response
        techs = analysis_data.get('techs', {})
        verdict = analysis_data.get('verdict', {})
        verdict_score = verdict.get('score', 0)
        verdict_reason = verdict.get('reason', '')
        verdict_risk = verdict.get('risk_level', 'UNKNOWN')
        verdict_action = verdict.get('action', 'UNKNOWN')
        
        config = await get_strategy_config_dict(db)
        
        # Calculate values for HTML rendering
        rsi_value = techs.get('rsi', 0)
        sma_value = techs.get('sma', 0)
        atr_value = techs.get('atr', 0)
        price_value = techs.get('price', 0)
        trend_value = techs.get('trend', 'UNKNOWN')
        adjusted_score = verdict_score + analysis_data.get('confidence', {}).get('boost', 0.0)
        confidence_boost = analysis_data.get('confidence', {}).get('boost', 0.0)
        meets_criteria = techs.get('rsi', 50) < config.get('rsi_threshold', 35) and adjusted_score >= config.get('ai_score_required', 7)
        similar_signals_list = analysis_data.get('similar_signals', [])
        win_rate = analysis_data.get('confidence', {}).get('win_rate', 0.0) * 100
        
        # Determine color classes
        ai_score_color = 'text-green-400' if verdict_score >= 7 else ('text-yellow-400' if verdict_score >= 5 else 'text-red-400')
        adjusted_score_color = 'text-green-400' if adjusted_score >= 7 else ('text-yellow-400' if adjusted_score >= 5 else 'text-red-400')
        trend_color = 'text-green-400' if trend_value == 'UP' else 'text-red-400'
        risk_color = 'text-green-400' if verdict_risk == 'LOW' else ('text-yellow-400' if verdict_risk == 'MEDIUM' else 'text-red-400')
        rsi_status = 'Oversold' if rsi_value < 30 else ('Overbought' if rsi_value > 70 else 'Neutral')
        
        # Calculate exit levels for buy low/sell high framing
        stop_loss_price = round(price_value - (2 * atr_value), 2) if atr_value > 0 else round(price_value * 0.95, 2)
        take_profit_price = round(price_value + (3 * atr_value), 2) if atr_value > 0 else round(price_value * 1.10, 2)
        
        # Prepare similar signals for template
        template_similar_signals = []
        if similar_signals_list:
            for s in similar_signals_list[:3]:
                timestamp_str = _convert_timestamp_to_iso(s.get('timestamp')) or 'N/A'
                
                score_val = 0
                if isinstance(s.get('analysis'), dict):
                    score_val = s.get('analysis', {}).get('verdict', {}).get('score', 0)
                elif hasattr(s.get('analysis'), 'score'):
                    score_val = s.get('analysis').score
                
                profitable_val = None
                if isinstance(s.get('outcome'), dict):
                    profitable_val = s.get('outcome', {}).get('profitable', None)
                
                template_similar_signals.append({
                    'timestamp': timestamp_str,
                    'analysis': {
                        'verdict': {
                            'score': score_val
                        }
                    },
                    'outcome': {
                        'profitable': profitable_val
                    }
                })
        
        # Render template
        return HTMLResponse(content=templates.get_template("pages/explanation.html").render(
            config=config,
            ai_score_color=ai_score_color,
            adjusted_score_color=adjusted_score_color,
            trend_color=trend_color,
            risk_color=risk_color,
            rsi_status=rsi_status,
            verdict_score=verdict_score,
            verdict_action=verdict_action,
            verdict_reason=verdict_reason,
            verdict_risk=verdict_risk,
            rsi_value=rsi_value,
            sma_value=sma_value,
            atr_value=atr_value,
            price_value=price_value,
            trend_value=trend_value,
            adjusted_score=adjusted_score,
            confidence_boost=confidence_boost,
            meets_criteria=meets_criteria,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            bars_count=bars_count,
            headlines=analysis_data.get('headlines', []),
            similar_signals_list=template_similar_signals,
            win_rate=win_rate,
            techs=techs,
            news_objects_list=analysis_data.get('news_objects', [])
        ))
    except Exception as e:
        logger.error(f"Failed to get explanation for {symbol}: {e}", exc_info=True)
        return HTMLResponse(content=templates.get_template("partials/status_message.html").render(
            message=str(e)[:200],
            color="red"
        ), status_code=500)

