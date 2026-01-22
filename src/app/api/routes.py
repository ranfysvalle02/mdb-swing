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
from fastapi.responses import HTMLResponse, JSONResponse, Response
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import json
from mdb_engine.observability import get_logger
from mdb_engine.dependencies import get_embedding_service
from mdb_engine.embeddings import EmbeddingService
from ..core.engine import get_scoped_db
from ..services.trading import place_order, calculate_position_size
from ..services.analysis import api
from ..services.analysis import get_market_data, analyze_technicals, analyze_comprehensive_stats
from ..services.alpaca_accounts import AlpacaAccountManager
from alpaca_trade_api.rest import TimeFrame
from ..services.ai import EyeAI, _ai_engine
from ..services.radar import RadarService
from ..services.positions import calculate_position_metrics, detect_sell_signal
from ..services.logo import get_logo_html
from ..api.templates import empty_positions, pending_order_card, position_card, lucide_init_script, toast_notification, htmx_response, htmx_modal_wrapper, error_response, transaction_history_item, empty_transactions, transactions_view, _sanitize_html_for_htmx
from ..core.templates import templates
from ..core.config import ALPACA_KEY, ALPACA_SECRET, STRATEGY_CONFIG, get_strategy_from_db, get_strategy_config as get_strategy_config_dict, FIRECRAWL_SEARCH_QUERY_TEMPLATE, get_active_custom_strategy, calculate_custom_strategy_hash
from ..services.ai_prompts import get_balanced_low_prompt
from ..services.custom_strategies import evaluate_strategy, get_available_metrics, format_strategy_description
from ..services.signal_hunter import get_screener_results, FILTER_PRESETS, DEFAULT_PRESET

logger = get_logger(__name__)

# In-memory cache for recently cancelled orders (to handle Alpaca API propagation delay)
# Key: order_id, Value: timestamp when cancelled
# TTL: 30 seconds to handle propagation window
_cancelled_orders_cache: Dict[str, float] = {}
import asyncio
from time import time


def error_response_html(message: str, status_code: int = 200) -> HTMLResponse:
    """Create a standardized HTML error response for HTMX."""
    return HTMLResponse(
        content=templates.get_template("partials/error_message.html").render(message=message),
        status_code=status_code
    )


def error_response_json(message: str, detail: Optional[str] = None, status_code: int = 400) -> JSONResponse:
    """Create a standardized JSON error response for API endpoints."""
    content = {"error": message}
    if detail:
        content["detail"] = detail
    return JSONResponse(content=content, status_code=status_code)


async def get_timeout_error() -> HTMLResponse:
    """Get timeout error template for HTMX timeout handling."""
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
                logger.info(f"Using watch list symbols ({len(symbols)} total): {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
                return symbols
        
        # Fallback to defaults
        from ..core.config import SYMBOLS
        logger.info(f"Using default symbols: {SYMBOLS}")
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

async def get_balance_compact(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get minimal account balance for nav bar.
    
    Returns a compact version showing only equity and P&L.
    """
    try:
        account_manager = AlpacaAccountManager(db)
        try:
            active_account = await account_manager.get_active_account()
        except Exception as account_error:
            logger.warning(f"Error getting active account: {account_error}")
            active_account = None
        
        if not active_account:
            if not check_alpaca_config() or api is None:
                return HTMLResponse(content=templates.get_template("pages/account_balance_compact.html").render(
                    equity=0.0,
                    pl=0.0,
                    pl_abs=0.0,
                    connected=False
                ))
            acct = api.get_account()
        else:
            api_client = account_manager.get_api_client(active_account)
            if not api_client:
                return HTMLResponse(content=templates.get_template("pages/account_balance_compact.html").render(
                    equity=0.0,
                    pl=0.0,
                    pl_abs=0.0,
                    connected=False
                ))
            acct = api_client.get_account()
        
        try:
            last_equity = float(acct.last_equity) if hasattr(acct, 'last_equity') and acct.last_equity else float(acct.equity)
        except (AttributeError, ValueError, TypeError):
            last_equity = float(acct.equity)
        
        pl = float(acct.equity) - last_equity
        
        try:
            template_content = templates.get_template("pages/account_balance_compact.html").render(
                equity=float(acct.equity),
                pl=pl,
                pl_abs=abs(pl),
                connected=True
            )
            return HTMLResponse(content=template_content, status_code=200)
        except Exception as template_error:
            logger.error(f"Template render error in compact balance: {template_error}", exc_info=True)
            # Fallback to simple HTML
            return HTMLResponse(
                content=f'<div class="flex items-center gap-2"><span class="text-sm text-white">${float(acct.equity):,.0f}</span></div>',
                status_code=200
            )
    except Exception as e:
        logger.error(f"Failed to get compact balance: {e}", exc_info=True)
        try:
            return HTMLResponse(
                content=templates.get_template("pages/account_balance_compact.html").render(
                    equity=0.0,
                    pl=0.0,
                    pl_abs=0.0,
                    connected=False
                ),
                status_code=200
            )
        except Exception as template_error:
            logger.error(f"Fallback template also failed: {template_error}", exc_info=True)
            # Ultimate fallback
            return HTMLResponse(
                content='<div class="flex items-center gap-2"><span class="text-sm text-white/60">--</span></div>',
                status_code=200
            )


async def get_balance(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get account balance.
    
    MDB-Engine Pattern: Returns HTMLResponse (not JSON) for htmx integration.
    HTMX expects HTML fragments that can be swapped into the DOM.
    This endpoint is polled every 10 seconds via hx-trigger="every 10s".
    
    Uses AlpacaAccountManager to get balance from active account.
    """
    account_manager = AlpacaAccountManager(db)
    
    try:
        # Get active account
        active_account = await account_manager.get_active_account()
        
        # Fallback to env vars if no account configured
        if not active_account:
            if not check_alpaca_config() or api is None:
                return HTMLResponse(content=templates.get_template("pages/account_balance.html").render(
                    equity=0.0,
                    pl=0.0,
                    pl_abs=0.0,
                    buying_power=0.0,
                    pl_color="text-gray-400",
                    pl_icon="fa-exclamation-triangle",
                    connected=False
                ))
            # Use global api as fallback
            acct = api.get_account()
        else:
            # Use account manager API client
            api_client = account_manager.get_api_client(active_account)
            if not api_client:
                return HTMLResponse(content=templates.get_template("pages/account_balance.html").render(
                    equity=0.0,
                    pl=0.0,
                    pl_abs=0.0,
                    buying_power=0.0,
                    pl_color="text-red-400",
                    pl_icon="fa-exclamation-triangle",
                    connected=False
                ))
            acct = api_client.get_account()
        
        # Handle case where last_equity might not exist (new account)
        try:
            last_equity = float(acct.last_equity) if hasattr(acct, 'last_equity') and acct.last_equity else float(acct.equity)
        except (AttributeError, ValueError, TypeError):
            last_equity = float(acct.equity)
        
        pl = float(acct.equity) - last_equity
        pl_color = "text-green-400" if pl >= 0 else "text-red-400"
        pl_icon = "fa-arrow-up" if pl >= 0 else "fa-arrow-down"
        
        try:
            return HTMLResponse(
                content=templates.get_template("pages/account_balance.html").render(
                    equity=float(acct.equity),
                    pl=pl,  # Include signed P&L for template logic
                    pl_abs=abs(pl),
                    buying_power=float(acct.buying_power),
                    pl_color=pl_color,
                    pl_icon=pl_icon,
                    connected=True
                ),
                status_code=200
            )
        except Exception as render_error:
            logger.error(f"Template render error: {render_error}", exc_info=True)
            # Return simple HTML fallback
            return HTMLResponse(
                content=f'<div class="text-red-400 text-sm">Error loading balance</div>',
                status_code=200
            )
    except Exception as e:
        logger.error(f"Failed to get balance: {e}", exc_info=True)
        try:
            return HTMLResponse(
                content=templates.get_template("pages/account_balance.html").render(
                    equity=0.0,
                    pl=0.0,
                    pl_abs=0.0,
                    buying_power=0.0,
                    pl_color="text-red-400",
                    pl_icon="fa-exclamation-triangle",
                    connected=False
                ),
                status_code=200
            )
        except Exception as render_error:
            logger.error(f"Error rendering error template: {render_error}", exc_info=True)
            # Return a simple error message that doesn't require template variables
            return HTMLResponse(
                content='<div class="account-balance-container"><div class="text-red-400 text-sm flex items-center gap-2"><i data-lucide="alert-circle" class="w-4 h-4"></i><span>Error loading balance</span></div></div>',
                status_code=200
            )

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




async def analyze_symbol(
    request: Request,
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
        logger.info(f"Using cached analysis for {symbol} (instant response)")
        
        # Extract data from cached analysis
        techs = analysis_data.get('techs', {})
        verdict_data = analysis_data.get('verdict', {})
        cnn_data = analysis_data.get('cnn_data')  # Get CNN data from cache
        
        # Handle verdict (dict from cache)
        from types import SimpleNamespace
        verdict = SimpleNamespace(
            score=verdict_data['score'],
            reason=verdict_data['reason'],
            risk_level=verdict_data['risk_level'],
            action=verdict_data['action'],
            key_factors=verdict_data['key_factors'],
            risks=verdict_data['risks'],
            opportunities=verdict_data['opportunities'],
            catalyst=verdict_data['catalyst']
        )
        
        color = "text-green-400" if verdict.score > 6 else "text-red-400"
        risk_badge = "badge-success" if verdict.risk_level.upper() == "LOW" else "badge-warning" if verdict.risk_level.upper() == "MEDIUM" else "badge-danger"
        action_color = "text-green-400" if verdict.action == "BUY" else "text-red-400"
        
        import time
        tradingview_id = f"tradingview_{int(time.time())}_{symbol}"
        
        cached_meta = await radar_service.get_cached_analysis(symbol, strategy_id)
        headlines_count = len(analysis_data.get('headlines', [])) if cached_meta else 0
        bars_count = 0  # We don't store bars count in cache, estimate from techs
        news_objects = analysis_data.get('news_objects', [])
        
        # Format dates in news_objects for display
        formatted_news = []
        for news in news_objects:
            formatted_news_item = news.copy()
            if news.get('created_at'):
                created_at = news['created_at']
                if isinstance(created_at, datetime):
                    formatted_news_item['created_at'] = created_at.strftime('%b %d, %Y')
                elif isinstance(created_at, str):
                    # Try to parse and format if it's an ISO string
                    try:
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        formatted_news_item['created_at'] = dt.strftime('%b %d, %Y')
                    except:
                        formatted_news_item['created_at'] = created_at
            formatted_news.append(formatted_news_item)
        news_objects = formatted_news
        
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
            bars_count=bars_count,
            news_objects=news_objects,
            cnn_data=None  # CNN data not available in cached results
        )
        
        # Sanitize HTML to prevent HTMX insertBefore errors from unescaped & characters
        analysis_content = _sanitize_html_for_htmx(analysis_content)
        
        # Wrap in modal only if targeting #modal-container (from watchlist card)
        if request and request.headers.get("HX-Target") == "modal-container":
            modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
                symbol=symbol,
                title=f"{symbol} - AI Analysis",
                content=analysis_content
            )
            # Sanitize modal HTML as well
            modal_html = _sanitize_html_for_htmx(modal_html)
            return HTMLResponse(content=modal_html)
        
        # Return unwrapped content for modal body swaps (from analysis_preview)
        return HTMLResponse(content=analysis_content)
    
    # Cache miss or stale - do full analysis
    logger.info(f"Cache miss for {symbol}, doing full analysis...")
    
    # This shows the user what's happening with polished loading state
    import asyncio
    progress_html = templates.get_template("components/ai_analysis_progress.html").render(
        symbol=symbol
    )
    
    # Small delay to ensure loading state is visible (polished UX)
    # This prevents the "flash" when analysis is very fast
    await asyncio.sleep(0.3)  # 300ms minimum display time for loading state
    
    # Fetch article content when user explicitly requests AI analysis
    bars, headlines, news_objects = await get_market_data(symbol, days=500, db=db, fetch_firecrawl_content=True)
    if bars is None or bars.empty:
        error_content = templates.get_template("partials/error_message.html").render(
            message="Data Error: Unable to fetch market data for this symbol. Symbol not found or market closed. Check if the symbol is valid."
        )
        return HTMLResponse(content=error_content)
    
    if len(bars) < 14:
        logger.error(f"Still only {len(bars)} days for {symbol} after requesting 500 days!")
        logger.error(f"This suggests an API issue. Check logs for details.")
        
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
        # Get strategy config to pass to AI
        from ..core.config import get_strategy_config
        from ..services.analysis import fetch_cnn_market_data
        strategy_config = await get_strategy_config(db)
        strategy_prompt = get_balanced_low_prompt(strategy_config)
        # Combine headlines for AI analysis
        headlines_text = "\n".join(headlines) if headlines else "No recent news found."
        logger.info(f"[AI] Analyzing {symbol} with {len(headlines)} headlines")
        
        # Fetch CNN market data (analyst ratings, price targets, financials) for better analysis
        cnn_data = None
        try:
            cnn_data = await fetch_cnn_market_data(symbol)
            if cnn_data:
                logger.info(f"[AI] Fetched CNN market data for {symbol}: analyst ratings={bool(cnn_data.get('analyst_ratings'))}, price_targets={bool(cnn_data.get('price_targets'))}")
            else:
                logger.debug(f"[AI] No CNN data available for {symbol} (will proceed without it)")
        except Exception as cnn_error:
            logger.warning(f"[AI] Failed to fetch CNN data for {symbol}: {cnn_error} (will proceed without it)")
            cnn_data = None
        
        # Wrap AI analysis in timeout (60 seconds)
        try:
            verdict = await asyncio.wait_for(
                asyncio.to_thread(
                    ai_engine.analyze,
                    ticker=symbol,
                    techs=techs,
                    headlines=headlines_text,
                    strategy_prompt=strategy_prompt,
                    strategy_config=strategy_config,
                    cnn_data=cnn_data
                ),
                timeout=60.0
            )
            logger.info(f"[AI] Analysis complete for {symbol}: Score {verdict.score}/10, Action: {verdict.action}")
        except asyncio.TimeoutError:
            logger.error(f"[AI] Analysis timed out for {symbol} after 60 seconds")
            error_content = templates.get_template("partials/error_message.html").render(
                message=f"AI analysis timed out after 60 seconds. The AI service may be slow or overloaded. Please try again."
            )
            return HTMLResponse(content=error_content)
        except Exception as e:
            logger.error(f"[AI] Analysis failed for {symbol}: {e}", exc_info=True)
            error_content = templates.get_template("partials/error_message.html").render(
                message=f"AI analysis failed: {str(e)[:200]}"
            )
            return HTMLResponse(content=error_content)
        
        # Cache the analysis for future requests (include CNN data)
        verdict_dict = {
            'score': verdict.score,
            'reason': verdict.reason,
            'risk_level': verdict.risk_level,
            'action': verdict.action,
            'key_factors': verdict.key_factors,
            'risks': verdict.risks,
            'opportunities': verdict.opportunities,
            'catalyst': verdict.catalyst
        }
        analysis_data = {
            'techs': techs,
            'verdict': verdict_dict,
            'headlines': headlines,
            'news_objects': news_objects,
            'cnn_data': cnn_data  # Cache CNN data for display
        }
        await radar_service.cache_analysis(symbol, strategy_id, analysis_data)
        
        color = "text-green-400" if verdict.score > 6 else "text-red-400"
        risk_badge = "badge-success" if verdict.risk_level.upper() == "LOW" else "badge-warning" if verdict.risk_level.upper() == "MEDIUM" else "badge-danger"
        action_color = "text-green-400" if verdict.action == "BUY" else "text-red-400"
        
        import time
        tradingview_id = f"tradingview_{int(time.time())}_{symbol}"
        
        # Format dates in news_objects for display
        formatted_news = []
        for news in news_objects:
            formatted_news_item = news.copy()
            if news.get('created_at'):
                created_at = news['created_at']
                if isinstance(created_at, datetime):
                    formatted_news_item['created_at'] = created_at.strftime('%b %d, %Y')
                elif isinstance(created_at, str):
                    # Try to parse and format if it's an ISO string
                    try:
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        formatted_news_item['created_at'] = dt.strftime('%b %d, %Y')
                    except:
                        formatted_news_item['created_at'] = created_at
            formatted_news.append(formatted_news_item)
        news_objects = formatted_news
        
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
            bars_count=len(bars) if bars is not None and not bars.empty else 0,
            news_objects=news_objects,
            cnn_data=cnn_data  # Pass CNN data to template for display
        )
        
        # Sanitize HTML to prevent HTMX insertBefore errors from unescaped & characters
        analysis_content = _sanitize_html_for_htmx(analysis_content)
        
        # Wrap in modal only if targeting #modal-container (from watchlist card)
        if request and request.headers.get("HX-Target") == "modal-container":
            modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
                symbol=symbol,
                title=f"{symbol} - AI Analysis",
                content=analysis_content
            )
            # Sanitize modal HTML as well
            modal_html = _sanitize_html_for_htmx(modal_html)
            return HTMLResponse(content=modal_html)
        
        # Return unwrapped content for modal body swaps (from analysis_preview)
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
                title=f"{symbol} ANALYSIS",
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
            title=f"{symbol} ANALYSIS",
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
            title=f"{symbol} ANALYSIS",
            content=error_content
        )
        return HTMLResponse(content=modal_html)

async def get_all_stats(request: Request, ticker: str = Form(...), db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get comprehensive stats for a symbol.
    
    Returns all technical indicators in a beautiful display.
    Wraps in modal if targeting #modal-container, otherwise returns unwrapped content for modal body swaps.
    """
    symbol = ticker.upper().strip()
    if not symbol:
        return HTMLResponse(content="<p class='text-red-400 p-4'>Invalid symbol</p>")
    
    try:
        bars, _, _ = await get_market_data(symbol, days=250, db=db)
        
        if bars is None or bars.empty or len(bars) < 20:
            insufficient_content = templates.get_template("partials/insufficient_data.html").render(
                symbol=symbol,
                days_available=len(bars) if bars is not None and not bars.empty else 0
            )
            # Check if we need to wrap in modal (targeting #modal-container from watchlist card)
            hx_target = request.headers.get("HX-Target", "") if request else ""
            if hx_target == "modal-container":
                modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
                    symbol=symbol,
                    title=f"{symbol} - All Stats",
                    content=insufficient_content
                )
                return HTMLResponse(content=modal_html)
            return HTMLResponse(content=insufficient_content)
        
        # Calculate comprehensive stats
        stats = analyze_comprehensive_stats(bars)
        
        # Render stats content
        stats_content = templates.get_template("pages/all_stats.html").render(
            symbol=symbol,
            stats=stats
        )
        
        # Wrap in modal only if targeting #modal-container (from watchlist card)
        if request and request.headers.get("HX-Target") == "modal-container":
            modal_html = templates.get_template("components/analysis_modal_wrapper.html").render(
                symbol=symbol,
                title=f"{symbol} - All Stats",
                content=stats_content
            )
            return HTMLResponse(content=modal_html)
        
        # Return unwrapped content for modal body swaps (from analysis_preview)
        return HTMLResponse(content=stats_content)
        
    except Exception as e:
        logger.error(f"Error getting stats for {symbol}: {e}", exc_info=True)
        error_content = templates.get_template("partials/error_message.html").render(
            message=f"Could not load stats for {symbol}. Please try again later."
        )
        return HTMLResponse(content=error_content)

async def analyze_rejection(ticker: str = Form(...), db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Analyze why a symbol does not meet entry criteria - simple explanation."""
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
            # Get API client from account manager
            account_manager = AlpacaAccountManager(db)
            active_account = await account_manager.get_active_account()
            api_client = account_manager.get_api_client(active_account) if active_account else None
            
            order = await place_order(sym, 'buy', "Manual Override", 10, db, api_client=api_client)
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
            # Check if position exists before trying to close
            try:
                position = api.get_position(sym)
                if not position or not hasattr(position, 'qty') or float(position.qty) <= 0:
                    return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                        message=f"Position for {sym} not found or already closed"
                    ))
            except Exception:
                # Position doesn't exist
                return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                    message=f"Position for {sym} not found or already closed"
                ))
            
            api.close_position(sym)
            return HTMLResponse(content=templates.get_template("partials/success_message.html").render(
                message="Position Sold"
            ))
        except Exception as e:
            logger.error(f"Failed to close position: {e}", exc_info=True)
            error_msg = str(e).lower()
            
            # Check for "already closed" patterns
            already_closed_patterns = [
                "position not found",
                "no position",
                "position does not exist",
                "position already closed",
                "insufficient qty",
                "available: 0",
                "404",
                "not found"
            ]
            
            if any(pattern in error_msg for pattern in already_closed_patterns):
                return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                    message=f"Position for {sym} already closed or not found"
                ))
            
            error_display = str(e)[:50] if str(e) else "No Position"
            return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                message=error_display
            ))

async def get_positions(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get current active positions and pending BUY orders.
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) for automatic connection management.
    Returns HTML with htmx buttons for closing positions (hx-post="/api/quick-sell").
    This endpoint is polled every 5 seconds via hx-trigger="every 5s".
    
    Shows pending BUY orders at the top, followed by active positions.
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
        
        # Get pending BUY orders
        pending_buy_orders = []
        try:
            all_orders = api.list_orders(status='open')
            # Filter for pending statuses (orders that haven't been filled yet)
            pending_statuses = ['new', 'pending_new', 'accepted', 'pending_replace', 'pending_cancel', 'partially_filled']
            pending_orders = [o for o in all_orders if o.status in pending_statuses]
            
            # Filter to only include BUY orders (exclude SELL orders which are bracket children)
            pending_buy_orders = [
                o for o in pending_orders 
                if o.side.lower() == 'buy'
            ]
            
            # Filter out recently cancelled orders (handle Alpaca API propagation delay)
            # Orders cancelled within the last 30 seconds are filtered out even if Alpaca still shows them
            current_time = time()
            pending_buy_orders = [
                o for o in pending_buy_orders
                if str(o.id) not in _cancelled_orders_cache or (current_time - _cancelled_orders_cache[str(o.id)]) > 30
            ]
            
            # Clean up expired entries from cache (older than 30 seconds)
            expired_keys = [oid for oid, ts in _cancelled_orders_cache.items() if current_time - ts > 30]
            for expired_key in expired_keys:
                del _cancelled_orders_cache[expired_key]
            
            logger.debug(f"Found {len(pending_buy_orders)} pending BUY orders (after filtering cancelled)")
        except Exception as e:
            logger.warning(f"Could not fetch pending orders: {e}", exc_info=True)
        
        html_parts = []
        
        # Render pending BUY orders first
        if pending_buy_orders:
            # Get market status once for all orders
            from ..services.trading import get_market_status, get_current_market_price, analyze_pending_order_status
            market_status = get_market_status(api)
            market_open = market_status.get("is_open", True)
            
            # Fetch trade records for pending orders to get stop_loss/take_profit
            trade_records_map = {}
            for order in pending_buy_orders:
                try:
                    trade_record = await db.history.find_one(
                        {"symbol": order.symbol, "action": "buy"},
                        sort=[("timestamp", -1)]
                    )
                    if trade_record:
                        key = (order.symbol, order.side.lower())
                        trade_records_map[key] = trade_record
                except Exception as e:
                    logger.debug(f"Could not fetch trade record for {order.symbol}: {e}")
            
            # Render each pending BUY order
            for order in pending_buy_orders:
                try:
                    trade_record = None
                    key = (order.symbol, order.side.lower())
                    trade_record = trade_records_map.get(key)
                    
                    # Get current price and analyze order status
                    current_price = await get_current_market_price(order.symbol, api)
                    status_info = analyze_pending_order_status(order, current_price, market_open)
                    
                    html_parts.append(pending_order_card(
                        symbol=order.symbol,
                        qty=int(order.qty),
                        side=order.side,
                        order_type=order.type,
                        limit_price=float(order.limit_price) if order.limit_price else None,
                        order_id=str(order.id),
                        stop_loss=trade_record.get("stop_loss") if trade_record else None,
                        take_profit=trade_record.get("take_profit") if trade_record else None,
                        current_price=current_price,
                        market_open=market_open,
                        status_message=status_info.get("status_message"),
                        price_diff_pct=status_info.get("price_diff_pct")
                    ))
                except Exception as e:
                    logger.warning(f"Error rendering pending order {order.symbol}: {e}")
        
        # Render active positions - show exactly what Alpaca returns
        # Trust Alpaca's data - if list_positions() returns it, show it
        if pos:
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
        
        # Empty state - only if no pending orders AND no positions
        if not pending_buy_orders and not pos:
            return HTMLResponse(content=empty_positions())
        
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
            
            # Filter out recently cancelled orders (handle Alpaca API propagation delay)
            # Orders cancelled within the last 30 seconds are filtered out even if Alpaca still shows them
            current_time = time()
            pending_orders = [
                o for o in pending_orders
                if str(o.id) not in _cancelled_orders_cache or (current_time - _cancelled_orders_cache[str(o.id)]) > 30
            ]
            
            # Clean up expired entries from cache (older than 30 seconds)
            expired_keys = [oid for oid, ts in _cancelled_orders_cache.items() if current_time - ts > 30]
            for expired_key in expired_keys:
                del _cancelled_orders_cache[expired_key]
            
            logger.debug(f"Found {len(pending_orders)} pending orders (excluded bracket children and cancelled)")
        except Exception as e:
            logger.warning(f"Could not fetch pending orders: {e}", exc_info=True)
        
        # Get cancelled orders to filter from transaction history
        cancelled_order_details = {}  # {(symbol, side, qty, price): True}
        try:
            # Get all orders (including cancelled ones) to identify which transactions to exclude
            # Try different approaches to get cancelled orders
            try:
                all_orders_all_statuses = api.list_orders(status='all', limit=500)
            except Exception:
                # Fallback: try getting closed orders
                try:
                    all_orders_all_statuses = api.list_orders(status='closed', limit=500)
                except Exception:
                    all_orders_all_statuses = []
            
            for order in all_orders_all_statuses:
                order_status = getattr(order, 'status', '').lower()
                # Check for cancelled statuses (Alpaca uses 'canceled', 'cancelled', 'expired', 'rejected')
                if order_status in ['canceled', 'cancelled', 'expired', 'rejected']:
                    # Store details for matching against transaction history
                    symbol = order.symbol.upper() if hasattr(order, 'symbol') else ''
                    side = order.side.lower() if hasattr(order, 'side') else ''
                    qty = int(order.qty) if hasattr(order, 'qty') and order.qty else 0
                    # Use limit_price if available, otherwise filled_avg_price (for cancelled orders, limit_price is more reliable)
                    price = 0.0
                    if hasattr(order, 'limit_price') and order.limit_price:
                        price = float(order.limit_price)
                    elif hasattr(order, 'filled_avg_price') and order.filled_avg_price:
                        price = float(order.filled_avg_price)
                    key = (symbol, side, qty, round(price, 2))
                    cancelled_order_details[key] = True
            logger.debug(f"Found {len(cancelled_order_details)} cancelled orders to filter from transaction history")
        except Exception as e:
            logger.warning(f"Could not fetch cancelled orders for filtering: {e}", exc_info=True)
        
        # Get transaction history from database
        transaction_history = []
        try:
            history_records = await db.history.find().sort("timestamp", -1).limit(50).to_list(length=50)
            if history_records:
                # Filter out transactions that correspond to cancelled orders
                filtered_history = []
                for record in history_records:
                    symbol = record.get("symbol", "")
                    action = record.get("action", "").lower()
                    qty = record.get("qty", 0)
                    price = round(float(record.get("price", 0.0)), 2)
                    key = (symbol, action, qty, price)
                    
                    # Only include if this transaction doesn't match a cancelled order
                    if key not in cancelled_order_details:
                        filtered_history.append(record)
                    else:
                        logger.debug(f"Filtered out cancelled transaction: {symbol} {action} {qty} @ ${price}")
                
                transaction_history = filtered_history
            logger.debug(f"Found {len(transaction_history)} transaction history records (after filtering cancelled)")
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
        
        # Get market status once for all orders
        from ..services.trading import get_market_status, get_current_market_price, analyze_pending_order_status
        market_status = get_market_status(api)
        market_open = market_status.get("is_open", True)
        
        # Fetch order status info for each pending order
        order_status_map = {}
        for order in pending_orders:
            try:
                current_price = await get_current_market_price(order.symbol, api)
                status_info = analyze_pending_order_status(order, current_price, market_open)
                order_status_map[order.id] = {
                    "current_price": current_price,
                    "market_open": market_open,
                    "status_message": status_info.get("status_message"),
                    "price_diff_pct": status_info.get("price_diff_pct")
                }
            except Exception as e:
                logger.debug(f"Could not get status info for order {order.id}: {e}")
                order_status_map[order.id] = {
                    "current_price": None,
                    "market_open": market_open,
                    "status_message": None,
                    "price_diff_pct": None
                }
        
        # Build HTML using transactions_view helper
        html = transactions_view(
            pending_orders=pending_orders,
            transaction_history=transaction_history,
            trade_records_map=trade_records_map,
            order_status_map=order_status_map
        )
        
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Failed to get transactions: {e}", exc_info=True)
        error_msg = str(e)[:50] if str(e) else "Unknown error"
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"API Error: {error_msg}"
        ))

async def cancel_order(order_id: str = Form(...), db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Cancel a pending order with proper synchronization.
    
    Staff Engineer Approach:
    1. Cancel order via Alpaca API
    2. Verify cancellation succeeded with retry mechanism
    3. Add to cancelled orders cache to filter during propagation window
    4. Filter out cancelled order from response even if Alpaca still shows it
    
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
        
        # Step 1: Cancel the order via Alpaca API
        try:
            api.cancel_order(order_id)
            logger.debug(f"Cancel request sent to Alpaca for order {order_id}")
        except Exception as cancel_error:
            # Check if error indicates order already cancelled or doesn't exist
            error_msg_lower = str(cancel_error).lower()
            if any(phrase in error_msg_lower for phrase in ['not found', 'already canceled', 'already cancelled', 'does not exist']):
                logger.info(f"Order {order_id} already cancelled or doesn't exist - treating as success")
                # Still add to cache to ensure it's filtered
                _cancelled_orders_cache[order_id] = time()
            else:
                raise  # Re-raise if it's a different error
        
        # Step 2: Verify cancellation with retry mechanism (handle Alpaca propagation delay)
        cancellation_verified = False
        max_retries = 3
        retry_delay = 0.3  # 300ms between retries
        
        for attempt in range(max_retries):
            try:
                # Check order status
                order = api.get_order(order_id)
                order_status = getattr(order, 'status', '').lower() if order else ''
                
                # Check if order is actually cancelled
                cancelled_statuses = ['canceled', 'cancelled', 'rejected', 'expired']
                if order_status in cancelled_statuses:
                    cancellation_verified = True
                    logger.debug(f"Order {order_id} cancellation verified (status: {order_status}) on attempt {attempt + 1}")
                    break
                elif attempt < max_retries - 1:
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    logger.debug(f"Order {order_id} still shows status '{order_status}', retrying... (attempt {attempt + 1}/{max_retries})")
            except Exception as verify_error:
                # If we can't get the order, it might be cancelled (order not found)
                error_msg_lower = str(verify_error).lower()
                if any(phrase in error_msg_lower for phrase in ['not found', 'does not exist', '404']):
                    cancellation_verified = True
                    logger.debug(f"Order {order_id} not found - assuming cancelled")
                    break
                elif attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.warning(f"Could not verify cancellation for order {order_id} after {max_retries} attempts: {verify_error}")
        
        # Step 3: Add to cancelled orders cache (TTL: 30 seconds to handle propagation window)
        _cancelled_orders_cache[order_id] = time()
        logger.debug(f"Added order {order_id} to cancelled orders cache")
        
        # Clean up old entries from cache (older than 30 seconds)
        current_time = time()
        expired_keys = [oid for oid, ts in _cancelled_orders_cache.items() if current_time - ts > 30]
        for expired_key in expired_keys:
            del _cancelled_orders_cache[expired_key]
        
        # Step 4: Get updated positions (will filter out cancelled order)
        positions_response = await get_positions(db=db)
        positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
        
        # Return HTML with hx-swap-oob to update positions and show success toast
        if cancellation_verified:
            success_toast = toast_notification("Order canceled successfully", "success", 3000)
        else:
            # Still show success but indicate verification pending
            success_toast = toast_notification("Cancel request sent - order will be removed shortly", "success", 3000)
        
        response_content = htmx_response(
            updates={
                "positions-list": positions_html,
                "toast-container": success_toast
            }
        )
        response = HTMLResponse(content=response_content)
        # hx-swap-oob handles all UI updates - no additional triggers needed
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
        name = config.get('name', 'Balanced Low')
        
        # Render template
        return HTMLResponse(content=templates.get_template("pages/strategy_display.html").render(
            name=name,
            rsi_threshold=rsi_threshold,
            rsi_min=rsi_min,
            sma_proximity_pct=sma_proximity_pct,
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
            for key in ['rsi_threshold', 'rsi_min']:
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
        
        logger.info(f"Strategy updated: RSI<{config['rsi_threshold']}")
        
        if is_htmx:
            # HTMX Gold Standard: Return updated strategy display HTML
            # Strategy changes affect watchlist evaluation, so trigger re-evaluation
            logger.info(f"Strategy updated: RSI {config.get('rsi_min', 20)}-{config['rsi_threshold']}, SMA proximity {config.get('sma_proximity_pct', 3.0)}%")
            
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
        valid_params = ['rsi_threshold', 'rsi_min', 'sma_proximity_pct']
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
            
            if param_name in ['rsi_threshold', 'rsi_min']:
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
        
        # Get current config
        current_config = await get_strategy_config_dict(db)
        
        # Update the parameter
        config = {
            "rsi_threshold": current_config.get('rsi_threshold', 35),
            "rsi_min": current_config.get('rsi_min', 20),
            "sma_proximity_pct": current_config.get('sma_proximity_pct', 3.0),
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
            logger.info(f"Strategy parameter updated: {param_name}={param_value} (new config: RSI {config.get('rsi_min', 20)}-{config.get('rsi_threshold', 35)}, SMA proximity {config.get('sma_proximity_pct', 3.0)}%)")
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
                    logger.error(f"Alpaca API not initialized - cannot fetch data for {symbol}")
                    rejection_reason = "Alpaca API not configured"
                else:
                    try:
                        logger.info(f"Fetching market data for {symbol}")
                        bars, _, _ = await get_market_data(symbol, days=250, db=db)
                        
                        if bars is None or bars.empty:
                            logger.error(f"No bars returned for {symbol} - bars is {bars}")
                            rejection_reason = "No market data returned from API"
                        elif len(bars) < 14:
                            logger.error(f"Insufficient bars for {symbol}: {len(bars)} bars (need 14+)")
                            rejection_reason = f"Insufficient data: only {len(bars)} bars (need 14+)"
                        else:
                            logger.info(f"Got {len(bars)} bars for {symbol}")
                        
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
                                logger.error(f"Could not calculate indicators for {symbol}: {e}", exc_info=True)
                                rejection_reason = f"Indicator calculation failed: {str(e)[:50]}"
                                meets_criteria = False
                        else:
                            # Bars were None or empty - already logged above
                            if not rejection_reason:
                                rejection_reason = "No market data available"
                                
                    except Exception as e:
                        logger.error(f"Could not get market data for {symbol}: {e}", exc_info=True)
                        rejection_reason = f"API error: {str(e)[:100]}"
                        # Fallback: try to get just price
                        try:
                            if api:
                                logger.info(f"Trying fallback price fetch for {symbol}")
                                bars = api.get_bars(symbol, TimeFrame.Day, limit=1, feed='iex').df
                                if bars is not None and not bars.empty:
                                    current_price = float(bars.iloc[-1]['close'])
                                    logger.info(f"Fallback succeeded for {symbol}: ${current_price}")
                                else:
                                    logger.error(f"Fallback returned empty data for {symbol}")
                        except Exception as fallback_error:
                            logger.error(f"Fallback price fetch failed for {symbol}: {fallback_error}", exc_info=True)
                
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
        
        logger.info(f"Analysis preview: {len(preview_data)} symbols, {ready_count} ready, {sum(1 for s in preview_data if s['available'])} available, api={api is not None}")
        logger.info(f"Preview data sample: {preview_data[0] if preview_data else 'No data'}")
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
        logger.info("🚀 get_watch_list called - starting watchlist load")
        # Get watch list from database
        watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
        if watch_list_settings and watch_list_settings.get("value"):
            symbols_str = watch_list_settings.get("value", "")
            symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
            logger.info(f"Got {len(symbols)} symbols from database: {symbols}")
        else:
            # Fallback to default from config
            from ..core.config import SYMBOLS
            symbols = SYMBOLS.copy() if SYMBOLS else []
            logger.info(f"Using default symbols from config: {symbols}")
        
        # Ensure symbols is always a list
        if not isinstance(symbols, list):
            symbols = []
            logger.warning("Symbols was not a list, reset to empty list")
        
        logger.info(f"Final symbols list: {symbols} (length: {len(symbols)})")
        
        from ..core.config import get_strategy_config, calculate_watchlist_config_hash, get_active_custom_strategy, calculate_custom_strategy_hash
        from ..services.custom_strategies import evaluate_strategy
        from ..services.analysis import analyze_comprehensive_stats
        
        # Check for active custom strategy first
        custom_strategy = await get_active_custom_strategy(db)
        use_custom_strategy = custom_strategy is not None
        
        # Initialize variables for both paths
        rsi_threshold = None
        rsi_min = None
        sma_proximity_pct = None
        
        if use_custom_strategy:
            # Use custom strategy
            config_hash = calculate_custom_strategy_hash(custom_strategy)
            logger.info(f"Re-evaluating watchlist with custom strategy: {custom_strategy.get('name')} (config hash: {config_hash[:8]}...)")
            strategy_config = custom_strategy
        else:
            # Fall back to default balanced_low strategy
            strategy_config = await get_strategy_config(db)
            rsi_threshold = strategy_config.get('rsi_threshold', 35)
            rsi_min = strategy_config.get('rsi_min', 20)
            sma_proximity_pct = strategy_config.get('sma_proximity_pct', 3.0)
            config_hash = calculate_watchlist_config_hash(strategy_config)
            logger.info(f"Re-evaluating watchlist with default strategy params: RSI {rsi_min}-{rsi_threshold}, SMA proximity {sma_proximity_pct}% (config hash: {config_hash[:8]}...)")
        
        # Get preview data for symbols (pre-calculated indicators)
        # Parallelize data fetching for better performance
        preview_data = []
        ready_count = 0
        logger.info(f"About to check if symbols list is empty. Symbols: {symbols}, Length: {len(symbols) if symbols else 0}")
        if symbols:
            logger.info(f"Symbols list is NOT empty, starting processing...")
            try:
                import asyncio
                import time
                from ..services.analysis import get_market_data, analyze_technicals
                from ..services.progress import calculate_rsi_progress, calculate_sma_progress
                
                logger.info(f"Processing {len(symbols)} symbols in parallel for watchlist")
                logger.info(f"Symbols: {symbols}")
                start_time = time.time()
                
                async def process_symbol(symbol: str) -> dict:
                    """Process a single symbol and return preview data."""
                    try:
                        # Check evaluation cache first
                        from ..services.watchlist_cache import get_cache
                        cache = get_cache()
                        cached_eval = cache.get_evaluation(symbol, config_hash, ttl_seconds=60)
                        
                        # ALWAYS fetch fresh news headlines (news changes frequently)
                        # Search for news headlines using Firecrawl (preview what AI will read)
                        from ..services.analysis import search_news_headlines
                        logger.info(f"Fetching fresh news headlines for {symbol}")
                        try:
                            firecrawl_headlines = await search_news_headlines(symbol, limit=5)
                            logger.info(f"Got {len(firecrawl_headlines)} headlines for {symbol}")
                        except Exception as e:
                            logger.error(f"Failed to fetch headlines for {symbol}: {e}", exc_info=True)
                            firecrawl_headlines = []
                        
                        # If we have cached evaluation, use it but update with fresh headlines
                        if cached_eval:
                            logger.debug(f"Using cached evaluation for {symbol}, updating with fresh headlines")
                            cached_eval['headlines'] = firecrawl_headlines if firecrawl_headlines else []
                            return cached_eval
                        
                        # No cache - do full processing
                        bars, headlines, news_objects = await get_market_data(symbol, days=100, db=db)
                        
                        current_price = None
                        rsi = None
                        trend = None
                        meets_criteria = False
                        
                        if bars is not None and not bars.empty and len(bars) >= 14:
                            current_price = float(bars.iloc[-1]['close'])
                            current_volume = float(bars.iloc[-1].get('volume', 0)) if 'volume' in bars.columns else None
                            
                            # Use custom strategy if available, otherwise use default
                            if use_custom_strategy:
                                # Use comprehensive stats for custom strategy evaluation
                                if len(bars) >= 20:
                                    stats = analyze_comprehensive_stats(bars)
                                    meets_criteria, rejection_reasons = evaluate_strategy(stats, custom_strategy)
                                    rejection_reason = "; ".join(rejection_reasons) if rejection_reasons else None
                                    
                                    # Extract common metrics for display
                                    rsi = stats.get('rsi')
                                    trend = stats.get('trend', 'NEUTRAL')
                                    sma_200 = stats.get('sma_200')
                                    atr = stats.get('atr')
                                    price_vs_sma_pct = stats.get('price_vs_sma_200_pct')
                                    
                                    # Progress percentages (simplified for custom strategies)
                                    rsi_progress_pct = None
                                    sma_progress_pct = None
                                    if rsi is not None:
                                        # Generic progress calculation
                                        rsi_progress_pct = max(0, min(100, (rsi / 100) * 100))
                                else:
                                    meets_criteria = False
                                    rejection_reason = "Insufficient data for custom strategy (need 20+ bars)"
                                    rsi = None
                                    trend = None
                                    sma_200 = None
                                    atr = None
                                    price_vs_sma_pct = None
                                    rsi_progress_pct = None
                                    sma_progress_pct = None
                                
                                # Additional metrics (not used in custom strategy evaluation but kept for display)
                                earnings_in_days = None
                                pe_ratio = None
                                market_cap = None
                            else:
                                # Default balanced_low strategy logic
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
                        
                        # Use yfinance news results (preview what AI will read)
                        # Fallback to news_objects or headlines if yfinance search failed
                        headlines_display = firecrawl_headlines if firecrawl_headlines else []
                        logger.info(f"[WATCHLIST] {symbol}: Got {len(headlines_display)} headlines from yfinance")
                        
                        if not headlines_display and news_objects:
                            # Fallback: Use news_objects which have headline and URL
                            for news_obj in news_objects[:5]:
                                headline_text = news_obj.get('headline', '').strip()
                                headline_url = news_obj.get('url')
                                if headline_text:
                                    headlines_display.append({
                                        'text': headline_text,
                                        'url': headline_url
                                    })
                        
                        if not headlines_display and headlines:
                            # Fallback: Headlines come as formatted strings like "Headline: ..." or "- ..."
                            # Extract just the headline text for cleaner display (no URL available)
                            for h in headlines[:5]:
                                # Remove "Headline: " prefix if present, or "- " prefix
                                headline_text = h.replace("Headline: ", "").replace("- ", "").strip()
                                if headline_text:
                                    headlines_display.append({
                                        'text': headline_text,
                                        'url': None
                                    })
                        
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
                            "rejection_reason": rejection_reason if not meets_criteria else None,
                            "headlines": headlines_display
                        }
                        
                        logger.info(f"[WATCHLIST] {symbol}: Final result has {len(result.get('headlines', []))} headlines")
                        if result.get('headlines'):
                            logger.info(f"[WATCHLIST] {symbol}: Sample headline: {result['headlines'][0].get('text', 'NO TEXT')[:60]}...")
                        
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
                            "error": str(e)[:100],  # Include error message for UI display
                            "headlines": []
                        }
                        # Don't cache error results
                        return error_result
                
                # Process all symbols in parallel with timeout protection
                tasks = [process_symbol(symbol) for symbol in symbols]
                # Use gather with return_exceptions to handle individual failures gracefully
                # Each symbol has a 30s timeout (via get_market_data internal timeouts)
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Processed {len(symbols)} symbols in {elapsed_time:.2f}s ({elapsed_time/len(symbols):.2f}s per symbol avg)")
                
                # Collect results and count ready symbols
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Exception in parallel processing: {result}", exc_info=True)
                        continue
                    preview_data.append(result)
                    if result.get("meets_criteria", False):
                        ready_count += 1
                
                # Sort preview_data intelligently:
                # 1. Candidates first (meets_criteria=True)
                # 2. Within candidates: sort by RSI (lower = more oversold = better), then by insertion order (newest first)
                # 3. Non-candidates after, sorted by insertion order (newest first)
                symbol_to_data = {item["symbol"]: item for item in preview_data}
                symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols)}
                
                # Create list with original order preserved
                ordered_data = []
                for symbol in symbols:
                    data = symbol_to_data.get(symbol, {
                        "symbol": symbol,
                        "price": None,
                        "rsi": None,
                        "trend": None,
                        "sma_200": None,
                        "atr": None,
                        "price_vs_sma_pct": None,
                        "meets_criteria": False,
                        "error": None,
                        "headlines": []
                    })
                    # Add original index for stable sorting (lower = newer, since we insert at 0)
                    data["_original_index"] = symbol_to_index[symbol]
                    ordered_data.append(data)
                
                # Sort: candidates first, then by RSI (lower is better), then by insertion order (newest first)
                def sort_key(item):
                    is_candidate = item.get("meets_criteria", False)
                    rsi = item.get("rsi")
                    original_idx = item.get("_original_index", 9999)
                    
                    # Candidates come first (0), non-candidates come after (1)
                    candidate_priority = 0 if is_candidate else 1
                    
                    # Within candidates, sort by RSI (lower = better), then by insertion order (newest first)
                    # Within non-candidates, sort by insertion order (newest first)
                    if is_candidate:
                        # Lower RSI is better (more oversold)
                        rsi_value = rsi if rsi is not None else 999
                        return (candidate_priority, rsi_value, original_idx)
                    else:
                        # Non-candidates: newest first (lower index = newer)
                        return (candidate_priority, original_idx, 0)
                
                preview_data = sorted(ordered_data, key=sort_key)
                
                # Remove the temporary _original_index field
                for item in preview_data:
                    item.pop("_original_index", None)
                
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
                
                card_headlines = preview.get("headlines", [])
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
                    "headlines": preview.get("headlines", []),  # CRITICAL: Copy headlines to template data!
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
        strategy_name = custom_strategy.get('name') if use_custom_strategy else strategy_config.get('name', 'Balanced Low')
        rsi_threshold_display = rsi_threshold if not use_custom_strategy else None
        sma_proximity_pct_display = sma_proximity_pct if not use_custom_strategy else None
        
        response = HTMLResponse(content=templates.get_template("pages/watchlist.html").render(
            symbols=symbols,
            preview_data=template_preview_data if template_preview_data else None,
            ready_count=ready_count,
            rsi_threshold=rsi_threshold_display,
            sma_proximity_pct=sma_proximity_pct_display,
            config_hash=config_hash,
            evaluation_timestamp=datetime.now().isoformat(),
            strategy_name=strategy_name,
            is_custom_strategy=use_custom_strategy
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


async def _check_candidates_for_tickers(results: List[Any], db: Any, include_names: bool) -> None:
    """Check candidate status for tickers and add is_candidate field to results.
    
    Modifies results in-place by adding is_candidate boolean field.
    Uses same logic as get_watch_list for consistency.
    """
    if not results:
        return
    
    try:
        from ..core.config import get_strategy_config, get_active_custom_strategy
        from ..services.custom_strategies import evaluate_strategy
        from ..services.analysis import get_market_data, analyze_technicals, analyze_comprehensive_stats
        import asyncio
        
        # Get strategy config (same logic as get_watch_list)
        custom_strategy = await get_active_custom_strategy(db)
        use_custom_strategy = custom_strategy is not None
        
        if use_custom_strategy:
            strategy_config = custom_strategy
        else:
            strategy_config = await get_strategy_config(db)
            rsi_threshold = strategy_config.get('rsi_threshold', 35)
            rsi_min = strategy_config.get('rsi_min', 20)
            sma_proximity_pct = strategy_config.get('sma_proximity_pct', 3.0)
        
        async def check_single_ticker(item: Any, index: int) -> None:
            """Check candidate status for a single ticker."""
            try:
                # Extract symbol from item (could be string or dict)
                if isinstance(item, dict):
                    symbol = item.get('symbol', '')
                else:
                    symbol = str(item).upper()
                
                if not symbol:
                    if isinstance(item, dict):
                        item['is_candidate'] = False
                    else:
                        # Convert string to dict
                        results[index] = {"symbol": symbol, "is_candidate": False}
                    return
                
                # Fetch market data
                bars, _, _ = await get_market_data(symbol, days=100, db=db)
                
                if bars is None or bars.empty or len(bars) < 14:
                    if isinstance(item, dict):
                        item['is_candidate'] = False
                    else:
                        results[index] = {"symbol": symbol, "is_candidate": False}
                    return
                
                current_price = float(bars.iloc[-1]['close'])
                meets_criteria = False
                
                if use_custom_strategy:
                    # Use custom strategy evaluation
                    if len(bars) >= 20:
                        stats = analyze_comprehensive_stats(bars)
                        meets_criteria, _ = evaluate_strategy(stats, custom_strategy)
                    else:
                        meets_criteria = False
                else:
                    # Default balanced_low strategy logic
                    techs = analyze_technicals(bars)
                    rsi = techs.get('rsi')
                    trend = techs.get('trend', 'NEUTRAL')
                    sma_200 = techs.get('sma')
                    
                    # Calculate price vs SMA percentage
                    price_vs_sma_pct = None
                    if current_price and sma_200:
                        price_vs_sma_pct = round(((current_price - sma_200) / sma_200) * 100, 1)
                    
                    # Check filters
                    passes_filters = True
                    earnings_in_days = techs.get('earnings_in_days')
                    pe_ratio = techs.get('pe_ratio')
                    market_cap = techs.get('market_cap')
                    
                    earnings_days_min = strategy_config.get('earnings_days_min', 3)
                    if earnings_in_days is not None and earnings_in_days < earnings_days_min:
                        passes_filters = False
                    
                    pe_ratio_max = strategy_config.get('pe_ratio_max', 50.0)
                    if pe_ratio and pe_ratio > pe_ratio_max:
                        passes_filters = False
                    
                    market_cap_min = strategy_config.get('market_cap_min', 300_000_000)
                    if market_cap and market_cap < market_cap_min:
                        passes_filters = False
                    
                    # Check if meets entry criteria
                    meets_criteria = (
                        passes_filters and
                        current_price is not None and
                        rsi is not None and
                        rsi > rsi_min and
                        rsi < rsi_threshold and
                        trend == "UP" and
                        (price_vs_sma_pct is None or price_vs_sma_pct <= sma_proximity_pct)
                    )
                
                # Add is_candidate field to result
                if isinstance(item, dict):
                    item['is_candidate'] = meets_criteria
                else:
                    # Convert string to dict if needed
                    results[index] = {"symbol": symbol, "is_candidate": meets_criteria}
                    
            except Exception as e:
                logger.debug(f"Error checking candidate status for ticker: {e}")
                # Set to False on error
                if isinstance(item, dict):
                    item['is_candidate'] = False
                else:
                    symbol = str(item).upper() if item else ""
                    results[index] = {"symbol": symbol, "is_candidate": False}
        
        # Process all tickers in parallel (limit to top 5)
        tasks = [check_single_ticker(item, idx) for idx, item in enumerate(results[:5])]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except Exception as e:
        logger.warning(f"Error in candidate checking: {e}")
        # Set all to False on error
        for idx, item in enumerate(results[:5]):
            if isinstance(item, dict):
                item['is_candidate'] = False
            else:
                symbol = str(item).upper() if item else ""
                results[idx] = {"symbol": symbol, "is_candidate": False}


async def search_tickers(
    query: Optional[str] = Query(None, description="Search query for ticker symbol"), 
    include_names: bool = Query(False, description="Include company names"),
    db: Any = Depends(get_scoped_db)
) -> JSONResponse:
    """Search for available tickers using Alpaca API.
    
    Returns a list of tickers that match the query (if provided) or popular tickers.
    Uses Alpaca list_assets API to get tradable stocks.
    Optionally includes company names for better UX.
    Always checks candidate status for top 5 results.
    """
    try:
        # Popular tickers with company names
        popular_with_names = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com Inc.",
            "NVDA": "NVIDIA Corporation",
            "META": "Meta Platforms Inc.",
            "TSLA": "Tesla Inc.",
            "AMD": "Advanced Micro Devices",
            "INTC": "Intel Corporation",
            "NFLX": "Netflix Inc.",
            "JPM": "JPMorgan Chase & Co.",
            "V": "Visa Inc.",
            "JNJ": "Johnson & Johnson",
            "WMT": "Walmart Inc.",
            "MA": "Mastercard Inc.",
            "PG": "Procter & Gamble Co.",
            "DIS": "The Walt Disney Company",
            "BAC": "Bank of America Corp.",
            "XOM": "Exxon Mobil Corporation",
            "CVX": "Chevron Corporation"
        }
        
        if not api:
            # Fallback to popular tickers if Alpaca API not available
            if query:
                query_upper = query.upper()
                filtered = [(t, popular_with_names.get(t, "")) for t in popular_with_names.keys() if query_upper in t]
                filtered = filtered[:20]
            else:
                filtered = list(popular_with_names.items())[:20]
            
            if include_names:
                results = [{"symbol": t, "name": n} for t, n in filtered]
            else:
                results = [t for t, _ in filtered]
            
            # Always check candidate status for top 5 (skip if API not available)
            if results and api:
                await _check_candidates_for_tickers(results[:5], db, include_names)
            
            if include_names:
                return JSONResponse(content={"tickers": results})
            return JSONResponse(content={"tickers": results})
        
        # Use Alpaca API to search for assets
        try:
            # Get all assets (stocks only, tradable)
            assets = api.list_assets(status='active', asset_class='us_equity')
            
            # Filter by query if provided
            if query:
                query_upper = query.upper().strip()
                filtered_assets = [
                    asset for asset in assets 
                    if query_upper in asset.symbol.upper() and asset.tradable
                ]
                # Sort by match position (exact matches first)
                filtered_assets.sort(key=lambda x: (not x.symbol.startswith(query_upper), x.symbol))
                filtered_assets = filtered_assets[:50]
                
                if include_names:
                    results = [{"symbol": asset.symbol, "name": asset.name} for asset in filtered_assets]
                else:
                    results = [asset.symbol for asset in filtered_assets]
                
                # Always check candidate status for top 5
                if results:
                    await _check_candidates_for_tickers(results[:5], db, include_names)
                
                return JSONResponse(content={"tickers": results})
            else:
                # Return popular/well-known tickers if no query
                popular_symbols = list(popular_with_names.keys())
                # Also include some from Alpaca's list
                alpaca_symbols = [asset.symbol for asset in assets[:30] if asset.tradable]
                combined = list(dict.fromkeys(popular_symbols + alpaca_symbols))[:50]
                
                if include_names:
                    results = [
                        {"symbol": sym, "name": popular_with_names.get(sym, "")} 
                        for sym in combined
                    ]
                    # Fill in names from Alpaca for symbols not in popular list
                    asset_dict = {asset.symbol: asset.name for asset in assets if asset.tradable}
                    for item in results:
                        if not item["name"]:
                            item["name"] = asset_dict.get(item["symbol"], "")
                else:
                    results = combined
                
                # Always check candidate status for top 5
                if results:
                    await _check_candidates_for_tickers(results[:5], db, include_names)
                
                return JSONResponse(content={"tickers": results})
        except Exception as e:
            logger.warning(f"Error fetching tickers from Alpaca: {e}")
            # Fallback to popular tickers
            if query:
                query_upper = query.upper()
                filtered = [(t, popular_with_names.get(t, "")) for t in popular_with_names.keys() if query_upper in t]
                filtered = filtered[:20]
            else:
                filtered = list(popular_with_names.items())[:20]
            
            if include_names:
                results = [{"symbol": t, "name": n} for t, n in filtered]
            else:
                results = [t for t, _ in filtered]
            
            # Always check candidate status for top 5
            if results:
                await _check_candidates_for_tickers(results[:5], db, include_names)
            
            if include_names:
                return JSONResponse(content={"tickers": results})
            return JSONResponse(content={"tickers": results})
    except Exception as e:
        logger.error(f"Error searching tickers: {e}", exc_info=True)
        return JSONResponse(content={"tickers": [], "error": str(e)[:100]}, status_code=500)


async def update_watch_list(request: Request, db: Any = Depends(get_scoped_db)) -> Response:
    """Update watch list - supports add, remove, and bulk update operations.
    
    No longer limited to 5 tickers - can have many tickers in watchlist.
    
    Handles both JSON (from fetch/API calls) and form data (from HTMX forms).
    """
    try:
        # Check content type to handle both JSON and form data
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
        else:
            # Handle form data from HTMX
            form_data = await request.form()
            body = dict(form_data)
            # Convert symbol to string if it's a list (form data can be lists)
            if "symbol" in body and isinstance(body["symbol"], list):
                body["symbol"] = body["symbol"][0] if body["symbol"] else ""
            elif "symbol" in body:
                body["symbol"] = str(body["symbol"])
        
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
            
            # Add symbol to the beginning (newest first) - no limit
            current_symbols.insert(0, symbol)
            
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
        
        # For HTMX requests (form submissions), return updated watchlist HTML
        if is_htmx:
            # Return the updated watchlist HTML by calling get_watch_list
            # Import here to avoid circular dependency
            watchlist_html_response = await get_watch_list(db)
            
            # Add success toast notification for add/remove actions
            if action in ["add", "remove"]:
                from .templates import toast_notification
                symbol = body.get("symbol", "").strip().upper()
                if action == "add":
                    toast_msg = toast_notification(
                        message=f"{symbol} added to watchlist",
                        type="success",
                        duration=3000,
                        target_id="toast-container"
                    )
                else:  # remove
                    toast_msg = toast_notification(
                        message=f"{symbol} removed from watchlist",
                        type="success",
                        duration=5000,
                        target_id="toast-container",
                        undo_action="/api/watch-list",
                        undo_symbol=symbol
                    )
                
                # Append toast to response using hx-swap-oob
                response_content = watchlist_html_response.body.decode('utf-8') + toast_msg
                response = HTMLResponse(content=response_content)
                # HTMX Best Practice: Trigger refresh of signal hunter section to update add/remove buttons
                response.headers["HX-Trigger"] = "watchlistUpdated"
                return response
            else:
                # For bulk updates, just return the watchlist HTML
                return watchlist_html_response
        
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
                # Fetch article content for AI analysis
                bars, headlines, news_objects = await get_market_data(symbol, days=500, db=db, fetch_firecrawl_content=True)
                if bars is None or bars.empty or len(bars) < 14:
                    continue
                
                techs = analyze_technicals(bars)
                
                # Get AI analysis if available
                ai_score = None
                ai_reason = None
                risk_level = "UNKNOWN"
                
                if _ai_engine:
                    try:
                        from ..core.config import get_strategy_config
                        strategy_config = await get_strategy_config(db)
                        strategy_prompt = get_balanced_low_prompt(strategy_config)
                        verdict = _ai_engine.analyze(
                            ticker=symbol,
                            techs=techs,
                            headlines="\n".join(headlines[:3]) if headlines else "No recent news",
                            strategy_prompt=strategy_prompt,
                            strategy_config=strategy_config
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
    This demonstrates htmx out-of-band swap capability.
    
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
        # Get API client from account manager
        account_manager = AlpacaAccountManager(db)
        active_account = await account_manager.get_active_account()
        api_client = account_manager.get_api_client(active_account) if active_account else None
        
        order = await place_order(symbol, 'buy', f"Quick buy from card ({qty} shares)", 10, db, qty=qty, api_client=api_client)
        
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
            toast_message = f"Buy order placed for {symbol} - Order ID: {order_id}"
        
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

async def get_buy_order_details(
    symbol: str = Form(...),
    db: Any = Depends(get_scoped_db)
) -> JSONResponse:
    """Get buy order details for confirmation modal.
    
    Calculates position size, prices, and risk metrics for a buy order.
    Returns JSON data that can be used to populate the buy confirmation modal.
    
    Args:
        symbol: Stock symbol to get order details for
        
    Returns:
        JSON with order details: price, quantity, stop_loss, take_profit, total_cost, buying_power, risk_amount
    """
    if not check_alpaca_config() or api is None:
        return JSONResponse(
            content={"error": "Alpaca API not configured"},
            status_code=400
        )
    
    try:
        symbol = symbol.upper().strip()
        if not symbol:
            return JSONResponse(
                content={"error": "Symbol is required"},
                status_code=400
            )
        
        # Get market data and technicals
        df, headlines, news_objects = await get_market_data(symbol)
        if df is None or df.empty:
            return JSONResponse(
                content={"error": f"No market data available for {symbol}"},
                status_code=404
            )
        
        techs = analyze_technicals(df)
        if not techs or 'price' not in techs or 'atr' not in techs:
            return JSONResponse(
                content={"error": f"Could not calculate technical indicators for {symbol}"},
                status_code=500
            )
        
        # Convert to Python types to avoid numpy/pandas serialization issues
        price = float(techs['price'])
        atr = float(techs['atr'])
        
        # Calculate position size
        qty = calculate_position_size(price, atr)
        if qty < 1:
            return JSONResponse(
                content={"error": f"Position size too small for {symbol} (risk/budget constraints)"},
                status_code=400
            )
        
        # Ensure qty is Python int
        qty = int(qty)
        
        # Calculate order prices
        limit_price = round(price, 2)
        stop_price = round(price - (2 * atr), 2)
        take_profit = round(price + (3 * atr), 2)
        
        # Get buying power
        buying_power = 10000.0  # Default fallback
        try:
            acct = api.get_account()
            buying_power = float(acct.buying_power)
        except Exception as e:
            logger.warning(f"Could not get account info: {e}, using default buying power")
        
        # Calculate costs and risks
        total_cost = qty * limit_price
        stop_distance = price - stop_price
        risk_amount = stop_distance * qty
        
        # Check if sufficient buying power (convert numpy bool_ to Python bool)
        sufficient_power = bool(buying_power >= total_cost)
        
        return JSONResponse(content={
            "symbol": symbol,
            "price": float(limit_price),
            "quantity": int(qty),
            "stop_loss": float(stop_price),
            "take_profit": float(take_profit),
            "total_cost": float(round(total_cost, 2)),
            "buying_power": float(round(buying_power, 2)),
            "risk_amount": float(round(risk_amount, 2)),
            "stop_distance": float(round(stop_distance, 2)),
            "sufficient_power": sufficient_power,
            "atr": float(round(atr, 2))
        })
        
    except Exception as e:
        logger.error(f"Error getting buy order details for {symbol}: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to get order details: {str(e)[:150]}"},
            status_code=500
        )

async def get_inline_buy_interface(
    symbol: str = Form(...),
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Get inline buy interface HTML fragment.
    
    Returns the buy interface with order details pre-loaded.
    This is loaded via HTMX to avoid Alpine.js async function issues.
    
    Args:
        symbol: Stock symbol to get buy interface for
        
    Returns:
        HTMLResponse with inline buy interface
    """
    if not check_alpaca_config() or api is None:
        return HTMLResponse(
            content='<div class="text-red-400 text-sm p-3">Alpaca API not configured</div>',
            status_code=400
        )
    
    try:
        symbol = symbol.upper().strip()
        if not symbol:
            return HTMLResponse(
                content='<div class="text-red-400 text-sm p-3">Symbol is required</div>',
                status_code=400
            )
        
        # Get order details (reuse logic from get_buy_order_details)
        df, headlines, news_objects = await get_market_data(symbol)
        if df is None or df.empty:
            return HTMLResponse(
                content=f'<div class="text-red-400 text-sm p-3">No market data available for {symbol}</div>',
                status_code=404
            )
        
        techs = analyze_technicals(df)
        if not techs or 'price' not in techs or 'atr' not in techs:
            return HTMLResponse(
                content=f'<div class="text-red-400 text-sm p-3">Could not calculate technical indicators for {symbol}</div>',
                status_code=500
            )
        
        # Convert to Python types
        price = float(techs['price'])
        atr = float(techs['atr'])
        
        # Calculate position size
        qty = calculate_position_size(price, atr)
        if qty < 1:
            return HTMLResponse(
                content=f'<div class="text-red-400 text-sm p-3">Position size too small for {symbol}</div>',
                status_code=400
            )
        
        qty = int(qty)
        
        # Calculate order prices
        limit_price = round(price, 2)
        stop_price = round(price - (2 * atr), 2)
        take_profit = round(price + (3 * atr), 2)
        
        # Get buying power
        buying_power = 10000.0
        try:
            acct = api.get_account()
            buying_power = float(acct.buying_power)
        except Exception as e:
            logger.warning(f"Could not get account info: {e}, using default buying power")
        
        max_quantity = min(99, int(buying_power / limit_price))
        
        # Render inline buy interface
        interface_html = templates.get_template("components/inline_buy_interface.html").render(
            symbol=symbol,
            quantity=qty,
            price=limit_price,
            stop_loss=stop_price,
            take_profit=take_profit,
            buying_power=buying_power,
            max_quantity=max_quantity
        )
        
        return HTMLResponse(content=interface_html)
        
    except Exception as e:
        logger.error(f"Error getting inline buy interface for {symbol}: {e}", exc_info=True)
        error_html = templates.get_template("partials/error_message.html").render(
            message=f"Failed to load buy interface: {str(e)[:150]}"
        )
        return HTMLResponse(content=error_html, status_code=500)

async def show_buy_confirmation_modal(
    symbol: str = Form(...),
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Show buy confirmation modal with order details.
    
    Fetches order details and returns the buy confirmation modal HTML.
    
    Args:
        symbol: Stock symbol to show buy confirmation for
        
    Returns:
        HTMLResponse with buy confirmation modal
    """
    if not check_alpaca_config() or api is None:
        return HTMLResponse(
            content=toast_notification(
                message="Alpaca API not configured",
                type="error"
            ),
            status_code=400
        )
    
    try:
        symbol = symbol.upper().strip()
        if not symbol:
            return HTMLResponse(
                content=toast_notification(
                    message="Symbol is required",
                    type="error"
                ),
                status_code=400
            )
        
        # Get order details
        df, headlines, news_objects = await get_market_data(symbol)
        if df is None or df.empty:
            return HTMLResponse(
                content=toast_notification(
                    message=f"No market data available for {symbol}",
                    type="error"
                ),
                status_code=404
            )
        
        techs = analyze_technicals(df)
        if not techs or 'price' not in techs or 'atr' not in techs:
            return HTMLResponse(
                content=toast_notification(
                    message=f"Could not calculate technical indicators for {symbol}",
                    type="error"
                ),
                status_code=500
            )
        
        price = techs['price']
        atr = techs['atr']
        
        # Calculate position size
        qty = calculate_position_size(price, atr)
        if qty < 1:
            return HTMLResponse(
                content=toast_notification(
                    message=f"Position size too small for {symbol} (risk/budget constraints)",
                    type="error"
                ),
                status_code=400
            )
        
        # Calculate order prices
        limit_price = round(price, 2)
        stop_price = round(price - (2 * atr), 2)
        take_profit = round(price + (3 * atr), 2)
        
        # Get buying power
        buying_power = 10000.0  # Default fallback
        try:
            acct = api.get_account()
            buying_power = float(acct.buying_power)
        except Exception as e:
            logger.warning(f"Could not get account info: {e}, using default buying power")
        
        # Render modal body content
        body_content = templates.get_template("components/buy_confirmation_body.html").render(
            symbol=symbol,
            quantity=qty,
            price=limit_price,
            stop_loss=stop_price,
            take_profit=take_profit,
            buying_power=buying_power
        )
        
        # Wrap in modal using standard pattern - use htmx_modal_wrapper helper for consistency
        from ..api.templates import htmx_modal_wrapper
        modal_html = htmx_modal_wrapper(
            modal_id=f"buy-modal-{symbol.lower()}",
            title=f"Confirm Buy Order - {symbol}",
            content=body_content,
            size="medium",
            icon="shopping-cart",
            icon_color="text-green-400"
        )
        
        return HTMLResponse(content=modal_html)
        
    except Exception as e:
        logger.error(f"Error showing buy confirmation modal for {symbol}: {e}", exc_info=True)
        # HTMX Best Practice: Return HTML error fragment, not JSON
        error_html = templates.get_template("partials/error_message.html").render(
            message=f"Failed to load buy confirmation: {str(e)[:150]}"
        )
        return HTMLResponse(content=error_html, status_code=500)

async def confirm_buy_order(
    request: Request,
    symbol: str = Form(...),
    qty: Optional[int] = Form(None),
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Confirm and execute buy order.
    
    Validates order details and executes the buy order via place_order.
    Returns success/error response with toast notification and updates positions list.
    
    Args:
        symbol: Stock symbol to buy
        qty: Number of shares (optional, will calculate if not provided)
        
    Returns:
        HTMLResponse with hx-swap-oob updates for positions list and toast notification
    """
    if not check_alpaca_config() or api is None:
        error_toast = toast_notification(
            message="Alpaca API not configured",
            type="error",
            duration=5000
        )
        return HTMLResponse(content=htmx_response(updates={"toast-container": error_toast}), status_code=400)
    
    try:
        # Fallback: Try to get form data directly if Form() parsing fails
        try:
            form_data = await request.form()
            if not symbol or symbol == "..." or (not qty and "qty" in form_data):
                symbol = form_data.get("symbol", symbol or "")
                qty_str = form_data.get("qty")
                if qty_str:
                    try:
                        qty = int(qty_str)
                    except (ValueError, TypeError):
                        qty = None
        except Exception as form_error:
            logger.debug(f"Could not read form data directly: {form_error}")
        
        # Log incoming request for debugging
        logger.info(f"confirm_buy_order called with symbol={symbol}, qty={qty}, qty_type={type(qty)}")
        
        # Validate symbol
        symbol = symbol.upper().strip() if symbol else ""
        if not symbol:
            logger.warning("confirm_buy_order: Symbol is empty")
            error_toast = toast_notification(
                message="Symbol is required",
                type="error",
                duration=5000
            )
            return HTMLResponse(content=htmx_response(updates={"toast-container": error_toast}), status_code=400)
        
        # Validate quantity - handle string conversion
        if qty is not None:
            try:
                # Handle both string and int types
                if isinstance(qty, str):
                    qty = int(qty.strip())
                else:
                    qty = int(qty)
                if qty < 1 or qty > 99:
                    logger.warning(f"confirm_buy_order: Quantity {qty} out of range")
                    error_toast = toast_notification(
                        message="Quantity must be between 1-99",
                        type="error",
                        duration=5000
                    )
                    return HTMLResponse(content=htmx_response(updates={"toast-container": error_toast}), status_code=400)
            except (ValueError, TypeError) as e:
                logger.error(f"confirm_buy_order: Invalid quantity conversion - qty={qty}, type={type(qty)}, error={e}")
                error_toast = toast_notification(
                    message=f"Invalid quantity: {qty}. Please enter a number between 1-99.",
                    type="error",
                    duration=5000
                )
                return HTMLResponse(content=htmx_response(updates={"toast-container": error_toast}), status_code=400)
        else:
            # qty is None - will be calculated automatically
            logger.info(f"confirm_buy_order: Quantity not provided, will calculate automatically")
        
        logger.info(f"Buy order confirmation requested for {symbol}, quantity: {qty}")
        
        # Get API client from account manager
        account_manager = AlpacaAccountManager(db)
        active_account = await account_manager.get_active_account()
        api_client = account_manager.get_api_client(active_account) if active_account else None
        
        # Place order (qty=None will trigger automatic calculation)
        order = await place_order(symbol, 'buy', f"Buy order from confirmation modal ({qty or 'auto'} shares)", 10, db, qty=qty, api_client=api_client)
        
        if order is None:
            logger.error(f"place_order returned None for {symbol}")
            error_toast = toast_notification(
                message=f"Order placement failed for {symbol}. Check server logs for details.",
                type="error",
                duration=5000
            )
            return HTMLResponse(content=htmx_response(updates={"toast-container": error_toast}), status_code=500)
        
        # Order succeeded - get updated positions
        logger.info(f"Buy order placed successfully for {symbol}: {order}")
        
        # Check if order is still pending (not immediately filled)
        is_pending = False
        order_id = None
        try:
            if hasattr(order, 'id'):
                order_id = str(order.id)
            if hasattr(order, 'status'):
                # Check if order status indicates it's still pending
                pending_statuses = ['new', 'pending_new', 'accepted', 'pending_replace', 'pending_cancel', 'partially_filled']
                is_pending = order.status.lower() in pending_statuses
            else:
                # If no status, check via API
                try:
                    if order_id and api:
                        current_order = api.get_order(order_id)
                        if current_order:
                            is_pending = current_order.status.lower() in pending_statuses
                except Exception:
                    pass  # If we can't check, assume filled
        except Exception as e:
            logger.debug(f"Could not determine order pending status: {e}")
        
        # Get updated positions HTML (includes pending orders)
        positions_response = await get_positions(db=db)
        positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
        
        # Create success toast notification
        success_toast = toast_notification(
            message=f"Buy order placed for {symbol} ({qty or 'auto'} shares)",
            type="success",
            duration=5000
        )
        
        # HTMX Best Practice: Use hx-swap-oob for multi-element updates
        # Update positions (includes pending orders) and toast
        response_content = htmx_response(
            updates={
                "positions-list": positions_html,
                "toast-container": success_toast
            }
        )
        
        response = HTMLResponse(content=response_content)
        # HTMX Best Practice: Use HX-Trigger header for cross-component events
        # Pass symbol and pending status in buyOrderPlaced event for smart UX
        # Trigger simple events immediately
        response.headers["HX-Trigger"] = "refreshPositions"
        # Trigger buyOrderPlaced after transactions settle with order status info
        event_data = {
            "symbol": symbol.upper(),
            "isPending": is_pending,
            "orderId": order_id
        }
        response.headers["HX-Trigger-After-Settle"] = json.dumps({"buyOrderPlaced": event_data})
        return response
        
    except Exception as e:
        logger.error(f"Buy order confirmation failed for {symbol}: {e}", exc_info=True)
        error_msg = str(e)[:150] if str(e) else "Unknown error occurred"
        error_toast = toast_notification(
            message=f"Buy failed: {error_msg}. Check server logs for details",
            type="error",
            duration=5000
        )
        # HTMX Best Practice: Use hx-swap-oob for error toast
        response_content = htmx_response(updates={"toast-container": error_toast})
        return HTMLResponse(content=response_content, status_code=500)

async def get_latest_scan(
    date: Optional[str] = None,
    db: Any = Depends(get_scoped_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> JSONResponse:
    """Get latest scan (or final scan if after 6 PM ET)."""
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
    
    symbol_upper = symbol.upper()
    
    try:
        # Check if position exists BEFORE trying to close it
        position_qty = 0
        position_exists = False
        try:
            position = api.get_position(symbol_upper)
            if position and hasattr(position, 'qty'):
                position_qty = float(position.qty)
                position_exists = position_qty > 0
        except Exception as pos_error:
            # Position doesn't exist or already closed
            error_msg = str(pos_error).lower()
            logger.debug(f"Position check for {symbol_upper}: {error_msg}")
            position_exists = False
        
        # If position doesn't exist, refresh and return early
        if not position_exists:
            positions_response = await get_positions(db=db)
            positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
            error_toast = toast_notification(
                f"⚠️ Position for {symbol_upper} already closed or not found. Refreshing positions...",
                "warning",
                5000
            )
            response_content = htmx_response(
                updates={
                    "positions-list": positions_html,
                    "toast-container": error_toast
                }
            )
            response = HTMLResponse(content=response_content, status_code=200)
            response.headers["HX-Trigger"] = "refreshPositions"
            return response
        
        # Try to close the position - Alpaca will handle validation
        try:
            api.close_position(symbol_upper)
        except Exception as close_error:
            error_msg = str(close_error).lower()
            logger.warning(f"Error closing position {symbol_upper}: {error_msg}")
            
            # Check for various "position already closed" error patterns
            already_closed_patterns = [
                "position not found",
                "no position",
                "position does not exist",
                "position already closed",
                "insufficient qty",
                "available: 0",
                "404",
                "not found"
            ]
            
            is_already_closed = any(pattern in error_msg for pattern in already_closed_patterns)
            
            if is_already_closed:
                # Position was already closed - refresh and show message
                positions_response = await get_positions(db=db)
                positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
                error_toast = toast_notification(
                    f"⚠️ Position for {symbol_upper} already closed. Refreshing positions...",
                    "warning",
                    5000
                )
                response_content = htmx_response(
                    updates={
                        "positions-list": positions_html,
                        "toast-container": error_toast
                    }
                )
                response = HTMLResponse(content=response_content, status_code=200)
                response.headers["HX-Trigger"] = "refreshPositions"
                return response
            
            # Handle partial position case (some shares already sold)
            if "insufficient qty" in error_msg and "available:" in error_msg:
                # Extract available qty from error message
                import re
                match = re.search(r'available:\s*(\d+)', error_msg)
                available_qty = int(match.group(1)) if match else 0
                
                if available_qty > 0:
                    # Try to close the available qty instead
                    try:
                        # Close partial position by specifying qty
                        api.submit_order(
                            symbol=symbol_upper,
                            qty=available_qty,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        logger.info(f"Closed partial position for {symbol_upper}: {available_qty} shares (out of {position_qty})")
                        # Successfully closed partial position - refresh and show success message
                        positions_response = await get_positions(db=db)
                        positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
                        success_toast = toast_notification(
                            f"✅ Sold {available_qty} shares of {symbol_upper} ({int(position_qty - available_qty)} already closed)",
                            "success",
                            6000
                        )
                        response_content = htmx_response(
                            updates={
                                "positions-list": positions_html,
                                "toast-container": success_toast
                            }
                        )
                        response = HTMLResponse(content=response_content, status_code=200)
                        response.headers["HX-Trigger"] = "refreshPositions"
                        return response
                    except Exception as partial_error:
                        # If partial close also fails, refresh and show message
                        positions_response = await get_positions(db=db)
                        positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
                        error_toast = toast_notification(
                            f"⚠️ Only {available_qty} of {int(position_qty)} shares available for {symbol_upper}. Position may be partially closed. Refreshing...",
                            "warning",
                            6000
                        )
                        response_content = htmx_response(
                            updates={
                                "positions-list": positions_html,
                                "toast-container": error_toast
                            }
                        )
                        response = HTMLResponse(content=response_content, status_code=200)
                        response.headers["HX-Trigger"] = "refreshPositions"
                        return response
                else:
                    # No shares available - refresh and show message
                    positions_response = await get_positions(db=db)
                    positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
                    error_toast = toast_notification(
                        f"⚠️ Position for {symbol_upper} already fully closed. Refreshing positions...",
                        "warning",
                        5000
                    )
                    response_content = htmx_response(
                        updates={
                            "positions-list": positions_html,
                            "toast-container": error_toast
                        }
                    )
                    response = HTMLResponse(content=response_content, status_code=200)
                    response.headers["HX-Trigger"] = "refreshPositions"
                    return response
            else:
                # Unknown error - log it but still refresh positions
                logger.error(f"Unexpected error closing position {symbol_upper}: {close_error}", exc_info=True)
                positions_response = await get_positions(db=db)
                positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
                error_toast = toast_notification(
                    f"⚠️ Error closing {symbol_upper}: {str(close_error)[:50]}. Refreshing positions...",
                    "error",
                    6000
                )
                response_content = htmx_response(
                    updates={
                        "positions-list": positions_html,
                        "toast-container": error_toast
                    }
                )
                response = HTMLResponse(content=response_content, status_code=200)
                response.headers["HX-Trigger"] = "refreshPositions"
                return response
        
        # Get updated positions HTML - pass db explicitly
        positions_response = await get_positions(db=db)
        positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
        
        # Determine success message based on what was closed
        if position_qty > 0:
            # Check if we closed a partial position
            try:
                remaining_pos = api.get_position(symbol_upper)
                remaining_qty = float(remaining_pos.qty) if hasattr(remaining_pos, 'qty') else 0
                if remaining_qty > 0:
                    closed_qty = int(position_qty - remaining_qty)
                    success_toast = toast_notification(
                        f"✅ Sold {closed_qty} shares of {symbol_upper} ({remaining_qty} remaining)",
                        "success",
                        6000
                    )
                else:
                    success_toast = toast_notification(
                        f"✅ Position closed: {symbol_upper} ({int(position_qty)} shares sold)",
                        "success",
                        5000
                    )
            except Exception:
                # Position fully closed
                success_toast = toast_notification(
                    f"✅ Position closed: {symbol_upper} ({int(position_qty)} shares sold)",
                    "success",
                    5000
                )
        else:
            success_toast = toast_notification(
                f"✅ Position closed: {symbol_upper}",
                "success",
                5000
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
        error_msg = str(e).lower()
        logger.error(f"Quick sell failed for {symbol_upper}: {e}", exc_info=True)
        
        # Check for various "position already closed" error patterns
        already_closed_patterns = [
            "position not found",
            "no position",
            "position does not exist",
            "position already closed",
            "insufficient qty",
            "available: 0",
            "404",
            "not found"
        ]
        
        is_already_closed = any(pattern in error_msg for pattern in already_closed_patterns)
        
        # Handle specific error cases with appropriate messages
        if is_already_closed or "insufficient qty" in error_msg or "available: 0" in error_msg:
            # Position was already closed or has 0 qty - force immediate refresh
            # Extract available qty if mentioned in error
            import re
            match = re.search(r'available:\s*(\d+)', error_msg)
            if match:
                available_qty = int(match.group(1))
                if available_qty > 0:
                    error_toast = toast_notification(
                        f"⚠️ Only {available_qty} shares available for {symbol_upper}. Position may be partially closed. Refreshing...",
                        "warning",
                        6000
                    )
                else:
                    error_toast = toast_notification(
                        f"⚠️ Position for {symbol_upper} already closed. Refreshing positions...",
                        "warning",
                        5000
                    )
            else:
                error_toast = toast_notification(
                    f"⚠️ Position for {symbol_upper} already closed. Refreshing positions...",
                    "warning",
                    5000
                )
            # Force immediate positions refresh to remove stale position
            try:
                positions_response = await get_positions(db=db)
                positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
                
                response_content = htmx_response(
                    updates={
                        "positions-list": positions_html,
                        "toast-container": error_toast
                    }
                )
                response = HTMLResponse(content=response_content, status_code=200)
                # Trigger immediate refresh
                response.headers["HX-Trigger"] = "refreshPositions"
                return response
            except Exception as pos_error:
                logger.warning(f"Could not refresh positions after sell error: {pos_error}")
                # Fallback: just show toast and let polling handle refresh
                response_content = error_toast
                response = HTMLResponse(content=response_content, status_code=200)
                response.headers["HX-Trigger"] = "refreshPositions"
                return response
        elif "position does not exist" in error_msg or "not found" in error_msg:
            # Position doesn't exist
            error_toast = toast_notification(
                f"⚠️ No position found for {symbol_upper}. Refreshing positions...",
                "warning",
                5000
            )
        else:
            # Generic error handling
            error_toast = toast_notification(
                f"❌ Sell failed for {symbol_upper}: {str(e)[:80]}",
                "error",
                6000
            )
        
        # Try to refresh positions on error to keep UI in sync, but don't fail if this also errors
        # Only update positions-list if we can successfully get positions
        try:
            positions_response = await get_positions(db=db)
            positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
            
            # Return response with both updates
            response_content = htmx_response(
                updates={
                    "positions-list": positions_html,
                    "toast-container": error_toast
                }
            )
        except Exception as pos_error:
            logger.warning(f"Could not refresh positions after sell error: {pos_error}")
            # Only return toast notification - let positions refresh naturally via polling
            response_content = error_toast
        
        return HTMLResponse(content=response_content, status_code=200)  # Return 200 so HTMX processes the response

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
    logger.info("[WS] WebSocket connection attempt received")
    
    try:
        await websocket.accept()
        logger.info("[WS] WebSocket connection accepted")
    except Exception as e:
        logger.error(f"[WS] Failed to accept WebSocket: {e}", exc_info=True)
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
            logger.info(f"[WS] Using custom symbols: {symbols}")
        else:
            symbols = await _get_target_symbols(db)
        total = len(symbols)
        stocks_with_analysis = []
        
        logger.debug(f"[WS] Starting analysis of {total} symbols: {symbols}")
        
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
        logger.debug("[WS] Sent initial progress message")
        
        import asyncio
        await asyncio.sleep(0.1)  # Small delay for UI update
        
        # Track failures for reporting
        failed_stocks = []
        
        # Analyze each stock with progress updates
        for idx, symbol in enumerate(symbols, 1):
            logger.debug(f"[WS] [{idx}/{total}] Starting analysis for {symbol}")
            
            try:
                # Check cache before expensive operations
                cached = await radar_service.get_cached_analysis(symbol, strategy_id)
                cache_hit = cached and cached['fresh']
                
                if cache_hit:
                    logger.debug(f"[WS] [{idx}/{total}] Cache hit for {symbol}")
                    cache_hits += 1
                    cached_analysis = cached['analysis']
                    config = await get_strategy_config_dict(db)
                    ai_score = cached_analysis.get('verdict', {}).get('score') if isinstance(cached_analysis.get('verdict'), dict) else (cached_analysis.get('verdict').score if hasattr(cached_analysis.get('verdict'), 'score') else None)
                    confidence_boost = cached_analysis.get('confidence', {}).get('boost', 0.0)
                    adjusted_score = (ai_score or 0) + confidence_boost
                    
                    # Candidate selection uses ONLY technical thresholds - NO AI score
                    techs_data = cached_analysis.get('techs', {})
                    rsi = techs_data.get('rsi', 0)
                    price = techs_data.get('price', 0)
                    sma_200 = techs_data.get('sma', 0)
                    trend = techs_data.get('trend', 'UNKNOWN')
                    
                    rsi_threshold = config.get('rsi_threshold', 35)
                    rsi_min = config.get('rsi_min', 20)
                    sma_proximity_pct = config.get('sma_proximity_pct', 3.0)
                    
                    # Calculate price vs SMA percentage
                    price_vs_sma_pct = None
                    if price and sma_200 and sma_200 > 0:
                        price_vs_sma_pct = ((price - sma_200) / sma_200) * 100
                    
                    # Candidate criteria: ONLY technical thresholds (RSI, Trend, SMA proximity)
                    meets_criteria = (
                        rsi is not None and rsi > 0 and
                        rsi > rsi_min and
                        rsi < rsi_threshold and
                        trend == "UP" and
                        (price_vs_sma_pct is None or price_vs_sma_pct <= sma_proximity_pct)
                    )
                    
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
                logger.debug(f"[WS] [{idx}/{total}] Sent progress update for {symbol}")
                
                # Fetch market data
                logger.debug(f"[WS] [{idx}/{total}] Fetching market data for {symbol}...")
                import time
                start_time = time.time()
                
                try:
                    # Fetch article content for AI analysis in WebSocket streaming
                    bars, headlines, news_objects = await get_market_data(symbol, days=500, db=db, fetch_firecrawl_content=True)
                    fetch_time = time.time() - start_time
                    logger.debug(f"[WS] [{idx}/{total}] Market data fetch for {symbol} took {fetch_time:.2f}s")
                except Exception as e:
                    error_msg = f"Failed to fetch market data: {str(e)[:100]}"
                    logger.error(f"[WS] [{idx}/{total}] {symbol}: {error_msg}", exc_info=True)
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
                    logger.warning(f"[WS] [{idx}/{total}] {symbol}: bars is None")
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
                    logger.warning(f"[WS] [{idx}/{total}] {symbol}: bars is empty")
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
                    logger.warning(f"[WS] [{idx}/{total}] {symbol}: Only {len(bars)} days of data (need 14+)")
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
                
                logger.debug(f"[WS] [{idx}/{total}] {symbol}: Got {len(bars)} days of data, {len(headlines)} headlines")
                
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
                logger.debug(f"[WS] [{idx}/{total}] Sent technicals progress for {symbol}")
                
                # Calculate technical indicators
                logger.debug(f"[WS] [{idx}/{total}] Calculating technical indicators for {symbol}...")
                techs_start = time.time()
                
                try:
                    techs = analyze_technicals(bars)
                    techs_time = time.time() - techs_start
                    logger.debug(f"[WS] [{idx}/{total}] Technicals calculated for {symbol} in {techs_time:.2f}s: RSI={techs['rsi']:.1f}, Trend={techs['trend']}, Price=${techs['price']:.2f}")
                except Exception as e:
                    error_msg = f"Technical analysis failed: {str(e)[:100]}"
                    logger.error(f"[WS] [{idx}/{total}] Technical analysis failed for {symbol}: {e}", exc_info=True)
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
                logger.debug(f"[WS] [{idx}/{total}] Sent AI progress for {symbol}")
                
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
                    logger.debug(f"[WS] [{idx}/{total}] Running AI analysis for {symbol}...")
                    ai_start = time.time()
                    try:
                        from ..core.config import get_strategy_config
                        strategy_config = await get_strategy_config(db)
                        strategy_prompt = get_balanced_low_prompt(strategy_config)
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
                            strategy_prompt=strategy_prompt,
                            strategy_config=strategy_config
                        )
                        ai_score = verdict.score
                        ai_reason = verdict.reason[:100] + "..." if len(verdict.reason) > 100 else verdict.reason
                        risk_level = verdict.risk_level
                        
                        ai_time = time.time() - ai_start
                        logger.debug(f"[WS] [{idx}/{total}] AI analysis complete for {symbol} in {ai_time:.2f}s: Score={ai_score}/10, Risk={risk_level}, Action={verdict.action}")
                    except Exception as e:
                        logger.error(f"[WS] [{idx}/{total}] AI analysis failed for {symbol}: {e}", exc_info=True)
                        logger.error(f"   Error type: {type(e).__name__}, Error: {str(e)}")
                else:
                    logger.warning(f"[WS] [{idx}/{total}] AI engine not available, skipping AI analysis for {symbol}")
                
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
                
                # Candidate selection uses ONLY technical thresholds - NO AI score
                rsi = techs.get('rsi', 0)
                price = techs.get('price', 0)
                sma_200 = techs.get('sma', 0)
                trend = techs.get('trend', 'UNKNOWN')
                
                rsi_threshold = config.get('rsi_threshold', 35)
                rsi_min = config.get('rsi_min', 20)
                sma_proximity_pct = config.get('sma_proximity_pct', 3.0)
                
                # Calculate price vs SMA percentage
                price_vs_sma_pct = None
                if price and sma_200 and sma_200 > 0:
                    price_vs_sma_pct = ((price - sma_200) / sma_200) * 100
                
                # Candidate criteria: ONLY technical thresholds (RSI, Trend, SMA proximity)
                meets_criteria = (
                    rsi is not None and rsi > 0 and
                    rsi > rsi_min and
                    rsi < rsi_threshold and
                    trend == "UP" and
                    (price_vs_sma_pct is None or price_vs_sma_pct <= sma_proximity_pct)
                )
                
                # Log if scoring failed but include analysis anyway
                if ai_score is None:
                    logger.debug(f"[WS] [{idx}/{total}] {symbol} - Analysis complete but AI scoring unavailable (included for retry)")
                elif ai_score == 0:
                    logger.debug(f"[WS] [{idx}/{total}] {symbol} - Score is 0/10 but including analysis for review/retry")
                
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
                logger.debug(f"[WS] [{idx}/{total}] {symbol} analysis complete, added to results")
                
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
                    logger.debug(f"[WS] [{idx}/{total}] Sent stock_complete for {symbol}")
                except Exception as e:
                    logger.error(f"[WS] Failed to send stock_complete for {symbol}: {e}", exc_info=True)
                
            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "rate limit" in error_msg.lower() or "rate limit exceeded" in error_msg.lower()
                
                logger.error(f"[WS] [{idx}/{total}] Exception analyzing {symbol}: {e}", exc_info=True)
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
        logger.info(f"[WS] Analysis complete! Processed {success_count}/{total} stocks successfully, {failed_count} failed")
        logger.info(f"[WS] Scanned {total} stocks | {success_count} with complete analysis (including those without scores for retry)")
        
        if failed_stocks:
            logger.warning(f"[WS] Failed stocks: {', '.join([f['symbol'] for f in failed_stocks])}")
            for failure in failed_stocks:
                logger.warning(f"   - {failure['symbol']}: {failure['reason']}")
        
        # Count stocks with scores vs without scores
        stocks_with_scores = sum(1 for s in stocks_with_analysis if s.get('ai_score') is not None)
        stocks_without_scores = success_count - stocks_with_scores
        if stocks_without_scores > 0:
            logger.info(f"[WS] {stocks_without_scores} stocks included without scores (available for retry)")
        
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
            logger.info("[WS] Sent completion message")
            
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
                logger.info(f"[WS] Saved daily scan: {len(stocks_with_analysis)} stocks, duration: {scan_duration:.2f}s")
            else:
                logger.warning("[WS] Failed to save daily scan")
        except Exception as e:
            logger.error(f"[WS] Failed to send completion message: {e}", exc_info=True)
        
    except WebSocketDisconnect:
        logger.info("[WS] WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"[WS] Fatal WebSocket error: {e}", exc_info=True)
        logger.error(f"   Error type: {type(e).__name__}")
        try:
            await _safe_send_json(websocket, {
                "type": "error",
                "message": f"Analysis failed: {str(e)[:100]}"
            })
        except:
            logger.error("[WS] Failed to send error message to client")

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
            # Fetch fresh data with article content for AI analysis
            bars, headlines, news_objects = await get_market_data(symbol, days=500, db=db, fetch_firecrawl_content=True)
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
            
            from ..core.config import get_strategy_config
            strategy_config = await get_strategy_config(db)
            strategy_prompt = get_balanced_low_prompt(strategy_config)
            verdict = _ai_engine.analyze(
                ticker=symbol,
                techs=techs,
                headlines="\n".join(headlines),
                strategy_prompt=strategy_prompt,
                strategy_config=strategy_config
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
        
        # Candidate selection uses ONLY technical thresholds - NO AI score
        rsi_threshold = config.get('rsi_threshold', 35)
        rsi_min = config.get('rsi_min', 20)
        sma_proximity_pct = config.get('sma_proximity_pct', 3.0)
        
        # Calculate price vs SMA percentage
        price_vs_sma_pct = None
        if price_value and sma_value and sma_value > 0:
            price_vs_sma_pct = ((price_value - sma_value) / sma_value) * 100
        
        # Candidate criteria: ONLY technical thresholds (RSI, Trend, SMA proximity)
        meets_criteria = (
            rsi_value is not None and rsi_value > 0 and
            rsi_value > rsi_min and
            rsi_value < rsi_threshold and
            trend_value == "UP" and
            (price_vs_sma_pct is None or price_vs_sma_pct <= sma_proximity_pct)
        )
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


# ============================================================================
# Custom Strategy Routes
# ============================================================================

async def get_strategies(db: Any = Depends(get_scoped_db)) -> JSONResponse:
    """Get all custom strategies.
    
    Returns:
        JSONResponse with list of all custom strategies
    """
    try:
        strategies = []
        async for strategy in db.custom_strategies.find({}).sort("created_at", -1):
            strategies.append({
                "label": strategy.get("label"),
                "name": strategy.get("name"),
                "description": strategy.get("description", ""),
                "is_active": strategy.get("is_active", False),
                "conditions": strategy.get("conditions", []),
                "created_at": strategy.get("created_at").isoformat() if strategy.get("created_at") else None,
                "updated_at": strategy.get("updated_at").isoformat() if strategy.get("updated_at") else None,
            })
        
        return JSONResponse(content={"strategies": strategies})
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}", exc_info=True)
        return error_response_json("Failed to get strategies", str(e))


async def get_strategy(label: str, db: Any = Depends(get_scoped_db)) -> JSONResponse:
    """Get a specific custom strategy by label.
    
    Args:
        label: Strategy label identifier
        
    Returns:
        JSONResponse with strategy details
    """
    try:
        strategy = await db.custom_strategies.find_one({"label": label})
        if not strategy:
            return error_response_json("Strategy not found", f"No strategy found with label: {label}", status_code=404)
        
        return JSONResponse(content={
            "label": strategy.get("label"),
            "name": strategy.get("name"),
            "description": strategy.get("description", ""),
            "is_active": strategy.get("is_active", False),
            "conditions": strategy.get("conditions", []),
            "created_at": strategy.get("created_at").isoformat() if strategy.get("created_at") else None,
            "updated_at": strategy.get("updated_at").isoformat() if strategy.get("updated_at") else None,
        })
    except Exception as e:
        logger.error(f"Failed to get strategy {label}: {e}", exc_info=True)
        return error_response_json("Failed to get strategy", str(e))


async def create_strategy(
    name: str = Form(...),
    description: str = Form(""),
    conditions: str = Form(...),  # JSON string
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Create a new custom strategy.
    
    Args:
        name: Strategy name
        description: Optional strategy description
        conditions: JSON string of conditions array
        
    Returns:
        HTMLResponse that closes modal and refreshes selector
    """
    try:
        import json
        conditions_list = json.loads(conditions)
        
        if not isinstance(conditions_list, list) or len(conditions_list) == 0:
            return error_response_json("Invalid conditions", "Conditions must be a non-empty array")
        
        # Generate unique label from name
        label_base = name.lower().replace(" ", "_").replace("-", "_")
        label_base = "".join(c for c in label_base if c.isalnum() or c == "_")
        
        # Ensure uniqueness
        label = label_base
        counter = 1
        while await db.custom_strategies.find_one({"label": label}):
            label = f"{label_base}_{counter}"
            counter += 1
        
        strategy_doc = {
            "label": label,
            "name": name,
            "description": description,
            "conditions": conditions_list,
            "is_active": False,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        
        await db.custom_strategies.insert_one(strategy_doc)
        
        logger.info(f"Created custom strategy: {label} ({name})")
        
        # Auto-activate the newly created strategy
        # Deactivate all other strategies first
        await db.custom_strategies.update_many(
            {"is_active": True},
            {"$set": {"is_active": False}}
        )
        
        # Activate the new strategy
        await db.custom_strategies.update_one(
            {"label": label},
            {"$set": {"is_active": True, "updated_at": datetime.now()}}
        )
        
        logger.info(f"Auto-activated newly created strategy: {label}")
        
        # Get updated strategies for selector refresh
        strategies = []
        async for strategy in db.custom_strategies.find({}).sort("created_at", -1):
            strategies.append({
                "label": strategy.get("label"),
                "name": strategy.get("name"),
                "description": strategy.get("description", ""),
                "is_active": strategy.get("is_active", False),
            })
        active_strategy = next((s for s in strategies if s["is_active"]), None)
        
        # Use HTMX best practices: hx-swap-oob for multi-element updates
        from ..api.templates import htmx_response, toast_notification
        
        # Build success message content
        success_content = templates.get_template("partials/strategy_created_success.html").render(
            strategy_name=name
        )
        
        # Refresh strategy selector
        selector_html = templates.get_template("components/strategy_selector.html").render(
            strategies=strategies,
            active_strategy=active_strategy
        )
        
        # Create toast notification
        toast_html = toast_notification(
            message=f'Strategy "{name}" created and activated successfully',
            type="success",
            duration=4000
        )
        
        # Use htmx_response helper for multi-element updates
        response_content = htmx_response({
            "modal-container": success_content,
            "strategy-selector-container": selector_html,
            "toast-container": toast_html
        })
        
        # Return with HX-Trigger header to refresh watchlist
        response = HTMLResponse(content=response_content, status_code=200)
        response.headers["HX-Trigger"] = "strategyActivated,refreshWatchlist"
        return response
    except json.JSONDecodeError as e:
        return error_response_json("Invalid JSON", f"Failed to parse conditions JSON: {e}")
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}", exc_info=True)
        return error_response_json("Failed to create strategy", str(e))


async def update_strategy(
    label: str,
    name: str = Form(None),
    description: str = Form(None),
    conditions: str = Form(None),  # JSON string
    db: Any = Depends(get_scoped_db)
) -> JSONResponse:
    """Update an existing custom strategy.
    
    Args:
        label: Strategy label identifier
        name: Optional new name
        description: Optional new description
        conditions: Optional JSON string of conditions array
        
    Returns:
        JSONResponse with updated strategy
    """
    try:
        strategy = await db.custom_strategies.find_one({"label": label})
        if not strategy:
            return error_response_json("Strategy not found", f"No strategy found with label: {label}", status_code=404)
        
        update_doc = {"updated_at": datetime.now()}
        
        if name is not None:
            update_doc["name"] = name
        if description is not None:
            update_doc["description"] = description
        if conditions is not None:
            import json
            conditions_list = json.loads(conditions)
            if not isinstance(conditions_list, list) or len(conditions_list) == 0:
                return error_response_json("Invalid conditions", "Conditions must be a non-empty array")
            update_doc["conditions"] = conditions_list
        
        await db.custom_strategies.update_one(
            {"label": label},
            {"$set": update_doc}
        )
        
        logger.info(f"Updated custom strategy: {label}")
        
        updated_strategy = await db.custom_strategies.find_one({"label": label})
        return JSONResponse(content={
            "label": updated_strategy.get("label"),
            "name": updated_strategy.get("name"),
            "description": updated_strategy.get("description", ""),
            "conditions": updated_strategy.get("conditions", []),
            "is_active": updated_strategy.get("is_active", False),
            "message": "Strategy updated successfully"
        })
    except json.JSONDecodeError as e:
        return error_response_json("Invalid JSON", f"Failed to parse conditions JSON: {e}")
    except Exception as e:
        logger.error(f"Failed to update strategy {label}: {e}", exc_info=True)
        return error_response_json("Failed to update strategy", str(e))


async def delete_strategy(label: str, db: Any = Depends(get_scoped_db)) -> JSONResponse:
    """Delete a custom strategy.
    
    Args:
        label: Strategy label identifier
        
    Returns:
        JSONResponse with success message
    """
    try:
        strategy = await db.custom_strategies.find_one({"label": label})
        if not strategy:
            return error_response_json("Strategy not found", f"No strategy found with label: {label}", status_code=404)
        
        # Don't allow deleting active strategy
        if strategy.get("is_active", False):
            return error_response_json("Cannot delete active strategy", "Please activate another strategy first")
        
        await db.custom_strategies.delete_one({"label": label})
        
        logger.info(f"Deleted custom strategy: {label}")
        
        return JSONResponse(content={"message": "Strategy deleted successfully"})
    except Exception as e:
        logger.error(f"Failed to delete strategy {label}: {e}", exc_info=True)
        return error_response_json("Failed to delete strategy", str(e))


async def activate_strategy(label: str, db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Set a custom strategy as active, or activate default by deactivating all.
    
    Args:
        label: Strategy label identifier, or 'default' to deactivate all custom strategies
        
    Returns:
        HTMLResponse with hx-swap-oob to update strategy selector and watchlist
    """
    try:
        # Handle "default" activation - deactivate all custom strategies
        if label == "default":
            await db.custom_strategies.update_many(
                {"is_active": True},
                {"$set": {"is_active": False, "updated_at": datetime.now()}}
            )
            logger.info("Activated default strategy (deactivated all custom strategies)")
        else:
            # Handle custom strategy activation
            strategy = await db.custom_strategies.find_one({"label": label})
            if not strategy:
                return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
                    message=f"Strategy not found: {label}"
                ), status_code=404)
            
            # Deactivate all other strategies
            await db.custom_strategies.update_many(
                {"is_active": True},
                {"$set": {"is_active": False}}
            )
            
            # Activate this strategy
            await db.custom_strategies.update_one(
                {"label": label},
                {"$set": {"is_active": True, "updated_at": datetime.now()}}
            )
            
            logger.info(f"Activated custom strategy: {label}")
        
        # Get updated strategies for the selector
        strategies = []
        async for strategy in db.custom_strategies.find({}).sort("created_at", -1):
            strategies.append({
                "label": strategy.get("label"),
                "name": strategy.get("name"),
                "description": strategy.get("description", ""),
                "is_active": strategy.get("is_active", False),
            })
        active_strategy = next((s for s in strategies if s["is_active"]), None)
        
        # Return updated strategy selector HTML
        # The hx-target on the button will swap this into #strategy-selector-container
        selector_html = templates.get_template("components/strategy_selector.html").render(
            strategies=strategies,
            active_strategy=active_strategy
        )
        
        return HTMLResponse(content=selector_html)
    except Exception as e:
        logger.error(f"Failed to activate strategy {label}: {e}", exc_info=True)
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"Failed to activate strategy: {str(e)[:200]}"
        ), status_code=500)


async def get_strategy_builder(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get the strategy builder page.
    
    Returns:
        HTMLResponse with strategy builder UI wrapped in modal
    """
    try:
        available_metrics = get_available_metrics()
        
        # Pass metrics directly to template - Alpine.js will handle JSON serialization
        builder_content = templates.get_template("pages/strategy_builder.html").render(
            available_metrics=available_metrics
        )
        
        # Wrap in modal
        modal_html = htmx_modal_wrapper(
            modal_id="strategy-builder-modal",
            title="Create Custom Strategy",
            content=builder_content,
            size="xl",
            icon="sliders-horizontal",
            icon_color="text-purple-400"
        )
        
        return HTMLResponse(content=modal_html)
    except Exception as e:
        logger.error(f"Failed to get strategy builder: {e}", exc_info=True)
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"Failed to load strategy builder: {str(e)[:200]}"
        ), status_code=500)


async def get_strategy_selector(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get the strategy selector button component.
    
    Returns:
        HTMLResponse with strategy selector button component
    """
    try:
        # Get current strategies to determine active strategy for display
        strategies = []
        async for strategy in db.custom_strategies.find({}).sort("created_at", -1):
            strategies.append({
                "label": strategy.get("label"),
                "name": strategy.get("name"),
                "description": strategy.get("description", ""),
                "is_active": strategy.get("is_active", False),
            })
        
        active_strategy = next((s for s in strategies if s["is_active"]), None)
        
        return HTMLResponse(content=templates.get_template("components/strategy_selector.html").render(
            strategies=strategies,
            active_strategy=active_strategy
        ))
    except Exception as e:
        logger.error(f"Failed to get strategy selector: {e}", exc_info=True)
        return HTMLResponse(content="<div>Strategy selector unavailable</div>", status_code=500)


async def get_strategy_selector_modal(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get the strategy selector modal (select or create).
    
    Returns:
        HTMLResponse with strategy selector modal, combining selection and creation
    """
    try:
        # Get current strategies
        strategies = []
        async for strategy in db.custom_strategies.find({}).sort("created_at", -1):
            strategies.append({
                "label": strategy.get("label"),
                "name": strategy.get("name"),
                "description": strategy.get("description", ""),
                "is_active": strategy.get("is_active", False),
            })
        
        active_strategy = next((s for s in strategies if s["is_active"]), None)
        
        modal_content = templates.get_template("pages/strategy_selector_modal.html").render(
            strategies=strategies,
            active_strategy=active_strategy
        )
        
        # Wrap in modal
        modal_html = htmx_modal_wrapper(
            modal_id="strategy-selector-modal",
            title="Select or Create Strategy",
            content=modal_content,
            size="large",
            icon="sliders-horizontal",
            icon_color="text-purple-400"
        )
        
        return HTMLResponse(content=modal_html)
    except Exception as e:
        logger.error(f"Failed to get strategy selector modal: {e}", exc_info=True)
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"Failed to load strategy selector: {str(e)[:200]}"
        ), status_code=500)


async def get_strategy_list(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get the strategy management page.
    
    Returns:
        HTMLResponse with strategy list UI wrapped in modal
    """
    try:
        strategies = []
        async for strategy in db.custom_strategies.find({}).sort("created_at", -1):
            strategies.append({
                "label": strategy.get("label"),
                "name": strategy.get("name"),
                "description": strategy.get("description", ""),
                "is_active": strategy.get("is_active", False),
                "conditions": strategy.get("conditions", []),
                "created_at": strategy.get("created_at"),
                "updated_at": strategy.get("updated_at"),
            })
        
        list_content = templates.get_template("pages/strategy_list.html").render(
            strategies=strategies
        )
        
        # Wrap in modal
        modal_html = htmx_modal_wrapper(
            modal_id="strategy-list-modal",
            title="Manage Strategies",
            content=list_content,
            size="large",
            icon="list",
            icon_color="text-purple-400"
        )
        
        return HTMLResponse(content=modal_html)
    except Exception as e:
        logger.error(f"Failed to get strategy list: {e}", exc_info=True)
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"Failed to load strategy list: {str(e)[:200]}"
        ), status_code=500)


async def get_signal_hunter_section(
    request: Request,
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Get Signal Hunter section with preset selector and results.
    
    Args:
        request: FastAPI request (may contain query param 'preset_name')
        db: MongoDB database instance
        
    Returns:
        HTMLResponse with Signal Hunter section HTML
    """
    try:
        from ..services.signal_hunter import FILTER_PRESETS, DEFAULT_PRESET, get_screener_results
        
        # Validate presets are available
        if not FILTER_PRESETS or not DEFAULT_PRESET or DEFAULT_PRESET not in FILTER_PRESETS:
            logger.error(f"[SIGNAL_HUNTER] Invalid FILTER_PRESETS or DEFAULT_PRESET configuration")
            return HTMLResponse(
                content="<div class='text-red-400 p-4'>Signal Hunter configuration error. Please check server logs.</div>",
                status_code=500
            )
        
        # Get preset from query param
        try:
            preset_name = request.query_params.get('preset_name', None)
        except AttributeError:
            preset_name = None
        
        preset = preset_name if preset_name and preset_name in FILTER_PRESETS else DEFAULT_PRESET
        filters_dict = FILTER_PRESETS[preset].copy()
        
        # Get current strategy for candidate evaluation
        custom_strategy = await get_active_custom_strategy(db)
        if custom_strategy:
            strategy_config = custom_strategy
        else:
            strategy_config = await get_strategy_config_dict(db)
        
        # Fetch screener results
        screener_results = []
        try:
            screener_results = get_screener_results(
                filters_dict=filters_dict,
                preset_name=preset,
                limit=20  # Limit to 20 for inline display
            )
        except Exception as e:
            logger.error(f"[SIGNAL_HUNTER] Error fetching results: {e}", exc_info=True)
        
        # Evaluate results against strategy in parallel (limit to 20 for performance)
        import asyncio
        evaluated_results = []
        use_custom_strategy = custom_strategy is not None
        
        async def evaluate_stock(stock_data: dict) -> dict:
            """Evaluate a single stock against the strategy."""
            symbol = stock_data.get('symbol', '').upper().strip()
            if not symbol:
                return None
            
            try:
                df, _, _ = await get_market_data(symbol, days=100, db=db, fetch_firecrawl_content=False)
                
                if df is None or df.empty or len(df) < 20:
                    return {
                        **stock_data,
                        'is_candidate': False,
                        'rejection_reasons': ['Insufficient market data'],
                        'rsi': None,
                        'price': None,
                        'trend': None,
                        'price_vs_sma_200_pct': None,
                        'sma_200': None
                    }
                
                try:
                    stats = analyze_comprehensive_stats(df)
                except Exception as stats_error:
                    logger.warning(f"[SIGNAL_HUNTER] Failed to analyze stats for {symbol}: {stats_error}")
                    return {
                        **stock_data,
                        'is_candidate': False,
                        'rejection_reasons': [f'Stats analysis failed: {str(stats_error)[:50]}'],
                        'rsi': None,
                        'price': None,
                        'trend': None,
                        'price_vs_sma_200_pct': None,
                        'sma_200': None
                    }
                
                if use_custom_strategy:
                    # Calculate price vs SMA-200 % for custom strategy
                    price = stats.get('price', 0) if stats else 0
                    sma_200 = stats.get('sma_200') if stats else None
                    price_vs_sma_pct = None
                    if price and sma_200:
                        try:
                            price_vs_sma_pct = ((price - sma_200) / sma_200) * 100
                        except (TypeError, ZeroDivisionError):
                            price_vs_sma_pct = None
                    try:
                        meets_criteria, rejection_reasons = evaluate_strategy(stats, custom_strategy)
                    except Exception as eval_error:
                        logger.warning(f"[SIGNAL_HUNTER] Failed to evaluate strategy for {symbol}: {eval_error}")
                        meets_criteria = False
                        rejection_reasons = [f'Strategy evaluation failed: {str(eval_error)[:50]}']
                else:
                    # Use full strategy evaluation logic (same as watchlist)
                    from ..services.analysis import analyze_technicals
                    
                    # Get strategy config values
                    rsi_threshold = strategy_config.get('rsi_threshold', 35)
                    rsi_min = strategy_config.get('rsi_min', 20)
                    sma_proximity_pct = strategy_config.get('sma_proximity_pct', 3.0)
                    earnings_days_min = strategy_config.get('earnings_days_min', 3)
                    pe_ratio_max = strategy_config.get('pe_ratio_max', 50.0)
                    market_cap_min = strategy_config.get('market_cap_min', 300_000_000)
                    
                    # Use analyze_technicals for default strategy (matches watchlist logic)
                    techs = analyze_technicals(df)
                    rsi = techs.get('rsi')
                    trend = techs.get('trend', 'NEUTRAL')
                    current_price = techs.get('price')
                    sma_200 = techs.get('sma')
                    earnings_in_days = techs.get('earnings_in_days')
                    pe_ratio = techs.get('pe_ratio')
                    market_cap = techs.get('market_cap')
                    
                    # Recalculate price vs SMA percentage
                    if current_price and sma_200:
                        price_vs_sma_pct = round(((current_price - sma_200) / sma_200) * 100, 1)
                    else:
                        price_vs_sma_pct = None
                    
                    # Check filters (earnings, P/E, market cap)
                    passes_filters = True
                    if earnings_in_days is not None and earnings_in_days < earnings_days_min:
                        passes_filters = False
                    if pe_ratio and pe_ratio > pe_ratio_max:
                        passes_filters = False
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
                    
                    # Build rejection reasons if not a candidate
                    rejection_reasons = []
                    if not meets_criteria:
                        if not passes_filters:
                            if earnings_in_days is not None and earnings_in_days < earnings_days_min:
                                rejection_reasons.append(f"Earnings in {earnings_in_days} day(s)")
                            if pe_ratio and pe_ratio > pe_ratio_max:
                                rejection_reasons.append(f"P/E {pe_ratio:.1f} > {pe_ratio_max:.0f}")
                            if market_cap and market_cap < market_cap_min:
                                rejection_reasons.append(f"Market cap < ${market_cap_min/1_000_000:.0f}M")
                        if rsi is not None:
                            if rsi <= rsi_min:
                                rejection_reasons.append(f"RSI {rsi:.1f} ≤ {rsi_min} (too extreme oversold)")
                            elif rsi >= rsi_threshold:
                                rejection_reasons.append(f"RSI {rsi:.1f} ≥ {rsi_threshold} (not oversold)")
                        if trend != "UP":
                            if trend == "DOWN":
                                rejection_reasons.append(f"Downtrend (Price ${current_price:.2f} < SMA-200 ${sma_200:.2f if sma_200 else 'N/A'})")
                            else:
                                rejection_reasons.append("Trend unclear")
                        if price_vs_sma_pct is not None and price_vs_sma_pct > sma_proximity_pct:
                            rejection_reasons.append(f"Price {price_vs_sma_pct:.1f}% above SMA-200 (exceeds {sma_proximity_pct}% proximity limit)")
                        if not rejection_reasons:
                            rejection_reasons.append("Does not meet entry criteria")
                
                # Prepare return values based on strategy type
                if use_custom_strategy:
                    return_rsi = stats.get('rsi') if stats else None
                    return_price = stats.get('price') if stats else None
                    return_trend = stats.get('trend') if stats else None
                    return_sma_200 = sma_200
                else:
                    return_rsi = rsi
                    return_price = current_price
                    return_trend = trend
                    return_sma_200 = sma_200
                
                return {
                    **stock_data,
                    'is_candidate': meets_criteria,
                    'rejection_reasons': rejection_reasons,
                    'rsi': return_rsi,
                    'price': return_price,
                    'trend': return_trend,
                    'price_vs_sma_200_pct': price_vs_sma_pct,
                    'sma_200': return_sma_200
                }
            except Exception as e:
                logger.debug(f"[SIGNAL_HUNTER] Error evaluating {symbol}: {e}")
                return {
                    **stock_data,
                    'is_candidate': False,
                    'rejection_reasons': [f'Error: {str(e)[:50]}'],
                    'rsi': None,
                    'price': None,
                    'trend': None,
                    'price_vs_sma_200_pct': None,
                    'sma_200': None
                }
        
        # Process stocks in parallel (limit concurrency to avoid overwhelming the API)
        stock_tasks = [evaluate_stock(stock_data) for stock_data in screener_results[:20]]
        results = await asyncio.gather(*stock_tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        for result in results:
            if result is None:
                continue
            if isinstance(result, Exception):
                logger.error(f"[SIGNAL_HUNTER] Exception in parallel evaluation: {result}", exc_info=True)
                continue
            evaluated_results.append(result)
        
        # Sort results: candidates first, then non-candidates
        evaluated_results.sort(key=lambda x: (not x.get('is_candidate', False), x.get('symbol', '')))
        
        # Get current watchlist symbols to check if stocks are already added
        watchlist_symbols = set()
        try:
            watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
            if watch_list_settings and watch_list_settings.get("value"):
                symbols_str = watch_list_settings.get("value", "")
                watchlist_symbols = {s.strip().upper() for s in symbols_str.split(',') if s.strip()}
        except Exception as e:
            logger.debug(f"Could not get watchlist for signal hunter: {e}")
        
        # Render section
        try:
            section_html = templates.get_template("pages/signal_hunter_section.html").render(
                presets=FILTER_PRESETS,
                current_preset=preset,
                results=evaluated_results,
                strategy_name=strategy_config.get('name', 'Default') if strategy_config else 'Default',
                watchlist_symbols=watchlist_symbols
            )
        except Exception as template_error:
            logger.error(f"[SIGNAL_HUNTER] Template rendering error: {template_error}", exc_info=True)
            return HTMLResponse(
                content=f"<div class='text-red-400 p-4'>Template error: {str(template_error)[:200]}</div>",
                status_code=500
            )
        
        return HTMLResponse(content=section_html)
    except Exception as e:
        logger.error(f"Failed to get Signal Hunter section: {e}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"[SIGNAL_HUNTER] Full traceback: {error_trace}")
        return HTMLResponse(
            content=f"<div class='text-red-400 p-4'>Error loading Signal Hunter: {str(e)[:200]}</div>",
            status_code=500
        )


async def get_signal_hunter_modal(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get the Signal Hunter modal with preset filter selector.
    
    Returns:
        HTMLResponse with Signal Hunter modal
    """
    try:
        # Get current strategy for candidate evaluation display
        custom_strategy = await get_active_custom_strategy(db)
        if custom_strategy:
            strategy_config = custom_strategy
        else:
            strategy_config = await get_strategy_config_dict(db)
        
        # Auto-load default preset filters
        default_filters = FILTER_PRESETS[DEFAULT_PRESET]
        
        # Get current watchlist symbols to check if stocks are already added
        watchlist_symbols = set()
        try:
            watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
            if watch_list_settings and watch_list_settings.get("value"):
                symbols_str = watch_list_settings.get("value", "")
                watchlist_symbols = {s.strip().upper() for s in symbols_str.split(',') if s.strip()}
        except Exception as e:
            logger.debug(f"Could not get watchlist for signal hunter modal: {e}")
        
        modal_content = templates.get_template("pages/signal_hunter_modal.html").render(
            results=None,
            loading=False,
            presets=FILTER_PRESETS,
            default_preset=DEFAULT_PRESET,
            current_filters=default_filters,
            strategy_name=strategy_config.get('name', 'Default'),
            watchlist_symbols=watchlist_symbols
        )
        
        # Wrap in modal
        modal_html = htmx_modal_wrapper(
            modal_id="signal-hunter-modal",
            title="Signal Hunter",
            content=modal_content,
            size="xl",
            icon="radar",
            icon_color="text-purple-400"
        )
        
        return HTMLResponse(content=modal_html)
    except Exception as e:
        logger.error(f"Failed to get Signal Hunter modal: {e}", exc_info=True)
        return HTMLResponse(content=templates.get_template("partials/error_message.html").render(
            message=f"Failed to load Signal Hunter: {str(e)[:200]}"
        ), status_code=500)


async def search_signal_hunter(
    request: Request,
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Search for stocks using Signal Hunter with Finviz filters, then evaluate against strategy.
    
    Args:
        request: FastAPI request (contains form data with Finviz filters)
        db: MongoDB database instance
        
    Returns:
        HTMLResponse with screener results and candidate evaluation
    """
    try:
        # Parse preset and filters from form data
        form_data = await request.form()
        preset_name = form_data.get("preset_name", DEFAULT_PRESET)
        filters_json = form_data.get("filters_json", "")
        
        # Parse filters from JSON or use preset
        filters_dict = None
        if filters_json:
            import json
            try:
                filters_dict = json.loads(filters_json)
                logger.info(f"[SIGNAL_HUNTER] Using filters from form: {filters_dict}")
            except json.JSONDecodeError:
                logger.warning(f"[SIGNAL_HUNTER] Failed to parse filters_json, using preset instead")
        
        if not filters_dict:
            if preset_name in FILTER_PRESETS:
                filters_dict = FILTER_PRESETS[preset_name].copy()
                logger.info(f"[SIGNAL_HUNTER] Using preset: {preset_name}")
            else:
                filters_dict = FILTER_PRESETS[DEFAULT_PRESET].copy()
                logger.info(f"[SIGNAL_HUNTER] Using default preset: {DEFAULT_PRESET}")
        
        if not filters_dict:
            custom_strategy = await get_active_custom_strategy(db)
            if custom_strategy:
                strategy_config = custom_strategy
            else:
                strategy_config = await get_strategy_config_dict(db)
            
            # Get current watchlist symbols
            watchlist_symbols = set()
            try:
                watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
                if watch_list_settings and watch_list_settings.get("value"):
                    symbols_str = watch_list_settings.get("value", "")
                    watchlist_symbols = {s.strip().upper() for s in symbols_str.split(',') if s.strip()}
            except Exception as e:
                logger.debug(f"Could not get watchlist for signal hunter modal: {e}")
            
            modal_content = templates.get_template("pages/signal_hunter_modal.html").render(
                results=None,
                loading=False,
                presets=FILTER_PRESETS,
                default_preset=DEFAULT_PRESET,
                current_filters=FILTER_PRESETS[DEFAULT_PRESET],
                strategy_name=strategy_config.get('name', 'Default'),
                watchlist_symbols=watchlist_symbols,
                error="Please select a preset"
            )
            modal_html = htmx_modal_wrapper(
                modal_id="signal-hunter-modal",
                title="Signal Hunter",
                content=modal_content,
                size="xl",
                icon="radar",
                icon_color="text-purple-400"
            )
            return HTMLResponse(content=modal_html)
        
        # Fetch screener results using filters
        # Note: This is slow (10-30 seconds) - HTMX loading indicator will show
        try:
            logger.info(f"[SIGNAL_HUNTER] Starting screener search (this may take 10-30 seconds)...")
            screener_results = get_screener_results(
                filters_dict=filters_dict,
                preset_name=preset_name,
                limit=100
            )
            logger.info(f"[SIGNAL_HUNTER] Screener search completed, found {len(screener_results)} stocks")
        except Exception as e:
            logger.error(f"[SIGNAL_HUNTER] Error in get_screener_results: {e}", exc_info=True)
            custom_strategy = await get_active_custom_strategy(db)
            if custom_strategy:
                strategy_config = custom_strategy
            else:
                strategy_config = await get_strategy_config_dict(db)
            
            # Get current watchlist symbols
            watchlist_symbols = set()
            try:
                watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
                if watch_list_settings and watch_list_settings.get("value"):
                    symbols_str = watch_list_settings.get("value", "")
                    watchlist_symbols = {s.strip().upper() for s in symbols_str.split(',') if s.strip()}
            except Exception as e2:
                logger.debug(f"Could not get watchlist for signal hunter modal: {e2}")
            
            modal_content = templates.get_template("pages/signal_hunter_modal.html").render(
                results=[],
                loading=False,
                presets=FILTER_PRESETS,
                default_preset=DEFAULT_PRESET,
                current_filters=filters_dict,
                strategy_name=strategy_config.get('name', 'Default'),
                watchlist_symbols=watchlist_symbols,
                error=f"Error fetching results: {str(e)}"
            )
            modal_html = htmx_modal_wrapper(
                modal_id="signal-hunter-modal",
                title="Signal Hunter",
                content=modal_content,
                size="xl",
                icon="radar",
                icon_color="text-purple-400"
            )
            return HTMLResponse(content=modal_html)
        
        # Get strategy config for evaluation (not for filtering)
        custom_strategy = await get_active_custom_strategy(db)
        use_custom_strategy = custom_strategy is not None
        
        if use_custom_strategy:
            strategy_config = custom_strategy
        else:
            strategy_config = await get_strategy_config_dict(db)
        
        if not screener_results:
            # Get current watchlist symbols
            watchlist_symbols = set()
            try:
                watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
                if watch_list_settings and watch_list_settings.get("value"):
                    symbols_str = watch_list_settings.get("value", "")
                    watchlist_symbols = {s.strip().upper() for s in symbols_str.split(',') if s.strip()}
            except Exception as e:
                logger.debug(f"Could not get watchlist for signal hunter modal: {e}")
            
            modal_content = templates.get_template("pages/signal_hunter_modal.html").render(
                results=[],
                loading=False,
                presets=FILTER_PRESETS,
                default_preset=DEFAULT_PRESET,
                current_filters=filters_dict,
                strategy_name=strategy_config.get('name', 'Default'),
                watchlist_symbols=watchlist_symbols,
                error="No stocks found matching your filters. Try a different preset."
            )
            modal_html = htmx_modal_wrapper(
                modal_id="signal-hunter-modal",
                title="Signal Hunter",
                content=modal_content,
                size="xl",
                icon="radar",
                icon_color="text-purple-400"
            )
            return HTMLResponse(content=modal_html)
        
        # Evaluate each stock against strategy
        evaluated_results = []
        
        for stock_data in screener_results[:50]:  # Limit to 50 for performance
            symbol = stock_data.get('symbol', '').upper().strip()
            if not symbol:
                continue
            
            try:
                # Fetch market data
                df, _, _ = await get_market_data(symbol, days=100, db=db, fetch_firecrawl_content=False)
                
                if df is None or df.empty or len(df) < 20:
                    evaluated_results.append({
                        **stock_data,
                        'is_candidate': False,
                        'rejection_reasons': ['Insufficient market data'],
                        'error': 'Insufficient data'
                    })
                    continue
                
                # Calculate comprehensive stats
                try:
                    stats = analyze_comprehensive_stats(df)
                except Exception as e:
                    logger.warning(f"[SIGNAL_HUNTER] Failed to calculate stats for {symbol}: {e}")
                    evaluated_results.append({
                        **stock_data,
                        'is_candidate': False,
                        'rejection_reasons': [f'Stats calculation failed: {str(e)[:50]}'],
                        'error': str(e)[:100]
                    })
                    continue
                
                # Evaluate against strategy
                if use_custom_strategy:
                    meets_criteria, rejection_reasons = evaluate_strategy(stats, custom_strategy)
                    bounce_strength = None
                else:
                    # Strategy-specific evaluation based on preset
                    rsi = stats.get('rsi')
                    sma_200 = stats.get('sma_200')
                    price = stats.get('price', 0)
                    trend = stats.get('trend') if stats else None
                    
                    if preset_name == "Buy Low":
                        # Buy Low: Oversold stocks ready to bounce
                        meets_criteria = (
                            rsi is not None and rsi < 40
                        )
                        
                        # Calculate bounce strength
                        bounce_strength = None
                        if rsi is not None:
                            if rsi < 25:
                                bounce_strength = "Very Strong"
                            elif rsi < 35:
                                bounce_strength = "Strong"
                            elif rsi < 40:
                                bounce_strength = "Moderate"
                        
                        rejection_reasons = []
                        if not meets_criteria:
                            if rsi is None:
                                rejection_reasons.append("RSI not available")
                            elif rsi >= 40:
                                rejection_reasons.append(f"RSI {rsi:.1f} not oversold (< 40)")
                    else:  # Catch Momentum
                        # Catch Momentum: Stocks with momentum (Month Up already filtered by Finviz)
                        # Trust Finviz filters - just verify RSI < 60 (already filtered by Finviz)
                        meets_criteria = (
                            rsi is not None and rsi < 60
                        )
                        
                        # Prefer stocks in uptrend or above SMA-200, but don't require it
                        # Finviz already filtered for "Month Up" performance
                        
                        bounce_strength = None
                        if rsi is not None:
                            if 40 <= rsi < 50:
                                bounce_strength = "Strong"
                            elif 50 <= rsi < 60:
                                bounce_strength = "Moderate"
                            elif rsi < 40:
                                bounce_strength = "Very Strong"
                        
                        rejection_reasons = []
                        if not meets_criteria:
                            if rsi is None:
                                rejection_reasons.append("RSI not available")
                            elif rsi >= 60:
                                rejection_reasons.append(f"RSI {rsi:.1f} overbought (>= 60)")
                
                # Add evaluation results
                evaluated_results.append({
                    **stock_data,
                    'is_candidate': meets_criteria,
                    'rejection_reasons': rejection_reasons,
                    'stats': stats,
                    'rsi': stats.get('rsi'),
                    'price': stats.get('price'),
                    'trend': stats.get('trend', 'UNKNOWN'),
                    'sma_200': stats.get('sma_200'),
                    'price_vs_sma_200_pct': stats.get('price_vs_sma_200_pct'),
                    'bounce_strength': bounce_strength,
                })
                
            except Exception as e:
                logger.error(f"[SIGNAL_HUNTER] Error evaluating {symbol}: {e}", exc_info=True)
                evaluated_results.append({
                    **stock_data,
                    'is_candidate': False,
                    'rejection_reasons': [f'Evaluation error: {str(e)[:50]}'],
                    'error': str(e)[:100]
                })
        
        # Get current watchlist symbols to check if stocks are already added
        watchlist_symbols = set()
        try:
            watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
            if watch_list_settings and watch_list_settings.get("value"):
                symbols_str = watch_list_settings.get("value", "")
                watchlist_symbols = {s.strip().upper() for s in symbols_str.split(',') if s.strip()}
        except Exception as e:
            logger.debug(f"Could not get watchlist for signal hunter modal: {e}")
        
        # Render results
        modal_content = templates.get_template("pages/signal_hunter_modal.html").render(
            results=evaluated_results,
            loading=False,
            presets=FILTER_PRESETS,
            default_preset=preset_name if preset_name in FILTER_PRESETS else DEFAULT_PRESET,
            current_filters=filters_dict,
            strategy_name=strategy_config.get('name', 'Default'),
            watchlist_symbols=watchlist_symbols
        )
        
        modal_html = htmx_modal_wrapper(
            modal_id="signal-hunter-modal",
            title="Signal Hunter",
            content=modal_content,
            size="xl",
            icon="radar",
            icon_color="text-purple-400"
        )
        
        return HTMLResponse(content=modal_html)
        
    except Exception as e:
        logger.error(f"[SIGNAL_HUNTER] Error in search: {e}", exc_info=True)
        # Get strategy config for error display
        try:
            custom_strategy = await get_active_custom_strategy(db)
            if custom_strategy:
                strategy_config = custom_strategy
            else:
                strategy_config = await get_strategy_config_dict(db)
        except:
            strategy_config = {"name": "Default", "rsi_min": 20, "rsi_threshold": 35}
        
        # Get current watchlist symbols
        watchlist_symbols = set()
        try:
            watch_list_settings = await db.app_settings.find_one({"key": "watch_list"})
            if watch_list_settings and watch_list_settings.get("value"):
                symbols_str = watch_list_settings.get("value", "")
                watchlist_symbols = {s.strip().upper() for s in symbols_str.split(',') if s.strip()}
        except Exception as e2:
            logger.debug(f"Could not get watchlist for signal hunter modal: {e2}")
        
        modal_content = templates.get_template("pages/signal_hunter_modal.html").render(
            results=None,
            loading=False,
            presets=FILTER_PRESETS,
            default_preset=DEFAULT_PRESET,
            current_filters=FILTER_PRESETS[DEFAULT_PRESET],
            strategy_name=strategy_config.get('name', 'Default'),
            watchlist_symbols=watchlist_symbols,
            error=f"Error: {str(e)[:200]}"
        )
        modal_html = htmx_modal_wrapper(
            modal_id="signal-hunter-modal",
            title="Signal Hunter",
            content=modal_content,
            size="xl",
            icon="radar",
            icon_color="text-purple-400"
        )
        return HTMLResponse(content=modal_html, status_code=500)


async def get_finviz_filters_api(
    type: str = Query("technical", description="Screener type: technical or overview"),
    db: Any = Depends(get_scoped_db)
) -> JSONResponse:
    """Get available Finviz filters and their options.
    
    Args:
        type: Screener type ("technical" or "overview")
        db: MongoDB database instance
        
    Returns:
        JSONResponse with filters and their options
    """
    try:
        filters = get_finviz_filters(type)
        return JSONResponse(content={
            "success": True,
            "screener_type": type,
            "filters": filters
        })
    except Exception as e:
        logger.error(f"Error getting Finviz filters: {e}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": str(e)[:200]},
            status_code=500
        )


# ============================================
# ALPACA ACCOUNT MANAGEMENT ROUTES
# ============================================

async def get_alpaca_accounts(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get list of all Alpaca accounts.
    
    Returns HTML for account management UI.
    """
    try:
        account_manager = AlpacaAccountManager(db)
        accounts = await account_manager.get_accounts()
        active_account = await account_manager.get_active_account()
        active_id = active_account.get('account_id') if active_account else None
        
        return HTMLResponse(content=templates.get_template("pages/alpaca_accounts.html").render(
            accounts=accounts,
            active_account_id=active_id
        ))
    except Exception as e:
        logger.error(f"Failed to get accounts: {e}", exc_info=True)
        return error_response_html(f"Failed to load accounts: {str(e)[:100]}")


async def create_alpaca_account(
    account_id: str = Form(...),
    nickname: str = Form(...),
    api_key: str = Form(...),
    api_secret: str = Form(...),
    base_url: str = Form("https://paper-api.alpaca.markets"),
    is_active: bool = Form(False),
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Create or update an Alpaca account.
    
    Args:
        account_id: Alpaca account identifier (e.g., "PA3AFGM5YBAO")
        nickname: User-friendly name
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        base_url: API base URL (default: paper trading)
        is_active: Whether to activate this account
    """
    try:
        account_manager = AlpacaAccountManager(db)
        account = await account_manager.create_account(
            account_id=account_id,
            nickname=nickname,
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
            is_active=is_active
        )
        
        if account:
            return htmx_response(
                content=toast_notification(
                    f"Account '{nickname}' saved successfully",
                    "success"
                ),
                hx_trigger={"accountUpdated": {}}
            )
        else:
            return error_response_html("Failed to create account. Check credentials.")
    except Exception as e:
        logger.error(f"Failed to create account: {e}", exc_info=True)
        return error_response_html(f"Error: {str(e)[:100]}")


async def set_active_alpaca_account(
    account_id: str = Form(...),
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Set an account as active.
    
    Args:
        account_id: Account identifier to activate
    """
    try:
        account_manager = AlpacaAccountManager(db)
        success = await account_manager.set_active_account(account_id)
        
        if success:
            return htmx_response(
                content=toast_notification(
                    "Account activated successfully",
                    "success"
                ),
                hx_trigger={"accountUpdated": {}, "balanceRefresh": {}}
            )
        else:
            return error_response_html("Failed to activate account")
    except Exception as e:
        logger.error(f"Failed to set active account: {e}", exc_info=True)
        return error_response_html(f"Error: {str(e)[:100]}")


async def delete_alpaca_account(
    account_id: str = Form(...),
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Delete an Alpaca account.
    
    Args:
        account_id: Account identifier to delete
    """
    try:
        account_manager = AlpacaAccountManager(db)
        success = await account_manager.delete_account(account_id)
        
        if success:
            return htmx_response(
                content=toast_notification(
                    "Account deleted successfully",
                    "success"
                ),
                hx_trigger={"accountUpdated": {}}
            )
        else:
            return error_response_html("Failed to delete account")
    except Exception as e:
        logger.error(f"Failed to delete account: {e}", exc_info=True)
        return error_response_html(f"Error: {str(e)[:100]}")


async def test_alpaca_account(
    account_id: str = Form(...),
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Test connection to an Alpaca account.
    
    Args:
        account_id: Account identifier to test
    """
    try:
        account_manager = AlpacaAccountManager(db)
        result = await account_manager.test_connection(account_id)
        
        if result.get('success'):
            equity = result.get('equity', 0)
            buying_power = result.get('buying_power', 0)
            status = result.get('status', 'N/A')
            equity_str = f"${equity:,.2f}"
            buying_power_str = f"${buying_power:,.2f}"
            return HTMLResponse(content=f"""
                <div class="glass rounded-lg p-3 border border-green-500/30 bg-green-500/10">
                    <div class="flex items-center gap-2 text-green-400 mb-2">
                        <i data-lucide="check-circle" class="w-4 h-4"></i>
                        <span class="font-semibold">Connection Successful</span>
                    </div>
                    <div class="text-xs text-muted space-y-1">
                        <div>Equity: <span class="text-white font-mono">{equity_str}</span></div>
                        <div>Buying Power: <span class="text-white font-mono">{buying_power_str}</span></div>
                        <div>Status: <span class="text-white">{status}</span></div>
                    </div>
                </div>
            """)
        else:
            return HTMLResponse(content=f"""
                <div class="glass rounded-lg p-3 border border-red-500/30 bg-red-500/10">
                    <div class="flex items-center gap-2 text-red-400 mb-1">
                        <i data-lucide="x-circle" class="w-4 h-4"></i>
                        <span class="font-semibold">Connection Failed</span>
                    </div>
                    <div class="text-xs text-red-300">{{ result.get('error', 'Unknown error') }}</div>
                </div>
            """)
    except Exception as e:
        logger.error(f"Failed to test account: {e}", exc_info=True)
        return HTMLResponse(content=f"""
            <div class="glass rounded-lg p-3 border border-red-500/30 bg-red-500/10">
                <div class="text-xs text-red-300">Error: {str(e)[:100]}</div>
            </div>
        """)

