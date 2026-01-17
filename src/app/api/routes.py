"""API route handlers for Balanced Low Buy System.

This module contains all FastAPI route handlers for the Sauron's Eye trading bot.
Routes are organized by functionality:
- Market scanning and analysis (analyze_symbol, discover_stocks)
- Position management (get_positions, quick_buy, quick_sell)
- Trade execution (execute_trade, panic_close)
- Strategy configuration (get_strategy_config, update_strategy_api)
- WebSocket streaming (get_trending_stocks_with_analysis_streaming)
- Trade history and logging (get_trade_logs)

All routes use mdb-engine's dependency injection pattern (get_scoped_db) for database access.
Routes return HTMLResponse for HTMX integration, or JSONResponse for API endpoints.
"""
from fastapi import Depends, Form, WebSocket, WebSocketDisconnect, Request, Body, Response
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime
from typing import Optional, Dict, Any
import json
from mdb_engine.observability import get_logger, set_correlation_id
from mdb_engine.dependencies import get_embedding_service
from mdb_engine.embeddings import EmbeddingService
from ..core.engine import get_scoped_db
from ..services.trading import place_order
from ..services.analysis import api
from ..services.analysis import get_market_data, analyze_technicals
from ..services.ai import EyeAI, _ai_engine
from ..services.radar import RadarService
from ..core.config import ALPACA_KEY, ALPACA_SECRET, STRATEGY_CONFIG, get_strategy_instance, get_strategy_from_db

logger = get_logger(__name__)

async def _safe_send_json(websocket, data: Dict[str, Any]) -> None:
    """Safely send JSON via WebSocket with automatic datetime sanitization.
    
    MDB-Engine Pattern: Ensures all WebSocket messages are JSON-serializable.
    This is a wrapper around websocket.send_json() that automatically sanitizes
    datetime objects and handles serialization errors gracefully.
    
    Args:
        websocket: WebSocket connection
        data: Data dictionary to send
    """
    try:
        # First attempt: sanitize and send
        sanitized_data = _sanitize_for_json(data)
        await websocket.send_json(sanitized_data)
    except (TypeError, ValueError) as e:
        # If sanitization didn't catch everything, try one more deep sanitization pass
        error_str = str(e).lower()
        if 'not json serializable' in error_str or 'datetime' in error_str or 'bool_' in error_str:
            logger.warning(f"First sanitization pass failed, attempting deep sanitization: {e}")
            try:
                # Deep sanitization: use json.dumps with default=str to convert any remaining non-serializable objects
                json_str = json.dumps(data, default=str, ensure_ascii=False)
                fallback_data = json.loads(json_str)
                await websocket.send_json(fallback_data)
            except Exception as e2:
                logger.error(f"Deep sanitization also failed: {e2}", exc_info=True)
                raise
        else:
            raise

def _convert_timestamp_to_iso(timestamp) -> Optional[str]:
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
    
    # Return as-is for other types (int, float, str, bool, etc.)
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
        return HTMLResponse(content="<span class='text-yellow-500'>Alpaca API not configured</span>")
    
    if api is None:
        return HTMLResponse(content="<span class='text-red-500'>Alpaca API not initialized</span>")
    
    try:
        acct = api.get_account()
        pl = float(acct.equity) - float(acct.last_equity)
        pl_color = "text-green-400" if pl >= 0 else "text-red-400"
        pl_icon = "fa-arrow-up" if pl >= 0 else "fa-arrow-down"
        return HTMLResponse(content=f"""
        <div class="text-3xl font-bold {pl_color} mb-1">${float(acct.equity):,.2f}</div>
        <div class="flex items-center gap-2 text-xs">
            <div class="flex items-center gap-1 {pl_color}">
                <i class="fas {pl_icon}"></i>
                <span>${abs(pl):,.2f}</span>
            </div>
            <span class="text-gray-500">•</span>
            <span class="text-gray-400">BP: ${float(acct.buying_power):,.2f}</span>
        </div>
        """)
    except Exception as e:
        logger.error(f"Failed to get balance: {e}", exc_info=True)
        error_msg = str(e)[:50] if str(e) else "Unknown error"
        return HTMLResponse(content=f"<span class='text-red-500'>API Error: {error_msg}</span>")

async def analyze_symbol(ticker: str = Form(...)) -> HTMLResponse:
    """Analyze a symbol using AI."""
    symbol = ticker.upper().strip()
    if not symbol:
        return HTMLResponse(content="Enter Symbol")
    
    # Always request plenty of data from the start
    bars, headlines, news_objects = get_market_data(symbol, days=500)  # Request 500 days minimum
    if bars is None or bars.empty:
        return HTMLResponse(content="""
            <div class="glass rounded-xl p-4 border border-red-500/30 fade-in">
                <div class="flex items-center gap-3 mb-2">
                    <i class="fas fa-exclamation-triangle text-red-500 text-xl"></i>
                    <span class="font-bold text-red-400">Data Error</span>
                </div>
                <p class="text-sm text-gray-300">Symbol not found or market closed. Check if the symbol is valid.</p>
            </div>
        """)
    
    if len(bars) < 14:
        # Log detailed info for debugging
        logger.error(f"🚨 Still only {len(bars)} days for {symbol} after requesting 500 days!")
        logger.error(f"   This suggests an API issue. Check logs for details.")
        
        return HTMLResponse(content=f"""
            <div class='glass rounded-xl p-4 border border-yellow-500/30 fade-in'>
                <div class="flex items-center gap-3 mb-3">
                    <i class="fas fa-exclamation-circle text-yellow-500 text-xl"></i>
                    <span class="font-bold text-yellow-400">Insufficient Data</span>
                </div>
                <div class='text-sm text-gray-300 mb-3'>Only {len(bars)} day(s) available for {symbol}.</div>
                <div class='text-xs text-gray-400 mb-2 font-semibold uppercase tracking-wider'>Possible reasons:</div>
                <ul class='text-xs text-gray-400 list-disc list-inside space-y-1 ml-4'>
                    <li>API rate limit or access issue</li>
                    <li>Symbol not available in paper trading</li>
                    <li>Market data subscription required</li>
                </ul>
                <div class='text-xs mt-3 text-gray-500 flex items-center gap-1'>
                    <i class="fas fa-info-circle"></i>
                    <span>Check application logs for detailed error information.</span>
                </div>
            </div>
        """)
    
    try:
        techs = analyze_technicals(bars)
    except ValueError as e:
        logger.error(f"Technical analysis failed for {symbol}: {e}")
        return HTMLResponse(content=f"<div class='text-red-500'>Analysis Error: {str(e)}</div>")
    except Exception as e:
        logger.error(f"Unexpected error analyzing {symbol}: {e}", exc_info=True)
        return HTMLResponse(content=f"<div class='text-red-500'>Unexpected Error: {str(e)[:100]}</div>")
    
    # Use EyeAI with current strategy
    ai_engine = _ai_engine
    if ai_engine:
        strategy = get_strategy_instance()
        strategy_prompt = strategy.get_ai_prompt()
        verdict = ai_engine.analyze(
            ticker=symbol,
            techs=techs,
            headlines="\n".join(headlines),
            strategy_prompt=strategy_prompt
        )
        color = "text-green-400" if verdict.score > 6 else "text-red-400"
        
        risk_badge = "badge-success" if verdict.risk_level.upper() == "LOW" else "badge-warning" if verdict.risk_level.upper() == "MEDIUM" else "badge-danger"
        action_color = "text-green-400" if verdict.action == "BUY" else "text-yellow-400" if verdict.action == "WAIT" else "text-red-400"
        
        return HTMLResponse(content=f"""
        <div class="glass-strong rounded-xl p-5 border border-red-500/20 fade-in">
            <div class="flex justify-between items-start mb-4">
                <div class="flex items-center gap-3">
                    <div class="w-14 h-14 rounded-xl bg-gradient-to-br from-red-500/20 to-orange-500/20 flex items-center justify-center">
                        <span class="font-bold text-white text-xl">{symbol}</span>
                    </div>
                    <div>
                        <div class="flex items-center gap-2 mb-1">
                            <span class="font-bold text-white text-xl">{symbol}</span>
                            <span class="badge {risk_badge}">{verdict.risk_level.upper()}</span>
                            <button onclick="openModal('risk-levels-modal')" class="text-gray-500 hover:text-yellow-400 transition cursor-pointer tooltip relative group ml-1" title="Learn about risk levels">
                                <i class="fas fa-info-circle text-xs"></i>
                                <div class="tooltip-text">Learn about risk levels</div>
                            </button>
                        </div>
                        <div class="text-xs text-gray-400">
                            <i class="fas fa-dollar-sign mr-1"></i>${techs['price']:.2f}
                        </div>
                    </div>
                </div>
                <div class="text-right">
                    <div class="flex items-center justify-end gap-2">
                        <div class="text-3xl font-bold {color}">{verdict.score}/10</div>
                        <button onclick="openModal('ai-score-modal')" class="text-gray-500 hover:text-purple-400 transition cursor-pointer tooltip relative group" title="Learn about AI scores">
                            <i class="fas fa-info-circle text-xs"></i>
                            <div class="tooltip-text">Learn how AI scores work</div>
                        </button>
                    </div>
                    <div class="text-xs text-gray-400">AI Score</div>
                </div>
            </div>
            
            <div class="grid grid-cols-3 gap-3 mb-4">
                <div class="glass rounded-lg p-3 text-center">
                    <div class="text-xs text-gray-400 mb-1 flex items-center justify-center gap-1">
                        <i class="fas fa-chart-line"></i>
                        <span>RSI</span>
                    </div>
                    <div class="text-lg font-bold text-white">{techs['rsi']:.1f}</div>
                    <div class="text-xs text-gray-500 mt-1">
                        {'Oversold' if techs['rsi'] < 30 else 'Overbought' if techs['rsi'] > 70 else 'Neutral'}
                    </div>
                </div>
                <div class="glass rounded-lg p-3 text-center">
                    <div class="text-xs text-gray-400 mb-1 flex items-center justify-center gap-1">
                        <i class="fas fa-arrow-trend-up"></i>
                        <span>Trend</span>
                    </div>
                    <div class="text-lg font-bold {'text-green-400' if techs['trend'] == 'UP' else 'text-red-400'}">{techs['trend']}</div>
                    <div class="text-xs text-gray-500 mt-1">vs SMA-200</div>
                </div>
                <div class="glass rounded-lg p-3 text-center">
                    <div class="text-xs text-gray-400 mb-1 flex items-center justify-center gap-1">
                        <i class="fas fa-wave-square"></i>
                        <span>ATR</span>
                    </div>
                    <div class="text-lg font-bold text-white">${techs['atr']:.2f}</div>
                    <div class="text-xs text-gray-500 mt-1">Volatility</div>
                </div>
            </div>
            
            <div class="glass rounded-lg p-4 mb-4 border-l-4 border-red-500/50">
                <div class="flex items-center justify-between mb-2">
                    <div class="text-xs text-gray-400 flex items-center gap-1">
                        <i class="fas fa-lightbulb"></i>
                        <span>Recommendation</span>
                    </div>
                    <span class="font-bold text-lg {action_color} uppercase">{verdict.action}</span>
                </div>
                <p class="text-sm text-gray-200 italic mb-3">"{verdict.reason}"</p>
                        <div class="text-xs text-gray-500 border-t border-gray-700/50 pt-2 mt-2">
                    <div class="flex items-center gap-4 flex-wrap">
                        <span class="flex items-center gap-1">
                            <i class="fas fa-info-circle"></i>
                            <span>Strategy: <strong>{STRATEGY_CONFIG['name']}</strong> - Balanced Low Buy System</span>
                        </span>
                        <span>RSI &lt; {STRATEGY_CONFIG['rsi_threshold']} (oversold)</span>
                        <span>Score ≥ {STRATEGY_CONFIG['ai_score_required']} (upside potential)</span>
                    </div>
                </div>
            </div>
            
            <script>
                new TradingView.widget({{
                    "autosize": true, "symbol": "{symbol}",
                    "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1",
                    "container_id": "tradingview_12345"
                }});
            </script>
        </div>
        """)
    else:
        return HTMLResponse(content="""
            <div class="glass rounded-xl p-4 border border-yellow-500/30 fade-in">
                <div class="flex items-center gap-3">
                    <i class="fas fa-eye-slash text-yellow-500 text-xl"></i>
                    <div>
                        <div class="font-bold text-yellow-400 mb-1">👁️ The Eye is Blind</div>
                        <p class="text-sm text-gray-300">Azure OpenAI configuration missing - The Eye cannot see</p>
                    </div>
                </div>
            </div>
        """)

async def backtest_symbol(ticker: str = Form("NVDA")) -> HTMLResponse:
    """Run backtest for a symbol."""
    if not check_alpaca_config():
        return HTMLResponse(content="<div class='text-yellow-500 p-2'>Alpaca API not configured</div>")
    
    if api is None:
        return HTMLResponse(content="<div class='text-red-500 p-2'>Alpaca API not initialized</div>")
    
    sym = ticker.upper()
    try:
        stats, trade_log = run_backtest_simulation(sym)
        
        if not stats:
            return HTMLResponse(content=f"<div class='text-red-500 p-2'>Backtest Failed: {trade_log}</div>")
    except Exception as e:
        logger.error(f"Backtest failed for {sym}: {e}", exc_info=True)
        return HTMLResponse(content=f"<div class='text-red-500 p-2'>Backtest Error: {str(e)[:100]}</div>")
    
    html = f"""
    <div class="fade-in">
        <div class="grid grid-cols-3 gap-3 mb-4">
            <div class="glass rounded-xl p-4 text-center border border-blue-500/20">
                <div class="text-xs text-gray-400 mb-2 uppercase tracking-wider flex items-center justify-center gap-1">
                    <i class="fas fa-exchange-alt"></i>
                    <span>Trades</span>
                </div>
                <div class="text-2xl font-bold text-white">{stats['trades']}</div>
                <div class="text-xs text-gray-500 mt-1">Total</div>
            </div>
            <div class="glass rounded-xl p-4 text-center border border-green-500/20">
                <div class="text-xs text-gray-400 mb-2 uppercase tracking-wider flex items-center justify-center gap-1">
                    <i class="fas fa-trophy"></i>
                    <span>Win Rate</span>
                </div>
                <div class="text-2xl font-bold {'text-green-400' if stats['win_rate'] > 50 else 'text-red-400'}">{stats['win_rate']}%</div>
                <div class="text-xs text-gray-500 mt-1">{'Excellent' if stats['win_rate'] > 60 else 'Good' if stats['win_rate'] > 50 else 'Needs Work'}</div>
            </div>
            <div class="glass rounded-xl p-4 text-center border border-yellow-500/20">
                <div class="text-xs text-gray-400 mb-2 uppercase tracking-wider flex items-center justify-center gap-1">
                    <i class="fas fa-chart-line"></i>
                    <span>Return</span>
                </div>
                <div class="text-2xl font-bold {'text-green-400' if stats['return'] > 0 else 'text-red-400'}">{stats['return']:+.1f}%</div>
                <div class="text-xs text-gray-500 mt-1">1 Year</div>
            </div>
        </div>
        
        <div class="glass rounded-xl p-3 mb-4 border border-gray-700/50">
            <div class="text-xs text-gray-400 mb-2 uppercase tracking-wider flex items-center gap-1">
                <i class="fas fa-chart-area"></i>
                <span>Equity Curve</span>
            </div>
            <img src="data:image/png;base64,{stats['plot']}" class="w-full rounded-lg" />
        </div>
        
        <div class="glass rounded-xl p-4 border border-gray-700/50">
            <div class="text-xs text-gray-400 mb-3 uppercase tracking-wider flex items-center gap-1">
                <i class="fas fa-list"></i>
                <span>Trade History</span>
            </div>
            <div class="max-h-40 overflow-y-auto">
                <table class="w-full text-xs">
                    <thead>
                        <tr class="text-gray-500 border-b border-gray-700/30">
                            <th class="px-2 py-2 text-left">Date</th>
                            <th class="px-2 py-2 text-left">Result</th>
                            <th class="px-2 py-2 text-right">P&L</th>
                        </tr>
                    </thead>
                    <tbody class="text-gray-300">
    """
    for t in trade_log:
        if t['type'] == 'SELL':
            color = "text-green-400" if t['pnl'] > 0 else "text-red-400"
            badge = "badge-success" if t['pnl'] > 0 else "badge-danger"
            html += f"""
                        <tr class="border-b border-gray-700/20 hover:bg-white/5">
                            <td class="px-2 py-2 font-mono">{t['date']}</td>
                            <td class="px-2 py-2">
                                <span class="badge {badge}">{t['result']}</span>
                            </td>
                            <td class="px-2 py-2 text-right font-bold {color}">${t['pnl']:.2f}</td>
                        </tr>
            """
            
    html += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    """
    return HTMLResponse(content=html)

async def execute_trade(ticker: str = Form(...), action: str = Form(...), db = Depends(get_scoped_db)) -> HTMLResponse:
    """Execute a manual trade."""
    if not check_alpaca_config():
        return HTMLResponse(content="<span class='text-yellow-500'>Alpaca API not configured</span>")
    
    if api is None:
        return HTMLResponse(content="<span class='text-red-500'>Alpaca API not initialized</span>")
    
    sym = ticker.upper()
    
    if action == 'buy':
        try:
            order = await place_order(sym, 'buy', "Manual Override", 10, db)
            if order:
                return HTMLResponse(content="""
                    <div class="glass rounded-lg p-3 border border-green-500/30 flex items-center gap-2 text-green-400">
                        <i class="fas fa-check-circle"></i>
                        <span class="font-semibold">Order Placed Successfully</span>
                    </div>
                """)
            return HTMLResponse(content="""
                <div class="glass rounded-lg p-3 border border-red-500/30 flex items-center gap-2 text-red-400">
                    <i class="fas fa-exclamation-circle"></i>
                    <span class="font-semibold">Order Failed - Check logs</span>
                </div>
            """)
        except Exception as e:
            logger.error(f"Failed to place order: {e}", exc_info=True)
            return HTMLResponse(content=f"<span class='text-red-500'>Error: {str(e)[:50]}</span>")
    else:
        try:
            api.close_position(sym)
            return HTMLResponse(content="""
                <div class="glass rounded-lg p-3 border border-yellow-500/30 flex items-center gap-2 text-yellow-400">
                    <i class="fas fa-check-circle"></i>
                    <span class="font-semibold">Position Closed</span>
                </div>
            """)
        except Exception as e:
            logger.error(f"Failed to close position: {e}", exc_info=True)
            error_msg = str(e)[:50] if str(e) else "No Position"
            return HTMLResponse(content=f"<span class='text-red-500'>{error_msg}</span>")

async def get_positions(db = Depends(get_scoped_db)) -> HTMLResponse:
    """Get current positions and pending orders.
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) for automatic connection management.
    Returns HTML with htmx buttons for closing positions (hx-post="/api/quick-sell")
    and canceling pending orders (hx-post="/api/cancel-order").
    This endpoint is polled every 5 seconds via hx-trigger="every 5s".
    """
    if not check_alpaca_config():
        return HTMLResponse(content="<div class='text-yellow-500 text-sm'>Alpaca API not configured</div>")
    
    if api is None:
        return HTMLResponse(content="<div class='text-red-500 text-sm'>Alpaca API not initialized</div>")
    
    try:
        # Get active positions
        pos = api.list_positions()
        
        # Get pending orders (new, pending_new, accepted, pending_replace, etc.)
        pending_orders = []
        try:
            all_orders = api.list_orders(status='open')
            # Filter for pending statuses (orders that haven't been filled yet)
            pending_statuses = ['new', 'pending_new', 'accepted', 'pending_replace', 'pending_cancel', 'partially_filled']
            pending_orders = [o for o in all_orders if o.status in pending_statuses]
            logger.debug(f"Found {len(pending_orders)} pending orders")
        except Exception as e:
            logger.warning(f"Could not fetch pending orders: {e}", exc_info=True)
        
        # If no positions and no pending orders
        if not pos and not pending_orders:
            return HTMLResponse(content="""
                <div class="text-center py-6 text-gray-500">
                    <i class="fas fa-wallet text-3xl mb-2 opacity-50"></i>
                    <p class="text-sm">Cash Gang 💰</p>
                    <p class="text-xs mt-1 text-gray-600">Waiting for the right low buy</p>
                </div>
            """)
        
        html = ""
        
        # Show pending orders first
        for order in pending_orders:
            symbol = order.symbol
            qty = int(order.qty)
            side = order.side  # 'buy' or 'sell'
            order_type = order.type  # 'limit', 'market', etc.
            limit_price = float(order.limit_price) if order.limit_price else None
            status = order.status
            order_id = order.id
            
            # Get order details from db.history if available
            trade_record = await db.history.find_one(
                {"symbol": symbol, "action": side},
                sort=[("timestamp", -1)]
            )
            
            stop_loss = trade_record.get("stop_loss") if trade_record else None
            take_profit = trade_record.get("take_profit") if trade_record else None
            
            # Status badge styling
            status_badge = "badge-warning"  # Yellow/orange for pending
            status_text = "PENDING"
            status_icon = "fa-clock"
            
            html += f"""
            <div class="glass rounded-lg p-3 mb-2 card-hover border border-yellow-500/40 bg-yellow-500/5">
                <div class="flex justify-between items-center mb-2">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 rounded-lg bg-yellow-500/20 flex items-center justify-center border border-yellow-500/40">
                            <span class="font-bold text-white text-sm">{symbol}</span>
                        </div>
                        <div>
                            <div class="flex items-center gap-2">
                                <div class="font-bold text-white">{symbol}</div>
                                <span class="badge {status_badge} text-[10px] px-2 py-0.5 animate-pulse">
                                    <i class="fas {status_icon} text-xs mr-1"></i>{status_text}
                                </span>
                            </div>
                            <div class="text-xs text-yellow-400 font-semibold">{side.upper()} Order • {qty} shares</div>
                            <div class="text-xs text-gray-400">
                                {order_type.upper()}{f' @ ${limit_price:.2f}' if limit_price else ''}
                            </div>
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="text-xs text-gray-400">Order ID</div>
                        <div class="text-xs text-gray-500 font-mono">{str(order_id)[:8]}...</div>
                    </div>
                </div>
                {(f'<div class="mb-2 pt-2 border-t border-yellow-500/20"><div class="grid grid-cols-2 gap-2 text-xs"><div class="glass rounded p-1.5 bg-red-500/10 border border-red-500/30"><div class="text-gray-400">Stop Loss</div><div class="text-red-400 font-bold">${stop_loss:.2f}</div></div><div class="glass rounded p-1.5 bg-green-500/10 border border-green-500/30"><div class="text-gray-400">Take Profit</div><div class="text-green-400 font-bold">${take_profit:.2f}</div></div></div></div>' if stop_loss and take_profit else '')}
                <div class="flex gap-2 pt-2 border-t border-yellow-500/20">
                    <button 
                        hx-post="/api/cancel-order"
                        hx-vals='{{"order_id": "{order_id}"}}'
                        hx-target="#positions-list"
                        hx-swap="innerHTML"
                        hx-confirm="Cancel {side.upper()} order for {symbol}?"
                        class="flex-1 glass-strong py-1.5 rounded-lg text-xs font-bold text-white hover:bg-red-600/30 transition-all bg-red-500/30 flex items-center justify-center gap-1 border border-red-500/50">
                        <i class="fas fa-times text-xs"></i>
                        <span>CANCEL</span>
                    </button>
                </div>
            </div>
            """
        
        # Then show active positions
        for p in pos:
            pl = float(p.unrealized_pl)
            pl_pct = (pl / (float(p.market_value) - pl)) * 100 if float(p.market_value) != pl else 0
            color = "text-green-400" if pl > 0 else "text-red-400"
            badge_color = "badge-success" if pl > 0 else "badge-danger"
            market_value = float(p.market_value)
            current_price = float(p.current_price)
            avg_entry_price = float(p.avg_entry_price)
            
            # Get entry/stop/take_profit from db.history
            trade_record = await db.history.find_one(
                {"symbol": p.symbol, "action": "buy"},
                sort=[("timestamp", -1)]
            )
            
            entry_price = trade_record.get("entry_price") if trade_record else avg_entry_price
            stop_loss = trade_record.get("stop_loss") if trade_record else None
            take_profit = trade_record.get("take_profit") if trade_record else None
            
            # Calculate distances
            distance_to_stop = current_price - stop_loss if stop_loss else None
            distance_to_target = take_profit - current_price if take_profit else None
            risk_reward_ratio = None
            if stop_loss and take_profit:
                risk_amount = entry_price - stop_loss
                reward_amount = take_profit - entry_price
                risk_reward_ratio = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0
            
            # Check for sell signals
            sell_signal = None
            sell_reason = None
            try:
                # Fetch current market data for analysis
                bars, _, _ = get_market_data(p.symbol, days=100)
                if bars is not None and not bars.empty and len(bars) >= 14:
                    techs = analyze_technicals(bars)
                    rsi = techs.get('rsi', 50)
                    
                    # Check sell signals
                    if take_profit and current_price >= take_profit:
                        sell_signal = "profit_target"
                        sell_reason = "Profit target reached! Time to sell high."
                    elif take_profit and current_price >= take_profit * 0.98:  # Within 2% of target
                        sell_signal = "near_target"
                        sell_reason = "Near profit target - consider selling high"
                    elif rsi > 70:
                        sell_signal = "overbought"
                        sell_reason = "RSI overbought - stock may be overvalued"
                    elif rsi > 65 and pl > 0:  # Profitable and getting overbought
                        sell_signal = "consider_sell"
                        sell_reason = "RSI rising, profit locked - consider selling high"
            except Exception as e:
                logger.debug(f"Could not analyze sell signals for {p.symbol}: {e}")
                # Continue without sell signal if analysis fails
            
            # Build risk/reward section HTML
            risk_reward_html = ""
            if distance_to_stop or distance_to_target:
                risk_html = f'<div class="glass rounded p-1.5 bg-red-500/10 border border-red-500/30"><div class="text-gray-400">Risk</div><div class="text-red-400 font-bold">-${abs(distance_to_stop):.2f}</div><div class="text-gray-500 text-[10px]">to stop loss</div></div>' if distance_to_stop else ''
                reward_html = f'<div class="glass rounded p-1.5 bg-green-500/10 border border-green-500/30"><div class="text-gray-400">Reward</div><div class="text-green-400 font-bold">+${distance_to_target:.2f}</div><div class="text-gray-500 text-[10px]">to profit target</div></div>' if distance_to_target else ''
                ratio_html = f'<div class="mt-2 text-center"><span class="badge badge-info text-[10px] px-2 py-0.5">Risk/Reward: {risk_reward_ratio}:1</span></div>' if risk_reward_ratio else ''
                risk_reward_html = f"""
                <div class="mb-2 pt-2 border-t border-gray-700/30">
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        {risk_html}
                        {reward_html}
                    </div>
                    {ratio_html}
                </div>
                """
            
            # Build sell signal indicator
            sell_signal_html = ""
            if sell_signal:
                signal_color = "bg-yellow-500/20 border-yellow-500/50" if sell_signal == "near_target" or sell_signal == "consider_sell" else "bg-green-500/20 border-green-500/50"
                signal_icon = "fa-exclamation-triangle" if sell_signal == "overbought" else "fa-chart-line" if sell_signal == "near_target" or sell_signal == "consider_sell" else "fa-check-circle"
                signal_text = "🎯 SELL HIGH" if sell_signal == "profit_target" else "⚠️ Consider Selling" if sell_signal in ["near_target", "consider_sell"] else "📈 Overbought"
                sell_signal_html = f"""
                <div class="mb-2 p-2 rounded-lg {signal_color} border flex items-center gap-2 animate-pulse">
                    <i class="fas {signal_icon} text-yellow-400"></i>
                    <div class="flex-1">
                        <div class="text-xs font-bold text-yellow-300">{signal_text}</div>
                        <div class="text-[10px] text-gray-300 mt-0.5">{sell_reason}</div>
                    </div>
                </div>
                """
            
            html += f"""
            <div class="glass rounded-lg p-3 mb-2 card-hover border border-gray-700/50">
                <div class="flex justify-between items-center mb-2">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center border border-purple-500/30">
                            <span class="font-bold text-white text-sm">{p.symbol}</span>
                        </div>
                        <div>
                            <div class="font-bold text-white">{p.symbol}</div>
                            <div class="text-xs text-green-400 font-semibold">Bought @ ${entry_price:.2f} (Low Buy)</div>
                            <div class="text-xs text-gray-400">{p.qty} shares • Now: ${current_price:.2f}</div>
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="font-bold {color}">${pl:.2f}</div>
                        <div class="text-xs {color}">{pl_pct:+.2f}%</div>
                    </div>
                </div>
                {sell_signal_html}
                {risk_reward_html}
                <div class="flex gap-2 pt-2 border-t border-gray-700/30">
                    <button 
                        hx-post="/api/quick-sell"
                        hx-vals='{{"symbol": "{p.symbol}"}}'
                        hx-target="#positions-list"
                        hx-swap="innerHTML"
                        hx-confirm="Close position for {p.symbol}?"
                            class="flex-1 glass-strong py-1.5 rounded-lg text-xs font-bold text-white hover:bg-red-600/30 transition-all bg-red-500/30 flex items-center justify-center gap-1 border border-red-500/50">
                        <i class="fas fa-times text-xs"></i>
                        <span>CLOSE</span>
                    </button>
                    <button 
                        hx-get="/api/position-chart/{p.symbol}"
                        hx-target="#position-chart-content"
                        hx-trigger="click"
                        onclick="openModal('position-chart-modal')"
                            class="glass-strong py-1.5 px-3 rounded-lg text-xs font-semibold text-white hover:bg-blue-600/20 transition-all bg-blue-500/20 flex items-center justify-center">
                        <i class="fas fa-chart-line text-xs"></i>
                    </button>
                </div>
            </div>
            """
        
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Failed to get positions: {e}", exc_info=True)
        error_msg = str(e)[:50] if str(e) else "Unknown error"
        return HTMLResponse(content=f"<span class='text-red-500'>API Error: {error_msg}</span>")

async def cancel_order(order_id: str = Form(...), db = Depends(get_scoped_db)) -> HTMLResponse:
    """Cancel a pending order.
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) for automatic connection management.
    Returns HTML with hx-swap-oob to update positions list and show toast notification.
    """
    if not check_alpaca_config() or api is None:
        error_html = """
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-red-500/30 bg-red-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-times-circle text-red-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Alpaca API not configured</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=400)
    
    try:
        logger.info(f"Canceling order: {order_id}")
        
        # Cancel the order via Alpaca API
        api.cancel_order(order_id)
        
        # Get updated positions HTML
        positions_response = await get_positions(db=db)
        positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
        
        # Return HTML with hx-swap-oob to update positions and show success toast
        success_html = f"""
        <div id="positions-list" hx-swap-oob="innerHTML">
            {positions_html}
        </div>
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-green-500/30 bg-green-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-check-circle text-green-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Order canceled successfully</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        response = HTMLResponse(content=success_html)
        response.headers["HX-Trigger"] = "refreshPositions"
        return response
        
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id}: {e}", exc_info=True)
        error_msg = str(e)[:150] if str(e) else "Unknown error occurred"
        error_html = f"""
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-red-500/30 bg-red-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-times-circle text-red-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Cancel failed: {error_msg}</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=500)

async def get_position_chart(symbol: str, db = Depends(get_scoped_db)) -> HTMLResponse:
    """Get position chart with entry, stop loss, and take profit levels.
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) for automatic connection management.
    Returns HTML with embedded matplotlib chart showing buy low/sell high levels.
    """
    if not check_alpaca_config():
        return HTMLResponse(content="<div class='text-yellow-500 text-sm'>Alpaca API not configured</div>")
    
    if api is None:
        return HTMLResponse(content="<div class='text-red-500 text-sm'>Alpaca API not initialized</div>")
    
    try:
        symbol = symbol.upper().strip()
        
        # Get position from Alpaca
        positions = api.list_positions()
        position = None
        for p in positions:
            if p.symbol == symbol:
                position = p
                break
        
        if not position:
            return HTMLResponse(content="<div class='text-yellow-500 text-sm'>Position not found</div>")
        
        avg_entry_price = float(position.avg_entry_price)
        current_price = float(position.current_price)
        qty = float(position.qty)
        
        # Get entry/stop/take_profit from db.history (most recent buy for this symbol)
        trade_record = await db.history.find_one(
            {"symbol": symbol, "action": "buy"},
            sort=[("timestamp", -1)]
        )
        
        entry_price = trade_record.get("entry_price") if trade_record else avg_entry_price
        stop_loss = trade_record.get("stop_loss") if trade_record else None
        take_profit = trade_record.get("take_profit") if trade_record else None
        
        # If not in DB, calculate from strategy defaults
        if stop_loss is None or take_profit is None:
            df, _, _ = get_market_data(symbol, days=60)
            if df is not None and not df.empty:
                techs = analyze_technicals(df)
                atr_value = techs.get('atr', 0)
                if stop_loss is None:
                    stop_loss = round(entry_price - (2 * atr_value), 2)
                if take_profit is None:
                    take_profit = round(entry_price + (3 * atr_value), 2)
            else:
                # Fallback: use 5% stop, 10% target
                stop_loss = round(entry_price * 0.95, 2)
                take_profit = round(entry_price * 1.10, 2)
        
        # Get price history for chart (last 60 days)
        df, _, _ = get_market_data(symbol, days=60)
        if df is None or df.empty:
            return HTMLResponse(content="<div class='text-yellow-500 text-sm'>Insufficient data for chart</div>")
        
        # Generate matplotlib chart
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import base64
        
        plt.figure(figsize=(10, 5), facecolor='#111827')
        ax = plt.gca()
        ax.set_facecolor('#111827')
        
        # Plot price history
        prices = df['close'].values
        dates = range(len(prices))
        plt.plot(dates, prices, color='#60a5fa', linewidth=2, label='Price')
        
        # Add entry price line (green)
        plt.axhline(y=entry_price, color='#10b981', linestyle='--', linewidth=2, label=f'Entry: ${entry_price:.2f}')
        plt.text(len(prices) * 0.02, entry_price, f'Entry: ${entry_price:.2f}', 
                color='#10b981', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#111827', edgecolor='#10b981', alpha=0.8))
        
        # Add stop loss line (red)
        plt.axhline(y=stop_loss, color='#ef4444', linestyle='--', linewidth=2, label=f'Stop Loss: ${stop_loss:.2f}')
        plt.text(len(prices) * 0.02, stop_loss, f'Risk Limit: ${stop_loss:.2f}', 
                color='#ef4444', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#111827', edgecolor='#ef4444', alpha=0.8))
        
        # Add take profit line (green)
        plt.axhline(y=take_profit, color='#10b981', linestyle='--', linewidth=2, label=f'Take Profit: ${take_profit:.2f}')
        plt.text(len(prices) * 0.02, take_profit, f'Profit Target: ${take_profit:.2f}', 
                color='#10b981', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#111827', edgecolor='#10b981', alpha=0.8))
        
        # Add current price indicator
        plt.axhline(y=current_price, color='#fbbf24', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Current: ${current_price:.2f}')
        plt.plot(len(prices) - 1, current_price, 'o', color='#fbbf24', markersize=10, label='Current Price')
        
        plt.title(f"{symbol} - Buy Low, Sell High", color='white', fontsize=12, fontweight='bold')
        plt.xlabel('Days', color='gray', fontsize=9)
        plt.ylabel('Price ($)', color='gray', fontsize=9)
        plt.tick_params(colors='gray', labelsize=8)
        plt.grid(color='#374151', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(loc='upper left', fontsize=8, facecolor='#111827', edgecolor='#374151', labelcolor='white')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', transparent=False, bbox_inches='tight', facecolor='#111827')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Calculate metrics
        risk_amount = entry_price - stop_loss
        reward_amount = take_profit - entry_price
        risk_reward_ratio = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0
        distance_to_stop = current_price - stop_loss
        distance_to_target = take_profit - current_price
        distance_to_stop_pct = round((distance_to_stop / entry_price) * 100, 2) if entry_price > 0 else 0
        distance_to_target_pct = round((distance_to_target / entry_price) * 100, 2) if entry_price > 0 else 0
        
        # Build HTML response
        html_content = f"""
        <div class="glass rounded-xl p-6 border border-purple-500/20" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
            <div class="flex items-center justify-between mb-6 pb-4 border-b border-gray-700/50">
                <h3 class="text-2xl font-bold text-white">{symbol} - Position Chart</h3>
                <button onclick="closeModal('position-chart-modal')" class="text-gray-400 hover:text-white transition text-2xl">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            
            <div class="mb-6">
                <img src="data:image/png;base64,{plot_url}" alt="{symbol} Chart" class="w-full rounded-lg border-2 border-gray-700/50">
            </div>
            
            <div class="grid grid-cols-2 gap-4 mb-6">
                <div class="glass rounded-lg p-5 border-2 border-green-500/30 bg-green-500/10">
                    <div class="text-sm text-gray-300 mb-2 font-medium">Entry Price (Low Buy)</div>
                    <div class="text-3xl font-bold text-green-400 mb-2">${entry_price:.2f}</div>
                    <div class="text-sm text-gray-400">{qty:.0f} shares</div>
                </div>
                <div class="glass rounded-lg p-5 border-2 border-yellow-500/30 bg-yellow-500/10">
                    <div class="text-sm text-gray-300 mb-2 font-medium">Current Price</div>
                    <div class="text-3xl font-bold text-yellow-400 mb-2">${current_price:.2f}</div>
                    <div class="text-sm text-gray-400">{((current_price - entry_price) / entry_price * 100):+.2f}%</div>
                </div>
                <div class="glass rounded-lg p-5 border-2 border-red-500/30 bg-red-500/10">
                    <div class="text-sm text-gray-300 mb-2 font-medium">Risk Limit (Stop Loss)</div>
                    <div class="text-3xl font-bold text-red-400 mb-2">${stop_loss:.2f}</div>
                    <div class="text-sm text-gray-400">-${abs(distance_to_stop):.2f} ({distance_to_stop_pct:.1f}%)</div>
                </div>
                <div class="glass rounded-lg p-5 border-2 border-green-500/30 bg-green-500/10">
                    <div class="text-sm text-gray-300 mb-2 font-medium">Profit Target (Sell High)</div>
                    <div class="text-3xl font-bold text-green-400 mb-2">${take_profit:.2f}</div>
                    <div class="text-sm text-gray-400">+${distance_to_target:.2f} ({distance_to_target_pct:.1f}%)</div>
                </div>
            </div>
            
            <div class="glass rounded-lg p-5 border-2 border-blue-500/30 bg-blue-500/10">
                <div class="flex items-center justify-between">
                    <div>
                        <div class="text-sm text-gray-300 mb-2 font-medium">Risk/Reward Ratio</div>
                        <div class="text-2xl font-bold text-blue-400 mb-1">{risk_reward_ratio}:1</div>
                        <div class="text-sm text-gray-400">For every $1 risked, potential ${risk_reward_ratio:.2f} reward</div>
                    </div>
                    <div class="text-right">
                        <div class="text-sm text-gray-300 mb-2 font-medium">Unrealized P&L</div>
                        <div class="text-2xl font-bold {'text-green-400' if float(position.unrealized_pl) > 0 else 'text-red-400'} mb-1">
                            ${float(position.unrealized_pl):.2f}
                        </div>
                        <div class="text-sm text-gray-400">{((current_price - entry_price) / entry_price * 100):+.2f}%</div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Failed to get position chart: {e}", exc_info=True)
        error_msg = str(e)[:50] if str(e) else "Unknown error"
        return HTMLResponse(content=f"<div class='text-red-500 text-sm'>Error: {error_msg}</div>")

async def get_trade_logs(db = Depends(get_scoped_db)) -> HTMLResponse:
    """Get trade history logs."""
    try:
        trades = await db.history.find({}).sort("timestamp", -1).limit(10).to_list(length=10)
        html = ""
        for t in trades:
            ts = t['timestamp'].strftime('%H:%M')
            action = t['action'].upper()
            style = "text-green-400" if action == 'BUY' else "text-red-400"
            badge_style = "badge-success" if action == 'BUY' else "badge-danger"
            html += f"""
            <tr class="glass border-b border-gray-700/30 hover:bg-white/5 transition fade-in">
                <td class="px-4 py-3 font-mono text-gray-400 text-xs">{ts}</td>
                <td class="px-4 py-3">
                    <div class="flex items-center gap-2">
                        <div class="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
                            <span class="font-bold text-white text-xs">{t['symbol']}</span>
                        </div>
                        <span class="font-bold text-white">{t['symbol']}</span>
                    </div>
                </td>
                <td class="px-4 py-3">
                    <span class="badge {badge_style}">{action}</span>
                </td>
                <td class="px-4 py-3 text-white font-semibold">${t.get('price', 0):.2f}</td>
                <td class="px-4 py-3">
                    <div class="flex items-start gap-2 max-w-[300px]">
                        <i class="fas fa-quote-left text-gray-600 text-xs mt-0.5 flex-shrink-0"></i>
                        <span class="text-gray-300 text-xs break-words leading-relaxed">{t['reason']}</span>
                    </div>
                </td>
            </tr>
            """
        if not html:
            html = """
            <tr>
                <td colspan="5" class="px-4 py-8 text-center text-gray-500">
                    <i class="fas fa-history text-3xl mb-2 opacity-50"></i>
                    <p class="text-sm">No trades yet</p>
                    <p class="text-xs mt-1 text-gray-600">Trade history will appear here</p>
                </td>
            </tr>
            """
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Failed to get trade logs: {e}", exc_info=True)
        return HTMLResponse(content="<tr><td colspan='5' class='text-red-500 text-center'>Error loading logs</td></tr>")

async def get_strategy_config() -> HTMLResponse:
    """Get current strategy configuration - Balanced Low Buy System."""
    color = STRATEGY_CONFIG.get('color', 'green')
    color_classes = {
        'green': 'border-green-500/30 bg-green-500/10',
        'blue': 'border-blue-500/30 bg-blue-500/10',
        'yellow': 'border-yellow-500/30 bg-yellow-500/10',
        'red': 'border-red-500/30 bg-red-500/10'
    }
    border_class = color_classes.get(color, 'border-green-500/30')
    
    return HTMLResponse(content=f"""
        <div class="glass rounded-xl p-4 border {border_class}">
            <div class="mb-3">
                <div class="text-xs text-gray-400 mb-1 uppercase tracking-wider">Strategy</div>
                <div class="text-lg font-bold text-white">{STRATEGY_CONFIG['name']}</div>
            </div>
            <p class="text-xs text-gray-400 mb-3">Buy Low: Enter when stocks are oversold (RSI &lt; {STRATEGY_CONFIG['rsi_threshold']}) with strong bounce-back signals.<br>Sell High: Exit at profit targets (entry + 3 ATR) or risk limits (entry - 2 ATR).</p>
            <div class="grid grid-cols-2 gap-3 text-xs">
                <div class="glass rounded-lg p-2">
                    <div class="text-gray-400 mb-1">RSI Threshold</div>
                    <div class="text-white font-bold">&lt; {STRATEGY_CONFIG['rsi_threshold']}</div>
                    <div class="text-gray-500 text-[10px] mt-1">On Sale (oversold)</div>
                </div>
                <div class="glass rounded-lg p-2">
                    <div class="text-gray-400 mb-1">AI Score Required</div>
                    <div class="text-white font-bold">≥ {STRATEGY_CONFIG['ai_score_required']}</div>
                    <div class="text-gray-500 text-[10px] mt-1">Bounce-back confidence</div>
                </div>
                <div class="glass rounded-lg p-2">
                    <div class="text-gray-400 mb-1">Risk/Reward</div>
                    <div class="text-white font-bold">3:1</div>
                    <div class="text-gray-500 text-[10px] mt-1">Reward-to-risk ratio</div>
                </div>
                <div class="glass rounded-lg p-2">
                    <div class="text-gray-400 mb-1">Risk Per Trade</div>
                    <div class="text-white font-bold">${STRATEGY_CONFIG['risk_per_trade']:.0f}</div>
                </div>
                <div class="glass rounded-lg p-2">
                    <div class="text-gray-400 mb-1">Max Capital</div>
                    <div class="text-white font-bold">${STRATEGY_CONFIG['max_capital']:,.0f}</div>
                </div>
            </div>
        </div>
    """)

async def get_strategy_api(db = Depends(get_scoped_db)) -> JSONResponse:
    """Get current active strategy config from MongoDB or env vars."""
    try:
        # Try to get from MongoDB first
        db_config = await get_strategy_from_db(db)
        if db_config:
            return JSONResponse(content={
                "success": True,
                "source": "database",
                "config": db_config
            })
        
        # Fallback to env vars
        return JSONResponse(content={
            "success": True,
            "source": "environment",
            "config": STRATEGY_CONFIG
        })
    except Exception as e:
        logger.error(f"Error getting strategy config: {e}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": str(e)[:100]},
            status_code=500
        )

async def update_strategy_api(request: Request, db = Depends(get_scoped_db)) -> JSONResponse:
    """Update strategy configuration in MongoDB."""
    try:
        body = await request.json()
        preset = body.get('preset')
        
        # Strategy presets
        presets = {
            "Conservative": {
                "rsi_threshold": 30,
                "ai_score_required": 9,
                "risk_per_trade": 25.0,
                "max_capital": 2500.0
            },
            "Moderate": {
                "rsi_threshold": 35,
                "ai_score_required": 7,
                "risk_per_trade": 50.0,
                "max_capital": 5000.0
            },
            "Aggressive": {
                "rsi_threshold": 40,
                "ai_score_required": 6,
                "risk_per_trade": 100.0,
                "max_capital": 10000.0
            }
        }
        
        # Determine config values
        if preset and preset in presets:
            config = presets[preset].copy()
            config['preset'] = preset
        elif preset == "Custom" or body.get('rsi_threshold'):
            # Custom parameters
            config = {
                "rsi_threshold": body.get('rsi_threshold', STRATEGY_CONFIG.get('rsi_threshold', 35)),
                "ai_score_required": body.get('ai_score_required', STRATEGY_CONFIG.get('ai_score_required', 7)),
                "risk_per_trade": body.get('risk_per_trade', STRATEGY_CONFIG.get('risk_per_trade', 50.0)),
                "max_capital": body.get('max_capital', STRATEGY_CONFIG.get('max_capital', 5000.0)),
                "preset": "Custom"
            }
        else:
            return JSONResponse(
                content={"success": False, "error": "Invalid request: provide 'preset' or custom parameters"},
                status_code=400
            )
        
        # Validate parameters
        if not (0 < config['rsi_threshold'] <= 100):
            return JSONResponse(
                content={"success": False, "error": "rsi_threshold must be between 0 and 100"},
                status_code=400
            )
        if not (0 <= config['ai_score_required'] <= 10):
            return JSONResponse(
                content={"success": False, "error": "ai_score_required must be between 0 and 10"},
                status_code=400
            )
        if config['risk_per_trade'] <= 0:
            return JSONResponse(
                content={"success": False, "error": "risk_per_trade must be positive"},
                status_code=400
            )
        if config['max_capital'] <= 0:
            return JSONResponse(
                content={"success": False, "error": "max_capital must be positive"},
                status_code=400
            )
        
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
        
        logger.info(f"Strategy updated: {config.get('preset', 'Custom')} - RSI<{config['rsi_threshold']}, Score≥{config['ai_score_required']}")
        
        return JSONResponse(content={
            "success": True,
            "config": config_doc
        })
    except Exception as e:
        logger.error(f"Error updating strategy config: {e}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": str(e)[:100]},
            status_code=500
        )

async def get_strategy_presets() -> JSONResponse:
    """List available strategy presets."""
    presets = [
        {
            "name": "Conservative",
            "description": "Low risk, high quality signals only",
            "rsi_threshold": 30,
            "ai_score_required": 9,
            "risk_per_trade": 25.0,
            "max_capital": 2500.0
        },
        {
            "name": "Moderate",
            "description": "Balanced risk and reward",
            "rsi_threshold": 35,
            "ai_score_required": 7,
            "risk_per_trade": 50.0,
            "max_capital": 5000.0
        },
        {
            "name": "Aggressive",
            "description": "Higher risk, more opportunities",
            "rsi_threshold": 40,
            "ai_score_required": 6,
            "risk_per_trade": 100.0,
            "max_capital": 10000.0
        },
        {
            "name": "Custom",
            "description": "Configure your own parameters",
            "rsi_threshold": None,
            "ai_score_required": None,
            "risk_per_trade": None,
            "max_capital": None
        }
    ]
    
    return JSONResponse(content={
        "success": True,
        "presets": presets
    })

async def debug_symbol(ticker: str = Form(...)) -> HTMLResponse:
    """Debug endpoint to see what data we're getting."""
    symbol = ticker.upper().strip()
    if not symbol:
        return HTMLResponse(content="Enter Symbol")
    
    debug_info = []
    debug_info.append(f"<div class='text-xs font-mono bg-gray-900 p-4 rounded border border-gray-700'>")
    debug_info.append(f"<div class='font-bold mb-2'>Debug Info for {symbol}</div>")
    
    # Check API
    if not api:
        debug_info.append(f"<div class='text-red-500'>❌ Alpaca API not initialized</div>")
        debug_info.append("</div>")
        return HTMLResponse(content="".join(debug_info))
    
    debug_info.append(f"<div class='text-green-500'>✅ Alpaca API initialized</div>")
    
    # Try to get bars
    try:
        from alpaca_trade_api.rest import TimeFrame
        from datetime import datetime, timedelta
        
        # Test 1: Limit approach with 'iex' feed
        try:
            bars_limit = api.get_bars(symbol, TimeFrame.Day, limit=250, feed='iex').df
            debug_info.append(f"<div class='mt-2'>Limit approach: {len(bars_limit)} bars</div>")
            if not bars_limit.empty:
                debug_info.append(f"<div>Columns: {list(bars_limit.columns)}</div>")
                debug_info.append(f"<div>Index: {bars_limit.index.name or 'None'}</div>")
                if len(bars_limit) > 0:
                    debug_info.append(f"<div>First row: {bars_limit.iloc[0].to_dict()}</div>")
                    debug_info.append(f"<div>Last row: {bars_limit.iloc[-1].to_dict()}</div>")
        except Exception as e:
            debug_info.append(f"<div class='text-red-500'>Limit approach failed: {e}</div>")
        
        # Test 2: Date range approach with 'iex' feed
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            bars_date = api.get_bars(
                symbol,
                TimeFrame.Day,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                feed='iex'  # Use 'iex' feed for paper trading
            ).df
            debug_info.append(f"<div class='mt-2'>Date range approach: {len(bars_date)} bars</div>")
        except Exception as e:
            debug_info.append(f"<div class='text-red-500'>Date range approach failed: {e}</div>")
        
    except Exception as e:
        debug_info.append(f"<div class='text-red-500'>Error: {e}</div>")
    
    debug_info.append("</div>")
    return HTMLResponse(content="".join(debug_info))


async def panic_close() -> HTMLResponse:
    """Close all positions and cancel all orders."""
    if not check_alpaca_config():
        return HTMLResponse(content="<span class='text-yellow-500'>Alpaca API not configured</span>")
    
    if api is None:
        return HTMLResponse(content="<span class='text-red-500'>Alpaca API not initialized</span>")
    
    try:
        api.cancel_all_orders()
        api.close_all_positions()
        logger.critical("⚫ The Eye closes all positions - All capital extracted")
        return HTMLResponse(content="<span class='text-red-500'>All positions closed</span>")
    except Exception as e:
        logger.error(f"Failed to panic close: {e}", exc_info=True)
        return HTMLResponse(content=f"<span class='text-red-500'>Error: {str(e)[:50]}</span>")

async def discover_stocks(db = Depends(get_scoped_db)) -> HTMLResponse:
    """Return curated stock lists organized by category."""
    logger.info("🔍 [DISCOVER] Starting discover_stocks endpoint")
    
    try:
        # Curated stock lists with descriptions
        categories = {
            "Tech Giants": {
                "description": "Large technology companies - stable and well-known",
                "stocks": [
                    {"symbol": "AAPL", "name": "Apple Inc."},
                    {"symbol": "MSFT", "name": "Microsoft Corporation"},
                    {"symbol": "GOOGL", "name": "Alphabet Inc. (Google)"},
                    {"symbol": "AMZN", "name": "Amazon.com Inc."},
                    {"symbol": "META", "name": "Meta Platforms Inc. (Facebook)"},
                    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
                    {"symbol": "TSLA", "name": "Tesla Inc."},
                ]
            },
            "Growth Stocks": {
                "description": "Companies with high growth potential",
                "stocks": [
                    {"symbol": "AMD", "name": "Advanced Micro Devices"},
                    {"symbol": "COIN", "name": "Coinbase Global"},
                    {"symbol": "PLTR", "name": "Palantir Technologies"},
                    {"symbol": "SNOW", "name": "Snowflake Inc."},
                    {"symbol": "CRWD", "name": "CrowdStrike Holdings"},
                    {"symbol": "NET", "name": "Cloudflare Inc."},
                ]
            },
            "Dividend Stocks": {
                "description": "Stocks that pay regular dividends - good for income",
                "stocks": [
                    {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
                    {"symbol": "VZ", "name": "Verizon Communications"},
                    {"symbol": "KO", "name": "The Coca-Cola Company"},
                    {"symbol": "PG", "name": "Procter & Gamble"},
                    {"symbol": "JNJ", "name": "Johnson & Johnson"},
                    {"symbol": "PEP", "name": "PepsiCo Inc."},
                ]
            },
            "Popular ETFs": {
                "description": "Exchange-Traded Funds - diversified baskets of stocks",
                "stocks": [
                    {"symbol": "SPY", "name": "SPDR S&P 500 ETF"},
                    {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
                    {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF"},
                    {"symbol": "ARKK", "name": "ARK Innovation ETF"},
                ]
            },
            "Semiconductors": {
                "description": "Chip makers - critical for technology",
                "stocks": [
                    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
                    {"symbol": "AMD", "name": "Advanced Micro Devices"},
                    {"symbol": "INTC", "name": "Intel Corporation"},
                    {"symbol": "TSM", "name": "Taiwan Semiconductor"},
                    {"symbol": "AVGO", "name": "Broadcom Inc."},
                ]
            }
        }
        
        logger.info(f"🔍 [DISCOVER] Generating HTML for {len(categories)} categories")
        html = ""
        for category_name, category_data in categories.items():
            logger.debug(f"   Processing category: {category_name} with {len(category_data['stocks'])} stocks")
            html += f"""
            <div class="glass rounded-lg p-4 border-l-4 border-green-500/50 mb-4">
                <h3 class="font-bold text-white mb-1 flex items-center gap-2">
                    <i class="fas fa-folder text-green-400"></i>
                    {category_name}
                </h3>
                <p class="text-xs text-gray-400 mb-3">{category_data['description']}</p>
                <div class="grid grid-cols-2 gap-2">
            """
            
            for stock in category_data['stocks']:
                symbol = stock['symbol']
                name = stock['name']
                html += f"""
                    <div class="stock-card glass rounded-lg p-3 border border-gray-700/50 hover:border-green-500/30 transition" 
                         data-symbol="{symbol}" data-name="{name}">
                        <div class="flex items-center justify-between">
                            <div>
                                <div class="font-bold text-white">{symbol}</div>
                                <div class="text-xs text-gray-400">{name}</div>
                            </div>
                        </div>
                    </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        logger.info(f"✅ [DISCOVER] Generated HTML ({len(html)} chars), returning response")
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"❌ [DISCOVER] Error in discover_stocks: {e}", exc_info=True)
        error_html = f"""
        <div class="glass rounded-lg p-4 border border-red-500/30">
            <div class="flex items-center gap-2 text-red-400">
                <i class="fas fa-exclamation-triangle"></i>
                <span class="font-semibold">Error loading stocks</span>
            </div>
            <p class="text-xs text-gray-400 mt-2">{str(e)[:100]}</p>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=500)


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
                bars, headlines, news_objects = get_market_data(symbol, days=500)
                if bars is None or bars.empty or len(bars) < 14:
                    continue
                
                techs = analyze_technicals(bars)
                
                # Get AI analysis if available
                ai_score = None
                ai_reason = None
                risk_level = "UNKNOWN"
                
                if _ai_engine:
                    try:
                        # Get strategy instance for AI prompt
                        strategy = get_strategy_instance()
                        strategy_prompt = strategy.get_ai_prompt()
                        
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

async def quick_buy(symbol: str = Form(...), qty: Optional[int] = Form(10), db = Depends(get_scoped_db)) -> HTMLResponse:
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
        error_html = """
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-red-500/30 bg-red-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-times-circle text-red-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Alpaca API not configured</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=400)
    
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
            error_html = f"""
            <div id="toast-container" hx-swap-oob="beforeend">
                <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-red-500/30 bg-red-500/10">
                    <div class="flex items-start gap-3">
                        <i class="fas fa-times-circle text-red-400 text-lg mt-0.5 flex-shrink-0"></i>
                        <div class="flex-1 min-w-0">
                            <p class="text-sm font-semibold text-white break-words">{error_msg}</p>
                            <p class="text-xs text-gray-400 mt-1">Possible reasons: No market data, position size too small, or API error</p>
                        </div>
                        <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                            <i class="fas fa-times text-xs"></i>
                        </button>
                    </div>
                </div>
            </div>
            """
            return HTMLResponse(content=error_html, status_code=500)
        
        # Order succeeded - get updated positions
        logger.info(f"Buy order placed successfully for {symbol}: {order}")
        
        # Get updated positions HTML - pass db explicitly
        positions_response = await get_positions(db=db)
        positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
        
        # Return HTML with hx-swap-oob to update positions and show success toast
        success_html = f"""
        <div id="positions-list" hx-swap-oob="innerHTML">
            {positions_html}
        </div>
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-green-500/30 bg-green-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-check-circle text-green-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Buy order placed for {symbol}</p>
                        <p class="text-xs text-gray-300 mt-1">Order ID: {getattr(order, 'id', 'N/A')}</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        response = HTMLResponse(content=success_html)
        response.headers["HX-Trigger"] = "refreshPositions"
        return response
        
    except Exception as e:
        logger.error(f"Quick buy failed for {symbol}: {e}", exc_info=True)
        error_msg = str(e)[:150] if str(e) else "Unknown error occurred"
        error_html = f"""
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-red-500/30 bg-red-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-times-circle text-red-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Buy failed: {error_msg}</p>
                        <p class="text-xs text-gray-400 mt-1">Check server logs for details</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=500)

async def quick_sell(symbol: str = Form(...), db = Depends(get_scoped_db)) -> HTMLResponse:
    """Quick sell/close position - returns HTML with hx-swap-oob for multiple updates.
    
    MDB-Engine Pattern: Uses Depends(get_scoped_db) for database access.
    """
    if not check_alpaca_config() or api is None:
        error_html = """
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-red-500/30 bg-red-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-times-circle text-red-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Alpaca API not configured</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=400)
    
    try:
        api.close_position(symbol.upper())
        
        # Get updated positions HTML - pass db explicitly
        positions_response = await get_positions(db=db)
        positions_html = positions_response.body.decode() if hasattr(positions_response.body, 'decode') else str(positions_response.body)
        
        # Return HTML with hx-swap-oob to update positions and show success toast
        success_html = f"""
        <div id="positions-list" hx-swap-oob="innerHTML">
            {positions_html}
        </div>
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-green-500/30 bg-green-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-check-circle text-green-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Position closed for {symbol}</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        response = HTMLResponse(content=success_html)
        response.headers["HX-Trigger"] = "refreshPositions"
        return response
    except Exception as e:
        error_msg = str(e)
        if "position does not exist" in error_msg.lower():
            error_html = """
            <div id="toast-container" hx-swap-oob="beforeend">
                <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-yellow-500/30 bg-yellow-500/10">
                    <div class="flex items-start gap-3">
                        <i class="fas fa-exclamation-triangle text-yellow-400 text-lg mt-0.5 flex-shrink-0"></i>
                        <div class="flex-1 min-w-0">
                            <p class="text-sm font-semibold text-white break-words">No position to close</p>
                        </div>
                        <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                            <i class="fas fa-times text-xs"></i>
                        </button>
                    </div>
                </div>
            </div>
            """
            return HTMLResponse(content=error_html, status_code=404)
        logger.error(f"Quick sell failed: {e}", exc_info=True)
        error_html = f"""
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-red-500/30 bg-red-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-times-circle text-red-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Error: {str(e)[:100]}</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=500)

async def cancel_orders(symbol: str = Form(None)) -> HTMLResponse:
    """Cancel orders - all or for specific symbol - returns HTML toast notification."""
    if not check_alpaca_config() or api is None:
        error_html = """
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-red-500/30 bg-red-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-times-circle text-red-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Alpaca API not configured</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=400)
    
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
        
        success_html = f"""
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-green-500/30 bg-green-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-check-circle text-green-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">{message}</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        return HTMLResponse(content=success_html)
    except Exception as e:
        logger.error(f"Cancel orders failed: {e}", exc_info=True)
        error_html = f"""
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast glass rounded-lg p-4 border pointer-events-auto shadow-lg min-w-[300px] max-w-[400px] border-red-500/30 bg-red-500/10">
                <div class="flex items-start gap-3">
                    <i class="fas fa-times-circle text-red-400 text-lg mt-0.5 flex-shrink-0"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-semibold text-white break-words">Error: {str(e)[:100]}</p>
                    </div>
                    <button onclick="this.closest('.toast').remove()" class="text-gray-400 hover:text-white transition flex-shrink-0">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=500)

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

async def _get_trending_stocks_with_analysis_streaming(websocket: WebSocket, db) -> None:
    """Stream stock analysis with progress updates via WebSocket.
    
    Simplified version - analyzes sample stocks without Firecrawl discovery.
    Includes cache integration and historical context.
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
    strategy = get_strategy_instance()
    strategy_id = strategy.get_name()
    
    try:
        # Sample stocks to analyze (popular tech stocks)
        # Scan more symbols to find quality signals (filtered to top 5 in frontend)
        symbols = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "AMD", "META", "AMZN", "INTC", "NFLX"]
        total = len(symbols)
        stocks_with_analysis = []
        
        logger.info(f"📊 [WS] Starting analysis of {total} symbols: {symbols}")
        
        # Send initial progress
        try:
            await websocket.send_json({
                "type": "progress",
                "message": "Analyzing sample stocks...",
                "current": 0,
                "total": total,
                "percentage": 0
            })
            logger.info("✅ [WS] Sent initial progress message")
        except Exception as e:
            logger.error(f"❌ [WS] Failed to send initial progress: {e}", exc_info=True)
        
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
                    cached_analysis = cached['analysis']
                    config = strategy.get_config()
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
                    
                    await _safe_send_json(websocket, {
                        "type": "stock_complete",
                        "stock": sanitized_stock_data,
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100)
                    })
                    continue
                
                # Send progress update
                try:
                    await websocket.send_json({
                        "type": "progress",
                        "message": f"Analyzing {symbol}...",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "cache_hit": False
                    })
                    logger.info(f"✅ [WS] [{idx}/{total}] Sent progress update for {symbol}")
                except Exception as e:
                    logger.error(f"❌ [WS] Failed to send progress for {symbol}: {e}")
                
                # Fetch market data
                logger.info(f"📈 [WS] [{idx}/{total}] Fetching market data for {symbol}...")
                import time
                start_time = time.time()
                
                try:
                bars, headlines, news_objects = get_market_data(symbol, days=500)
                fetch_time = time.time() - start_time
                logger.info(f"⏱️ [WS] [{idx}/{total}] Market data fetch for {symbol} took {fetch_time:.2f}s")
                except Exception as e:
                    error_msg = f"Failed to fetch market data: {str(e)[:100]}"
                    logger.error(f"❌ [WS] [{idx}/{total}] {symbol}: {error_msg}", exc_info=True)
                    failed_stocks.append({"symbol": symbol, "reason": error_msg})
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error fetching data for {symbol}: {str(e)[:50]}",
                        "symbol": symbol,
                        "current": idx,
                        "total": total
                    })
                    continue
                
                if bars is None:
                    logger.warning(f"⚠️ [WS] [{idx}/{total}] {symbol}: bars is None")
                    failed_stocks.append({"symbol": symbol, "reason": "No market data returned"})
                    await websocket.send_json({
                        "type": "progress",
                        "message": f"Skipping {symbol} - no data returned",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "skipped": True
                    })
                    continue
                
                if bars.empty:
                    logger.warning(f"⚠️ [WS] [{idx}/{total}] {symbol}: bars is empty")
                    failed_stocks.append({"symbol": symbol, "reason": "Empty market data"})
                    await websocket.send_json({
                        "type": "progress",
                        "message": f"Skipping {symbol} - empty data",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "skipped": True
                    })
                    continue
                
                if len(bars) < 14:
                    logger.warning(f"⚠️ [WS] [{idx}/{total}] {symbol}: Only {len(bars)} days of data (need 14+)")
                    failed_stocks.append({"symbol": symbol, "reason": f"Insufficient data ({len(bars)} days, need 14+)"})
                    await websocket.send_json({
                        "type": "progress",
                        "message": f"Skipping {symbol} - insufficient data ({len(bars)} days)",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "skipped": True
                    })
                    continue
                
                logger.info(f"✅ [WS] [{idx}/{total}] {symbol}: Got {len(bars)} days of data, {len(headlines)} headlines")
                
                # Send technical analysis progress
                try:
                    await websocket.send_json({
                        "type": "progress",
                        "message": f"Calculating indicators for {symbol}...",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "stage": "technicals"
                    })
                    logger.info(f"✅ [WS] [{idx}/{total}] Sent technicals progress for {symbol}")
                except Exception as e:
                    logger.error(f"❌ [WS] Failed to send technicals progress: {e}")
                
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
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Technical analysis failed for {symbol}: {str(e)[:50]}",
                        "symbol": symbol,
                        "current": idx,
                        "total": total
                    })
                    continue
                
                # Send AI analysis progress
                try:
                    await websocket.send_json({
                        "type": "progress",
                        "message": f"AI analyzing {symbol}...",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "stage": "ai"
                    })
                    logger.info(f"✅ [WS] [{idx}/{total}] Sent AI progress for {symbol}")
                except Exception as e:
                    logger.error(f"❌ [WS] Failed to send AI progress: {e}")
                
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
                        strategy_prompt = strategy.get_ai_prompt()
                        logger.debug(f"   Strategy: {strategy.get_name()}, Prompt length: {len(strategy_prompt)}")
                        
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
                    verdict_dict = verdict.model_dump() if hasattr(verdict, 'model_dump') else verdict.dict() if hasattr(verdict, 'dict') else {
                        'score': verdict.score,
                        'action': verdict.action,
                        'reason': verdict.reason,
                        'risk_level': verdict.risk_level
                    }
                    
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
                
                # Skip stocks with 0/10 score
                if ai_score is not None and ai_score == 0:
                    logger.info(f"⏭️ [WS] [{idx}/{total}] Skipping {symbol} - score is 0/10")
                    await websocket.send_json({
                        "type": "progress",
                        "message": f"Skipping {symbol} - score 0/10",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "skipped": True
                    })
                    continue
                
                config = strategy.get_config()
                adjusted_score = (ai_score or 0) + confidence.get('boost', 0.0)
                meets_criteria = techs.get('rsi', 50) < config.get('rsi_threshold', 35) and adjusted_score >= config.get('ai_score_required', 7)
                
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
                    "ai_score_required": config.get('ai_score_required', 7)
                }
                
                # Sanitize stock_data BEFORE appending (so completion message is also sanitized)
                sanitized_stock_data = _sanitize_for_json(stock_data)
                stocks_with_analysis.append(sanitized_stock_data)
                logger.info(f"✅ [WS] [{idx}/{total}] {symbol} analysis complete, added to results")
                
                # Send completed stock
                try:
                    await _safe_send_json(websocket, {
                        "type": "stock_complete",
                        "stock": sanitized_stock_data,
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100)
                    })
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
                
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error analyzing {symbol}: {str(e)[:50]}",
                        "symbol": symbol,
                        "current": idx,
                        "total": total,
                        "is_rate_limit": is_rate_limit,
                        "error_type": "rate_limit" if is_rate_limit else "general"
                    })
                except Exception as send_err:
                    logger.error(f"❌ [WS] Failed to send error message: {send_err}")
                continue
        
        # Send completion
        # Note: stocks_with_analysis is already sanitized (we sanitize before appending)
        success_count = len(stocks_with_analysis)
        failed_count = len(failed_stocks)
        logger.info(f"🎉 [WS] Analysis complete! Processed {success_count}/{total} stocks successfully, {failed_count} failed")
        
        if failed_stocks:
            logger.warning(f"⚠️ [WS] Failed stocks: {', '.join([f['symbol'] for f in failed_stocks])}")
            for failure in failed_stocks:
                logger.warning(f"   - {failure['symbol']}: {failure['reason']}")
        
        try:
            warnings = []
            if success_count < len(symbols):
                warnings.append(f"Only {success_count} of {len(symbols)} stocks analyzed successfully")
                if failed_stocks:
                    failed_symbols = [f['symbol'] for f in failed_stocks]
                    warnings.append(f"Failed: {', '.join(failed_symbols)}")
            
            await _safe_send_json(websocket, {
                "type": "complete",
                "stocks": stocks_with_analysis,  # Already sanitized
                "total": success_count,
                "expected": len(symbols),
                "failed": failed_stocks,  # Include failure details
                "source": "sample",
                "warnings": warnings
            })
            logger.info("✅ [WS] Sent completion message")
        except Exception as e:
            logger.error(f"❌ [WS] Failed to send completion message: {e}", exc_info=True)
        
    except WebSocketDisconnect:
        logger.info("🔌 [WS] WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"❌ [WS] Fatal WebSocket error: {e}", exc_info=True)
        logger.error(f"   Error type: {type(e).__name__}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Analysis failed: {str(e)[:100]}"
            })
        except:
            logger.error("❌ [WS] Failed to send error message to client")

async def get_explanation(
    symbol: str = Form(...), 
    db = Depends(get_scoped_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> HTMLResponse:
    """Get detailed explanation for a stock analysis including calculations, data, insights, and news.
    
    MDB-Engine Pattern: Dependency Injection
    - db: Scoped database via get_scoped_db()
    - embedding_service: EmbeddingService via get_embedding_service()
    """
    try:
        symbol = symbol.upper().strip()
        strategy = get_strategy_instance()
        strategy_id = strategy.get_name()
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
            bars, headlines, news_objects = get_market_data(symbol, days=500)
            if bars is None or bars.empty or len(bars) < 14:
                error_html = f"""
                <div class="glass rounded-lg p-4 border border-red-500/30">
                    <div class="flex items-center gap-2 text-red-400 mb-2">
                        <i class="fas fa-exclamation-circle"></i>
                        <span class="font-semibold">Error</span>
                    </div>
                    <p class="text-sm text-gray-300">Insufficient data for {symbol}</p>
                </div>
                """
                return HTMLResponse(content=error_html, status_code=400)
            
            bars_count = len(bars)
            techs = analyze_technicals(bars)
            
            # Get AI analysis
            if not _ai_engine:
                error_html = """
                <div class="glass rounded-lg p-4 border border-red-500/30">
                    <div class="flex items-center gap-2 text-red-400 mb-2">
                        <i class="fas fa-exclamation-circle"></i>
                        <span class="font-semibold">Error</span>
                    </div>
                    <p class="text-sm text-gray-300">AI engine not available</p>
                </div>
                """
                return HTMLResponse(content=error_html, status_code=503)
            
            strategy_prompt = strategy.get_ai_prompt()
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
        if isinstance(verdict, dict):
            verdict_score = verdict.get('score', 0)
            verdict_reason = verdict.get('reason', '')
            verdict_risk = verdict.get('risk_level', 'UNKNOWN')
            verdict_action = verdict.get('action', 'UNKNOWN')
        else:
            verdict_score = verdict.score if hasattr(verdict, 'score') else 0
            verdict_reason = verdict.reason if hasattr(verdict, 'reason') else ''
            verdict_risk = verdict.risk_level if hasattr(verdict, 'risk_level') else 'UNKNOWN'
            verdict_action = verdict.action if hasattr(verdict, 'action') else 'UNKNOWN'
        
        config = strategy.get_config()
        
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
        
        # Build HTML explanation - matching frontend structure
        # Escape HTML in text fields
        def escape_html(text):
            if text is None:
                return ''
            return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
        
        # Build similar signals HTML
        similar_signals_html = ''
        if similar_signals_list:
            similar_signals_html = '<div class="glass rounded-lg p-5 border-l-4 border-blue-500/50"><h3 class="text-base font-bold text-white mb-4 flex items-center gap-2"><i class="fas fa-history text-blue-400"></i>Similar Signals (' + str(len(similar_signals_list)) + ')</h3><div class="space-y-2">'
            for s in similar_signals_list[:3]:
                timestamp_str = _convert_timestamp_to_iso(s.get('timestamp')) or 'N/A'
                
                score_val = 0
                if isinstance(s.get('analysis'), dict):
                    score_val = s.get('analysis', {}).get('verdict', {}).get('score', 0)
                elif hasattr(s.get('analysis'), 'score'):
                    score_val = s.get('analysis').score
                
                profitable_val = False
                if isinstance(s.get('outcome'), dict):
                    profitable_val = s.get('outcome', {}).get('profitable', False)
                
                profitable_class = 'badge-success' if profitable_val is True else ('badge-danger' if profitable_val is False else 'badge-info')
                profitable_symbol = '✓' if profitable_val is True else ('✗' if profitable_val is False else '?')
                
                similar_signals_html += f'<div class="glass rounded-lg p-3 bg-blue-500/10 flex items-center justify-between"><span class="text-sm text-gray-200">{escape_html(timestamp_str)} • {score_val}/10</span><span class="badge {profitable_class} text-sm px-3 py-1">{profitable_symbol}</span></div>'
            similar_signals_html += '</div></div>'
        
        # Build news articles HTML
        news_articles_html = ''
        news_objects_list = analysis_data.get('news_objects', [])
        if news_objects_list:
            news_articles_html = '<div class="glass rounded-lg p-5 border-l-4 border-green-500/50"><h3 class="text-base font-bold text-white mb-4 flex items-center gap-2"><i class="fas fa-newspaper text-green-400"></i>Recent News (' + str(len(news_objects_list)) + ')</h3><div class="space-y-3 max-h-96 overflow-y-auto">'
            for article in news_objects_list[:5]:
                headline = escape_html(article.get('headline', ''))
                url = escape_html(article.get('url', '#'))
                summary = escape_html((article.get('summary', '')[:500] if article.get('summary') else ''))
                author = escape_html(article.get('author', 'Unknown'))
                news_articles_html += f'<div class="glass rounded-lg p-4 bg-green-500/10"><a href="{url}" target="_blank" class="text-sm font-semibold text-white hover:text-green-400 transition block mb-2 leading-snug">{headline}</a><p class="text-xs text-gray-300 leading-relaxed line-clamp-3">{summary}</p><p class="text-xs text-gray-400 mt-2">— {author}</p></div>'
            news_articles_html += '</div></div>'
        
        # Determine color classes
        ai_score_color = 'text-green-400' if verdict_score >= 7 else ('text-yellow-400' if verdict_score >= 5 else 'text-red-400')
        adjusted_score_color = 'text-green-400' if adjusted_score >= 7 else ('text-yellow-400' if adjusted_score >= 5 else 'text-red-400')
        trend_color = 'text-green-400' if trend_value == 'UP' else 'text-red-400'
        risk_color = 'text-green-400' if verdict_risk == 'LOW' else ('text-yellow-400' if verdict_risk == 'MEDIUM' else 'text-red-400')
        rsi_status = 'Oversold' if rsi_value < 30 else ('Overbought' if rsi_value > 70 else 'Neutral')
        
        # Calculate exit levels for buy low/sell high framing
        stop_loss_price = round(price_value - (2 * atr_value), 2) if atr_value > 0 else round(price_value * 0.95, 2)
        take_profit_price = round(price_value + (3 * atr_value), 2) if atr_value > 0 else round(price_value * 1.10, 2)
        risk_amount = price_value - stop_loss_price
        reward_amount = take_profit_price - price_value
        risk_reward_ratio = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0
        
        # Build the complete HTML
        html_content = f"""
            <!-- Header: Why This Could Be a Low Buy -->
            <div class="mb-6 pb-4 border-b border-gray-700/50">
                <h3 class="text-2xl font-bold text-white mb-3 leading-tight">Why This Could Be a Low Buy</h3>
                <p class="text-base text-gray-300 leading-relaxed">Stock is oversold (RSI &lt; {config.get('rsi_threshold', 35)}) with signals suggesting recovery. The Eye found this at a low price with bounce-back potential.</p>
            </div>
            
            <!-- Key Metrics Row -->
            <div class="grid grid-cols-4 gap-4 mb-6">
                <div class="glass rounded-lg p-4 bg-purple-500/10 border border-purple-500/20">
                    <div class="text-xs text-gray-400 uppercase tracking-wider mb-2 font-semibold">AI Score</div>
                    <div class="text-3xl font-bold {ai_score_color} mb-1">{verdict_score}/10</div>
                    <div class="text-sm text-gray-300 mt-1">{escape_html(verdict_action)}</div>
                </div>
                <div class="glass rounded-lg p-4 bg-green-500/10 border border-green-500/20">
                    <div class="text-xs text-gray-400 uppercase tracking-wider mb-2 font-semibold">RSI</div>
                    <div class="text-3xl font-bold text-green-400 mb-1">{rsi_value:.1f}</div>
                    <div class="text-sm text-gray-300 mt-1">{rsi_status}</div>
                </div>
                <div class="glass rounded-lg p-4 bg-blue-500/10 border border-blue-500/20">
                    <div class="text-xs text-gray-400 uppercase tracking-wider mb-2 font-semibold">Trend</div>
                    <div class="text-3xl font-bold {trend_color} mb-1">{trend_value}</div>
                    <div class="text-sm text-gray-300 mt-1">vs SMA-200</div>
                </div>
                <div class="glass rounded-lg p-4 bg-yellow-500/10 border border-yellow-500/20">
                    <div class="text-xs text-gray-400 uppercase tracking-wider mb-2 font-semibold">Risk</div>
                    <div class="text-2xl font-bold {risk_color} mb-1">{verdict_risk}</div>
                    <div class="text-sm text-gray-300 mt-1">Bounce-back confidence</div>
                </div>
            </div>
            
            <!-- Two Column Layout -->
            <div class="grid grid-cols-2 gap-6">
                <!-- Left Column: Calculations & Technical Details -->
                <div class="space-y-4">
                    <!-- Calculations -->
                    <div class="glass rounded-lg p-5 border-l-4 border-green-500/50">
                        <h3 class="text-base font-bold text-white mb-4 flex items-center gap-2">
                            <i class="fas fa-calculator text-green-400"></i>
                            Calculations
                        </h3>
                        <div class="grid grid-cols-2 gap-3">
                            <div class="glass rounded-lg p-3 bg-green-500/10">
                                <div class="text-xs text-gray-400 mb-1 font-medium">RSI</div>
                                <div class="text-xl font-bold text-green-400">{rsi_value:.2f}</div>
                                <div class="text-xs text-gray-400 mt-1">14-day</div>
                            </div>
                            <div class="glass rounded-lg p-3 bg-blue-500/10">
                                <div class="text-xs text-gray-400 mb-1 font-medium">SMA-200</div>
                                <div class="text-xl font-bold text-blue-400">${sma_value:.2f}</div>
                                <div class="text-xs text-gray-400 mt-1">200-day avg</div>
                            </div>
                            <div class="glass rounded-lg p-3 bg-purple-500/10">
                                <div class="text-xs text-gray-400 mb-1 font-medium">ATR</div>
                                <div class="text-xl font-bold text-purple-400">${atr_value:.2f}</div>
                                <div class="text-xs text-gray-400 mt-1">Volatility</div>
                            </div>
                            <div class="glass rounded-lg p-3 bg-yellow-500/10">
                                <div class="text-xs text-gray-400 mb-1 font-medium">Price</div>
                                <div class="text-xl font-bold text-white">${price_value:.2f}</div>
                                <div class="text-xs text-gray-400 mt-1">Current</div>
                            </div>
                        </div>
                        <!-- Collapsible Formula Details -->
                        <details class="mt-4">
                            <summary class="text-sm text-purple-400 cursor-pointer hover:text-purple-300 font-medium">Show Formulas</summary>
                            <div class="mt-3 space-y-2 text-sm text-gray-300 leading-relaxed">
                                <div><strong class="text-white">RSI:</strong> RSI = 100 - (100 / (1 + RS)) where RS = Average Gain / Average Loss over 14 periods</div>
                                <div><strong class="text-white">SMA-200:</strong> SMA-200 = Sum of closing prices over last 200 days / 200</div>
                                <div><strong class="text-white">ATR:</strong> ATR = Average of True Range over 14 periods, where True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)</div>
                                <div><strong class="text-white">Trend:</strong> Price (${price_value:.2f}) {'>' if trend_value == 'UP' else '<'} SMA-200 (${sma_value:.2f})</div>
                            </div>
                        </details>
                    </div>
                    
                    <!-- Inputs -->
                    <div class="glass rounded-lg p-5 border-l-4 border-blue-500/50">
                        <h3 class="text-base font-bold text-white mb-4 flex items-center gap-2">
                            <i class="fas fa-arrow-right text-blue-400"></i>
                            Inputs
                        </h3>
                        <div class="grid grid-cols-2 gap-3">
                            <div>
                                <div class="text-xs text-gray-400 mb-1 font-medium">Data Points</div>
                                <div class="text-lg font-semibold text-white">{bars_count}</div>
                            </div>
                            <div>
                                <div class="text-xs text-gray-400 mb-1 font-medium">News Count</div>
                                <div class="text-lg font-semibold text-white">{len(analysis_data.get('headlines', []))}</div>
                            </div>
                            <div>
                                <div class="text-xs text-gray-400 mb-1 font-medium">RSI Threshold</div>
                                <div class="text-lg font-semibold text-white">&lt; {config.get('rsi_threshold', 35)}</div>
                            </div>
                            <div>
                                <div class="text-xs text-gray-400 mb-1 font-medium">Score Required</div>
                                <div class="text-lg font-semibold text-white">≥ {config.get('ai_score_required', 7)}</div>
                            </div>
                        </div>
                    </div>
                    
                    {similar_signals_html}
                </div>
                
                <!-- Right Column: Outputs, Insights & News -->
                <div class="space-y-3">
                    <!-- Outputs & Reasoning -->
                    <div class="glass rounded-lg p-5 border-l-4 border-yellow-500/50">
                        <h3 class="text-base font-bold text-white mb-4 flex items-center gap-2">
                            <i class="fas fa-arrow-left text-yellow-400"></i>
                            Outputs & Reasoning
                        </h3>
                        <div class="mb-4">
                            <div class="text-xs text-gray-400 mb-2 font-medium">Adjusted Score</div>
                            <div class="text-2xl font-bold {adjusted_score_color}">
                                {adjusted_score:.2f}/10
                                {f'<span class="text-sm {"text-green-400" if confidence_boost > 0 else "text-red-400"} ml-2">({"+", "" if confidence_boost > 0 else ""}{confidence_boost:.1f})</span>' if confidence_boost != 0 else ''}
                            </div>
                        </div>
                        <div class="glass rounded-lg p-4 bg-gray-500/10 mb-4">
                            <div class="text-xs text-gray-400 mb-2 font-medium">AI Reasoning</div>
                            <div class="text-sm text-gray-200 leading-relaxed">{escape_html(verdict_reason)}</div>
                        </div>
                        <div class="flex items-center justify-between text-sm mb-4 pb-4 border-b border-gray-700/30">
                            <span class="text-gray-300 font-medium">Meets Criteria:</span>
                            <span class="badge {'badge-success' if meets_criteria else 'badge-danger'} text-sm px-3 py-1">
                                {'✓ Yes' if meets_criteria else '✗ No'}
                            </span>
                        </div>
                        <!-- Exit Levels: Buy Low, Sell High -->
                        <div class="glass rounded-lg p-4 bg-green-500/10 border-2 border-green-500/30">
                            <div class="text-sm text-gray-300 mb-3 font-semibold">If we buy here (Low Buy):</div>
                            <div class="space-y-3">
                                <div class="flex justify-between items-center">
                                    <span class="text-sm text-gray-300">Stop Loss (Risk Limit):</span>
                                    <span class="text-lg text-red-400 font-bold">${stop_loss_price:.2f}</span>
                                </div>
                                <div class="flex justify-between items-center">
                                    <span class="text-sm text-gray-300">Take Profit (Sell High):</span>
                                    <span class="text-lg text-green-400 font-bold">${take_profit_price:.2f}</span>
                                </div>
                                <div class="flex justify-between items-center pt-3 border-t border-gray-700/30">
                                    <span class="text-sm text-gray-300 font-medium">Risk/Reward:</span>
                                    <span class="text-xl text-blue-400 font-bold">{risk_reward_ratio}:1</span>
                                </div>
                                <div class="text-xs text-gray-400 mt-2 pt-2 border-t border-gray-700/20">For every $1 risked, potential ${risk_reward_ratio:.2f} reward</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Insights -->
                    <div class="glass rounded-lg p-5 border-l-4 border-purple-500/50">
                        <h3 class="text-base font-bold text-white mb-4 flex items-center gap-2">
                            <i class="fas fa-lightbulb text-purple-400"></i>
                            Insights
                        </h3>
                        <div class="glass rounded-lg p-4 bg-purple-500/10 mb-4">
                            <div class="text-xs text-gray-400 mb-2 font-medium">Historical Context</div>
                            <p class="text-sm text-gray-200 leading-relaxed">Similar low-buy signals have {win_rate:.1f}% win rate ({len(similar_signals_list)} signals found)</p>
                        </div>
                        <div class="space-y-3">
                            <div class="flex items-center justify-between py-2">
                                <span class="text-sm text-gray-300">Signal:</span>
                                <span class="text-sm font-semibold text-white">{'Signal detected' if techs.get('rsi', 50) < config.get('rsi_threshold', 35) else 'No signal'}</span>
                            </div>
                            <div class="flex items-center justify-between py-2">
                                <span class="text-sm text-gray-300">Historical:</span>
                                <span class="text-sm font-semibold text-white">{len(similar_signals_list)} signals{(' (' + str(int(win_rate)) + '% win)') if win_rate > 0 else ''}</span>
                            </div>
                            <div class="flex items-center justify-between py-2">
                                <span class="text-sm text-gray-300">AI Validation:</span>
                                <span class="text-sm font-semibold text-white">Score {verdict_score}/10 indicates {escape_html(verdict_action)}</span>
                            </div>
                        </div>
                    </div>
                    
                    {news_articles_html}
                </div>
            </div>
        """
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Failed to get explanation for {symbol}: {e}", exc_info=True)
        error_html = f"""
        <div class="glass rounded-lg p-4 border border-red-500/30">
            <div class="flex items-center gap-2 text-red-400 mb-2">
                <i class="fas fa-exclamation-circle"></i>
                <span class="font-semibold">Error</span>
            </div>
            <p class="text-sm text-gray-300">{str(e)[:200]}</p>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=500)

