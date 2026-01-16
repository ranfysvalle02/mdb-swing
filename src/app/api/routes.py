"""API route handlers."""
from fastapi import Depends, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime
import json
import asyncio
from mdb_engine.dependencies import get_scoped_db
from mdb_engine.observability import get_logger
from ..services.trading import place_order, run_backtest_simulation
from ..services.analysis import api
from ..services.analysis import get_market_data, analyze_technicals
from ..services.ai import eye
from ..core.config import ALPACA_KEY, ALPACA_SECRET, STRATEGY_CONFIG, STRATEGY_PRESETS, FIRECRAWL_API_KEY
from ..services.analysis import firecrawl_client

logger = get_logger(__name__)

def check_alpaca_config() -> bool:
    """Check if Alpaca API is properly configured."""
    if not ALPACA_KEY or not ALPACA_SECRET:
        logger.warning("Alpaca API credentials not configured")
        return False
    return True

async def get_balance() -> HTMLResponse:
    """Get account balance."""
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
    bars, headlines = get_market_data(symbol, days=500)  # Request 500 days minimum
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
    
    if eye:
        verdict = eye.analyze(symbol, techs['price'], techs['rsi'], techs['atr'], techs['trend'], "\n".join(headlines))
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
                            <span>Current Strategy: <strong>{STRATEGY_CONFIG['name']}</strong></span>
                        </span>
                        <span>RSI &lt; {STRATEGY_CONFIG['rsi_threshold']}</span>
                        <span>Score ≥ {STRATEGY_CONFIG['ai_score_required']}</span>
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
                        <div class="font-bold text-yellow-400 mb-1">The Eye is Blind</div>
                        <p class="text-sm text-gray-300">Azure OpenAI configuration missing</p>
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

async def get_positions() -> HTMLResponse:
    """Get current positions."""
    if not check_alpaca_config():
        return HTMLResponse(content="<div class='text-yellow-500 text-sm'>Alpaca API not configured</div>")
    
    if api is None:
        return HTMLResponse(content="<div class='text-red-500 text-sm'>Alpaca API not initialized</div>")
    
    try:
        pos = api.list_positions()
        if not pos:
            return HTMLResponse(content="""
                <div class="text-center py-6 text-gray-500">
                    <i class="fas fa-wallet text-3xl mb-2 opacity-50"></i>
                    <p class="text-sm">Cash Gang 💰</p>
                    <p class="text-xs mt-1 text-gray-600">No open positions</p>
                </div>
            """)
        
        html = ""
        for p in pos:
            pl = float(p.unrealized_pl)
            pl_pct = (pl / (float(p.market_value) - pl)) * 100 if float(p.market_value) != pl else 0
            color = "text-green-400" if pl > 0 else "text-red-400"
            badge_color = "badge-success" if pl > 0 else "badge-danger"
            market_value = float(p.market_value)
            html += f"""
            <div class="glass rounded-lg p-3 mb-2 card-hover border border-gray-700/50">
                <div class="flex justify-between items-center mb-2">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center border border-purple-500/30">
                            <span class="font-bold text-white text-sm">{p.symbol}</span>
                        </div>
                        <div>
                            <div class="font-bold text-white">{p.symbol}</div>
                            <div class="text-xs text-gray-400">{p.qty} shares @ ${market_value / float(p.qty):.2f}</div>
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="font-bold {color}">${pl:.2f}</div>
                        <div class="text-xs {color}">{pl_pct:+.2f}%</div>
                    </div>
                </div>
                <div class="flex gap-2 pt-2 border-t border-gray-700/30">
                    <button onclick="quickSell('{p.symbol}', event)" 
                            class="flex-1 glass-strong py-1.5 rounded-lg text-xs font-bold text-white hover:bg-red-600/30 transition-all bg-red-500/30 flex items-center justify-center gap-1 border border-red-500/50">
                        <i class="fas fa-times text-xs"></i>
                        <span>CLOSE</span>
                    </button>
                    <button onclick="quickScan('{p.symbol}')" 
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
                    <div class="flex items-center gap-2 max-w-[200px]">
                        <i class="fas fa-quote-left text-gray-600 text-xs"></i>
                        <span class="text-gray-300 text-xs truncate" title="{t['reason']}">{t['reason']}</span>
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
    """Get current strategy configuration."""
    preset_color = STRATEGY_CONFIG.get('color', 'yellow')
    color_classes = {
        'blue': 'border-blue-500/30 bg-blue-500/10',
        'yellow': 'border-yellow-500/30 bg-yellow-500/10',
        'red': 'border-red-500/30 bg-red-500/10'
    }
    border_class = color_classes.get(preset_color, 'border-yellow-500/30')
    
    return HTMLResponse(content=f"""
        <div class="glass rounded-xl p-4 border {border_class}">
            <div class="flex items-center justify-between mb-3">
                <div>
                    <div class="text-xs text-gray-400 mb-1 uppercase tracking-wider">Active Strategy</div>
                    <div class="text-lg font-bold text-white">{STRATEGY_CONFIG['name']}</div>
                </div>
                <button onclick="openModal('strategy-modal')" class="text-xs text-gray-400 hover:text-white transition">
                    <i class="fas fa-edit"></i>
                </button>
            </div>
            <p class="text-xs text-gray-400 mb-3">{STRATEGY_CONFIG['description']}</p>
            <div class="grid grid-cols-2 gap-3 text-xs">
                <div class="glass rounded-lg p-2">
                    <div class="text-gray-400 mb-1">RSI Threshold</div>
                    <div class="text-white font-bold">&lt; {STRATEGY_CONFIG['rsi_threshold']}</div>
                </div>
                <div class="glass rounded-lg p-2">
                    <div class="text-gray-400 mb-1">AI Score Required</div>
                    <div class="text-white font-bold">≥ {STRATEGY_CONFIG['ai_score_required']}</div>
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

async def get_pending_trades(db = Depends(get_scoped_db)) -> HTMLResponse:
    """Get pending trades awaiting approval."""
    try:
        pending = await db.pending_trades.find({"status": "pending"}).sort("timestamp", -1).limit(10).to_list(length=10)
        if not pending:
            return HTMLResponse(content="""
                <div class="text-center py-8 text-gray-500">
                    <i class="fas fa-check-circle text-4xl mb-2 opacity-50"></i>
                    <p class="text-sm">No pending trades</p>
                    <p class="text-xs mt-1 text-gray-600">The Eye is watching...</p>
                </div>
            """)
        
        html = ""
        for trade in pending:
            ts = trade['timestamp'].strftime('%H:%M')
            score = trade['score']
            score_color = "text-green-400" if score >= 8 else "text-yellow-400"
            risk_level = trade.get('risk_level', 'N/A').upper()
            risk_badge = "badge-success" if risk_level == "LOW" else "badge-warning" if risk_level == "MEDIUM" else "badge-danger"
            
            html += f"""
            <div class="glass rounded-xl p-4 mb-3 border border-orange-500/20 card-hover fade-in">
                <div class="flex justify-between items-start mb-3">
                    <div class="flex items-center gap-3">
                        <div class="w-12 h-12 rounded-lg bg-gradient-to-br from-orange-500/20 to-red-500/20 flex items-center justify-center">
                            <span class="font-bold text-white text-lg">{trade['symbol']}</span>
                        </div>
                        <div>
                            <div class="flex items-center gap-2">
                                <span class="font-bold text-white">{trade['symbol']}</span>
                                <span class="badge {risk_badge}">{risk_level}</span>
                                <button onclick="openModal('risk-levels-modal')" class="text-gray-500 hover:text-yellow-400 transition cursor-pointer tooltip relative group ml-1" title="Learn about risk levels">
                                    <i class="fas fa-info-circle text-xs"></i>
                                    <div class="tooltip-text">Learn about risk levels</div>
                                </button>
                            </div>
                            <div class="text-xs text-gray-400 mt-1">
                                <i class="fas fa-clock mr-1"></i>{ts}
                            </div>
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="flex items-center justify-end gap-2">
                            <div class="text-2xl font-bold {score_color}">{score}/10</div>
                            <button onclick="openModal('ai-score-modal')" class="text-gray-500 hover:text-purple-400 transition cursor-pointer tooltip relative group" title="Learn about AI scores">
                                <i class="fas fa-info-circle text-xs"></i>
                                <div class="tooltip-text">Learn how AI scores work</div>
                            </button>
                        </div>
                        <div class="text-xs text-gray-400">AI Score</div>
                    </div>
                </div>
                
                <div class="grid grid-cols-3 gap-2 mb-3">
                    <div class="glass rounded-lg p-2 text-center">
                        <div class="text-xs text-gray-400 mb-1">Price</div>
                        <div class="font-bold text-white">${trade['price']:.2f}</div>
                    </div>
                    <div class="glass rounded-lg p-2 text-center">
                        <div class="text-xs text-gray-400 mb-1">RSI</div>
                        <div class="font-bold text-white">{trade['rsi']:.1f}</div>
                    </div>
                    <div class="glass rounded-lg p-2 text-center">
                        <div class="text-xs text-gray-400 mb-1">Trend</div>
                        <div class="font-bold {'text-green-400' if trade['trend'] == 'UP' else 'text-red-400'}">{trade['trend']}</div>
                    </div>
                </div>
                
                <div class="glass rounded-lg p-2 mb-3 border border-gray-700/50">
                    <p class="text-xs text-gray-300 line-clamp-2">"{trade['reason'][:120]}"</p>
                </div>
                
                <div class="flex gap-2">
                    <button 
                        hx-post="/api/approve-trade" 
                        hx-vals='{{"trade_id": "{str(trade["_id"])}"}}'
                        hx-target="#pending-trades"
                        hx-swap="innerHTML"
                        class="flex-1 glass-strong py-2 rounded-lg font-bold text-white hover:bg-green-600/20 transition-all glow-green flex items-center justify-center gap-2">
                        <i class="fas fa-check"></i>
                        APPROVE
                    </button>
                    <button 
                        hx-post="/api/reject-trade" 
                        hx-vals='{{"trade_id": "{str(trade["_id"])}"}}'
                        hx-target="#pending-trades"
                        hx-swap="innerHTML"
                        class="flex-1 glass-strong py-2 rounded-lg font-bold text-white hover:bg-red-600/20 transition-all flex items-center justify-center gap-2">
                        <i class="fas fa-times"></i>
                        REJECT
                    </button>
                </div>
            </div>
            """
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Failed to get pending trades: {e}", exc_info=True)
        return HTMLResponse(content="<div class='text-red-500 text-sm'>Error loading pending trades</div>")

async def approve_trade(trade_id: str = Form(...), db = Depends(get_scoped_db)) -> HTMLResponse:
    """Approve and execute a pending trade."""
    try:
        from bson import ObjectId
        trade = await db.pending_trades.find_one({"_id": ObjectId(trade_id)})
        if not trade:
            return HTMLResponse(content="<span class='text-red-500'>Trade not found</span>")
        
        # Execute the trade
        order_result = await place_order(
            trade['symbol'], 
            trade['action'], 
            trade['reason'], 
            trade['score'], 
            db
        )
        
        if order_result:
            # Update status to approved
            await db.pending_trades.update_one(
                {"_id": ObjectId(trade_id)},
                {"$set": {"status": "approved", "executed_at": datetime.now()}}
            )
            logger.info(f"✅ Trade approved and executed: {trade['symbol']}")
            # Return empty to trigger refresh, but show a brief success message
            return HTMLResponse(content="")  # Empty to trigger refresh
        else:
            await db.pending_trades.update_one(
                {"_id": ObjectId(trade_id)},
                {"$set": {"status": "failed", "error": "Order placement failed"}}
            )
            return HTMLResponse(content="""
                <div class="glass rounded-lg p-3 border border-red-500/30 flex items-center gap-2 text-red-400 mb-2">
                    <i class="fas fa-times-circle"></i>
                    <span class="font-semibold">Order Failed</span>
                </div>
            """)
    except Exception as e:
        logger.error(f"Failed to approve trade: {e}", exc_info=True)
        return HTMLResponse(content=f"<span class='text-red-500'>Error: {str(e)[:50]}</span>")

async def reject_trade(trade_id: str = Form(...), db = Depends(get_scoped_db)) -> HTMLResponse:
    """Reject a pending trade."""
    try:
        from bson import ObjectId
        await db.pending_trades.update_one(
            {"_id": ObjectId(trade_id)},
            {"$set": {"status": "rejected", "rejected_at": datetime.now()}}
        )
        logger.info(f"❌ Trade rejected: {trade_id}")
        return HTMLResponse(content="")  # Empty to trigger refresh
    except Exception as e:
        logger.error(f"Failed to reject trade: {e}", exc_info=True)
        return HTMLResponse(content=f"<span class='text-red-500'>Error: {str(e)[:50]}</span>")

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
    
    # Get current watchlist to show which stocks are already added
    try:
        watchlist_doc = await db.watchlist.find_one({})
        watchlist_symbols = set(watchlist_doc.get('symbols', []) if watchlist_doc else [])
    except Exception as e:
        logger.warning(f"Could not fetch watchlist: {e}")
        watchlist_symbols = set()
    
    html = ""
    for category_name, category_data in categories.items():
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
            is_in_watchlist = symbol in watchlist_symbols
            button_class = "bg-gray-600/30 cursor-not-allowed" if is_in_watchlist else "bg-green-500/20 hover:bg-green-500/30 cursor-pointer"
            button_text = '<i class="fas fa-check mr-1"></i>In Watchlist' if is_in_watchlist else '<i class="fas fa-plus mr-1"></i>Add to Watchlist'
            
            disabled_attr = 'disabled' if is_in_watchlist else ''
            html += f"""
                <div class="stock-card glass rounded-lg p-3 border border-gray-700/50 hover:border-green-500/30 transition" 
                     data-symbol="{symbol}" data-name="{name}">
                    <div class="flex items-center justify-between">
                        <div>
                            <div class="font-bold text-white">{symbol}</div>
                            <div class="text-xs text-gray-400">{name}</div>
                        </div>
                        <button onclick="addToWatchlist('{symbol}', '{name}', event)" 
                                class="text-xs px-3 py-1 rounded {button_class} text-white transition"
                                {disabled_attr}>
                            {button_text}
                        </button>
                    </div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    return HTMLResponse(content=html)

async def get_watchlist(db = Depends(get_scoped_db)) -> HTMLResponse:
    """Get current watchlist."""
    try:
        watchlist_doc = await db.watchlist.find_one({})
        if not watchlist_doc or not watchlist_doc.get('symbols'):
            return HTMLResponse(content="""
                <div class="text-center py-4 text-gray-500">
                    <i class="fas fa-eye-slash text-2xl mb-2 opacity-50"></i>
                    <p class="text-xs">No stocks in watchlist yet</p>
                    <p class="text-[10px] mt-1 text-gray-600">Click "Add Stocks" to get started</p>
                </div>
            """)
        
        symbols = watchlist_doc.get('symbols', [])
        html = ""
        for symbol in symbols:
            html += f"""
            <div class="glass rounded-lg p-2 flex items-center justify-between border border-gray-700/50 hover:border-green-500/30 transition">
                <div class="flex items-center gap-2">
                    <div class="w-8 h-8 rounded bg-green-500/20 flex items-center justify-center">
                        <span class="font-bold text-white text-xs">{symbol}</span>
                    </div>
                </div>
                <button onclick="removeFromWatchlist('{symbol}')" 
                        class="text-xs px-2 py-1 rounded bg-red-500/20 hover:bg-red-500/30 text-red-400 transition">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            """
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Failed to get watchlist: {e}", exc_info=True)
        return HTMLResponse(content="<div class='text-red-500 text-xs'>Error loading watchlist</div>")

async def add_to_watchlist(symbol: str = Form(...), name: str = Form(""), db = Depends(get_scoped_db)) -> HTMLResponse:
    """Add a stock to the watchlist."""
    symbol = symbol.upper().strip()
    if not symbol:
        return HTMLResponse(content="<span class='text-red-500'>Invalid symbol</span>")
    
    try:
        watchlist_doc = await db.watchlist.find_one({})
        if not watchlist_doc:
            # Create new watchlist
            await db.watchlist.insert_one({
                "symbols": [symbol],
                "updated_at": datetime.now()
            })
        else:
            symbols = watchlist_doc.get('symbols', [])
            if symbol not in symbols:
                symbols.append(symbol)
                await db.watchlist.update_one(
                    {},
                    {"$set": {"symbols": symbols, "updated_at": datetime.now()}}
                )
        
        logger.info(f"Added {symbol} to watchlist")
        # Return updated watchlist HTML
        return await get_watchlist(db)
    except Exception as e:
        logger.error(f"Failed to add to watchlist: {e}", exc_info=True)
        return HTMLResponse(content="<div class='text-red-500 text-xs'>Error adding stock</div>")

async def remove_from_watchlist(symbol: str = Form(...), db = Depends(get_scoped_db)) -> HTMLResponse:
    """Remove a stock from the watchlist."""
    symbol = symbol.upper().strip()
    
    try:
        watchlist_doc = await db.watchlist.find_one({})
        if watchlist_doc:
            symbols = watchlist_doc.get('symbols', [])
            if symbol in symbols:
                symbols.remove(symbol)
                await db.watchlist.update_one(
                    {},
                    {"$set": {"symbols": symbols, "updated_at": datetime.now()}}
                )
                logger.info(f"Removed {symbol} from watchlist")
        
        # Return updated watchlist HTML
        return await get_watchlist(db)
    except Exception as e:
        logger.error(f"Failed to remove from watchlist: {e}", exc_info=True)
        return HTMLResponse(content="<div class='text-red-500 text-xs'>Error removing stock</div>")

async def _search_trending_stocks_internal() -> dict:
    """Internal function to search trending stocks - returns dict."""
    if not firecrawl_client:
        # Fallback to curated list if Firecrawl not available
        fallback_stocks = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA"]
        return {
            "success": True,
            "stocks": fallback_stocks,
            "source": "fallback"
        }
    
    try:
        import re
        import json
        
        # Search for trending stocks
        search_queries = [
            "trending stocks today 2024",
            "most active stocks today",
            "popular tech stocks"
        ]
        
        discovered_symbols = set()
        
        for query in search_queries:
            try:
                response = firecrawl_client.search(
                    query=query,
                    limit=10,
                    sources=["web", "news"],
                    scrapeOptions={
                        "formats": [{"type": "markdown"}]
                    }
                )
                
                # Extract stock symbols from search results
                # Handle different response formats
                if isinstance(response, dict):
                    data = response
                elif hasattr(response, 'data'):
                    data = response.data
                elif hasattr(response, 'json'):
                    data = response.json()
                else:
                    data = response
                
                # Check web results
                web_results = data.get('data', {}).get('web', []) if isinstance(data, dict) and 'data' in data else data.get('web', []) if isinstance(data, dict) else []
                news_results = data.get('data', {}).get('news', []) if isinstance(data, dict) and 'data' in data else data.get('news', []) if isinstance(data, dict) else []
                
                all_text = ""
                for result in web_results + news_results:
                    title = result.get('title', '')
                    description = result.get('description', '')
                    markdown = result.get('markdown', '')
                    all_text += f"{title} {description} {markdown} "
                
                # Extract stock symbols (3-5 letter uppercase codes)
                # Common patterns: "AAPL", "NVDA stock", "$TSLA", etc.
                symbol_pattern = r'\b([A-Z]{2,5})\b'
                potential_symbols = re.findall(symbol_pattern, all_text.upper())
                
                # Filter to common stock symbols (exclude common words)
                common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'WAY', 'USE', 'HER', 'SHE', 'MAN', 'HAS', 'HAD', 'ITS', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'WAY', 'USE'}
                
                # Common stock symbols to prioritize
                popular_stocks = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'NFLX', 'DIS', 'JPM', 'V', 'MA', 'PG', 'JNJ', 'WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'BA', 'CAT', 'GE', 'IBM', 'XOM', 'CVX', 'COP', 'SLB'}
                
                for symbol in potential_symbols:
                    if len(symbol) >= 2 and len(symbol) <= 5:
                        if symbol in popular_stocks or (symbol not in common_words and len(symbol) >= 3):
                            discovered_symbols.add(symbol)
                            if len(discovered_symbols) >= 5:  # Get at least 5 stocks
                                break
                
                if len(discovered_symbols) >= 5:
                    break
                        
            except Exception as e:
                logger.warning(f"Firecrawl search failed for query '{query}': {e}")
                continue
        
        # Fallback if we didn't find enough
        if len(discovered_symbols) < 3:
            discovered_symbols.update(["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA"])
        
        # Return top 5-7 stocks
        result_stocks = list(discovered_symbols)[:7]
        
        return {
            "success": True,
            "stocks": result_stocks,
            "source": "firecrawl"
        }
        
    except Exception as e:
        logger.error(f"Failed to search trending stocks: {e}", exc_info=True)
        # Fallback to curated list
        fallback_stocks = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA"]
        return {
            "success": True,
            "stocks": fallback_stocks,
            "source": "fallback"
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
                bars, headlines = get_market_data(symbol, days=500)
                if bars is None or bars.empty or len(bars) < 14:
                    continue
                
                techs = analyze_technicals(bars)
                
                # Get AI analysis if available
                ai_score = None
                ai_reason = None
                risk_level = "UNKNOWN"
                
                if eye:
                    try:
                        verdict = eye.analyze(
                            symbol, 
                            techs['price'], 
                            techs['rsi'], 
                            techs['atr'], 
                            techs['trend'], 
                            "\n".join(headlines[:3]) if headlines else "No recent news"
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

async def quick_buy(symbol: str = Form(...), db = Depends(get_scoped_db)) -> JSONResponse:
    """Quick buy order from stock card."""
    if not check_alpaca_config() or api is None:
        return JSONResponse(content={"success": False, "error": "Alpaca API not configured"})
    
    try:
        order = await place_order(symbol.upper(), 'buy', f"Quick buy from card", 10, db)
        if order:
            return JSONResponse(content={"success": True, "message": f"Buy order placed for {symbol}"})
        return JSONResponse(content={"success": False, "error": "Order failed - check logs"})
    except Exception as e:
        logger.error(f"Quick buy failed: {e}", exc_info=True)
        return JSONResponse(content={"success": False, "error": str(e)[:100]})

async def quick_sell(symbol: str = Form(...)) -> JSONResponse:
    """Quick sell/close position."""
    if not check_alpaca_config() or api is None:
        return JSONResponse(content={"success": False, "error": "Alpaca API not configured"})
    
    try:
        api.close_position(symbol.upper())
        return JSONResponse(content={"success": True, "message": f"Position closed for {symbol}"})
    except Exception as e:
        error_msg = str(e)
        if "position does not exist" in error_msg.lower():
            return JSONResponse(content={"success": False, "error": "No position to close"})
        logger.error(f"Quick sell failed: {e}", exc_info=True)
        return JSONResponse(content={"success": False, "error": str(e)[:100]})

async def cancel_orders(symbol: str = Form(None)) -> JSONResponse:
    """Cancel orders - all or for specific symbol."""
    if not check_alpaca_config() or api is None:
        return JSONResponse(content={"success": False, "error": "Alpaca API not configured"})
    
    try:
        if symbol:
            # Cancel orders for specific symbol
            orders = api.list_orders(status='open', symbols=[symbol.upper()])
            for order in orders:
                api.cancel_order(order.id)
            return JSONResponse(content={"success": True, "message": f"Cancelled orders for {symbol}"})
        else:
            # Cancel all orders
            api.cancel_all_orders()
            return JSONResponse(content={"success": True, "message": "All orders cancelled"})
    except Exception as e:
        logger.error(f"Cancel orders failed: {e}", exc_info=True)
        return JSONResponse(content={"success": False, "error": str(e)[:100]})

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

async def _get_trending_stocks_with_analysis_streaming(websocket: WebSocket) -> None:
    """Stream trending stocks analysis with progress updates via WebSocket."""
    await websocket.accept()
    
    try:
        # Get trending stocks
        search_data = await _search_trending_stocks_internal()
        
        if not search_data.get('success') or not search_data.get('stocks'):
            symbols = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "AMD"]
        else:
            symbols = search_data['stocks'][:6]  # Top 6 trending stocks
        
        total = len(symbols)
        stocks_with_analysis = []
        
        # Send initial progress
        await websocket.send_json({
            "type": "progress",
            "message": f"Discovering trending stocks...",
            "current": 0,
            "total": total,
            "percentage": 0
        })
        
        await asyncio.sleep(0.1)  # Small delay for UI update
        
        # Analyze each stock with progress updates
        for idx, symbol in enumerate(symbols, 1):
            try:
                # Send progress update
                await websocket.send_json({
                    "type": "progress",
                    "message": f"Analyzing {symbol}...",
                    "current": idx,
                    "total": total,
                    "percentage": int((idx / total) * 100),
                    "symbol": symbol
                })
                
                bars, headlines = get_market_data(symbol, days=500)
                if bars is None or bars.empty or len(bars) < 14:
                    await websocket.send_json({
                        "type": "progress",
                        "message": f"Skipping {symbol} - insufficient data",
                        "current": idx,
                        "total": total,
                        "percentage": int((idx / total) * 100),
                        "symbol": symbol,
                        "skipped": True
                    })
                    continue
                
                # Send technical analysis progress
                await websocket.send_json({
                    "type": "progress",
                    "message": f"Calculating indicators for {symbol}...",
                    "current": idx,
                    "total": total,
                    "percentage": int((idx / total) * 100),
                    "symbol": symbol,
                    "stage": "technicals"
                })
                
                techs = analyze_technicals(bars)
                
                # Send AI analysis progress
                await websocket.send_json({
                    "type": "progress",
                    "message": f"AI analyzing {symbol}...",
                    "current": idx,
                    "total": total,
                    "percentage": int((idx / total) * 100),
                    "symbol": symbol,
                    "stage": "ai"
                })
                
                # Get AI analysis if available
                ai_score = None
                ai_reason = None
                risk_level = "UNKNOWN"
                
                if eye:
                    try:
                        verdict = eye.analyze(
                            symbol, 
                            techs['price'], 
                            techs['rsi'], 
                            techs['atr'], 
                            techs['trend'], 
                            "\n".join(headlines[:3]) if headlines else "No recent news"
                        )
                        ai_score = verdict.score
                        ai_reason = verdict.reason[:100] + "..." if len(verdict.reason) > 100 else verdict.reason
                        risk_level = verdict.risk_level
                    except Exception as e:
                        logger.warning(f"AI analysis failed for {symbol}: {e}")
                
                stock_data = {
                    "symbol": symbol,
                    "price": techs['price'],
                    "rsi": techs['rsi'],
                    "atr": techs['atr'],
                    "trend": techs['trend'],
                    "sma": techs['sma'],
                    "ai_score": ai_score,
                    "ai_reason": ai_reason,
                    "risk_level": risk_level
                }
                
                stocks_with_analysis.append(stock_data)
                
                # Send completed stock
                await websocket.send_json({
                    "type": "stock_complete",
                    "stock": stock_data,
                    "current": idx,
                    "total": total,
                    "percentage": int((idx / total) * 100)
                })
                
            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "rate limit" in error_msg.lower() or "rate limit exceeded" in error_msg.lower()
                
                logger.warning(f"Failed to analyze {symbol}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error analyzing {symbol}: {str(e)[:50]}",
                    "symbol": symbol,
                    "current": idx,
                    "total": total,
                    "is_rate_limit": is_rate_limit,
                    "error_type": "rate_limit" if is_rate_limit else "general"
                })
                continue
        
        # Send completion with warnings if needed
        warnings = []
        if len(stocks_with_analysis) < len(symbols):
            warnings.append(f"Only {len(stocks_with_analysis)} of {len(symbols)} stocks analyzed successfully")
        
        await websocket.send_json({
            "type": "complete",
            "stocks": stocks_with_analysis,
            "total": len(stocks_with_analysis),
            "expected": len(symbols),
            "source": search_data.get('source', 'fallback'),
            "warnings": warnings
        })
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Analysis failed: {str(e)[:100]}"
            })
        except:
            pass

async def auto_setup_watchlist(db = Depends(get_scoped_db)) -> JSONResponse:
    """Check if watchlist exists and return trending stocks for 'What's New' section."""
    try:
        # Check if watchlist already exists
        watchlist_doc = await db.watchlist.find_one({})
        has_watchlist = watchlist_doc and watchlist_doc.get('symbols')
        
        # Get trending stocks with full analysis (use internal function)
        analysis_data = await _get_trending_stocks_with_analysis_internal()
        
        if analysis_data.get('success') and analysis_data.get('stocks'):
            return JSONResponse(content={
                "success": True,
                "already_setup": has_watchlist,
                "stocks": analysis_data['stocks'],
                "source": analysis_data.get('source', 'unknown')
            })
        else:
            # Fallback - try to get analysis for fallback stocks
            fallback_symbols = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "AMD"]
            fallback_with_analysis = []
            for symbol in fallback_symbols[:3]:  # Analyze first 3 to avoid timeout
                try:
                    bars, headlines = get_market_data(symbol, days=500)
                    if bars is not None and not bars.empty and len(bars) >= 14:
                        techs = analyze_technicals(bars)
                        ai_score = None
                        ai_reason = None
                        if eye:
                            try:
                                verdict = eye.analyze(symbol, techs['price'], techs['rsi'], techs['atr'], techs['trend'], "\n".join(headlines[:3]) if headlines else "")
                                ai_score = verdict.score
                                ai_reason = verdict.reason[:100] + "..." if len(verdict.reason) > 100 else verdict.reason
                            except:
                                pass
                        fallback_with_analysis.append({
                            "symbol": symbol,
                            "price": techs['price'],
                            "rsi": techs['rsi'],
                            "atr": techs['atr'],
                            "trend": techs['trend'],
                            "sma": techs['sma'],
                            "ai_score": ai_score,
                            "ai_reason": ai_reason
                        })
                except:
                    fallback_with_analysis.append({"symbol": symbol})
            
            # Add remaining without analysis
            for symbol in fallback_symbols[3:]:
                fallback_with_analysis.append({"symbol": symbol})
            
            return JSONResponse(content={
                "success": True,
                "already_setup": has_watchlist,
                "stocks": fallback_with_analysis,
                "source": "fallback"
            })
            
    except Exception as e:
        logger.error(f"Failed to get trending stocks: {e}", exc_info=True)
        fallback_stocks = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "AMD"]
        return JSONResponse(content={
            "success": True,
            "already_setup": False,
            "stocks": [{"symbol": s} for s in fallback_stocks],
            "source": "fallback"
        })
