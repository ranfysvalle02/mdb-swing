import time
import os
import math
import requests
import logging
import io
import base64
from datetime import datetime

# --- DATA & MATH ---
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for server
import matplotlib.pyplot as plt

# --- TRADING ---
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

# --- WEB & SCHEDULING ---
from flask import Flask, request, jsonify, render_template_string
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient

# --- AI & LOGIC ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Sentinel_Commander")

# Environment Variables
SYMBOLS_ENV = os.getenv("TARGET_SYMBOLS", "NVDA,AMD,MSFT,GOOGL,AMZN,TSLA,COIN,AAPL")
SYMBOLS = [s.strip() for s in SYMBOLS_ENV.split(',')]
MAX_CAPITAL_DEPLOYED = float(os.getenv("MAX_CAPITAL_DEPLOYED", "5000.00"))

# ALPACA KEYS
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version='v2')

# MONGO CONNECTION
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
try:
    client = MongoClient(MONGO_URI)
    db = client['sentinel_pro']
    history_col = db['history']
except Exception as e:
    logger.critical(f"DB Fail: {e}")
    # Fallback for no DB (avoids crash, creates in-memory list)
    class MockCol:
        def __init__(self): self.data = []
        def insert_one(self, d): self.data.append(d)
        def find(self): 
            # Mock cursor
            class Cursor:
                def sort(self, *args): return self
                def limit(self, *args): return mock_col.data[-10:]
            return Cursor()
    history_col = MockCol()
    logger.warning("Running without Database (In-Memory Only)")

app = Flask(__name__)

# ==========================================
# 1. AI AGENT DEFINITIONS
# ==========================================
class TradeVerdict(BaseModel):
    score: int = Field(description="Bullish score 0-10.")
    action: str = Field(description="One of: 'BUY', 'WAIT', 'SELL_NOW'.")
    reason: str = Field(description="Concise strategic reasoning (max 20 words).")
    risk_level: str = Field(description="Low, Medium, or High based on volatility.")

class SentinelBrain:
    def __init__(self):
        # Ensure OPENAI_API_KEY is set in env
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2) 
        
    def analyze(self, ticker, price, rsi, atr, trend, headlines):
        system_prompt = (
            "You are a Senior Quant Trader. Analyze Technicals + Sentiment. "
            "Technicals: RSI < 30 is Oversold (Buy zone), > 70 Overbought. "
            "ATR indicates volatility. If news is catastrophic (fraud/bankruptcy), VETO the trade (Score 0)."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", (
                f"Ticker: {ticker} | Price: ${price}\n"
                f"Technicals: RSI={rsi}, ATR={atr}, Trend={trend}\n"
                f"News:\n{headlines}"
            ))
        ])
        chain = prompt | self.llm.with_structured_output(TradeVerdict)
        return chain.invoke({})

# Initialize AI
try:
    brain = SentinelBrain()
except:
    logger.warning("OpenAI Key missing. AI features will fail.")
    brain = None

# ==========================================
# 2. MARKET DATA & MATH
# ==========================================
def get_market_data(symbol, days=100):
    try:
        # Fetch data (Day candles)
        bars = api.get_bars(symbol, TimeFrame.Day, limit=days).df
        if bars.empty: return None, []
        if isinstance(bars.index, pd.MultiIndex): bars = bars.reset_index()
        
        # Fetch News
        news = api.get_news(symbol=symbol, limit=3)
        headlines = [f"- {n.headline}" for n in news] if news else ["No recent news."]
        
        return bars, headlines
    except Exception as e:
        logger.error(f"Data fail {symbol}: {e}")
        return None, []

def analyze_technicals(df):
    """Calculates RSI, SMA, and ATR."""
    df.columns = [c.lower() for c in df.columns]
    
    # Pandas-TA
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['sma_200'] = ta.sma(df['close'], length=200)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    current = df.iloc[-1]
    
    # Determine Trend (Simple: Price > 200 SMA)
    sma_val = current['sma_200'] if not math.isnan(current['sma_200']) else 0
    trend = "UP" if current['close'] > sma_val else "DOWN"
    
    return {
        "price": round(current['close'], 2),
        "rsi": round(current['rsi'], 2) if not math.isnan(current['rsi']) else 50,
        "atr": round(current['atr'], 2) if not math.isnan(current['atr']) else 1,
        "trend": trend,
        "sma": round(sma_val, 2)
    }

# ==========================================
# 3. EXECUTION LOGIC
# ==========================================
def calculate_position_size(price, atr, risk_per_trade_dollars=50.0):
    """Volatility Sizing: High ATR = Fewer Shares."""
    stop_distance = 2 * atr 
    if stop_distance == 0: return 0
    
    shares_risk = math.floor(risk_per_trade_dollars / stop_distance)
    
    # Cap at budget
    try:
        acct = api.get_account()
        buying_power = float(acct.buying_power)
    except:
        buying_power = 10000.0 # Fallback
        
    max_shares_budget = math.floor(min(buying_power, MAX_CAPITAL_DEPLOYED) / price)
    
    return int(min(shares_risk, max_shares_budget))

def place_order(symbol, side, reason, score):
    df, _ = get_market_data(symbol)
    if df is None: return False
    techs = analyze_technicals(df)
    
    qty = calculate_position_size(techs['price'], techs['atr'])
    if qty < 1: 
        logger.info(f"Skipping {symbol}: Position size 0 (Risk/Budget)")
        return False

    # Dynamic Bracket Orders
    limit_price = techs['price']
    stop_price = round(techs['price'] - (2 * techs['atr']), 2) # 2x ATR Stop
    take_profit = round(techs['price'] + (3 * techs['atr']), 2) # 3x ATR Target
    
    try:
        order = api.submit_order(
            symbol=symbol, qty=qty, side=side, type='limit', limit_price=limit_price,
            time_in_force='day', order_class='bracket',
            take_profit={'limit_price': take_profit},
            stop_loss={'stop_price': stop_price}
        )
        
        # Log to DB
        history_col.insert_one({
            "timestamp": datetime.now(), "symbol": symbol, "action": side,
            "qty": qty, "price": limit_price, "reason": reason, "score": score
        })
        return order
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return None

# ==========================================
# 4. BACKTESTING ENGINE
# ==========================================
def run_backtest_simulation(symbol, days=365):
    try:
        # Fetch extra data for SMA calculation
        limit = days + 250
        bars = api.get_bars(symbol, TimeFrame.Day, limit=limit).df
        if bars.empty: return None, "No Data"
        
        bars = bars.reset_index()
        bars.columns = [c.lower() for c in bars.columns]
        
        # Calc Indicators on full set
        bars['rsi'] = ta.rsi(bars['close'], length=14)
        bars['sma_200'] = ta.sma(bars['close'], length=200)
        bars['atr'] = ta.atr(bars['high'], bars['low'], bars['close'], length=14)
        
        # Slice to requested period
        data = bars.iloc[-days:].copy().reset_index(drop=True)
        
    except Exception as e:
        return None, str(e)

    # Simulation Variables
    initial_capital = 10000.0
    cash = initial_capital
    equity_curve = [initial_capital]
    trades = []
    
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    state = 'SCANNING' # 'SCANNING' | 'HOLDING'

    for index, row in data.iterrows():
        current_price = row['close']
        current_date = row['timestamp'].strftime('%Y-%m-%d')
        
        if math.isnan(row['sma_200']) or math.isnan(row['atr']):
            equity_curve.append(cash + (position * current_price))
            continue

        if state == 'SCANNING':
            # LOGIC: Trend UP + RSI LOW
            if current_price > row['sma_200'] and row['rsi'] < 35:
                atr = row['atr']
                stop_dist = 2 * atr
                target_dist = 3 * atr
                
                risk_dollars = 100 # Risk $100 per trade
                shares = math.floor(risk_dollars / stop_dist)
                cost = shares * current_price
                
                if shares > 0 and cost < cash:
                    state = 'HOLDING'
                    position = shares
                    entry_price = current_price
                    stop_loss = current_price - stop_dist
                    take_profit = current_price + target_dist
                    cash -= cost
                    trades.append({"date": current_date, "type": "BUY", "price": entry_price})

        elif state == 'HOLDING':
            sell_price = None
            result = ""
            
            # Simplified exits based on High/Low
            if row['low'] <= stop_loss:
                sell_price = stop_loss
                result = "STOP LOSS"
            elif row['high'] >= take_profit:
                sell_price = take_profit
                result = "WIN"
            
            if sell_price:
                pnl = (sell_price - entry_price) * position
                cash += (position * sell_price)
                state = 'SCANNING'
                trades.append({"date": current_date, "type": "SELL", "price": sell_price, "result": result, "pnl": pnl})
                position = 0

        # Update Equity
        current_val = cash + (position * current_price)
        equity_curve.append(current_val)

    # Stats
    sells = [t for t in trades if t['type'] == 'SELL']
    total_trades = len(sells)
    wins = len([t for t in sells if t.get('result') == 'WIN'])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_return = ((equity_curve[-1] - initial_capital) / initial_capital) * 100
    
    # Plotting
    plt.figure(figsize=(6, 2.5), facecolor='#111827')
    ax = plt.gca()
    ax.set_facecolor('#111827')
    plt.plot(equity_curve, color='#4ade80', linewidth=1.5)
    plt.title(f"{symbol} 1yr Sim (Pure Technical)", color='white', fontsize=10)
    plt.tick_params(colors='gray', labelsize=8)
    plt.grid(color='#374151', linestyle='--', linewidth=0.5)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=False, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return {
        "trades": total_trades,
        "win_rate": round(win_rate, 1),
        "return": round(total_return, 2),
        "plot": plot_url
    }, trades

# ==========================================
# 5. AUTOMATED SCANNER JOB
# ==========================================
def scanner_job():
    logger.info("📡 Scanning Markets...")
    for sym in SYMBOLS:
        try:
            # Check existing position
            try:
                pos = api.get_position(sym)
                if float(pos.qty) > 0: continue
            except: pass

            bars, headlines = get_market_data(sym)
            if bars is None: continue
            
            techs = analyze_technicals(bars)
            
            # PRE-FILTER: Technical Setup (Trend + Oversold)
            if techs['rsi'] < 35 and techs['trend'] == "UP":
                logger.info(f"🔎 Technical Match: {sym}")
                
                if brain:
                    verdict = brain.analyze(sym, techs['price'], techs['rsi'], techs['atr'], techs['trend'], "\n".join(headlines))
                    
                    if verdict.score >= 8:
                        logger.info(f"🚀 AI APPROVED: {sym} (Score: {verdict.score})")
                        place_order(sym, 'buy', verdict.reason, verdict.score)
                    else:
                        logger.info(f"🛑 AI VETO: {sym} (Score: {verdict.score})")
                else:
                    logger.warning("AI Brain missing, skipping trade.")

        except Exception as e:
            logger.error(f"Scan error {sym}: {e}")

# ==========================================
# 6. HTML TEMPLATE (Modern Dashboard)
# ==========================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <title>Sentinel Pro Commander</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1f2937; }
        ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
        .htmx-indicator{display:none} 
        .htmx-request .htmx-indicator{display:inline} 
        .htmx-request.htmx-indicator{display:inline}
    </style>
</head>
<body class="bg-gray-900 text-gray-100 font-mono min-h-screen p-6">

    <div class="flex justify-between items-center mb-8 border-b border-gray-700 pb-4">
        <div>
            <h1 class="text-3xl font-bold text-green-400">⚡ SENTINEL PRO</h1>
            <p class="text-sm text-gray-400">AI-Augmented Algorithmic Trading</p>
        </div>
        <div class="text-right">
            <div id="account-info" hx-get="/api/balance" hx-trigger="load, every 10s">
                <span class="animate-pulse text-gray-500">Connecting to Broker...</span>
            </div>
        </div>
    </div>

    <div class="grid grid-cols-12 gap-6">
        
        <div class="col-span-12 md:col-span-4 space-y-6">
            
            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                <h2 class="text-xl font-bold text-blue-400 mb-4">🧠 AI Analyst</h2>
                <form hx-post="/api/analyze" hx-target="#ai-result" hx-indicator="#loading">
                    <div class="flex gap-2">
                        <input type="text" name="ticker" placeholder="Symbol (e.g. AAPL)" 
                               class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500 uppercase">
                        <button type="submit" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-bold">SCAN</button>
                    </div>
                </form>
                <div id="loading" class="htmx-indicator mt-4 text-center text-gray-500 text-xs">
                    🛰️ Querying GPT-4 & Market Data...
                </div>
                <div id="ai-result" class="mt-4"></div>
            </div>

            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                <h2 class="text-xl font-bold text-pink-400 mb-4">🕰️ Time Machine</h2>
                <form hx-post="/api/backtest" hx-target="#backtest-results" hx-indicator="#bt-loading">
                    <div class="flex gap-2">
                        <input type="text" name="ticker" placeholder="Validate (e.g. NVDA)" 
                               class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white uppercase">
                        <button type="submit" class="bg-pink-700 hover:bg-pink-800 px-4 py-2 rounded font-bold">TEST</button>
                    </div>
                </form>
                <div id="bt-loading" class="htmx-indicator mt-4 text-center text-gray-500 text-xs">
                    🧮 Crunching 365 days of OHLCV data...
                </div>
                <div id="backtest-results" class="mt-4"></div>
            </div>

            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                <h2 class="text-xl font-bold text-yellow-400 mb-4">⚡ Manual Override</h2>
                <form hx-post="/api/trade" hx-target="#trade-status">
                    <div class="flex gap-2 mb-2">
                        <input type="text" name="ticker" placeholder="Symbol" class="w-20 bg-gray-900 border border-gray-600 rounded px-2 uppercase">
                        <select name="action" class="bg-gray-900 border border-gray-600 rounded px-2">
                            <option value="buy">BUY</option>
                            <option value="sell">SELL</option>
                        </select>
                        <button type="submit" class="w-full bg-green-600 hover:bg-green-700 rounded font-bold">EXECUTE</button>
                    </div>
                </form>
                <div id="trade-status" class="text-sm mt-2"></div>
            </div>

            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                <h2 class="text-xl font-bold text-purple-400 mb-2">💼 Positions</h2>
                <div id="positions-list" hx-get="/api/positions" hx-trigger="load, every 5s"></div>
            </div>
        </div>

        <div class="col-span-12 md:col-span-8 space-y-6">
            
            <div class="bg-gray-800 p-1 rounded-lg border border-gray-700 h-96 relative" id="chart-container">
                <div class="tradingview-widget-container" style="height:100%;width:100%">
                  <div id="tradingview_12345" style="height:100%;width:100%"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                  <script type="text/javascript">
                  new TradingView.widget({
                    "autosize": true, "symbol": "NASDAQ:NVDA",
                    "interval": "D", "timezone": "Etc/UTC", "theme": "dark",
                    "style": "1", "locale": "en", "toolbar_bg": "#f1f3f6",
                    "enable_publishing": false, "container_id": "tradingview_12345"
                  });
                  </script>
                </div>
            </div>

            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-bold text-gray-300">📜 Neural Ledger</h2>
                    <button class="text-xs bg-red-900 text-red-200 px-2 py-1 rounded border border-red-700 hover:bg-red-800" 
                            hx-post="/api/panic" 
                            onclick="return confirm('WARNING: This will liquidate ALL positions and cancel ALL orders. Proceed?')">
                        ☢️ LIQUIDATE ALL
                    </button>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm text-left text-gray-400">
                        <thead class="text-xs text-gray-500 uppercase bg-gray-700">
                            <tr>
                                <th class="px-4 py-3">Time</th>
                                <th class="px-4 py-3">Sym</th>
                                <th class="px-4 py-3">Action</th>
                                <th class="px-4 py-3">Price</th>
                                <th class="px-4 py-3">Reason</th>
                            </tr>
                        </thead>
                        <tbody hx-get="/api/logs" hx-trigger="load, every 5s"></tbody>
                    </table>
                </div>
            </div>

        </div>
    </div>
</body>
</html>
"""

# ==========================================
# 7. FLASK ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/balance')
def api_balance():
    try:
        acct = api.get_account()
        pl = float(acct.equity) - float(acct.last_equity)
        pl_color = "text-green-400" if pl >= 0 else "text-red-400"
        return f"""
        <div class="text-2xl font-bold {pl_color}">${float(acct.equity):,.2f}</div>
        <div class="text-xs text-gray-500">BP: ${float(acct.buying_power):,.2f}</div>
        """
    except:
        return "<span class='text-red-500'>API Error</span>"

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    symbol = request.form.get('ticker').upper().strip()
    if not symbol: return "Enter Symbol"
    
    bars, headlines = get_market_data(symbol)
    if bars is None: return "<div class='text-red-500'>Data Error / Symbol Not Found</div>"
    
    techs = analyze_technicals(bars)
    
    # Run AI
    if brain:
        verdict = brain.analyze(symbol, techs['price'], techs['rsi'], techs['atr'], techs['trend'], "\n".join(headlines))
        color = "text-green-400" if verdict.score > 6 else "text-red-400"
        
        return f"""
        <div class="border-l-4 border-blue-500 pl-4 py-2 bg-gray-900 rounded fade-in">
            <div class="flex justify-between mb-2">
                <span class="font-bold text-xl">{symbol}</span>
                <span class="font-bold text-xl {color}">{verdict.score}/10</span>
            </div>
            <div class="grid grid-cols-2 gap-2 text-xs text-gray-400 mb-2">
                <div>RSI: {techs['rsi']}</div>
                <div>Trend: {techs['trend']}</div>
                <div class="col-span-2">Rec: <span class="uppercase font-bold text-white">{verdict.action}</span></div>
            </div>
            <p class="text-sm italic text-gray-300">"{verdict.reason}"</p>
            <script>
                new TradingView.widget({{
                    "autosize": true, "symbol": "{symbol}",
                    "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1",
                    "container_id": "tradingview_12345"
                }});
            </script>
        </div>
        """
    else:
        return "AI Module Not Loaded."

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    sym = request.form.get('ticker', 'NVDA').upper()
    stats, trade_log = run_backtest_simulation(sym)
    
    if not stats:
        return f"<div class='text-red-500 p-2'>Backtest Failed: {trade_log}</div>"
    
    html = f"""
    <div class="animate-pulse-once">
        <div class="grid grid-cols-3 gap-2 mb-4 text-center">
            <div class="bg-gray-900 p-2 rounded border border-gray-700">
                <div class="text-[10px] text-gray-500 uppercase">Trades</div>
                <div class="text-lg font-bold text-white">{stats['trades']}</div>
            </div>
            <div class="bg-gray-900 p-2 rounded border border-gray-700">
                <div class="text-[10px] text-gray-500 uppercase">Win Rate</div>
                <div class="text-lg font-bold {'text-green-400' if stats['win_rate'] > 50 else 'text-red-400'}">{stats['win_rate']}%</div>
            </div>
            <div class="bg-gray-900 p-2 rounded border border-gray-700">
                <div class="text-[10px] text-gray-500 uppercase">Return</div>
                <div class="text-lg font-bold {'text-green-400' if stats['return'] > 0 else 'text-red-400'}">{stats['return']}%</div>
            </div>
        </div>
        
        <div class="bg-gray-900 rounded p-1 mb-4 border border-gray-700">
            <img src="data:image/png;base64,{stats['plot']}" class="w-full rounded" />
        </div>
        
        <div class="max-h-32 overflow-y-auto text-xs border-t border-gray-700 pt-2">
             <table class="w-full text-left text-gray-400">
                <thead><tr class="text-gray-600"><th>Date</th><th>Res</th><th class="text-right">PnL</th></tr></thead>
                <tbody>
    """
    for t in trade_log:
        if t['type'] == 'SELL':
            color = "text-green-400" if t['pnl'] > 0 else "text-red-400"
            html += f"<tr><td>{t['date']}</td><td>{t['result']}</td><td class='{color} text-right'>${t['pnl']:.2f}</td></tr>"
            
    html += "</tbody></table></div></div>"
    return html

@app.route('/api/trade', methods=['POST'])
def api_trade():
    sym = request.form.get('ticker').upper()
    action = request.form.get('action')
    
    if action == 'buy':
        order = place_order(sym, 'buy', "Manual Override", 10)
        if order: return f"<span class='text-green-500'>Order Placed</span>"
        return "<span class='text-red-500'>Failed</span>"
    else:
        try:
            api.close_position(sym)
            return "<span class='text-yellow-500'>Closed</span>"
        except:
             return "<span class='text-red-500'>No Position</span>"

@app.route('/api/positions')
def api_positions():
    try:
        pos = api.list_positions()
        if not pos: return "<div class='text-gray-500 text-sm italic'>Cash Gang 💰</div>"
        
        html = ""
        for p in pos:
            pl = float(p.unrealized_pl)
            color = "text-green-400" if pl > 0 else "text-red-400"
            html += f"""
            <div class="flex justify-between items-center bg-gray-900 p-2 mb-1 rounded text-sm border border-gray-700">
                <div>
                    <span class="font-bold text-white">{p.symbol}</span>
                    <span class="text-xs text-gray-500 ml-1">{p.qty}sh</span>
                </div>
                <span class="{color}">${pl:.2f}</span>
            </div>
            """
        return html
    except:
        return "API Error"

@app.route('/api/logs')
def api_logs():
    trades = list(history_col.find().sort("timestamp", -1).limit(10))
    html = ""
    for t in trades:
        ts = t['timestamp'].strftime('%H:%M')
        style = "text-green-400" if t['action'] == 'buy' else "text-red-400"
        html += f"""
        <tr class="bg-gray-800 border-b border-gray-700 hover:bg-gray-700 transition">
            <td class="px-4 py-2 font-mono text-gray-500">{ts}</td>
            <td class="px-4 py-2 font-bold text-white">{t['symbol']}</td>
            <td class="px-4 py-2 {style} uppercase text-xs font-bold">{t['action']}</td>
            <td class="px-4 py-2 text-gray-300">${t.get('price', 0)}</td>
            <td class="px-4 py-2 text-gray-500 text-xs truncate max-w-[100px]" title="{t['reason']}">{t['reason']}</td>
        </tr>
        """
    return html

@app.route('/api/panic', methods=['POST'])
def panic():
    api.cancel_all_orders()
    api.close_all_positions()
    logger.critical("☢️ PANIC PRESSED - LIQUIDATING ALL")
    return ""

# ==========================================
# 8. STARTUP
# ==========================================
if __name__ == '__main__':
    # Scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(scanner_job, 'interval', minutes=15)
    scheduler.start()
    
    print("\n⚡ SENTINEL COMMANDER ONLINE")
    print("👉 UI: http://localhost:5000")
    print("👉 LOGS: See terminal output\n")
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000)
