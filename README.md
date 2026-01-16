# Building "The Sentinel": A Hybrid AI Swing Trading Bot

**The Problem:** Automated trading bots are terrifying. If you leave them alone, they can drain your account in minutes due to a single bug or a market crash.

**The Solution:** A "Hybrid" system. The bot does the heavy lifting (scanning, math, AI analysis), but it keeps a human in the loop.

In this guide, we are going to build **The Sentinel**, a Python-based agent that:

1. **Scans the market** every hour for technical setups (Trend + RSI).
2. **Reads the news** using an LLM to validate the move.
3. **Executes a "Shadow Trade"** (Paper Trade) to track performance.
4. **Pings your phone** so you can decide if you want to mirror the trade with real money.

## Executive Summary

**The Sentinel** is a hybrid AI trading agent designed to mitigate the risks of fully autonomous trading while leveraging the speed of algorithmic analysis. It solves the "black box" problem by acting as a high-speed analyst rather than a reckless trader—automating the research, but leaving the final financial decision to a human.

* **Goal:** Automate the 99% of trading that is boring (scanning, math, news reading) to focus on the 1% that matters (decision making).
* **Mechanism:** It scans for technical dips in uptrending stocks, uses an LLM (GPT-4) to verify the move against real-time news sentiment, and executes a "paper trade" to track hypothetical performance.
* **Stack:** Python, Docker, MongoDB (Storage), OpenAI (Sentiment Analysis), Alpaca (Market Data & Paper Execution).
* **Outcome:** A push notification on your phone containing a pre-validated, high-probability trade setup with entry, stop-loss, and take-profit targets calculated for you.

---

## The Strategy: Explained Simply

Think of **The Sentinel** like a professional house flipper looking for a deal in a nice neighborhood. It uses three specific checks before it calls you:

### 1. The "Good Neighborhood" Check (Trend)

* **Technical Term:** 50-Day Moving Average (SMA).
* **The Analogy:** We only want to buy houses in neighborhoods where property values are generally going *up* over time. If the whole neighborhood is crashing (Price < SMA), we don't care how cheap the house is—we aren't interested.

### 2. The "Ugly Paint" Check (Discount)

* **Technical Term:** RSI (Relative Strength Index) < 35.
* **The Analogy:** We found a great neighborhood, but we don't want to pay full price. We look for the one house that looks a little ugly right now—maybe the paint is peeling or the lawn is overgrown (Oversold). It’s a good asset temporarily looking bad.

### 3. The "Foundation" Check (AI Analysis)

* **Technical Term:** LLM Sentiment Analysis.
* **The Analogy:** This is the most critical step. Before we buy that ugly house, we hire an inspector ( The AI) to check *why* it's cheap.
* **Good Cheap:** The lawn is overgrown because the owner was lazy. (Market overreaction to minor news). **Result: BUY.**
* **Bad Cheap:** The foundation is cracked and the house is sinking. (CEO fraud, bankruptcy, major lawsuit). **Result: WALK AWAY.**



Only when a stock passes all three checks—it's in a good trend, it's currently cheap, and the news confirms the "foundation" is solid—does the bot ping your phone.

---

## The Architecture

We are ditching complex cloud architectures for a robust **All-In-One** Python script running in Docker.

* **The Brain:** Python (Logic) + OpenAI (Sentiment).
* **The Hands:** Alpaca API (Paper Trading Execution).
* **The Eyes:** Alpaca Data API (Price History & News).
* **The Memory:** MongoDB (Trade History).
* **The Nervous System:** `ntfy.sh` (Push Notifications).

---

## Part 1: The Setup

### 1. `requirements.txt`

We need a lightweight environment. We use `alpaca-trade-api` for both data and execution to keep things consistent.

```text
flask
apscheduler
requests
pymongo
langchain
langchain-openai
alpaca-trade-api
pandas
numpy

```

### 2. `Dockerfile`

This ensures our bot runs identically on your laptop or a cheap cloud VPS.

```dockerfile
FROM python:3.9-slim

# Keep Python from buffering stdout so logs show up immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
CMD ["python", "main.py"]

```

### 3. `docker-compose.yml`

We spin up a local MongoDB instance alongside our bot.

```yaml
version: '3.8'
services:
  mongo:
    image: mongo:latest
    container_name: sentinel_mongo
    restart: always
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  sentinel:
    build: .
    restart: on-failure
    depends_on:
      - mongo
    environment:
      # SYSTEM CONFIG
      MONGO_URI: "mongodb://mongo:27017/"
      NTFY_TOPIC: "my_sentinel_bot"
      PUBLIC_URL: "http://localhost:5000"
      
      # API KEYS (Replace these!)
      OPENAI_API_KEY: "sk-..."
      ALPACA_API_KEY: "PK..."
      ALPACA_SECRET_KEY: "..."
      # Use Paper URL for safety
      ALPACA_BASE_URL: "https://paper-api.alpaca.markets" 

volumes:
  mongo_data:

```

---

## Part 2: The Code (`main.py`)

This is the core logic. It implements the "Three-Green-Lights" strategy: **Trend + Discount + Sentiment**.

```python
import time
import os
import math
import requests
import pandas as pd
import pandas_ta as ta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from flask import Flask, request, jsonify, render_template_string
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
import logging
from datetime import datetime

# --- LANGCHAIN & PYDANTIC ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Sentinel_Commander")

# --- CONFIG & SECRETS ---
SYMBOLS_ENV = os.getenv("TARGET_SYMBOLS", "NVDA,AMD,MSFT,GOOGL,AMZN,TSLA,COIN")
SYMBOLS = [s.strip() for s in SYMBOLS_ENV.split(',')]
MAX_CAPITAL_DEPLOYED = float(os.getenv("MAX_CAPITAL_DEPLOYED", "5000.00"))

# ALPACA
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version='v2')

# MONGO
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
try:
    client = MongoClient(MONGO_URI)
    db = client['sentinel_pro']
    history_col = db['history']
except Exception as e:
    logger.critical(f"DB Fail: {e}")
    exit(1)

app = Flask(__name__)

# --- 1. SOPHISTICATED AI AGENT ---
class TradeVerdict(BaseModel):
    score: int = Field(description="Bullish score 0-10.")
    action: str = Field(description="One of: 'BUY', 'WAIT', 'SELL_NOW'.")
    reason: str = Field(description="Concise strategic reasoning (max 20 words).")
    risk_level: str = Field(description="Low, Medium, or High based on volatility.")

class SentinelBrain:
    def __init__(self):
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

brain = SentinelBrain()

# --- 2. ADVANCED DATA & MATH ---
def get_market_data(symbol):
    try:
        # Fetch enough data for ATR calculation
        bars = api.get_bars(symbol, TimeFrame.Day, limit=100).df
        if bars.empty: return None
        if isinstance(bars.index, pd.MultiIndex): bars = bars.reset_index()
        
        # News
        news = api.get_news(symbol=symbol, limit=3)
        headlines = [f"- {n.headline}" for n in news] if news else ["No recent news."]
        
        return bars, headlines
    except Exception as e:
        logger.error(f"Data fail {symbol}: {e}")
        return None, []

def analyze_technicals(df):
    """Calculates RSI, SMA, and ATR for dynamic stops."""
    df.columns = [c.lower() for c in df.columns]
    
    # Pandas-TA Magic
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['sma_200'] = ta.sma(df['close'], length=200)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    current = df.iloc[-1]
    
    trend = "UP" if current['close'] > current['sma_200'] else "DOWN"
    return {
        "price": round(current['close'], 2),
        "rsi": round(current['rsi'], 2),
        "atr": round(current['atr'], 2),
        "trend": trend
    }

# --- 3. SMART EXECUTION ---
def calculate_position_size(price, atr, risk_per_trade_dollars=50.0):
    """
    Volatility Sizing: If ATR is high (volatile), buy fewer shares.
    Target Risk: Lose max $50 if stop is hit.
    """
    stop_distance = 2 * atr # Stop loss is 2x ATR
    if stop_distance == 0: return 0
    
    shares_risk = math.floor(risk_per_trade_dollars / stop_distance)
    
    # Cap at budget
    acct = api.get_account()
    buying_power = float(acct.buying_power)
    max_shares_budget = math.floor(min(buying_power, MAX_CAPITAL_DEPLOYED) / price)
    
    return min(shares_risk, max_shares_budget)

def place_order(symbol, side, reason, score):
    df, _ = get_market_data(symbol)
    if df is None: return False
    techs = analyze_technicals(df)
    
    qty = calculate_position_size(techs['price'], techs['atr'])
    if qty < 1: return False

    # Dynamic Bracket
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
        
        # Log
        history_col.insert_one({
            "timestamp": datetime.now(), "symbol": symbol, "action": side,
            "qty": qty, "price": limit_price, "reason": reason, "score": score
        })
        return order
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return None

# --- 4. AUTO-PILOT LOOP ---
def scanner_job():
    logger.info("📡 Scanning Markets...")
    for sym in SYMBOLS:
        try:
            # Skip if we own it
            try:
                if float(api.get_position(sym).qty) > 0: continue
            except: pass

            bars, headlines = get_market_data(sym)
            if bars is None: continue
            
            techs = analyze_technicals(bars)
            
            # Filter: Only bother AI if technicals look interesting
            if techs['rsi'] < 35 and techs['trend'] == "UP":
                verdict = brain.analyze(sym, techs['price'], techs['rsi'], techs['atr'], techs['trend'], "\n".join(headlines))
                
                if verdict.score >= 8:
                    logger.info(f"🚀 AI BUY SIGNAL: {sym}")
                    place_order(sym, 'buy', verdict.reason, verdict.score)
        except Exception as e:
            logger.error(f"Scan error {sym}: {e}")

# --- 5. MODERN UI (Tailwind + HTMX) ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <title>Sentinel Pro Commander</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>
        /* Custom scrollbar for terminal feel */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1f2937; }
        ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
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
                <span class="animate-pulse">Loading data...</span>
            </div>
        </div>
    </div>

    <div class="grid grid-cols-12 gap-6">
        
        <div class="col-span-4 space-y-6">
            
            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                <h2 class="text-xl font-bold text-blue-400 mb-4">🧠 AI Analyst</h2>
                <form hx-post="/api/analyze" hx-target="#ai-result" hx-indicator="#loading">
                    <div class="flex gap-2">
                        <input type="text" name="ticker" placeholder="Symbol (e.g. AAPL)" 
                               class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500 uppercase">
                        <button type="submit" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-bold">SCAN</button>
                    </div>
                </form>
                
                <div id="loading" class="htmx-indicator mt-4 text-center text-gray-500 text-sm">
                    🛰️ Querying GPT-4 & Market Data...
                </div>
                
                <div id="ai-result" class="mt-4">
                    </div>
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
                <div id="positions-list" hx-get="/api/positions" hx-trigger="load, every 5s">
                    </div>
            </div>

        </div>

        <div class="col-span-8 space-y-6">
            
            <div class="bg-gray-800 p-1 rounded-lg border border-gray-700 h-96 relative" id="chart-container">
                <div class="tradingview-widget-container">
                  <div id="tradingview_12345"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                  <script type="text/javascript">
                  new TradingView.widget({
                    "width": "100%", "height": "100%", "symbol": "NASDAQ:AAPL",
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
                    <button class="text-xs bg-red-900 text-red-200 px-2 py-1 rounded" hx-post="/api/panic" 
                            onclick="return confirm('CLOSE ALL POSITIONS?')">☢️ LIQUIDATE ALL</button>
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
                        <tbody hx-get="/api/logs" hx-trigger="load, every 10s">
                            </tbody>
                    </table>
                </div>
            </div>

        </div>
    </div>
</body>
</html>
"""

# --- 6. FLASK API ENDPOINTS ---

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/balance')
def api_balance():
    acct = api.get_account()
    pl_color = "text-green-400" if float(acct.equity) > float(acct.last_equity) else "text-red-400"
    html = f"""
    <div class="text-2xl font-bold {pl_color}">${float(acct.equity):,.2f}</div>
    <div class="text-xs text-gray-500">BP: ${float(acct.buying_power):,.2f}</div>
    """
    return html

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    symbol = request.form.get('ticker').upper().strip()
    if not symbol: return "Enter Symbol"
    
    bars, headlines = get_market_data(symbol)
    if bars is None: return "<div class='text-red-500'>Symbol not found or data error.</div>"
    
    techs = analyze_technicals(bars)
    
    # Run AI
    verdict = brain.analyze(symbol, techs['price'], techs['rsi'], techs['atr'], techs['trend'], "\n".join(headlines))
    
    color = "text-green-400" if verdict.score > 6 else "text-red-400"
    
    # Return HTML Snippet
    return f"""
    <div class="border-l-4 border-blue-500 pl-4 py-2 bg-gray-900 rounded">
        <div class="flex justify-between mb-2">
            <span class="font-bold text-xl">{symbol}</span>
            <span class="font-bold text-xl {color}">{verdict.score}/10</span>
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs text-gray-400 mb-2">
            <div>RSI: {techs['rsi']}</div>
            <div>ATR: {techs['atr']}</div>
            <div>Trend: {techs['trend']}</div>
            <div>Rec: {verdict.action}</div>
        </div>
        <p class="text-sm italic text-gray-300">"{verdict.reason}"</p>
        
        <script>
            new TradingView.widget({{
                "width": "100%", "height": "100%", "symbol": "{symbol}",
                "interval": "D", "timezone": "Etc/UTC", "theme": "dark",
                "container_id": "tradingview_12345"
            }});
        </script>
    </div>
    """

@app.route('/api/trade', methods=['POST'])
def api_trade():
    sym = request.form.get('ticker').upper()
    action = request.form.get('action')
    
    if action == 'buy':
        order = place_order(sym, 'buy', "Manual Override", 10)
        if order: return f"<span class='text-green-500'>Order Placed: {order.id}</span>"
        return "<span class='text-red-500'>Failed (Budget/Risk)</span>"
    else:
        # Simple sell all
        try:
            api.close_position(sym)
            return "<span class='text-yellow-500'>Position Closed</span>"
        except:
             return "<span class='text-red-500'>No Position</span>"

@app.route('/api/positions')
def api_positions():
    pos = api.list_positions()
    if not pos: return "<div class='text-gray-500 text-sm'>Cash Gang 💰</div>"
    
    html = ""
    for p in pos:
        pl = float(p.unrealized_pl)
        color = "text-green-400" if pl > 0 else "text-red-400"
        html += f"""
        <div class="flex justify-between items-center bg-gray-900 p-2 mb-1 rounded text-sm">
            <span class="font-bold">{p.symbol}</span>
            <span>{p.qty} sh</span>
            <span class="{color}">${pl:.2f}</span>
        </div>
        """
    return html

@app.route('/api/logs')
def api_logs():
    trades = list(history_col.find().sort("timestamp", -1).limit(10))
    html = ""
    for t in trades:
        ts = t['timestamp'].strftime('%H:%M')
        style = "text-green-400" if t['action'] == 'buy' else "text-red-400"
        html += f"""
        <tr class="bg-gray-800 border-b border-gray-700 hover:bg-gray-700">
            <td class="px-4 py-2">{ts}</td>
            <td class="px-4 py-2 font-bold">{t['symbol']}</td>
            <td class="px-4 py-2 {style} uppercase">{t['action']}</td>
            <td class="px-4 py-2">${t.get('price', 0)}</td>
            <td class="px-4 py-2 text-gray-400 truncate max-w-xs" title="{t['reason']}">{t['reason']}</td>
        </tr>
        """
    return html

@app.route('/api/panic', methods=['POST'])
def panic():
    api.cancel_all_orders()
    api.close_all_positions()
    logger.critical("☢️ PANIC PRESSED")
    return ""

if __name__ == '__main__':
    # Scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(scanner_job, 'interval', minutes=15)
    scheduler.start()
    
    print("⚡ SENTINEL COMMANDER ONLINE: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)
```

---

## How It Works

### 1. The Trend Filter (SMA 50)

We never trade against the trend. The bot calculates the **50-Day Moving Average**. If the current price is *below* this line, we assume the stock is dying, and we do nothing.

### 2. The Discount Filter (RSI)

If the trend is UP, we look for a dip. We use **RSI (Relative Strength Index)**.

* **RSI < 35:** The stock is "Oversold."
* If RSI > 35, we wait.

### 3. The AI Filter (The Vibe Check)

If the math looks good, the AI reads the last 3 news headlines.

* **Scenario A:** "CEO resigns in scandal" -> AI Score: 2/10 -> **BLOCK TRADE.**
* **Scenario B:** "Market dips on inflation fears" -> AI Score: 8/10 -> **EXECUTE.**

### 4. The Shadow Trade

If all lights are green, the bot executes a **Paper Trade** via Alpaca. This tracks your hypothetical P&L. Simultaneously, it sends a notification to your phone. If you agree with the logic, you open your brokerage app and place the real trade yourself.

---

## Next Steps: Making it Smarter

To take this from a fun project to an enterprise-grade tool, consider implementing **Vector Search**.

Currently, the AI has no memory. By using **MongoDB Atlas Vector Search**, you could embed every news headline you analyze. Before placing a trade, the bot could query its own memory:

> *"Last time I saw headlines about 'CEO Investigation' for this stock, did the price recover?"*

If the answer is "No," the bot can veto the trade based on historical performance, creating a self-improving feedback loop.
