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
```python
import time
import os
import math
import requests
import pandas as pd
import pandas_ta as ta  # NEW: Robust technical analysis library
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from flask import Flask, request, jsonify, render_template_string
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
import logging

# --- LANGCHAIN & PYDANTIC ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Sentinel_Pro")

# --- CONFIGURATION & SAFETY ---
# Symbols to trade
SYMBOLS_ENV = os.getenv("TARGET_SYMBOLS", "NVDA,AMD,MSFT,GOOGL,AMZN,TSLA")
SYMBOLS = [s.strip() for s in SYMBOLS_ENV.split(',')]

# 🛡️ BUDGET CAP (Crucial Safety Feature)
# Max dollars the bot is allowed to have in the market at any time.
MAX_CAPITAL_DEPLOYED = float(os.getenv("MAX_CAPITAL_DEPLOYED", "2000.00"))

# Risk Parameters
RISK_PER_TRADE_PCT = 0.015  # Risk 1.5% of equity per trade
MAX_POS_SIZE_PCT = 0.20     # Max 20% allocation per stock
TP_PCT = 1.08               # Take Profit +8%
SL_PCT = 0.96               # Stop Loss -4%
LIMIT_BUFFER = 1.002        # Pay up to 0.2% above current price to ensure fill (Slippage protection)

# Infrastructure
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "sentinel_alerts")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:5000")

# --- ALPACA CONNECTION ---
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version='v2')
app = Flask(__name__)

# --- MONGO CONNECTION ---
try:
    client = MongoClient(MONGO_URI)
    db = client['shadow_trading']
    history_col = db['history']
except Exception as e:
    logger.critical(f"MongoDB Failed: {e}")
    exit(1)

# --- 1. ENHANCED AI AGENT ---
class TradeSignal(BaseModel):
    score: int = Field(description="Sentiment score 0-10 (10 is Bullish/Buy).")
    reason: str = Field(description="A strategic justification under 25 words.")

class SentinelAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3) 
        self.chain = self._build_chain()
    
    def _build_chain(self):
        system_prompt = (
            "You are a Risk Manager at a Hedge Fund. "
            "We have a Technical BUY Signal (Oversold). "
            "Your job is to VETO the trade if there is catastrophic news (bankruptcy, fraud, lawsuits). "
            "If news is quiet or generic market noise, confirm the trade with a high score (8-10). "
            "If news is actively bad, score low (0-3). Output JSON."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", (
                "Ticker: {ticker}\n"
                "Current Price: ${price}\n"
                "Technical Context: RSI is {rsi} (Oversold), Price is above SMA.\n"
                "Recent Headlines:\n{headlines}"
            ))
        ])
        return prompt | self.llm.with_structured_output(TradeSignal)

    def analyze(self, ticker, headlines, price, rsi):
        if not headlines: 
            return TradeSignal(score=6, reason="No news found; technicals valid.")
        
        return self.chain.invoke({
            "ticker": ticker, 
            "headlines": "\n".join(headlines),
            "price": price,
            "rsi": round(rsi, 2)
        })

agent = SentinelAgent()

# --- 2. DATA METHODS ---
def get_clean_data(symbol):
    try:
        # Get 200 days to be safe for SMA calc
        bars = api.get_bars(symbol, TimeFrame.Day, limit=200).df
        if bars.empty: return None
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index()
        return bars
    except Exception as e:
        logger.error(f"Data Error {symbol}: {e}")
        return None

def get_market_news(symbol):
    try:
        news_list = api.get_news(symbol=symbol, limit=4)
        return [f"- {n.headline} ({n.created_at.strftime('%Y-%m-%d')})" for n in news_list]
    except Exception as e:
        return []

# --- 3. RISK & MATH (UPDATED WITH PANDAS-TA) ---
def calculate_technicals(data):
    """
    Calculates RSI(14) and SMA(50) using pandas-ta for robustness.
    """
    # Safety check for minimum data points
    if len(data) < 50: 
        return 50.0, 0.0
    
    # Ensure column names are lowercase for pandas-ta
    data.columns = [c.lower() for c in data.columns]
    
    try:
        # Calculate indicators using pandas-ta
        # This handles edge cases, smoothing, and NaN values automatically
        data['rsi'] = ta.rsi(data['close'], length=14)
        data['sma_50'] = ta.sma(data['close'], length=50)
        
        # Drop NaNs created by the calculation window to avoid errors
        data = data.dropna()

        if data.empty:
            return 50.0, 0.0

        # Get the most recent values
        rsi = data['rsi'].iloc[-1]
        sma = data['sma_50'].iloc[-1]
        
        return rsi, sma
        
    except Exception as e:
        logger.error(f"Math Error: {e}")
        return 50.0, 0.0

def check_budget_availability(symbol_price, quantity):
    """
    🛡️ SAFETY CHECK: Ensures we do not exceed MAX_CAPITAL_DEPLOYED.
    """
    try:
        positions = api.list_positions()
        current_exposure = sum([float(p.market_value) for p in positions])
        
        trade_cost = symbol_price * quantity
        
        if (current_exposure + trade_cost) > MAX_CAPITAL_DEPLOYED:
            logger.warning(f"🚫 BUDGET CAP HIT: Exposure ${current_exposure:.2f} + Trade ${trade_cost:.2f} > ${MAX_CAPITAL_DEPLOYED}")
            return False
        return True
    except Exception as e:
        logger.error(f"Budget check failed: {e}")
        return False

def calculate_smart_size(price):
    try:
        account = api.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        
        # Risk Math
        risk_amt = equity * RISK_PER_TRADE_PCT
        stop_dist = price * (1 - SL_PCT)
        if stop_dist <= 0: return 0
        
        qty_risk = math.floor(risk_amt / stop_dist)
        
        # Max Allocation Math
        max_alloc = equity * MAX_POS_SIZE_PCT
        qty_cap = math.floor(max_alloc / price)
        
        final_qty = min(qty_risk, qty_cap)
        
        if (final_qty * price) > buying_power:
            final_qty = math.floor(buying_power / price)
            
        return max(0, final_qty)
    except Exception as e:
        logger.error(f"Size Calc Error: {e}")
        return 0

# --- 4. EXECUTION ENGINE (Limit Orders) ---
def execute_smart_trade(symbol, price, side, reason, score, rsi):
    qty = calculate_smart_size(price)
    if qty < 1: 
        logger.info(f"⚠️ {symbol}: Qty calculated to 0 (Risk or Funds issue).")
        return None

    if not check_budget_availability(price, qty):
        return None

    # Limit Price Logic: Bid slightly higher to get filled, but cap slippage
    limit_price = round(price * LIMIT_BUFFER, 2)
    take_profit = round(price * TP_PCT, 2)
    stop_loss = round(price * SL_PCT, 2)

    try:
        # Submit Bracket Order with LIMIT entry
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',          # Changed from market
            limit_price=limit_price, 
            time_in_force='gtc',
            order_class='bracket',
            take_profit={'limit_price': take_profit},
            stop_loss={'stop_price': stop_loss}
        )
        
        logger.info(f"✅ ORDER SENT: {symbol} | {qty} shares | Limit: ${limit_price}")
        
        # Log to Mongo
        history_col.insert_one({
            "action": side,
            "symbol": symbol,
            "qty": qty,
            "entry_price": price,
            "limit_price": limit_price,
            "tp": take_profit,
            "sl": stop_loss,
            "reason": reason,
            "ai_score": score,
            "rsi": rsi,
            "order_id": order.id,
            "timestamp": time.time(),
            "status": "filled" # Assumption for simple log
        })
        
        notify_user(symbol, price, qty, score, reason)
        return order
        
    except Exception as e:
        logger.error(f"Execution Failed {symbol}: {e}")
        return None

def notify_user(symbol, price, qty, score, reason):
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", 
            data=f"🚀 {symbol} BUY | {qty} sh @ ${price} | AI: {score}/10 | {reason}",
            headers={"Title": f"Sentinel: Bought {symbol}", "Tags": "moneybag"}
        )
    except: pass

# --- 5. MAIN LOGIC ---
def scan_market():
    # 1. Market Hours Check
    try:
        clock = api.get_clock()
        if not clock.is_open:
            logger.info("🌙 Market Closed.")
            return
    except: pass

    logger.info("🔎 Scanning Market...")
    
    for symbol in SYMBOLS:
        try:
            # Check existing position first
            try:
                pos = api.get_position(symbol)
                if float(pos.qty) > 0: continue # Skip if we hold it
            except: pass 

            # Technicals
            df = get_clean_data(symbol)
            if df is None: continue
            
            price = df['close'].iloc[-1].item()
            rsi, sma = calculate_technicals(df)
            
            # Strategy: Simple Dip Buy (Uptrend + Oversold)
            is_uptrend = price > sma
            is_oversold = rsi < 35  # Aggressive dip buying
            
            if is_uptrend and is_oversold:
                logger.info(f"⚡ MATCH: {symbol} (RSI {rsi:.1f})")
                
                # AI Analysis
                headlines = get_market_news(symbol)
                decision = agent.analyze(symbol, headlines, price, rsi)
                
                if decision.score >= 7:
                    execute_smart_trade(symbol, price, "buy", decision.reason, decision.score, rsi)
                else:
                    logger.info(f"🛑 AI VETO: {symbol} | Score {decision.score} | {decision.reason}")
                    
        except Exception as e:
            logger.error(f"Scan error {symbol}: {e}")

# --- 6. FLASK SERVER & DASHBOARD ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentinel Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body { font-family: sans-serif; background: #111; color: #eee; padding: 20px; }
        .card { background: #222; padding: 15px; margin-bottom: 20px; border-radius: 8px; border: 1px solid #333; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #444; }
        th { color: #888; }
        .buy { color: #4caf50; font-weight: bold; }
        .panic { background: #d32f2f; color: white; border: none; padding: 15px; width: 100%; font-size: 1.2em; cursor: pointer; }
    </style>
</head>
<body>
    <h1>🛡️ Sentinel Control Deck</h1>
    
    <div class="card">
        <h3>💰 Portfolio Status</h3>
        <p><b>Equity:</b> ${{ equity }} | <b>Buying Power:</b> ${{ bp }}</p>
        <p><b>Budget Used:</b> ${{ exposure }} / ${{ max_budget }}</p>
    </div>

    <div class="card">
        <h3>🚨 Emergency Controls</h3>
        <form action="/panic" method="post" onsubmit="return confirm('ARE YOU SURE? THIS SELLS EVERYTHING.');">
            <button type="submit" class="panic">☢️ PANIC BUTTON: SELL ALL POSITIONS ☢️</button>
        </form>
    </div>

    <div class="card">
        <h3>📜 Recent Trade Log</h3>
        <table>
            <tr><th>Time</th><th>Symbol</th><th>Action</th><th>Price</th><th>AI Score</th><th>Reason</th></tr>
            {% for t in trades %}
            <tr>
                <td>{{ t.timestamp }}</td>
                <td class="buy">{{ t.symbol }}</td>
                <td>{{ t.action }}</td>
                <td>${{ t.entry_price }}</td>
                <td>{{ t.ai_score }}</td>
                <td>{{ t.reason }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    try:
        # Get Account Info
        acct = api.get_account()
        positions = api.list_positions()
        exposure = sum([float(p.market_value) for p in positions])
        
        # Get DB History
        trades = list(history_col.find().sort("timestamp", -1).limit(10))
        for t in trades:
            t['timestamp'] = time.strftime('%H:%M:%S', time.localtime(t['timestamp']))
        
        return render_template_string(HTML_TEMPLATE, 
            equity=acct.equity, 
            bp=acct.buying_power,
            exposure=round(exposure, 2),
            max_budget=MAX_CAPITAL_DEPLOYED,
            trades=trades
        )
    except Exception as e:
        return f"Dashboard Error: {e}"

@app.route('/panic', methods=['POST'])
def panic():
    """
    🚨 KILL SWITCH: Cancels all orders and Liquidates all positions.
    """
    try:
        api.cancel_all_orders()
        api.close_all_positions()
        logger.critical("☢️ PANIC TRIGGERED: SELLING EVERYTHING ☢️")
        return "<h1>🚨 LIQUIDATION INITIATED. CHECK BROKER.</h1>"
    except Exception as e:
        return f"Panic Failed: {e}"

if __name__ == '__main__':
    # Initialize Scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(scan_market, 'interval', minutes=15) # Check every 15 mins
    scheduler.start()
    
    # Run App
    print(f"🛡️ Sentinel Active on {PUBLIC_URL}")
    app.run(host='0.0.0.0', port=5000)

```
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
